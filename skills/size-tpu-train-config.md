---
name: size-tpu-train-config
description: Keep a training config slice-agnostic — fix the global batch, derive per-device parallelism / gradient accumulation per TPU slice.
---

# Size a TPU Training Config

Keep the training config **slice-agnostic**: fix the global batch (and the rest of the
hyperparameters) once, and let the only slice-dependent knob — `per_device_parallelism` —
be derived at submission from `TPUS[tpu]`. Config-time `chips` equals runtime
`jax.device_count()` and `per_device_parallelism` is per chip, so never probe `jax.devices()`.

## Stats

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class TpuStats:
    chips: int      # total accelerator chips in the slice (= jax.device_count())
    hbm_gib: int    # HBM per chip
    tflops: int     # peak bf16 TFLOP/s per chip
    hosts: int      # host VMs in the slice (> 1 ⇒ multi-host)


# HBM/TFLOPS per chip from docs/tpu-stats.md; chips (= chip_count) and hosts
# (= host_count) per fray TPU_TOPOLOGIES — do not guess these, look them up.
# Non-exhaustive — add any slice you use the same way. Slices below the rule are
# single-host (hosts == 1); the rest are multi-host.
TPUS: dict[str, TpuStats] = {
    # ---- single-host ----
    "v4-8":        TpuStats(chips=4,  hbm_gib=32, tflops=275, hosts=1),
    "v5litepod-4": TpuStats(chips=4,  hbm_gib=16, tflops=197, hosts=1),
    "v5litepod-8": TpuStats(chips=8,  hbm_gib=16, tflops=197, hosts=1),
    "v6e-4":       TpuStats(chips=4,  hbm_gib=32, tflops=918, hosts=1),
    "v6e-8":       TpuStats(chips=8,  hbm_gib=32, tflops=918, hosts=1),
    "v5p-8":       TpuStats(chips=4,  hbm_gib=95, tflops=459, hosts=1),
    # ---- multi-host ----
    "v5p-16":      TpuStats(chips=8,  hbm_gib=95, tflops=459, hosts=2),
    "v5p-32":      TpuStats(chips=16, hbm_gib=95, tflops=459, hosts=4),
    "v5p-64":      TpuStats(chips=32, hbm_gib=95, tflops=459, hosts=8),
    "v6e-16":      TpuStats(chips=16, hbm_gib=32, tflops=918, hosts=4),
    "v6e-32":      TpuStats(chips=32, hbm_gib=32, tflops=918, hosts=8),
}
```

## Fixed global batch → gradient accumulation

`train_batch_size` is the global batch; `per_device_parallelism` is the only knob. `-1` ⇒
`batch // devices` (no accumulation); otherwise Levanter derives `microbatch = pdp × chips`
and `accum_steps = batch // microbatch` (`grad_accum.py`). Requires
`batch % (pdp × chips) == 0`.

Given one tuned `per_chip_microbatch` (examples/chip fitting the 16 GiB floor), richer
chips get a static HBM multiple; the largest divisor under the cap yields integer
accumulation:

```python
def per_device_parallelism(tpu: str, global_batch: int, per_chip_microbatch: int, floor_gib: int = 16) -> int:
    s = TPUS[tpu]
    if global_batch % s.chips:
        raise ValueError(f"{global_batch} not divisible by {s.chips} chips ({tpu})")
    cap = per_chip_microbatch * (s.hbm_gib // floor_gib)
    full = global_batch // s.chips  # per-chip load with no accumulation
    if full <= cap:
        return -1
    return next(d for d in range(cap, 0, -1) if full % d == 0)


def gradient_accumulation(tpu: str, global_batch: int, pdp: int) -> int:
    chips = TPUS[tpu].chips
    microbatch = global_batch if pdp < 0 else pdp * chips  # pdp == -1 -> whole batch, 1 step
    return global_batch // microbatch
```

The two config params — `per_device_parallelism` and `gradient_accumulation`:

```python
GB, PCM = 128, 8  # global batch; measured per-chip microbatch
for tpu in ("v5p-8", "v6e-4", "v5litepod-8"):
    pdp = per_device_parallelism(tpu, GB, PCM)
    gacc = gradient_accumulation(tpu, GB, pdp)
    print(tpu, "per_device_parallelism=", pdp, "gradient_accumulation=", gacc)
# v5p-8        per_device_parallelism= -1  gradient_accumulation= 1
# v6e-4        per_device_parallelism= 16  gradient_accumulation= 2
# v5litepod-8  per_device_parallelism=  8  gradient_accumulation= 2
```
