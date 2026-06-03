---
name: size-tpu-train-config
description: Keep a training config slice-agnostic — fix the global batch, derive per-device parallelism / gradient accumulation per TPU slice.
---

# Size a TPU Training Config

Keep the training config **slice-agnostic**: fix the global batch (and the rest of the
hyperparameters) once, and let the only slice-dependent knob — `per_device_parallelism` —
be derived at submission from `TPU_STATS[tpu]`. Config-time `chips` equals runtime
`jax.device_count()` and `per_device_parallelism` is per chip, so never probe `jax.devices()`.

## Stats

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class TpuStats:
    chips: int      # total accelerator chips in the slice (= jax.device_count())
    hbm_gib: int    # HBM per chip
    tflops: int     # peak bf16 TFLOP/s per chip


# HBM/TFLOPS per chip from docs/tpu-clusters.md; chips per fray TPU_TOPOLOGIES
# (N/2 for v4/v5p, N for v5e/v6e). Non-exhaustive — add any slice you use the same
# way (multi-host included).
TPU_STATS: dict[str, TpuStats] = {
    "v4-8":        TpuStats(chips=4, hbm_gib=32, tflops=275),
    "v5litepod-1": TpuStats(chips=1, hbm_gib=16, tflops=197),
    "v5litepod-2": TpuStats(chips=2, hbm_gib=16, tflops=197),
    "v5litepod-4": TpuStats(chips=4, hbm_gib=16, tflops=197),
    "v5litepod-8": TpuStats(chips=8, hbm_gib=16, tflops=197),
    "v5p-8":       TpuStats(chips=4, hbm_gib=95, tflops=459),
    "v6e-1":       TpuStats(chips=1, hbm_gib=32, tflops=918),
    "v6e-4":       TpuStats(chips=4, hbm_gib=32, tflops=918),
    "v6e-8":       TpuStats(chips=8, hbm_gib=32, tflops=918),
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
    s = TPU_STATS[tpu]
    if global_batch % s.chips:
        raise ValueError(f"{global_batch} not divisible by {s.chips} chips ({tpu})")
    cap = per_chip_microbatch * (s.hbm_gib // floor_gib)
    full = global_batch // s.chips                      # per-chip load with no accumulation
    if full <= cap:
        return -1
    return next(d for d in range(cap, 0, -1) if full % d == 0)


def gradient_accumulation(tpu: str, global_batch: int, pdp: int) -> int:
    chips = TPU_STATS[tpu].chips
    microbatch = global_batch if pdp < 0 else pdp * chips   # pdp == -1 -> whole batch, 1 step
    return global_batch // microbatch
```

The two config params — `per_device_parallelism` and `gradient_accumulation`:

```python
GB, PCM = 128, 8                                  # global batch; measured per-chip microbatch
for tpu in ("v5p-8", "v6e-4", "v5litepod-8"):
    pdp = per_device_parallelism(tpu, GB, PCM)
    gacc = gradient_accumulation(tpu, GB, pdp)
    print(tpu, "per_device_parallelism=", pdp, "gradient_accumulation=", gacc)
# v5p-8        per_device_parallelism= -1  gradient_accumulation= 1
# v6e-4        per_device_parallelism= 16  gradient_accumulation= 2
# v5litepod-8  per_device_parallelism=  8  gradient_accumulation= 2
```
