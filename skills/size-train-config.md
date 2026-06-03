---
name: size-train-config
description: Keep a training config slice-agnostic — fix the global batch, derive per-device parallelism / gradient accumulation per TPU slice.
---

# Size a Training Config

Keep the training config **slice-agnostic**: fix the global batch (and the rest of the
hyperparameters) once, and let the only slice-dependent knob — `per_device_parallelism` —
be derived at submission from `SINGLE_HOST[tpu]`. Config-time `chips` equals runtime
`jax.device_count()` and `per_device_parallelism` is per chip, so never probe `jax.devices()`.

## Stats

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class TpuStats:
    chips: int      # accelerator chips on the single host
    hbm_gib: int    # HBM per chip
    tflops: int     # peak bf16 TFLOP/s per chip


# Single-host slices. HBM/TFLOPS per chip from docs/tpu-clusters.md; chips per fray
# TPU_TOPOLOGIES (N/2 for v4/v5p, N for v5e/v6e). Add slices as needed.
SINGLE_HOST: dict[str, TpuStats] = {
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
    s = SINGLE_HOST[tpu]
    if global_batch % s.chips:
        raise ValueError(f"{global_batch} not divisible by {s.chips} chips ({tpu})")
    cap = per_chip_microbatch * (s.hbm_gib // floor_gib)
    full = global_batch // s.chips                      # per-chip load with no accumulation
    if full <= cap:
        return -1
    return next(d for d in range(cap, 0, -1) if full % d == 0)
```

The two config params this yields — `per_device_parallelism` and the implied grad-accum
(`batch // (pdp × chips)`, `1` when `pdp == -1`):

```python
GB, PCM = 128, 8                                  # global batch; measured per-chip microbatch
for tpu in ("v5p-8", "v6e-4", "v5litepod-8"):
    chips = SINGLE_HOST[tpu].chips
    pdp = per_device_parallelism(tpu, GB, PCM)
    microbatch = GB if pdp < 0 else pdp * chips
    print(tpu, "per_device_parallelism=", pdp, "grad_accum=", GB // microbatch)
# v5p-8        per_device_parallelism= -1  grad_accum= 1
# v6e-4        per_device_parallelism= 16  grad_accum= 2
# v5litepod-8  per_device_parallelism=  8  grad_accum= 2
```

## Scope

Single-host only; no memory estimation (`per_chip_microbatch` is measured). The
`hbm // floor` multiple is conservative — override per slice if a chip deviates.
