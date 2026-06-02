# TPU Device Utilization (v5p-8)

What `jax` and `tpu-info` report on a Marin dev TPU (Iris-allocated v5p-8). Versions
seen: JAX/jaxlib `0.9.2`, libtpu `0.0.39`; v5p = single host, 4 chips, 1 core/chip.

## Allocate a dev TPU

```bash
uv run scripts/iris/dev_tpu.py --config lib/iris/config/marin.yaml \
  --tpu-name "$USER-v5p8" allocate --tpu-type v5p-8 --no-setup-env
# then: setup_env  |  execute -- <cmd>  |  release   (when done)
```

(See the marin repo's `reserve-tpu` skill for the full reserve workflow.)

## Probing

- **JAX:** `jax.devices()` and `d.memory_stats()` (keys `bytes_limit`, `bytes_in_use`,
  `peak_bytes_in_use`, …). A v5p chip reports `bytes_limit ≈ 95.74 GiB`.
- **libtpu:** `uv run --with tpu-info --package marin --extra tpu tpu-info` — per-chip
  HBM, Duty cycle, TensorCore Utilization, and the owning PID.

## Takeaways

- **JAX vs tpu-info HBM differ:** JAX counts only user buffers; `tpu-info` reads
  libtpu's total, which includes runtime overhead the framework doesn't see.
- **Counters warm up:** right after a workload attaches, TensorCore Utilization is live
  but HBM/Duty cycle still read zero — they need a few-second sampling window.
- **`HBM Usage` shows `N/A` when no framework is attached** — `tpu-info` on an idle TPU
  is useless for capacity. Use `memory_stats()['bytes_limit']` (95.74 GiB on v5p).
- **PID column is the best triage field:** it tells you which process owns each
  `/dev/vfio/N` when chips look stuck.
- **TensorCore Util vs Duty cycle:** TensorCore = fraction of MXU cycles doing useful
  work; Duty cycle = fraction of wall time the chip was active. E.g. 93% MXU at 71%
  duty = busy 71% of the time, MXU 93% saturated when busy.
