# TPU Peak FLOPS & MFU

Theoretical peak bf16 FLOPS used as the **MFU denominator** in Marin/Levanter
training. MFU = model FLOPs / (per-device peak × device count). These are
*theoretical peak* rates (the denominator), not achieved throughput.

Source of truth (no hand-derived constants):
- Chip counts: `fray.types.TPU_TOPOLOGIES` (`lib/fray/src/fray/types.py`)
- Per-device peak: `fray.device_flops.DEVICE_FLOPS` (`lib/fray/src/fray/device_flops.py`)

## Key constants

| Device | Per-device bf16 peak | Notes |
|---|---:|---|
| TPU v5p (per chip) | **459 TFLOP/s** | dense, no sparsity inflation |
| NVIDIA H100 (per GPU) | **989.5 TFLOP/s** | `1.979e15 / 2` — de-rated from NVIDIA's 2:4-sparsity quote to the dense rate used for dense training |

For `v5p-N`, **chips = N/2** (e.g. `v5p-8` = 4 chips), per `TPU_TOPOLOGIES`.
H100-equiv = slice peak / 989.5 TFLOP/s (≈ 0.464 H100 per v5p chip) — a pure dense
peak-FLOPS ratio; it ignores memory, interconnect, and real utilization.

## v5p slices — peak bf16

| Slice | Chips | Peak TFLOP/s | Peak PFLOP/s | H100-equiv |
|---|---:|---:|---:|---:|
| v5p-8 | 4 | 1,836 | 1.836 | 1.86 |
| v5p-16 | 8 | 3,672 | 3.672 | 3.71 |
| v5p-32 | 16 | 7,344 | 7.344 | 7.42 |
| v5p-64 | 32 | 14,688 | 14.688 | 14.84 |
| v5p-128 | 64 | 29,376 | 29.376 | 29.69 |
| v5p-256 | 128 | 58,752 | 58.752 | 59.38 |
| v5p-512 | 256 | 117,504 | 117.504 | 118.75 |
| v5p-1024 | 512 | 235,008 | 235.008 | 237.50 |
| v5p-2048 | 1024 | 470,016 | 470.016 | 475.00 |
| v5p-4096 | 2048 | 940,032 | 940.032 | 950.01 |
| v5p-8192 | 4096 | 1,880,064 | 1,880.064 | 1,900.01 |
| v5p-12288 | 6144 | 2,820,096 | 2,820.096 | 2,850.02 |
