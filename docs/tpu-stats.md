# TPU Stats

Per-chip TPU stats — peak bf16 FLOPS and HBM — plus per-slice peak-FLOPS totals. Peak
FLOPS is the **MFU denominator** in Marin/Levanter training (MFU = model FLOPs /
(per-device peak × device count)); these are *theoretical peak* rates, not achieved
throughput.

Source of truth (no hand-derived constants):
- Chip counts: `fray.types.TPU_TOPOLOGIES` (`lib/fray/src/fray/types.py`)
- Per-device peak FLOPS: `fray.device_flops.DEVICE_FLOPS` (`lib/fray/src/fray/device_flops.py`)
- HBM/chip: Google TPU docs (mirrored in `docs/tpu-clusters.md`)

## Per-chip stats

| Device | bf16 peak / chip | HBM / chip | Notes |
|---|---:|---:|---|
| TPU v5p | **459 TFLOP/s** | 95 GiB | dense, no sparsity inflation |
| TPU v5e (`v5litepod-*`) | **197 TFLOP/s** | 16 GB | dense |
| TPU v6e | **918 TFLOP/s** | 32 GB | dense |
| NVIDIA H100 | **989.5 TFLOP/s** | 80 GB | `1.979e15 / 2` — de-rated from NVIDIA's 2:4-sparsity quote to the dense rate used for dense training |

Chips per slice: **N/2** for `v5p` (e.g. `v5p-8` = 4), **N** for `v5e`/`v6e` (e.g. `v6e-8` = 8), per `TPU_TOPOLOGIES`.
Total slice HBM = chips × HBM/chip. H100-equiv = slice peak / 989.5 TFLOP/s (≈ 0.464 H100
per v5p chip) — a pure dense peak-FLOPS ratio; it ignores memory, interconnect, and real utilization.

## v5p slices — peak bf16

| Slice | Chips | Peak TFLOP/s | Peak PFLOP/s | Total HBM (GiB) | H100-equiv |
|---|---:|---:|---:|---:|---:|
| v5p-8 | 4 | 1,836 | 1.836 | 380 | 1.86 |
| v5p-16 | 8 | 3,672 | 3.672 | 760 | 3.71 |
| v5p-32 | 16 | 7,344 | 7.344 | 1,520 | 7.42 |
| v5p-64 | 32 | 14,688 | 14.688 | 3,040 | 14.84 |
| v5p-128 | 64 | 29,376 | 29.376 | 6,080 | 29.69 |
| v5p-256 | 128 | 58,752 | 58.752 | 12,160 | 59.38 |
| v5p-512 | 256 | 117,504 | 117.504 | 24,320 | 118.75 |
| v5p-1024 | 512 | 235,008 | 235.008 | 48,640 | 237.50 |
| v5p-2048 | 1024 | 470,016 | 470.016 | 97,280 | 475.00 |
| v5p-4096 | 2048 | 940,032 | 940.032 | 194,560 | 950.01 |
| v5p-8192 | 4096 | 1,880,064 | 1,880.064 | 389,120 | 1,900.01 |
| v5p-12288 | 6144 | 2,820,096 | 2,820.096 | 583,680 | 2,850.02 |

## v5e (`v5litepod-N`) slices — peak bf16

| Slice | Chips | Peak TFLOP/s | Peak PFLOP/s | Total HBM (GB) | H100-equiv |
|---|---:|---:|---:|---:|---:|
| v5litepod-1 | 1 | 197 | 0.197 | 16 | 0.20 |
| v5litepod-2 | 2 | 394 | 0.394 | 32 | 0.40 |
| v5litepod-4 | 4 | 788 | 0.788 | 64 | 0.80 |
| v5litepod-8 | 8 | 1,576 | 1.576 | 128 | 1.59 |
| v5litepod-16 | 16 | 3,152 | 3.152 | 256 | 3.19 |
| v5litepod-32 | 32 | 6,304 | 6.304 | 512 | 6.37 |
| v5litepod-64 | 64 | 12,608 | 12.608 | 1,024 | 12.74 |
| v5litepod-128 | 128 | 25,216 | 25.216 | 2,048 | 25.48 |
| v5litepod-256 | 256 | 50,432 | 50.432 | 4,096 | 50.97 |

## v6e slices — peak bf16

| Slice | Chips | Peak TFLOP/s | Peak PFLOP/s | Total HBM (GB) | H100-equiv |
|---|---:|---:|---:|---:|---:|
| v6e-1 | 1 | 918 | 0.918 | 32 | 0.93 |
| v6e-4 | 4 | 3,672 | 3.672 | 128 | 3.71 |
| v6e-8 | 8 | 7,344 | 7.344 | 256 | 7.42 |
| v6e-16 | 16 | 14,688 | 14.688 | 512 | 14.84 |
| v6e-32 | 32 | 29,376 | 29.376 | 1,024 | 29.69 |
| v6e-64 | 64 | 58,752 | 58.752 | 2,048 | 59.38 |
| v6e-128 | 128 | 117,504 | 117.504 | 4,096 | 118.75 |
| v6e-256 | 256 | 235,008 | 235.008 | 8,192 | 237.50 |
