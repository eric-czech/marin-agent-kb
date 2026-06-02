# TPU Clusters

Iris cluster configs live in `lib/iris/config/(marin|marin-dev).yaml` and define
these TPU slice types. Use this to pick a `--region`/`--zone` that actually has the
slice you want (Iris won't move a running job to follow availability — see
`docs/iris-scheduling-notes.md`).

| Cluster | Zone | Family | HBM/chip | TFLOPS/chip | Single-VM | All slice sizes |
|---|---|---|---:|---:|---|---|
| `big-run` | us-central2-b | v4 | 32 GiB | 275 | v4-8 | `v4-{8,16,32,64,128,256,512,1024,2048,4096}` |
| `eu-west4` | europe-west4-b | v5e | 16 GB | 197 | v5litepod-4, -8 | `v5litepod-{4,8,16,32,64,128,256}` |
| `eu-west4-a` | europe-west4-a | v6e | 32 GB | 918 | v6e-4, v6e-8 | `v6e-{4,8,16,32,64,128,256}` |
| `us-central1` | us-central1-a | v5p | 95 GiB | 459 | v5p-8 | `v5p-{8,16,32,64,128,256,512,1024,2048}` |
| `us-central2` | us-central2-b | v4 | 32 GiB | 275 | v4-8 | `v4-{8,16,32,64,128,256,512,1024,2048,4096}` |
| `us-east1` | us-east1-d | v6e | 32 GB | 918 | v6e-4, v6e-8 | `v6e-{4,8,16,32,64,128,256}` |
| `us-east5` | us-east5-b | v6e | 32 GB | 918 | v6e-4, v6e-8 | `v6e-{4,8,16,32,64,128,256}` |
| `us-east5-a` | us-east5-a | v5p | 95 GiB | 459 | v5p-8 | `v5p-{8,16,32,64,128,256,512,1024,2048}` |
| `us-west4` | us-west4-a | v5e | 16 GB | 197 | v5litepod-4, -8 | `v5litepod-{4,8,16,32,64,128,256}` |

Slice column shorthand: `family-{N, …}` expands to one slice per `N` (e.g. `v5p-{8,16}` =
`v5p-8`, `v5p-16`). Chips per slice = `N/2` for v4/v5p, `N` for v6e/v5e.

**HBM sources** (Google TPU docs): v4 32 GiB/chip, v5e 16 GB/chip, v5p 95 GiB/chip,
v6e 32 GB/chip.
