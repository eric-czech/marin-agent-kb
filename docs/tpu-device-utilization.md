# TPU Device Utilization (v5p-8)

What `jax` and `tpu-info` report on a Marin dev TPU — captured on an Iris-allocated v5p-8
in `us-east5-a`. Allocate one with the marin `reserve-tpu` skill or `scripts/iris/dev_tpu.py`
(Driver, below). For slice → chip/HBM and peak FLOPS, see `docs/tpu-clusters.md` and
`docs/tpu-peak-flops-mfu.md`.

The `tpu-info` snapshots below walk one workload through four states —
**idle → starting → steady → idle** — to show what each metric actually reports under load.

## Versions

| Component | Version |
| --- | --- |
| JAX | `0.9.2` |
| jaxlib | `0.9.2` (pinned in `lib/{marin,levanter,fray}/pyproject.toml`) |
| libtpu | `0.0.39` (per `tpu-info` and Marin pyproject pin) |
| `tpu-info` | latest from PyPI (resolved by `uv run --with tpu-info`) |
| Accelerator | TPU v5p (single host, 4 chips, 1 core/chip) |

## Probe scripts

Three throwaway scripts (delete after use); driven from a laptop via `dev_tpu.py`.

### `dev_tpu_probe.py` - one-shot JAX device + HBM probe

```python
"""One-shot probe: print JAX device info and HBM stats. Delete after use."""
import jax
import jax.numpy as jnp

GIB = 1024**3

print(f"jax.__version__       = {jax.__version__}")
print(f"jax.default_backend() = {jax.default_backend()}")
print(f"jax.process_count()   = {jax.process_count()}")
print(f"jax.process_index()   = {jax.process_index()}")
print(f"jax.local_device_count() = {jax.local_device_count()}")
print(f"jax.device_count()       = {jax.device_count()}")
print()
print("jax.devices():")
for d in jax.devices():
    print(f"  {d!r}")
    print(f"    platform={d.platform}  device_kind={d.device_kind}  id={d.id}")
    coords = getattr(d, "coords", None)
    core = getattr(d, "core_on_chip", None)
    if coords is not None or core is not None:
        print(f"    coords={coords}  core_on_chip={core}")

print()
print("Per-device memory_stats() (HBM):")
for d in jax.local_devices():
    try:
        ms = d.memory_stats() or {}
    except Exception as e:
        print(f"  {d}: memory_stats() failed: {e}")
        continue
    limit = ms.get("bytes_limit", 0)
    in_use = ms.get("bytes_in_use", 0)
    peak = ms.get("peak_bytes_in_use", 0)
    reserved = ms.get("bytes_reservable_limit", 0)
    print(
        f"  {d}: limit={limit / GIB:.2f} GiB  "
        f"in_use={in_use / GIB:.3f} GiB  "
        f"peak={peak / GIB:.3f} GiB  "
        f"reservable={reserved / GIB:.2f} GiB"
    )
    print(f"    raw keys: {sorted(ms.keys())}")

print()
print("Forcing a small allocation on every local device so HBM is non-zero...")
arrs = []
for d in jax.local_devices():
    arrs.append(jax.device_put(jnp.ones((512, 512), dtype=jnp.bfloat16), d))
jax.block_until_ready(arrs)

print("Post-allocation memory_stats():")
for d in jax.local_devices():
    ms = d.memory_stats() or {}
    print(
        f"  {d}: in_use={ms.get('bytes_in_use', 0) / GIB:.3f} GiB  "
        f"peak={ms.get('peak_bytes_in_use', 0) / GIB:.3f} GiB"
    )
```

### `dev_tpu_workload.py` - sustained workload to drive duty cycle

```python
"""Sustained JAX workload: keep all 4 v5p chips busy until /tmp/stop_workload exists."""
import os
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

GIB = 1024**3
STOP_FILE = "/tmp/stop_workload"

devices = np.array(jax.local_devices()).reshape(4)
mesh = Mesh(devices, axis_names=("data",))
sharding_replicated = NamedSharding(mesh, P())

# Sizable static buffer (per-chip ~512 MiB bf16) so HBM in_use looks like a real model.
N_STATIC = 16384  # 16384 * 16384 * 2 bytes = 512 MiB per replica
static = jax.device_put(jnp.zeros((N_STATIC, N_STATIC), dtype=jnp.bfloat16), sharding_replicated)
static.block_until_ready()

# Matmul size that keeps the MXU busy.
N = 8192
A = jax.device_put(jnp.ones((N, N), dtype=jnp.bfloat16), sharding_replicated)
B = jax.device_put(jnp.ones((N, N), dtype=jnp.bfloat16), sharding_replicated)


@jax.jit
def step(a, b):
    # 8 chained matmuls per call -> sustained compute.
    for _ in range(8):
        a = a @ b
    return a


out = step(A, B)
out.block_until_ready()

ms = jax.local_devices()[0].memory_stats()
print(f"[workload] warmup done; HBM in_use={ms['bytes_in_use'] / GIB:.2f} GiB / {ms['bytes_limit'] / GIB:.2f} GiB")
print("[workload] READY", flush=True)

iters = 0
t0 = time.time()
while not os.path.exists(STOP_FILE):
    out = step(out, A)
    if iters % 25 == 0:
        out.block_until_ready()
        ms = jax.local_devices()[0].memory_stats()
        elapsed = time.time() - t0
        print(
            f"[workload] iter={iters} elapsed={elapsed:.1f}s "
            f"in_use={ms['bytes_in_use'] / GIB:.2f} GiB "
            f"peak={ms['peak_bytes_in_use'] / GIB:.2f} GiB",
            flush=True,
        )
    iters += 1
out.block_until_ready()
print(f"[workload] stop signalled after {iters} iters; exiting")
try:
    os.remove(STOP_FILE)
except FileNotFoundError:
    pass
sys.exit(0)
```

### `dev_tpu_probe.sh` - orchestrates idle / running / idle snapshots

```bash
#!/usr/bin/env bash
set -u
cd "$HOME/marin"
source "$HOME/.local/bin/env"

TPU_INFO="uv run --with tpu-info --package marin --extra tpu tpu-info"
WORKLOAD_LOG="/tmp/workload.log"
STOP_FILE="/tmp/stop_workload"
rm -f "$STOP_FILE" "$WORKLOAD_LOG"

echo "===== [1] tpu-info (idle, no JAX process) ====="
$TPU_INFO 2>&1 || echo "[tpu-info exit=$?]"

echo
echo "===== [2] launching sustained JAX workload in background ====="
uv run --package marin --extra tpu python dev_tpu_workload.py > "$WORKLOAD_LOG" 2>&1 &
WORKLOAD_PID=$!
echo "workload pid=$WORKLOAD_PID, log=$WORKLOAD_LOG"

# Wait until workload writes "READY" (post-warmup) or dies.
echo "waiting for warmup..."
for i in $(seq 1 90); do
    if grep -q "READY" "$WORKLOAD_LOG" 2>/dev/null; then
        echo "warmup complete after ${i}s"
        break
    fi
    if ! kill -0 "$WORKLOAD_PID" 2>/dev/null; then
        echo "workload died during warmup"
        cat "$WORKLOAD_LOG"
        exit 1
    fi
    sleep 1
done

echo
echo "===== [3] tpu-info (workload running) ====="
$TPU_INFO 2>&1 || echo "[tpu-info exit=$?]"

echo
echo "===== [4] tpu-info (workload still running, second snapshot) ====="
sleep 3
$TPU_INFO 2>&1 || echo "[tpu-info exit=$?]"

echo
echo "===== [5] stopping workload ====="
touch "$STOP_FILE"
wait "$WORKLOAD_PID"
echo "workload exit=$?"
echo "--- workload log (tail) ---"
tail -20 "$WORKLOAD_LOG"

echo
echo "===== [6] tpu-info (idle again, after JAX exit) ====="
$TPU_INFO 2>&1 || echo "[tpu-info exit=$?]"
```

### Driver (local laptop, in the `marin` repo)

```bash
# one-time per session
uv run scripts/iris/dev_tpu.py --config lib/iris/config/marin.yaml \
  --tpu-name "$USER-v5p8-devinfo" allocate --tpu-type v5p-8 --no-setup-env
uv run scripts/iris/dev_tpu.py --config lib/iris/config/marin.yaml \
  --tpu-name "$USER-v5p8-devinfo" setup_env

# JAX-only probe
uv run scripts/iris/dev_tpu.py --config lib/iris/config/marin.yaml \
  --tpu-name "$USER-v5p8-devinfo" execute -- bash -c \
  'cd ~/marin && source ~/.local/bin/env && uv run --package marin --extra tpu python dev_tpu_probe.py'

# idle / running / idle demo
uv run scripts/iris/dev_tpu.py --config lib/iris/config/marin.yaml \
  --tpu-name "$USER-v5p8-devinfo" execute -- bash dev_tpu_probe.sh

# cleanup
uv run scripts/iris/dev_tpu.py --config lib/iris/config/marin.yaml \
  --tpu-name "$USER-v5p8-devinfo" release
```

## JAX-only probe output

```
jax.__version__       = 0.9.2
jax.default_backend() = tpu
jax.process_count()   = 1
jax.process_index()   = 0
jax.local_device_count() = 4
jax.device_count()       = 4

jax.devices():
  TpuDevice(id=0, process_index=0, coords=(0,0,0), core_on_chip=0)
    platform=tpu  device_kind=TPU v5  id=0
    coords=[0, 0, 0]  core_on_chip=0
  TpuDevice(id=1, process_index=0, coords=(1,0,0), core_on_chip=0)
    platform=tpu  device_kind=TPU v5  id=1
    coords=[1, 0, 0]  core_on_chip=0
  TpuDevice(id=2, process_index=0, coords=(0,1,0), core_on_chip=0)
    platform=tpu  device_kind=TPU v5  id=2
    coords=[0, 1, 0]  core_on_chip=0
  TpuDevice(id=3, process_index=0, coords=(1,1,0), core_on_chip=0)
    platform=tpu  device_kind=TPU v5  id=3
    coords=[1, 1, 0]  core_on_chip=0

Per-device memory_stats() (HBM):
  TPU_0(process=0,(0,0,0,0)): limit=95.74 GiB  in_use=0.000 GiB  peak=0.000 GiB  reservable=95.74 GiB
    raw keys: ['bytes_in_use', 'bytes_limit', 'bytes_reservable_limit', 'bytes_reserved', 'largest_alloc_size', 'largest_free_block_bytes', 'num_allocs', 'peak_bytes_in_use', 'peak_bytes_reserved']
  TPU_1(process=0,(1,0,0,0)): limit=95.74 GiB  in_use=0.000 GiB  peak=0.000 GiB  reservable=95.74 GiB
  TPU_2(process=0,(0,1,0,0)): limit=95.74 GiB  in_use=0.000 GiB  peak=0.000 GiB  reservable=95.74 GiB
  TPU_3(process=0,(1,1,0,0)): limit=95.74 GiB  in_use=0.000 GiB  peak=0.000 GiB  reservable=95.74 GiB

Post-allocation memory_stats() (after device_put of 512x512 bf16 on each chip):
  TPU_0(process=0,(0,0,0,0)): in_use=0.001 GiB  peak=0.002 GiB
  TPU_1(process=0,(1,0,0,0)): in_use=0.001 GiB  peak=0.001 GiB
  TPU_2(process=0,(0,1,0,0)): in_use=0.001 GiB  peak=0.001 GiB
  TPU_3(process=0,(1,1,0,0)): in_use=0.001 GiB  peak=0.001 GiB
```

## `tpu-info` under three states

### [1] idle - no JAX process

```
Libtpu version: 0.0.39
Accelerator type: v5p

TPU Chips
┏━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━┓
┃ Chip        ┃ Type         ┃ Devices ┃ PID  ┃
┡━━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━┩
│ /dev/vfio/0 │ TPU v5p chip │ 1       │ None │
│ /dev/vfio/1 │ TPU v5p chip │ 1       │ None │
│ /dev/vfio/2 │ TPU v5p chip │ 1       │ None │
│ /dev/vfio/3 │ TPU v5p chip │ 1       │ None │
└─────────────┴──────────────┴─────────┴──────┘
╭───────────────────────── Runtime Utilization Status ─────────────────────────╮
│ WARNING: Libtpu metrics unavailable. Is there a framework using the TPU? See │
│ tpu_info docs for more information.                                          │
╰──────────────────────────────────────────────────────────────────────────────╯
TPU Runtime Utilization
┏━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┓
┃ Chip ┃ HBM Usage (GiB) ┃ Duty cycle ┃
┡━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━┩
│ 0    │ N/A             │ N/A        │
│ 1    │ N/A             │ N/A        │
│ 2    │ N/A             │ N/A        │
│ 3    │ N/A             │ N/A        │
└──────┴─────────────────┴────────────┘
TensorCore Utilization
┏━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Core ID ┃ TensorCore Utilization ┃
┡━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━┩
│ 0       │ 0.00%                  │
│ 1       │ 0.00%                  │
│ 2       │ 0.00%                  │
│ 3       │ 0.00%                  │
└─────────┴────────────────────────┘
```

### [3] JAX workload just attached - TensorCore live, HBM/duty cycle not yet sampled

```
Libtpu version: 0.0.39
Accelerator type: v5p

TPU Chips
┏━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━┓
┃ Chip        ┃ Type         ┃ Devices ┃ PID    ┃
┡━━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━┩
│ /dev/vfio/0 │ TPU v5p chip │ 1       │ 529279 │
│ /dev/vfio/1 │ TPU v5p chip │ 1       │ 529279 │
│ /dev/vfio/2 │ TPU v5p chip │ 1       │ 529279 │
│ /dev/vfio/3 │ TPU v5p chip │ 1       │ 529279 │
└─────────────┴──────────────┴─────────┴────────┘
TPU Runtime Utilization
┏━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┓
┃ Chip ┃ HBM Usage (GiB)      ┃ Duty cycle ┃
┡━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━┩
│ 0    │ 0.00 GiB / 95.74 GiB │ 0.00%      │
│ 1    │ 0.00 GiB / 95.74 GiB │ 0.00%      │
│ 2    │ 0.00 GiB / 95.74 GiB │ 0.00%      │
│ 3    │ 0.00 GiB / 95.74 GiB │ 0.00%      │
└──────┴──────────────────────┴────────────┘
TensorCore Utilization
┏━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Core ID ┃ TensorCore Utilization ┃
┡━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━┩
│ 0       │ 80.59%                 │
│ 1       │ 64.01%                 │
│ 2       │ 85.93%                 │
│ 3       │ 91.98%                 │
└─────────┴────────────────────────┘
```

### [4] JAX workload steady-state (3s later) - metrics fully populated

```
Libtpu version: 0.0.39
Accelerator type: v5p

TPU Chips
┏━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━┓
┃ Chip        ┃ Type         ┃ Devices ┃ PID    ┃
┡━━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━┩
│ /dev/vfio/0 │ TPU v5p chip │ 1       │ 529279 │
│ /dev/vfio/1 │ TPU v5p chip │ 1       │ 529279 │
│ /dev/vfio/2 │ TPU v5p chip │ 1       │ 529279 │
│ /dev/vfio/3 │ TPU v5p chip │ 1       │ 529279 │
└─────────────┴──────────────┴─────────┴────────┘
TPU Runtime Utilization
┏━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┓
┃ Chip ┃ HBM Usage (GiB)      ┃ Duty cycle ┃
┡━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━┩
│ 0    │ 1.26 GiB / 95.74 GiB │ 71.11%     │
│ 1    │ 1.26 GiB / 95.74 GiB │ 71.09%     │
│ 2    │ 1.26 GiB / 95.74 GiB │ 71.09%     │
│ 3    │ 1.26 GiB / 95.74 GiB │ 71.09%     │
└──────┴──────────────────────┴────────────┘
TensorCore Utilization
┏━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Core ID ┃ TensorCore Utilization ┃
┡━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━┩
│ 0       │ 93.48%                 │
│ 1       │ 93.46%                 │
│ 2       │ 93.45%                 │
│ 3       │ 93.48%                 │
└─────────┴────────────────────────┘
```

Workload log (steady-state) showed JAX-side accounting:

```
[workload] warmup done; HBM in_use=0.88 GiB / 95.74 GiB
[workload] iter=0 elapsed=0.0s in_use=0.88 GiB peak=1.01 GiB
[workload] iter=25 elapsed=0.5s in_use=0.88 GiB peak=4.01 GiB
[workload] iter=50 elapsed=1.0s in_use=0.88 GiB peak=4.01 GiB
[workload] iter=75 elapsed=1.5s in_use=0.88 GiB peak=4.01 GiB
[workload] iter=100 elapsed=2.1s in_use=0.88 GiB peak=4.01 GiB
[workload] iter=125 elapsed=2.6s in_use=0.88 GiB peak=4.01 GiB
[workload] iter=150 elapsed=3.1s in_use=0.88 GiB peak=4.01 GiB
[workload] iter=175 elapsed=3.6s in_use=0.88 GiB peak=4.01 GiB
[workload] iter=200 elapsed=4.1s in_use=0.88 GiB peak=4.01 GiB
[workload] iter=225 elapsed=4.6s in_use=0.88 GiB peak=4.01 GiB
[workload] iter=250 elapsed=5.1s in_use=0.88 GiB peak=4.01 GiB
[workload] stop signalled after 251 iters; exiting
```

### [6] idle again, after JAX exit

```
Libtpu version: 0.0.39
Accelerator type: v5p

TPU Chips
┏━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━┓
┃ Chip        ┃ Type         ┃ Devices ┃ PID  ┃
┡━━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━┩
│ /dev/vfio/0 │ TPU v5p chip │ 1       │ None │
│ /dev/vfio/1 │ TPU v5p chip │ 1       │ None │
│ /dev/vfio/2 │ TPU v5p chip │ 1       │ None │
│ /dev/vfio/3 │ TPU v5p chip │ 1       │ None │
└─────────────┴──────────────┴─────────┴──────┘
TPU Runtime Utilization
┏━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┓
┃ Chip ┃ HBM Usage (GiB) ┃ Duty cycle ┃
┡━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━┩
│ 0    │ N/A             │ N/A        │
│ 1    │ N/A             │ N/A        │
│ 2    │ N/A             │ N/A        │
│ 3    │ N/A             │ N/A        │
└──────┴─────────────────┴────────────┘
TensorCore Utilization
┏━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Core ID ┃ TensorCore Utilization ┃
┡━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━┩
│ 0       │ 0.00%                  │
│ 1       │ 0.00%                  │
│ 2       │ 0.00%                  │
│ 3       │ 0.00%                  │
└─────────┴────────────────────────┘
```

## Takeaways

- **JAX vs `tpu-info` HBM**: JAX reports `0.88 GiB in_use`, `tpu-info` reports `1.26 GiB`. JAX accounts only for user-allocated buffers; `tpu-info` reads libtpu's HBM total, which includes runtime overhead the framework doesn't see.
- **Counter warm-up matters**: snapshot [3] taken right after the workload started shows `TensorCore Utilization` already populated but `HBM Usage`/`Duty cycle` still at zeros - the libtpu metrics server needs a sampling window of a few seconds before those columns mean anything. Snapshot [4] (3s later) shows the real steady-state values.
- **PID column is the most useful field for triage**: it tells you which OS process owns each `/dev/vfio/N`. When chips look "stuck", that's where you start.
- **`HBM Usage` row disappears entirely when no framework is attached** (snapshots [1] and [6] show `N/A`) - so `tpu-info` on an idle TPU is not useful for capacity planning. Use `jax.local_devices()[0].memory_stats()['bytes_limit']` for the chip's actual HBM size (95.74 GiB on v5p — cf. the 95 GiB/chip in `docs/tpu-clusters.md`).
- **TensorCore vs Duty cycle**: TensorCore Utilization is what fraction of MXU cycles did useful work; Duty cycle is what fraction of wall time the chip was active. Steady state here: 93% MXU at 71% duty - i.e., the chip is busy 71% of the time, and when busy it's keeping the MXU 93% saturated. Gap between them is fed by the inner `for _ in range(8): a = a @ b` Python-side scheduling overhead.
