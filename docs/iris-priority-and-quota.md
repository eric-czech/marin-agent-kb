# Iris Priority Bands & Quota

Iris has three **priority bands** (`lib/iris/src/iris/rpc/job.proto:578`), set with
`--priority <band>`:

| Band | Use | Preemptible | Counts vs quota |
|---|---|---|---|
| `production` | supersedes daily work; admin-only | never preempted | counts toward spend, but never downgraded |
| `interactive` | normal work — **the default** | yes | **yes** |
| `batch` | opportunistic background | yes | **no** |

An unspecified band resolves to `interactive`.

## Quota (per-user budget)

Spend is computed per user over active tasks and capped by a per-user **budget limit**
(set in cluster config — **currently ~75,000**; `0` = unlimited). The relevant module is
`lib/iris/src/iris/cluster/controller/budget.py`:

- **Cost per task** (`resource_value`): `1000 × accel_chips + RAM_GB + 5 × CPU_cores`
  (integer-truncated), summed over a job's tasks (= VMs).
- `batch` jobs are **excluded** from spend; `production` is included but **never downgraded**.
- When a user is **over budget**, their `interactive` tasks are downgraded to `batch`
  (preemptible) until spend drops — `production` is immune.

The `1000 × chips` term dominates, so quota use ≈ **chips in flight × 1000** (≈75 chips at ~75,000).

## Inspect budgets

`iris --cluster marin budget list` prints each user's limit, current spend, and max band
(`budget get <user>` for one user; `budget set` is admin-only). Use it to see how much of
your quota is in flight. The exact columns may change — what matters is that the command exists.

## Examples

Chip counts per `TPU_TOPOLOGIES` (see `docs/tpu-clusters.md`). With `with_tpu` defaults
(32 cores, 128 GiB RAM) the RAM/CPU add-on is `128 + 5×32 = 288` per VM:

| Slice | Chips | Cost ≈ | Concurrent at ~75,000 |
|---|---:|---:|---:|
| `v5p-8` | 4 | 4,288 | ~17 |
| `v6e-8` | 8 | 8,288 | ~9 |
| `v5e-8` (`v5litepod-8`) | 8 | 8,288 | ~9 |

See also `docs/iris-execution-model.md` (bands drive preemption/retries) and
`docs/tpu-stats.md` (chips per slice).
