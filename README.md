# marin-agent-kb

Private agent skills and context for the
[marin](https://github.com/marin-community/marin) project, consolidated from personal
notes and gists. Three folders:

- **`skills/`** — repeatable, actionable playbooks an agent invokes to *do* a task.
- **`docs/`** — atomic, static reference an agent *reads* for context; kept current.
- **`guides/`** — opinionated, point-in-time write-ups preserved for historical
  reference; **not maintained** and may be out of date.

## Skills

| Skill | What |
|---|---|
| [`run-iris-job`](skills/run-iris-job.md) | Submit / monitor / stop a job on an Iris cluster. |
| [`monitor-sweep`](skills/monitor-sweep.md) | Loop that auto-resubmits failed/killed sweep runs. |
| [`write-sweep`](skills/write-sweep.md) | Develop a sweep as code: target list raced by workers via `claim_and_run`. |
| [`size-tpu-train-config`](skills/size-tpu-train-config.md) | Slice-agnostic training config: fix global batch, derive per-device parallelism / grad accum per slice. |
| [`eval-checkpoints`](skills/eval-checkpoints.md) | Fan out one idempotent eval job per checkpoint. |
| [`prep-hf-dataset-zephyr`](skills/prep-hf-dataset-zephyr.md) | Download an HF dataset + Zephyr pipeline → GCS parquet. |
| [`setup-dev-vm`](skills/setup-dev-vm.md) | Bootstrap a fresh VM: gcloud/SA, GitHub auth, env, skills. |
| [`clone-marin-branch`](skills/clone-marin-branch.md) | Clone marin onto a new branch at `repos/marin-br/<slug>`. |

## Docs

Named by topical prefix — `setup-`, `tpu-` (hardware reference), `iris-` (execution),
`training-` (Levanter), `ops-` (lifecycle).

| Doc | What |
|---|---|
| [`setup-service-account`](docs/setup-service-account.md) | Create + activate the `eczech-agent` SA. |
| [`tpu-clusters`](docs/tpu-clusters.md) | Cluster → zone → family → HBM/TFLOPS → slice sizes. |
| [`tpu-stats`](docs/tpu-stats.md) | Per-chip bf16 FLOPS + HBM; per-slice peak-FLOPS tables (v5p/v5e/v6e); MFU denominator. |
| [`tpu-device-utilization`](docs/tpu-device-utilization.md) | jax / `tpu-info` HBM & utilization on v5p-8. |
| [`iris-execution-model`](docs/iris-execution-model.md) | Fray vs Iris, TPU allocation, preemption, CPU-driver-vs-direct-on-TPU. |
| [`iris-scheduling-notes`](docs/iris-scheduling-notes.md) | Zone/region control, Executor caveat, dashboards. |
| [`iris-priority-and-quota`](docs/iris-priority-and-quota.md) | Priority bands (batch/interactive/production) + per-user budget formula. |
| [`iris-submit-tpu-job`](docs/iris-submit-tpu-job.md) | Worked TPU-direct `iris job run` example. |
| [`training-continuation`](docs/training-continuation.md) | `initialize_from*` options + continuing the LR schedule. |
| [`training-disable-grad-tracking`](docs/training-disable-grad-tracking.md) | Turn off `WatchConfig` per-param logging. |
| [`ops-delete-tpu-vm`](docs/ops-delete-tpu-vm.md) | Delete a stuck / misconfigured TPU worker VM. |

## Guides

Opinionated, point-in-time write-ups, **not maintained** — for historical reference only
(contrast the atomic, current `docs/`).

| Guide | What |
|---|---|
| [`marin-dev-guide-2026q2`](guides/marin-dev-guide-2026q2.md) | Opinionated Marin user/dev guide, Q2 2026. |
| [`marin-dev-guide-2025q3`](guides/marin-dev-guide-2025q3.md) | Earlier snapshot, Q3 2025 (Ray-era pipeline setup). |

## Conventions

- Source secrets from `~/marin.env`; pass `--user $USERNAME`; use `--cluster marin`.
- Lint via `./infra/pre-commit.py` (never `uv run pre-commit`).
- These complement the marin repo's own `.agents/skills/` — they don't replace them.
