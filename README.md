# marin-agent-kb

Private agent skills and context for the
[marin](https://github.com/marin-community/marin) project, consolidated from personal
notes and gists. Two folders:

- **`skills/`** — repeatable, actionable playbooks an agent invokes to *do* a task.
- **`docs/`** — static reference an agent *reads* for context.

## Skills

| Skill | What |
|---|---|
| `run-iris-job` | Submit / monitor / stop a job on an Iris cluster. |
| `monitor-sweep` | Loop that auto-resubmits failed/killed sweep runs. |
| `write-sweep` | Develop a sweep as code: target list raced by workers via `claim_and_run`. |
| `eval-checkpoints-offline` | Fan out one idempotent eval job per checkpoint. |
| `prep-hf-dataset-zephyr` | Download an HF dataset + Zephyr pipeline → GCS parquet. |
| `setup-dev-vm` | Bootstrap a fresh VM: gcloud/SA, GitHub auth, env, skills. |
| `clone-marin-branch` | Clone marin onto a new branch at `repos/marin-br/<slug>`. |

## Docs

Named by topical prefix — `setup-`, `tpu-` (hardware reference), `iris-` (execution),
`training-` (Levanter), `ops-` (lifecycle).

| Doc | What |
|---|---|
| `setup-dev-environment` | Local repo setup: clone, buckets, HF Pro, auth, `make dev_setup`. |
| `setup-service-account` | Create + activate the `eczech-agent` SA. |
| `tpu-clusters` | Cluster → zone → family → HBM/TFLOPS → slice sizes. |
| `tpu-peak-flops-mfu` | Peak bf16 FLOPS + MFU denominator; v5p slice table. |
| `tpu-device-utilization` | jax / `tpu-info` HBM & utilization on v5p-8. |
| `iris-execution-model` | Fray vs Iris, TPU allocation, preemption, CPU-driver-vs-direct-on-TPU. |
| `iris-scheduling-notes` | Zone/region control, Executor caveat, dashboards. |
| `iris-priority-and-quota` | Priority bands (batch/interactive/production) + per-user budget formula. |
| `iris-submit-tpu-job` | Worked TPU-direct `iris job run` example. |
| `training-continuation` | `initialize_from*` options + LR re-warmup fix. |
| `training-disable-grad-tracking` | Turn off `WatchConfig` per-param logging. |
| `ops-delete-tpu-vm` | Delete a stuck / misconfigured TPU worker VM. |

## Conventions

- Source secrets from `~/marin.env`; pass `--user $USERNAME`; use `--cluster marin`.
- Lint via `./infra/pre-commit.py` (never `uv run pre-commit`).
- These complement the marin repo's own `.agents/skills/` — they don't replace them.
