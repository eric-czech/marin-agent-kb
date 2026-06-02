---
name: run-iris-job
description: Submit, monitor, and stop a job on an Iris cluster.
---

# Run Iris Job

## Required context

Infer from the user; if CLUSTER or SCRIPT is unclear, ask first.

- **CLUSTER** — `--cluster <name>`, resolved to `lib/iris/config/<name>.yaml`.
  Valid: `marin` (prod, default), `marin-dev` (dev, smaller caps, restarted often),
  `coreweave` / `coreweave-rno2a` / `coreweave-usw09b` (GPU). `iris cluster list` shows all.
- **SCRIPT** — Python script path, e.g. `experiments/dna/exp109_bolinas_sweep_eval.py`.

## Optional context

- **USERNAME** (`--user`) — resolve in order, first match wins:
  1. Value the operator gave in context.
  2. `$USERNAME` env var, if set.
  3. `USERNAME` in `~/marin.env`, if present.
  4. System user (`whoami`), only if not generic (`ubuntu`, `exedev`, `root`, …).

  If none yields a real username (only a generic system user remains), **stop and ask** —
  don't guess.
- **REGION** — restrict scheduling: omit (Iris picks) or repeat `--region us-central1`.
- Extra env vars. Wait behavior defaults to `--no-wait`.

## Submit

```bash
source ~/marin.env && uv run iris --cluster marin job run \
  --user <USERNAME> \
  --no-wait \
  -e WANDB_API_KEY $WANDB_API_KEY \
  -e HUGGING_FACE_HUB_TOKEN $HUGGING_FACE_HUB_TOKEN \
  <EXTRA_ENV_VARS> \
  -- python <SCRIPT>
```

## Monitor / stop

```bash
iris --cluster marin job logs <JOB_ID> --follow             # stream
iris --cluster marin job logs <JOB_ID> --since-seconds 300  # recent
iris --cluster marin job stop <JOB_ID>                       # stop + children (kill = alias)
iris --cluster marin job stop <JOB_ID> --no-include-children
```

Job IDs are namespaced paths (e.g. `/myuser/my-job`).

## Conventions

- `WANDB_API_KEY` + `HUGGING_FACE_HUB_TOKEN` come from `~/marin.env`, always passed via `-e`.
- TPU/GPU type and resources belong in the Python script, not the CLI.
- After submit, report: Job ID, cluster, region(s).

For TPU-direct submission with `--zone`/`--priority` control, see `docs/iris-submit-tpu-job.md`.

$ARGUMENTS
