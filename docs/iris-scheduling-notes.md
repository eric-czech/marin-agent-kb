# Iris Scheduling Notes

Operational notes for running jobs on Iris (all marin scheduling goes through Iris;
Ray is gone). See `skills/run-iris-job.md` for the submit command.

## Region / zone control

Iris **starts** jobs flexibly across regions but won't resume or move a job to follow
TPU availability. Once a run has made progress in one region, you're usually better
off pinning a region/zone that has the slice and scheduling there explicitly:

- Pass `--zone <zone>` (and `--reserve` when needed) to control placement.
- Current availability: https://iris.oa.dev/#/autoscaler
- Full TPU-direct example: `docs/submit-job-on-tpu-vm.md`; zone↔TPU map: `docs/tpu-clusters.md`.

## Executor

The Executor now runs **with** training jobs rather than spawning them. Running a
multi-step sweep via `executor_main` directly on a preemptible TPU VM fails with
`RuntimeError: distributed.initialize should only be called once` — you can't train
multiple times in one Python process. Submit one job per trial instead
(see `skills/design-tpu-sweep.md`).

## Dashboards

- Prod: https://iris.oa.dev/ — config `lib/iris/config/marin.yaml`
- Dev: https://iris-dev.oa.dev/ — `marin-dev.yaml`; controller updated more often but
  stricter TPU limits. Mainly useful if the iris protobuf schema drifts from your branch.
- Local (if public endpoints unreachable): `uv run iris --cluster marin cluster dashboard`.
