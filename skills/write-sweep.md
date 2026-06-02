---
name: write-sweep
description: Develop a config/hyper-parameter sweep as code — a target list raced by independent workers via claim_and_run.
---

# Write a Sweep

Pattern for a config sweep (one trial per grid point) in marin. Start from the tutorial
`experiments/tutorials/train_tiny_sweep_tpu.py`; the underlying primitive is
`marin.execution.sweep.claim_and_run` (`docs/tutorials/executor-sweeps.md`).

## Layout

Build every trial at submission time, then submit `NUM_WORKERS` independent workers. Each
worker races peers on a per-target lock (`claim_and_run`) and trains the claimed trial
**inline**; a target a peer already finished is a no-op.

```python
trials  = [build_trial(c) for c in grid]                    # configs carry placeholders
targets = [SweepTarget(target_id=t.name, config=t) for t in trials]

def run_one(target):                                        # resolve + train inline
    run_training_on_worker(name=target.config.name, raw_config=target.config.raw, ...)

def worker(sweep_root):                                     # one worker loops the targets
    claim_and_run(sweep_root, targets, run_one)

handles = [client.submit(JobRequest(                        # N-way parallelism
    entrypoint=from_callable(worker, args=[SWEEP_ROOT]),
    resources=ResourceConfig.with_tpu("v4-8"))) for _ in range(NUM_WORKERS)]
for h in handles: h.wait(raise_on_failure=True)
```

- `SWEEP_ROOT` is a **region-pinned** lock path; workers across regions / resubmissions
  contend on the same namespace. Bump `SWEEP_NAME` to fork a fresh sweep.
- Build trials at submit time so workers do no config work; configs carry placeholders
  resolved in the *worker's* region (survives cross-region preemption).
- Inline-on-accelerator vs a CPU driver is a tradeoff, not a rule — see
  `docs/iris-execution-model.md`.

## Subcommands (env-var dispatch)

One `python -m …` entrypoint, behavior selected by env var:

```python
def preview(): return os.environ.get("PREVIEW", "").lower() in {"yes", "true", "1"}
# PREVIEW=yes -> print the resolved target list and submit nothing
```

## Subset the grid (RUNS)

Optional `RUNS` CSV substring filter over target ids; empty = all:

```python
def selected(targets):
    needles = [s.strip() for s in os.environ.get("RUNS", "").split(",") if s.strip()]
    return targets if not needles else [t for t in targets if any(n in t.target_id for n in needles)]
```

## Conventions

- Reuse prebuilt tokenized caches; re-tokenize only when data changes.
- Bump `SWEEP_NAME` to fork run names + the lock root on a recipe change.
- Keep launchers minimal; add fan-out / monitors only when asked (see `monitor-sweep`).
