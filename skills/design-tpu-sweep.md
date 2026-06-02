---
name: design-tpu-sweep
description: Structure a TPU training/eval sweep in marin — CPU driver, Fray TPU worker, idempotent claim_and_run.
---

# Design a TPU Sweep

Pattern for a config sweep (one trial per config) on marin TPUs. Reference:
`experiments/protein/exp29_arch_sweep.py`. Start from the closest existing
`experiments/.../expNN_*.py` and ask for one rather than inventing structure.

## Layout

A bash launcher submits **one CPU iris job per config**. Each CPU job is a driver
that submits **one Fray `with_tpu` worker**; the worker trains under `claim_and_run`
for idempotency (a target a peer already finished is a no-op).

```bash
# one CPU iris job per config (see run-iris-job)
for cfg in llama qwen3; do
  iris --cluster marin job run --user $USERNAME --no-wait \
    --region us-east5 -e RUNS $cfg -- python -m experiments.<pkg>.expNN_sweep
done
```

```python
def main():                                   # CPU driver
    variants = selected()                     # RUNS filter; raise if empty
    if preview(): print_preview(variants); return
    client.submit(JobRequest(
        entrypoint=Entrypoint.from_callable(worker_entrypoint, args=[ids(variants)]),
        resources=ResourceConfig.with_tpu(tpu(), zone=ZONE),
    )).wait(raise_on_failure=True)

def worker_entrypoint(ids):                   # runs ON the TPU
    targets = [SweepTarget(trial_name(v), config=v.id) for v in selected()]
    claim_and_run(SWEEP_ROOT, targets, run_one)    # run_one builds trial + trains
```

**Why a worker, not in-process:** training directly under `iris --tpu` caps host
RAM and OOM-kills the XLA compile.

## Subcommands (env-var dispatch)

One `python -m …` entrypoint, behavior selected by env var:

```python
def preview(): return os.environ.get("PREVIEW", "").lower() in {"yes", "true", "1"}
def tpu():     return os.environ.get("TPU") or DEFAULT_TPU    # override slice
# PREVIEW=yes -> list targets, submit nothing;  default -> submit
```

## RUNS selector

`RUNS` is a CSV **substring** filter over config ids; empty = all:

```python
def selected():
    needles = [s.strip() for s in os.environ.get("RUNS", "").split(",") if s.strip()]
    return ALL if not needles else [v for v in ALL if any(n in v.id for n in needles)]
```

## Conventions

- Reuse prebuilt tokenized caches; re-tokenize only when data changes.
- Bump a `VERSION` constant to fork run names + the `SWEEP_ROOT` claim dir on a recipe change.
- Keep launchers minimal; add fan-out / monitors only when asked (see `monitor-sweep-resubmit`).
