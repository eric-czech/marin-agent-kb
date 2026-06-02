---
name: monitor-sweep
description: Keep a sweep alive by auto-resubmitting runs that failed or were killed.
---

# Monitor Sweep

A recurring loop that watches a set of sweep runs and resubmits any that failed.
**Read + submit only** — never stop/kill, never any destructive op.

## State file

`/tmp/<sweep>_monitor_state.json` — single source of truth, mapping logical run
name → current job id. The operator edits it in-conversation to add new runs or
drop completed ones.

```json
{"exp135-m1": "/eczech/exp135-m1-20260601", "exp135-m2": "/eczech/exp135-m2-20260601"}
```

## Per-tick logic

```python
state = read_json(state_file)
for name, job_id in state.items():
    s = job_summary(job_id, timeout=180)             # iris --cluster marin job summary <id>
    if s in {"failed", "killed", "worker_failed"}:
        new_id = resubmit(name, ...standard flags...) # see run-iris-job
        state[name] = new_id
        write_json(state_file, state)
    report_line(name, s)
```

## Invariants

- **Query error → skip.** Timeout / non-parseable output ≠ failure; wait for next tick.
- **In-flight is a no-op:** `pending` / `running` / `submitted` / `queued`.
- Resubmit silently; don't flag the operator on every recovery.

## Resubmit shape

A TPU-direct `iris ... job run` (see `run-iris-job`) with a pinned region, fixed
resource extras, and a date-stamped `--job-name` so each submission is unique.
Capture the new job id from the last stdout line and write it back to state.

## Scheduling

Drive with `/loop 15m` or a `*/15 * * * *` cron that re-fires this skill. Cron
auto-expires after 7 days; recreate to extend.
