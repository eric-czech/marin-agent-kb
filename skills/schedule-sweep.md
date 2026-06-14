---
name: schedule-sweep
description: Run a sweep to completion fastest — place each training run on the highest wall-clock-throughput TPU slice/band, with budget, recovery, and relocation. Supersedes monitor-sweep.
---

# Schedule Sweep

Place/re-place a sweep's runs to finish the whole sweep fastest, within its definition of
done. **Supersedes `monitor-sweep` — this skill is a superset of it.** It owns the full run
lifecycle: submit/place, keep alive by resubmitting terminal failures (incl. parent jobs),
relocate, plus throughput and budget. Use this **instead of** `monitor-sweep`; never run both.

Composable references (their knowledge is assumed, not repeated — the default for every
related skill/doc):

- `monitor-sweep` — the failure-recovery loop this skill absorbs; consult for resubmit shape.
- `run-iris-job` — submit/monitor/stop one job.
- `size-tpu-train-config` — makes a config slice-agnostic (any slice launchable).
- `docs/iris-priority-and-quota.md` — priority bands + per-user budget.

## State files

- Per-config data (config → job id, slice, band, …) → `/tmp/<sweep>_monitor_state.json`,
  this skill's single source of truth (same format `monitor-sweep` uses).
- Throughput → `/tmp/<sweep>_throughput.md`.
- Default temp; on request, put either in the sweep repo.

## Scope

- **Single region only.** Pin every run to one `--region`; never span regions.
- **Per-config management required.** Track each config as its own run (job + W&B run);
  reason per config, never an aggregate count.
- Configs are slice-agnostic (`size-tpu-train-config`) → slice is a free knob.

## Done (infer from context)

- **Fixed set** (common): immutable config list. Done = every config has a completed run;
  you do placement + recovery only.
- **Open/agent-defined**: you pick configs as results arrive (search). Done = objective met;
  you also choose what to launch.
- Placement/throughput/budget handling are identical; only the config source differs.

## Placement: decide on `wts`

Pick each new/relocated run's slice by measured **`wts`** — the only metric folding in slice
speed *and* real availability/preemption. Fastest ≠ biggest ≠ highest compute throughput.
Re-measure periodically.

| metric | meaning | W&B summary |
|---|---|---|
| **`wts`** (decide) | wall-clock tok/s; real rate incl. preemption + availability | `throughput/total_tokens` ÷ (`_timestamp` − `created_at`) |
| `ats` | active tok/s; excludes dead/preempted time | `throughput/total_tokens` ÷ `_runtime` |
| `tts` | compute tok/s; instantaneous ceiling "if always scheduled" | `throughput/tokens_per_second` |
| `mfu` | model FLOPs util (diagnostic) | `throughput/mfu` |

`tts ≥ ats ≥ wts` always; `wts ≪ tts` = fast-but-poorly-scheduled → deprioritize.

Source: W&B `runs(ENTITY_PROJECT, filters={"group": SWEEP_GROUP})`, slice from each run's
`tpu=<slice>` tag. Per run, each value is one summary scalar (no history scan):

```
tok  = summary["throughput/total_tokens"]      # cumulative tokens
rt   = summary["_runtime"]                      # active seconds (excludes dead/preempted time)
wall = summary["_timestamp"] - to_unix(run.created_at)   # elapsed clock, incl. preemption
tts  = summary["throughput/tokens_per_second"]  # instantaneous rate
mfu  = summary["throughput/mfu"]                # instantaneous %

# Per slice — SUM cumulative scalars over its runs, AVERAGE rates:
wts  = Σtok / Σwall      # DECISION METRIC — rank slices desc
ats  = Σtok / Σrt
tts  = mean(tts)         # rate: average, never sum
dead = (Σwall - Σrt) / Σwall
```

Record the per-slice table in the throughput file.

Two caveats:

- If a relocation split a config into several W&B run objects, dedup per config before
  summing — cumulative `total_tokens` double-counts otherwise.
- `wts` clocks from training start, so it omits pre-start queue time — rely on the
  availability check for scarce slices.

## Bands & budget

- Fill `interactive` (on-budget) to ~the cap with highest-`wts` work, then add `batch`
  (off-budget) for free capacity. Set band explicitly per run — it is not implied by slice.
  Mild brief overage is fine (Iris downgrades overflow to `batch`); act only if well over for
  a sustained stretch.
- Gate on **total** user spend (it includes the user's unrelated jobs).

## Availability: read CHILD state, not parent

Each run = parent driver + child worker. The parent stays alive and handles rescheduling
across preemptions/capacity gaps → its state ≠ availability. Read the child:

- child `pending`/`unschedulable` → no slice room now.
- child `running` → training.

Iris is not faultless: occasionally a parent itself fails (no self-recovery) → resubmit it
(the recovery step in Loop).

## Relocate stuck runs

Child unschedulable ≳ 1 h (ignore short waits; preemptible scheduling is slow to turn around)
→ resubmit that config on the highest-`wts` slice currently scheduling; spread demand so runs
place. Never relocate a progressing run. Region fixed. Relocation = stop-then-resubmit → update
state with the new job id so the recovery step tracks the live run, not the killed one.

## Loop

**Exactly one recurring task** — a single `/loop` or cron firing this skill. Never a second
loop; scheduling, throughput tracking, and failure recovery all happen in this one tick. Each
tick, over the state file:

- recover failures: resubmit terminal-failed runs (incl. failed parent jobs); update state.
- read spend; check child states/availability; record finished results.
- refresh `wts` in the throughput file if it moved.
- place new / relocate stuck runs per above.
