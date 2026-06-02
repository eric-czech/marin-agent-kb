# Continued Pretraining (Levanter)

Resuming / continuing from a prior Levanter checkpoint. Levanter docs:
*Getting-Started-Training#resume-training-runs* and
*Hardware-Agnostic-Training#resuming-a-training-run*.

## Checkpoint init options

- `initialize_from` — load full state, continue at the same step.
- `initialize_from_checkpoint_path` — load optimizer state but reset data loader, LR
  schedule, and step (to 0).
- `initialize_from_hf` — load weights only.

Use `allow_partial_checkpoint` + `reset_data_loader_on_init` with
`initialize_from_checkpoint_path` to set a new `num_train_steps` from prior weights.
See tootsie `experiments/tootsie/exp600_tootsie.py` (L107, L166).

## LR-schedule gotcha

With `initialize_from_checkpoint_path` the LR schedule does **not** reset cleanly (runs
got stuck at 0, or decayed from a small start). Fix: `dataclasses.replace` these on the
optimizer config set by `SimpleTrainConfig`, all as step counts:

- `cycle_length` — list `[last_run_steps, new_run_steps]`.
- `rewarmup` — warmup steps for the new cycle (separate from the first warmup).
- `decay` — same as the first cycle.

Reference: `_set_continuation_schedule` in `experiments/dna/exp135_bolinas_mix_sweep.py`;
commit `9c719e7` shows how parent + new steps combine (≈80% of parent steps if starting
before cooldown with 20% decay) and how to set the new schedule.
