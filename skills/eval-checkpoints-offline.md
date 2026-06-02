---
name: eval-checkpoints-offline
description: Offline-eval Levanter checkpoints by fanning out one idempotent Iris job per checkpoint.
---

# Offline Checkpoint Eval

Evaluate trained Levanter checkpoints without an Executor or status files: a CPU
iris coordinator fans out one Fray TPU sub-job per checkpoint, and the done-signal
is the result JSON's presence at a deterministic path. Reference:
`experiments/dna/exp135_bolinas_epoch_sweep_eval.py`.

## Discovery

Use levanter's helper, not a hand-rolled `fs.ls` walk. Pick latest-only (eval the
final model) or all-steps (e.g. an epoch sweep):

```python
from levanter.checkpoint import discover_latest_checkpoint

def discover(train_dir, *, all_steps):
    root = f"{train_dir}/checkpoints"
    if not all_steps:                                    # newest valid step, or none
        ckpt = discover_latest_checkpoint(root)
        return [(int(ckpt.rsplit("step-", 1)[1]), ckpt)] if ckpt else []
    # every complete checkpoint: step-N/ dirs with a metadata.json (written last)
    return sorted(
        (int(name.split("-", 1)[1]), f"gs://{d}")
        for d in FS.ls(root, detail=False)
        if (name := d.rsplit("/", 1)[-1]).startswith("step-") and FS.exists(f"{d}/metadata.json")
    )
```

## Idempotent fan-out

```python
def submit_one(run, step, ckpt):
    out = f"{TMP_BUCKET}/{run}/step-{step}.json"
    if FS.exists(out): return None                  # already done -> skip
    return submit_tpu_job(eval_on_worker, [run, step, ckpt, out], EVAL_RESOURCES)

def main():
    handles = [h for r in runs for (step, ckpt) in discover(r, all_steps=ALL_STEPS)
               if (h := submit_one(r, step, ckpt))]
    for h in handles: h.wait(raise_on_failure=True)
    merge(TMP_BUCKET)                               # glob step-*.json -> merged.json

def eval_on_worker(run, step, ckpt, out):
    if FS.exists(out): return                       # double-check on the worker
    results = run_eval_harness_main(EvalHarnessMainConfig(
        eval_harness=LmEvalHarnessConfig(task_spec=..., include_path=...),
        checkpoint_path=ckpt, checkpoint_is_hf=False,
        trainer=TrainerConfig(tracker=NoopConfig(), mp=...), model=model_config()))
    if jax.process_index() == 0 and results:
        write_json(out, {...results...})            # FailSafeJSONEncoder (task class refs)
```

## Notes

- **Pin the region** to where the checkpoints were written so `marin_prefix()`
  resolves to the right bucket via the GCE metadata server.
- Stage results in a `marin_temp_bucket(ttl_days=…)`.
- XLA compile is reused across jobs via the shared `JAX_COMPILATION_CACHE_DIR` that
  `resolve_training_env` sets — one compile per `(arch, mesh)`, cached thereafter.
- Worker extras: `extras_for_resources(resources)` + `lm_eval`.
