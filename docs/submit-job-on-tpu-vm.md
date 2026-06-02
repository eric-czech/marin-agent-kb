# Submit a Job on a TPU VM

Worked example of a TPU-direct `iris job run` with explicit placement and resources —
the most reliable way to land a job where the requested slice actually exists (see
`docs/iris-scheduling-notes.md` and `docs/tpu-clusters.md`). For the general,
parameterized submit, see `skills/run-iris-job.md`.

```bash
MIX=exp135-zoonomia-m1
DATE=$(date +%Y%m%d-%H%M%S)
source ~/marin.env && uv run iris --cluster marin job run \
  --region us-east5 \
  --user eczech \
  --no-wait \
  --priority interactive \
  --tpu v5p-8 \
  --enable-extra-resources \
  --extra tpu \
  --extra lm_eval \
  --memory 128GB \
  --job-name "dna-bolinas-mix-v0.9-${MIX}-${DATE}" \
  -e WANDB_API_KEY "$WANDB_API_KEY" \
  -e SWEEP_MIX_NAMES "${MIX}" \
  -- python -m experiments.dna.exp135_bolinas_mix_sweep
```

Notes:
- `--zone <zone>` pins placement tighter than `--region`; `--reserve` requests reserved capacity.
- `--priority interactive` for fast, short-lived debugging jobs.
- `--extra` adds dependency groups (`tpu`, `lm_eval`) to the worker image.
