# Dev Environment Setup

Local setup for working in the marin repo (laptop / personal dev box). For a fresh
agent VM, see `skills/setup-dev-vm.md`.

## Clone & buckets

- Clone single-branch; avoid forks / multiple remotes (agents handle them poorly):
  `git clone --single-branch git@github.com:marin-community/marin.git`
- **Don't create custom GCP buckets.** The marin Executor rejects bucket names that
  don't match its pattern (`lib/marin/src/marin/execution/executor.py`). Use the
  standard regional buckets `gs://marin-<region>`, region ∈ {us-central1, us-central2,
  us-east1, us-east5, us-west4, eu-west4}. List: `gsutil ls -b 'gs://marin-*'`.

## Accounts & auth

- Get a Hugging Face **PRO** account ($9/mo) — not worth fighting tokenization rate
  limits otherwise.
- Authenticate to `hai-gcp-models`: ask (Discord/Slack) to be added to the `marin-dev`
  role, then `gcloud auth login --update-adc --quiet`. Expect to re-run 1–2× daily.

## Initialize

```bash
git clone --single-branch git@github.com:marin-community/marin.git && cd marin
uv sync --all-packages
make dev_setup          # precommits + node; installs uv/gcloud if missing
```

Config + secrets live in `~/marin.env`, sourced from `~/.bashrc` — see
`skills/setup-dev-vm.md` for the full block (`USERNAME`, `PROJECT_ID`, HF, WANDB).
Add `SSL_CERT_FILE` so requests find certifi's bundle:

```bash
export SSL_CERT_FILE="$PWD/.venv/lib/python3.11/site-packages/certifi/cacert.pem"
```

Lint entry point is `./infra/pre-commit.py` (never `uv run pre-commit`).
