> **Historical reference — not maintained.** An opinionated guide that was most
> relevant at a particular point in time. Unlike the `docs/` and `skills/` in this repo,
> it is *not* kept up to date as Marin changes (some of it may now be wrong); it is
> preserved for historical context only.

Setup instructions for Marin pipeline development 

## Overview

General flow of steps to follow:

1. Get your email added to the Marin GCP org (for the `hai-gcp-models` project, at least)
2. Create a bucket for your work if not relevant for the standard `marin-<region>` buckets
    - See existing buckets here https://console.cloud.google.com/storage/browser?project=hai-gcp-models
    - Consider `gs://marin-<team>-<region>` [[1](https://marin.readthedocs.io/en/latest/tutorials/storage-bucket/#step-1-choose-a-region-and-name)]
    - E.g. I used `marin-dna-us-central1`: https://console.cloud.google.com/storage/browser/marin-dna-us-central1
3. Configure dev env with region/cluster env vars
4. Run Ray dashboard
5. Run jobs
6. Analyze storage

See details for each step below.

## 1. Get added to GCP

Ask David.

## 2. Create bucket

See https://marin.readthedocs.io/en/latest/tutorials/storage-bucket/#step-2-create-the-bucket.

```bash
PROJECT_ID=hai-gcp-models
BUCKET=gs://marin-dna-us-central1
REGION=us-central1

gcloud storage buckets create "$BUCKET" \
  --project "$PROJECT_ID" \
  --location "$REGION" \
  --uniform-bucket-level-access \
  --default-storage-class=STANDARD --enable-autoclass

gcloud storage buckets update "$BUCKET" --clear-soft-delete
gcloud storage buckets describe "$BUCKET" --format="value(soft_delete_policy)"

cat << EOF > lifecycle.json
{
  "rule": [
    {
      "action": {"type": "Delete"},
      "condition": {"age": 7, "matchesPrefix": ["tmp/"]}
    }
  ]
}
EOF
gsutil lifecycle set lifecycle.json "$BUCKET"
```

## 3. Configure dev env

See https://marin.readthedocs.io/en/latest/tutorials/tpu-cluster-setup/#executing-a-job-on-the-cluster.

```bash
# Clone the repo
cd marin

# Install env
# See: https://github.com/marin-community/marin/blob/main/docs/tutorials/installation.md
uv sync --all-packages


# Set local env
cat << EOF > .env
export MARIN_PREFIX=$BUCKET
export WANDB_PROJECT=marin-dna
export WANDB_ENTITY=eric-czech
# DO NOT use tokens/keys you use elsewhere; create new ones just
# for Marin because they will not be hidden within cluster logs
export HUGGING_FACE_HUB_TOKEN=xxxxx
export WANDB_API_KEY=xxxxx
export RAY_AUTH_MODE=token
export RAY_AUTH_TOKEN_PATH="${RAY_AUTH_TOKEN_PATH:-$HOME/.ray/auth_token}"
EOF

# Note that this will run `config set project hai-gcp-models`
make dev_setup

# Fetch credentials for `hai-gcp-models` project if you haven't done so yet
gcloud auth login --update-adc --quiet
```

## 4. Run Ray dashboard

You need to do this in order to submit jobs as well as monitor them:

```
# Run and view dashboard at localhost:8265
# See: https://marin.readthedocs.io/en/latest/dev-guide/guidelines-internal/#connecting-to-the-ray-cluster-dashboard-and-submitting-jobs
# This will authenticate and open tunnels for remote dashboard
uv run scripts/ray/cluster.py --config infra/marin-$REGION.yaml auth

# I'm confused as to when you would use this instead per:
# https://marin.readthedocs.io/en/latest/dev-guide/guidelines-internal/#connecting-to-the-ray-cluster-dashboard-and-submitting-jobs
uv run scripts/ray/cluster.py --config infra/marin-$REGION.yaml dashboard
```

## 5. Submit jobs

Try https://marin.readthedocs.io/en/latest/tutorials/tpu-cluster-setup/#using-ray-submit:

```bash
# This fails immediately on the cluster with "`marin` not found", so I'm not sure why this example is in the docs
# without more details on how to make it work
export RAY_ADDRESS=http://localhost:8265
ray job submit --working-dir . -- python experiments/tutorials/hello_world.py
```

Nevermind, use `ray_run` instead (https://marin.readthedocs.io/en/latest/tutorials/tpu-cluster-setup/#using-ray-run):

```bash
# This will look for dashboard on port 8265 by default
uv run lib/marin/src/marin/run/ray_run.py \
  --env_vars WANDB_API_KEY ${WANDB_API_KEY} \
  --env_vars HUGGING_FACE_HUB_TOKEN ${HUGGING_FACE_HUB_TOKEN} \
  --  python experiments/tutorials/hello_world.py \
  --prefix $MARIN_PREFIX --force_run_failed true
  
# TODO: figure out why neither having WANDB_PROJECT set in local env
# for ray_run nor passing it explicitly result in using any project other than "marin"
# --env_vars WANDB_PROJECT ${WANDB_PROJECT} \
```

To stop a job:

```bash
# Get "Submission ID" (or "Job ID"?) from dashboard and pass it here:
ray job stop --address http://127.0.0.1:8265 ray-run-eczech-plantcad_isoflops_batch_size-20251124-171923
```

To monitor jobs logs if you get disconnected from the terminal you used to launch the job in the first place, you can reconnect with:

```bash
# Get "Submission ID" or "Job ID" from dashboard
ray job logs ray-run-eczech-plantcad_isoflops_batch_size-20251124-171923 --address http://127.0.0.1:8265 --follow
```

## 6. Analyze storage

For training runs, especially across big grids, checkpoint storage utilization can become fairly large.  To check and delete checkpoints (or anything else), here are some convenient commands:

```bash
# Use -s for total results rather than one per individual object
gsutil -m du -ch -s gs://marin-dna-us-central1/checkpoints

# Delete by glob
gsutil -m rm -r 'gs://marin-dna-us-central1/checkpoints/plantcad_isoflop_v1.3-*'
```

## 7. Clusters & TPUs

Cluster configs for iris live in `lib/iris/examples/(marin|marin-dev).yaml` and define these TPU slice types:

  | Cluster | Zone | Family | HBM/chip | TFLOPS/chip | Single-VM | All slice sizes |
  |---|---|---|---:|---:|---|---|
  | `us-central1` | us-central1-a | v5p | 95 GiB | 459 | v5p-8 | `v5p-8`, `v5p-16`, `v5p-32`, `v5p-64`, `v5p-128`, `v5p-256`, `v5p-512`, `v5p-1024`, `v5p-2048` |
  | `us-east5-a` | us-east5-a | v5p | 95 GiB | 459 | v5p-8 | `v5p-8`, `v5p-16`, `v5p-32`, `v5p-64`, `v5p-128`, `v5p-256`, `v5p-512`, `v5p-1024`, `v5p-2048` |
  | `us-central2` | us-central2-b | v4 | 32 GiB | 275 | v4-8 | `v4-8`, `v4-16`, `v4-32`, `v4-64`, `v4-128`, `v4-256`, `v4-512`, `v4-1024`, `v4-2048`, `v4-4096` |
  | `big-run` | us-central2-b | v4 | 32 GiB | 275 | v4-8 | `v4-8`, `v4-16`, `v4-32`, `v4-64`, `v4-128`, `v4-256`, `v4-512`, `v4-1024`, `v4-2048`, `v4-4096` |
  | `us-east5` | us-east5-b | v6e | 32 GB | 918 | v6e-4, v6e-8 | `v6e-4`, `v6e-8`, `v6e-16`, `v6e-32`, `v6e-64`, `v6e-128`, `v6e-256` |
  | `us-east1` | us-east1-d | v6e | 32 GB | 918 | v6e-4, v6e-8 | `v6e-4`, `v6e-8`, `v6e-16`, `v6e-32`, `v6e-64`, `v6e-128`, `v6e-256` |
  | `eu-west4-a` | europe-west4-a | v6e | 32 GB | 918 | v6e-4, v6e-8 | `v6e-4`, `v6e-8`, `v6e-16`, `v6e-32`, `v6e-64`, `v6e-128`, `v6e-256` |
  | `eu-west4` | europe-west4-b | v5e | 16 GB | 197 | v5litepod-4, v5litepod-8 | `v5litepod-4`, `v5litepod-8`, `v5litepod-16`, `v5litepod-32`, `v5litepod-64`, `v5litepod-128`, `v5litepod-256` |
  | `us-west4` | us-west4-a | v5e | 16 GB | 197 | v5litepod-4, v5litepod-8 | `v5litepod-4`, `v5litepod-8`, `v5litepod-16`, `v5litepod-32`, `v5litepod-64`, `v5litepod-128`, `v5litepod-256` |

  **HBM sources**: v4 ([GCP](https://cloud.google.com/tpu/docs/v4)): 32 GiB/chip, v5e ([GCP](https://cloud.google.com/tpu/docs/v5e)): 16 GB/chip, v5p ([GCP](https://cloud.google.com/tpu/docs/v5p)): 95 GiB/chip, v6e
  ([GCP](https://cloud.google.com/tpu/docs/v6e)): 32 GB/chip. Repo values (`marin/scaling_laws/tpu_utils.py`): v4=32 GiB, v5p=95 GiB — both match. v5e/v6e not in repo.

  **Topology source**: [`types.py#L634-L678`](https://github.com/marin-community/marin/blob/9148807ec2a53298309fa6a177fb0c9413912508/lib/iris/src/iris/cluster/types.py#L634-L678)

**TPU generations summary:**
- **v4**: Older gen, 32 GiB HBM per chip. Only in us-central2.
- **v5e**: Cost-optimized for inference and lighter training. Lower HBM/FLOPS than v5p.
- **v5p**: Training-focused, high HBM (95 GiB/chip) and FLOPS. Primary choice for pretraining.
- **v6e (Trillium)**: Newest gen, improved perf/watt over v5p. Good for both training and inference.

### Cluster Allocations

Here are some stats from a single analysis of recent allocations over all clusters:

| Zone | Slice | Count | Oldest | Newest |
|---|---|---|---|---|
| **europe-west4-a** | v6e-4 | 3 | 2026-03-10 | 2026-04-04 |
| | v6e-8 | 2 | 2026-04-01 | 2026-04-03 |
| | v6e-128 | 2 | 2026-04-04 | 2026-04-04 |
| **europe-west4-b** | v5litepod-4 | 5 | 2026-04-03 | 2026-04-04 |
| | v5litepod-16 | 1 | 2026-04-04 | 2026-04-04 |
| | v5litepod-128 | 2 | 2026-04-04 | 2026-04-04 |
| **us-central1-a** | v5p-8 | 31 | 2026-03-17 | 2026-04-04 |
| | v5p-16 | 1 | 2026-04-04 | 2026-04-04 |
| | v5p-32 | 4 | 2026-04-03 | 2026-04-04 |
| | v5p-64 | 1 | 2026-04-04 | 2026-04-04 |
| **us-central2-b** | v4-8 | 8 | 2026-04-03 | 2026-04-04 |
| | v4-32 | 54 | 2026-04-03 | 2026-04-04 |
| | v4-512 | 1 | 2026-04-03 | 2026-04-03 |
| **us-east1-d** | v6e-4 | 1 | 2026-04-03 | 2026-04-03 |
| | v6e-8 | 5 | 2026-03-10 | 2026-04-04 |
| | v6e-128 | 1 | 2026-04-03 | 2026-04-03 |
| **us-east5-a** | v5p-8 | 1 | 2026-04-04 | 2026-04-04 |
| | v5p-1024 | 1 | 2026-04-04 | 2026-04-04 |
| **us-east5-b** | v6e-4 | 10 | 2026-03-06 | 2026-04-04 |
| | v6e-8 | 1 | 2026-04-03 | 2026-04-03 |
| **us-west4-a** | v5litepod-4 | 6 | 2026-04-03 | 2026-04-03 |
| | v5litepod-8 | 3 | 2026-01-25 | 2026-02-28 |
| | v5litepod-16 | 2 | 2026-04-04 | 2026-04-04 |
| | v5litepod-128 | 1 | 2026-04-03 | 2026-04-03 |

<details><summary>Code</summary>
    
```
for zone in europe-west4-a europe-west4-b us-central1-a us-central2-b us-east1-d us-east5-a us-east5-b us-west4-a; do
  echo "| **$zone** |"
  gcloud compute tpus tpu-vm list --zone=$zone --project=hai-gcp-models \
    --format="table[no-heading](acceleratorType,createTime)" 2>/dev/null \
  | awk '{
      type=$1; time=$2
      if (!(type in min) || time < min[type]) min[type]=time
      if (!(type in max) || time > max[type]) max[type]=time
      count[type]++
    } END {
      for (t in count) {
        split(min[t], a, "T"); split(max[t], b, "T")
        printf "| | %s | %d | %s | %s |\n", t, count[t], a[1], b[1]
      }
    }' | sort -t'|' -k3,3
done
```
</details>

### Cluster Status

To query job and task status, see https://gist.github.com/eric-czech/bab558e275b478fb60a6cc97545d65fb.

## Troubleshooting

----

SSL errors with `gcsfs` reads:

```
aiohttp.client_exceptions.ClientConnectorCertificateError: Cannot connect to host storage.googleapis.com:443 ssl:True [SSLCertVerificationError: (1, '[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: unable to get local issuer
     certificate (_ssl.c:1006)')]
```

Fix by setting `SSL_CERT_FILE` in env to output of:

```bash
uv run python -c "import certifi; print(certifi.where())"
# ~/repos/crfm/marin/.venv/lib/python3.11/site-packages/certifi/cacert.pem
```
----
    
See more common errors, mostly with no known root cause or fix (other than retrying), at [Marin Errors](https://gist.github.com/eric-czech/3f65064cef914f05433e84c05df9d98f).