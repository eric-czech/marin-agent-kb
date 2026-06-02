> **Historical reference — not maintained.** An opinionated guide that was most
> relevant at a particular point in time. Unlike the `docs/` and `skills/` in this repo,
> it is *not* kept up to date as Marin changes (some of it may now be wrong); it is
> preserved for historical context only.

Opinionated Marin user guide most relevant in Q2 2026.

## Setup

- Clone the repo, avoid forks / multiple remotes (YMMV but Claude does not always handle that well)
  - `git clone --single-branch git@github.com:marin-community/marin.git`
- Don't use or try to setup custom GCP buckets as suggested in docs like [this](https://marin.readthedocs.io/en/latest/tutorials/storage-bucket/#step-2-create-the-bucket)
  - The Marin Executor will fail on bucket names not matching a particular pattern (see [here](https://github.com/marin-community/marin/blob/246077026f3405622dff654d2e0abc02788f78e0/lib/marin/src/marin/execution/executor.py#L168-L171))
  - Assume all data will be stored in standard marin buckets like `gs://marin-<region>`
    - `region` ∈ `{us-central1, us-central2, us-east1, us-east5, us-west4, eu-west4}`
      - There are a few more, but those are the primary region buckets worth knowing about
    - Show all buckets: `gsutil ls -b 'gs://marin-*'`
- Sign up for a Hugging Face PRO account ($9 / month): https://huggingface.co/pro
  - It's not worth fighting more rate limits on tokenization otherwise
- Authenticate with the `hai-gcp-models` project
  - Ask to be added to the `marin-dev` role on GCP on Discord or Slack ([example](https://openathena.slack.com/archives/C09AUMZ3QUA/p1775489847472469?thread_ts=1775489212.798239&cid=C09AUMZ3QUA))
    - Verify by viewing it [here](https://console.cloud.google.com/iam-admin/iam?project=hai-gcp-models)
  - Run `gcloud auth login --update-adc --quiet`
    - Assume you will need to re-run 1-2x daily -- haven't figured out how to avoid that
- Initialize the project:

```bash
# Clone the repo
git clone --single-branch git@github.com:marin-community/marin.git
cd marin

# Install env;
# see: https://github.com/marin-community/marin/blob/main/docs/tutorials/installation.md
uv sync --all-packages

# Set project env
cat << EOF > .env
export PROJECT_ID=hai-gcp-models
export WANDB_PROJECT=marin-dna
export WANDB_ENTITY=eric-czech
export WANDB_API_KEY=xxxxx
export HUGGING_FACE_HUB_TOKEN=xxxxx
export SSL_CERT_FILE=".venv/lib/python3.11/site-packages/certifi/cacert.pem"
EOF

# Setup precommits+node, and uv/gcloud if not already installed;
# see: https://github.com/marin-community/marin/blob/7677822a2897ec00b40f0cda95bd9e8d88f8914c/Makefile#L205-L206
make dev_setup
```

## Job Management

### Scheduling

- All job scheduling now goes through Iris (no more Ray)
- Basic skill for running and managing iris jobs: [run-iris-job.md](https://gist.github.com/eric-czech/35cd4a526f3404c2c368d78cbec8fb2c#file-run-iris-job-md)
- Notably, this skill does not include explicit constraints on scheduling like `--zone` and `--reserve` for `iris job run`
  - In practice, these often become necessary to control where jobs get run [[1](https://discord.com/channels/1354881461060243556/1364827114670657616/1493941317275746356)]
  - This is especially true if they have already made significant progress within one region [[2](https://discord.com/channels/1354881461060243556/1364827114670657616/1495783846690427015)]
  - Think of it like iris provides a way to **start** jobs flexibly across regions, but not resume or move those jobs to follow TPU availability
  - Currently, you are often better off picking a region/zone that supports the TPU slice you want and then scheduling there explicitly
- See https://iris.oa.dev/#/autoscaler for current availability of TPUs across regions
- Here is one way to manually schedule a job with control over where it is run via `--zone`:

```
DATE=$(date +%Y%m%d-%H%M%S) && uv run iris --config lib/iris/examples/marin.yaml job run \
--no-wait \
--job-name "bolinas-dna-4b-sweep-${DATE}" \
--zone us-east5-a \
-e WANDB_API_KEY $WANDB_API_KEY \
-e HUGGING_FACE_HUB_TOKEN $HUGGING_FACE_HUB_TOKEN \
-e WARMUP_MODE yes \
-- python experiments/dna/exp_bolinas_4b_sweep.py
```

This is the most direct way to make ensure that a job is run where the TPU requested actually exists. See "Clusters" below for correspondence between TPUs and zones.

### Executor

- This now runs with training jobs instead of spawning them [#5279](https://github.com/marin-community/marin/pull/5279)
- If you try to run a sweep with `executor_main` directly on a TPU VM via `--preemptible` and it has multiple training `steps`, it will fail on `RuntimeError: distributed.initialize should only be called once` when training multiple times in the same Python process (you can't do this anymore)

### Monitoring

- **Production Cluster**: https://iris.oa.dev/
  - Defined at [lib/iris/examples/marin.yaml](https://github.com/marin-community/marin/blob/7677822a2897ec00b40f0cda95bd9e8d88f8914c/lib/iris/examples/marin.yaml))
- **Dev Cluster**: https://iris-dev.oa.dev/
  - Defined at ([lib/iris/examples/marin-dev.yaml](https://github.com/marin-community/marin/blob/main/lib/iris/examples/marin-dev.yaml))
  - The iris controller for this cluster is restarted/updated more frequently but the cluster itself has much stricter limits on TPU availability
  - This is typically only worth using if the protobuf schema defining the iris client/server interface becomes out of sync with your branch [[3](https://discord.com/channels/1354881461060243556/1490717706167648409/1490741407214993618)]

To run the dashboard locally if the public endpoints are not reachable: 

```bash
# Dashboard is global for all regions and runs in us-central1-a at TOW
uv run iris --config lib/iris/examples/marin.yaml cluster dashboard
```

## Clusters

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

  **HBM sources**: v4 ([1](https://cloud.google.com/tpu/docs/v4)): 32 GiB/chip, v5e ([2](https://cloud.google.com/tpu/docs/v5e)): 16 GB/chip, v5p ([3](https://cloud.google.com/tpu/docs/v5p)): 95 GiB/chip, v6e
  ([4](https://cloud.google.com/tpu/docs/v6e)): 32 GB/chip.
  
## Ops

### Delete VM 

Deleting a VM can be necessary to avoid misconfigured workers, see [here](https://discord.com/channels/1354881461060243556/1364827114670657616/1495031090623025443).  Example:

```
gcloud compute tpus tpu-vm delete marin-tpu-v6e-preemptible-4-europe-west4-20260423-1141-ea0e0871 --zone=europe-west4-a --quiet 2>&1
```

### Auth as SA

In order to use Claude w/o permissions, it makes far more sense to first find a CPU VM and configure it for scoped GitHub and GCP access.  This [thread](https://discord.com/channels/1354881461060243556/1364827114670657616/1497360523979657346) contains some more details.  On using an SA, here is how I set it up on my VM (from exe.dev):

```
# Local (auth as self):
gcloud iam service-accounts create eczech-agent \
    --project=hai-gcp-models \
    --display-name="eczech-agent" \
    --description="Personal agent SA for eczech with Marin Agent role"
gcloud projects add-iam-policy-binding hai-gcp-models \
    --member="serviceAccount:eczech-agent@hai-gcp-models.iam.gserviceaccount.com" \
    --role="projects/hai-gcp-models/roles/marin_agent" \
    --condition=None

# Remote (auth as SA on VM):
# ssh oa-dev-1.exe.xyz
gcloud iam service-accounts keys create eczech-agent.json \
   --iam-account=eczech-agent@hai-gcp-models.iam.gserviceaccount.com
gcloud auth activate-service-account --key-file=eczech-agent.json
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/eczech-agent.json
```

### Run job on TPU VM

```bash
MIX=exp135-zoonomia-m1
DATE=$(date +%Y%m%d-%H%M%S)
source ~/repos/marin/.env && uv run iris --cluster=marin job run \
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

### Continued pretraining

- Docs:
  -  [Getting-Started-Training/#resume-training-runs](https://levanter.readthedocs.io/en/latest/Getting-Started-Training/#resume-training-runs)
  -  [Hardware-Agnostic-Training/#resuming-a-training-run](https://levanter.readthedocs.io/en/latest/Hardware-Agnostic-Training/#resuming-a-training-run)
- Levanter options for checkpoints:
  - `initialize_from`: Loads full state of previous run and continues at the same step
  - `initialize_from_checkpoint_path`: Loads optimizer states (I think) but resets data loader, LR schedule and step (to 0)
  - `initialize_from_hf`: Loads nothing but weights
- Examples:
  - Both of these from tootsie show how to set new `num_train_steps` w/ `allow_partial_checkpoint` and `reset_data_loader_on_init` while using prior levanter checkpoints via `initialize_from_checkpoint_path`:
    - [exp600_tootsie.py#L107](https://github.com/marin-community/marin/blob/6e37517d2e907a0a4a415cebc1d8c46516ac07ad/experiments/tootsie/exp600_tootsie.py#L107)
    - [exp600_tootsie.py#L166](https://github.com/marin-community/marin/blob/6e37517d2e907a0a4a415cebc1d8c46516ac07ad/experiments/tootsie/exp600_tootsie.py#L166)
- LR schedule problems with `initialize_from_checkpoint_path`
  - I expected these to reset in these bolinas runs [exp135-zoonomia-m4](https://wandb.ai/eric-czech/marin/runs/dna-bolinas-mix-v0.9-p1B-i23-exp135-zoonomia-m4-ee223a) and [exp135-zoonomia-m5-b1fe59](https://wandb.ai/eric-czech/marin/runs/dna-bolinas-mix-v0.9-p1B-i24-exp135-zoonomia-m5-b1fe59)
  - Instead one started out at 0 and is stuck there, the other is weirdly just decreasing from a small starting value
  - What you need to do is override these properties on the optimizer config set by `SimpleTrainConfig` (replace on the dataclass): `cycle_length`, `rewarmup` and `decay`
  - These are each set as step counds where `cycle_length` is passed as a list like `[last_run_steps, new_run_steps]`
  - `rewarmup` steps tells the LR schedule how many steps to do warmup separately from the first warmup
  - `decay` steps is the same as in the first cycle
  - See [_set_continuation_schedule](https://github.com/marin-community/marin/blob/14a6766e46e2c66788157445ceee13fadd443388/experiments/dna/exp135_bolinas_mix_sweep.py#L819) in `exp135_bolinas_mix_sweep.py`
  - This commit [9c719e7](https://github.com/marin-community/marin/commit/9c719e722186a90247f57aa9eb7f2c1f6d1c76bc) shows how parent and new run steps must be added (80% of parent steps if starting prior to cooldown w/ 20% decay), how to set new LR schedules, and how to support continuation prior to cooldown or after (cooldown alone would not be much different)

### Grad/param tracking

- These include a lot of data in wandb runs and may require non-trivial compute
- To disable, add `WatchConfig(watch_targets=[], interval=0)` to a train config