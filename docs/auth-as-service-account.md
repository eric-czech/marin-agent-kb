# Auth as a Service Account

To run agents without permission prompts, configure a CPU VM for scoped GitHub + GCP
access via a personal service account holding the `marin_agent` role.

## Create the SA (run locally, as yourself)

```bash
gcloud iam service-accounts create eczech-agent \
  --project=hai-gcp-models --display-name="eczech-agent" \
  --description="Personal agent SA for eczech with Marin Agent role"
gcloud projects add-iam-policy-binding hai-gcp-models \
  --member="serviceAccount:eczech-agent@hai-gcp-models.iam.gserviceaccount.com" \
  --role="projects/hai-gcp-models/roles/marin_agent" --condition=None
```

## Activate on the VM (as the SA)

```bash
gcloud iam service-accounts keys create eczech-agent.json \
  --iam-account=eczech-agent@hai-gcp-models.iam.gserviceaccount.com
gcloud auth activate-service-account --key-file=eczech-agent.json
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/eczech-agent.json
```

GCP can't re-download key material, so keep the key safe — `skills/setup-dev-vm.md`
reuses an existing key rather than minting a new one.
