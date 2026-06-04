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

## Scoped by design (no object deletes)

The `marin_agent` role deliberately omits `storage.objects.delete`, so the SA can read
and write GCS but **not delete** objects. Expect errors like this:

```
the active service account eczech-agent@hai-gcp-models.iam.gserviceaccount.com
has no storage.objects.delete permission on marin-us-east5
```

If a task genuinely needs a delete, do it as yourself (not the SA) rather than widening
the role.
