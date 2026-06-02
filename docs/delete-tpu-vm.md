# Delete a TPU VM

Deleting a VM is sometimes necessary to clear a misconfigured / stuck worker. Find the
worker name (e.g. from `tpu-info`'s PID triage or the autoscaler) and delete it in its
zone:

```bash
gcloud compute tpus tpu-vm delete <worker-name> --zone=<zone> --quiet
# e.g.
gcloud compute tpus tpu-vm delete marin-tpu-v6e-preemptible-4-europe-west4-20260423-1141-ea0e0871 \
  --zone=europe-west4-a --quiet
```

**Never** stop/restart an Iris cluster or kill jobs you don't own without explicit
permission — this is for clearing a single broken worker VM only.
