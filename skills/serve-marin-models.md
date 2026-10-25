---
name: serve-marin-models
description: Serve a Marin model through Iris with a browser chat UI and OpenAI-compatible API, using batch priority for personal serving.
---

# Serve Marin Models

Use `marin-serve iris` to launch the serving backend and chat dashboard inside an
Iris job. Launch personal serving at **batch priority**. Admission can wait, and
higher-priority work can preempt the service.

## Resolve the current launch path

Start with `marin-serve iris --help`, the checkout's `lib/iris/OPS.md`, and its
federation guide. Resolve cluster names, resource sizes, and launch flags from
current code and configuration.

Use the current model recipe for backend and parallelism choices. Custom Marin
architectures can require the Marin vLLM fork; GPU count alone does not determine
tensor, data, or expert parallelism. For placement and whole-node GPU requests,
consult [CoreWeave](../docs/coreweave.md).

Iris authentication and model-storage credentials are separate. Verify the
submitting identity and a small model-storage read before allocating accelerators.
`~/marin.env` can contain stale storage keys despite a working Iris login. Follow
[Browsing CoreWeave storage](../docs/coreweave.md#browsing-coreweave-storage) for
key refresh and use the current Iris secret mechanism for delivery.

## Keep the entire job tree at batch priority

Set `--priority batch` explicitly on the root job. If the serving CLI supports
setting priority directly, use it. Otherwise run the launcher inside a small CPU
Iris job submitted with `--priority batch`; the serving child inherits that band.
Allow the driver enough memory for workspace setup and keep it alive through
readiness and access-URL creation.

For federation, place the **root driver** on the target peer. Only whole root job
trees federate. From that driver, submit the serving child to its local controller
with the task's credentials. Discover that controller through Iris task metadata.

Before the driver requests accelerators, check its stored priority and refuse to
continue unless it is batch. After submission, verify the serving child's
controller-recorded priority too. Leave capacity waits at batch priority.

## Worked launch: Snowball, 2026-10-24

This recipe served `open-athena/Grug-67B-A2B-Datakit-SFT-262K-2026.09.21`
on eight H100s through Iris, with a chat UI and OpenAI-compatible API. Both the
CPU parent and GPU child were verified as batch priority. The UI, health endpoint,
model discovery, and a short completion all worked.

**These commands may have changed. Consult current CLI help and Marin docs
first.** The successful run used Marin commit
[`2cae063c8771`](https://github.com/marin-community/marin/tree/2cae063c877172844fe6ff90e94354cc08a273fa).
The commands below express its options as a shell wrapper; the original driver
invoked the same serving CLI from Python. Model paths, resource sizes, and
parallelism settings belong to this example.

### Prepare the model and credentials

Run from a Marin checkout. Load the current `CW_KEY_ID` and `CW_KEY_SECRET` using
[the credential procedure](../docs/coreweave.md#browsing-coreweave-storage).
A working Iris login alone does not validate those storage credentials.

The model below was already staged in CoreWeave storage. Its path has a TTL;
check that it still exists before reusing it. Resolve an available model location
if it has expired.

```bash
: "${CW_KEY_ID:?Load current CoreWeave credentials first}"
: "${CW_KEY_SECRET:?Load current CoreWeave credentials first}"
export CW_KEY_ID CW_KEY_SECRET

USERNAME=eczech
JOB_NAME="snowball-$(date -u +%Y%m%d-%H%M%S)"
SNOWBALL_ENDPOINT="/serve/${USERNAME}-${JOB_NAME}"
SNOWBALL_MODEL='s3://marin-us-east-02a/tmp/ttl=14d/quick-serve-models/open-athena_Grug-67B-A2B-Datakit-SFT-262K-2026.09.21_b8c07f7df1df65525abbfdbcd1572318ba11c42f-4be62e99f28b1e32'
```

### Build the driver

The guard checks the CPU parent's stored priority and model access before the
serving launcher can request GPUs. The child uses the controller and task token
of the cluster running the parent. Its unspecified priority inherits batch.

```bash
serve_driver_file="$(mktemp)"
cat > "$serve_driver_file" <<'SH'
set -euo pipefail

python - <<'PY'
import os

from iris.cli.connect import rpc_client
from iris.cluster.client.job_info import get_job_info
from iris.rpc import controller_pb2, job_pb2
from rigging.filesystem.factory import url_to_fs

info = get_job_info()
assert info is not None and info.controller_address and info.task_token
with rpc_client(info.controller_address, info.credentials) as rpc:
    response = rpc.get_job_status(
        controller_pb2.Controller.GetJobStatusRequest(job_id=info.job_id.to_wire())
    )
    assert response.request.priority_band == job_pb2.PRIORITY_BAND_BATCH
fs, path = url_to_fs(os.environ['SNOWBALL_MODEL'] + '/config.json')
assert fs.info(path)['size'] > 0
print('Verified batch parent and model storage access', flush=True)
PY

export MARIN_CLUSTER_TOKEN="${IRIS_TASK_TOKEN:?Missing Iris task token}"
exec marin-serve iris "$SNOWBALL_MODEL" \
  --controller "${IRIS_CONTROLLER_ADDRESS:?Missing local controller}" \
  --gpu H100x8 \
  --name serve \
  --endpoint-name "$SNOWBALL_ENDPOINT" \
  --vllm-source marin-fork \
  --tensor-parallel-size 1 \
  --max-model-len 262144 \
  --memory 512g --disk 500g \
  --timeout-hours 24 \
  --wait-timeout 3600 \
  --proxy-timeout 1800 \
  --no-cache \
  --vllm-arg=--data-parallel-size --vllm-arg=8 \
  --vllm-arg=--enable-expert-parallel \
  --vllm-arg=--model-loader-extra-config \
  --vllm-arg='{"distributed":true}'
SH
```

Here TP=1, DP=8, and expert parallelism spread this MoE across one eight-GPU node.
The launcher loaded a staged S3 model; `--no-cache` disabled its separate model
cache. These choices need review for a different architecture or backend.

### Submit the batch parent

This small isolated local environment supplied the Iris CLI without installing
all of Marin on the laptop. The parent installs `marin-core` inside Iris. Only the
root names the target peer; its whole job tree stays on that peer.

```bash
uv run --no-project \
  --with ./lib/rigging --with ./lib/finelog \
  --with ./lib/finestore --with ./lib/iris \
  iris --cluster marin job run --no-wait \
  --user "$USERNAME" --job-name "$JOB_NAME" \
  --priority batch --target-cluster cw-rno2a \
  --cpu 1 --memory 6GB --disk 20GB --enable-extra-resources \
  --sync-package marin-core --max-retries 0 --timeout 90000 \
  --secret-behavior isolate \
  --secret CW_KEY_ID env:CW_KEY_ID \
  --secret CW_KEY_SECRET env:CW_KEY_SECRET \
  -e MARIN_PREFIX s3://marin-us-east-02a/marin \
  -e SNOWBALL_MODEL "$SNOWBALL_MODEL" \
  -e SNOWBALL_ENDPOINT "$SNOWBALL_ENDPOINT" \
  -- bash -c "$(cat "$serve_driver_file")"
rm "$serve_driver_file"
```

The root is `/$USERNAME/$JOB_NAME`; its serving child is
`/$USERNAME/$JOB_NAME/serve`. Check both recorded priorities after submission.
The driver task's logs contain readiness and the public capability URL.

For this version, the 24-hour serving timer starts at readiness. The parent's
90,000-second deadline bounds the whole tree to 25 hours from its task start;
queueing or startup can consume that budget. Batch preemption can interrupt
service earlier.

## Bound the lifetime

Set a finite serving lifetime and parent deadline, allowing time for admission
and model loading. Check when each clock starts: the serving lifetime can begin
at readiness while the parent's deadline starts earlier. Token expiry is another
limit. Report expected shutdown and hard cutoff in the user's timezone.

## Verify access and hand it over

A running Iris job can still be building its environment or loading weights. Wait
for endpoint registration, then verify the browser page, health, model discovery,
and one short completion through the **public** proxy.

Use the capability URL minted by the serving launcher. For a federated service it
must route through the public parent to the correct peer; an internal controller
address is unusable from a laptop. Read the driver's task logs to recover the URL
without mixing in the serving child's logs. The Iris job dashboard shows status;
the serving dashboard is the chat UI.

The OpenAI base URL ends in `/v1`; discover the served model ID from that endpoint.
With a capability URL, the URL carries authorization and an SDK may use any
nonempty API-key string. Treat that URL as a credential and keep it out of Git.

Return the chat UI link, API base URL and model ID, job dashboard and IDs, verified
priority, expiry times, and the command for stopping this job tree. Stop for a
missing credential or an unclear resource/runtime failure. Keep recovery within
the serving jobs authorized by the user.
