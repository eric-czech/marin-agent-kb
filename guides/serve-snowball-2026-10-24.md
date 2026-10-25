# Snowball serving launch, 2026-10-24

This recipe served `open-athena/Grug-67B-A2B-Datakit-SFT-262K-2026.09.21`
on eight H100s through Iris, with a chat UI and OpenAI-compatible API. Both the
CPU parent and GPU child were verified as batch priority. The UI, health endpoint,
model discovery, and a short completion all worked.

**These commands may have changed. Consult current CLI help and Marin docs
first.** The successful run used Marin commit
[`2cae063c8771`](https://github.com/marin-community/marin/tree/2cae063c877172844fe6ff90e94354cc08a273fa).
The commands below express its options as a shell wrapper; the original driver
invoked the same serving CLI from Python. Model paths, resource sizes, and
parallelism settings belong to this example. Follow
[serve-marin-models](../skills/serve-marin-models.md) for the general workflow.

## Prepare the model and credentials

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

## Build the driver

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

## Submit the batch parent

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
