# CoreWeave

Operational context for running Marin drivers, GPU gangs, and data pipelines on
CoreWeave. This is a workload guide, not the Iris cluster-operator runbook.

Last audited 2026-08-05 against `~/repos/marin-br/main`, including the Iris cluster
configs, federation and Kubernetes backends, Fray submission path, Rigging storage
resolution, Levanter attention backends, and Zephyr resource defaults. Historical
storage measurements are dated separately.

## Critical rules

### Submission

- Run the driver inside the cluster that will run its gang. Submit the root job with
  `--target-cluster <cw-cluster>`; Iris federates root jobs only, so a driver cannot
  hand a child gang to another cluster.
- Put `--priority batch` on the driver only. An unspecified child priority resolves by
  walking the parent chain, so the gang inherits `batch`. Do not submit as
  `interactive` and rely on quota demotion later.
- Request whole GPU nodes and use power-of-two gang sizes. H100 replicas request all
  8 GPUs on a node; GB200 replicas request all 4. Sub-node GPU pods consume schedulable
  GPU/RDMA resources while leaving unusable fragments in the InfiniBand pool.

### Storage

- The three production clusters all use `s3://marin-us-east-02a/marin`. Treat it as
  one shared namespace: there is no regional data identity, cross-region restart
  procedure, or dataset/checkpoint seeding fan-out.
- Forward `MARIN_PREFIX=s3://marin-us-east-02a/marin` explicitly to the gang's job
  environment. A Kubernetes pod does not inherit the driver's process environment.
  Current Iris defaults and normal child submission both help propagate it, but they
  are not a workload contract to depend on.
- `MARIN_CLUSTER` is normally unset in task pods. Without an explicit `MARIN_PREFIX`,
  Rigging selects the default `marin` data config and can silently resolve relative
  paths onto GCS.

### Runtime

- Set model `attn_backend=AttentionBackend.JAX_FLASH` explicitly for GPU training.
  Do not leave the backend implicit: the GPU default is NVTE, and an unsupported
  configuration may fall back. The failure mode observed at sequence length 8192 was
  a quadratic attention path and an OOM.
- Kubernetes memory is a hard request and limit, not the softer VM/Docker behavior.
  Give drivers at least 3 GB. Zephyr's coordinator default is only 1 GB, so every
  CoreWeave `ZephyrContext` needs an explicit `coordinator_resources` override of at
  least 3 GB; allow more for a full workspace `uv sync`.

## Execution model

CoreWeave Iris clusters use the Kubernetes backend. Each task is a pod, Kueue admits
multi-replica jobs as a gang, and the pod requests GPU and RDMA devices directly.
CoreWeave's cluster autoscaler and reserved NodePools provide nodes; there are no Iris
worker daemons on these clusters.

Federation moves an entire root job once:

1. Submit the driver to the GCP `marin` controller with `--target-cluster`.
2. The Marin controller queues and hands that root to the selected CoreWeave peer.
3. The driver starts inside CoreWeave and submits its child gang to that same local
   controller.

This is why cluster placement belongs on the driver rather than on the gang request.
It also means that a target-cluster driver can use the peer's CPU pool while waiting
on or coordinating GPU work.

### Submission template

```bash
uv run iris --cluster marin job run \
  --target-cluster cw-us-east-02a \
  --priority batch \
  --user "$USERNAME" \
  --job-name <driver-name> \
  --cpu 1 --memory 3GB --disk 16GB --extra cpu \
  -e MARIN_PREFIX s3://marin-us-east-02a/marin \
  -- python -m <driver-module>
```

Inside the driver, leave the gang priority unspecified so it inherits `batch`, but
make its storage environment explicit:

```python
from fray.types import EnvironmentConfig, JobRequest, ResourceConfig

gang = JobRequest(
    name="train",
    entrypoint=...,
    resources=ResourceConfig.with_gpu(
        "H100",
        count=8,
        replicas=32,
        cpu=32,
        ram="256g",
        disk="256g",
    ),
    environment=EnvironmentConfig(
        workspace=".",
        env_vars={"MARIN_PREFIX": "s3://marin-us-east-02a/marin"},
    ),
)
```

The exact environment construction may differ by launcher. The invariant is that the
gang's `JobRequest` carries `MARIN_PREFIX`; setting it only with `os.environ` in the
driver is insufficient.

## Cluster map

The current `marin` federation config has three CoreWeave peers. A fourth checked-in
cluster is reserved for CI and is not a federation peer.

| Cluster | Fleet in checked-in config | CPU pool | Shared prefix | Use |
|---|---|---|---|---|
| `cw-us-east-02a` | 32 nodes × 8 H100 = 256 GPUs | 4 × 192-vCPU Genoa | `s3://marin-us-east-02a/marin` | Default H100 training/eval cluster. It has dense and sparse MoE, vLLM e2e, and MarinFold precedent. |
| `cw-rno2a` | 64 nodes × 8 H100 = 512 GPUs | 1 × 192-vCPU Turin | `s3://marin-us-east-02a/marin` | Data processing. Reference Datakit and Zephyr pipelines run here; avoid it for routine training when the fleet is busy. |
| `cw-us-east-08a` | 216 nodes × 4 GB200 = 864 GPUs (12 NVL72 racks) | 4 × 64-vCPU Emerald Rapids | `s3://marin-us-east-02a/marin` | Blackwell and very large jobs. Budget bring-up: the repository has an eval hardware mapping but no established dense-LM training precedent. |
| `cw-us-west-04a` | up to 2 nodes × 8 H100 | 1 × 64-vCPU Emerald Rapids | `s3://marin-us-west-04a/marin` | CI only. It runs the Iris gang smoke and daily canary ferry; never target it for experiments. |

Practical selection:

- H100 training: `cw-us-east-02a`.
- Data processing: `cw-rno2a`, while respecting its single-node CPU pool.
- GB200 or jobs too large for the H100 fleets: `cw-us-east-08a`, with an explicit
  bring-up phase.
- Experiments: never `cw-us-west-04a`.

Direct CoreWeave controller endpoints are IP-restricted. The supported user path is
the `marin` controller and federation; `iris rpc controller list-peers` is the useful
view of live peer capacity when cluster dashboards or direct controller URLs are not
reachable.

## Gang shape and resource behavior

The physical unit is a node, even though Kubernetes can express smaller GPU counts.

| Node | Correct per-replica request | Avoid |
|---|---:|---:|
| H100 | `H100x8` | `H100x1`, `H100x2`, `H100x4` |
| GB200 NVL72 | `GB200x4` | `GB200x1`, `GB200x2` |

Use power-of-two replica counts for ordinary training gangs: 1, 2, 4, 8, 16, 32,
64, and so on within cluster capacity. Kueue admits all replicas together and Iris
requests `rdma/ib` alongside each GPU. A partial-node request can therefore strand
both GPU and network capacity even if Kubernetes reports free individual GPUs.

GB200 additionally has rack/NVLink topology constraints. Let the current Iris CLI
validate the requested multi-node shape; do not bypass a topology validation error
with hand-built coscheduling annotations.

## Priority inheritance

Iris stores an unspecified priority as `UNSPECIFIED` and resolves it at scheduling
time by walking `parent_job_id` until it finds an explicit ancestor band. If every
ancestor is unspecified, it falls back to `interactive`.

For a batch workload:

- set the root driver to `batch`;
- leave all child requests unspecified;
- do not set individual gang or Zephyr jobs to `interactive`.

Batch work is excluded from user quota spend. Interactive work may be demoted when a
user is over budget, but that is an overload mechanism rather than a submission
strategy.

## Storage model

### One production namespace

`cw-us-east-02a`, `cw-rno2a`, and `cw-us-east-08a` all set both controller state and
task storage to the `marin-us-east-02a` CoreWeave bucket. A checkpoint written by one
cluster is immediately addressable by either of the others at the same URL. Moving a
run between those clusters changes compute placement, not its storage identity.

CoreWeave AI Object Storage currently has no ingress, egress, transfer, or request
fees. Cross-region reads are supported directly. The expensive boundary is generally
CoreWeave ↔ GCS, where the non-CoreWeave provider's transfer charges still apply.

### LOTA is a cache, not another data copy

Inside CoreWeave, S3 requests use `http://cwlota.com`. LOTA is an LRU object cache
distributed over the GPU and CPU nodes of one CKS cluster. Each production cluster
therefore has its own cache state even though all three read the same bucket:

- a cold cross-region read fetches from the bucket's home repository;
- later reads in that cluster are served from its local distributed cache;
- another cluster may still be cold;
- writes can cross regions but are not cached in the same way as reads.

This is a latency distinction, not a reason to clone datasets or rewrite paths by
region. Pre-staging is available for a carefully chosen working set, but it is an
optional performance optimization rather than a correctness step or mandatory
fan-out.

### Environment resolution

Rigging resolves `marin_prefix()` in this order:

1. `MARIN_PREFIX`;
2. the selected data config's explicit root;
3. a bucket inferred from region metadata;
4. a local fallback.

The selected data config comes from `MARIN_CLUSTER`, defaulting to `marin`. CoreWeave
pods normally have no `MARIN_CLUSTER`, so an omitted `MARIN_PREFIX` selects Marin's GCS
configuration. The CoreWeave Iris configs currently inject the correct S3 prefix into
every task and the Iris client merges a parent's explicit job environment into normal
children. Still put the value in the gang environment: it makes the dependency visible
and protects paths created through a launcher that does not use normal Iris child
submission.

### Buckets and credentials

| Bucket | Store | Intended use |
|---|---|---|
| `marin-us-east-02a` | CoreWeave, US-EAST-02A | Shared production data, checkpoints, MarinFold data, and controller state for all three production clusters. |
| `marin-us-west-04a` | CoreWeave, US-WEST-04A | CI state and scratch for `cw-us-west-04a`. |
| `marin-na` | Cloudflare R2 | Global zero-egress staging, finelog archives, and Datakit distribution. Access must be explicit. |
| `rhoarnet-us-east-08a` | CoreWeave, another tenant | Not ours. It may be visible to the access key; never write to it. |

Task pods carry one S3-compatible endpoint and credential set. The default CoreWeave
credentials do not also grant access to R2. A job that deliberately uses `marin-na`
must pass the appropriate `R2_*` credentials and endpoint configuration explicitly.

### Dated storage inventory

The 2026-07-25 recursive inventory measured at least 359 TB in
`marin-us-east-02a`, plus a `datakit/` prefix whose 100,000+ parts were not totaled:

| Prefix | Objects | Size |
|---|---:|---:|
| `marin/` | 1,194,074 | 228.7 TB |
| `iris/` | 3,754,994 | 118.9 TB |
| `tmp/` | 128,465 | 7.3 TB |
| `protein-structure/` | 6,786 | 2.7 TB |
| `MarinFold/` | 16,249 | 0.95 TB |
| `models/` | 49 | 0.16 TB |
| `user/` | 1,807 | 0.11 TB |
| `finelog/` + `scratch/` | 1,936 | 0.005 TB |

The `iris/` state prefix was roughly one-third of measured storage, and `tmp/` held
7.3 TB despite 1–7 day lifecycle tiers. At this scale, a new 10 TB experiment adds
about 3%, 20 TB about 6%, and 100 TB about 28%; the last is a storage decision, not a
routine default.

## GPU training environment

### Attention backend

Levanter's accelerator default is NVTE on GPU. Make the intended kernel explicit in
the model config:

```python
import dataclasses

from levanter.layers.attention import AttentionBackend

model = dataclasses.replace(
    model,
    attn_backend=AttentionBackend.JAX_FLASH,
)
```

Apply the setting at the model configuration site that feeds every layer; changing a
local attention call or only one stage is not sufficient. This is a memory-safety
requirement for long sequences, especially sequence length 8192, not merely a
performance preference.

### Hard Kubernetes memory limits

The Iris Kubernetes backend writes the requested memory as both the pod request and
the pod limit. Crossing it produces an OOM kill (often exit 137).

- Driver: request at least 3 GB. Use more when startup performs a full `uv sync` or
  holds large submission/config objects.
- Zephyr coordinator: always pass `coordinator_resources`; the library default is
  `cpu=0.1, ram="1g"`, which is too small for a CoreWeave workspace startup.
- Zephyr workers: size `resources` independently for the workload; do not infer their
  requirement from the coordinator.

Example:

```python
ctx = ZephyrContext(
    name="pipeline",
    max_workers=16,
    resources=ResourceConfig(cpu=2, ram="8g", disk="8g"),
    coordinator_resources=ResourceConfig(
        cpu=1,
        ram="6g",
        disk="16g",
        preemptible=False,
    ),
)
```

Three GB is the floor; 6 GB is the established safer value when the coordinator runs
the standard workspace setup.

## Browsing CoreWeave storage

CoreWeave Object Storage requires virtual-hosted addressing. Use the LOTA endpoint
inside a cluster and `https://cwobject.com` outside it.

```bash
set -a
source ~/marin.env
set +a

cfg=$(mktemp)
printf '[default]\ns3 =\n    addressing_style = virtual\n' > "$cfg"

AWS_CONFIG_FILE="$cfg" \
AWS_ACCESS_KEY_ID="$CW_KEY_ID" \
AWS_SECRET_ACCESS_KEY="$CW_KEY_SECRET" \
AWS_REGION=US-EAST-02A \
AWS_ENDPOINT_URL_S3=https://cwobject.com \
AWS_PAGER="" \
uv run --with awscli aws s3 ls s3://marin-us-east-02a/MarinFold/
```

In Python, `rigging.filesystem.s3_compat.configure_coreweave_s3()` configures the
same keys; plain `fsspec.filesystem("s3")` can then access the bucket.

## Sources

- Marin: `lib/iris/config/{marin,cw-*}.yaml`, `config/coreweave.yaml`,
  `lib/iris/docs/{coreweave,federation}.md`, the Iris Kubernetes and scheduling
  backends, `lib/iris/src/iris/client/client.py`,
  `lib/rigging/src/rigging/filesystem/cluster_config.py`,
  `lib/levanter/src/levanter/layers/attention.py`, and
  `lib/zephyr/src/zephyr/execution.py`.
- [Original CoreWeave cluster and bucket audit](https://gist.github.com/eric-czech/9184f1403acdc70257ae1fefa2ae76a4)
  (2026-07-25).
- [CoreWeave pricing](https://coreweave.com/pricing) for current storage transfer
  and request fees.
- [CoreWeave LOTA documentation](https://docs.coreweave.com/products/storage/object-storage/improving-performance/about-lota)
  and [performance guidance](https://docs.coreweave.com/products/storage/object-storage/improving-performance/best-practices)
  for cluster-local caching and cross-region reads.
