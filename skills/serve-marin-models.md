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

## Worked launch

The [Snowball launch from 2026-10-24](../guides/serve-snowball-2026-10-24.md)
records the working parent submission and full GPU serving command, including the
priority check. These commands may have changed: consult current CLI help and
Marin documentation first, then adapt the example to the model and cluster.

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
