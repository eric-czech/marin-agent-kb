# Job Execution Model (Fray + Iris)

How a marin job reaches hardware, and why sweeps use a CPU driver *by default* (not
by necessity).

## Fray vs Iris

- **Iris** — the cluster job orchestrator/scheduler: owns worker VMs, assigns tasks,
  and runs the task state machine (queue → run → retry → done).
- **Fray** — the Python execution-client API (`JobRequest`, `ResourceConfig`,
  `current_client().submit(...)`); its Iris backend converts a `JobRequest` into an
  Iris task.

So: Python builds a Fray `JobRequest` → Fray submits it to Iris → Iris schedules it
on a worker. `iris job run -- python …` is the CLI frontend onto the same controller.

## Allocation

- `ResourceConfig.with_tpu("v5p-8", zone=…)` — a TPU slice. The **TPU VM is the atomic
  scheduling unit** (`v5p-8` = 1 VM / 4 chips; bigger slices = more VMs). Defaults:
  cpu=32, ram=128g.
- `ResourceConfig(cpu=…, ram=…)` — a CPU-only worker.

## Preemption & retries

TPU VMs are **preemptible by default**. The **Iris controller** auto-retries a preempted
task — effectively unlimited in practice (`max_retries_preemption` = 100 via Fray, 1000
via the CLI), while failures (`max_retries_failure` = 0) are not. A retry re-runs the entrypoint from scratch,
so resuming *progress* (vs restarting at step 0) is the workload's job, via checkpoints.
Neither layer comes from a CPU driver — **the controller is what makes a job
preemption-resilient, not the driver.**

## CPU driver vs direct-on-TPU

Either works; Iris retries both. The CPU-driver → Fray-TPU-worker shape is the **sweep
default** only because the long-lived CPU process holds dispatch / setup / fan-out logic
that then isn't re-run on each TPU restart — run **directly on the TPU** when the CPU pool
is contended.

Logically the driver is a **dependency, not a safety net**: the TPU worker is its *child*,
so if the driver terminates Iris cascades and **cancels the children** (`KILLED` is
terminal — never retried). Small CPU jobs are auto-tagged non-preemptible to keep the
driver stable.
