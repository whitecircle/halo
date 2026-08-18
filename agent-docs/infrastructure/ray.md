# Ray Cluster

Ray runs the rollout side of [Environmental GRPO](../training-methods/grpo/environmental-grpo.md).
Each training rank (one per training GPU) drives a pool of `EnvironmentActor` processes; an actor
plays one episode against the rollout servers and returns the trajectory with its reward. That is
Ray's entire job — weight sync and gradients go over NCCL, generation is HTTP to
[Rollout Servers](rollout-servers.md), and Ray carries only episode dispatch and results.

On a single node Ray configures itself. The implementation lives in
`src/environments/ray_actors.py` and the trainer's `_init_async_components`; the pin is
`ray[default] 2.52.x` (`pyproject.toml`). Terms: [glossary](../reference/glossary.md).

## Lifecycle

Every training rank calls `ray.init` (`ignore_reinit_error=True`) and builds its own actor pool at
the start of `train()`, not at trainer construction — there is no shared coordinator rank, and a
constructed-but-untrained trainer starts no Ray. With `ray_address: null` (the
default) each rank starts a local Ray instance, unless a `ray start` instance already runs on the
host, in which case `ray.init` joins it.

Two init settings differ from Ray's defaults:

- **The dashboard is off** (`include_dashboard=False`): Ray's dashboard/jobs API carries known
  RCE/DNS-rebinding CVEs (CVE-2023-48022, CVE-2025-62593), so a Ray instance the toolkit starts never
  opens port 8265. There is no YAML knob to re-enable it; monitor from the host shell instead (below).
  It governs only the instances `ray.init` starts: joining a head someone else launched inherits that
  head's setting, so a manual `ray start` must pass `--include-dashboard=false` itself ([below](#multi-node)).
- **Temp-dir fallback**: Ray's object-store socket is an AF_UNIX path with a ~107-byte cap, so
  when `TMPDIR` is longer than 40 characters Ray is redirected to `/tmp/ray`.

Teardown runs in `train()`'s `finally`: the prefetch thread stops, every actor's `shutdown()` is
awaited sequentially, weight-sync communicators close. `ray.shutdown()` is never called — actors
die by reference counting. A truly wedged actor can stall this teardown; kill the trainer
container if exit hangs.

## The actor pool

`EnvironmentActor` is declared `@ray.remote(max_restarts=-1)` and nothing more:

- **Unlimited restarts** — actors are stateless between episodes, so a restarted actor beats a
  permanent hole in the round-robin. Episodes routed to a dead actor become masked rows — zero
  reward, excluded from the loss — rather than a crash. A *wholly* dead pool still halts the run:
  a step in which no episode anywhere in the world survived warns, and the second consecutive one
  raises (`EMPTY_ROLLOUT_STEP_LIMIT = 2`) rather than training on all-masked batches.
- **Default resources** — Ray reserves one CPU to place an actor, then releases it: a running
  actor is charged nothing. So Ray's scheduler never caps what the pool actually consumes;
  what does depends on the environment ([below](#what-bounds-actor-concurrency)).
- **Asyncio actor** — episode methods are `async def`, so one actor serves up to Ray's default
  1000 concurrent calls. The pool size is *not* the concurrency limit; `max_concurrent_rollouts`
  is.
- **Soft node affinity** — each rank prefers its own node
  (`NodeAffinitySchedulingStrategy(soft=True, _spill_on_unavailable=True)`), so a shared cluster
  does not scatter one rank's pool cluster-wide; `soft` alone would pin an actor to a saturated live
  node and wait, so the spill flag is what keeps the preference a preference.
- **Lazy environment build** — the env is constructed inside the actor on its first episode. A
  broken environment constructor surfaces as per-episode errors, not a startup crash.

The actor's episode loop shares the per-turn step-context stamp (`step_context_from_generation`,
`src/environments/episode.py`) and the effort binding with the offline eval driver (`run_episode` in
`src/environments/eval_runner.py`), but keeps its own aiohttp session, backoff with retryable-4xx
classification, and engine token/routing capture.

Dispatch is round-robin over actors and over server URLs, one asyncio task per prompt, gated by a
semaphore of `max_concurrent_rollouts` (default 4× the actor count the rank actually built, clamped
to at least that count — so with `ray_address` set it derives from `num_rollout_workers //
world_size`, not from `num_rollout_workers`).

Each episode carries an `episode_timeout` deadline; on expiry the task is cancelled and the episode
returns as a masked row. The cancel is `ray.cancel(force=False)` — Ray forbids force-cancel on
async-actor tasks — so a hard-wedged task is not killable. Any actor failure, `RayActorError`
included, also becomes a masked row; the in-flight episode is lost, not retried.

## Pool sizing

`num_rollout_workers` (default 64) is **per training rank**:

- `ray_address: null` — cluster-wide total = `world_size × num_rollout_workers`.
- `ray_address` set — each rank creates `num_rollout_workers // world_size` actors (floored at 1),
  so the cluster-wide pool approximates `num_rollout_workers`. The floor means a world size larger
  than `num_rollout_workers` overshoots.

The split keys off the **config field**, not Ray's own `RAY_ADDRESS` environment variable: an
exported `RAY_ADDRESS` joins the cluster (Ray substitutes it for the `address=None` the toolkit
passes) while `ray_address: null` still has every rank build the full pool. Set the config field.

Three cluster-wide totals bound a large run: actors (above), in-flight episodes
(`world_size × max_concurrent_rollouts`), and per-node sandbox slots (next section).

**Under TP/ETP, most of that load is discarded.** Every rank collects a full batch, then
`_broadcast_rollouts_for_tp` replaces each rank's results with its TP-group leader's, so all ranks
tokenize identical trajectories. Actor and rollout-server load therefore scales with `world_size`
while useful rollouts scale with `world_size / tp_size`. Size the pool and the servers for the
former.

Actors are async, so size the pool to the env's **blocking** per-episode cost, not to HTTP
concurrency. Sandbox-backed envs (`code_contests`, `swe`) are CPU-bound and bursty; search/MCP
envs are latency-bound, where more actors help only until the remote endpoint rate-limits. Leave
the default unless `async/mean_rollout_latency` rises while the rollout servers sit idle. The
trainer-side batch knobs live under
[Environmental GRPO — Batch construction](../training-methods/grpo/environmental-grpo.md#batch-construction).

## What bounds actor concurrency

Not Ray — what binds depends on the environment class:

- **Sandbox-backed** (`code_contests`, `codeforces`, `swe`, any env with a `python`/code tool):
  a process-global gate of `HALO_SANDBOX_MAX_CONCURRENCY` slots (default: host core count) —
  **per process**. Each actor is its own process, so N actors on a node can run `N × cpu_count`
  concurrent sandbox subprocesses unless the var is set so the slots **sum** to the core budget.
  Per-run rlimits and scratch sizing:
  [Sandboxes](../training-methods/grpo/environments/sandbox.md#sizing-at-scale).
- **Network-bound** (`qa_search`, `react_search`, `mcp`, open-book `exam_qa`): no gate — size
  against the search/MCP endpoint's rate limit, not against cores.

On a single node the actors inherit the trainer's environment, so `HALO_SANDBOX_*` set at launch
reaches them; on a shared cluster see the export rule below.

## Multi-node

The intended large topology is a training node (or nodes), GPU-less CPU nodes for the actor tier,
and inference nodes for the rollout servers — the two deployment scenarios are drawn in
[Environmental GRPO — Multi-node deployment](../training-methods/grpo/environmental-grpo.md#multi-node-deployment).
Start the cluster manually — there are no helper scripts:

```bash
ray start --head --port=6379 --include-dashboard=false   # head node
ray start --address="head-node:6379"                     # each worker node
```

then set `ray_address: "head-node:6379"`. `--include-dashboard=false` is required, not optional:
`ray start` defaults it to on wherever the `ray[default]` dashboard deps are installed (they are, it
is the pin), and `ray.init` joining that head cannot turn it back off — the head would serve the
jobs API on 8265 the section above says is never opened. The toolkit configures no other Ray ports;
upstream defaults apply. What must hold across nodes:

- **Rollout-server URLs must resolve from every actor node** — actors call the servers directly
  over HTTP. The trainer warns when a loopback URL meets a multi-node run (a scattered actor
  would hit a node with no server and return masked rows).
- **Actor nodes need the environment's runtime**, not GPUs — per env: a sandbox plus a large
  `TMPDIR` for the coding envs and `swe` (an SWE session's whole working tree lives in `TMPDIR`
  for its episode); outbound network plus the backend's key for `react_search` / `qa_search` /
  open-book `exam_qa`; `npx`/`uvx` for `mcp`. The per-env list:
  [Environments](../training-methods/grpo/environments/index.md).
- **Actor processes inherit the `ray start` daemon's environment, not the trainer's.** Export
  `HALO_SANDBOX_*`, `TMPDIR`, search/MCP keys (`SERPER_API_KEY` / `BRAVE_API_KEY` /
  `TAVILY_API_KEY`), and any HF/proxy vars *before* `ray start` on each node; nothing forwards
  the trainer's env to remote actors.

## Data through the object store

Episode results are the heavy Ray payload: a full multi-turn trajectory with token ids, per-token
logprobs, and — under [R3 routing replay](../training-methods/grpo/environmental-grpo.md#off-policy-mismatch-and-stability-knobs) —
the routing mask, which stays base64-encoded end to end and so remains compact on this path. The
trainer gathers lightweight per-episode summaries across ranks, never full trajectories. The
object store itself runs at Ray's defaults (~30% of RAM, capped by `/dev/shm`); a
`--shm-size=128g` container covers it, and the toolkit configures no spilling.

## Monitoring

The trainer logs three cumulative counters on every **logging** step (from `log()`, not per training
step): `async/total_rollouts`, `async/cumulative_mean_rollout_latency`, and
`async/total_generation_tokens`. They are accumulated from the per-episode summaries the trainer
already gathers across ranks each round, so they are **job** totals, not one rank's pool — and no
collective, and no call into the actors, happens inside `log()`. Nothing is read back off the actors:
a `ray.get` fan-out there would block this rank inside `log()` on single-threaded actors busy
mid-rollout while its peers move on to the next NCCL call, and NCCL then times out the job.

From the host: `ray status` (cluster resources, PENDING actors), `ray logs` (actor stderr). The
dashboard stays off (above). `py-spy` note: the toolkit's stack-dump helpers discover torchrun
PIDs only — actor processes are children of the raylet, so attach explicitly with
`python scripts/profiling/py_spy_diag.py dump --pid <actor-pid>` (the subcommand is required; actor
PIDs show as `ray::EnvironmentActor` in `ps`).

| Symptom | Cause → fix |
|---|---|
| Actors `PENDING`, rollouts never start | No free CPU anywhere in the cluster (a saturated local node spills, it does not pin) → free cores or lower `num_rollout_workers` |
| `ray.exceptions.RayActorError` in logs | Actor process died (OOM, native crash); the episode is masked and the actor restarts — check `ray logs`, reduce per-node concurrency if OOM |
| Rollouts fail with connection refused on some ranks only | Loopback/unresolvable server URL from an actor node → use routable hostnames in `rollout_server_configs` |
| Search/MCP env scores all zero on a shared cluster | Backend key or `npx`/`uvx` missing from the actor node's env → export before `ray start` |
| Sandbox runs queue far beyond core count | Per-process gate × many actors → set `HALO_SANDBOX_MAX_CONCURRENCY` so slots sum to cores |
| `episode_timeout` errors while the env is healthy | Deadline below real episode latency (long SWE sessions hit this first) → raise it, keeping it under the NCCL watchdog (validated when the trainer builds the rollout config, not at parse) |
