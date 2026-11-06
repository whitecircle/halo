# Rollout Servers

A rollout server is a **separate inference container**, vLLM 0.32.0 or SGLang 0.5.22, that generates
the rollouts during RL. The trainer sends prompts over HTTP and pushes updated weights back over a
per-server NCCL group through the clients in `halo/weight_sync/`; it never imports either engine
(ABI-incompatible stacks). [Online GRPO](../training-methods/grpo/online.md) is vLLM-only;
[Async GRPO with Environments](../training-methods/async-grpo/README.md) selects the engine with
`rollout_backend`. Every other server knob is engine-neutral:

```yaml
rollout_backend: vllm            # default; or sglang
rollout_server_configs:          # one entry per server
- url: "http://localhost:8000"
  group_port: 51216              # optional; server N defaults to rollout_group_port + N (0: kernel-assigned)
- url: "http://localhost:8001"
  group_port: 51217
  group_host: "127.0.0.1"        # optional per-server override of VLLM_/SGLANG_GROUP_HOST
```

`url` is required. Group ports are bound on the trainer host and must be unique across servers
(enforced); the defaults sit inside Linux's ephemeral range (32768–60999), so reserve them
(`net.ipv4.ip_local_reserved_ports`) or pick ports outside it. The config refuses any other key, a
missing `url`, a `group_port` outside 0–65535, and a non-default `rollout_server_url` beside a non-empty list. The
trainer-side mechanics (prefetch and its staleness, sync cadence, trajectory length) live on
[Async GRPO → Weight synchronization](../training-methods/async-grpo/setup.md#weight-synchronization).

| | vLLM 0.32.0 | SGLang 0.5.22 |
|---|---|---|
| Trainers | online GRPO, online SDPG, async GRPO | async GRPO |
| Sampled-token ids | `--return-tokens-as-token-ids` (server flag) | per-request `return_meta_info` |
| IS-reference logprobs (the sampling distribution's) | `--logprobs-mode processed_logprobs` (server flag; the default `raw_logprobs` is pre-temperature and refused at any `rollout_temperature` ≠ 1) | default (post-temperature, pre-nucleus; keep `SGLANG_RETURN_ORIGINAL_LOGPROB` unset) |
| [R3 routing replay](../training-methods/async-grpo/objective.md#routing-replay) | `--enable-return-routed-experts` + `--moe-backend triton` | `--enable-return-routed-experts` + `--moe-runner-backend triton` |
| Thinking budget (`rollout_max_thinking_tokens`) | enforced engine-side by Model Runner V2 with a reasoning parser; refused for Mistral 4 ([Reasoning budgets](../training-methods/async-grpo/reasoning.md#reasoning-budget)) | refused at config time |
| Expert layout on sync | the layout the family's `ExpertLayout` writes, per-expert or fused, under its hub keys | the same, read by 0.5.22's per-family loaders |
| Trainer expert distribution (EP/ETP) | supported | supported |

Both engines fill a sampling field a request omits from the model's `generation_config.json` (vLLM
`--generation-config auto`, SGLang `--sampling-defaults model`), so every request states `top_k`,
`min_p` and `repetition_penalty`: async GRPO and its evals send `rollout_top_k`, `rollout_min_p` and
`rollout_repetition_penalty` (off by default; SGLang refuses a `top_k` of `0`), online GRPO TRL's own.
A model default such as Qwen3.6's `top_k: 20` then never filters the distribution the trainer scores.

Use vLLM unless a run needs SGLang specifically.

## Weight sync

Both engines receive the **full model** every sync: merged LoRA touches ~95% of bytes, so there is no
delta path (~42 GB at 20B), unless a LoRA run serves its adapter apart ([Adapter sync](#adapter-sync)).
Re-assembling the sharded weights is a collective every rank joins.

One rank owns the clients and sends: the **forwarding rank** (global main; TP rank 0 under TP). A
failure there (an engine 500, a refused tensor, a host OOM on the snapshot, a failed receive while it holds earlier
experts, which it then drops) is recorded rather than raised, so the rank stays in every remaining collective and the
whole world raises together at the end of the sync, naming the failing rank and its cause. A receive that fails with
nothing to drop is the peers' own footprint: it raises on that rank and the peers wait for the NCCL watchdog, as under
any collective a rank never enters.

The gather **reshards the FSDP2 modules first**: a forward leaves transient unsharded params
registered while the optimizer steps the shards, so reading them would ship a policy one step behind.
A LoRA run folds each adapter into its base weight out of place as it sends it, one tensor's
temporaries at a time ([PEFT](../optimization/peft.md#online-rl--rollout-server-weight-sync)).

### Group rendezvous

Each client hosts its group's rendezvous on the trainer: a TCP store on `group_port` that the engine's
workers dial at the advertised group address to fetch the NCCL bootstrap. The store takes
unauthenticated connections until the client closes its communicator, and vLLM's workers unpickle what
they read from it, so a host that reaches the port while a group forms can run code in the server
(SGLang's c10d reads raw bytes under the same trust assumption). **The group port must be reachable
only by trusted hosts.**

The store listens on the advertised address alone: `group_host` / `VLLM_GROUP_HOST` /
`SGLANG_GROUP_HOST` when set; else loopback when this host binds the server's address itself; else
this host's address on the route to the server (the interface the kernel would send from; a server no
route reaches is refused). A name is resolved once to IPv4 and the engine is sent that address. An
address this host cannot bind (a NAT or port-mapped address, `0.0.0.0`, a name resolving to loopback
for a remote server) or one with no IPv4 address raises before the engine is asked to join; IPv6 is
unsupported.

The client binds the listener on `group_port` (a kernel-assigned port at `0`, which is the one the
engine is sent) before the join request and hands it to torch's `TCPStore` (`master_listen_fd`), so a
port another socket holds fails with `Could not bind the weight-transfer group port` and no server is
left waiting on a group that cannot form.

`HALO_WEIGHT_SYNC_BIND_ALL=1` listens on every interface and advertises the configured address
unresolved, for a trainer the server reaches through NAT or a port mapping. The mapping must keep the
port, and the trainer must reach its own advertised address, since the store's own client dials it.

### Streaming and quiesce

The push is **streamed**: an update is a sequence of declared chunks inside one quiesce.
`/start_weight_update` … N × `/update_weights` … `/finish_weight_update` on vLLM (the last closes the
layerwise-reload phase); N × `/update_weights_from_distributed` inside a `/begin_weight_update` …
`/end_weight_update` session between `/pause_generation` and `/continue_generation` on SGLang. The
forwarding rank stages one chunk on its sync GPU, not the model (~800 GB at 400B): packed, double-
buffered buffers on vLLM, typed chunks on SGLang, each the size of its chunk. A chunk is cut before
`HALO_WEIGHT_SYNC_CHUNK_MB` (1 GiB) is exceeded, and a larger tensor rides alone. A chunk staged in
pinned host memory instead crosses PCIe twice before the NIC sees it: 19–24 GB/s against 53–80 GB/s
staged on the device (8× B300 nodes over EFA; the vLLM rate measured on vLLM 0.26.0, the SGLang rate
measured on SGLang 0.5.17).

The sync's expert gather delivers each layer to the forwarding rank alone, over the EP group's ring
(`gather_cat_to_rank`: SUM-reduced byte windows, bit exact, one window on every other rank), so no other rank
assembles a layer (a ring reduce moves about as many bytes per link as an all-gather; the gain is the absent
assembly on peers, the skipped repeat in other EP groups and the overlap with the send). Under deferred DP the EP groups that do not hold the forwarding rank join no expert collective, and
expert-TP shards still gather across their group. Layer k+1's gather is issued on a side stream before layer k is
forwarded (`HALO_WEIGHT_SYNC_GATHER_AHEAD`, 1 by default, 0 for strictly in turn), in the same collective order on
every rank; each layer ahead keeps one more assembled layer on the forwarding GPU. Gathered checkpoint saves keep the
all-rank gather.

The forwarding rank's peak during a sync is the assembled EP layer being sent (~13 GB for one 397B
layer, and the same again for the layer issued ahead), plus the snapshot of the chunk being sent and the engine path's staged copy of it, each the size of that
chunk: SGLang's arena, grown to the largest chunk, or vLLM's packed buffers, so 12.9 + 12.9 + 8.6 + 8.6 GB at 397B
with the default one layer ahead (calculated from the fused bank's 8.6 GB; 12.9 GB less at `HALO_WEIGHT_SYNC_GATHER_AHEAD=0`).
vLLM stages a whole chunk before broadcasting it, so a single server also holds one chunk's packed copy, where it held
two fixed buffers before; lower `HALO_WEIGHT_SYNC_CHUNK_MB` to bound it. The staging is made once per chunk and every server's client broadcasts
from that one read-only buffer, in `per_server` and `single_group` alike, so the figure does not grow with the number
of servers (`halo/weight_sync/staging.py`). A family whose hub split is a reverse converter (Step-3.7) also holds the
split copies beside the layer.

With several servers (`rollout_server_configs`) one snapshot per parameter is shared and released once every
server has it. Under `per_server` each chunk goes to every server on concurrent threads over a group of its own; the
threads share the forwarding rank's GPU, NICs and process, so that fan-out costs the sum, not the max: two servers
each push at half of one server's rate. `rollout_sync_topology: single_group` sends each chunk once
([below](#one-group-over-every-server)), and `auto` picks it for a homogeneous fleet.

**The quiesce spans the streaming.** It opens with the first full chunk and closes when the last
lands, so a server stops serving for as long as the gather runs (minutes at 397B), and every push
pauses all servers together
([single vs multi-server](../training-methods/async-grpo/setup.md#multiple-servers-and-prefetch)).

- **vLLM** is paused with `/pause?mode=keep`: in-flight generations (the prefetched rounds) freeze and
  resume under the new weights, the staleness the sampling-logprob IS ratio corrects. vLLM's
  default `abort` would hand every in-flight request back as a fragment with an ordinary stop reason.
- **SGLang** is paused with `{"mode": "abort"}` (its post-update cache flush refuses while a request is
  running), so in-flight generations drop. The rollout actor re-issues an aborted turn
  (`finish_reason: abort`) against the same observation, up to `max_retries` times; an abort is never
  charged as a length cut. A prefetch whose rounds span more syncs than `max_retries` is refused
  ([Prefetch](../training-methods/async-grpo/setup.md#multiple-servers-and-prefetch)). The refusal
  reads the client's `PAUSE_ABORTS_IN_FLIGHT`, true only on SGLang's client.

The paused window is charged to neither the episode nor the request: every rank credits the push to
its in-flight episodes' `episode_timeout` and, through the Ray actors' clock, to their requests'
`request_timeout`; `async/requests_expired_in_sync` counts the ones that overran anyway.

A chunk cannot be replayed. A server that fails before any chunk went out (an engine restarted between
syncs) is recovered by a reconnect and re-flush; one that fails **after** its first chunk, the close of the update
included, holds neither the old policy nor the new, and vLLM's layerwise reload materializes any layer straddling the
boundary from *uninitialized* storage. The abort therefore leaves that engine **paused**, refuses every later
sync on that client and logs `RESTART the … server`: restart the container. A resume that fails after a good push is
retried three times, then raises naming the server, since every rollout would otherwise queue until its request
timeout.

Forming one group is bounded by `HALO_WEIGHT_SYNC_FORMATION_TIMEOUT_SECONDS` (120 s): the engine's join and the
trainer's rendezvous. It does not reach the SGLang process group, whose watchdog uses the chunk deadline below.
Each chunk runs under one 600 s deadline on both engines (`HALO_NCCL_SYNC_TIMEOUT_SECONDS`): the
update request that declares it and the drain of its sends. A failed chunk drains the stream under a
30 s deadline, and on expiry the group is aborted instead of parking the trainer. Closing an SGLang group is bounded
the same way: a server that does not answer `/health` is aborted outright, and a destroy still blocked after 30 s, as
under a wedged engine that never enters its half, is aborted too, so the failure that brought the close raises. An
abort runs on its own thread and the close returns after 30 s even when torch's abort does not, as it waits on a
communicator another thread may be destroying; a group left that way is not released.
SGLang's `/health` generates through the scheduler, so the client asks it before the destroy occupies the scheduler,
for longer than the 20 s the server allows a step. It waits for a server on `/v1/models`, which a pause does not hold
and which answers ahead of the warm-up generate, then through `/health`'s 503s until that generate ends, as its first
forward can outlast the 60 s generation probe on a cold kernel cache.
vLLM unloads the model when an update session opens, and a failed chunk ends the session with it, so the client reads
`instance_id` off `/server_info` before each open. A restart between syncs then reconnects and re-sends instead of
opening a session on an engine with no group; a session no close can end leaves the server paused, to be restarted.

Both clients send from their own CUDA streams and allocate their buffers on the caller's stream,
recorded for the send stream, so a reconnect strands no allocator pool; `HALO_WEIGHT_SYNC_MEM_LOG=1`
brackets each sync with a per-rank `[mem rankNN] weight-sync pre/post` line, and `reserved` far above
`peak_alloc` on the forwarding rank means stranded pools
([Debugging](../reference/debugging.md#3-gpu-memory-profiling)).

### One group over every server

`rollout_sync_topology: single_group` forms one NCCL group of the trainer (rank 0) and every server's workers, so
each chunk is broadcast once. It belongs to environmental GRPO, which drives several servers; online GRPO talks to
one server and has no such field. The default `per_server` is unchanged. Both engines take the join's `rank_offset` and
`world_size` as given, so no engine patch is involved: SGLang ranks a worker `rank_offset + tp_rank`, vLLM
`rank_offset + dp_index * (TP * PP) + rank_in_dp`, and each server takes the next block after the one before it.

- **GPU visibility.** Servers on one host talk over NCCL peer-to-peer, which needs every peer's GPU visible to each
  server process: mount all GPUs per server and select with `CUDA_VISIBLE_DEVICES`, as the compose files do. Servers that
  each see only their own GPU (one GPU per container or pod) fail the first push with `invalid device ordinal`; keep them
  on `per_server`.
- **Layout.** Servers sort by their entry's `node` label (default: the URL host; config order kept within a node), so one
  node's servers are one contiguous rank block and a ring over the group crosses each node's links once. Two labels
  resolving to one address raise; an IP and a DNS name of one node are such labels, so give them one `node`. Neither
  engine reports its host, and a server whose TP spans nodes cannot be placed: the label is a requirement.
- **`auto` resolves.** `rollout_sync_topology: auto` (opt-in; the default is `per_server`) is `single_group` for two or more servers whose
  `/server_info` report one served model and one value of every flag `verify_group_servers` compares, and `per_server`
  for one server and for any fleet that does not qualify: a mixed fleet, a server that does not answer, a
  pipeline-parallel server, local servers beside remote ones with no `group_host`, a node split across labels, a
  per-entry `group_port` or differing `group_host` values. The pool reads the servers when it forms the groups
  (`resolve_sync_topology`, `halo/weight_sync/group_eligibility.py`), logs `Weight-sync topology <name>: <why>` once, and
  an explicit `single_group` or `per_server` is never overridden. `auto` refuses a `node` label, which only orders
  one group. A restart re-forms the topology it resolved; it is not read again.
- **Config.** One rendezvous on `rollout_group_port`, one `group_host`. A per-entry `group_port`, differing
  `group_host` values and a `node` label under `per_server` raise where the config is built, and again where the pool
  is built, from one validator (`validate_sync_topology`). Servers on this host beside servers on others advertise one
  address that is loopback to the remote ones or the external NIC to the local ones, whose hairpin traffic firewalls
  drop, so formation refuses that mix unless `group_host` is set. `verify_group_servers` (startup preflight) refuses
  servers that serve different models or disagree on any launch flag `/server_info` reports (dtype, quantization, MoE
  backend, parsers, strict tool calling, routing capture, `torch.compile`), and a server that does not answer for
  them; a flag a server does not report is not compared. The image identity is checked per server when its engine
  world is read. SGLang with `--pp-size > 1` is refused at formation, as its stages would all rank
  `rank_offset + tp_rank`; `per_server` is not touched.
- **Formation.** Every server's join RPC and the trainer's half start together, as vLLM's join blocks until the whole group
  arrived and SGLang's first collective needs every rank; the first failure destroys every engine's half and raises
  naming the server. The group name carries a generation (`weight_update_group_g{n}`), and an SGLang join clears a dead
  trainer's group under the group's name, and under the per-server name on a client's first join and later only if it
  formed that name. A vLLM server whose `instance_id` cannot be read
  refuses formation, as its restart could not be told from a live group.
- **A sync.** Each server is probed (vLLM's `instance_id`), then all are paused with their sessions opened together; the
  log names each server's pause and how long the first chunk waited on the slowest. SGLang then takes a membership
  probe, an empty declare that a server holding the group answers as a no-op. A server found restarted by either check
  re-forms the group once, with every server paused, reading the layout and launch of the servers that lost the group
  again (one that kept it is still the engine it formed with, and a scheduler parked behind the lost member's destroy
  would stall the read), and the sync replays; a second miss closes the sessions, resumes every server and raises. A
  re-form aborts the trainer's half before any server destroys its own, as a lost member never enters a graceful
  destroy; the wait is bounded, logs its expiry, and a survivor that stays parked is not freed by it. An SGLang join
  whose session close gets no answer in 30 s raises naming the server, which then needs a restart. Every bounded wait of
  a re-form or teardown (a layout read, a group destroy, an abort, a session close) logs its server, purpose and bound
  once when it starts, and once more when the bound expires. A wait expires when its call is still running after the
  join or ended in a request timeout. A survivor's destroy during a re-form is waited out to its bound, since SGLang has no route
  to abort a server's half. Its scheduler runs `destroy_process_group` on the old group inline and serves no other
  request until it returns, about 60 s after a peer died, so an SGLang re-form after a server restart takes about 60 s. An SGLang release stops the
  rendezvous socket listening itself, so a destroy or abort left parked, which keeps the store alive through its group, does not hold the group
  port. Each chunk is declared on every server and sent once, paced by the slowest server; new weights go live per
  server at its own close.
- **A failure after the first declare is global.** A server that refuses its declare after peers accepted theirs
  leaves them parked in a receive no trainer-side call frees, so the group aborts the trainer's half, marks **every**
  server unusable and leaves it paused, and raises `RESTART the … servers:` with the failed one first. The pause stays
  until all are restarted; chunk 0 is not replayed. A resume that fails after a good push is retried three times, per
  server.
- **Not covered.** An engine-side bounded receive, a dedicated SGLang membership route and the engines' reporting of
  host and NCCL environment need image changes. SGLang `single_group` rests on the probe: the GPU row
  `trainers/grpo/test_sglang_membership_probe.py` decides whether it stays. The measured gain over `per_server` waits on the GPU rows of `test_single_group_sync.py` and `test_sync_topology_auto.py` (1, 2, 4 and 8
  servers per engine, cross-node and colocated containers). The `single_group` row bounds the second push of a group,
  which follows process and communicator warm-up, and reports the first beside it. It bounds the transfer (the push
  less a one-tensor push's fixed cost) at 1.5× one server's, and the fixed cost, the pause, begin and end calls and
  chunk declares each server answers concurrently, at 20 ms per extra server.

### Adapter sync

`rollout_adapter_sync: true` serves a LoRA policy as an adapter beside the frozen base instead of pushing it folded
into the weights. SGLang only: vLLM loads an adapter from a directory both containers mount, and its decode with an
adapter served ran over 5% slower than merged weights on the model measured.

- **First sync.** The base goes out as above, adapters left out (`fold_adapters=False`), then the adapter loads. A
  restarted trainer first retires the `halo-adapter-*` versions its predecessor left on the engines.
- **Each later sync.** The adapter is gathered whole on the forwarding rank, B is multiplied by `lora_alpha / r` (or
  `/ sqrt(r)` under rsLoRA) in fp32 and rounded to bf16 once, and the adapter loads under a new name with
  `lora_alpha = r`, so the engine's own scaling is exactly 1. No pause and no session: rounds in flight keep running.
- **Transport.** `/load_lora_adapter_from_tensors` with a plain pickle inline in the JSON body, one copy per tensor
  parallel rank. The route's own `ForkingPickler` form hands CPU tensors over by file descriptor and CUDA tensors by
  IPC handle, which works on one host only. A per-version directory needs the same mount in the trainer and every
  engine and cannot be used where containers forbid bind mounts. A path load of a 7 to 27 MB adapter took 0.03 to
  0.04 s on Qwen3.5-35B-A3B; the tensors route is timed by the GPU row.
- **Selection.** Every episode leases the newest version for its whole length and its requests send
  `model: <model_name>:<version>`, so a multi-turn episode and a prefetched round keep the version they started on.
  A request with no version published raises, since the engine would serve the base. The `isr_engine_reference`
  re-score sends `lora_path` with the newest version.
- **Retirement.** After each load the ranks join the versions their episodes still hold, and rank 0 unloads the
  others. An episode cancelled at its deadline can outlive its lease; the engine's unload then waits for its request.
- **Restart.** A server whose `/v1/models` lacks the newest version (it restarted) gets the base and the adapter
  again at the next sync; the episodes that were running under the lost version fail and are counted as errored.
- **Launch flags.** `SGLANG_LORA_MAX_RANK` turns the compose flags on ([variables](#sglang)). The startup preflight refuses a server whose rank, module list or loaded
  limit is too small for the run (`prefetch_depth` + 2 versions are live at once), one with speculative decoding
  (its draft runs without the adapter), a quantized base or data-parallel replicas. A module outside
  `--lora-target-modules` is dropped by the engine without a trace, hence the module check.
- **Cost.** On Qwen3.5-35B-A3B, SGLang 0.5.22 decoded 2.6% slower with `--enable-lora` and no adapter against a warm
  base, and 3.0 to 4.6% slower with an adapter selected; vLLM missed the 5% bar.

Refused at construction, with the reason in the message: a backend other than SGLang, `routing_replay: rollout`, a
`model_name` that is unset or holds a colon, QLoRA, TP, PP and CP, native expert LoRA, DoRA, `lora_bias`,
`modules_to_save`, `rank_pattern` and `alpha_pattern`, a grown vocabulary, any trainable parameter that is not an
adapter, adapters not in bf16, an adapted module other than the attention and dense-MLP linears
(`ADAPTER_SERVABLE_MODULES`), and one whose hub name a converter fuses or splits. Router bias-update balancing is
already refused by every weight sync. Run it with the GPU rows in
`tests/gpu/trainers/grpo/test_adapter_sync_e2e.py`, which decide whether it stays.

### Served weights stay in checkpoint layout

The sync writes bf16 checkpoint-layout tensors into the server's parameter storage in place, so any
load-time transformation of that storage silently corrupts every later update. On vLLM the
auto-selected Blackwell MoE backends (`FLASHINFER_TRTLLM` / `CUTLASS`) repack expert weights:
`--moe-backend triton` is **required** for MoE RL. On SGLang the `triton` runner loads bf16 unpacked;
the `flashinfer_trtllm*`, aiter and quantized runners repack after the load, and an update into them is
not validated.

The same rule excludes quantized serving: serve bf16. For gpt-oss that means the **BF16** checkpoint,
not the stock MXFP4 `openai/gpt-oss-20b`, whose expert loader drops every synced bf16 expert weight
while the biases land, silently, leaving only a slow log-ratio drift. Checkpoint layout and expert
un-fuse rules: [Checkpoints](../reference/checkpoints.md#serving-on-vllm--sglang).

### vLLM server patches

The image applies six runtime patches through `sitecustomize` in the API server and every engine-core
worker (`docker/vllm/patches/`), and every one is asserted at build:

- **Layerwise reload** (`vllm_layerwise_reload_patch.py`). vLLM's reload moves each layer to the meta
  device and wraps its `weight_loader`s; model code that writes with a direct `param.copy_()` (gpt-oss
  experts and attention `sinks`) writes a no-op, and the reload re-registers the saved tensors, so
  every sync reverts those weights while `/update_weights` returns `200 OK`. The patch excludes them
  from the reload (`SKIP_LAYER_NAMES`: `RoutedExperts` / `FusedMoE`, `OAIAttention`, and `Gemma4Router`,
  whose partial load would materialize an uninitialized buffer), checked against the installed vLLM's
  classes through the MRO. Missing, the server logs `RoutedExperts: Failed to load weights` per expert
  layer per sync while `sampling/logratio_mean` drifts negative (`YaRNScalingRotaryEmbedding: Failed to
  load weights` is benign). Do not override `PYTHONPATH` at `docker run`: dropping `/opt/nccl_compat`
  kills weight-transfer init (`No module named 'halo'`).
- **Weight-transfer re-init** (`vllm_weight_transfer_reinit_patch.py`) destroys the previous NCCL
  communicator before `/init_weight_transfer_engine` builds the next. Stock 0.32.0 drops it without
  destroying it, so each trainer connection strands a communicator holding device memory on every
  engine-core worker until `ncclCommInitRank` fails while `/health` still answers `200`. The trainer
  aborts its own half on `close_communicator()`; on SGLang the client also asks the engine to destroy
  the group (`/destroy_weights_update_group`) and clears a stale registration at every join.
  `tests/gpu/trainers/grpo/test_weight_transfer_reinit.py` cycles both backends.
- **Two thinking-budget patches** (`vllm_reasoning_forced_end_patch.py`,
  `vllm_thinking_budget_completion_patch.py`) let the gpt-oss parser force its whole move into the final
  channel when the analysis channel overruns its budget, on Model Runner V2's budget
  ([GPT-OSS](../models/gpt-oss.md#serving-for-grpo-vllm)).
- **Split top-p guard** (`vllm_topp_split_guard_patch.py`) adds the monolithic kernel's logit-space
  no-mask check to the split top-p pipeline vLLM runs for the top-k-off rows of a batch of at most 64.
  Without it a budget cut (the forced token's logit is 1e9) at `rollout_top_p < 1` masks the whole row,
  and the engine samples token 0 and reports a NaN log-prob.

- **Image identity** (`vllm_image_identity_patch.py`) lists these patches as `VLLM_HALO_PATCHES` on `/server_info`,
  where the trainer refuses a server without it ([Startup preflight](#startup-preflight)). It lands only where
  `sitecustomize` ran, so a `PYTHONPATH` override loses it with the patches.

**Build-time source edits** (`docker/vllm/patches/patch_vllm_source.py`, through
`docker/engine_source_patch.py`, the patcher the SGLang image shares) turn vLLM's harmony pipeline off
for gpt-oss, point the server log at stderr and move FlashInfer CUTLASS to the back of the unquantized
MoE backends. Each edit asserts its pre- and post-image; `python3 /opt/halo/patch_vllm_source.py
--verify` re-checks a running container.

**Model Runner V2** is pinned (`VLLM_USE_V2_MODEL_RUNNER=1`): it enforces thinking budgets, is the only
runner R3 capture starts on, and computes prompt log-probs before the drafter under speculative
decoding (the V1 runner's drafter graph overwrites the target's hidden states, so prompt log-probs come
back garbage, [vllm#53488](https://github.com/vllm-project/vllm/issues/53488)). The
`isr_engine_reference` startup probe refuses a server whose prompt log-probs do not follow from its
tokens, whatever the cause ([Engine reference](../training-methods/async-grpo/objective.md#engine-reference)).

**Build gates** (`docker/vllm/verify_image.py`, `docker/vllm/patches/verify_topp_split_guard.py`,
`docker/vllm/plugins/verify_gptoss_plugins.py`) assert the patches, the transformers/tokenizers pins,
the `ncclUniqueId` pickle shim, the weight-transfer wire (init and update request fields, the packed
chunk boundary, the routes), that `/pause` accepts `mode=keep` and resets the prefix cache under it,
that the split top-p guard keeps a budget-cut row whole while masking an ordinary row as upstream does,
that the gpt-oss parser plugins drive Model Runner V2's budget, that `EngineArgs` carries
`weight_transfer_config` / `enable_return_routed_experts` and `KernelConfig.moe_backend`, and the
per-family [config-schema parity](#config-schema-parity). A green build is what proves the server can
be synced.

### Partly loaded layers

vLLM's layerwise reload counts every tensor a module holds, less the names on its skip list
(`SKIP_LOAD_TENSORS`), and processes the module once the count is met. A module that received some of its
tensors is processed at `/finish_weight_update` instead: the missing ones are materialized with
`torch.empty_strided`, and that storage is copied over the live parameter. The update answers `200 OK`.
A module that received nothing keeps its tensors, and a name on the skip list stays out of the count and
off the meta device.

So a hub tensor the trainer holds as a buffer and the engine holds as a parameter beside pushed ones has
to ride the push. Ling 2.0's router bias is that case: vLLM spells it `GateLinear.expert_bias`, which the
reload counts, so the family spec declares it (`engine_loaded_buffers`) and the gather forwards it whenever
`moe_router_enable_expert_bias`, the flag both engines create the tensor on, is set. Left out, the served
router bias reads back as whatever that storage holds, which keeps the model plausible and moves most prompt
log-probs. Weight-sync RL runs `moe_balancing: none`, so the buffer holds its checkpoint's value and the push
leaves it where the server loaded it.

The other router biases stay out of the push and stay exact:

| Family | Hub tensor | Why vLLM keeps it |
|---|---|---|
| GLM-4 MoE Lite | `gate.e_score_correction_bias` | the name is on the skip list (`docker/vllm/verify_image.py` asserts it) |
| LFM-2 MoE | `expert_bias`, mapped onto `gate.e_score_correction_bias` | the same |
| Step-3.7 | `moe.router_bias` | it sits alone on a module that receives nothing |
| Gemma 4 | `layer_scalar` (not a bias) | the same |

Laguna's bias and GPT-OSS's `router.bias` are trainer parameters, so they are pushed anyway. SGLang's loaders
copy name by name with nothing to count, so a tensor left out keeps its value on every family, and Ling's bias
lands on a parameter of the same name. `tests/cpu/grpo/test_weight_sync_engine_loaded_buffers.py` holds every
syncable family's push to its gathered checkpoint, and `tests/gpu/trainers/grpo/test_weight_push_exact.py`
proves on a live server, per family and engine, that an unchanged push leaves the served log-probs bit-identical
([Coverage](#coverage)).

The TP=2 rows stop there. The others take one real optimizer step and push it. The served log-probs must then sit
within twice the trainer-versus-served gap measured before the step (at least 0.01 nats), since the two forwards disagree on identical
weights ([noise](../training-methods/async-grpo/objective.md#trainer-and-engine-noise)). On its fixed rows that gap
reads 0.02–0.06 nats on Qwen3 MoE, Qwen3.5/3.6, GLM-4 MoE Lite and Ling 2.0, 0.16 on LFM-2, 0.31–0.47 on Gemma 4,
0.38–0.40 on GPT-OSS and 0.60–0.65 on Laguna. At the top of that range the bar is too loose to catch a small sync
error, so there the bit-identical rows, the CPU test and `--reach` hold the sync: `--reach` scales each forwarded name
in turn and requires the served log-probs to move, since an engine that drops a name still answers `200`, and only a
change that fails to reach the model shows it.

### First-push parity probe

The probe detects a scrambled or mis-shaped load. It does not check that the engine consumed every pushed name: an
engine answers `200` for a tensor its loader does not map and keeps the launch value, and one it stacks or reshapes
wrongly serves another model. After a trainer's first push (online GRPO's, the environmental trainer's forced start
push) every rank scores one fixed 1,100-token counting row with the trainer's own log-prob path
(`_get_per_token_logps_and_entropies`), and rank 0 scores it through the client's `score_completion_logprobs` (a pool
scores on its first server; the row is cut to the served window). A mean gap above 1 nat per position
(`PARITY_MAX_MEAN_GAP_NATS`) is logged as a warning. The bound sits above the 0.65 floor measured on Laguna; no other
family, QLoRA trainer or composite has been measured, so the default warns instead of stopping a run that may work.
`HALO_WEIGHT_SYNC_PARITY_PROBE=raise` stops every rank on a gap, `off` skips the probe; context parallelism skips it,
as its batches shard the row. Later pushes pay nothing.

On a fresh run the trainer and the engine hold the same weights, so a dropped name reads exact until training moves
it; the probe sees one only on a resumed run or a server launched from another revision.
`test_weight_push_exact.py --reach` is the per-name check on a live server, and the recorded loader names
([GLM-5 Next](../models/glm5-next.md#rl-weight-sync)) pin the names per engine on the CPU tier.

### Construction gates

The online and async GRPO trainers refuse at construction a model the selected engine cannot take a
sync for (`reject_unsyncable_model`, `halo/trainers/grpo/rollout/weight_push.py`), on every rank alike. Their scripts
settle the family's verdict off the checkpoint's config first (`reject_unsyncable_config`), before the datasets, the
model load and the server preflights:

- **The family.** An engine fact on that engine's client (`UNSERVABLE_MODEL_TYPES`,
  [below](#which-families-each-engine-serves)), or a `weight_sync_refusal` on the family spec, which
  Inkling (no engine run has validated its sync), Qwen3-Next, Qwen3.8-Flash-Next (no engine run has validated its sync, and
  SGLang caches a hyper-connection weight copy on SM100-family GPUs that an update leaves stale,
  [RL weight sync](../models/qwen4_exp.md#rl-weight-sync)), Nemotron-H (no engine run has validated its sync)
  and Cohere2 MoE carry on both engines.
- **A stock module tree no engine layout covers**, in three cases. A family with an EP layer but no live
  one (`use_grouped_gemm: false` at `ep_size: 1`): the experts ship in the layout the EP layer's gather
  emits, and the stock tree's fused tensors under module names are dropped by the engine's loader with no
  error; `use_grouped_gemm: true` (the torchrun default) installs the EP layers. A dense checkpoint under a
  spelling a MoE family claims (`lfm2_vl` over `lfm2`, `mistral3` over `mistral`, `gemma4` with
  `enable_moe_block: false`): no `use_grouped_gemm` remedy applies, so train it offline. Any MoE with no
  family spec, read off the config's routed-expert count: SGLang's loaders drop its fused experts with a
  `200`, vLLM's take them only where the checkpoint spells `gate_proj`/`up_proj` and drop `w1`/`w3` names,
  and no sync into one has run. Train it offline, or [add the family](../models/adding-a-model.md).
- **A bnb-quantized (QLoRA) base**, and PEFT LoRA on a weight the dense push does not send (EP expert
  weights take native expert LoRA instead).
- **GptOss without live sinks.** `reset_sinks: true` under `flash_attention_2` removes the sinks, and
  the sync forwards `named_parameters()` only, so the server keeps the pretrained sinks against a
  sink-free trainer, permanently off-policy. On-policy GptOss needs `reset_sinks: false` with FA4 or
  eager, as the shipped GRPO configs set; `train_sinks: true` (SFT-only) is refused too.
- **Enabled bias-update balancing**, adopted or transient: the push carries no bias that moves between
  syncs. The shipped GRPO scripts downgrade `moe_balancing` to `none` themselves.
- **A declared buffer the EP layer no longer holds** ([Partly loaded layers](#partly-loaded-layers)): an
  upstream rename would otherwise raise mid-push, after the engine took a chunk.

Offline GRPO needs no server and passes none of these gates.

### Startup preflight

Before the second model load rank 0 waits for each server, then reads `/server_info` and refuses, on every rank, what
it reports wrong (`verify_server_flags`, `halo/trainers/grpo/rollout/engine_preflight.py`). A field the reply does not
carry reads as unread, so a renamed upstream field cannot refuse a good server.

Rank 0 does this work alone, so the peers wait for it over the c10d store (`DIST_STORE_TIMEOUT_HOURS`), not in an NCCL
collective under the 30-minute watchdog: a server that loads slowly ends in `ConnectionError` naming
`rollout_connection_timeout`, never in a watchdog abort. Group formation joins the same way
(`rollout/weight_sync_group`). The probes of the servers run concurrently on rank 0, up to `HALO_PREFLIGHT_PARALLELISM`
at once, with results read in server order. Each server's launch-flags line also carries the engine's own launch
facts from `/server_info` (parallel sizes, batch caps, attention backend, graph mode; `ServerFlags.extras`), and a
server that does not answer is logged as unread.

| Refused | Because |
|---|---|
| A server off the toolkit's image (no `VLLM_HALO_PATCHES` in `vllm_env`, no `halo_patches` on SGLang) | Upstream takes the same routes at the same version, with none of the patches: experts revert behind a `200`. Each client also refuses it at group formation; serving-only use without a sync stays free |
| A MoE on a backend other than `triton` | The auto backends repack expert weights and a push corrupts them |
| Servers of a `single_group` run that serve different models or launch flags | One broadcast reaches all of them, and each would load it differently |
| `VLLM_ENFORCE_STRICT_TOOL_CALLING` on, with a tool-bearing environment | A grammar the trainer's log-probs never see |
| No tool parser, with a tool-bearing environment (`/server_info` on SGLang, the `400` a tool-bearing request gets on vLLM) | Calls stay text, and every episode scores alike with no error |
| `routing_replay: rollout` with no `--enable-return-routed-experts`, or with `--enable-torch-compile` | No rows, or corrupt ones |
| No reasoning parser, with `reasoning_price`, `reasoning_floor`, `carry_reasoning` or a vLLM thinking budget | A budget answers `400` on every rollout, the price charges nothing, the floor scores maximal under-use |
| `train_on_sampled_tokens` and a vLLM server whose logprobs carry no `token_id:` entries | Every turn trains on a re-tokenized render |
| A server that, timed on a 64-token completion (the better of two, one idle sequence), would need over twice `request_timeout` to decode a full `rollout_max_tokens` turn (`verify_turn_deadline`); over half of it only warns, since a loaded server decodes slower per sequence | A timed-out episode trains as a masked row, so the batch keeps only the episodes that finished fast ([Monitoring](../training-methods/async-grpo/monitoring.md#halts)) |
| A server whose prompt does not move with the reasoning level, where the trainer's does (`verify_server_chat_template`) | Its template ignores the level, and the reasoning terms charge for a contract the model never read ([Chat template](../training-methods/async-grpo/episodes.md#chat-template)) |

**Hub names.** The sync forwards every tensor under the key a gathered checkpoint would carry: a
family's `export_key_renames`, then transformers' own save-side revert of what the load converted
(`export_conversions` in `halo/checkpoint/hub_layout.py`, the list the gathered save inverts), so
Step-3.7's per-layer `moe.*` stacks and a SigLIP tower's `vision_model` level reach the engine under
their hub keys. One-to-one renames stream tensor by tensor; a tensor a reverse converter claims (a
fused `gate_up_proj` the hub stores split, a vision tower's fused q/k/v) is held until its sources are
complete, since the engine loads one tensor at a time. Held dense tensors are bounded at
`HALO_WEIGHT_SYNC_CHUNK_MB`, and experts are not: the gather keeps them alive until each layer's flush anyway.

### Which families each engine serves

Both engines read a family's experts in the layout its `ExpertLayout` writes: per-expert tensors for
Qwen3 MoE, Qwen3-Next, GLM-4 MoE Lite, GLM-5 Next, Laguna, Bailing and LFM-2, the fused pair for Qwen3.5/3.6 and Gemma 4,
GptOss's interleaved pair. What differs is which families each pinned release can take an online
update for; each client declares those with the loader fact (`UNSERVABLE_MODEL_TYPES`), which the
[construction gate](#construction-gates) quotes.

| Family (`model_type`) | vLLM 0.32.0 | SGLang 0.5.22 | Loader fact |
|---|:--:|:--:|---|
| Mistral4 (`mistral4`) | ✗ | ✗ | neither registers a class ([Mistral4](../models/mistral4.md#serving)) |
| Ling 3.0 (`bailing_hybrid`) | ✗ | ✗ | both register `BailingMoeV3ForCausalLM`; no online update into it is validated |
| Ring (`bailing_moe_linear`) | ✗ | ✗ | the checkpoints declare `BailingMoeLinearV2ForCausalLM`; both register `BailingMoeV2_5ForCausalLM` |
| Zaya (`zaya`) | ✗ | ✗ | vLLM ships no class; SGLang's loader reads the pre-transformers-5.14 per-expert checkpoint (`zaya_block.experts.local_experts.N.linear_fc1`) |
| DeepSeek-V4 (`deepseek_v4`) | ✗ | ✗ | both map per-expert `w1/w2/w3` where the gather emits the fused pair |
| Laguna (`laguna`) | ✓ | ✗ | SGLang's `load_weights` asserts every routed-expert tensor of every sparse layer per call; vLLM's loader drops the exported balancing bias ([Callbacks](../training-methods/callbacks.md#moe-balancing-modes)) |
| Step-3.7 (`step3p7`, `step3p5`) | ✓ | ✗ | `Step3p5ForCausalLM.load_weights` asserts full parameter coverage per call |

Inkling, Qwen3-Next, Qwen3.8-Flash-Next, Nemotron-H and Cohere2 MoE are refused on both engines by their
family spec ([construction gates](#construction-gates)). These sync on both, expert distribution included:
dense families (not a dense checkpoint under a MoE family's spelling), GptOss, Qwen3 MoE, Qwen3.5/3.6 MoE,
GLM-4 MoE Lite, GLM-5 Next ([RL weight sync](../models/glm5-next.md#rl-weight-sync)), Gemma 4, Ling 2.0, LFM-2 MoE. A MoE with no family spec is refused on both. Per-family server
flags: [Step-3.7](../models/step3p7.md#serving-for-grpo-vllm), [Nemotron-H](../models/nemotron-h.md#checkpoints-and-serving), [GLM-5 Next](../models/glm5-next.md#rl-weight-sync).

The sync ships hub names and full unsharded tensors into the engine's own `load_weights`: each TP rank
narrows its slice, and under `--ep-size` the loader keeps its local experts. An expert name that
mapping does not cover leaves **no server-side signal** in most MoE loaders (Qwen3 MoE, GLM-4,
gpt-oss, Bailing, LFM-2): they `continue` before their `not found in params_dict` warning and the
update returns `200 OK`, so the engine keeps its launch experts under a synced router. The server
tier's expert-only round catches it; the construction gate is the whole guard.

Three SGLang 0.5.22 loader facts shape its image and client:

- **Routers that cache a derived form.** Upstream, the Gemma 4 router folds `scale` into its norm once,
  behind a latch, so a synced `scale` never reaches routing. `docker/sglang/patches/patch_sglang.py`
  makes a load of `scale` release the latch, and asserts that the GLM-4 gates still read their fp32
  weight live. It checks its pre- and post-images at build and stays in the image (`/opt/halo/`) for
  `--verify`.
- **Fused a-projection halves.** The MLA loaders (GLM-4 MoE Lite) concatenate `q_a_proj` and
  `kv_a_proj_with_mqa` from a cache local to one `load_weights` call and drop a half that arrives alone.
  The client declares the pair (`CO_LOADED_PARAM_GROUPS`) and the chunker keeps it in one chunk; a
  pair still incomplete at the close refuses the close. A block without `q_lora_rank` has no
  `q_a_proj`, so every push scopes the pair to the pushed model's modules first.
- **The triton runner** (`SGLANG_MOE_RUNNER_BACKEND=triton`, the compose default) is the runner whose
  weights an update reaches unchanged, and the one R3 capture needs.

## vLLM

`Dockerfile.vllm` builds `halo:vllm-0.32.0` from upstream `vllm/vllm-openai:v0.32.0` (CUDA 13) with
the native NCCL weight-transfer engine (`VLLM_SERVER_DEV_MODE=1`, which exposes the pause and update
routes, so keep the server off untrusted networks), the patches above, the gpt-oss parser plugins, R3
capture (base64-npy `routed_experts` per completion choice), the EFA userspace and `nvidia-nccl-cu13`
at `uv.lock`'s exact pin. The build asserts the installed NCCL against the lock and the `libnccl.so.2`
a process resolves, and `VLLM_NCCL_SO_PATH` points pynccl at that wheel so the base image's older
system copy never wins. A skew fails or hangs `ncclCommInitRank` at `/init_weight_transfer_engine`;
rebuild the image after any bump of the pin.

The client reads `GET /version` and refuses a server older than 0.32.0 or one that reports no version:
the packed-transfer geometry (`packed`, `packed_buffer_size_bytes`, `packed_num_buffers`) rides
`/init_weight_transfer_engine`, which older engines refuse. No newer release is validated.

### Config-schema parity

The server parses every checkpoint with its own transformers, the base image's 5.19.0
(`Dockerfile.vllm` asserts it sits inside vLLM's declared range). `docker/vllm/parity/check.py` runs at
image build: vLLM's `ModelConfig` must resolve what the toolkit exports, to a per-layer attention
geometry, for every family whose EP layer admits weight sync and the vLLM client does not list as
unservable. One `config.json` fixture per family is rendered from the tiny roster config in
`tests/common/models.py` (Ling 2.0 through a checked-in vendor config module), or for Step-3.7 from the release
config at a pinned revision. Step-3.7 also has a control under `unparseable/` that must be refused (a control
the server reads fails the build, since its rewrite is then redundant).
`tests/cpu/checkpoint/test_vllm_parity_fixtures.py` fails when the roster or the fixtures drift, and a
transformers bump on either side fails the build rather than the first live sync.

**Step-3.7 is the one family whose export is not transformers' own serialization.** The server's
transformers has a native `step3p7` class, but vLLM's config parsing raises
`AmbiguousGlobalPerLayerAttributeError` on it, so the server reads the family only through the
release's `auto_map` modules, and its exports carry the source repo's config schema and those modules
([Exported `config.json`](../reference/checkpoints.md#exported-configjson)).

### Running it

`docker-compose.vllm.yml` runs the server with `network_mode: host` + `ipc: host`: the workers dial the
trainer's rendezvous on its group port and then NCCL's bootstrap on an ephemeral trainer port, neither
of which a bridge network publishes (formation times out at "1/2 clients joined"). The compose file
takes the image from `VLLM_IMAGE` and builds nothing; it needs `HALO_SCRATCH`, the training service's
scratch volume, even to start the server alone, because compose interpolates the whole file (set it
once in the repo-root `.env`).

```bash
make build-vllm      # or: export VLLM_IMAGE=public.ecr.aws/whitecircle/halo:vllm-0.32.0
VLLM_MODEL=Qwen/Qwen3-30B-A3B VLLM_CUDA_DEVICES=6,7 VLLM_TP=2 TRAINER_CUDA_DEVICES=0,1,2,3,4,5 \
  docker compose -f docker-compose.vllm.yml up vllm-server
```

Without compose, the ENTRYPOINT is `vllm serve` and any args replace the image's `CMD` wholesale, so
restate every load-bearing flag, or the server silently loses weight transfer and falls back to the
expert-repacking auto MoE backend:

```bash
docker run --gpus all --network=host --ipc=host \
  halo:vllm-0.32.0 Qwen/Qwen3-4B-Instruct-2507 --port 8000 \
  --weight-transfer-config '{"backend": "nccl"}' --moe-backend triton \
  --return-tokens-as-token-ids --logprobs-mode processed_logprobs \
  --enable-auto-tool-choice --tool-call-parser hermes
```

| Variable | Default | Purpose |
|---|---|---|
| `VLLM_IMAGE` | `halo:vllm-0.32.0` | The image compose runs; set it to the prebuilt `public.ecr.aws/whitecircle/halo:vllm-0.32.0` |
| `VLLM_MODEL` | `Qwen/Qwen3-0.6B` | Hub id or local checkpoint |
| `VLLM_PORT` | `8000` | Bound on the host; drives the serve command, the healthcheck, the container's `VLLM_SERVER_URL` and the Makefile's |
| `VLLM_CUDA_DEVICES` | `7` | Server GPUs, disjoint from `TRAINER_CUDA_DEVICES` (default `0,1,2,3,4,5,6`; a rank cannot broadcast to itself). Selected via `CUDA_VISIBLE_DEVICES` in a container that sees every GPU: hiding devices instead (`--gpus device=N`) breaks the cross-container P2P import and `/init_weight_transfer_engine` fails with `unhandled cuda error` |
| `VLLM_TP` / `VLLM_GPU_MEM` | `1` / `0.85` | `--tensor-parallel-size` / `--gpu-memory-utilization` |
| `VLLM_MOE_BACKEND` | `triton` | Keep `triton` for MoE RL |
| `VLLM_ENABLE_R3` | *(unset)* | Any value adds `--enable-return-routed-experts`; MoE models only (the engine exits on a dense one, and the `restart` policy loops it forever) |
| `VLLM_ATTENTION_BACKEND` | *(unset = auto)* | `--attention-backend`; GLM-4 MoE Lite (MLA) on Blackwell needs `CUTLASS_MLA` ([MLA backend](../reference/checkpoints.md#serving-on-vllm--sglang)) |
| `VLLM_SPECULATIVE_CONFIG` | *(unset)* | `--speculative-config` JSON for a checkpoint with an MTP head ([Throughput](#throughput)) |
| `VLLM_PREFIX_CACHING_FLAG` | `--enable-prefix-caching` | `--no-enable-prefix-caching` turns the cache off |
| `VLLM_ENFORCE_STRICT_TOOL_CALLING` | `0` | Env var the server reads, also the image's ENV; grammar-constrained tool calling stays off so the served distribution is the policy's |
| `VLLM_TUNED_CONFIG_FOLDER` | *(unset)* | Env var the server reads: tuned Triton MoE tile configs, visible inside the container |
| `VLLM_TOOL_PARSER` | `hermes` | `--tool-call-parser`; per-family values below |
| `VLLM_TOOL_CALLING_FLAGS` | `--enable-auto-tool-choice --tool-call-parser $VLLM_TOOL_PARSER` | Set empty (`VLLM_TOOL_CALLING_FLAGS=`) to serve with no tool parser, as the ReAct recipes do |
| `VLLM_TOOL_PARSER_PLUGIN` / `VLLM_REASONING_PARSER_PLUGIN` | *(unset)* | Parser plugin paths (gpt-oss: the baked `/opt/gpt_oss_text_tool_parser.py`, `/opt/gpt_oss_reasoning_parser.py`) |
| `VLLM_REASONING_PARSER` | *(unset)* | Required whenever a turn can carry a thinking cap: without one vLLM answers every request carrying `thinking_token_budget` with a 400 |
| `VLLM_CHAT_TEMPLATE` | *(unset)* | The same `.jinja` the trainer's `chat_template:` uses, visible inside the container, where compose mounts the repo's `jinja-templates/` at `/jinja-templates` |
| `VLLM_TRUST_REMOTE_CODE` | *(unset)* | Any value adds `--trust-remote-code`, which the Bailing (Ling 2.0) and Step-3.7 repos need |
| `VLLM_EXTRA_ARGS` | *(unset)* | Further flags, appended last, such as `--served-model-name` when the server's path differs from the trainer's model id |
| `TRAIN_IMAGE` | `halo:blackwell` | The image of the compose file's optional trainer service |

The flag-valued variables are `${VAR:+--flag ${VAR}}` slots, so an unset one omits its flag; vLLM never
reads them as env vars. Load-bearing flags the compose file sets:

- `--weight-transfer-config '{"backend": "nccl"}'` enables the transfer engine.
- `--return-tokens-as-token-ids`: `train_on_sampled_tokens` (default on) recovers the sampled ids from
  the logprobs, which vLLM spells out only under this flag. The [preflight](#startup-preflight) refuses a
  server without it, as every turn would fall back to re-tokenizing a chat-template re-render and train on
  tokens the engine never sampled; `train_on_sampled_tokens: false` takes that path on purpose.
- `--logprobs-mode processed_logprobs`: the reported logprobs are the sampling distribution's
  (temperature and top-p applied). The trainer scores at the sampling temperature and divides by them;
  the default `raw_logprobs` are pre-temperature, so every IS weight would be π^T / π^1 while
  `sampling/is_ratio_mean` still reads ≈ 1. The trainer probes each server at startup (temperature 2
  must halve the top-1/top-2 gap) and refuses a raw server whenever the temperature ≠ 1. Under this
  mode a top-p < 1, a top-k or a min-p also renormalizes every uncertain position, which a per-sequence
  log-ratio reads as drift: leave all three off under the geometric band (`isr_geo_band_min/max`), OPSM
  (`isr_opsm_delta`) or an online `sequence_*` `vllm_importance_sampling_mode` (also probed and refused).
- `--enable-prefix-caching`: R3 refuses to start without it, and a multi-turn episode re-prefills its
  whole context every turn when it is off.

`--max-model-len` is left unset, so the server serves the model's native window; the trainer's startup
probe reads it off `/v1/models` and raises when `max_prompt_length` plus one turn's generation exceeds
it (the worst-case multi-turn budget only warns).

`VLLM_USE_V2_MODEL_RUNNER` is an env var the compose file passes through bare, so unset keeps the
image's `1`. `0` falls back to the V1 runner, where R3 refuses to start and the gpt-oss budget loses
its patches; vLLM reads it with `int()`, so `false` or an empty value (an empty key in `.env` counts)
kills the server with a bare `ValueError`.

**Tool parsers are per family.** For native-tool environments (`code_contests`, `swe`, `mcp`,
`qa_search`, open-book `exam_qa`) a missing or wrong parser leaves calls as text with no `tool_calls`, so every
episode ties. The [preflight](#startup-preflight) refuses a server with no parser. A wrong one shows only in the
replies, so the trainer reads the opening its chat template renders for a call and raises when two rounds hold it in
their text with no call parsed anywhere (`halo/trainers/grpo/environmental/tool_call_watch.py`); one parsed call stands
the watch down for the run, so a later malformed call never trips it, and a turn the engine cut is no evidence, as any
parser leaves a partial call as text. `tool_parser_watch_rounds` sets the round count, and `null` turns the watch off for
a model that writes the opening in prose. ReAct environments send no `tools` schema and parse actions from the text, so
their recipes serve without a parser.

| Family | `--tool-call-parser` |
|---|---|
| Qwen3, the 2507 releases included; most others | `hermes` (JSON inside `<tool_call>` tags) |
| Qwen3.5 / 3.6, Qwen3-Coder | `qwen3_xml` (hermes does not parse their XML `<function=…>` calls) |
| GPT-OSS | plugin `gpt_oss_text` via `VLLM_TOOL_PARSER_PLUGIN`; reasoning plugin `/opt/gpt_oss_reasoning_parser.py`, parser `openai_gptoss` ([GPT-OSS](../models/gpt-oss.md#serving-for-grpo-vllm)) |
| GLM-4.7-Flash | `glm47` |
| Gemma 4 | `gemma4` (hermes leaves its `<\|tool_call>call:…<tool_call\|>` calls as text); with a thinking cap also `VLLM_REASONING_PARSER=gemma4` |
| LFM-2 | `lfm2` (pythonic calls between `<\|tool_call_start\|>` and `<\|tool_call_end\|>`) |
| Laguna | `poolside_v1` (`<tool_call>` with `<arg_key>`/`<arg_value>` pairs) |
| Step-3.7 Flash | `step3p5` (`<tool_call>` around `<function=…>`/`<parameter=…>` XML) |

**Transport.** The compose file defaults both containers to the no-fabric recipe (`NCCL_IB_DISABLE=1`
and `NCCL_NET=Socket` on each, `NCCL_P2P_LEVEL=NVL` on the server), so the cross-container group takes
NVLink and sockets by declaration on both ends: two containers on different nets form the group and
hang at the first collective. A trainer that pins `NCCL_PROTO` needs the server pinned the same way
(both files pass it through). On an EFA host layer `docker-compose.vllm.efa.yml`
([Servers on other nodes](#servers-on-other-nodes-efa)). `VLLM_GROUP_HOST` (or an entry's
`group_host`) names the trainer address the server dials back to and touches nothing else; a same-host
stack needs none ([Group rendezvous](#group-rendezvous)).

## SGLang

`Dockerfile.sglang` builds `halo:sglang-0.5.22` from `lmsysorg/sglang:v0.5.22` to align NCCL with the
training images (upstream's wheel trails `uv.lock`'s pin), install the EFA userspace and apply the
loader patches an online update needs ([above](#which-families-each-engine-serves)). It also patches
the prefill result loop to step its log-prob offset over a request retracted while its prefill ran;
upstream leaves the offset behind, so each later request in the batch reads another request's
log-probs and the `isr_engine_reference` re-score is silently wrong. The build asserts the weight-sync
routes, request schemas, rendezvous convention and expert loader layouts the client relies on
(`docker/sglang/verify_image.py`) and drives the patched loop on a retracted request.

The server keeps SGLang's own transformers (5.19.0) and torch. Serving-only use can run upstream
(`SGLANG_IMAGE=lmsysorg/sglang:v0.5.22`); weight sync needs this image, which names itself on `/server_info`
([Startup preflight](#startup-preflight)). No newer release is validated.
DeepEP v2 (`--moe-a2a-backend deepep_v2`) forces `--moe-runner-backend deep_gemm` and leaves gpt-oss
off its model allowlist; no online update into that runner is validated.

```bash
make build-sglang    # or: export SGLANG_IMAGE=public.ecr.aws/whitecircle/halo:sglang-0.5.22
SGLANG_MODEL=Qwen/Qwen3-0.6B SGLANG_MODEL_DIR=/path/to/checkpoints docker compose -f docker-compose.sglang.yml up
```

| Variable | Default | Purpose |
|---|---|---|
| `SGLANG_MODEL` | *(required)* | Hub id or local checkpoint directory |
| `SGLANG_MODEL_DIR` | *(required)* | Host directory mounted read-only at the same path: the one holding a local `SGLANG_MODEL`, any existing directory for a hub id (`down` needs both too) |
| `SGLANG_IMAGE` | `halo:sglang-0.5.22` | The image compose runs |
| `SGLANG_PORT` | `30000` | Bound on the host |
| `SGLANG_TP` / `SGLANG_GPU_MEM` | `1` / `0.85` | `--tp-size` / `--mem-fraction-static` |
| `SGLANG_CUDA_DEVICES` | `7` | Server GPUs, disjoint from the trainer's |
| `SGLANG_TOOL_PARSER` | `auto` | `--tool-call-parser`; `auto` reads the parser off the chat template and **disables it** on a template it cannot detect ([below](#tool-parsers-on-sglang)); `qwen25` for the Qwen3 2507 releases, `hermes` for Ling-mini-2.0 |
| `SGLANG_MOE_RUNNER_BACKEND` | `triton` | Keep `triton` for MoE RL and R3 |
| `SGLANG_ENABLE_R3` | *(unset)* | Any value adds `--enable-return-routed-experts` |
| `SGLANG_LORA_MAX_RANK` | *(unset)* | At least the run's `lora_r`; adds `--enable-lora --max-lora-rank` and the three rows below for `rollout_adapter_sync` ([Adapter sync](#adapter-sync)) |
| `SGLANG_LORA_TARGET_MODULES` / `SGLANG_MAX_LOADED_LORAS` / `SGLANG_MAX_LORAS_PER_BATCH` | `q_proj k_proj v_proj o_proj` / `8` / `3` | `--lora-target-modules` (every adapted projection), `--max-loaded-loras` (`prefetch_depth` + 2 or more), `--max-loras-per-batch` (2 or more) |
| `SGLANG_REASONING_PARSER` | *(unset)* | `--reasoning-parser` (`gpt-oss` for harmony models) |
| `SGLANG_CHAT_TEMPLATE` | *(unset)* | The same `.jinja` the trainer uses, visible inside the container, where compose mounts the repo's `jinja-templates/` at `/jinja-templates` |
| `SGLANG_ATTENTION_BACKEND` | *(unset)* | `triton` for GLM-4 MoE Lite on Blackwell: its MLA head size has no kernel in the default backend (`Unsupported head dimensions`) |
| `SGLANG_TRUST_REMOTE_CODE` | *(unset)* | Any value adds `--trust-remote-code`, which the Bailing (Ling 2.0) repos need |
| `SGLANG_EXTRA_ARGS` | *(unset)* | Further flags, appended last. `--enable-deterministic-inference` for GLM-4 MoE Lite: under triton attention its greedy logits differ between a prefill and a prefix-cache hit, which the tier's zero-noise probe refuses |

`SGLANG_GROUP_HOST` belongs to the trainer (the `VLLM_GROUP_HOST` equivalent). Engine behavior under RL:

- **Sampled ids** arrive per request: SGLang's OpenAI `logprobs` reports tokens as text, so the trainer
  sets `return_meta_info` + `return_prompt_token_ids` and reads `meta_info.output_token_logprobs`.
- **Logprobs are post-temperature, pre-nucleus**: the logits are divided by the temperature before the
  reported log-softmax, and top-p, top-k and min-p renormalize only the sampling probabilities, so they
  are the IS reference at any temperature and no cut moves the geometric band.
  `SGLANG_RETURN_ORIGINAL_LOGPROB` switches to raw values, which the startup probe refuses.
- **A length cut-off is a `stop_reason`**, not a `finish_reason`; `get_finish_reason`
  (`halo/inference/response.py`) reads either.
- **R3**: the fused runners (`triton_kernel`, `flashinfer_trtllm`, `flashinfer_mxfp4`) bypass the
  capture hook. The payload (base64 raw int32, decoded by the model's own layer and top-k counts) sits
  in each choice's `meta_info` when sampled ids are captured, response-level under
  `sglext.routed_experts` otherwise; rows cover the full sequence. The capturer exits at start for a dense
  model and Gemma 4 (no `num_experts_per_tok`; Gemma 4 has no routing replay on either engine). Families with
  a passing R3 row: [Coverage](#coverage).
- **`--enable-torch-compile`** is refused under `routing_replay: rollout`, as capture under it returns corrupt rows.
  With no replay, on 0.5.22, a server under the flag does not start on Qwen3-30B-A3B and takes about 10 minutes to
  start on gpt-oss-20b.
- **`rollout_max_thinking_tokens` is refused** for every model (SGLang ignores unknown request fields).
  Steer with `reasoning_effort`, which reaches the chat template only; a per-level `thinking_tokens`
  reaches no request field and is warned once.
- **Plain data-parallel replicas (`--dp-size > 1`) take no update**: the engine asserts `dp_size == 1`
  on the update routes and the client refuses at group formation (replicas restart `tp_rank` at 0 and
  collide on `rank_offset + tp_rank`). Attention DP (`--attn-dp-size N`) is accepted; the client reads
  the layout off `/server_info`.
- **`isr_engine_reference`** re-scores through `/generate` with `logprob_start_len`.

### Tool parsers on SGLang

`--tool-call-parser auto` resolves where one of SGLang's template rules matches: gpt-oss (`gpt-oss`), Qwen3.5 and
Qwen3.6 (`qwen3_coder`), Gemma 4 (`gemma4`), GLM-4.7-Flash (`glm47`), LFM-2 (`lfm2`). On a
template no rule matches it logs one warning (`could not detect tool-call parser from chat template. Disabling
tool-call parser.`) and serves with no parser, which `/server_info` reports and the
[preflight](#startup-preflight) refuses on a native-tool environment. Qwen3-30B-A3B-Instruct-2507 and
Ling-mini-2.0 land there, since their templates carry no `enable_thinking` toggle for the `qwen` rule to read.

Serve the Qwen3 2507 releases with `SGLANG_TOOL_PARSER=qwen25`, which ran on Qwen3-30B-A3B-Instruct-2507 (0.5.22 logs it
as the old spelling of `qwen`), and Ling-mini-2.0 with `SGLANG_TOOL_PARSER=hermes`, the parser its rows ran on.

Sampling stays off any grammar. By upstream's code, `glm47`, the parser `auto` resolves on GLM-4.7, builds a
full-assistant grammar for every tool-bearing request: the sampler masks tokens under it, the reported log-probs carry
the mask, and the trainer scores the unmasked distribution. `docker/sglang/patches/patch_sglang.py` removes that flag
and the build asserts, on the real parser, that `tool_choice: auto` carries no constraint on any parser a family
resolves to, so a malformed call stays text the environment scores, as on vLLM. On GLM-4.7-Flash the patch makes no
visible change on the prompts that ran.

### NCCL transport (SGLang)

The group is ordinary NCCL: CUDA IPC between the containers on one host (`P2P/CUMEM`, NVLink), the
fabric across nodes. The one engine-side requirement is **cuMem parity**: SGLang sets
`NCCL_CUMEM_ENABLE=0` process-wide unless it is set, while the trainer's NCCL has cuMem on, and the
mismatch fails the first cross-container buffer import (`ncclP2pImportShareableBuffer ... invalid
argument`, then `The full weights of the ModelRunner are partially updated`).
`docker-compose.sglang.yml` sets `NCCL_CUMEM_ENABLE=1`; keep it on a hand-run server.

Both base compose files pass `NCCL_SOCKET_IFNAME=^docker,veth` (the vLLM file to its training service
too): on a host running other containers NCCL otherwise enumerates their bridge and veth interfaces,
which carry no host-to-host traffic (the first collective after the sync hangs) or vanish with their
container (`Call to bind failed: No such device`, then a `400` on the update). Override it to pin one
NIC on a multi-homed host, one every collective can use: the variable is process-wide and filters what
the default process group and DeepEP pick too. It is not a group-address knob.

## Servers on other nodes (EFA)

The weight-sync group rides the fabric when both containers can drive it. All three images install one
EFA userspace whole (`docker/efa/install_efa_userspace.sh`: rdma-core, AWS libfabric and one
`aws-ofi-nccl` build; inventory on [Docker](docker.md#rdma-networking-infiniband-and-efa)). The stack
is wire-sensitive down to rdma-core: at init the plugin probes libfabric for in-order RDMA writes and
forces `NCCL_PROTO=simple` when the probe fails, and the answer comes from rdma-core's EFA provider
(the NGC base's MOFED `libefa` fails it, the installer's passes). Two containers that answer
differently form the group with different protocol tables and hang at the first collective, the same
signature as a plugin-version mismatch. So both ends run images built from the same script, or a pair
that cannot be rebuilt runs `NCCL_PROTO=simple` on both. The upstream engine bases ship no EFA
userspace, and NCCL then falls back to sockets with no message.

Server side, layer the EFA overlay after the base compose file:

```bash
docker compose -f docker-compose.vllm.yml -f docker-compose.vllm.efa.yml up -d vllm-server
SGLANG_MODEL=... SGLANG_MODEL_DIR=... docker compose -f docker-compose.sglang.yml -f docker-compose.sglang.efa.yml up -d
```

Each overlay adds `devices: /dev/infiniband` (a `-v` bind mount does not grant access),
`ulimits: memlock: -1` and `NCCL_NET=Libfabric NCCL_NET_PLUGIN=ofi NCCL_IB_DISABLE=0
NCCL_SOCKET_IFNAME=^lo,docker,veth,tailscale`. Naming the plugin's net makes a missing or mismatched
plugin fail group formation instead of falling back to sockets. `lo` is excluded because an
excluded-only list still ranks loopback first, and a server advertising `127.0.0.1` fails formation
with a trainer on another node (`remote process exited or there was a network error`).

Trainer side, the same env: `make ... EFA=1` adds `--device=/dev/infiniband` and those variables to
every `DOCKER_RUN` (and drops the socket forcing from `test-gpu-vllm` / `test-gpu-sglang`); a
hand-written `docker run` passes them itself. `NCCL_NET` / `NCCL_NET_PLUGIN` are process-global, shared
by the sync group and every trainer collective, so a trainer left at `Socket` on an EFA host sends
everything over TCP and breaks DeepEP; this is the env
[multi-node training](../parallelism/multi-node.md#rdma-fabrics) already uses. GPUDirect RDMA runs
through dmabuf with no `nvidia_peermem`; `/dev/gdrdrv` is a cross-node DeepEP requirement, not a sync
one. Set the entry's `group_host` (or `VLLM_GROUP_HOST` / `SGLANG_GROUP_HOST`) when the server cannot
dial back to the trainer's address on the route to it (asymmetric routing); Online GRPO has only the env
var. An address the trainer cannot bind, as behind NAT or a port mapping, raises unless
`HALO_WEIGHT_SYNC_BIND_ALL=1` is set ([Group rendezvous](#group-rendezvous)).

Verify the transport before training, from the trainer container launched as the trainer would be,
on a GPU the server does not own:

```bash
python scripts/profiling/weight_sync_transport.py --server_url http://<server>:8000 --backend vllm --expect efa
```

It pushes one real parameter of the served checkpoint, unchanged, through the trainer's own sync path
(`--model_id` when the served id is a server-side path or an alias), and reports the transport NCCL
formed on (`NET/Libfabric/…/GDRDMA` is EFA with GPUDirect; `NET/IB`, `NET/Socket`, `P2P/CUMEM`, `SHM`),
whether the plugin forced the simple protocol, the `aws-ofi-nccl` build, the libfabric provider and
GB/s. It exits 1 when the group fails to form or the served model changed, and `--expect` adds the
transport to that gate ([flags](../reference/scripts-reference.md#profiling--benchmarks)).

On 8× B300 nodes, trainer node → server node, the streamed path syncs at 53 GB/s through the vLLM
client over EFA (measured on vLLM 0.26.0), 80 GB/s through the SGLang client (measured on SGLang
0.5.17) and 9.7 GB/s over sockets (NIC line rate ~100 GB/s per GPU). The vLLM client's rate is set by
its one HTTP round trip per chunk and rises with `HALO_WEIGHT_SYNC_CHUNK_MB`
(73 GB/s at 8 GiB, measured on vLLM 0.26.0); the SGLang client's by the fabric.

## Checking a server

```bash
curl -s localhost:8000/health          # vLLM (30000 for SGLang)
curl -s localhost:8000/v1/models
curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen3-0.6B","messages":[{"role":"user","content":"Say OK"}],"max_tokens":16}'
```

The eval runners under `scripts/environments/inference/` take the server URL as the OpenAI base and
`--training_config <yaml>` to sample under the training contract. `make test-gpu-vllm` and
`make test-gpu-sglang` run the GPU tiers against a live server, a behaviorally verified weight sync
included (greedy output must change after a broadcast); the server must own a GPU outside
`TRAINER_CUDA_DEVICES` ([Contributing → Dev loop](../contributing/README.md#dev-loop)).

## Throughput

**Decode is CPU-bound at RL concurrency.** The engine core is one Python thread and saturates a core
from a few dozen concurrent sequences; step latency grows with running sequences, so per-sequence
speed falls as concurrency rises while aggregate throughput rises sub-linearly. Fewer steps per token
help; kernel-level gains (tuned MoE tiles, another attention backend) do not show. A trainer sharing
the host slows every step, so pin each server container to its own cores (`docker run --cpuset-cpus`,
compose `cpuset:`). Batch sizes and timeouts:
[Sizing a run](../training-methods/async-grpo/performance.md#sizing-a-run).

**Servers × TP.** Run one engine per GPU, grouped into the fewest servers that hold the model: a second
engine on one GPU only time-slices it (and two starting at once fail their memory profiling), and
fewer servers shrink the sync fan-out. Per-server request load is
`max_concurrent_rollouts × rollout ranks / num_servers` per round in flight (`prefetch_depth` of them under
prefetch): every rollout rank (a TP/ETP group's leader) builds its own actor pool
([Ray → Pool sizing](ray.md#pool-sizing)), and dispatch is round-robin regardless of load, each rank starting at
the server its data-parallel rank names, so a round narrower than the server count still spreads.

**Prefix caching** stays on: a multi-turn episode re-sends its whole context every turn, and the
`mode=keep` pause that brackets every update clears the cache, so no cached prefix serves stale
weights. Turn it off with speculative decoding (`VLLM_PREFIX_CACHING_FLAG=--no-enable-prefix-caching`;
R3 then refuses to start): speculation can poison the align-mode prefix cache and the engine has no
guard for the pair.

**Speculative decoding (MTP).** A checkpoint with a multi-token-prediction head (Qwen3.5/3.6 or Nemotron-H, `mtp.*`
tensors) drafts with it through `VLLM_SPECULATIVE_CONFIG='{"method":"mtp","num_speculative_tokens":1}'`.
Rejection sampling keeps the target distribution, up to the bf16 rounding a draft's batch shape changes
([Serving](../optimization/mtp-heads.md#serving), SGLang's flags too), `processed_logprobs` come from the
target logits, the thinking budget is enforced on the target, and the drafter's embeddings and
`lm_head` are the target's modules, so the sync updates them (the MTP layer keeps launch weights, which
only moves acceptance), except Step-3.5/3.7, which keep one head per draft layer, and Nemotron-H, whose family the sync refuses. The server base must store
the head, and exports of a run that did not [train it](../optimization/mtp-heads.md) do not, so vLLM
refuses them at startup; `patch_vocab` and `convert_to_bf16` write a stored Qwen3.5/3.6, Qwen3-Next, GLM-4 Lite or Nemotron-H head back, and drop any other family's. SGLang has no such check on Qwen3.5/3.6, so its export must carry the tail. Under speculation a request with `min_p > 0` or a `logit_bias` raises; never
use `ngram` on linear-attention families (an unguarded output-corruption bug); `suffix` fails at
startup on Model Runner V2. The prompt-log-prob probe passes under MTP on Model Runner V2 (Qwen3.6-35B-A3B,
one speculative token).

**Strict tool calling off** (`VLLM_ENFORCE_STRICT_TOOL_CALLING=0`, the image's ENV, so a hand-run server has it too):
at the default strict level the GLM parsers (`glm45`, `glm47`) take a structural-tag grammar at `tool_choice: auto`,
which would mask tokens the trainer's log-probabilities never see masked and add CPU work per step; a
`--tool-strict-level` above `auto` or a strict tool would add other parsers. With it off a malformed call becomes
text the environment scores.

**Triton MoE tile configs.** `--moe-backend triton` reads a per-shape tuned config
(`E=<experts>,N=<intermediate>,device_name=<GPU>.json`) from `VLLM_TUNED_CONFIG_FOLDER`, then vLLM's
`fused_moe/configs`, and logs `Using default MoE config` when neither matches. They matter for
prefill-heavy phases, not CPU-bound decode. `benchmarks/kernels/benchmark_moe.py --tune` in the vLLM
image writes one (needs `pip install ray`; pass `--tp-size 1` and `--batch-size` with the decode batch
and the prefill chunk).

**Generation volume is the step-time lever** once prefetch overlaps collection with training
(`async/prefetch_hit_rate` > 0.8): step time tracks mean episode tokens. `rollout_max_tokens` caps a
turn; on vLLM the turn's reasoning cap caps CoT engine-side, while a soft budget, and every budget on
SGLang, is held only by the reasoning terms.

**fp32 heads.** Both engines serve bf16 logits by default, while the trainer's chunked scoring is fp32. Each has a switch: `--enable-fp32-lm-head` on SGLang and `--hf-overrides '{"head_dtype":"float32"}'` on vLLM, both through `*_EXTRA_ARGS` under compose, with vLLM's single quotes kept in the value. The served weight stays bf16, so the sync is unaffected. Use them only with `use_chunked_grpo_logprobs: true`, as the full path scores bf16 and would then mismatch the other way ([size of the gap](../training-methods/async-grpo/performance.md#chunked-log-probs)).

**Memory.** Raise `--gpu-memory-utilization` / `--mem-fraction-static` to 0.9 on dedicated server GPUs,
but serve at ≤ 0.80, or with a smaller `--max-num-batched-tokens`, under `isr_engine_reference`, whose
`prompt_logprobs` re-scores allocate outside the engine's memory profile. Do not pass
`--enforce-eager`: the in-place sync keeps captured CUDA graphs valid, and graph decode is several-fold
faster on long generations. On B200 pin `VLLM_ATTENTION_BACKEND=FLASH_ATTN` only where FlashInfer
JIT-fails on SM 10.0.

**Sync cadence.** `sync_weights_every_n_steps: 2–4` for slow environments
([Async GRPO](../training-methods/async-grpo/setup.md#weight-synchronization)).

## Coverage

What passed against a live server on this release, on one node with each server at TP=1 unless a row says otherwise.
A dash is a shape the code admits with no row on that engine.

**Weight push** (`test_weight_push_exact.py`, 40 rows). A push of the unchanged checkpoint leaves every served
log-prob bit-identical, on 2,010 positions (987 on Step-3.7's tiny checkpoint): dense Qwen3-0.6B plain and at TP=2;
gpt-oss-20b, every checkpoint of the second table below and Step-3.7 at `ep1` and EP=2, on each engine that takes the
family. The Step-3.7 release passes at EP=2 against a vLLM server at TP=2 (2,010 positions).

**Async GRPO, full fine-tuning**: vLLM on Qwen3-30B-A3B at 2 ranks and gpt-oss-20b at 4, SGLang on gpt-oss-20b at 2 and
4, dense rows on Qwen3-0.6B.

| Trainer shape | vLLM | SGLang |
|---|---|---|
| Plain FSDP2, dense | 1 and 2 ranks | - (a weight push and the re-init row ran, with no training loop) |
| `ep1`; `ep1` ZeRO-3 | ran | ran |
| `ep1` with `fsdp_shard_ep1_experts: false` | ran | - |
| EP=2, EP=4 | ran | ran |
| EP=2 as two groups, on one domain and on two simulated domains | ran | - |
| TP=2 | ran, also TP=2 × DP=2 | ran, gpt-oss included (its hand-sliced `sinks` are sent once from the gathered drain) |
| EP=2 + TP=2, EP=2 + ETP=2 | ran | ran |
| HSDP on two simulated domains | ran | - |
| Attention LoRA | `ep1`, EP=2, pure ETP=2, EP=2 + ETP=2 | `ep1`, EP=2 + ETP=2 |
| Native expert LoRA | EP=2, EP=4, EP=2 resume | EP=2, EP=4, EP=2 resume |
| Resume | EP=2, `ep1` LoRA | `ep1`, `ep1` LoRA |
| Prefetch | EP=2 at depth 1 and 2 | `ep1` at depth 1 |
| Two servers | EP=2 at prefetch depth 1; one rank at depth 1 and 2 | 2 ranks, no prefetch |
| R3 | `ep1`, EP=2, chunked, prefetch depth 1 | `ep1`, `ep1` LoRA, TP=2, `ep1` resume, chunked, prefetch depth 1 |
| R2 | EP=2 | EP=2 |
| A server at TP=2 | Qwen3-30B-A3B at `ep1` | Qwen3-30B-A3B at `ep1` |
| Trainer and server on two machines ([overlays](#servers-on-other-nodes-efa)) | Qwen3-30B-A3B at `ep1` | Qwen3-30B-A3B at `ep1` |

Other checkpoints, async GRPO at 2 ranks:

| Checkpoint | vLLM | SGLang |
|---|---|---|
| Qwen3-30B-A3B | R2 and R3 at EP=2 | `ep1`, EP=2, each also with R3 |
| Qwen3.5-35B-A3B | R2 and R3 at EP=2 | R3 at EP=2 |
| Qwen3.6-35B-A3B | EP=2, TP=2, `ep1` attention LoRA, R2 and R3 at EP=2 | `ep1`, EP=2, `ep1` attention LoRA, R3 at EP=2 |
| GLM-4.7-Flash | `ep1`, EP=2, TP=2, each also with R3; R2 at EP=2 | `ep1`, EP=2, each also with R3 |
| LFM2-24B-A2B | `ep1`, EP=2, TP=2, each also with R3; R2 at EP=2 | `ep1`, EP=2, each also with R3 |
| Gemma 4 26B-A4B | `ep1`, EP=2, `ep1` attention LoRA | `ep1`, EP=2, `ep1` attention LoRA |
| Laguna-XS-2.1 | `ep1`, EP=2, R2 and R3 at EP=2 | refused |
| Ling-mini-2.0 | `ep1`, EP=2, R2 and R3 at EP=2 | R2 and R3 at EP=2 |

Online GRPO and SDPG ran on Qwen3-30B-A3B (`ep1`, EP=2, EP=2 × DP=2, EP=2 on two simulated domains, EP=2 + TP=2, EP=2 +
ETP=2, attention and expert LoRA) and Qwen3-0.6B (FSDP2, one rank, TP=2, TP=2 × DP=2, ZeRO-3, HSDP on two simulated
domains, LoRA); online GRPO also on gpt-oss-20b, Qwen3.6-35B-A3B and Gemma 4 26B-A4B at EP=2. Step-3.7 ran as a weight
push on a tiny hub-layout checkpoint and on the release, with no training loop, and on the release as one step of expert
LoRA at EP=4. The R2 and R3 rows show capture and replay work; what they
change: [Trainer and engine noise](../training-methods/async-grpo/objective.md#trainer-and-engine-noise).

An SGLang server under its own expert parallelism (`SGLANG_TP=2 SGLANG_EXTRA_ARGS="--ep-size 2"`) takes
the sync for Qwen3 MoE. Measured on SGLang 0.5.17 (not re-run on 0.5.22), gpt-oss fails at start under
`--ep-size`: its bf16 fused expert loader narrows only the TP dimension and copies the global expert count
into local storage (`_load_w2`).

## Troubleshooting

Server-side symptoms; the trainer-side weight-sync errors (group address, port bind, transport
mismatch, SGLang cuMem) are rows of [Troubleshooting](../reference/troubleshooting.md#symptom--cause--fix).

| Symptom | Cause → fix |
|---|---|
| Every completion hangs after a trainer died mid-sync, with vLLM's `/health` still 200 and SGLang's hanging too | SIGKILL (`docker rm -f`) skipped the client's `atexit` resume, so the engine stays paused. The next trainer refuses it at startup (`is PAUSED`, or `paused or wedged` after a 60 s probe on SGLang, which has no pause route) → POST `/resume` (vLLM) or `/continue_generation` (SGLang), or restart it if that sync had already streamed part of the model. vLLM's update session unloads the model when it opens, and `/resume` then takes the engine down: restart it |
| A server dies mid-run and the trainer goes on for minutes | Rollouts and the sync run out their timeouts first: a killed engine (`docker kill`, one server) raises after about 4 minutes on either engine, naming the server, and a frozen one (`docker pause`, SGLang at 2 ranks) after about 20 → let it raise or stop the run, then restart the server |
| Group formation times out, "1/2 clients joined" | A bridge network hides the rendezvous port → `network_mode: host` on both containers |
| First sync takes minutes, `/health` unanswered | One-time NCCL group init per server; the engine pauses mid-update → wait |
| 400 on every rollout (vLLM) | A thinking budget without a reasoning parser → set `VLLM_REASONING_PARSER`; otherwise rollouts return zero tokens and the run trains on all-masked batches |
| `… hold '<marker>' … and the server parsed no call` at the second round | Wrong tool parser, or vLLM `--tool-call-parser` without `--enable-auto-tool-choice` ([parsers](#running-it)) |
| Log-ratio drifts on SGLang while the server log stays clean | The MoE loaders skip unmapped expert names without a warning → a clean log proves nothing; the construction gates are the guard |
| `RoutedExperts: Failed` (vLLM log) | Layerwise-reload patch missing → expert syncs silently reverted; rebuild (`make build-vllm`) |
| `/init_weight_transfer_engine` answers 500 (`NCCL error: unhandled cuda error`) while `/health` is 200 | Re-init patch missing → a stranded communicator per connection; rebuild and recreate the container |
