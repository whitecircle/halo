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
(`net.ipv4.ip_local_reserved_ports`) or pick ports outside it. Entry keys are read by name without
validation, so a misspelled `group_port` silently falls back to `rollout_group_port + index`. The
trainer-side mechanics (prefetch and its staleness, sync cadence, trajectory length) live on
[Async GRPO → Weight synchronization](../training-methods/async-grpo/setup.md#weight-synchronization).

| | vLLM 0.32.0 | SGLang 0.5.22 |
|---|---|---|
| Trainers | online, async GRPO | async GRPO |
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
delta path (~42 GB at 20B). Re-assembling the sharded weights is a collective every rank joins.

One rank owns the clients and sends: the **forwarding rank** (global main; TP rank 0 under TP). A
failure there (an engine 500, a refused tensor, a host OOM on the snapshot) is recorded rather than
raised, so the rank stays in every remaining collective and the whole world raises together at the
end of the sync, naming the failing rank and its cause.

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
buffered 1 GiB buffers on vLLM, typed 1 GiB chunks on SGLang. A chunk is cut before
`HALO_WEIGHT_SYNC_CHUNK_MB` (1 GiB) is exceeded, and a larger tensor rides alone. A chunk staged in
pinned host memory instead crosses PCIe twice before the NIC sees it: 19–24 GB/s against 53–80 GB/s
staged on the device (8× B300 nodes over EFA; the vLLM rate measured on vLLM 0.26.0, the SGLang rate
measured on SGLang 0.5.17).

The forwarding rank's peak during a sync is the assembled EP layer being sent (~28 GB for one 397B
layer), the staged chunk, the snapshot of the largest tensor and the engine path's buffers: SGLang's
arena (grown to the largest chunk) or vLLM's two fixed 1 GiB packed buffers.

With several servers (`rollout_server_configs`) one snapshot per parameter is shared, each chunk goes
to every server on concurrent threads, and it is released once all have it. The threads share the
forwarding rank's GPU, NICs and process, so the fan-out costs the sum, not the max: two servers each
push at half of one server's rate.

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
syncs) is recovered by a reconnect and re-flush; one that fails **after** its first chunk holds neither
the old policy nor the new, and vLLM's layerwise reload materializes any layer straddling the boundary
from *uninitialized* storage. The abort therefore leaves that engine **paused**, refuses every later
sync on that client and logs `RESTART the … server`: restart the container.

Each chunk runs under one 600 s deadline on both engines (`HALO_NCCL_SYNC_TIMEOUT_SECONDS`): the
update request that declares it and the drain of its sends. A failed chunk drains the stream under a
30 s deadline, and on expiry the group is aborted instead of parking the trainer.

Both clients send from their own CUDA streams and allocate their buffers on the caller's stream,
recorded for the send stream, so a reconnect strands no allocator pool; `HALO_WEIGHT_SYNC_MEM_LOG=1`
brackets each sync with a per-rank `[mem rankNN] weight-sync pre/post` line, and `reserved` far above
`peak_alloc` on the forwarding rank means stranded pools
([Debugging](../reference/debugging.md#3-gpu-memory-profiling)).

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

The image applies five runtime patches through `sitecustomize` in the API server and every engine-core
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

### Construction gates

The trainer refuses at construction a model the selected engine cannot take a sync for
(`reject_unsyncable_model`, `halo/trainers/grpo/rollout/weight_push.py`), on every rank alike:

- **The family.** An engine fact on that engine's client (`UNSERVABLE_MODEL_TYPES`,
  [below](#which-families-each-engine-serves)), or a `weight_sync_refusal` on the family spec, which
  Inkling (no engine run has validated its sync), GLM-5 Next (both engines' per-expert loaders refuse
  the gathered fused pair), Qwen3-Next and Cohere2 MoE carry on both engines.
- **A MoE without a live EP layer** (`use_grouped_gemm: false` at `ep_size: 1`), for every family: the
  experts ship in the layout the EP layer's gather emits, and the stock module tree's fused tensors
  under module names are dropped by the engine's loader with no error. `use_grouped_gemm: true` (the
  torchrun default) installs the EP layers.
- **A bnb-quantized (QLoRA) base**, and PEFT LoRA on a weight the dense push does not send (EP expert
  weights take native expert LoRA instead).
- **GptOss without live sinks.** `reset_sinks: true` under `flash_attention_2` removes the sinks, and
  the sync forwards `named_parameters()` only, so the server keeps the pretrained sinks against a
  sink-free trainer, permanently off-policy. On-policy GptOss needs `reset_sinks: false` with FA4 or
  eager, as the shipped GRPO configs set; `train_sinks: true` (SFT-only) is refused too.
- **Enabled bias-update balancing**, adopted or transient: the parameter-only payload never pushes the
  bias. The shipped GRPO scripts downgrade `moe_balancing` to `none` themselves.

**Hub names.** The sync forwards every tensor under the key a gathered checkpoint would carry: a
family's `export_key_renames`, then transformers' own save-side revert of what the load converted
(`export_conversions` in `halo/checkpoint/hub_layout.py`, the list the gathered save inverts), so
Step-3.7's per-layer `moe.*` stacks and a SigLIP tower's `vision_model` level reach the engine under
their hub keys. One-to-one renames stream tensor by tensor; a tensor a reverse converter claims (a
fused `gate_up_proj` the hub stores split, a vision tower's fused q/k/v) is held until its sources are
complete, since the engine loads one tensor at a time.

### Which families each engine serves

Both engines read a family's experts in the layout its `ExpertLayout` writes: per-expert tensors for
Qwen3 MoE, Qwen3-Next, GLM-4 MoE Lite, Laguna, Bailing and LFM-2, the fused pair for Qwen3.5/3.6 and Gemma 4,
GptOss's interleaved pair. What differs is which families each pinned release can take an online
update for; each client declares those with the loader fact (`UNSERVABLE_MODEL_TYPES`), which the
[construction gate](#construction-gates) quotes.

| Family (`model_type`) | vLLM 0.32.0 | SGLang 0.5.22 | Loader fact |
|---|:--:|:--:|---|
| Mistral4 | ✗ | ✗ | neither registers a class ([Mistral4](../models/mistral4.md#serving)) |
| Ling 3.0 (`bailing_hybrid`) | ✗ | ✗ | both register `BailingMoeV3ForCausalLM`; no online update into it is validated |
| Ring (`bailing_moe_linear`) | ✗ | ✗ | the checkpoints declare `BailingMoeLinearV2ForCausalLM`; both register `BailingMoeV2_5ForCausalLM` |
| Zaya | ✗ | ✗ | vLLM ships no class; SGLang's loader reads the pre-transformers-5.14 per-expert checkpoint (`zaya_block.experts.local_experts.N.linear_fc1`) |
| DeepSeek-V4 | ✗ | ✗ | both map per-expert `w1/w2/w3` where the gather emits the fused pair |
| Laguna | ✓ | ✗ | SGLang's `load_weights` asserts every routed-expert tensor of every sparse layer per call; vLLM's loader drops the exported balancing bias ([Callbacks](../training-methods/callbacks.md#moe-balancing-modes)) |
| Step-3.7 (`step3p7`, `step3p5`) | ✓ | ✗ | `Step3p5ForCausalLM.load_weights` asserts full parameter coverage per call |

Every family not in the table syncs on both engines, expert distribution included: dense families,
GptOss, Qwen3 MoE, Qwen3.5/3.6 MoE, GLM-4 MoE Lite, Gemma 4, Ling 2.0, LFM-2 MoE. Inkling, GLM-5 Next,
Qwen3-Next and Cohere2 MoE are refused on both by their family spec. Per-family server flags:
[Step-3.7](../models/step3p7.md#serving-for-grpo-vllm).

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
`tests/common/models.py`, plus a control under `unparseable/` that must be refused (a control the
server reads fails the build, since its rewrite is then redundant).
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
| `VLLM_ENFORCE_STRICT_TOOL_CALLING` | `0` | Env var the server reads; grammar-constrained tool calling stays off so the served distribution is the policy's |
| `VLLM_TUNED_CONFIG_FOLDER` | *(unset)* | Env var the server reads: tuned Triton MoE tile configs, visible inside the container |
| `VLLM_TOOL_PARSER` | `hermes` | `--tool-call-parser`; per-family values below |
| `VLLM_TOOL_CALLING_FLAGS` | `--enable-auto-tool-choice --tool-call-parser $VLLM_TOOL_PARSER` | Set empty (`VLLM_TOOL_CALLING_FLAGS=`) to serve with no tool parser, as the ReAct recipes do |
| `VLLM_TOOL_PARSER_PLUGIN` / `VLLM_REASONING_PARSER_PLUGIN` | *(unset)* | Parser plugin paths (gpt-oss: the baked `/opt/gpt_oss_text_tool_parser.py`, `/opt/gpt_oss_reasoning_parser.py`) |
| `VLLM_REASONING_PARSER` | *(unset)* | Required whenever a turn can carry a thinking cap: without one vLLM answers every request carrying `thinking_token_budget` with a 400 |
| `VLLM_CHAT_TEMPLATE` | *(unset)* | The same `.jinja` the trainer's `chat_template:` uses, visible inside the container |
| `TRAIN_IMAGE` | `halo:blackwell` | The image of the compose file's optional trainer service |

The flag-valued variables are `${VAR:+--flag ${VAR}}` slots, so an unset one omits its flag; vLLM never
reads them as env vars. Load-bearing flags the compose file sets:

- `--weight-transfer-config '{"backend": "nccl"}'` enables the transfer engine.
- `--return-tokens-as-token-ids`: `train_on_sampled_tokens` (default on) recovers the sampled ids from
  the logprobs, which vLLM spells out only under this flag. Without it every turn falls back to
  re-tokenizing a chat-template re-render: one warning, then a run training on tokens the engine never
  sampled.
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
`qa_search`, open-book `exam_qa`) a missing or wrong parser is silent and fatal: calls stay text, no
`tool_calls`, every episode reward 0, a flat zero gradient. ReAct environments send no `tools` schema
and parse actions from the text, so their recipes serve without a parser.

| Family | `--tool-call-parser` |
|---|---|
| Qwen3, the 2507 releases included; most others | `hermes` (JSON inside `<tool_call>` tags) |
| Qwen3.5 / 3.6, Qwen3-Coder | `qwen3_xml` (hermes does not parse their XML `<function=…>` calls) |
| GPT-OSS | plugin `gpt_oss_text` via `VLLM_TOOL_PARSER_PLUGIN`; reasoning plugin `/opt/gpt_oss_reasoning_parser.py`, parser `openai_gptoss` ([GPT-OSS](../models/gpt-oss.md#serving-for-grpo-vllm)) |
| GLM-4 | `glm45` / `glm47` |
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
(`SGLANG_IMAGE=lmsysorg/sglang:v0.5.22`); weight sync needs this image. No newer release is validated.
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
| `SGLANG_TOOL_PARSER` | `auto` | `--tool-call-parser`; `auto` reads the parser off the chat template (gpt-oss resolves to the harmony parser) |
| `SGLANG_MOE_RUNNER_BACKEND` | `triton` | Keep `triton` for MoE RL and R3 |
| `SGLANG_ENABLE_R3` | *(unset)* | Any value adds `--enable-return-routed-experts` |
| `SGLANG_REASONING_PARSER` | *(unset)* | `--reasoning-parser` (`gpt-oss` for harmony models) |
| `SGLANG_CHAT_TEMPLATE` | *(unset)* | The same `.jinja` the trainer uses, visible inside the container |
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
  `sglext.routed_experts` otherwise; rows cover the full sequence. The capturer serves GptOss, Qwen3
  MoE, Qwen3.5/3.6 and GLM-4 MoE Lite, exits at start for a dense model and Gemma 4 (no
  `num_experts_per_tok`; Gemma 4 has no routing replay on either engine) and raises at the first capture
  for Bailing. Keep `--enable-torch-compile` off under R3: capture × compile produces isolated
  catastrophic log-ratio rows.
- **`rollout_max_thinking_tokens` is refused** for every model (SGLang ignores unknown request fields).
  Steer with `reasoning_effort`, which reaches the chat template only; a per-level `thinking_tokens`
  reaches no request field and is warned once.
- **Plain data-parallel replicas (`--dp-size > 1`) take no update**: the engine asserts `dp_size == 1`
  on the update routes and the client refuses at group formation (replicas restart `tp_rank` at 0 and
  collide on `rank_offset + tp_rank`). Attention DP (`--attn-dp-size N`) is accepted; the client reads
  the layout off `/server_info`.
- **`isr_engine_reference`** re-scores through `/generate` with `logprob_start_len`.

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
dial back to the trainer's address on the route to it (NAT, asymmetric routing); Online GRPO has only
the env var.

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
fewer servers shrink the sync fan-out. Where rollout collection dominates (a tool-heavy environment
such as `code_contests`), 2×(TP=2) and 4×(TP=1) give the same step time; re-measure short single-turn
environments before consolidating. Per-server request load is
`max_concurrent_rollouts × world_size / num_servers` per round in flight (`prefetch_depth` of them under
prefetch): every rank builds its own actor pool, under TP
too ([Ray → Pool sizing](ray.md#pool-sizing)), and dispatch is round-robin regardless of load.

**Prefix caching** stays on: a multi-turn episode re-sends its whole context every turn, and the
`mode=keep` pause that brackets every update clears the cache, so no cached prefix serves stale
weights. Turn it off with speculative decoding (`VLLM_PREFIX_CACHING_FLAG=--no-enable-prefix-caching`;
R3 then refuses to start): speculation can poison the align-mode prefix cache and the engine has no
guard for the pair.

**Speculative decoding (MTP).** A checkpoint with a multi-token-prediction head (Qwen3.5/3.6, `mtp.*`
tensors) drafts with it through `VLLM_SPECULATIVE_CONFIG='{"method":"mtp","num_speculative_tokens":2}'`.
It is sound for RL: rejection sampling keeps the target distribution, `processed_logprobs` come from the
target logits, the thinking budget is enforced on the target, and the drafter's embeddings and
`lm_head` are the target's modules, so the sync updates them (the MTP layer keeps launch weights, which
only moves acceptance). Under speculation a request with `min_p > 0` or a `logit_bias` raises; never
use `ngram` on linear-attention families (an unguarded output-corruption bug); `suffix` fails at
startup on Model Runner V2. The prompt-log-prob probe passes under MTP on Model Runner V2 (Qwen3.6-35B-A3B,
one speculative token).

**Strict tool calling off** (`VLLM_ENFORCE_STRICT_TOOL_CALLING=0`): the structural-tag grammar would
mask tokens the trainer's log-probabilities never see masked and add CPU work per step; a malformed
call becomes text the environment scores.

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

**Memory.** Raise `--gpu-memory-utilization` / `--mem-fraction-static` to 0.9 on dedicated server GPUs,
but serve at ≤ 0.80, or with a smaller `--max-num-batched-tokens`, under `isr_engine_reference`, whose
`prompt_logprobs` re-scores allocate outside the engine's memory profile. Do not pass
`--enforce-eager`: the in-place sync keeps captured CUDA graphs valid, and graph decode is several-fold
faster on long generations. On B200 pin `VLLM_ATTENTION_BACKEND=FLASH_ATTN` only where FlashInfer
JIT-fails on SM 10.0.

**Sync cadence.** `sync_weights_every_n_steps: 2–4` for slow environments
([Async GRPO](../training-methods/async-grpo/setup.md#weight-synchronization)).

## Coverage

Async GRPO against a live server, per trainer parallelism axis:

| Trainer axis | vLLM | SGLang |
|---|---|---|
| FSDP2 DP (dense) | supported | supported |
| TP=2 | supported | supported, gpt-oss included (its hand-sliced `sinks` are sent once from the gathered drain) |
| EP=2; EP=2 + ETP=2, EP=2 + TP=2, EP=4, with or without LoRA / expert LoRA; expert LoRA resume | supported | supported |
| Undistributed MoE (`ep_group_size == 1`, EP layers present) with 2×TP=2 / 4×TP=1 servers, expert sync and R3 | supported | supported |
| Trainer and server on different nodes over EFA ([above](#servers-on-other-nodes-efa)) | supported; not run on the 0.32.0 image | supported; not run on the 0.5.22 image |

An SGLang server under its own expert parallelism (`SGLANG_TP=2 SGLANG_EXTRA_ARGS="--ep-size 2"`) takes
the sync for Qwen3 MoE. Measured on SGLang 0.5.17 (not re-run on 0.5.22), gpt-oss fails at start under
`--ep-size`: its bf16 fused expert loader narrows only the TP dimension and copies the global expert count
into local storage (`_load_w2`).

## Troubleshooting

Server-side symptoms; the trainer-side weight-sync errors (group address, port bind, transport
mismatch, SGLang cuMem) are rows of [Troubleshooting](../reference/troubleshooting.md#symptom--cause--fix).

| Symptom | Cause → fix |
|---|---|
| `/health` is 200 but every completion hangs after a trainer died mid-sync | SIGKILL (`docker rm -f`) skipped the client's `atexit` resume, so the engine stays paused → POST `/resume` (vLLM) or `/continue_generation` (SGLang), or restart it |
| Group formation times out, "1/2 clients joined" | A bridge network hides the rendezvous port → `network_mode: host` on both containers |
| First sync takes minutes, `/health` unanswered | One-time NCCL group init per server; the engine pauses mid-update → wait |
| 400 on every rollout (vLLM) | A thinking budget without a reasoning parser → set `VLLM_REASONING_PARSER`; otherwise rollouts return zero tokens and the run trains on all-masked batches |
| Run completes with a flat zero gradient | Missing or wrong tool parser on a native-tool environment ([parsers](#running-it)) |
| Log-ratio drifts on SGLang while the server log stays clean | The MoE loaders skip unmapped expert names without a warning → a clean log proves nothing; the construction gates are the guard |
| `RoutedExperts: Failed` (vLLM log) | Layerwise-reload patch missing → expert syncs silently reverted; rebuild (`make build-vllm`) |
| `/init_weight_transfer_engine` answers 500 (`NCCL error: unhandled cuda error`) while `/health` is 200 | Re-init patch missing → a stranded communicator per connection; rebuild and recreate the container |
