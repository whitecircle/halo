# Rollout Servers

A rollout server is a **separate inference container**, vLLM or SGLang, that generates the model's
responses (rollouts) during RL. The trainer sends prompts to it over HTTP and pushes updated weights
back over a per-server NCCL group. Every on-policy GRPO trainer needs one:
[Online](../training-methods/grpo/online-grpo.md) and
[Async GRPO with Environments](../training-methods/grpo/async-grpo/README.md).

The training environment never imports either engine (ABI-incompatible stacks); it speaks to them
through the vendored client in `src/distributed/nccl/`. Select the engine with
`rollout_backend: vllm | sglang` (async GRPO only; Online GRPO is vLLM-only by construction). Every
other server knob is the same for both engines:

```yaml
rollout_backend: vllm            # default
rollout_server_configs:          # one entry per server
- url: "http://localhost:8000"
  group_port: 51216              # optional; server N defaults to vllm_group_port + N
- url: "http://localhost:8001"
  group_port: 51217
  group_host: "127.0.0.1"        # optional per-server override of VLLM_/SGLANG_GROUP_HOST
```

`url` is required; `group_port` and `group_host` are optional. Ports must be unique across servers
(enforced). They must also not be handed out as ephemeral source ports by the trainer host: the
defaults sit inside Linux's 32768–60999 range, so reserve them (`net.ipv4.ip_local_reserved_ports`)
or choose ports outside it.

An entry's keys are read by name with no per-entry validation, so a misspelled `group_port` silently
falls back to `vllm_group_port + index`. The top-level keys are engine-neutral and the parser renames
none: any other spelling raises the unknown-key error. Trainer-side rollout mechanics (prefetch and
its one-step staleness, sync cadence, trajectory-length knobs) stay on the
[Async GRPO with Environments](../training-methods/grpo/async-grpo/setup.md#weight-synchronization) page.

| | vLLM 0.26.0 | SGLang 0.5.17 |
|---|---|---|
| Trainers | online, async GRPO | async GRPO |
| Sampled-token ids | `--return-tokens-as-token-ids` (server flag) | per-request `return_meta_info` |
| IS-reference logprobs (the sampling distribution's) | `--logprobs-mode processed_logprobs` (server flag; the default `raw_logprobs` is pre-temperature and refused at any `rollout_temperature` ≠ 1) | default (post-temperature, pre-nucleus; keep `SGLANG_RETURN_ORIGINAL_LOGPROB` unset) |
| [R3 routing replay](../training-methods/grpo/async-grpo/objective.md#routing-replay) | `--enable-return-routed-experts` + `--moe-backend triton` | `--enable-return-routed-experts` + `--moe-runner-backend triton` |
| Thinking budget (`rollout_max_thinking_tokens`) | enforced engine-side with a reasoning parser and `VLLM_USE_V2_MODEL_RUNNER=0`; harmony-disabled gpt-oss arms it off the toolkit plugin's marker ([GPT-OSS](../models/gpt-oss.md#serving-for-grpo-vllm)) | rejected at config time |
| Expert layout on sync | the layout the family's own `gather_expert_state_dict` emits, per-expert or fused; 0.26.0's expert loader reads both. A family whose hub namespace differs from its module tree (Step-3.7's per-layer `moe.gate_proj`/`up_proj` stacks) is re-spelled through transformers' save-side revert, so the engine receives its hub keys | the same layouts, read by 0.5.17's per-family loaders; the families they cannot update are listed under [Which families each engine serves](#which-families-each-engine-serves) |
| Trainer expert distribution ([EP/ETP](../reference/glossary.md#parallelism)) | supported | supported |

Use vLLM unless a run needs SGLang specifically.

## Weight sync

Both engines receive the **full model** every sync: merged LoRA touches ~95% of bytes, so there is no
delta path (~42 GB at 20B, ~1–2 s steady over loopback). Re-assembling the sharded weights is a
collective; every rank takes part and none may skip.

One rank owns the clients and does the sending: the **forwarding rank** (global main; TP-rank 0 under
TP). Its sends sit between the gathers, so a failure there (an engine 500, a refused tensor, a host
OOM on the snapshot) is recorded rather than raised: the rank stays in every remaining collective,
and the whole world raises together at the end of the sync, naming the failing rank and its cause.

The gather **reshards the FSDP2 modules first**. A forward leaves their transient unsharded params
registered while the optimizer steps the shards. Reading the registered params would ship a policy
one optimizer step behind and fold a PEFT merge into a copy the next unshard discards.

### Group rendezvous

Each client hosts its group's rendezvous on the trainer: a TCP store on `group_port` that the
engine's workers dial at the advertised group address to fetch the NCCL bootstrap. The store takes
unauthenticated connections until the client closes its communicator. vLLM's workers unpickle the
bootstrap they read from it, so a host that reaches the port while a group forms can run code in the
server; SGLang's c10d reads raw bytes, but the same trust assumption holds. **The group port must be
reachable only by trusted hosts.**

On both engines the store listens on the advertised address alone: `group_host` /
`VLLM_GROUP_HOST` / `SGLANG_GROUP_HOST` when set, else loopback for a server on the same host and
the default-route NIC for one on another node. A name is resolved once to its IPv4 address, and the
engine is sent that address. The client binds the listening socket and hands it to torch's
`TCPStore` (`master_listen_fd`); a store master that opens its own listens on every interface
whatever address it is given. An address this host cannot bind alone (a NAT or port-mapped address,
the `0.0.0.0` wildcard, a name resolving to loopback for a remote server) raises before the engine
is asked to join, and so does one with no IPv4 address: IPv6 is unsupported.

`HALO_WEIGHT_SYNC_BIND_ALL=1` puts the store on every interface instead and advertises the
configured address unresolved, for a trainer the server reaches through NAT or a port mapping. The
mapping must keep the port, and the trainer must reach its own advertised address: the store's own
client dials it.

### Streaming and quiesce

The push is **streamed, not buffered**. Both engines take an update as a sequence of declared chunks
inside one quiesce: `/start_weight_update` … N × `/update_weights` … `/finish_weight_update` on
vLLM, N × `/update_weights_from_distributed` between `/pause_generation` and `/continue_generation`
on SGLang. The forwarding rank sends each chunk as the gather fills it and stages one chunk on its
sync GPU, not one model (~800 GB at 400B). `/finish_weight_update` closes the layerwise reload phase
on every path; the broadcast is packed (~1 GB buffers, double-buffered) on vLLM and typed 1 GB chunks
on SGLang.

The chunk stays on the device, not in host memory: a chunk that transits pinned host memory is copied
out and back over PCIe before the NIC sees it, 19–24 GB/s against 53–80 GB/s staged on the device.
The chunk is cut before the budget (`HALO_WEIGHT_SYNC_CHUNK_MB`, 1 GiB) is exceeded; a tensor above
it is a chunk of its own.

The forwarding rank's peak during a sync is therefore the assembled EP layer being sent (the largest
rank-local allocation, ~28 GB for one 397B layer), the staged chunk, the snapshot of the largest
tensor, and the engine path's buffers: the SGLang arena, grown to the largest chunk seen, or the two
fixed 1 GiB vLLM packed buffers.

In multi-server mode (`rollout_server_configs`) one snapshot per parameter is shared across all
servers. Each chunk goes out to every server on concurrent threads and is released once they all
have it. The threads share the forwarding rank's GPU, NICs and process, so the fan-out costs the sum
rather than the max: two servers each push at 27 GB/s over EFA, half of one server's rate.

A chunk cannot be replayed. A server that fails **after** its first chunk is reported rather than
reconnected; the trainer does not hold what already landed. One that fails before any chunk went out
(an engine restarted between syncs, the common case) is recovered by the reconnect + re-flush. Each
client owns its own NCCL connection to its server on a `group_port` bound on the *trainer* host
([group ports](../training-methods/grpo/async-grpo/setup.md#weight-synchronization)).

**The quiesce spans the streaming, not just the final broadcast.** The update opens with the first
full chunk (~1 GB into the gather) and closes when the last one lands, so a server stops serving for
as long as the gather runs: minutes at 397B. Only the raw-model path is rolling, one server at a time
— a single training process, no adapters, no EP wrappers, in multi-server mode; every other shape
pauses all servers together
([single vs multi-server](../training-methods/grpo/async-grpo/setup.md#multiple-servers-and-prefetch)).

The client pauses vLLM with `/pause?mode=keep`: in-flight generations (the prefetched rollout round)
freeze and resume under the new weights on `/resume`. That is the one-step staleness the
sampling-logprob IS ratio corrects. vLLM's own default is `abort`, which hands every in-flight request
back as a fragment with an ordinary stop reason; once the training pass is shorter than a rollout
round that is most long turns, every step.

SGLang's pause is `/pause_generation {"mode": "abort"}` (its post-update cache flush asserts an idle
scheduler), so in-flight generations are dropped across that window. The rollout actor re-issues an
aborted turn (`finish_reason: abort`) against the same observation, up to `max_retries` aborts per
turn, rather than stepping the environment with the fragment. An abort is never charged as a length
cut: it consumes no length-cutoff recovery.

The paused window is not charged to the episode: every rank credits the push's duration to its
in-flight episodes' `episode_timeout` (the deadline counts engine-serving time). `request_timeout`,
aiohttp's total per request, is not credited, so it must still exceed the longest turn plus one sync.
Otherwise the frozen request times out and its retry re-issues a turn the engine is still completing.

**An interrupted mid-stream sync leaves that server unusable.** The engine then holds neither the
old policy nor the new one, and vLLM's layerwise reload materializes a layer whose tensors straddled
the boundary from *uninitialized* storage while it waits for the rest. The abort therefore leaves
that engine **paused** instead of resuming it, refuses every later sync on that client, and logs
`RESTART the … server`. Restart the container: the trainer kept no copy of what landed and cannot
repair it.

On vLLM, each client owns **one persistent CUDA stream pair** for the pack uploads. PyTorch's caching
allocator keeps freed blocks in per-stream pools, so a fresh stream per sync would strand one payload
of reserved memory every sync; the forwarding rank's steady state is therefore its training footprint
plus about one sync of pack buffers. The SGLang client keeps one persistent send stream for the same
reason and drops its arena at the end of every sync, so between syncs that memory is the allocator's
rather than pinned at the largest chunk's size.

`HALO_WEIGHT_SYNC_MEM_LOG=1` (off by default) brackets each collective sync with a per-rank
`[mem rankNN] weight-sync pre/post` line ([Debugging](../reference/debugging.md#3-gpu-memory-profiling));
`reserved` far above `peak_alloc` on the forwarding rank means stranded allocator pools.

### Served weights stay in checkpoint layout

The sync writes bf16 checkpoint-layout tensors into the server's parameter storage in place, so any
load-time transformation of that storage silently corrupts every later update.

On vLLM the auto-selected Blackwell MoE backends (`FLASHINFER_TRTLLM`/`CUTLASS`) repack expert
weights; `--moe-backend triton` is **required** for MoE RL (its kernels read the checkpoint layout
directly). On SGLang the `triton`/`triton_kernel` runners load bf16 unpacked; `flashinfer_trtllm`
repacks and must not serve RL.

The same rule excludes quantized serving: a weight-quantized engine stores transformed tensors the
broadcast cannot update. Serve bf16. For gpt-oss that means the **BF16** checkpoint, not the stock
MXFP4 one (`openai/gpt-oss-20b`, whose quantization the engine auto-detects with no flag to fail on):
its MXFP4 expert loader has no branch for a bf16 expert tensor, so every synced expert weight is
dropped while the biases land, silently, and the trainer sees only a slow log-ratio drift.

### vLLM server patches

**The layerwise-reload patch** (`docker/vllm/patches/vllm_layerwise_reload_patch.py`, baked into the
image, applied via `sitecustomize` in the API server and every engine-core worker) closes a silent
corruption path.

vLLM's reload moves each layer to the meta device and wraps its `weight_loader`s. Model code that
writes weights with a direct `param.copy_()` (gpt-oss experts and attention `sinks`) lands on a meta
tensor as a no-op, and the reload then re-registers the **saved** tensors: every sync reverts those
weights while `/update_weights` returns `200 OK`.

The patch excludes the affected classes from the reload lifecycle (`SKIP_LAYER_NAMES`:
`RoutedExperts` / `FusedMoE`, `OAIAttention`, and `Gemma4Router`, whose partial load would
materialize an uninitialized buffer into live memory). The class set is version-dependent and
asserted against the installed vLLM at build, so an upstream refactor fails the image build.

Missing, it shows as `RoutedExperts: Failed to load weights` per expert layer per sync in the server
log, with the trainer's `sampling/logratio_mean` drifting monotonically negative
(`YaRNScalingRotaryEmbedding: Failed to load weights` is benign: no loadable weights). Do not
override `PYTHONPATH` at `docker run`: dropping `/opt/nccl_compat` kills weight-transfer init
(`No module named 'src'`).

**The weight-transfer re-init patch** (`docker/vllm/patches/vllm_weight_transfer_reinit_patch.py`,
applied through the same `sitecustomize` hook) destroys the engine's previous NCCL communicator
before `/init_weight_transfer_engine` builds the next one, and again on engine shutdown, so one
server outlives any number of trainer connections.

Stock vLLM 0.26.0 only drops the reference, and `PyNcclCommunicator` has no `__del__`, so each
connection strands a live communicator (~633 MiB of device memory per connection on every
engine-core worker) until `ncclCommInitRank` fails while `/health` still answers `200`. The trainer
half is symmetric: `close_communicator()` aborts its own communicator instead of dropping it. Both
halves are asserted by `tests/gpu/trainers/grpo/test_vllm_weight_transfer_reinit.py`.

Checkpoint layout and expert un-fuse rules live in
[Checkpoints](../reference/checkpoints.md#serving-on-vllm--sglang).

### Construction gates

Which families each backend accepts for RL is gated trainer-side at construction
(`validate_weight_sync_support`, `src/trainers/grpo/rollout/weight_sync.py`). Inkling, GLM-5 Next
and Cohere2 MoE declare `_supports_weight_sync = False` and are refused on both engines; every other
refusal is an engine fact on that engine's client
([Which families each engine serves](#which-families-each-engine-serves)).

A bnb-quantized (QLoRA) base is refused on both engines. So is any MoE without a live EP wrapper
(`use_grouped_gemm: false` at `ep_size: 1`), for every MoE family: the experts ship in the layout the
wrapper's gather emits, and without one the dense walk would forward the stock module tree's fused
expert tensors under module names, which the engine's loader drops with no error.

The gate reads the family's contract off the live wrapper, or off the `model_type` registry when a
run has none; `use_grouped_gemm: true` (the torchrun default) installs the wrappers.

**Hub-namespace families.** The sync forwards every tensor under the key a gathered checkpoint would
carry. Where the live module tree and the hub checkpoint differ, the rewrite is derived, not
tabulated: Laguna's `_EXPORT_KEY_RENAMES` pairs, and, for a family declaring
`_EXPORTS_HUB_NAMESPACE` (Step-3.7 Flash), transformers' own save-side conversion revert (the
reversed `WeightRenaming`/`WeightConverter` entries `save_pretrained` applies).

One-to-one renames stream tensor by tensor. A tensor a reverse converter claims (a fused
`gate_up_proj` the hub stores split, a vision tower's q/k/v the hub stores fused) is held until its
sources are complete, since the engine loads one tensor at a time.

Any family whose hub checkpoint sits behind such a conversion joins by declaring the flag on its EP
layer once a pinned engine serves it (`_supports_weight_sync` stays off for GLM-5 Next and Inkling
because none does). Per-family server flags: [Step-3.7](../models/step3p7.md#serving-for-grpo-vllm).

**GptOss needs live attention sinks.** `reset_sinks: true` under `flash_attention_2` rebinds
`attn.sinks = None`, and the sync forwards `named_parameters()` only, so nothing is ever pushed for
those slots. The server keeps generating with the pretrained sinks against a sink-free trainer:
permanently off-policy with no error at sync time.

The validator refuses that shape at construction; on-policy GptOss needs `reset_sinks: false` with a
sink-carrying implementation (FA4 or eager), which is what the shipped GRPO configs set.

It refuses two more shapes for the same reason, state the sync cannot carry: `train_sinks: true`
(sinks that change every step, SFT-only), and an enabled router bias-update balancing bias, adopted
or transient, which the parameter-only payload never pushes. The shipped GRPO scripts downgrade
`moe_balancing` to `none` themselves.

### Which families each engine serves

Both engines' loaders read a family's experts in the layout its own `gather_expert_state_dict`
emits: per-expert tensors for Qwen3 MoE, GLM-4 MoE Lite, Laguna, Bailing and LFM-2, the fused pair
for Qwen3.5/3.6 and Gemma 4, GptOss's interleaved pair. The sync carries one layout per family on
either engine.

What differs per engine is which families its pinned release can take an online update for at all.
Each client declares those with the loader fact (`UNSERVABLE_MODEL_TYPES`), quoted by the
[construction gate](#construction-gates).

| Family (`model_type`) | vLLM 0.26.0 | SGLang 0.5.17 | Loader fact |
|---|:--:|:--:|---|
| Mistral4 | ✗ | ✗ | neither registers a class ([Mistral4](../models/mistral4.md#serving)) |
| Ling 3.0 (`bailing_hybrid`) | ✗ | ✗ | no class for `BailingMoeV3ForCausalLM` |
| Ring (`bailing_moe_linear`) | ✗ | ✗ | the checkpoints declare `BailingMoeLinearV2ForCausalLM`; both register `BailingMoeV2_5ForCausalLM` |
| Zaya | ✗ | ✗ | vLLM ships no native class; SGLang's loader reads the pre-transformers-5.14 per-expert checkpoint (`zaya_block.experts.local_experts.N.linear_fc1`) |
| DeepSeek-V4 | ✗ | ✗ | vLLM's loader targets the fp8/fp4-packed release layout; SGLang's maps per-expert `w1/w3/w2` |
| Laguna | ✓ | ✗ | SGLang's `load_weights` asserts every routed-expert tensor of every sparse layer per call |
| Step-3.7 (`step3p7`, `step3p5`) | ✓ | ✗ | `Step3p5ForCausalLM.load_weights` asserts full parameter coverage per call |

The families not in that table sync on both engines, expert distribution included ([CI](ci.md) has
the per-family pass): dense families, GptOss, Qwen3 MoE, Qwen3.5/3.6 MoE, GLM-4 MoE Lite, Gemma 4,
Ling 2.0, LFM-2 MoE. Inkling, GLM-5 Next and Cohere2 MoE are the exception — they declare
`_supports_weight_sync = False` and are refused on both ([Construction gates](#construction-gates)).

Three SGLang 0.5.17 loader facts shape its image and its client.

**Routers that cache a derived form.** Upstream, the GLM-4 gate caches an fp32 copy of its weight at
the first forward and never re-reads the parameter, and the Gemma 4 router folds `scale` into its
norm once, behind a latch. A synced router weight lands in the parameter while routing keeps the
launch values, with no error.

`Dockerfile.sglang` applies `docker/sglang/patches/patch_sglang_weight_updates.py`: the gate reads
its fp32 weight live, and a load of `scale` releases the latch. The script asserts its pre-images
before rewriting and its post-images after, at build, and stays in the image (`/opt/halo/`) so
`--verify` re-checks a running container.

**Fused a-projection halves.** The MLA loaders (GLM-4 MoE Lite here) concatenate `q_a_proj` and
`kv_a_proj_with_mqa` from a cache local to one `load_weights` call, one chunk, and drop a half that
arrives alone. The client declares the pair (`CO_LOADED_PARAM_GROUPS`) and the chunker keeps it in
one chunk, deferring the first half when the byte budget would cut between them; a pair still
incomplete when the sync closes refuses the close.

**The triton runner.** The `flashinfer_trtllm*`, aiter and quantized runners repack expert weights
after the load, and an online update writes the canonical layout into the repacked buffer.
`SGLANG_MOE_RUNNER_BACKEND=triton` (the compose default) is the runner whose weights an update
reaches unchanged; R3 capture needs it too.

The sync ships hub names and full unsharded tensors into the engine's own `load_weights` mapping;
each TP rank narrows its slice, and under `--ep-size` the loader keeps its local experts and drops
the rest.

An expert name that mapping does not cover leaves **no server-side signal**: the MoE loaders
`continue` on an unmatched `mlp.experts` name *before* the `not found in params_dict` warning
(reachable only from the dense loaders), and the update still returns `200 OK`. The engine keeps
serving its launch-weight experts under a freshly synced router; the server tier's expert-only round
is what catches it. The trainer-side construction gate is the whole guard.

Expert distribution (EP, ETP) is accepted on both engines: the sync group is ordinary NCCL beside
DeepEP's.

## vLLM

`Dockerfile.vllm` builds `vllm-server:0.26.0` with the native NCCL weight-transfer engine, the
layerwise-reload patch, R3 routed-experts capture (base64-npy `routed_experts` per completion
choice), and `nvidia-nccl-cu13` installed at `uv.lock`'s exact pin, the same NCCL the training
images run. `VLLM_NCCL_SO_PATH` is baked to that wheel so the base image's older system copy can
never win the soname race.

A skew fails `ncclCommInitRank` at `/init_weight_transfer_engine` or hangs it with no error;
rebuild the image after any lock bump of the pin.

0.26.0 is the last vLLM release on torch 2.11, the training image's torch and NCCL generation; 0.27
moves to torch 2.13, whose NCCL does not match that pin. The image also installs the EFA userspace
the training image runs (`docker/efa/install_efa_userspace.sh`), so the group can ride EFA from a
trainer on another node ([Servers on other nodes](#servers-on-other-nodes-efa)).

### Config-schema parity {#config-schema-parity}

The server parses every checkpoint with **its** transformers, pinned to the 5.14 line, one line below
the training image's 5.16 (`Dockerfile.vllm` asserts the pin at build). **Gemma 4 is what pins that
line.**

vLLM's Gemma 4 model code (0.25.1 through 0.28.0) reads the 5.14 config schema (flat
`global_head_dim` / `num_global_key_value_heads`, a global `num_attention_heads`). 5.16 folds those
into `per_layer_config` and raises `AmbiguousGlobalPerLayerAttributeError` on vLLM's
`get_head_size`, so a 5.16 server makes Gemma 4 unservable on every one of those vLLM versions.
Toolkit exports are therefore written in the flat form.

**Step-3.7 is a different constraint**, not a dialect: this transformers has no `step3p7` class at
all and reads the family only through the release's `auto_map` modules, which its release config
loads cleanly on either line. Its exports therefore carry the source repo's own config schema and
those modules ([Checkpoints](../reference/checkpoints.md#what-gets-saved)).

`docker/vllm/parity/check.py` runs at image build: the server's transformers must parse what the
toolkit exports for every family whose EP layer admits weight sync, since the server loads that
checkpoint before a single tensor can be synced into it. One `config.json` fixture per family,
rendered offline from the tiny roster config in `tests/common/models.py` (only the source-schema
carry is pinned to a release config at a fixed revision, the one thing a tiny config cannot express),
plus a negative control under `unparseable/` (the folded Gemma 4 form, the native-schema Step-3.7
export) that must still be refused.

A transformers bump on either side then fails the build rather than the first live sync, and a family
no pinned engine can load (Mistral4) surfaces as a refusal rather than a dead sync.
`tests/cpu/checkpoint/test_vllm_parity_fixtures.py` fails when the roster or the fixtures drift.

`docker-compose.vllm.yml` runs the server with `network_mode: host` + `ipc: host`: to form the NCCL
group the two sides first find each other on an ephemeral trainer port (the rendezvous), which a
bridge network would hide, and group formation then times out at "1/2 clients joined".

Prebuilt: `docker pull public.ecr.aws/whitecircle/halo:vllm-0.26.0` (anonymous, no AWS account), then
`docker tag public.ecr.aws/whitecircle/halo:vllm-0.26.0 vllm-server:0.26.0` — compose names that tag.

```bash
make build-vllm                        # or the pull + tag above
VLLM_MODEL=Qwen/Qwen3-30B-A3B VLLM_CUDA_DEVICES=6,7 VLLM_TP=2 \
  TRAINER_CUDA_DEVICES=0,1,2,3,4,5 \
  docker compose -f docker-compose.vllm.yml up vllm-server
```

| Variable | Default | Purpose |
|---|---|---|
| `VLLM_MODEL` | `Qwen/Qwen3-0.6B` | Hub id or local checkpoint |
| `VLLM_PORT` | `8000` | Bound on the host (`network_mode: host`). One knob for the whole stack: it drives the serve command, the healthcheck, the container's `VLLM_SERVER_URL` and the readiness banner, and the Makefile derives its own `VLLM_SERVER_URL` from it (`SGLANG_PORT` is the SGLang equivalent) |
| `VLLM_CUDA_DEVICES` | `7` | Server GPUs; must exclude the trainer's (a rank cannot broadcast to itself). Selects via `CUDA_VISIBLE_DEVICES` inside a container that sees every GPU: hiding devices from the container instead (`--gpus device=N`) breaks the cross-container NCCL P2P import of the trainer's buffers, and `/init_weight_transfer_engine` fails with `unhandled cuda error` on the first connection (the stranded-communicator case under [Troubleshooting](#troubleshooting) fails only after several) |
| `VLLM_TP` | `1` | `--tensor-parallel-size` |
| `VLLM_GPU_MEM` | `0.85` | `--gpu-memory-utilization` |
| `VLLM_MOE_BACKEND` | `triton` | Keep `triton` for MoE RL ([Weight sync](#weight-sync)) |
| `VLLM_ENABLE_R3` | *(unset)* | Any non-empty value adds `--enable-return-routed-experts` (R3 capture); the `triton` MoE backend is the one the capture hook reaches |
| `VLLM_ATTENTION_BACKEND` | *(unset = auto)* | `--attention-backend`. GLM-4 MoE Lite (MLA) on Blackwell needs `CUTLASS_MLA`: the auto-selected FlashInfer MLA decode kernel rejects its head config at graph capture ([MLA backend](../reference/checkpoints.md#serving-on-vllm--sglang)) |
| `VLLM_SPECULATIVE_CONFIG` | *(unset)* | `--speculative-config` JSON, e.g. `{"method":"mtp","num_speculative_tokens":2}` for a checkpoint that ships an MTP head ([Throughput](#throughput)); pair it with `VLLM_PREFIX_CACHING_FLAG=--no-enable-prefix-caching` on 0.26.0 |
| `VLLM_PREFIX_CACHING_FLAG` | `--enable-prefix-caching` | Set to `--no-enable-prefix-caching` to turn the cache off |
| `VLLM_ENFORCE_STRICT_TOOL_CALLING` | `0` | vLLM's grammar-constrained tool calling; off so the served distribution is the policy's and the engine core skips per-step grammar work ([Throughput](#throughput)) |
| `VLLM_TUNED_CONFIG_FOLDER` | *(unset)* | Directory of tuned Triton MoE tile configs, visible inside the container ([Throughput](#throughput)) |
| `VLLM_TOOL_PARSER` | `hermes` | `--tool-call-parser`; per-family values below |
| `VLLM_TOOL_PARSER_PLUGIN` | *(unset)* | `--tool-parser-plugin` path (gpt-oss uses the baked `/opt/gpt_oss_text_tool_parser.py`) |
| `VLLM_CHAT_TEMPLATE` | *(unset)* | Set to the SAME `.jinja` the trainer's `chat_template:` uses; the file must be visible inside the server container |
| `VLLM_REASONING_PARSER` | *(unset)* | Required when training sets `rollout_max_thinking_tokens` |
| `VLLM_REASONING_PARSER_PLUGIN` | *(unset)* | `--reasoning-parser-plugin` path for families without a built-in parser |
| `TRAIN_IMAGE` | `halo:blackwell` | Image the compose file's optional trainer service runs |

Flags the compose file already sets that are load-bearing for RL:

- `--weight-transfer-config '{"backend": "nccl"}'` enables the transfer engine.

- `--return-tokens-as-token-ids`: `train_on_sampled_tokens` (default on) recovers the sampled ids
  from the logprobs, which vLLM spells out only under this flag. Without it every turn falls back to
  re-tokenizing a chat-template re-render: one warning, then a whole run training on tokens the
  engine never sampled.

- `--logprobs-mode processed_logprobs`: the reported logprobs are the sampling distribution's
  (temperature and top-p applied). The default `raw_logprobs` are pre-temperature, so at any
  `rollout_temperature` ≠ 1 every IS weight is π^T / π^1 while `sampling/is_ratio_mean` still reads
  ≈ 1.

    The trainer scores its log-probs at the sampling temperature (`rollout_temperature`, or TRL's
    `temperature` on the online arm) and divides by the reported values. Above 1 the weights tilt
    toward improbable tokens (entropy climbs step over step); below 1 toward confident ones (entropy
    collapses). The trainer probes each server at startup (temperature 2 must halve the top-1/top-2
    gap) and refuses a raw server whenever that temperature ≠ 1.

    Under this mode a top-p < 1 also renormalizes every uncertain position over its nucleus, lifting
    it by the nucleus mass. Whenever the per-token log-ratios are summed per sequence — the env arm's
    trajectory geometric band (`isr_geo_band_min/max`), or an online `sequence_*`
    `vllm_importance_sampling_mode` — that pairing is probed and refused: the band reads the sum as
    drift and a sequence-level IS weight collapses toward 0, stalling the run silently. Fix it with
    `rollout_top_p: 1.0` / `top_p: 1.0`, a `token_*` IS mode, or by dropping the band.

`--max-model-len` is left unset: the server serves the model's native context window. The trainer's
startup probe reads it off `/v1/models` and **raises** when `max_prompt_length` plus one turn's
generation exceeds it. The worst-case multi-turn budget only warns, since a rollout growing past the
window OOMs the training forward before the fail-on-overflow check.

R3 runs (`routing_replay: rollout`) set `VLLM_ENABLE_R3=1`, which adds
`--enable-return-routed-experts`; without the flag the trainer raises at the first capture. The
FlashInfer monolithic MoE kernels bypass the capturer and return all-zero expert ids, hence the
triton backend. Set `VLLM_ENABLE_R3` for MoE models only: the capturer reads the experts-per-token count off the
config and the engine exits at start on a dense model, which under the compose `restart` policy
shows up as a server that restarts forever and never turns healthy.

`VLLM_USE_V2_MODEL_RUNNER=0` is not a serve flag but an env var the compose file already passes
through from your shell (`VLLM_USE_V2_MODEL_RUNNER=0 docker compose -f docker-compose.vllm.yml up`).
Any run sending thinking budgets needs it, R3 or not: Model Runner V2 rejects `thinking_token_budget`
with a 400 on every request, so each rollout errors instead of generating (zero tokens,
`episode/error_rate` 1).

Spell it `0` or `1` and nothing else: vLLM reads it with `int()`, so `false` or an empty value (an
empty key in the repo-root `.env` counts) kills the server at startup with a bare `ValueError`.

Tool parsers are per family. For **native-tool** envs (`code_contests`, `swe`, `mcp`, `qa_search`,
open-book `exam_qa`) the absence of the right one is silent and fatal to RL: calls stay text, no
`tool_calls`, every episode reward 0, flat zero gradient.

ReAct envs parse actions from the response text, so a mismatched parser costs them nothing. A
*missing* one still 400s, since the trainer sends `tools` for any env with a tool registry:

| Family | `--tool-call-parser` |
|---|---|
| Qwen3 / Qwen3.5 / 3.6 | `qwen3_xml` (hermes does NOT parse their XML calls) |
| GPT-OSS | bundled plugin `gpt_oss_text` via `VLLM_TOOL_PARSER_PLUGIN`; reasoning plugin `/opt/gpt_oss_reasoning_parser.py`, parser `openai_gptoss` ([GPT-OSS](../models/gpt-oss.md#serving-for-grpo-vllm)) |
| GLM-4 | `glm45` / `glm47` |
| Gemma 4 | `gemma4` (hermes leaves its `<\|tool_call>call:…<tool_call\|>` calls as text, so no tool ever runs); with a thinking budget (`rollout_max_thinking_tokens`, or an env's per-effort `thinking_tokens` profile) also `VLLM_REASONING_PARSER=gemma4` and `VLLM_USE_V2_MODEL_RUNNER=0`, else every request 400s |
| most others | `hermes` (`<tool_call>` XML) |

`docker-compose.vllm.yml` defaults **both** containers to the no-fabric recipe (`NCCL_IB_DISABLE=1`
and `NCCL_NET=Socket` on each, `NCCL_P2P_LEVEL=NVL` on the server). On a host without a fabric the
cross-container group then takes NVLink + sockets on both ends by declaration rather than by each
side's own fallback; two containers that land on different nets form the group and hang at the first
collective ([Troubleshooting](#troubleshooting)).

On an EFA host layer `docker-compose.vllm.efa.yml` over it
([Servers on other nodes](#servers-on-other-nodes-efa)). `NCCL_NET` is process-global, so a trainer
left at `Socket` there sends every collective over TCP and breaks DeepEP.

`VLLM_GROUP_HOST` (or an entry's `group_host`, for that one server) names the trainer address the
server dials back to and the rendezvous store binds ([Group rendezvous](#group-rendezvous)), and
touches nothing else; a same-host compose stack needs no value, and the
resolution order is on
[Online GRPO — Multi-homed nodes](../training-methods/grpo/online-grpo.md#multi-homed-nodes-vllm_group_host).
`NCCL_SOCKET_IFNAME` is not a transfer-group knob ([NCCL transport](#nccl-transport-sglang)).

## SGLang

`Dockerfile.sglang` builds `sglang-server:0.5.17` to align NCCL: upstream's wheel trails `uv.lock`'s
exact pin (what the training images run), and weight sync needs both ends on one runtime.

The build bumps the NCCL wheel, installs the EFA userspace the training image runs
(`docker/efa/install_efa_userspace.sh`), and applies the loader patches an online update needs
(`docker/sglang/patches/`, [Which families each engine serves](#which-families-each-engine-serves)).
It asserts that the weight-sync routes, request schemas and rendezvous convention still exist, so an
upstream refactor fails the build instead of a training run.

The server's transformers stays at SGLang's own exact pin (5.12.1 for 0.5.17); only NCCL and the EFA
userspace are rebuilt. Serving-only use can run upstream directly
(`SGLANG_IMAGE=lmsysorg/sglang:v0.5.17`); weight sync needs this image. 0.5.17 is the last SGLang
release on torch 2.11, the training image's torch and NCCL generation; 0.5.18 moves to torch 2.13,
whose NCCL does not match the pin weight sync needs on both ends. A later release changes nothing
here: 0.5.19's `--moe-a2a-backend deepep_v2` forces `--moe-runner-backend deep_gemm`, which an online
update does not reach.

Prebuilt: `docker pull public.ecr.aws/whitecircle/halo:sglang-0.5.17`, then set
`SGLANG_IMAGE` to that tag (it defaults to the locally built `sglang-server:0.5.17`).

```bash
make build-sglang                      # or the pull above
SGLANG_MODEL=Qwen/Qwen3-0.6B docker compose -f docker-compose.sglang.yml up
```

| Variable | Default | Purpose |
|---|---|---|
| `SGLANG_MODEL` | *(required)* | Hub id or local checkpoint directory |
| `SGLANG_IMAGE` | `sglang-server:0.5.17` | The image compose runs; set it to the prebuilt `public.ecr.aws/whitecircle/halo:sglang-0.5.17`, or to `lmsysorg/sglang:v0.5.17` for serving-only use |
| `SGLANG_MODEL_DIR` | `/mnt/models` | Host dir mounted read-only, at the same path inside — point it at wherever your checkpoints live when `SGLANG_MODEL` is a path |
| `SGLANG_PORT` | `30000` | Bound on the host (`network_mode: host`) |
| `SGLANG_TP` | `1` | Tensor-parallel size; pair with `SGLANG_CUDA_DEVICES` |
| `SGLANG_CUDA_DEVICES` | `7` | Server GPUs; must exclude the trainer's |
| `SGLANG_GPU_MEM` | `0.85` | `--mem-fraction-static` |
| `SGLANG_TOOL_PARSER` | `auto` | `--tool-call-parser`. `auto` reads the parser off the chat template; its harmony channel-marker rule resolves gpt-oss to the harmony parser, so no per-family pin is needed. Override only for a template the detector does not cover |
| `SGLANG_MOE_RUNNER_BACKEND` | `triton` | `--moe-runner-backend`; keep `triton` for MoE RL and for R3 capture |
| `SGLANG_ENABLE_R3` | *(unset)* | Any non-empty value adds `--enable-return-routed-experts` (R3 capture) |
| `SGLANG_REASONING_PARSER` | *(unset)* | `--reasoning-parser` (`gpt-oss` for harmony models); separates reasoning from content in the response |
| `SGLANG_CHAT_TEMPLATE` | *(unset)* | `--chat-template`: the SAME `.jinja` the trainer's `chat_template:` uses; the file must be visible inside the server container |
| `SGLANG_ATTENTION_BACKEND` | *(unset)* | `--attention-backend`. `triton` for GLM-4 MoE Lite on Blackwell: its MLA head size has no kernel in the engine's default backend and the server exits at start (`Unsupported head dimensions`) |
| `SGLANG_TRUST_REMOTE_CODE` | *(unset)* | Any non-empty value adds `--trust-remote-code`, which the Bailing (Ling 2.0) repos need; their modeling code ships in the checkpoint |
| `SGLANG_EXTRA_ARGS` | *(unset)* | Further launch flags, verbatim. `--enable-deterministic-inference` for GLM-4 MoE Lite: under the triton attention backend its greedy logits differ between a prefill and a prefix-cache hit of the same prompt, which the tier's zero-noise baseline probe refuses |

`SGLANG_GROUP_HOST` belongs to the **trainer** (the `VLLM_GROUP_HOST` equivalent, separate because
the engines can sit on different hosts).

Engine behavior under RL:

- **Sampled ids** arrive per request: SGLang's OpenAI `logprobs` reports tokens as text, so the
  trainer sets `return_meta_info` + `return_prompt_token_ids` and reads
  `choice.meta_info.output_token_logprobs[i][1]`. No server flag.

- **Logprobs are post-temperature, pre-nucleus** by default: `sampler.py` divides the logits by the
  temperature before the log-softmax the reported values come from, and top-p renormalizes only the
  sampling probabilities.

    They are therefore the IS reference the trainer expects at any `rollout_temperature`, and a
    top-p < 1 leaves the geometric band untouched. Do not set `SGLANG_RETURN_ORIGINAL_LOGPROB`: it
    switches to raw values, and the startup probe refuses them.

- **A length cut-off is a `stop_reason`**, not a `finish_reason`. `get_finish_reason`
  (`src/inference/response.py`) reads `finish_reason or stop_reason`, so the rollout path grades an
  engine-truncated completion as truncated rather than as an answer.

- **R3**: serve with `--enable-return-routed-experts --moe-runner-backend triton`. The fused runners
  (`triton_kernel`, flashinfer; auto-selection picks one for most MoE shapes) bypass the capture hook
  and return nothing.

    The trainer opts in per request and decodes the wire format (response-level
    `sglext.routed_experts`, base64 raw int32) by the model's own layer/top-k counts. Rows cover the
    full sequence, so prompt spans replay too.

    The engine's capturer is per family: it serves GptOss, Qwen3 MoE, Qwen3.5/3.6 and GLM-4 MoE Lite,
    exits at start for a dense model and for Gemma 4 (whose config carries no
    `num_experts_per_tok`; the family has no routing replay on either engine) and raises at the
    first capture for Bailing. Serve those without `SGLANG_ENABLE_R3`.

- **`rollout_max_thinking_tokens` is rejected at config time** for every model: SGLang ignores
  unknown request fields, and the trainer wires neither of its budget mechanisms. Steer with the
  environment's `reasoning_effort`, which reaches the **chat template** only — a per-level
  `thinking_tokens` profile reaches no SGLang request field and is warned once, not enforced.

    The custom-logit-processor path needs a per-model class; the strict-thinking grammar needs a
    detector exposing `think_excluded_tokens`. For harmony models neither exists server-side.

- **`--dp-size > 1` needs `--enable-dp-attention`**, or the client refuses at group formation: plain
  DP replicas each restart `tp_rank` at 0, so their workers collide on `rank_offset + tp_rank` in the
  update group and no sizing can address them. The client reads the layout off `/server_info`.

- **`isr_engine_reference`** re-scores through `/generate` with `logprob_start_len`; no server flag.

- **`--enable-torch-compile` must stay off under R3 capture**: no step-time gain, and capture ×
  compile produces isolated catastrophic log-ratio rows (the IS veto/geo-band masks them, the trust
  region absorbing an engine numerics fault).

### NCCL transport (SGLang)

The group is ordinary NCCL, the same as vLLM's: on one host it takes CUDA IPC between the two
containers (`P2P/CUMEM`, NVLink), across nodes the fabric. The one engine-side requirement is
**cuMem parity**. SGLang's engine entry point sets `NCCL_CUMEM_ENABLE=0` process-wide unless the
variable is already set, while the trainer's NCCL has cuMem on.

The mismatch fails the first cross-container buffer import (`ncclP2pImportShareableBuffer ... invalid
argument`, then `Cuda failure 'invalid argument'. The full weights of the ModelRunner are partially
updated`). `docker-compose.sglang.yml` sets `NCCL_CUMEM_ENABLE=1`; keep it on a hand-run server. The
trainer keeps NVLink for its own FSDP2 collectives, so nothing about this engine changes the
trainer's NCCL env.

`NCCL_SOCKET_IFNAME` keeps the socket bootstrap off Docker's bridge and the per-container `veth`
pairs. On a host running other containers NCCL otherwise enumerates them too, and a veth carries no
host-to-host traffic (the first collective after the sync hangs) or disappears when its container
exits (`Call to bind failed: No such device` on the server, `400` on the update).

Both base compose files pass `NCCL_SOCKET_IFNAME=^docker,veth` by default (the vLLM file to its
training service too), a same-host value; the fabric recipe excludes `lo` as well
([Servers on other nodes](#servers-on-other-nodes-efa)). Override it to pin one NIC on a multi-homed
host, one every collective can use: the variable is process-wide and filters the interfaces the
default process group and DeepEP pick too.

The sync's sends drain under a 600 s deadline, the same on both engines
(`HALO_NCCL_SYNC_TIMEOUT_SECONDS` overrides it); a failed chunk drains the whole stream under a 30 s
deadline. On expiry the group is aborted instead of parking the trainer.

## Servers on other nodes (EFA)

The weight-sync group rides the node's fabric when both containers can drive it. All three images
carry one EFA userspace, installed whole by `docker/efa/install_efa_userspace.sh`: rdma-core, AWS
libfabric and one `aws-ofi-nccl` build. The pins live in the script, the inventory on
[Docker](docker.md#rdma-networking-infiniband-and-efa).

The stack is wire-sensitive down to rdma-core. At init the plugin probes libfabric for in-order RDMA
writes and forces `NCCL_PROTO=simple` when the probe fails; the answer comes from rdma-core's EFA
provider (`libefa`). The NGC base's MOFED `libefa` fails it and the installer's passes. Two
containers whose answers differ form the group with different NCCL protocol tables and hang at the
first collective, the same signature as a plugin-version mismatch.

So both ends must run images built from the same script. A pair that cannot be rebuilt runs with
`NCCL_PROTO=simple` on both ends instead; either restores the full rate. The preflight below reports
a side whose plugin forced the simple protocol. The upstream vLLM and SGLang bases ship no EFA
userspace, and NCCL then falls back to sockets with no message.

Server side, layer the EFA overlay after the base compose file:

```bash
docker compose -f docker-compose.vllm.yml -f docker-compose.vllm.efa.yml up -d vllm-server
SGLANG_MODEL=... docker compose -f docker-compose.sglang.yml -f docker-compose.sglang.efa.yml up -d
```

Each overlay adds `devices: /dev/infiniband` (a `-v` bind mount does not grant access), `ulimits:
memlock: -1`, and `NCCL_NET=Libfabric NCCL_NET_PLUGIN=ofi NCCL_IB_DISABLE=0
NCCL_SOCKET_IFNAME=^lo,docker,veth,tailscale`. Naming the plugin's net is deliberate: a missing or
mismatched plugin then fails group formation instead of silently falling back to sockets.

`lo` is in the exclusion because an excluded-only list still ranks loopback first: a server started
with the base files' `^docker,veth` advertises its NCCL bootstrap address as `127.0.0.1`, and a
trainer on another node fails group formation with `remote process exited or there was a network
error`. `tailscale` is excluded because `^docker,veth` leaves a Tailscale interface eligible. The
base files keep the no-fabric, same-host recipe.

Trainer side, the same env: `make ... EFA=1` adds `--device=/dev/infiniband` and those variables to
every `DOCKER_RUN` (and drops the socket forcing from `test-gpu-vllm` / `test-gpu-sglang`); a
hand-written `docker run` passes them itself. The trainer sets no NCCL env in code, and `NCCL_NET` /
`NCCL_NET_PLUGIN` are process-global, shared by the sync group and every trainer collective, so this
is the env the [multi-node recipes](../parallelism/launch-recipes.md#environment-variables) already
use.

GPUDirect RDMA runs through dmabuf with no `nvidia_peermem`; `/dev/gdrdrv` is a cross-node DeepEP
GIN requirement, not a sync one. Name the trainer address the server dials back to with the entry's
`group_host` (or `VLLM_GROUP_HOST` / `SGLANG_GROUP_HOST`) when the default-route NIC is not the one
the server can reach; Online GRPO has only the env var (TRL builds its client without `group_host`).

Verify the transport before training:

```bash
python scripts/profiling/weight_sync_transport.py --server-url http://<server>:8000 --backend vllm --expect efa
```

Run it from the trainer container, launched as the trainer would be (same image, devices and NCCL
env), on a GPU the server does not own.

It forms the group against the live server (either backend) and pushes one real parameter of the
served checkpoint: the input embedding by default, value unchanged, so the served model is
unchanged. The checkpoint read on the trainer side must be the one the server loaded (`--model-id`
when the served id is a server-side path or an alias).

It reports the transport NCCL formed on (`NET/Libfabric/…/GDRDMA` is EFA with GPUDirect; `NET/IB`,
`NET/Socket`, `P2P/CUMEM` for same-host CUDA IPC, or `SHM`), the `aws-ofi-nccl` build string, the
libfabric provider, and GB/s. It exits 1 when the group fails to form or the push changed the served
model; `--expect efa|ib|socket|p2p|shm` adds the transport to that gate. Flags:
[Scripts](../reference/scripts-reference.md#profiling--benchmarks).

Measured on 8× B300 nodes, trainer node → server node, one worker, the trainer's own streamed path, full
model per sync (NIC line rate ~100 GB/s per GPU; a raw NCCL broadcast reaches 93 GB/s): EFA with the
vLLM client 53 GB/s, EFA with the SGLang client 80 GB/s, sockets over the ENA 9.7 GB/s. At those
rates gpt-oss-120b (234 GB) syncs in 4.4 s, 2.9 s and 24 s.

The vLLM client's rate is set by its one HTTP round trip per chunk and rises with
`HALO_WEIGHT_SYNC_CHUNK_MB` (1 GiB → 54 GB/s, 8 GiB → 73 GB/s at this payload); the SGLang client's
by the fabric.

## Checking a server

```bash
curl -s localhost:8000/health          # vLLM (30000 for SGLang)
curl -s localhost:8000/v1/models

curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen3-0.6B","messages":[{"role":"user","content":"Say OK"}],"max_tokens":16}'
```

The eval runners under `scripts/environments/inference/` take the server URL as the OpenAI base and
`--training_config <yaml>` to sample under the training contract. `make test-gpu-vllm` and
`make test-gpu-sglang` run the GPU tiers against a live server (the capture path plus a behaviorally
verified weight sync: greedy output must change after a broadcast); the server must own a GPU
outside `TRAINER_CUDA_DEVICES`.

## Throughput

**Servers × TP.** Run one engine per GPU, grouped into the fewest servers that hold the model.

Two half-memory engines on one B300 deliver less at 96 sequences (5.4k tok/s) than a single engine
at the same load (6.25k), because the second engine only time-slices the GPU, and two engines
starting on one GPU at the same moment fail their memory profiling. Two servers instead of four also
halve the weight-sync fan-out.

The condition: where rollout collection dominates serving (a tool-heavy env such as `code_contests`
at gpt-oss-20b scale on 4 B300s), 2×(TP=2) and 4×(TP=1) measure the same step time, so consolidate.
On short single-turn envs serving is a larger share of the step; re-measure before consolidating.

Per-server request load is `max_concurrent_rollouts × world_size / num_servers`: every rank builds
its own actor pool and collects a full batch, under TP too, so the load scales with `world_size`
rather than `data_parallel_size` ([Ray — Pool sizing](ray.md#pool-sizing)). Dispatch is round-robin
and ignores whether a server is busy.

**Prefix caching.** The compose passes `--enable-prefix-caching` because vLLM keeps it opt-in for
hybrid families (Qwen3.5/3.6's linear-attention layers; its "Mamba cache mode 'align'" warning is
that opt-in). A multi-turn episode re-sends its whole context every turn, and without the cache each
turn's prefill takes scheduler steps away from every other request's decode.

The pause that brackets every weight update (`/pause?mode=keep`) demotes in-flight requests to
waiting and wipes the cache, so no cached prefix serves stale weights.

vLLM 0.26.0 poisons the align-mode prefix cache under speculative decoding: turn the cache off with
MTP (`VLLM_PREFIX_CACHING_FLAG=--no-enable-prefix-caching`). Measured on Qwen3.6-35B-A3B on a B300 at
TP=1, uncached prefill runs near 22k tok/s, so re-prefilling each turn costs far less than MTP
returns at RL concurrency.

**The decode step is CPU-bound at RL concurrency.** The V1 engine core is one Python thread and sits
at 100% of a core from ~48 concurrent sequences (Qwen3.6-35B-A3B on a B300: ~15 ms per step).

A trainer sharing the host (its ranks and the environments' judge sandboxes) slows every step: 22–25
tok/s per sequence against 63–66 for the same server alone. Pin each server container to
its own cores (`docker run --cpuset-cpus`, compose `cpuset:`) and the trainer to the rest.

Step latency grows with running sequences (about 14 ms + 0.22 ms per sequence here under MTP), so
per-sequence speed falls as concurrency rises, 101 tok/s at 48 running and 71 at 96, while aggregate
throughput rises sub-linearly (+44% for that doubling).

Kernel-level gains (tuned MoE tiles, another attention backend) do not show at this concurrency;
fewer steps per token do. Turning these numbers into batch sizes and timeouts:
[Sizing a run](../training-methods/grpo/async-grpo/performance.md#sizing-a-run).

**Speculative decoding (MTP).** A checkpoint that ships a multi-token-prediction head (Qwen3.5/3.6,
`mtp.*` tensors) drafts with it through
`--speculative-config '{"method":"mtp","num_speculative_tokens":2}'` (compose slot
`VLLM_SPECULATIVE_CONFIG`); the drafter loads from the same checkpoint.

Measured on Qwen3.6-35B-A3B on a B300 at 48 concurrent sequences, temperature 1.0: 59–67 → 108–112
tok/s per sequence, 2.4 tokens accepted per step, output statistics unchanged.

Sound for RL: rejection sampling applies temperature and top-p to the target logits and keeps the
target distribution, `processed_logprobs` come from those same logits, and the thinking budget is
enforced on the target under speculation. The drafter's embeddings and `lm_head` are the target's
modules, so weight sync updates them; its MTP layer keeps launch weights, which only moves
acceptance.

Not applied under speculation: `min_p`, `logit_bias`. Never `ngram` or `suffix` on linear-attention
families (open output-corruption bug; they also turn async scheduling off).
`scripts/profiling/weight_sync_transport.py` passes against an MTP server.

**Strict tool calling off** (`VLLM_ENFORCE_STRICT_TOOL_CALLING=0`, the compose default). vLLM
otherwise constrains every tool-bearing request with a structural-tag grammar, masking tokens the
trainer's log-probabilities never see masked and adding per-step grammar work on the CPU-bound engine
core. The tool parser still parses the calls; a malformed call becomes text the environment scores.

**Triton MoE tile configs.** The mandatory `--moe-backend triton` reads a per-shape tuned config
(`E=<experts>,N=<intermediate>,device_name=<GPU>.json`) from `VLLM_TUNED_CONFIG_FOLDER`, then from
vLLM's `fused_moe/configs`, and logs `Using default MoE config. Performance might be sub-optimal!`
when neither matches; the image ships none for B300 in bf16. At RL concurrency the step is CPU-bound
(above) and the tuned tiles change nothing measurable; they matter for prefill-heavy phases.

`benchmarks/kernels/benchmark_moe.py --tune` in the vLLM image writes one (needs `pip install ray`;
one batch size per visible GPU in parallel, 1920 tile configs each, 8–27 minutes, JSON written only
at the end; a single config's Triton compile failure aborts the run unless its `OutOfResources`
handler also catches `RuntimeError`). Pass `--tp-size 1` (the default of 2 halves `N`) and
`--batch-size` with the decode batch (concurrent sequences × (1 + speculative tokens)) and the
prefill chunk. The file is read once per process, so a new one needs a server restart.

**Generation volume is the step-time lever** once prefetch overlaps collection into training
(`async/prefetch_hit_rate` > 0.8): step time tracks mean episode tokens. `rollout_max_tokens` caps a
turn; on vLLM `rollout_max_thinking_tokens` caps CoT engine-side, on SGLang only the environment's
per-effort budgets price it.

**Memory.** Raise `--gpu-memory-utilization` / `--mem-fraction-static` to 0.9 (0.80 with
`isr_engine_reference`, below) when the server GPUs are dedicated; more KV cache means more
concurrent rollouts per server. Do not pass `--enforce-eager`: the in-place weight sync keeps captured
CUDA graphs valid, and CUDA-graph decode is several-fold faster on long generations.

On B200 pin the backend through the compose slot `VLLM_ATTENTION_BACKEND=FLASH_ATTN` (FlashInfer can
JIT-fail on SM 10.0). On head-dim-256 families `FLASH_ATTN` resolves to FA2 (FA4 refuses the head
size) and measured 30% slower than FlashInfer on a B300; keep auto unless FlashInfer fails.

**`isr_engine_reference` headroom.** The trainer's engine re-score sends `prompt_logprobs` requests,
and vLLM materializes an fp32 log-softmax over the vocabulary for every prefill chunk of one
(`max_num_batched_tokens × vocab × 4 B`, 8 GB at 8192 × 248k) outside its memory profile; at 0.90
the engine dies of CUDA OOM under load. Serve at `--gpu-memory-utilization` ≤ 0.80 or a smaller
`--max-num-batched-tokens` when that knob is on.

**Sync cadence.** `sync_weights_every_n_steps: 2–4` for slow environments
([Async GRPO with Environments](../training-methods/grpo/async-grpo/setup.md#weight-synchronization)).

**Weight-quantized serving is excluded** by the in-place sync ([Weight sync](#weight-sync)).

## Coverage

Support per trainer parallelism axis, async GRPO against a live server:

| Trainer axis | vLLM | SGLang |
|---|---|---|
| FSDP2 DP (dense) | supported | supported |
| TP=2 | supported | supported, gpt-oss included: its hand-sliced attention `sinks` are skipped by the dense parameter walk and sent once from the gathered-full drain, so each hub name reaches the engine exactly once |
| EP=2 (MoE) | supported | supported |
| EP=2 + ETP=2, EP=2 + TP=2, EP=4, with or without LoRA / expert LoRA | supported | supported |
| Expert LoRA (EP=2), resume included | supported | supported |
| Undistributed MoE (`ep_group_size == 1`, EP wrappers present) with multi-server serving (2×TP=2, 4×TP=1), expert sync and R3 rollout replay | supported | supported |
| Trainer and server on different nodes over EFA ([Servers on other nodes](#servers-on-other-nodes-efa)) | supported, a trainer spanning two nodes (DTensor experts over the fabric) against a server on a third included | supported |

An SGLang server under its own expert parallelism (`SGLANG_TP=2 SGLANG_EXTRA_ARGS="--ep-size 2"`)
takes the sync for Qwen3 MoE: the loader keeps its local experts and drops the rest. gpt-oss cannot
be served that way on 0.5.17: its fused expert loader crash-loops at start under `--ep-size`
(`_load_w2`, local against global expert count), before any sync.

## Troubleshooting

| Symptom | Cause → fix |
|---|---|
| `/health` is 200 but every completion hangs after a trainer died mid-sync | The trainer was killed with SIGKILL (`docker rm -f`), which skips the `atexit` resume the client registers; the engine stays paused. POST its resume route (`/resume` on vLLM, `/continue_generation` on SGLang) or restart it |
| Group formation times out, "1/2 clients joined" | Bridge network hides the rendezvous port → `network_mode: host` on both containers |
| First sync takes minutes, `/health` unanswered | One-time NCCL group init per server; the engine pauses mid-update — wait |
| 400 on every rollout (vLLM) | a thinking budget the server cannot take: set `VLLM_REASONING_PARSER`, **and** `VLLM_USE_V2_MODEL_RUNNER=0` (Model Runner V2 does not implement `thinking_token_budget`). Rollouts otherwise return zero tokens and the run trains on all-masked batches |
| Run completes with flat zero gradient | Missing/wrong tool parser on a native-tool env: calls stay text, no `tool_calls`, every episode reward 0 (parser table above; ReAct envs parse text and are immune to a *mismatch*) |
| Log-ratio drifts on SGLang while the server log stays clean | No server-side signal exists: the MoE loaders skip unmapped expert names before their `not found in params_dict` warning → do not read a clean log as proof of a landed sync; the construction gates are the guard |
| `RoutedExperts: Failed` (vLLM log) | Layerwise-reload patch missing → expert syncs silently reverted; rebuild `vllm-server` |
| `/init_weight_transfer_engine` answers 500 (`NCCL error: unhandled cuda error`) while `/health` is 200 | Re-init patch missing → the engine strands a communicator per trainer connection until the GPU runs out; rebuild `vllm-server` and recreate the server container |
| First sync hangs at the first collective after the group formed, both ends idle | The two containers drive different transports (`NCCL_NET` / `NCCL_NET_PLUGIN` differ) or different `aws-ofi-nccl` + libfabric builds (an upstream server image, a host-installed plugin) — the pair forms the group and then hangs → both ends from the Halo images with the same recipe (compose EFA overlay + `make ... EFA=1`, or both on the no-fabric defaults); `scripts/profiling/weight_sync_transport.py` reports the transport and build each side formed on |
| `ncclP2pImportShareableBuffer ... invalid argument` on the first update, `The full weights of the ModelRunner are partially updated` (SGLang) | cuMem off on the server only — SGLang sets `NCCL_CUMEM_ENABLE=0` unless it is pre-set → `NCCL_CUMEM_ENABLE=1` in the server container (the compose default); restart the server, it holds a half-written model |
