# Rollout Servers

A rollout server is a **separate inference container** — vLLM or SGLang — that generates the
model's responses (rollouts) during RL. The trainer sends prompts to it over HTTP and pushes
updated weights back over a per-server NCCL group. Every on-policy GRPO trainer needs one:
[Online](../training-methods/grpo/online-grpo.md) and
[Environmental GRPO](../training-methods/grpo/environmental-grpo.md).

The trainer never imports either engine — their torch/transformers stacks cannot load in the
training process — so it speaks to them through the vendored client in `src/distributed/nccl/`.
Select the engine with `rollout_backend: vllm | sglang` (env-GRPO only — Online GRPO is
vLLM-only by construction); every other server knob is the same for both engines:

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
(enforced), and an entry's keys are read by name with no per-entry validation — a misspelled
`group_port` silently falls back to `vllm_group_port + index`. The top-level keys are engine-neutral
and the parser renames none: any other spelling raises the unknown-key error. Trainer-side rollout mechanics — prefetch and its one-step
staleness, sync cadence, trajectory-length knobs — stay on the
[Environmental GRPO](../training-methods/grpo/environmental-grpo.md#nccl-weight-synchronization) page.

| | vLLM 0.26.0 | SGLang 0.5.17 |
|---|---|---|
| Trainers | online, env-GRPO | env-GRPO |
| Sampled-token ids | `--return-tokens-as-token-ids` (server flag) | per-request `return_meta_info` |
| IS-reference logprobs (the sampling distribution's) | `--logprobs-mode processed_logprobs` (server flag; the default `raw_logprobs` is pre-temperature and refused at any `rollout_temperature` ≠ 1) | default (post-temperature, pre-nucleus; keep `SGLANG_RETURN_ORIGINAL_LOGPROB` unset) |
| [R3 routing replay](../training-methods/grpo/environmental-grpo.md#off-policy-mismatch-and-stability-knobs) | `--enable-return-routed-experts` + `--moe-backend triton` | `--enable-return-routed-experts` + `--moe-runner-backend triton` |
| Thinking budget (`rollout_max_thinking_tokens`) | enforced engine-side with a reasoning parser and `VLLM_USE_V2_MODEL_RUNNER=0`; harmony-disabled gpt-oss arms it off the toolkit plugin's marker ([GPT-OSS](../models/gpt-oss.md#serving-for-grpo-vllm)) | rejected at config time |
| Expert layout on sync | whatever the family's own `gather_expert_state_dict` emits — per-expert (Qwen3 MoE, GLM-4/Laguna, Bailing, LFM-2) or fused where that is the family's base gather (Qwen3.5/3.6, Gemma 4); 0.26.0's expert loader reads both. A family whose hub namespace differs from its module tree (Step-3.7's per-layer `moe.gate_proj`/`up_proj` stacks) is re-spelled through transformers' save-side revert, so the engine receives its hub keys | the same layouts, read by 0.5.17's per-family loaders; the families its loaders cannot take an update for are listed under [Which families each engine serves](#which-families-each-engine-serves) |
| Trainer expert distribution ([EP/ETP](../reference/glossary.md#parallelism)) | supported | supported |

Use vLLM unless you need SGLang specifically: it is the only backend for Online GRPO, the only
one that enforces a thinking budget, and its 0.26.0 loaders take the sync for two families SGLang's
do not.

## Weight sync

Both engines receive the **full model** every sync — merged LoRA touches ~95% of bytes, so there is
no delta path (~42 GB at 20B, ~1–2 s steady over loopback). Re-assembling the sharded weights is a
collective: every rank takes part, none may skip. One rank then owns the clients and does the
sending — the **forwarding rank** (global main; TP-rank 0 under TP). Its sends sit *between* those
gathers, so a failure there (an engine 500, a refused tensor, a host OOM on the snapshot) is
recorded rather than raised: the forwarding rank stays in every remaining collective and the whole
world raises together at the end of the sync, naming the failing rank and its cause.

The gather **reshards the FSDP2 modules first**. A forward leaves their transient unsharded params
registered while the optimizer steps the shards, so reading the registered params would ship a
policy one optimizer step behind — and fold a PEFT merge into a copy the next unshard discards.

The push is **streamed, not buffered**: both engines take an update as a sequence of declared chunks
inside one quiesce (`/start_weight_update` … N × `/update_weights` … `/finish_weight_update` on vLLM,
N × `/update_weights_from_distributed` between `/pause_generation` and `/continue_generation` on
SGLang), so the forwarding rank sends each chunk as the gather fills it and stages one chunk on its
sync GPU — not one model (~800 GB at 400B), and not in host memory: a chunk that transits pinned
host memory is copied out and back over PCIe before the NIC sees it — 19-24 GB/s against 53-80 GB/s
staged on the device. The chunk is cut before the budget
(`HALO_WEIGHT_SYNC_CHUNK_MB`, 1 GiB) is exceeded; a tensor above it is a chunk of its own. The
forwarding rank's peak during a sync is therefore the assembled EP layer being sent (the largest
rank-local allocation, ~28 GB for one 397B layer), the staged chunk, the snapshot of the largest
tensor, and the engine path's buffers (two vLLM packed buffers, one SGLang arena), each grown to
the largest chunk seen. In multi-server mode (`rollout_server_configs`)
one snapshot per parameter is shared across all servers and each chunk goes out to every server on
concurrent threads, released once they all have it; the threads share the forwarding rank's GPU,
NICs and process, and the fan-out costs the sum rather than the max — two servers each push at
27 GB/s over EFA, half of one server's rate — so the per-sync cost grows with the server count.
The trade is that a chunk cannot be replayed: a server that fails **after** its first chunk is
reported rather than reconnected — the trainer does not hold what already landed. One that fails
before any chunk went out (an engine restarted between syncs, the common case) is still recovered by
the reconnect + re-flush. Each client owns its own NCCL connection to its server on a `group_port`
bound on the *trainer* host
([group ports](../training-methods/grpo/environmental-grpo.md#nccl-weight-synchronization)).

**The quiesce spans the streaming, not just the final broadcast.** The update opens with the first
full chunk — ~1 GB into the gather — and closes when the last one lands, so a server stops serving
for as long as the gather runs: minutes at 397B, and for every server at once outside the
[rolling path](../training-methods/grpo/environmental-grpo.md#single-server-vs-multi-server). The
client pauses vLLM with `/pause?mode=keep`: in-flight generations — the prefetched rollout round —
freeze and resume under the new weights on `/resume`, the one-step staleness the sampling-logprob IS
ratio corrects. vLLM's own default is `abort`, which hands every in-flight request back as a fragment
with an ordinary stop reason; once the training pass is shorter than a rollout round that is most
long turns, every step. SGLang's is `/pause_generation {"mode": "abort"}` (its post-update cache
flush asserts an idle scheduler), so there in-flight generations are dropped across that window.

**An interrupted mid-stream sync leaves that server unusable.** The engine then holds neither the
old policy nor the new one, and vLLM's layerwise reload materializes a layer whose tensors straddled
the boundary from *uninitialized* storage while it waits for the rest. The abort therefore leaves
that engine **paused** instead of resuming it, refuses every later sync on that client, and logs
`RESTART the … server`. Restart the container — the trainer kept no copy of what landed and cannot
repair it.

The first sync takes minutes (one-time NCCL group formation per server) and the server stops
answering `/health` mid-update — an update in progress, not a hang. Every sync pauses the engine
and resumes it after: `/pause?mode=keep` … `/resume` on vLLM, with the broadcast itself bracketed by
`/start_weight_update` … `/finish_weight_update` (the layerwise reload phase, closed on every path);
`/pause_generation` … `/continue_generation` on SGLang. The broadcast is packed (~1 GB buffers,
double-buffered) on vLLM and typed 1 GB chunks on SGLang.

On vLLM, each client owns **one persistent CUDA stream pair** for the pack uploads, because PyTorch's
caching allocator keeps freed blocks in per-stream pools: a fresh stream per sync would strand one
payload of reserved memory every sync. The forwarding rank's steady state is therefore its training
footprint plus about one sync of pack buffers. The SGLang client keeps one persistent send stream
for the same reason and drops its arena at the end of every sync, so between syncs that memory is
the allocator's rather than pinned at the largest chunk's size.

Each rank logs a `[mem rankNN] weight-sync pre/post` line per collective sync to watch exactly
this ([Debugging](../reference/debugging.md#3-gpu-memory-profiling)); `reserved` far above
`peak_alloc` on the forwarding rank means stranded allocator pools.

**Served weights must stay in checkpoint layout.** The sync writes bf16 checkpoint-layout tensors
into the server's parameter storage in place, so any load-time transformation of that storage
silently corrupts every later update: on vLLM the auto-selected Blackwell MoE backends
(`FLASHINFER_TRTLLM`/`CUTLASS`) repack expert weights — `--moe-backend triton` is **required** for
MoE RL (its kernels read the checkpoint layout directly). On SGLang the `triton`/`triton_kernel`
runners load bf16 unpacked; `flashinfer_trtllm` repacks and must not serve RL.

The same rule excludes quantized serving: a weight-quantized engine stores transformed tensors the
broadcast cannot update, and `--quantization` on a MoE model fails loudly against the reload patch
below. Serve bf16 — for gpt-oss that means the **BF16** checkpoint, not the stock MXFP4 one
(`openai/gpt-oss-20b`, whose quantization the engine auto-detects with no flag to fail on). Its MXFP4
expert loader branches on packed blocks and on biases with no branch for a bf16 expert tensor, so
every synced expert weight is dropped while the biases land — the one quantized path that does **not**
fail loudly, leaving the trainer only a slow log-ratio drift.

**The vLLM layerwise-reload patch** (`docker/vllm/patches/vllm_layerwise_reload_patch.py`, baked into
the image, applied via `sitecustomize` in the API server and every engine-core worker) closes a silent
corruption path. vLLM's reload moves each layer to the meta device and wraps its `weight_loader`s, but
model code that writes weights with a direct `param.copy_()` (gpt-oss experts and attention `sinks`)
lands on a meta tensor as a no-op, and the reload then re-registers the **saved** tensors — every sync
reverting those weights while `/update_weights` returns `200 OK`.

The patch excludes the affected classes from the reload lifecycle (`SKIP_LAYER_NAMES`: `RoutedExperts`
/ `FusedMoE`, `OAIAttention`, and `Gemma4Router`, whose partial load would materialize an uninitialized
buffer into live memory). The class set is version-dependent and asserted against the installed vLLM at
build, so an upstream refactor fails the image build.

Missing, it shows as `RoutedExperts: Failed to load weights` per expert layer per sync in the server
log, with the trainer's `sampling/logratio_mean` drifting monotonically negative
(`YaRNScalingRotaryEmbedding: Failed to load weights` is benign — no loadable weights). Do not override
`PYTHONPATH` at `docker run`: dropping `/opt/nccl_compat` kills weight-transfer init
(`No module named 'src'`).

**The vLLM weight-transfer re-init patch**
(`docker/vllm/patches/vllm_weight_transfer_reinit_patch.py`, applied through the same `sitecustomize`
hook) destroys the engine's previous NCCL communicator before `/init_weight_transfer_engine` builds
the next one, and again on engine shutdown, so one server outlives any number of trainer connections.
Stock vLLM 0.26.0 only drops the reference and `PyNcclCommunicator` has no `__del__`, so each
connection strands a live communicator — ~633 MiB of device memory per connection on every
engine-core worker — until `ncclCommInitRank` fails while `/health` still answers `200`. The trainer
half is symmetric: `close_communicator()` aborts its own communicator instead of dropping it. Both
halves are asserted by `tests/gpu/trainers/grpo/test_vllm_weight_transfer_reinit.py`.

Checkpoint layout and expert un-fuse rules live in
[Checkpoints](../reference/checkpoints.md#serving-on-vllm-sglang).

Which families each backend accepts for RL is gated trainer-side at construction. Inkling, GLM-5
Next and Cohere2 MoE declare `_supports_weight_sync = False` and are refused on both engines; every
other refusal is an engine fact on that engine's client
([Which families each engine serves](#which-families-each-engine-serves)). A bnb-quantized (QLoRA)
base is refused on both as well.

**Hub-namespace families.** The sync forwards every tensor under the key a gathered checkpoint would
carry. Where the live module tree and the hub checkpoint differ, the rewrite is derived, not
tabulated: Laguna's `_EXPORT_KEY_RENAMES` pairs, and — for a family declaring
`_EXPORTS_HUB_NAMESPACE` (Step-3.7 Flash) — transformers' own save-side conversion revert, the
reversed `WeightRenaming`/`WeightConverter` entries `save_pretrained` applies. One-to-one renames
stream tensor by tensor; a tensor a reverse converter claims (a fused `gate_up_proj` the hub stores
split, a vision tower's q/k/v the hub stores fused) is held until its sources are complete, since the
engine loads one tensor at a time. Any family whose hub checkpoint sits behind such a conversion joins
by declaring the flag on its EP layer once a pinned engine serves it (`_supports_weight_sync` stays
off for GLM-5 Next and Inkling because none does). Per-family server flags:
[Step-3.7](../models/step3p7.md#serving-for-grpo-vllm).

**GptOss needs live attention sinks.** `reset_sinks: true` under `flash_attention_2` rebinds
`attn.sinks = None`, and the sync forwards `named_parameters()` only, so nothing is ever pushed for
those slots and the server keeps generating with the pretrained sinks against a sink-free trainer —
permanently off-policy with no error at sync time. `validate_weight_sync_support`
(`src/trainers/grpo/rollout/weight_sync.py`) refuses that shape at construction: on-policy GptOss
needs `reset_sinks: false` with a sink-carrying implementation (FA4 or eager), which is what the
shipped GRPO configs set. The same validator refuses two more shapes for the same reason — state the
sync cannot carry: `train_sinks: true` (sinks that change every step, SFT-only), and an enabled
router bias-update balancing bias, adopted or transient, which the parameter-only payload never
pushes. The shipped GRPO scripts downgrade `moe_balancing` to `none` themselves.

### Which families each engine serves

Both engines' loaders read a family's experts in the layout its own `gather_expert_state_dict`
emits — per-expert tensors for Qwen3 MoE, GLM-4 MoE Lite, Laguna, Bailing and LFM-2, the fused pair
for Qwen3.5/3.6 and Gemma 4, GptOss's interleaved pair — so the sync carries one layout per family
on either engine. What differs per engine is which families its pinned release can take an online
update for at all. Each client declares those with the loader fact (`UNSERVABLE_MODEL_TYPES`), and
`validate_weight_sync_support` refuses the pair at construction, quoting it. A family no gather can
spell on any engine stays a family flag (`_supports_weight_sync`: Inkling, GLM-5 Next, Cohere2 MoE).

| Family (`model_type`) | vLLM 0.26.0 | SGLang 0.5.17 | Loader fact |
|---|:--:|:--:|---|
| Mistral4 | ✗ | ✗ | neither registers a class ([Mistral4](../models/mistral4.md#serving)) |
| Ling 3.0 (`bailing_hybrid`) | ✗ | ✗ | no class for `BailingMoeV3ForCausalLM` |
| Ring (`bailing_moe_linear`) | ✗ | ✗ | the checkpoints declare `BailingMoeLinearV2ForCausalLM`; both register `BailingMoeV2_5ForCausalLM` |
| Zaya | ✗ | ✗ | vLLM ships no native class; SGLang's loader reads the pre-transformers-5.14 per-expert checkpoint (`zaya_block.experts.local_experts.N.linear_fc1`) |
| DeepSeek-V4 | ✗ | ✗ | vLLM's loader targets the fp8/fp4-packed release layout; SGLang's maps per-expert `w1/w3/w2` |
| Laguna | ✓ | ✗ | SGLang's `load_weights` asserts every routed-expert tensor of every sparse layer per call |
| Step-3.7 (`step3p7`, `step3p5`) | ✓ | ✗ | `Step3p5ForCausalLM.load_weights` asserts full parameter coverage per call |

Every other family the trainer trains — dense families, GptOss, Qwen3 MoE, Qwen3.5/3.6 MoE, GLM-4
MoE Lite, Gemma 4, Ling 2.0, LFM-2 MoE — syncs on both engines, expert distribution included
([CI](ci.md) has the per-family pass).

Three SGLang 0.5.17 loader facts shape its image and its client:

- **Routers that cache a derived form.** Upstream, the GLM-4 gate caches an fp32 copy of its
  weight at the first forward and never re-reads the parameter, and the Gemma 4 router folds `scale`
  into its norm once, behind a latch: a synced router weight lands in the parameter while routing
  keeps the launch values, with no error. `Dockerfile.sglang` applies
  `docker/sglang/patches/patch_sglang_weight_updates.py` (the gate reads its fp32 weight live, a
  load of `scale` releases the latch); the script asserts its pre-images before rewriting and its
  post-images after, at build, and stays in the image (`/opt/halo/`) so `--verify` re-checks a
  running container.
- **Fused a-projection halves.** The MLA loaders (GLM-4 MoE Lite here) concatenate `q_a_proj` and
  `kv_a_proj_with_mqa` from a cache local to one `load_weights` call — one chunk — and drop a half
  that arrives alone. The client declares the pair (`CO_LOADED_PARAM_GROUPS`) and the chunker keeps
  it in one chunk, deferring the first half when the byte budget would cut between them; a pair
  still incomplete when the sync closes refuses the close.
- **The triton runner.** The `flashinfer_trtllm*`, aiter and quantized runners repack expert weights
  after the load, and an online update writes the canonical layout into the repacked buffer.
  `SGLANG_MOE_RUNNER_BACKEND=triton` (the compose default) is the runner whose weights an update
  reaches unchanged; R3 capture needs it too.

The sync ships hub names and full unsharded tensors into the engine's own `load_weights` mapping;
each TP rank narrows its slice, and under `--ep-size` the loader keeps its local experts and drops
the rest. An expert name that mapping does not cover leaves **no server-side signal**: the MoE
loaders `continue` on an unmatched `mlp.experts` name *before* the `not found in params_dict`
warning (reachable only from the dense loaders), the update still returns `200 OK`, and the engine
keeps serving its launch-weight experts under a freshly synced router; the server tier's
expert-only round is what catches it.

The trainer-side construction gate is the whole guard. It reads the family's contract off a live EP
wrapper, or — when a run has none (`use_grouped_gemm: false` at `ep_size: 1`) — off the `model_type`
registry. Expert distribution (EP, ETP) is accepted on both engines: the sync group is ordinary NCCL
beside DeepEP's.

## vLLM

`Dockerfile.vllm` builds `vllm-server:0.26.0` with the native NCCL weight-transfer engine, the
layerwise-reload patch, R3 routed-experts capture (base64-npy `routed_experts` per completion
choice), and `nvidia-nccl-cu13` installed at `uv.lock`'s exact pin — the same NCCL the training
images run — with `VLLM_NCCL_SO_PATH` baked to that wheel so the base image's older system copy
can never win the soname race. A skew fails `ncclCommInitRank` at `/init_weight_transfer_engine`
(HTTP 500, `NCCL error: internal error`) — rebuild the image after any lock bump of the pin. 0.26.0
is the last vLLM release on torch 2.11 — the training image's torch and NCCL generation; 0.27 moves
to torch 2.13, whose NCCL does not match that pin. The image also installs the EFA userspace the
training image runs (`docker/efa/install_efa_userspace.sh`), so the group can ride EFA from a
trainer on another node ([Servers on other nodes](#servers-on-other-nodes-efa)).

### Config-schema parity {#config-schema-parity}

The server parses every checkpoint with **its** transformers, pinned to the 5.14 line — one line
below the training image's 5.16 (`Dockerfile.vllm` asserts the pin at build). **Gemma 4 is what pins
that line.** vLLM's Gemma 4 model code (0.25.1 through 0.28.0 alike) reads the 5.14 config schema
(flat `global_head_dim` / `num_global_key_value_heads`, a global `num_attention_heads`), which 5.16
folds into `per_layer_config` and raises `AmbiguousGlobalPerLayerAttributeError` on vLLM's
`get_head_size` — a 5.16 server makes Gemma 4 unservable on every one of those vLLM versions — so
toolkit exports are written in the flat form.

**Step-3.7 is a different constraint**, not a dialect: this transformers has no `step3p7` class at
all and reads the family only through the release's `auto_map` modules, which its release config
loads cleanly on either line. Its exports therefore carry the source repo's own config schema and
those modules ([Checkpoints](../reference/checkpoints.md#what-gets-saved)).

`docker/vllm/parity/check.py` runs at image build over one `config.json` per fixtured family — what
the toolkit exports for it, built from the tiny roster config in `tests/common/models.py`, through
the vendor config module the fixture ships where transformers carries no class for the family
(Bailing/Ling). Only the source-schema carry is pinned to a release config at a fixed revision — the
one thing a tiny config cannot express — so everything else re-renders offline (`generate.py`,
regenerated in the training image). The roster is derived from the EP
registry: every family whose layer class admits weight sync owes a fixture, because the server has to
parse that family's checkpoint before a single tensor can be synced into it — which is how a family
no pinned engine can load (Mistral4) surfaces as a refusal rather than a dead sync.
`tests/cpu/checkpoint/test_vllm_parity_fixtures.py` fails when the roster or the rendered fixtures
drift. Each rewrite also ships its negative control under `unparseable/` — the folded Gemma 4 form,
the native-schema Step-3.7 export — which must still be refused: a transformers bump on either side
that breaks the schema fails the build, not the first live sync.

`docker-compose.vllm.yml` runs it with `network_mode: host` + `ipc: host`: to form the NCCL group the
two sides first find each other on an ephemeral trainer port (the rendezvous), which a bridge network
would hide, and group formation then times out at "1/2 clients joined".

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
| `VLLM_CUDA_DEVICES` | `7` | Server GPUs — must exclude the trainer's (a rank cannot broadcast to itself). Selects via `CUDA_VISIBLE_DEVICES` inside a container that sees every GPU: hiding devices from the container instead (`--gpus device=N`) breaks the cross-container NCCL P2P import of the trainer's buffers (`Cuda failure 101 'invalid device ordinal'`, `500` on `/init_weight_transfer_engine`) |
| `VLLM_TP` | `1` | `--tensor-parallel-size` |
| `VLLM_GPU_MEM` | `0.85` | `--gpu-memory-utilization` |
| `VLLM_MOE_BACKEND` | `triton` | Keep `triton` for MoE RL ([Weight sync](#weight-sync)) |
| `VLLM_ENABLE_R3` | *(unset)* | Any non-empty value adds `--enable-return-routed-experts` (R3 capture); the `triton` MoE backend is the one the capture hook reaches |
| `VLLM_ATTENTION_BACKEND` | *(unset = auto)* | `--attention-backend`. GLM-4 MoE Lite (MLA) on Blackwell needs `CUTLASS_MLA`: the auto-selected FlashInfer MLA decode kernel rejects its head config at graph capture ([MLA backend](../reference/checkpoints.md#serving-on-vllm-sglang)) |
| `VLLM_TOOL_PARSER` | `hermes` | `--tool-call-parser`; per-family values below |
| `VLLM_TOOL_PARSER_PLUGIN` | *(unset)* | `--tool-parser-plugin` path (gpt-oss uses the baked `/opt/gpt_oss_text_tool_parser.py`) |
| `VLLM_CHAT_TEMPLATE` | *(unset)* | Set to the SAME `.jinja` the trainer's `chat_template:` uses; the file must be visible inside the server container |
| `VLLM_REASONING_PARSER` | *(unset)* | Required when training sets `rollout_max_thinking_tokens` |
| `VLLM_REASONING_PARSER_PLUGIN` | *(unset)* | `--reasoning-parser-plugin` path for families without a built-in parser |
| `TRAIN_IMAGE` | `halo:blackwell` | Image the compose file's optional trainer service runs |

Flags the compose file already sets that are load-bearing for RL:

- `--weight-transfer-config '{"backend": "nccl"}'` — enables the transfer engine.
- `--return-tokens-as-token-ids` — `train_on_sampled_tokens` (default on) recovers the sampled ids
  from the logprobs, which vLLM only spells out under this flag. Without it every turn falls back to
  re-tokenizing a chat-template re-render: one warning, then a whole run training on tokens the
  engine never sampled.
- `--logprobs-mode processed_logprobs` — the reported logprobs are the sampling distribution's
  (temperature and top-p applied). The default `raw_logprobs` are pre-temperature: the trainer scores
  its log-probs at `rollout_temperature` and divides by these, so at any temperature ≠ 1 every IS
  weight is π^T / π^1 — tilted toward improbable tokens above 1 (entropy climbs step over step) and
  toward confident ones below (entropy collapses) — while `sampling/is_ratio_mean` still reads ≈ 1.
  The trainer probes each server at startup (temperature 2 must halve the top-1/top-2 gap) and
  refuses a raw server whenever `rollout_temperature` ≠ 1. Under this mode a top-p < 1 also
  renormalizes every uncertain position over its nucleus, which the trajectory geometric band reads
  as drift: `rollout_top_p: 1.0` whenever `isr_geo_band_min/max` is set (also probed and refused).

`--max-model-len` is left unset — the server serves the model's native context window. The trainer's
startup probe reads it off `/v1/models` and **raises** when `max_prompt_length` plus one turn's
generation exceeds it; the worst-case multi-turn budget only warns, since a rollout growing past the
window OOMs the training forward before the fail-on-overflow check.

R3 runs add one flag, `--enable-return-routed-experts` (`routing_replay: rollout`) — without it the
trainer raises at the first capture. The compose `command:` has no interpolation slot for it, so edit
it or launch `vllm serve` directly, as the R3 example config headers instruct. The FlashInfer
monolithic MoE kernels bypass the capturer and return all-zero expert ids, hence the triton backend.

`VLLM_USE_V2_MODEL_RUNNER=0` is not a serve flag but an env var the compose file already passes
through from your shell (`VLLM_USE_V2_MODEL_RUNNER=0 docker compose -f docker-compose.vllm.yml up`).
Any run sending thinking budgets needs it, R3 or not: Model Runner V2 rejects
`thinking_token_budget` with a 400 on every request, so each rollout errors instead of generating
(zero tokens, `episode/error_rate` 1). Spell it `0` or `1` and nothing else — vLLM reads it with
`int()`, so `false` or an empty value (an empty key in the repo-root `.env` counts) kills the server
at startup with a bare `ValueError`.

Tool parsers by family — for **native-tool** envs (`code_contests`, `swe`, `mcp`, `qa_search`,
open-book `exam_qa`) the absence of the right one is silent and fatal to RL: calls stay text, no
`tool_calls`, every episode reward 0, flat zero gradient. ReAct envs parse actions from the
response text, so a mismatched parser costs them nothing — but a *missing* one still 400s, since
the trainer sends `tools` for any env with a tool registry:

| Family | `--tool-call-parser` |
|---|---|
| Qwen3 / Qwen3.5 / 3.6 | `qwen3_xml` (hermes does NOT parse their XML calls) |
| GPT-OSS | bundled plugin `gpt_oss_text` via `VLLM_TOOL_PARSER_PLUGIN`; reasoning plugin `/opt/gpt_oss_reasoning_parser.py`, parser `openai_gptoss` ([GPT-OSS](../models/gpt-oss.md#serving-for-grpo-vllm)) |
| GLM-4 | `glm45` / `glm47` |
| Gemma 4 | `hermes`; with a thinking budget (`rollout_max_thinking_tokens`, or an env's per-effort `thinking_tokens` profile) also `VLLM_REASONING_PARSER=gemma4` and `VLLM_USE_V2_MODEL_RUNNER=0`, else every request 400s |
| most others | `hermes` (`<tool_call>` XML) |

`docker-compose.vllm.yml` defaults **both** containers to the no-fabric recipe (`NCCL_IB_DISABLE=1`
and `NCCL_NET=Socket` on each, `NCCL_P2P_LEVEL=NVL` on the server), so on a host without a fabric
the cross-container group takes NVLink + sockets on both ends by declaration rather than by each
side's own fallback — two containers that land on different nets form the group and hang at the
first collective ([Troubleshooting](#troubleshooting)). On an EFA host layer
`docker-compose.vllm.efa.yml` over it ([Servers on other nodes](#servers-on-other-nodes-efa)):
`NCCL_NET` is process-global, so a trainer left at `Socket` there sends every collective over TCP
and breaks DeepEP.

Steer the transfer group with `VLLM_GROUP_HOST`: it names the trainer address the server dials back
to and touches nothing else. Unset, the client resolves loopback for a local
server and the default-route NIC otherwise, so a same-host compose stack needs no value; a
`rollout_server_configs` entry's `group_host` overrides it for that one server. `NCCL_SOCKET_IFNAME`
is not a transfer-group knob; it is process-wide, filtering the interfaces the default process group
and DeepEP pick too, so on a multi-homed host set it only to an interface every collective can use.

## SGLang

`Dockerfile.sglang` builds `sglang-server:0.5.17` to align NCCL: upstream's wheel trails
`uv.lock`'s exact pin (what the training images run), and weight sync needs both ends on one
runtime. The build bumps the wheel, installs the EFA userspace the training image runs
(`docker/efa/install_efa_userspace.sh`), applies the loader patches an online update needs
(`docker/sglang/patches/`, [Which families each engine serves](#which-families-each-engine-serves)),
and asserts the weight-sync routes, request schemas, and rendezvous convention still exist, so an
upstream refactor fails the build instead of a training run. Serving-only use can run upstream
directly (`SGLANG_IMAGE=lmsysorg/sglang:v0.5.17`); weight sync needs this image. 0.5.17 is
the last SGLang release on torch 2.11 — the training image's torch and NCCL generation; 0.5.18
moves to torch 2.13, whose NCCL does not match the pin weight sync needs on both ends. 0.5.19's
weight updater is byte-identical and it keeps the same `NCCL_CUMEM_ENABLE=0`-unless-set default;
its `--moe-a2a-backend deepep_v2` (DeepEP's ElasticBuffer engine) is allowlisted to
`DeepseekV3ForCausalLM`, `DeepseekV4ForCausalLM` and `Qwen3MoeForCausalLM` and forces
`--moe-runner-backend deep_gemm`, a runner an online update does not reach (the triton runner
above), so it adds nothing to the recipe and the pin stays at 0.5.17.

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
| `SGLANG_CUDA_DEVICES` | `7` | Server GPUs — must exclude the trainer's |
| `SGLANG_GPU_MEM` | `0.85` | `--mem-fraction-static` |
| `SGLANG_TOOL_PARSER` | `auto` | `--tool-call-parser`. `auto` reads the parser off the chat template — its harmony channel-marker rule resolves gpt-oss to the harmony parser on 0.5.14 and 0.5.17 alike, so no per-family pin is needed. Override only for a template the detector does not cover |
| `SGLANG_MOE_RUNNER_BACKEND` | `triton` | `--moe-runner-backend`; keep `triton` for MoE RL and for R3 capture |
| `SGLANG_ENABLE_R3` | *(unset)* | Any non-empty value adds `--enable-return-routed-experts` (R3 capture) |
| `SGLANG_REASONING_PARSER` | *(unset)* | `--reasoning-parser` (`gpt-oss` for harmony models) — separates reasoning from content in the response |
| `SGLANG_CHAT_TEMPLATE` | *(unset)* | `--chat-template`: the SAME `.jinja` the trainer's `chat_template:` uses; the file must be visible inside the server container |
| `SGLANG_ATTENTION_BACKEND` | *(unset)* | `--attention-backend`. `triton` for GLM-4 MoE Lite on Blackwell: its MLA head size has no kernel in the engine's default backend and the server exits at start (`Unsupported head dimensions`) |
| `SGLANG_TRUST_REMOTE_CODE` | *(unset)* | Any non-empty value adds `--trust-remote-code`, which the Bailing (Ling 2.0) repos need — their modeling code ships in the checkpoint |
| `SGLANG_EXTRA_ARGS` | *(unset)* | Further launch flags, verbatim. `--enable-deterministic-inference` for GLM-4 MoE Lite: under the triton attention backend its greedy logits differ between a prefill and a prefix-cache hit of the same prompt, which the tier's zero-noise baseline probe refuses |

`SGLANG_GROUP_HOST` belongs to the **trainer** (the `VLLM_GROUP_HOST` equivalent, separate because
the engines can sit on different hosts).

Engine behavior under RL:

- **Sampled ids** arrive per request: SGLang's OpenAI `logprobs` reports tokens as text, so the
  trainer sets `return_meta_info` + `return_prompt_token_ids` and reads
  `choice.meta_info.output_token_logprobs[i][1]`. No server flag.
- **Logprobs are post-temperature, pre-nucleus** by default (`sampler.py` divides the logits by the
  temperature before the log-softmax the reported values come from; top-p renormalizes only the
  sampling probabilities), so they are the IS reference the trainer expects at any
  `rollout_temperature`, and a top-p < 1 leaves the geometric band untouched. Do not set
  `SGLANG_RETURN_ORIGINAL_LOGPROB`: it switches to raw values, and the startup probe refuses them.
- **A length cut-off is a `stop_reason`**, not a `finish_reason`. `get_finish_reason`
  (`src/inference/response.py`) reads `finish_reason or stop_reason`, so the rollout path grades an
  engine-truncated completion as truncated rather than as an answer.
- **R3**: serve with `--enable-return-routed-experts --moe-runner-backend triton` — the fused
  runners (`triton_kernel`, flashinfer; auto-selection picks one for most MoE shapes) bypass the
  capture hook and return nothing. The trainer opts in per request and decodes the wire format
  (response-level `sglext.routed_experts`, base64 raw int32) by the model's own layer/top-k counts.
  Rows cover the full sequence, so prompt spans replay too. The engine's capturer is per family:
  it serves GptOss, Qwen3 MoE, Qwen3.5/3.6 and GLM-4 MoE Lite, exits at start for Gemma 4 (whose
  config carries no `num_experts_per_tok`; the family has no routing replay on either engine) and
  raises at the first capture for Bailing — serve those without `SGLANG_ENABLE_R3`.
- **`rollout_max_thinking_tokens` is rejected at config time** for every model: SGLang ignores
  unknown request fields, and the trainer wires neither of its budget mechanisms (the
  custom-logit-processor path needs a per-model class; the strict-thinking grammar needs a detector
  exposing `think_excluded_tokens` — for harmony models neither exists server-side). Steer with the
  environment's `reasoning_effort`.
- **`--dp-size > 1` needs `--enable-dp-attention`**, or the client refuses at group formation: plain
  DP replicas each restart `tp_rank` at 0, so their workers collide on `rank_offset + tp_rank` in the
  update group and no sizing can address them. The client reads the layout off `/server_info`.
- **`--enable-torch-compile` must stay off under R3 capture**: no step-time gain, and capture ×
  compile produces isolated catastrophic log-ratio rows (the IS veto/geo-band masks them — the
  trust region absorbing an engine numerics fault).

### NCCL transport (SGLang)

The group is ordinary NCCL, the same as vLLM's: on one host it takes CUDA IPC between the two
containers (`P2P/CUMEM`, NVLink), across nodes the fabric. The one engine-side requirement is
**cuMem parity**: SGLang's engine entry point sets `NCCL_CUMEM_ENABLE=0` process-wide unless the
variable is already set, the trainer's NCCL has cuMem on, and the mismatch fails the first
cross-container buffer import (`ncclP2pImportShareableBuffer ... invalid argument`, then
`Cuda failure 'invalid argument'. The full weights of the ModelRunner are partially updated`).
`docker-compose.sglang.yml` sets `NCCL_CUMEM_ENABLE=1`; keep it on a hand-run server. The trainer
keeps NVLink for its own FSDP2 collectives, so nothing about this engine changes the trainer's
NCCL env.

`NCCL_SOCKET_IFNAME` keeps the socket bootstrap off Docker's bridge and the per-container `veth`
pairs: on a host running other containers NCCL otherwise enumerates them too, and a veth carries no
host-to-host traffic (the first collective after the sync hangs) or disappears when its container
exits (`Call to bind failed: No such device` on the server, `400` on the update). Both base compose
files pass `NCCL_SOCKET_IFNAME=^docker,veth` by default (the vLLM file to its training service too)
— a same-host value; the fabric recipe excludes `lo` as well
([Servers on other nodes](#servers-on-other-nodes-efa)). Override it to pin one NIC on a
multi-homed host.

The trainer's default process group cannot contain the engine's ranks (new groups split from a
parent can only subset it), so `create_weight_update_group` forms the trainer↔engine group through
a fresh TCP-store handshake both sides can reach. Teardown asks the engine to drop its half of the
group concurrently with the local destroy — under cuMem transports each side's finalize waits for
the other. A chunk's uploads go through one device arena on the sync GPU (the chunk budget,
`HALO_WEIGHT_SYNC_CHUNK_MB`, grown for a tensor above it and released at the end of each sync),
reused across chunks by stream order alone: uploads and broadcasts share one stream, so a chunk's
copies queue behind the previous chunk's sends with no host wait. The host settles a chunk's sends
two chunks later, under the same 600 s drain deadline as the vLLM path
(`HALO_NCCL_SYNC_TIMEOUT_SECONDS` overrides it), and a failed chunk drains the whole stream under a
30 s deadline; on expiry the group is aborted instead of parking the trainer. The settle is
deferred rather than per chunk because the engine acknowledges a chunk as soon as its data arrived
while the sender's kernels retire a little later, and a host-side drain after every chunk would
serialize that tail with the next chunk's declaration, at a fifth of the push rate.

## Servers on other nodes (EFA)

The weight-sync group rides the node's fabric when both containers can drive it. All three images
carry one EFA userspace, installed whole by `docker/efa/install_efa_userspace.sh` — rdma-core, AWS
libfabric and one `aws-ofi-nccl` build; the pins live in the script, the inventory on
[Docker](docker.md#rdma-networking-infiniband-and-efa). The stack is wire-sensitive down to rdma-core: at init the
plugin probes libfabric for in-order RDMA writes and forces `NCCL_PROTO=simple` when the probe
fails, and the answer comes from rdma-core's EFA provider (`libefa`). The NGC base's MOFED `libefa`
fails it, the installer's passes, and two containers whose answers differ form the group
with different NCCL protocol tables and hang at the first collective — the same signature as a
plugin-version mismatch. So both ends must run images built from the same script; a pair that
cannot be rebuilt runs with `NCCL_PROTO=simple` on both ends instead — either restores the full
rate. The preflight below reports a side whose plugin forced the simple protocol. The
upstream vLLM and SGLang bases ship no EFA userspace, and NCCL then falls back to sockets with no
message.

Server side, layer the EFA overlay after the base compose file:

```bash
docker compose -f docker-compose.vllm.yml -f docker-compose.vllm.efa.yml up -d vllm-server
SGLANG_MODEL=... docker compose -f docker-compose.sglang.yml -f docker-compose.sglang.efa.yml up -d
```

Each overlay adds `devices: /dev/infiniband` (a `-v` bind mount does not grant access), `ulimits:
memlock: -1`, and `NCCL_NET=Libfabric NCCL_NET_PLUGIN=ofi NCCL_IB_DISABLE=0
NCCL_SOCKET_IFNAME=^lo,docker,veth,tailscale`. Naming the plugin's net is deliberate: a missing or
mismatched plugin then fails group formation instead of silently falling back to sockets. `lo` is
in the exclusion because an excluded-only list still ranks loopback first: a server started with
the base files' `^docker,veth` advertises its NCCL bootstrap address as `127.0.0.1`, and a trainer
on another node fails group formation with `remote process exited or there was a network error`;
`tailscale` because `^docker,veth` leaves a Tailscale interface eligible. The base files keep the
no-fabric, same-host recipe.

Trainer side, the same env: `make ... EFA=1` adds `--device=/dev/infiniband` and those variables to
every `DOCKER_RUN` (and drops the socket forcing from `test-gpu-vllm` / `test-gpu-sglang`); a
hand-written `docker run` passes them itself. The trainer sets no NCCL env in code, and `NCCL_NET`
/ `NCCL_NET_PLUGIN` are process-global — shared by the sync group and every trainer collective — so
this is the env the [multi-node recipes](../parallelism/launch-recipes.md#environment-variables)
already use. GPUDirect RDMA runs through dmabuf with no `nvidia_peermem`; `/dev/gdrdrv` is a
cross-node DeepEP GIN requirement, not a sync one. Name the trainer address the server dials back
to with the entry's `group_host` (or `VLLM_GROUP_HOST` / `SGLANG_GROUP_HOST`) when the default-route
NIC is not the one the server can reach; Online GRPO has only the env var (TRL builds its client
without `group_host`).

Verify the transport before training:

```bash
python scripts/profiling/weight_sync_transport.py --server-url http://<server>:8000 --backend vllm --expect efa
```

Run it from the trainer container, launched as the trainer would be (same image, devices and NCCL
env), on a GPU the server does not own. It forms the group against the live server (either
backend), pushes one real parameter of the served checkpoint (the input embedding by default, value
unchanged, so the served model is unchanged — the checkpoint read on the trainer side must be the
one the server loaded, `--model-id` when the served id is a server-side path or an alias) and
reports the transport NCCL formed on — `NET/Libfabric/…/GDRDMA` (EFA with GPUDirect), `NET/IB`,
`NET/Socket`, `P2P/CUMEM` (same-host CUDA IPC) or `SHM` — the `aws-ofi-nccl` build string, the
libfabric provider, and GB/s. `--expect efa|ib|socket|p2p|shm` makes it a gate (exit 1 on a
mismatch, on an altered served model, or when the group fails to form); flags on
[Scripts](../reference/scripts-reference.md#profiling--benchmarks).

Measured on 4× p6-b300, trainer node → server node, one worker, the trainer's own streamed path
(`update_named_param` per gathered tensor, `reset_prefix_cache`), full model per sync (NIC line
rate ~100 GB/s per GPU; a raw NCCL broadcast reaches 93 GB/s); the per-model columns are the
per-sync cost at that rate:

| Transport | Rate | Qwen3-8B (16.4 GB, measured) | gpt-oss-20b (42 GB) | Qwen3-30B-A3B (61 GB) | gpt-oss-120b (234 GB) |
|---|---|---|---|---|---|
| EFA, vLLM client | 53 GB/s | 0.31 s | 0.8 s | 1.2 s | 4.4 s |
| EFA, SGLang client | 80 GB/s | 0.21 s | 0.5 s | 0.8 s | 2.9 s |
| Sockets over the ENA | 9.7 GB/s | 1.7 s | 4.3 s | 6.3 s | 24 s |

The vLLM client's rate is set by its one HTTP round trip per chunk (`HALO_WEIGHT_SYNC_CHUNK_MB`:
1 / 2 / 4 / 8 GiB → 54 / 63 / 70 / 73 GB/s at this payload); the SGLang client's by the fabric.

## Checking a server

```bash
curl -s localhost:8000/health          # vLLM (30000 for SGLang)
curl -s localhost:8000/v1/models

curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen3-0.6B","messages":[{"role":"user","content":"Say OK"}],"max_tokens":16}'
```

The eval runners under `scripts/environments/inference/` take the server URL as the OpenAI base.
`make test-gpu-vllm` and `make test-gpu-sglang` run the GPU tiers against a live server — the
capture path plus a behaviorally verified weight sync (greedy output must change after a broadcast);
the server must own a GPU outside `TRAINER_CUDA_DEVICES`.

## Throughput

- **Servers × TP**: split the serving GPUs into the fewest servers that hold the model — at
  ~20B-MoE scale on 4 GPUs, 2×(TP=2) and 4×(TP=1) measure the same step time on a tool-heavy
  env (`code_contests`), where rollout collection dominates serving — and fewer servers halve the
  weight-sync fan-out. On short single-turn envs serving is a larger share of the step;
  re-measure before consolidating. Every rank builds its own actor pool and collects a full batch, so
  per-server request load is `max_concurrent_rollouts × world_size / num_servers`; dispatch is
  round-robin and ignores whether a server is busy.
- **Generation volume is the step-time lever** once prefetch overlaps collection into training
  (`async/prefetch_hit_rate` > 0.8): step time tracks mean episode tokens. `rollout_max_tokens`
  caps a turn; on vLLM `rollout_max_thinking_tokens` caps CoT engine-side, on SGLang only the
  environment's per-effort budgets price it.
- **Memory**: raise `--gpu-memory-utilization` / `--mem-fraction-static` to 0.9 when the server GPUs
- **`isr_engine_reference` headroom**: the trainer's engine re-score sends `prompt_logprobs` requests, and vLLM materializes an fp32 log-softmax over the vocabulary for every prefill chunk of one (`max_num_batched_tokens × vocab × 4 B`, 8 GB at 8192 × 248k) outside its memory profile — at 0.90 the engine dies of CUDA OOM under load. Serve at `--gpu-memory-utilization` ≤ 0.80 or a smaller `--max-num-batched-tokens` when that knob is on.
  are dedicated; more KV cache means more concurrent rollouts per server. Do not pass
  `--enforce-eager` — the in-place weight sync keeps captured CUDA graphs valid, and CUDA-graph
  decode is several-fold faster on long generations. On B200 pin the backend through the compose
  slot `VLLM_ATTENTION_BACKEND=FLASH_ATTN` (FlashInfer can JIT-fail on SM 10.0).
- **Sync cadence**: `sync_weights_every_n_steps: 2–4` for slow environments
  ([Environmental GRPO](../training-methods/grpo/environmental-grpo.md#nccl-weight-synchronization)).
- **Not available**: speculative decoding — no config knob passes draft-model arguments, and weight
  sync covers only the target model, so a draft would serve a stale policy from the first update.
  Weight-quantized serving is excluded by the in-place sync ([Weight sync](#weight-sync)).

## Coverage

What works, per parallelism axis (env-GRPO against a live server; dense rows on the dense server
tier, the EP rows on Qwen3-30B-A3B and gpt-oss; two trainer ranks unless the row says four):

| axis | vLLM | SGLang |
|---|---|---|
| FSDP2 DP (dense) | works | works |
| TP=2 | works | works |
| EP=2 (MoE, Qwen3-30B-A3B) | works | works |
| EP=2 + ETP=2, EP=2 + TP=2, EP=4 (four trainer ranks; gpt-oss and Qwen3-30B-A3B, with and without LoRA / expert LoRA) | works | works |
| Expert LoRA (EP=2, with resume) | works | works |

gpt-oss syncs cleanly under trainer TP=2 on SGLang: the hand-sliced attention `sinks` are skipped
by the dense parameter walk and sent once from the gathered-full drain, so each hub name reaches
the engine exactly once.

An SGLang server under its own expert parallelism (`SGLANG_TP=2 SGLANG_EXTRA_ARGS="--ep-size 2"`)
takes the sync for Qwen3 MoE: the loader keeps its local experts and drops the rest, and the
expert-only round moves the served policy. gpt-oss cannot be served that way on 0.5.17 at all — its
fused expert loader crash-loops at start under `--ep-size` (`_load_w2`, local against global expert
count), before any sync.

Undistributed MoE (`ep_group_size == 1`, EP wrappers present) works at 20B-MoE scale with
multi-server serving (2×TP=2 and 4×TP=1), expert sync, R3 rollout replay, and a flat
trainer↔engine log-ratio. Over EFA, with the trainer on one p6-b300 node and the server on another,
the end-to-end rows pass for both engines: vLLM at EP=2, TP=2 and EP=1 + LoRA (Qwen3-30B-A3B),
SGLang at EP=2 (gpt-oss-20b), and a trainer spanning two nodes (one GPU each, EP=1, DTensor
experts over the fabric) against vLLM on a third ([Servers on other nodes](#servers-on-other-nodes-efa)).

## Troubleshooting

| Symptom | Cause → fix |
|---|---|
| Group formation times out, "1/2 clients joined" | Bridge network hides the rendezvous port → `network_mode: host` on both containers |
| First sync takes minutes, `/health` unanswered | One-time NCCL group init; the engine pauses mid-update — wait |
| 400 on every rollout (vLLM) | a thinking budget the server cannot take: set `VLLM_REASONING_PARSER`, **and** `VLLM_USE_V2_MODEL_RUNNER=0` (Model Runner V2 does not implement `thinking_token_budget`). Rollouts otherwise return zero tokens and the run trains on all-masked batches |
| Run completes with flat zero gradient | Missing/wrong tool parser on a native-tool env: calls stay text, no `tool_calls`, every episode reward 0 (parser table above; ReAct envs parse text and are immune to a *mismatch*) |
| Log-ratio drifts on SGLang while the server log stays clean | No server-side signal exists: the MoE loaders skip unmapped expert names before their `not found in params_dict` warning → do not read a clean log as proof of a landed sync; the construction gates are the guard |
| `RoutedExperts: Failed` (vLLM log) | Layerwise-reload patch missing → expert syncs silently reverted; rebuild `vllm-server` |
| `/init_weight_transfer_engine` answers 500 (`NCCL error: unhandled cuda error`) while `/health` is 200 | Re-init patch missing → the engine strands a communicator per trainer connection until the GPU runs out; rebuild `vllm-server` and recreate the server container |
| First sync hangs at the first collective after the group formed, both ends idle | The two containers drive different transports (`NCCL_NET` / `NCCL_NET_PLUGIN` differ) or different `aws-ofi-nccl` + libfabric builds (an upstream server image, a host-installed plugin) — the pair forms the group and then hangs → both ends from the Halo images with the same recipe (compose EFA overlay + `make ... EFA=1`, or both on the no-fabric defaults); `scripts/profiling/weight_sync_transport.py` reports the transport and build each side formed on |
| `ncclP2pImportShareableBuffer ... invalid argument` on the first update, `The full weights of the ModelRunner are partially updated` (SGLang) | cuMem off on the server only — SGLang sets `NCCL_CUMEM_ENABLE=0` unless it is pre-set → `NCCL_CUMEM_ENABLE=1` in the server container (the compose default); restart the server, it holds a half-written model |
| Sync rate in the single-digit GB/s on an EFA host | The group formed on sockets: a server without its EFA overlay or an image without the EFA userspace → `weight_sync_transport.py --expect efa` on each server, then fix the end it names |
| `/init_weight_transfer_engine` answers 500 with `ncclP2pImportShareableBuffer ... Cuda failure 101 'invalid device ordinal'` in the server log | The server container does not see the trainer's GPU — expose all GPUs to it and select with `CUDA_VISIBLE_DEVICES` (`VLLM_CUDA_DEVICES`), as the compose file does |
| `Call to bind failed: No such device` in the server log (`400` on `/update_weights_from_distributed`), or a trainer collective hanging right after the first sync | NCCL's socket bootstrap picked a Docker `veth` — set `NCCL_SOCKET_IFNAME=^docker,veth` on both ends (the base compose default, same host only; the fabric recipe is `^lo,docker,veth,tailscale`) |
| Cross-node group formation fails with `remote process exited or there was a network error` | The server's `NCCL_SOCKET_IFNAME` does not exclude `lo`, so it advertised `127.0.0.1` as its bootstrap address → start it under the EFA overlay (`^lo,docker,veth,tailscale`), not the base file alone |
| `Errno 98` binding the group port at trainer start | Previous run's port in TIME_WAIT → wait for `ss -tln` to clear, or change `group_port` |
| `/health` answers but generation is wedged after a killed trainer | Scheduler left attached to the dead transfer group → restart the server container |
| `RESTART the … server` in the trainer log; that server stays paused and refuses the next sync | A sync was interrupted after part of the model went out → the engine holds a half-written model on purpose ([Weight sync](#weight-sync)); restart it, do not `/resume` it |
| `EngineDeadError` on the first request after a restart, `reshape_and_cache_flash … Meta tensors` in the engine log | The reloaded AOT compile cache does not match the attention backend the restarted engine auto-selected (free GPU memory steers that choice, so a co-tenant server changes it) → pin the backend (compose `VLLM_ATTENTION_BACKEND=FLASH_ATTN`, which becomes `--attention-backend`; 0.26.0 reads no such environment variable itself), or clear `/root/.cache/vllm/torch_compile_cache` before restarting. vLLM respawns the engine core, so later requests answer |
| Rollouts from a stale policy after a server swap | A reconnected server holds launch weights until the next full sync |
