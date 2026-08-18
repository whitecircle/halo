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
| [R3 routing replay](../training-methods/grpo/environmental-grpo.md#off-policy-mismatch-and-stability-knobs) | `--enable-return-routed-experts` + `--moe-backend triton` | `--enable-return-routed-experts` + `--moe-runner-backend triton` |
| Thinking budget (`rollout_max_thinking_tokens`) | enforced engine-side with a reasoning parser and `VLLM_USE_V2_MODEL_RUNNER=0`; harmony-disabled gpt-oss arms it off the toolkit plugin's marker ([GPT-OSS](../models/gpt-oss.md#serving-for-grpo-vllm)) | rejected at config time |
| Expert layout on sync | whatever the family's own `gather_expert_state_dict` emits — per-expert (Qwen3 MoE, GLM-4/Laguna, Bailing, LFM-2) or fused where that is the family's base gather (Qwen3.5/3.6, Gemma 4); 0.26.0's expert loader reads both. A family whose hub namespace differs from its module tree (Step-3.7's per-layer `moe.gate_proj`/`up_proj` stacks) is re-spelled through transformers' save-side revert, so the engine receives its hub keys | fused only (GptOss) |
| Trainer expert distribution ([EP/ETP](../reference/glossary.md#parallelism)) | supported | refused at construction |

Use vLLM unless you need SGLang specifically — vLLM is the only backend for expert-distributed
runs and for Online GRPO.

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
SGLang), so the forwarding rank sends each 1 GB chunk as the gather fills it and holds one chunk of
pinned host memory — not one model (~800 GB at 400B). The chunk is cut before the budget is
exceeded, and the recycled pinned buffers are themselves capped at one chunk. In multi-server mode
(`rollout_server_configs`) one page-locked snapshot per parameter is shared across all servers and
each chunk goes out to every server **concurrently**, its buffers recycled once they all have it.
The trade is that a chunk cannot be replayed: a server that fails **after** its first chunk is
reported rather than reconnected — the trainer does not hold what already landed. One that fails
before any chunk went out (an engine restarted between syncs, the common case) is still recovered by
the reconnect + re-flush. Each client owns its own NCCL connection to its server on a `group_port`
bound on the *trainer* host
([group ports](../training-methods/grpo/environmental-grpo.md#nccl-weight-synchronization)).

**The quiesce spans the streaming, not just the final broadcast.** The update opens with the first
full chunk — ~1 GB into the gather — and closes when the last one lands, so a server stops serving
for as long as the gather runs: minutes at 397B, and for every server at once outside the
[rolling path](../training-methods/grpo/environmental-grpo.md#single-server-vs-multi-server). vLLM
queues requests behind its pause; SGLang's is `/pause_generation {"mode": "abort"}` (its post-update
cache flush asserts an idle scheduler), so in-flight generations — prefetched rollouts included —
are dropped across that window.

**An interrupted mid-stream sync leaves that server unusable.** The engine then holds neither the
old policy nor the new one, and vLLM's layerwise reload materializes a layer whose tensors straddled
the boundary from *uninitialized* storage while it waits for the rest. The abort therefore leaves
that engine **paused** instead of resuming it, refuses every later sync on that client, and logs
`RESTART the … server`. Restart the container — the trainer kept no copy of what landed and cannot
repair it.

The first sync takes minutes (one-time NCCL group formation per server) and the server stops
answering `/health` mid-update — an update in progress, not a hang. Every sync pauses the engine
and resumes it after: `/pause` … `/resume` on vLLM, with the broadcast itself bracketed by
`/start_weight_update` … `/finish_weight_update` (the layerwise reload phase, closed on every path);
`/pause_generation` … `/continue_generation` on SGLang. The broadcast is packed (~1 GB buffers,
double-buffered) on vLLM and typed 1 GB chunks on SGLang.

On vLLM, each client owns **one persistent CUDA stream pair** for the pack uploads, because PyTorch's
caching allocator keeps freed blocks in per-stream pools: a fresh stream per sync would strand one
payload of reserved memory every sync. The forwarding rank's steady state is therefore its training
footprint plus about one sync of pack buffers. SGLang stages its chunks on the default stream and has
no equivalent hazard.

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

Which families each backend accepts for RL is gated trainer-side. Two gates are SGLang-specific — the
[EP refusal](#ep-cannot-be-combined-with-sglang-weight-sync) and the
[fused-layout](#the-fused-expert-layout-is-declared-per-family) sections below. Two apply to **both**
backends: DeepSeek-V4, Inkling, Zaya, Cohere2 MoE, GLM-5 Next and Mistral4 declare
`_supports_weight_sync = False` (Cohere2 MoE because its sync is unverified against
the pinned server; Mistral4 because vLLM 0.26.0 registers no `mistral4` class at all, so its
composite loader has no text tower to build — [Mistral4](../models/mistral4.md#serving)), and Ling 3.0
(`bailing_hybrid`) and Ring (`bailing_moe_linear`) are refused by `model_type` — no pinned engine
registers a model class for those spellings, so the server cannot even load the base model. A
bnb-quantized (QLoRA) base is refused for both as well.

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

## vLLM

`Dockerfile.vllm` builds `vllm-server:0.26.0` with the native NCCL weight-transfer engine, the
layerwise-reload patch, R3 routed-experts capture (base64-npy `routed_experts` per completion
choice), and `nvidia-nccl-cu13` installed at `uv.lock`'s exact pin — the same NCCL the training
images run — with `VLLM_NCCL_SO_PATH` baked to that wheel so the base image's older system copy
can never win the soname race. A skew fails `ncclCommInitRank` at `/init_weight_transfer_engine`
(HTTP 500, `NCCL error: internal error`) — rebuild the image after any lock bump of the pin. 0.26.0
is the last vLLM release on torch 2.11 — the training image's torch and NCCL generation; 0.27 moves
to torch 2.13, whose NCCL does not match that pin.

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
| `VLLM_CUDA_DEVICES` | `7` | Server GPUs — must exclude the trainer's (a rank cannot broadcast to itself) |
| `VLLM_TP` | `1` | `--tensor-parallel-size` |
| `VLLM_GPU_MEM` | `0.85` | `--gpu-memory-utilization` |
| `VLLM_MOE_BACKEND` | `triton` | Keep `triton` for MoE RL ([Weight sync](#weight-sync)) |
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
| most others | `hermes` (`<tool_call>` XML) |

`docker-compose.vllm.yml` defaults **both** containers to the no-InfiniBand recipe, overriding the
training image's baked IB/Gin NCCL env so the cross-container transfer group takes NVLink + socket
instead of the uninitialized OFI/Gin NET path: `NCCL_IB_DISABLE=1` + `NCCL_NET=Socket` on the
trainer, `NCCL_IB_DISABLE=1` + `NCCL_P2P_LEVEL=NVL` on the server. Without the recipe the first
collective wedges: both GPUs spin at 100% and the trainer raises after 120 s (`NCCL weight-sync
warm-up all-reduce did not complete`).

`NCCL_NET=Socket` disables the Gin plugin and is process-global, so those defaults fit EP=1 /
no-DeepEP runs on single-node, no-RDMA hosts. On a multi-node IB/EFA cluster **override both**
(`NCCL_IB_DISABLE=0` and a non-Socket `NCCL_NET`) or every trainer collective goes to TCP and DeepEP
breaks. Steer the transfer group with `VLLM_GROUP_HOST` instead: it names the trainer address the
server dials back to and touches nothing else. Unset, the client resolves loopback for a local
server and the default-route NIC otherwise, so a same-host compose stack needs no value; a
`rollout_server_configs` entry's `group_host` overrides it for that one server. `NCCL_SOCKET_IFNAME`
is not a transfer-group knob; it is process-wide, filtering the interfaces the default process group
and DeepEP pick too, so on a multi-homed host set it only to an interface every collective can use.

## SGLang

`Dockerfile.sglang` builds `sglang-server:0.5.17` to align NCCL: upstream's wheel trails
`uv.lock`'s exact pin (what the training images run), and weight sync needs both ends on one
runtime. The build bumps the wheel and asserts the weight-sync routes, request schemas, and
rendezvous convention still exist, so an upstream refactor fails the build instead of a training
run. Serving-only use can run upstream directly (`SGLANG_IMAGE=lmsysorg/sglang:v0.5.17`). 0.5.17 is
the last SGLang release on torch 2.11 — the training image's torch and NCCL generation; 0.5.18
moves to torch 2.13, whose NCCL does not match the pin weight sync needs on both ends.

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

`SGLANG_GROUP_HOST` belongs to the **trainer** (the `VLLM_GROUP_HOST` equivalent, separate because
the engines can sit on different hosts).

Engine behavior under RL:

- **Sampled ids** arrive per request: SGLang's OpenAI `logprobs` reports tokens as text, so the
  trainer sets `return_meta_info` + `return_prompt_token_ids` and reads
  `choice.meta_info.output_token_logprobs[i][1]`. No server flag.
- **A length cut-off is a `stop_reason`**, not a `finish_reason`. `get_finish_reason`
  (`src/inference/response.py`) reads `finish_reason or stop_reason`, so the rollout path grades an
  engine-truncated completion as truncated rather than as an answer.
- **R3**: serve with `--enable-return-routed-experts --moe-runner-backend triton` — the fused
  runners (`triton_kernel`, flashinfer; auto-selection picks one for most MoE shapes) bypass the
  capture hook and return nothing. The trainer opts in per request and decodes the wire format
  (response-level `sglext.routed_experts`, base64 raw int32) by the model's own layer/top-k counts.
  Rows cover the full sequence, so prompt spans replay too.
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
- **Set `fsdp_reshard_after_backward: false` on the trainer** for step time: the
  socket-global NCCL below puts FSDP2's per-microstep re-gather of the full model
  ([ZeRO reshard](../parallelism/data-parallelism.md#zero-2-vs-zero-3-reshard_after_forward)) on
  loopback TCP (~15 s × `gradient_accumulation_steps` per step otherwise; the flag leaves one
  such gather per optimizer step). Plain-DP/CP/EP runs
  only — the flag is rejected under trainer TP or PP, which pay the re-gather. With it, R3-mode SGLang runs within
  ~1.4× of the same config on vLLM (20B MoE, 2×(TP=2)+DP=4: ~215 s vs ~150 s per step);
  engine kernels are near parity, and the residual is the weight-sync gather and once-per-step
  collectives over sockets.

### NCCL transport (SGLang only)

The trainer and engine are same-host but different containers; NCCL's CUDA-IPC path fails across
that boundary (`ncclP2pImportShareableBuffer ... invalid argument`) mid-update. Both ends therefore
need the socket transport, process-global:

```text
NCCL_P2P_DISABLE=1  NCCL_SHM_DISABLE=1  NCCL_NET=Socket  NCCL_IB_DISABLE=1  NCCL_NET_PLUGIN=none
```

`NCCL_NET_PLUGIN=none` is separate and load-bearing: the images bundle the aws-ofi plugin, which
NCCL prefers and then wedges group formation on a host with no OFI fabric. Setting the flags only on
the server is not enough — the group still forms (a TCP rendezvous) and the first broadcast hangs,
because the trainer still reaches for CUDA-IPC.

The cost is process-global: a multi-rank trainer
loses NVLink between its own ranks for the whole job (the reason for
`fsdp_reshard_after_backward: false` above). vLLM's pynccl cannot form rings under this environment
— the two engines' sync transports are **mutually exclusive per process**. (The vLLM no-IB recipe
above uses only `NCCL_IB_DISABLE`/`NCCL_NET=Socket`, which both engines tolerate; the other three
are SGLang-only.)

The trainer's default process group cannot contain the engine's ranks (new groups split from a
parent can only subset it), so `create_weight_update_group` forms the trainer↔engine group through
a fresh TCP-store handshake both sides can reach.

### EP cannot be combined with SGLang weight sync

`rollout_backend: sglang` with any expert distribution (`ep_size × expert_tp_size > 1`) is refused
at construction (`validate_backend_parallelism`): the sync communicator needs CUDA-IPC disabled,
DeepEP needs it enabled for symmetric memory, and both are process-global. All four flag
combinations fail — DeepEP symmetric-memory error, truncated broadcast, engine-side
`ncclUnhandledCudaError`, or truncation under a scoped window; NCCL caches the flags on first read.
`ep_buffer_backend: legacy` removes only the DeepEP half; the cross-container broadcast still fails.

The gate fires at trainer construction — after the model loads, before any rollout or sync —
because the failure it prevents lands at the first sync with the served weights partly overwritten.
Use `rollout_backend: vllm` for expert-distributed runs.

### The fused expert layout is declared per family

SGLang loads experts fused (`experts.gate_up_proj` / `experts.down_proj`), the way transformers
stores them; vLLM takes whichever layout the family's own gather emits. A family declares the fused
gather through `gather_fused_expert_state_dict`, and **only the GptOss layer implements it** — the
base raises for every other family. `gpt_oss` is also the only 0.5.17 loader reaching for the fused
helper (`make_expert_params_mapping_fused`); `qwen3_moe` builds its mapping over per-expert names
(`make_expert_params_mapping`), so a fused pair matches nothing there.

A family without the override is refused at construction under `rollout_backend: sglang`, naming the
class. The hook is additive — a family opts in by overriding `gather_fused_expert_state_dict` on its
EP layer ([Adding a Model](../models/adding-a-model.md)) — but add it only against an engine whose
loader verifiably consumes the fused pair.

The sync ships hub names and full unsharded tensors into the engine's own `load_weights` mapping, and
each TP rank narrows its slice. An expert name that mapping does not cover leaves **no server-side
signal**: both MoE loaders `continue` on an unmatched `mlp.experts` name *before* the
`not found in params_dict` warning (reachable only from the dense loaders), the update still returns
`200 OK`, and the engine keeps serving its launch-weight experts under a freshly synced router.

The trainer-side construction gates are the whole guard. They read the family's contract off a live EP
wrapper, or — when a run has none (`use_grouped_gemm: false` at `ep_size: 1`) — off the `model_type`
registry. This gate is independent of the transport conflict.

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

What works, per parallelism axis (env-GRPO, 2 trainer ranks, live server; dense rows on the dense
server tier, the EP row on Qwen3-30B-A3B):

| axis | vLLM | SGLang |
|---|---|---|
| FSDP2 DP (dense) | works | works |
| TP=2 | works | works |
| EP=2 (MoE) | works | refused (above) |

SGLang under trainer TP=2 runs without `fsdp_reshard_after_backward: false` (rejected under TP)
and pays the per-microstep re-gather — correct, but not tuned for step time. gpt-oss syncs cleanly
there: the hand-sliced attention `sinks` are skipped by the dense parameter walk and sent once from
the gathered-full drain, so each hub name reaches the engine exactly once.

Undistributed MoE (`ep_group_size == 1`, EP wrappers present) works at 20B-MoE scale with
multi-server serving (2×TP=2 and 4×TP=1), fused expert sync, R3 rollout replay, and a flat
trainer↔engine log-ratio. Multi-node group formation (`*_GROUP_HOST` across hosts) is **not**
covered.

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
| `ncclBuildRings: ring 0 does not contain rank 1` (vLLM) | Trainer launched with SGLang's five socket vars — the sync transports are mutually exclusive |
| Broadcast hangs at the first sync (SGLang) | The five NCCL vars missing on one end — both processes need all five |
| `Errno 98` binding the group port at trainer start | Previous run's port in TIME_WAIT → wait for `ss -tln` to clear, or change `group_port` |
| `/health` answers but generation is wedged after a killed trainer | Scheduler left attached to the dead transfer group → restart the server container |
| `RESTART the … server` in the trainer log; that server stays paused and refuses the next sync | A sync was interrupted after part of the model went out → the engine holds a half-written model on purpose ([Weight sync](#weight-sync)); restart it, do not `/resume` it |
| `EngineDeadError` on the first request after a restart, `reshape_and_cache_flash … Meta tensors` in the engine log | The reloaded AOT compile cache does not match the attention backend the restarted engine auto-selected (free GPU memory steers that choice, so a co-tenant server changes it) → pin the backend (compose `VLLM_ATTENTION_BACKEND=FLASH_ATTN`, which becomes `--attention-backend`; 0.26.0 reads no such environment variable itself), or clear `/root/.cache/vllm/torch_compile_cache` before restarting. vLLM respawns the engine core, so later requests answer |
| Rollouts from a stale policy after a server swap | A reconnected server holds launch weights until the next full sync |
