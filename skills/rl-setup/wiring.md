# rl-setup — wiring reference

Concrete commands, config fields, and launch examples for online / async-environment
GRPO. Source of truth: `Dockerfile.vllm`, `docker-compose.vllm.yml`,
`src/distributed/nccl/clients/vllm.py`, `src/trainers/grpo/environmental.py`,
`src/configs/async_training_config.py`, `src/configs/environment_config.py`,
`src/rewards/spec.py` (the reward terms), `src/environments/registry.py`. Cross-link the docs:
`agent-docs/infrastructure/rollout-servers.md` (server setup, weight sync, SGLang),
`agent-docs/training-methods/grpo/rewards.md` (the `rewards:` term list),
`agent-docs/training-methods/grpo/online-grpo.md`,
`agent-docs/training-methods/grpo/async-grpo/{README,setup}.md`.

## 1. Bring up the vLLM container

vLLM 0.26.0 (cu13) is a **separate** image — the training env cannot import vLLM
(ABI-incompatible: the server image ships its own `transformers 5.14.1` / vllm 0.26.0,
training is on `transformers 5.16` / torch 2.11). `Dockerfile.vllm` pins that 5.14 line and asserts
it at build — vLLM's Gemma 4 code reads the 5.14 config schema 5.16 folds into `per_layer_config`
(`agent-docs/infrastructure/rollout-servers.md`, config-schema parity). Colocated in-process vLLM is also
rejected on the trainer: it would build its own NCCL/TP communicators on the training GPUs
and deadlock against the FSDP2/EP/CP/TP groups. Server mode keeps the two NCCL
worlds in separate processes (the trainer rejects `vllm_mode != server`).

```bash
# Build the server image (or `make build-vllm`). VLLM_VERSION is the one functional build arg.
docker build -f Dockerfile.vllm -t vllm-server:0.26.0 .
# The tool parser is NOT a build arg, and vLLM reads no VLLM_TOOL_PARSER env. It is a runtime
# compose variable interpolated into the command (`--tool-call-parser ${VLLM_TOOL_PARSER:-hermes}`).

# Compose: GPUs split between server and trainer, ipc=host + all GPUs visible
# on both for NCCL P2P. The server advertises native weight-transfer endpoints.
VLLM_MODEL=Qwen/Qwen3-4B-Instruct-2507 \
VLLM_CUDA_DEVICES=0 VLLM_TP=1 VLLM_PORT=8000 \
TRAINER_CUDA_DEVICES=1,2,3,4,5,6,7 \
HF_HOME=$HALO_SCRATCH/hf HF_TOKEN=$HF_TOKEN \
  docker compose -f docker-compose.vllm.yml up vllm-server
```

Four build gates keep the weight sync honest, failing the image rather than the run:
the layerwise-reload patch must be applied **and** its `SKIP_LAYER_NAMES` must cover the
classes that `copy_` their weights directly on this vLLM (`RoutedExperts`, `OAIAttention`,
`Gemma4Router`, checked through the MRO); the weight-transfer **re-init** patch must replace
both `NCCLWeightTransferEngine.init_transfer_engine` and `.shutdown` (unpatched, every trainer
that connects strands a live communicator on each engine-core worker until
`ncclCommInitRank` fails outright); `/pause` must still accept `mode=keep`, the form the client pauses with so in-flight
requests survive a sync; and `verify_gptoss_plugins.py` must exercise the gpt-oss
tool + reasoning parser plugins, which no test suite can reach (they import vLLM). Two more
assert the transformers 5.14 line with a per-family config-schema parity fixture, and that
`EngineArgs` still carries `weight_transfer_config` / `enable_return_routed_experts` and
`KernelConfig.moe_backend`.

Compose env knobs (`docker-compose.vllm.yml`): `VLLM_MODEL`, `VLLM_PORT` (8000), `VLLM_TP`
(tensor-parallel size), `VLLM_GPU_MEM` (`--gpu-memory-utilization`, default 0.85),
`VLLM_TOOL_PARSER` (default `hermes`) and `VLLM_TOOL_PARSER_PLUGIN`, `VLLM_MOE_BACKEND`
(default `triton` — required for MoE weight sync), `VLLM_ATTENTION_BACKEND` (unset = auto;
GLM-4 MoE Lite / GLM-4.7-Flash needs `CUTLASS_MLA` on Blackwell), `VLLM_CHAT_TEMPLATE`,
`VLLM_REASONING_PARSER` / `VLLM_REASONING_PARSER_PLUGIN`, `VLLM_USE_V2_MODEL_RUNNER`,
`VLLM_ENABLE_R3` (`--enable-return-routed-experts`, required by `routing_replay: rollout`),
`VLLM_CUDA_DEVICES` / `TRAINER_CUDA_DEVICES`. Each optional one is a `${VAR:+--flag ${VAR}}`
slot, so leaving it unset omits the flag entirely.
The server command always includes `--weight-transfer-config '{"backend": "nccl"}'`,
`--return-tokens-as-token-ids` (load-bearing: `train_on_sampled_tokens` defaults
on, so without it the trainer re-tokenizes a re-render), `--logprobs-mode
processed_logprobs` (also load-bearing: a rank-0 startup probe refuses a server returning raw
pre-temperature logprobs at any `rollout_temperature != 1.0`) and
`--enable-auto-tool-choice`, plus `--moe-backend ${VLLM_MOE_BACKEND:-triton}`. Both services run
`network_mode: host`; healthcheck polls `/health`; the `training` service
`depends_on` it being healthy and gets `VLLM_SERVER_URL=http://localhost:8000`.

`docker-compose.sglang.yml`: `SGLANG_MODEL` (**required**), `SGLANG_CUDA_DEVICES`,
`SGLANG_PORT` (30000), `SGLANG_TP`, `SGLANG_GPU_MEM` (`--mem-fraction-static`),
`SGLANG_MOE_RUNNER_BACKEND` (default `triton` — required for MoE weight sync and R3),
`SGLANG_ENABLE_R3`, `SGLANG_ATTENTION_BACKEND`, `SGLANG_TOOL_PARSER` (default `auto`),
`SGLANG_REASONING_PARSER`, `SGLANG_CHAT_TEMPLATE`, `SGLANG_TRUST_REMOTE_CODE`,
`SGLANG_EXTRA_ARGS`, `NCCL_CUMEM_ENABLE=1`. Weight sync needs the repo's `sglang-server:0.5.17` image, not upstream: it
carries the GLM-4 gate and Gemma 4 router loader patches (`docker/sglang/patches/`) without which a
synced router never reaches routing.

**Thinking budget is enforced engine-side.** `rollout_max_thinking_tokens` becomes the
per-request `thinking_token_budget`, which vLLM honors only with a reasoning parser
(`--reasoning-parser qwen3` for Qwen3.x; for gpt-oss the bundled
`/opt/gpt_oss_reasoning_parser.py` plugin, whose budget arms on the generation prompt's own
trailing `<|start|>assistant` so the count starts at the first sampled token). The run must
also set `VLLM_USE_V2_MODEL_RUNNER=0` — Model Runner V2 answers `thinking_token_budget` with a
400 on every rollout. The field is vLLM-only: under `rollout_backend: sglang` it is refused
at config time.

Standalone (no compose):
```bash
docker run --gpus all --network=host --ipc=host \
  vllm-server:0.26.0 Qwen/Qwen3-4B-Instruct-2507 --port 8000 \
  --weight-transfer-config '{"backend": "nccl"}' --moe-backend triton \
  --return-tokens-as-token-ids --logprobs-mode processed_logprobs \
  --enable-auto-tool-choice --tool-call-parser hermes
```

> Checkpoint gotcha (`agent-docs/infrastructure/docker.md`): a model saved with
> `model_type: qwen3_5_moe_text` is rejected by vLLM (expects
> `Qwen3_5MoeConfig`) — patch `config.json` to `model_type: qwen3_5_moe`
> before serving.

## 2. AsyncTrainingConfig fields that matter

(`src/configs/async_training_config.py` — parsed from the async-GRPO YAML)

| Field | Default | Purpose |
|---|---|---|
| `rollout_backend` | `vllm` | engine: `vllm` or `sglang` (async GRPO only). SGLang's 0.5.17 loaders refuse a longer family list than vLLM's, quoted at trainer construction; its server needs `NCCL_CUMEM_ENABLE=1` (compose default) and must be the repo's `Dockerfile.sglang` image — `agent-docs/infrastructure/rollout-servers.md#which-families-each-engine-serves` |
| `rollout_server_url` | `http://localhost:8000` | single-server URL (weight sync + generation) |
| `rollout_server_configs` | `None` | multi-server: `[{"url": ..., "group_port": ...}]`; overrides `rollout_server_url`, enables prefetch overlap |
| `rollout_connection_timeout` | `120.0` | wait for `/health` |
| `sync_weights_every_n_steps` | `1` | NCCL weight push cadence |
| `num_rollout_workers` | `64` | Ray env actors (per DP rank when `ray_address` set) |
| `max_concurrent_rollouts` | `None` | pipeline depth; default 4 × this rank's share of `num_rollout_workers` (the whole pool locally, `÷ world_size` on a shared Ray cluster) |
| `ray_address` | `None` | shared Ray cluster; `None` = per-rank local |
| `rollout_temperature` / `rollout_top_p` / `rollout_max_tokens` | `0.7` / `0.95` / `32768` | rollout sampling (max tokens per turn) |
| `rollout_max_thinking_tokens` | `None` | per-turn CoT cap (vLLM `thinking_token_budget`); needs a server reasoning parser + `VLLM_USE_V2_MODEL_RUNNER=0`, refused under `sglang` |
| `rollout_chat_template_kwargs` | `{}` | chat-template variables sent on every rollout request **and** applied to the trainer's own renders (Qwen3.x `preserve_thinking`); `reasoning_effort` is refused here — it travels per episode |
| `max_train_row_tokens` | `None` | longest training row a rank takes; must exceed `rollout_max_tokens`. Over-cap per-turn rows are left out, whole-trajectory rows train at zero weight (`sampling/rows_over_cap_frac`) |
| `eval_rollout_batch_size` | `None` | rows per rank in one eval rollout round (eval runs without prefetch); `None` = the eval batch |
| `effort_length_penalty_k0` / `effort_length_floor_weight` | `None` / `0.0` | both off by default; the first prices an episode's reasoning tokens by its effort level (capped at `effort_length_penalty_c_max`), the second its shortfall against `effort_length_floor_budgets` × the thinking budget it ran under |
| `episode_timeout` | `1200.0` | per-episode wall clock, checked against the NCCL watchdog — raise `DIST_NCCL_TIMEOUT_MINUTES` with it |
| `train_on_sampled_tokens` | `True` | train on the server's actual sampled ids (needs `--return-tokens-as-token-ids`) rather than a re-tokenized re-render |
| `enable_prefetch` | `True` | overlap rollout with training (auto-disabled in single-server mode) |
| `num_prefetch_batches` | `1` | batches prefetched ahead |
| `model_name` / `request_timeout` / `max_retries` / `retry_base_wait` | — | per-request HTTP behavior |

`vllm_group_port` is **not** in `AsyncTrainingConfig` — it's a TRL `GRPOConfig`
field (`self.args.vllm_group_port`) read when constructing `VLLMWeightSyncClient`;
set it in YAML (e.g. `vllm_group_port: 51216`).

### EnvironmentConfig fields (`src/configs/environment_config.py`)

`environment_type` (registry name), `rewards` (the term list, default
`[{source: environment}]`), `max_turns` (`int | None`, default **`None`** = keep the environment
class's own default — `code_contests`/`codeforces` 15, `swe` 20, `exam_qa` 8, every other
environment 10), `environment_kwargs` (per-env dict).
`to_env_config()` forwards the terms as `reward_terms` plus an explicitly-set `max_turns`,
merged with `environment_kwargs`, to `resolve_environment(environment_type, config)`.

`rewards` is parsed into typed dataclasses at config time (`src/rewards/spec.py`), so a bad term
fails before any server is touched. Each term prices one source's score in `[0, 1]` as
`weight × score ^ exponent` — `environment` (the episode grade), `judge` (a generative judge over
`requirements`), `reward_model` (a served BT / seq-cls model); the online arm adds `accuracy` and
`format`. Partial credit is the `exponent` (`> 0`, above 1 convex); there is **no failure offset**,
since a constant cancels in the group baseline. Per-term reference:
`agent-docs/training-methods/grpo/rewards.md`.

## 3. NCCL client connect flow

`VLLMWeightSyncClient` (`src/distributed/nccl/clients/vllm.py`), driven by
the trainer's `_init_weight_sync_client` → `_sync_weights_to_engine`:

1. `VLLMWeightSyncClient(base_url=..., group_port=..., connection_timeout=...)`
   → `check_server()` polls `/health`.
2. `init_communicator(device=cuda:N)` — GETs `/get_world_size`, computes
   `world_size = inference_ws + 1`, advertises `master_address` (arg →
   `VLLM_GROUP_HOST` env → default-route NIC) and binds the TCPStore on
   `0.0.0.0`; POSTs `/init_weight_transfer_engine` (server rank_offset=1) while
   the trainer (rank 0) builds the `StatelessProcessGroup` + `PyNcclCommunicator`
   concurrently. Done once, while vLLM is idle.
3. Each sync: `sync_model_weights()` → `/pause` → `/start_weight_update` →
   `packed_broadcast_producer` (~1 GB packed buffers) alongside server
   `/update_weights` → `/finish_weight_update` → `/resume`. LoRA:
   `update_named_param()` buffers, `reset_prefix_cache()` flushes in one bulk
   broadcast (adapters merged first, names stripped of `base_model.model.`).

The parallelism-aware gather is the shared `gather_and_send_weights`
(`src/trainers/grpo/rollout/weight_sync.py`) that both the online trainer and env
GRPO both reach through `sync_trainer_weights`: EP experts via
`gather_expert_state_dict` / `gather_ep_layer_weights` (collective across the EP group); non-EP /
router / shared / dense params through `materialize_dtensor`, which returns each FSDP2-DP and
TP-mesh shard full (both HF's `tp_plan` and the toolkit's attention-only TP place theirs as
DTensors). The hand-sliced non-DTensor TP shards — GptOss sinks — are skipped there and drained
gathered by `iter_tp_sharded_non_dtensor_full`; shipping this rank's slice under the full-tensor
name would corrupt the served weights. **All ranks must enter
the gather; only the global-main tp_rank-0 process sends.** PEFT adapters are
merged into the base and forwarded under base-model names. Multi-homed clusters:
pin the control-plane NIC via `VLLM_GROUP_HOST` (distinct from
`NCCL_SOCKET_IFNAME`). A server on another EFA node: compose EFA overlay on the
server, `make ... EFA=1` on the trainer, `scripts/profiling/weight_sync_transport.py
--server-url http://<server>:8000 --expect efa` as the preflight (`agent-docs/infrastructure/rollout-servers.md#servers-on-other-nodes-efa`).

## 4. Environment registry (`src/environments/registry.py`)

| `environment_type` | Class / factory | Tools | Output |
|---|---|---|---|
| `react_math` | `ReActEnvironment` | Calculator, Python | Thought/Action/Observation |
| `react_search` | `ReActEnvironment` | Web search | Thought/Action/Observation |
| `native_math` | `NativeToolUseEnvironment` | Calculator + Python | OpenAI function calling |
| `native_coding` | `NativeToolUseEnvironment` | Python REPL | OpenAI function calling |
| `native_combined` | `NativeToolUseEnvironment` | All native tools | OpenAI function calling |
| `swe` | `SweEnvironment` | File ops + Python REPL | OpenAI function calling |
| `mcp` | `NativeMCPClientEnvironment` | MCP server tools (`mcp_server`) | MCP protocol |
| `qa_search` | `NativeToolUseEnvironment` (via `create_qa_search_environment`) | Web search (+ optional Python) | OpenAI function calling |
| `code_contests` | `CodeContestsEnvironment` | Python REPL + test runner (`timeout_per_test`, `output_comparison`) | OpenAI function calling |
| `codeforces` | `CodeContestsEnvironment` (tokens preset) | Python REPL + test runner (token compare + special-judge checkers) | OpenAI function calling |
| `exam_qa` | `ExamQAEnvironment` | Optional search (`open_book`) | OpenAI function calling |

Per-env `environment_kwargs`: `search_backend` (qa_search/exam_qa only — `react_search` builds its
tools with the default backend and refuses the key; `mock` needs `HALO_ALLOW_MOCK_SEARCH=1`),
`open_book` (exam_qa), `mcp_server` (mcp), `timeout_per_test`, `max_grading_seconds` and
`compiled_time_limit_scale` (code_contests), `include_python_tools` (qa_search). Every environment
also takes the `BaseEnvironment` knobs (`src/environments/base.py`): `reasoning_effort` /
`reasoning_effort_profiles`, `carry_reasoning` (refused under `rollout_backend: sglang`),
`requires_answer`, `max_observation_chars`, `max_length_cutoff_recoveries` and the
`tool_success_reward` / `tool_error_penalty` / `tool_reward_cap` turn shaping — the shipped
`examples/grpo/environmental/environmental-grpo-template.yaml` enumerates them. Custom env: pass `environment_cls`
(a `src.environments.base.BaseEnvironment` subclass) + `environment_kwargs`
instead of `environment_config`.

## 5. Launch examples

### Online GRPO (RLVR) — rule-based verifiable rewards

Config: `examples/grpo/online/qwen3/online-grpo-qwen3-4b-smoke.yaml`
(template: `rlvr-online-grpo-template.yaml`). Uses TRL-native vLLM fields
(`use_vllm: true`, `vllm_mode: server`, `vllm_server_host`, `vllm_server_port`) and the same
`rewards:` term list, defaulting to `[{source: accuracy}]` — the RLVR graders `accuracy` and
`format`, plus `judge` / `reward_model` (`src/args/rlvr_online_grpo_args.py`).

```bash
# vLLM on GPU 0
docker compose -f docker-compose.vllm.yml up vllm-server   # or standalone docker run

# Trainer on GPUs 1+ (accelerate for plain DP/LoRA)
CUDA_VISIBLE_DEVICES=1 accelerate launch \
  scripts/training/online_grpo/rlvr.py \
  examples/grpo/online/qwen3/online-grpo-qwen3-4b-smoke.yaml

# MoE with EP (torchrun, not accelerate)
torchrun --nproc_per_node=8 \
  scripts/training/online_grpo/rlvr.py \
  examples/grpo/online/qwen3_5/online-grpo-qwen3.6-35b-a3b-dapo-math.yaml \
  --expert_parallel_size=8
```

(`scripts/training/online_grpo/rlvr.py` — verifiable rewards.)

### Async GRPO with Environments — multi-turn tool-use

Recipes live at `examples/grpo/environmental/<family>/<backend>/` (`gemma4`, `gptoss`, `qwen3_5`,
each with `vllm/` and — ep1 only — `sglang/`); in a filename `-lora-`/`-full-` is the adapter and
`-ep1`/`-ep4` the expert distribution. Start from
`qwen3_5/vllm/qwen3.6-35b-a3b-react-math-full-ep4.yaml` (template:
`environmental-grpo-template.yaml`), `qwen3_5/vllm/qwen3.6-35b-a3b-code-contests-lora-ep1.yaml` for
a graded tool-heavy one, or `gptoss/sglang/gptoss-20b-code-contests-full-ep1.yaml` on SGLang. Each
reads `EnvironmentConfig` + `AsyncTrainingConfig` from YAML.

```bash
# vLLM (tool parser matters here — env GRPO uses /v1/chat/completions with tools)
VLLM_MODEL=Qwen/Qwen3.6-35B-A3B VLLM_TOOL_PARSER=hermes \
  docker compose -f docker-compose.vllm.yml up vllm-server

# Trainer — single entry point; resolves the env from environment_type in the YAML
CUDA_VISIBLE_DEVICES=1 accelerate launch \
  scripts/training/environmental_grpo.py \
  examples/grpo/environmental/qwen3_5/vllm/qwen3.6-35b-a3b-react-math-full-ep4.yaml

# EP/TP: same script under torchrun + parallel flags
torchrun --nproc_per_node=8 \
  scripts/training/environmental_grpo.py \
  examples/grpo/environmental/qwen3_5/vllm/qwen3.6-35b-a3b-code-contests-lora-ep1.yaml \
  --expert_parallel_size=8
```

`scripts/training/environmental_grpo.py` is the single async-GRPO
entry point: it resolves the env from `environment_type` in the YAML (`accelerate
launch` for plain DP, `torchrun` for EP/TP/ETP). `halo launch environmental-grpo <config>
--nproc N` builds the same line. For a non-registry environment,
call `register_environment(name, factory)` at import time and set that name as
`environment_type`.

Minimal async-GRPO YAML shape (the load-bearing keys):

```yaml
model_name_or_path: Qwen/Qwen3-4B-Instruct-2507
attn_implementation: flash_attention_2

# Environment (EnvironmentConfig)
environment_type: react_math
rewards:
  - source: environment         # weight 1.0, exponent 1.0 — the episode grade
max_turns: 10
# environment_kwargs: { search_backend: duckduckgo }

# Rollout server + Ray (AsyncTrainingConfig)
rollout_server_url: "http://localhost:8000"
vllm_group_port: 51216          # TRL GRPOConfig field, read by the NCCL client
rollout_connection_timeout: 120.0
sync_weights_every_n_steps: 1
num_rollout_workers: 4
rollout_temperature: 0.7
rollout_top_p: 0.95
rollout_max_tokens: 512
enable_prefetch: true

# GRPO hyperparameters (GRPOConfig)
num_generations: 4
beta: 0.01                      # 0.0 disables the ref model
epsilon: 0.2
scale_rewards: group           # TRL takes 'group' | 'batch' | 'none'
max_prompt_length: 512
max_completion_length: 512
```
