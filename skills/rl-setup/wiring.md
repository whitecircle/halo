# rl-setup — launch examples

The facts behind these commands live on their owner pages; read them, not a copy:

- server images, build gates, compose knobs, weight sync, transport, per-family serving:
  `agent-docs/infrastructure/rollout-servers.md`;
- every `AsyncRolloutConfig` / `EnvironmentConfig` / `GRPOConfig` field with its default:
  `agent-docs/reference/configuration-reference.md`;
- the environment registry and per-environment knobs:
  `agent-docs/training-methods/async-grpo/environments/README.md#registry`;
- the `rewards:` term list: `agent-docs/training-methods/grpo/rewards/README.md`;
- method setup: `agent-docs/training-methods/grpo/online.md`,
  `agent-docs/training-methods/async-grpo/setup.md`.

## Launch examples

### Online GRPO (RLVR) — rule-based verifiable rewards

Config: `examples/grpo/online/qwen3/online-grpo-qwen3-4b-smoke.yaml`
(template: `rlvr-online-grpo-template.yaml`). Uses TRL-native vLLM fields
(`use_vllm: true`, `vllm_mode: server`, `vllm_server_host`, `vllm_server_port`) and the same
`rewards:` term list, defaulting to `[{source: accuracy}]` — the RLVR graders `accuracy` and
`format`, plus `judge` / `reward_model` (`halo/rewards/graders/verifiable.py`).

Server and trainer GPUs must be disjoint (a rank cannot NCCL-broadcast to itself), and the server
must serve the trained model — bare compose serves `Qwen/Qwen3-0.6B` on GPU 7.

```bash
# Dense: server on GPU 0, FSDP2 DP on 1-3 (the recipe's own split)
VLLM_MODEL=Qwen/Qwen3-4B-Instruct-2507 VLLM_CUDA_DEVICES=0 \
  docker compose -f docker-compose.vllm.yml up -d vllm-server
CUDA_VISIBLE_DEVICES=1,2,3 torchrun --nproc_per_node=3 \
  scripts/training/grpo/online.py \
  examples/grpo/online/qwen3/online-grpo-qwen3-4b-smoke.yaml

# MoE with EP: one DeepEP group across the trainer GPUs (ep_size = trainer-GPU count)
VLLM_MODEL=Qwen/Qwen3.6-35B-A3B VLLM_CUDA_DEVICES=4,5,6,7 \
  docker compose -f docker-compose.vllm.yml up -d vllm-server
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 \
  scripts/training/grpo/online.py \
  examples/grpo/online/qwen3_5/online-grpo-qwen3.6-35b-a3b-dapo-math.yaml \
  --expert_parallel_size=4
```

### Async GRPO with Environments — multi-turn tool-use

Recipes live at `examples/grpo/environmental/<family>/<backend>/` (`gemma4`, `gptoss`, `qwen3_5`,
each with `vllm/` and — ep1 only — `sglang/`); in a filename `-lora-`/`-full-` is the adapter and
`-ep1`/`-ep4` the expert distribution. Start from
`qwen3_5/vllm/qwen3.6-35b-a3b-react-math-full-ep4.yaml` (template:
`environmental-grpo-template.yaml`), `qwen3_5/vllm/qwen3.6-35b-a3b-code-contests-lora-ep1.yaml` for
a graded tool-heavy one, or `gptoss/sglang/gptoss-20b-code-contests-full-ep1.yaml` on SGLang. Each
reads `EnvironmentConfig` + `AsyncRolloutConfig` from YAML.

Each recipe's header carries its server flags and GPU split. React-math (EP=4) serves on GPUs 4-7
with `--reasoning-parser qwen3` and no tool-call parser — react_math reads its Action from the
text — so compose runs it with the tool-calling flags emptied:

```bash
VLLM_MODEL=Qwen/Qwen3.6-35B-A3B VLLM_CUDA_DEVICES=4,5,6,7 VLLM_REASONING_PARSER=qwen3 \
VLLM_TOOL_CALLING_FLAGS= docker compose -f docker-compose.vllm.yml up -d vllm-server

# Trainer — single entry point; resolves the env from environment_type in the YAML
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 \
  scripts/training/grpo/environmental.py \
  examples/grpo/environmental/qwen3_5/vllm/qwen3.6-35b-a3b-react-math-full-ep4.yaml
```

Launch with `torchrun` (`halo launch grpo/environmental <config> --nproc N` builds the line); a
non-registry environment registers through `register_environment(name, factory)` at import time.

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

# Rollout server + Ray (AsyncRolloutConfig)
rollout_server_url: "http://localhost:8000"
rollout_group_port: 51216       # NCCL weight-sync group port, bound on the trainer host
rollout_connection_timeout: 120.0
sync_weights_every_n_steps: 1
num_rollout_workers: 4
rollout_temperature: 0.7
rollout_top_p: 0.95
rollout_max_tokens: 512         # per-turn generation cap; the script pins max_completion_length to it
enable_prefetch: false          # needs 2+ rollout_server_configs; one server auto-disables it

# GRPO hyperparameters (GRPOConfig)
num_generations: 4
beta: 0.01                      # 0.0 disables the ref model
epsilon: 0.2
scale_rewards: batch           # 'batch' (the recipes') | 'group' | 'none'
max_prompt_length: 512
```
