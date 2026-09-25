# Online GRPO (RLVR)

RLVR (Reinforcement Learning with Verifiable Rewards) is on-policy GRPO scored by the config's [reward terms](rewards.md): `\boxed{}` exact match by default, a regex format check, a generative judge or a served reward model. Trainer `DistributedGRPOTrainer`, script `scripts/training/online_grpo/rlvr.py`. Generation runs on a separate vLLM server that receives weights over NCCL; rewards are scored in the trainer.

Parallelism: EP, TP, EP+TP, EP+ETP and pure ETP (`ep_size=1`). CP and PP are rejected at config time.

Use [Offline GRPO](offline-grpo.md) for pre-collected data, [Async GRPO with Environments](async-grpo/README.md) for multi-turn, tool-calling or judge-scored tasks, [SMPO](../preference/smpo.md) or DPO for pairwise data ([overview](README.md)).

![Online GRPO (RLVR) as one step-long cycle: each prompt and answer is repeated num_generations times, the trainer renders and tokenizes the prompt, the vLLM server (separate container, GPU 7 by compose default) returns completions with their sampling log-probs, the reward terms score them into a weighted reward, the group normalizes it into advantages, the GRPO loss steps the model, and the new weights go back over NCCL before the next generation](../../assets/diagrams/online_grpo_pipeline.png)

## Dataset

Two columns: `prompt` (a string or a `{"role", "content"}` message list) and `answer` (the ground truth).

```jsonl
{"prompt": "What is 2 + 3? Put your answer in \\boxed{}.", "answer": "5"}
{"prompt": [{"role": "user", "content": "What is 2 + 3?"}], "answer": "5"}
```

`prompt_field` and `answer_field` must name real columns — a typo raises at load instead of yielding all-zero rewards. `system_prompt` is prepended only when the row has no system turn. A rendered prompt over `max_prompt_length` is **dropped**, never truncated. The recipes pull `openai/gsm8k`, `trl-lib/DeepMath-103K` and `open-r1/DAPO-Math-17k-Processed`.

## Rewards

The reward is the `rewards:` list of [terms](rewards.md): each becomes one TRL reward function named after the term, weighted through `GRPOConfig.reward_weights`, and TRL logs it as `rewards/<name>/mean`. The default list is the one `accuracy` term. Sources here: `accuracy`, `format`, `judge`, `reward_model`.

| `source` | Scores `1.0` when | Options |
|---|---|---|
| `accuracy` | the last `\boxed{...}` equals `answer` | none |
| `format` | the completion matches `pattern` | `pattern`, default `<think>.*?</think>\s*<answer>.*?</answer>` under `re.DOTALL` |

Accuracy normalizes both sides: the answer keeps only the text after a GSM8K `####` marker, both drop `,` and `$`, and `extract_last_boxed` (`src/rewards/matching.py`) matches braces by depth. An empty extraction never scores; the match is strict, while the environments' chain also lowercases and accepts `rtol=0.01` numeric matches. The graders live in `src/rewards/verifiable.py`.

A `judge` or `reward_model` term is an async reward function TRL gathers on its own loop; a judge reads the rendered `answer` column as the reference. The list is YAML-only: a container field cannot be set from the CLI.

## Configuration

The script parses `RLVROnlineGRPOScriptArguments`, TRL's `GRPOConfig`, `ModelConfig` and `DistributedArguments` ([fields](../../reference/configuration-reference.md#rlvronlinegrposcriptarguments)).

```yaml
model_name_or_path: Qwen/Qwen3-4B-Instruct-2507
dataset: trl-lib/DeepMath-103K
answer_field: solution

beta: 0.0                     # no KL anchor, no reference model
num_generations: 8            # the GRPO group size
max_prompt_length: 1024       # dataset filter
max_completion_length: 2048   # generation budget; required

use_vllm: true                # server mode is mandatory
vllm_mode: server
vllm_server_port: 8000

learning_rate: 1.0e-06
gradient_accumulation_steps: 8
```

`max_completion_length` is the budget TRL hands vLLM, so it must be a positive int; `null` raises. Bounded on both knobs, their sum becomes `model_max_length`.

TRL defaults `loss_type` to `dapo` and `scale_rewards` to `group`. The recipes under `examples/grpo/online/` all run `loss_type: grpo`, `beta: 0`, `epsilon` `0.15`–`0.2`, `num_generations` 4 or 8, and `max_completion_length` 128–4096.

Five optional objective changes ride on the script arguments, all off by default: `advantage_mode` (`qae` / `asymmetric` / `neg_mask_hard` — [Advantages](async-grpo/objective.md#advantages)), `scale_rewards_std_floor` (a floor on the std divisor), `drop_degenerate_groups` (mask all-equal-reward groups out of the loss and normalizer), `use_rlrr` (intra-group ranking advantages, [arXiv:2601.23058](https://arxiv.org/abs/2601.23058)) and `use_sdpg` ([Online SDPG](../distillation/online-sdpg.md)).

The first four recompute on the gathered reward set, so they raise unless `multi_objective_aggregation` is `sum_then_normalize`, and RLRR excludes the other three. `neg_mask_hard` gates on the **total weighted reward**, not the accuracy reward alone.

RLRR's nine tunables (`RLRRConfig`, `src/args/mixins.py`): `rlrr_mode` (`hrr` default, or `prr`), `rlrr_tau` (`0.1`), `rlrr_lambda` (`2048.0` — the config field is `lam`, `lambda` being a keyword), `rlrr_xi_pos` / `rlrr_xi_neg` (the Eq. 5 clip band, `1e-3` / `-1e-3`; `xi_neg > xi_pos` is refused), `rlrr_std_normalize` (`false`), `rlrr_length_rerank` (`true`), `rlrr_correctness_clip` (`true`) and `rlrr_correctness_threshold` (`0.5` — the only correctness signal; no path supplies gold labels). All nine are range-validated whether or not `use_rlrr` is on, and refused at a non-default value with it off. Worked recipe: `examples/grpo/online/qwen3/online-grpo-qwen3-8b-rlrr-math.yaml`.

**Importance-sampling correction** (`vllm_importance_sampling_correction`, TRL default on) weights the loss by `exp(logπ_recompute − logπ_sampling)`; under the default `vllm_importance_sampling_mode: sequence_mask` a sequence whose ratio leaves `[clip_min, clip_max]` (`3.0`) is masked out of the loss, and the `*_truncate` modes clamp it instead. It gates the KL tail clamp, so `beta > 0` with it off warns.

A rank-0 startup probe refuses a server whose logprobs are raw pre-temperature values at `temperature != 1.0` — serve vLLM with `--logprobs-mode processed_logprobs`. It also refuses nucleus-renormalized logprobs under `top_p < 1` while the correction runs a `sequence_*` mode, which stalls the run silently; keep `top_p: 1.0`.

**Dropout is forced off** after the parallel modes are realized, so it reaches EP grouped expert-LoRA dropout and overrides `disable_dropout: false`: active dropout biases every recomputed log-prob against vLLM's dropout-free sampling. A config-level float only warns.

**MoE balancing is off on this path**: `aux_loss` is inert under a policy-gradient loss, and both bias-update modes downgrade to `none` because the weight sync ships parameters only ([Callbacks](../callbacks.md#routerbiasbalancingcallback)). `reasoning_effort` (`low` / `medium` / `high` / `random`) is applied at dataset-map time and needs a template that reads it.

## GRPO objective for verifiable rewards

A graded-fraction reward in `[0, 1]` over a small group wants a different objective than the math defaults — this is the reward's *shape*, not the task. The knobs are TRL `GRPOConfig` fields and apply equally to async GRPO.

| Knob | Set it to | Why |
|---|---|---|
| `loss_type` | `dapo` (TRL's default) | Normalizes over the global active-token count, so it is length-unbiased; `grpo`'s per-sequence mean is not |
| `epsilon_high` | `0.28` with `epsilon: 0.2` | DAPO clip-higher: only the upper bound loosens |
| `scale_rewards` | `batch` for a shaped reward | `group` divides by the group's own spread, turning shaping noise into full-scale advantages |
| `mask_truncated_completions` | `true` (default `false`) | Drops a completion cut at the budget instead of scoring it a loss |

A single binary reward is the exception to the `scale_rewards` row: a uniformly-failed group's std is exactly 0, and the `1e-4` added to the divisor leaves its advantages at zero rather than NaN, so the recipes keep `group`.

The clip binds only when a generation batch outlives one optimizer step. At the defaults (`num_iterations: 1`, `steps_per_generation` = `gradient_accumulation_steps`) `old_per_token_logps` share the loss forward's weights, so the ratio is 1 and `epsilon` is inert. `epsilon_high` falls back to `epsilon`.

`beta` is the KL anchor to the base model, and every recipe runs `beta: 0`. Without PEFT, `beta != 0` makes TRL build its own reference: an fp32 dense per-rank replica from the hub's default revision, warned about as wasteful under EP.

That reference is **rejected when the policy carries live attention sinks** (`reset_sinks: false`), in every mode — it is not sink-restricted, so the KL is biased on every token. Use `beta: 0`, or `use_peft: true` with an attention target; expert-only LoRA is never PEFT-wrapped, so TRL still builds one.

## Chat template handling

The script renders each prompt to text with the trainer's tokenizer (`render_generation_prompt`, `src/data/pipeline/rendered.py`); TRL tokenizes that text and sends **token ids** to vLLM, which applies no template of its own. A `tools_field` column rides through as `tools=`.

The rendered leading BOS is stripped **only** when the tokenizer's post-processor prepends one of its own; stripping unconditionally would delete BOS where the template emits it and the post-processor does not (Gemma 4). The tokenizer is the single source of truth, so a `chat_template:` override needs no server-side setting — unlike async GRPO ([Chat template](async-grpo/rollouts.md#chat-template)).

## Chunked log-probs

`use_chunked_grpo_logprobs: true` (default off) computes the completion log-probs from the backbone's `last_hidden_state` and a vocab-chunked softmax instead of a full `[B, T, vocab]` logits tensor, bounding the loss-forward peak. Turn it on for large-vocab models on long completions.

Text-only: a multimodal batch routes every rank back to the full path, and a PEFT adapter on `lm_head` raises. [Offline GRPO](offline-grpo.md#memory-chunked-log-probs) shares the switch ([limits](async-grpo/performance.md#chunked-log-probs)).

## Server mode only

A colocated in-process `vllm.LLM()` builds its own NCCL communicators on the training GPUs, and two NCCL worlds sharing devices and streams under `CUDA_DEVICE_MAX_CONNECTIONS=1` deadlock, so `DistributedGRPOTrainer` rejects `use_vllm: false` and `vllm_mode: colocate` at init ([Rollout Servers](../../infrastructure/rollout-servers.md#vllm)).

A rank-0 preflight reads the served `max_model_len` from `/v1/models` and raises on every rank when `max_prompt_length + max_completion_length` exceeds it, before the trainer is built.

> [!WARNING]
> **GPT-OSS: toolkit vLLM image, sinks on**
>
> Serve GPT-OSS from the toolkit `vllm-server:0.26.0` image with sinks on (the default) and keep `reset_sinks: false` on the trainer; stock upstream vLLM produces garbage GPT-OSS output on Blackwell ([GPT-OSS](../../models/gpt-oss.md#serving-for-grpo-vllm)).

### Weight sync

Weights are pushed before the **next** generation, not at the optimizer step: TRL syncs when `state.global_step` has moved since `_last_loaded_step`. That sentinel is per-process (`-1`) and `TrainerState` never carries it, so a resumed run pushes before its first rollout ([mechanics](../../infrastructure/rollout-servers.md#weight-sync)).

`validate_weight_sync_support` refuses seven shapes at construction:

- QLoRA (`load_in_4bit` / `load_in_8bit`): bnb-packed storage corrupts the served policy.
- GptOss sinks removed by the `flash_attention_2` `reset_sinks` reset, and `train_sinks: true`.
- Model types in the client's `UNSERVABLE_MODEL_TYPES`, and EP families setting `_supports_weight_sync = False` ([per-family restrictions](../../parallelism/expert-parallelism.md#per-family-ep-restrictions)).
- An EP family with no live EP wrapper: `ep_size: 1` with `use_grouped_gemm: false`.
- Live `bias_update` balancing state: the payload is parameters only.

### Multi-homed nodes (`VLLM_GROUP_HOST`)

Rank 0 advertises an address for the vLLM workers to dial back and binds the NCCL weight-sync TCPStore on that address alone ([group rendezvous](../../infrastructure/rollout-servers.md#group-rendezvous)). Resolution order: an explicit `group_host` argument → `VLLM_GROUP_HOST` → **loopback when the server address is one this host itself binds** → default-route NIC.

Loopback for a local server is deliberate: an external NIC there lets a provider firewall drop the hairpin traffic, and the first collective spins forever. Set `VLLM_GROUP_HOST` to the trainer IP a **remote** server should dial when the default route is not on its subnet. It is the control-plane address, not `NCCL_SOCKET_IFNAME`.

> [!NOTE]
> **Multi-server sync belongs to Async GRPO with Environments**
>
> The list-of-dicts `server_configs` API (per-server `url`/`group_port`/`group_host` via `InferenceClientManager`) belongs to [Async GRPO with Environments](async-grpo/setup.md) (`AsyncTrainingConfig.rollout_server_configs`), not this single-server RLVR trainer.

## Launch

vLLM must own GPUs **no trainer rank uses** — NCCL cannot share one between them. The compose defaults split them: server on GPU 7 (`VLLM_CUDA_DEVICES`), trainer on 0–6 (`TRAINER_CUDA_DEVICES`).

```bash
# vLLM on GPU 7 (server flags and variables: Rollout Servers)
VLLM_MODEL=Qwen/Qwen3-30B-A3B docker compose -f docker-compose.vllm.yml up vllm-server

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 torchrun --nproc_per_node=7 \
    scripts/training/online_grpo/rlvr.py examples/grpo/online/rlvr-online-grpo-template.yaml \
    --model_name_or_path=Qwen/Qwen3-30B-A3B
```

`halo launch rlvr <config> --nproc 7` builds the same `torchrun` line — the device split stays yours — landing on FSDP2 data parallelism. Add `--tensor_parallel_size` for dense TP or `--expert_parallel_size` for MoE.

Set `ep_size` to the training-GPU count so the trainer ranks form **one** DeepEP group. Above `ep_size: 2` a group narrower than the NVLink domain is rejected ([DeepEP](../../infrastructure/deepep.md#ep-grouping-what-is-reliable)), and `num_experts` must divide by `ep_size` — so most rosters want a power-of-two split: trainer on `CUDA_VISIBLE_DEVICES=0,1,2,3` with `--expert_parallel_size=4`, server on `VLLM_CUDA_DEVICES=4,5,6,7`. Under EP+TP, `tp_size` must divide the NVLink domain and `ep_size` must be a multiple of it.

### LoRA

Add `use_peft: true` and the `lora_*` fields. LoRA runs under FSDP2 DP, EP and pure ETP; any adapter on the TP-sharded backbone is **rejected under TP and EP+TP** (`_validate_lora_tp_compatibility`, `src/trainers/mixins/validation.py`), because PEFT keeps `lora_A`/`lora_B` as plain tensors outside the TP graph. Native expert adapters too.

The weight sync merges the adapter into the base before broadcasting, so vLLM serves the plain base; under EP the gather folds native expert-LoRA in too ([PEFT](../../optimization/peft.md#hyperparameters)).

## Data flow and batch construction

One optimizer step: each `{prompt, answer}` row is rendered and replicated `num_generations`× into a group, `_generate_single_turn` sends the token ids to vLLM, completions come back right-**padded** with a `completion_mask` that is `1` only on generated tokens, the rewards score them, and `scale_rewards` normalizes each group into advantages. Under TP the group leader generates and broadcasts, so every rank forwards identical tokens.

**Collator.** GRPO uses TRL's dataset-**row** collator, not the SFT [collators](../../data/collators.md): left-padded prompts, right-padded completions, never packed — hence no CP.

**Counting.** Rollouts per optimizer step = `per_device_train_batch_size × gradient_accumulation_steps × data_parallel_size`. `per_device_train_batch_size` counts completions, since `RepeatSampler` already repeated each prompt; times the completion length it sets the per-forward memory. `num_generations` trades unique prompts for samples per prompt.

**Geometry.** Under TP, ETP or a pre-sharded dataset the dataloader rebuilds `RepeatSampler` at its own rate (`src/trainers/grpo/mixins/dataloader.py`): `per_device_train_batch_size × steps_per_generation × data_parallel_size` must divide by `num_generations`, and a pre-sharded dataset drops the DP factor. Under TP/ETP the per-rank product must **itself** divide by it, since TP siblings inject duplicate slices into TRL's world-order gather. Both raise when the dataloader is built.

## vLLM on Blackwell (B200)

Server-side, pin `VLLM_ATTENTION_BACKEND=FLASH_ATTN` only where FlashInfer JIT-fails on SM 10.0; on head-dim-256 families it resolves to FA2 and measures slower ([Rollout Servers](../../infrastructure/rollout-servers.md#throughput)).

Trainer-side, all three GRPO scripts default `attn_implementation` to **SDPA** for their right-padded batches, and drop that default under `reset_sinks: false`, where only a sink-carrying implementation is accepted ([padded workloads](../../optimization/flash-attention.md#model-specific-handling)). A pinned value wins.

## Testing a setup

Run a smoke config against a live server first — `examples/grpo/online/qwen3/online-grpo-qwen3-4b-smoke.yaml` (dense) or `.../gptoss/online-grpo-gptoss-20b-ep-smoke.yaml` (EP). Each sets `max_steps: 3` and `save_strategy: "no"`, exercising rendering, rewards, the loss and one weight sync in minutes.

CPU: `pytest tests/cpu/grpo -m cpu`. GPU: `tests/gpu/trainers/grpo/test_online_grpo_mock.py` needs no server; the `test_online_grpo_vllm_*_e2e.py` suites run the PEFT × parallelism × resume matrix against one.

## What to watch

Metric names follow TRL's `GRPOTrainer`, plus `kl_clamp_frac` (reference log-ratios hitting the 5-nat clamp, at `beta > 0`) and `sampling/degenerate_group_frac`. Read every run: `rewards/accuracy/mean`, `frac_reward_zero_std`, `sampling/importance_sampling_ratio/mean` (near 1 means trainer and engine agree), `completions/clipped_ratio`, `entropy`, and `<name>/scored_frac` per externally scored term — the share of calls that returned a usable verdict, so a failing judge shows up as a falling fraction rather than a quiet zero ([Reward Terms](rewards.md#generative-judge)).

`save_completions` (default on) writes `<output_dir>/completions/completions_<step>.parquet` (step zero-padded to five digits); `log_completions` is console-only.

| Symptom | Cause | Fix |
|---|---|---|
| All rewards zero | The policy emits no `\boxed{}` | Ask for it in `system_prompt`, raise `max_completion_length`, read the parquet |
| IS ratio far from 1 | Trainer and engine disagree (template, sinks, dropout) | Match the served model; `reset_sinks: false` for GPT-OSS |
| Hang at sync or generation | Server down or wedged, a GPU shared with a trainer rank, blocked ports | `curl .../health`; check both containers' `CUDA_VISIBLE_DEVICES`; raise `DIST_NCCL_TIMEOUT_MINUTES` (default `30`); [Rollout Servers](../../infrastructure/rollout-servers.md#troubleshooting) |
| OOM | Full-vocab logits or a long completion | `use_chunked_grpo_logprobs: true`, lower the batch or `max_completion_length`, raise EP size |

GRPO perturbs a tuned policy, so the learning rate sits near the SFT floor: `1e-6` to `5e-6` in the recipes, roughly 10× that for LoRA ([sizing](../sft.md#learning-rate-and-global-batch-size)).

## Related pages

- [Offline GRPO](offline-grpo.md) · [Async GRPO with Environments](async-grpo/README.md) · [GRPO overview](README.md)
- [Rollout Servers](../../infrastructure/rollout-servers.md) · [Expert Parallelism](../../parallelism/expert-parallelism.md) · [Trainer Architecture](../../reference/trainer-architecture.md)
