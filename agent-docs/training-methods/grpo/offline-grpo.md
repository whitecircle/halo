# Offline GRPO

Offline GRPO trains on pre-collected, off-policy data: several completions per prompt, each with a pre-computed reward, and no generation during training. Trainer `OfflineGRPOTrainer`, script `scripts/training/offline_grpo.py`, config `OfflineGRPOConfig` ([field reference](../../reference/configuration-reference.md#offlinegrpoconfig)).

Use it when the completions already exist, when scoring them is expensive, or when a reproducible run matters. For live generation use [Online GRPO](online-grpo.md) or [Async GRPO with Environments](async-grpo/README.md); for pairwise data, [SMPO](../preference/smpo.md) or DPO ([GRPO overview](README.md) compares all three).

Parallelism: EP, TP, EP+TP, EP+ETP and pure ETP (`ep_size=1`). CP is rejected at config time — the forward is trimmed with `logits_to_keep`, which CP's sequence splitting breaks; use [SMPO](../preference/smpo.md) for long sequences there. The trainer declares `_supports_pp`, but [pipeline parallelism](../../parallelism/pipeline-parallelism.md) is not yet available in this release.

![Offline GRPO in two bands: tokenization turns one dataset row (prompt, completions, rewards) into per-group advantages and then one training row per completion carrying its advantage, group_id and group_size; each training step draws a micro-batch from MultiGroupSampler, computes the per-token loss with the min_log_prob floor and the optional k3 KL, and normalizes it with bnpo, grpo or dr_grpo, every row weighted 1/group_size](../../assets/diagrams/offline_grpo_pipeline.png)

## Dataset

Three conversational columns. A group is one **row**, keyed by row index: put every completion of a prompt in one row, since two rows with the same prompt stay two groups.

| Column | Type | Contents |
|---|---|---|
| `prompt` | `list[dict]` | Message list (`{"role", "content"}`) |
| `completions` | `list[list[dict]]` | One message list per completion (typically 4–16) |
| `rewards` | `list[float]` | One reward per completion |

```jsonl
{"prompt": [{"role": "user", "content": "What is 2+2?"}], "completions": [[{"role": "assistant", "content": "4"}], [{"role": "assistant", "content": "5"}]], "rewards": [1.0, -0.5]}
```

A singleton group, and any exactly-tied one, normalizes to advantage 0 under every method. A `completions`/`rewards` length mismatch raises with the row index; a non-finite reward raises with the offending group.

`scripts/inference/reward_model/rm_rejection_sampling.py` emits this shape with `--output_format offline_grpo`.

The training script renders `template(prompt + completion)` and strips the rendered-prompt prefix, so a strict template (Qwen3.5) never sees an assistant-only message list. A `tools_field` column is passed as `tools=`.

## Configuration

```yaml
model_name_or_path: Qwen/Qwen3.6-35B-A3B
dataset:
  - s3://bucket/gsm8k-qwen3.6-35b-a3b-offline-grpo
advantage_method: quantile_norm
loss_type: bnpo
kl_beta: 0.0
initial_min_log_prob: -0.5
max_prompt_length: 512
max_completion_length: 6144
expert_parallel_size: 8
learning_rate: 5.0e-06
```

| Knob | Default | Effect |
|---|---|---|
| `advantage_method` | `quantile_norm` | Reward → advantage within each group (table below) |
| `best_completion_emphasis` | `0.0` | Boost the top-reward rows by a factor above `1.0`, or `auto` |
| `loss_type` | `bnpo` | Loss normalization: `bnpo`, `grpo`, `dr_grpo` |
| `policy_gradient_formulation` | `prob_weighted` | Per token `-(π·A)`; `reinforce` uses `-(log π·A)` |
| `min_log_prob` | `-3.0` | Log-prob floor, on negative-advantage rows only |
| `initial_min_log_prob` | `null` | Linearly schedule that floor from this start |
| `kl_beta` | `0.0` | Coefficient of the k3 KL penalty to a reference policy |
| `drop_degenerate_groups` | `false` | Drop exactly-tied and singleton groups at tokenization |
| `use_chunked_grpo_logprobs` | `false` | Vocab-chunked log-probs, not full logits |
| `max_prompt_length` | `512` | Prompt budget, left-truncated; `null` = no cap |
| `max_completion_length` | `null` | Completion budget, cut from the end; `null` = no cap |
| `max_length` | `null` | Pipeline-parallel only; rejected at construction off PP — on every run, since [PP is not yet available](../../parallelism/pipeline-parallelism.md) |

The shipped recipes run `learning_rate: 5.0e-06`: GRPO refines a tuned policy, so the rate sits below the SFT band. Advantages are group-relative, so a wider batch helps ([sizing](../sft.md#learning-rate-and-global-batch-size)).

Both length knobs are truncation caps, and the tokenizer's window is pinned only when both are bounded. At a cut the sequence gets no terminator — EOS is appended only when the completion ends inside its budget — so a completion cut at the cap is trained without a stop token rather than taught to stop there.

### Advantages

Rewards map to advantages per group during preprocessing, then clip to `[-10, 10]` — emphasis can push one past a method's own range.

| `advantage_method` | Formula | Best for |
|---|---|---|
| `quantile_norm` (default) | inverse normal CDF of ranks | ordinal or outlier-heavy rewards |
| `z_norm` | `(r - mean) / (std + ε)` | normal reward distributions |
| `minmax` | `2(r - min)/(max - min) - 1` | bounded `[-1, +1]` advantages |
| `quantile_uniform` | uniform from ranks | most outlier-tolerant |
| `robust` | `(r - median) / IQR` | extreme outliers |

`best_completion_emphasis` takes a float above `1.0` (`2.0` for 2× weight) or `"auto"`, which adapts to reward variance as `3.0 + 2.0·std/(1.0 + std)`. Any other number raises — negatives and `(0.0, 1.0]` alike — since the factor applies only above `1.0`.

`drop_degenerate_groups: true` drops **exactly**-tied groups and groups of fewer than two before they spend forward compute or dilute the loss normalizer, and raises if that empties the dataset. Near-ties are kept — the rank methods train them at full scale.

### Loss types

All three weight each example by `1/group_size`, so every group contributes equally however many completions it has.

- `bnpo` (default): global weighted token average, over the micro-batch's group-weighted token sum.
- `grpo`: per-sequence average, then a weighted mean across groups. Use when completion length varies.
- `dr_grpo`: normalized by effective group count × `max_completion_length` — a constant denominator that removes length bias. It is not a generation budget here, so a `null` or non-positive value raises at trainer init.

### Memory: chunked log-probs

The default path materializes `[B, T_completion, vocab]` logits, twice per micro-batch at `kl_beta > 0` — on wide vocabularies with long completions, the memory peak. `use_chunked_grpo_logprobs: true` computes the same log-probs from the backbone's `last_hidden_state` via a vocab-chunked softmax, covering the policy and both reference paths. It is inert under PP and warns there. Limits: [Chunked log-probs](async-grpo/performance.md#chunked-log-probs).

### Reference model

`kl_beta` is the KL anchor, and a reference exists only at `kl_beta > 0`. A PEFT-wrapped policy builds none, since `disable_adapter()` reverts to the base weights. Dense full fine-tuning deepcopies the live policy, so a resume re-anchors the KL there.

EP and grouped-GEMM wrapped MoE models hold live NCCL groups `deepcopy` cannot pickle, so their reference — and an expert-only LoRA run's, which is not PEFT-wrapped — is a dense per-rank replica the script loads through the DPO/KTO reference path (`load_frozen_reference_model`) and passes as `ref_model`. It carries the policy's weights source (the resumed checkpoint on a resume), revision, `trust_remote_code`, attention request and sinks policy, with the non-persistent buffers repaired. The trainer raises when a wrapped MoE arrives at `kl_beta > 0` without one, and when `ref_model` is passed to a run that holds no reference. The reference log-ratio is capped at 5 nats, bounding the k3 estimator's tail.

## Launch

```bash
torchrun --nproc_per_node=8 scripts/training/offline_grpo.py \
    examples/grpo/offline/qwen3_5/offline-grpo-qwen3.6-35b-a3b-gsm8k.yaml --expert_parallel_size=8
```

`halo launch offline-grpo <config> --nproc 8` builds the same line. From Python, load the model through `load_distributed_model` and pass the same `ParallelismConfig` to the trainer.

**MoE balancing.** A policy-gradient loss never adds the router aux term, so `moe_balancing: aux_loss` warns and does nothing. With no weight sync here, `bias_update` is the working choice on a family whose bias exports — unlike the on-policy trainers, where it is downgraded ([Callbacks](../callbacks.md#routerbiasbalancingcallback)); the shipped recipes set `none`.

## Testing a setup

Run a short config on a slice of the data first: `max_steps: 10` with `save_strategy: "no"` exercises tokenization, advantage normalization and the loss in minutes. A bad `loss_type`, `advantage_method` or `policy_gradient_formulation` is caught at construction.

CPU: `pytest tests/cpu/grpo -m cpu`. GPU: `tests/gpu/trainers/grpo/test_offline_grpo.py` and its `_bnpo` / `_bs4` / `_chunked` / `_tp_resume` siblings, plus the LoRA suites.

## What to watch

Per-sample diagnostics split by advantage sign, once per log step: `{positive,negative}/{logps,rewards,kl,ref_logps}_{mean,std,min,max,range}`, plus the pre-clamp `negative/*_unclamped_*`. The `kl` and `ref_logps` families exist only at `kl_beta > 0`, `min_log_prob_restriction` is the live clamp floor, and evaluation prefixes every key with `eval_`.

Read `positive/logps_mean` (stable or rising), `negative/logps_mean` (falling) and `positive/rewards_mean`.

| Symptom | Cause | Fix |
|---|---|---|
| Loss NaN or exploding | Low-probability tokens on negative-advantage rows | `min_log_prob: -3.0` and `max_grad_norm: 1.0` |
| Rewards fall through training | Distribution shift from the offline data | Add `kl_beta: 0.05`, lower the learning rate |
| `Batch-count equalization left 0 full batches` | The split is too small for this world size | Lower `per_device_train_batch_size`, shrink the world |
| OOM | Full-vocab logits over long completions | `use_chunked_grpo_logprobs: true`, lower the batch or `max_completion_length`, or raise EP size |

## Related pages

- [Online GRPO (RLVR)](online-grpo.md) · [Async GRPO with Environments](async-grpo/README.md) · [GRPO overview](README.md)
- [Expert Parallelism](../../parallelism/expert-parallelism.md) · [Pipeline Parallelism](../../parallelism/pipeline-parallelism.md)
- [Trainer Architecture](../../reference/trainer-architecture.md) · [Configuration Reference](../../reference/configuration-reference.md#offlinegrpoconfig)
