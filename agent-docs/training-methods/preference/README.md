# Preference Optimization

Two methods train on paired comparison data (chosen vs. rejected): **SMPO** — reference-model-free, with a dynamic-margin objective and an SFT anchor — and **DPO** (reference-model-based). For **unpaired** binary feedback (one labeled completion per prompt) use [KTO](kto.md). [Reward Modeling](reward-modeling.md) trains a Bradley-Terry scalar reward model from the same pairwise format.

## Methods at a glance

| Aspect | [SMPO](smpo.md) | [DPO](dpo.md) |
|--------|------|-----|
| Reference model | Not required | Required (or PEFT, or precomputed log probs) |
| Objective | Margin: `log p(chosen) - log p(rejected) >= margin` | KL-constrained optimal policy |
| Loss variants | `sigmoid`, `hinge`, `ipo`, `smooth_lower_bound` | 15 TRL types; `loss_type` is a list, so RPO = a preference loss combined with `sft` |
| Auxiliary SFT loss | Built-in (weighted chosen/rejected CE) | Only via `loss_type: [..., sft]` |
| Token-level clipping | Built-in percentile clipping | Not available |
| Margin scheduling | Curriculum via margin schedule | Not available |
| Trainer | `SmoothMarginPOTrainer` | `DistributedDPOTrainer` |
| Script | `scripts/training/preference/smpo.py` | `scripts/training/preference/dpo.py` |

Different shape? Multiple completions with reward scores → [Offline GRPO](../grpo/offline-grpo.md); on-policy generation during training → [Online GRPO](../grpo/online-grpo.md); a single good completion per prompt → [SFT](../sft.md).

## Dataset format

Both methods use the same pairwise format with `list[dict]` messages:

```jsonl
{"prompt": [{"role": "user", "content": "What is the capital of France?"}], "chosen": [{"role": "assistant", "content": "Paris is the capital of France."}], "rejected": [{"role": "assistant", "content": "France is in Europe."}]}
```

`prompt` is the conversation up to divergence; `chosen` and `rejected` are the competing completions.

## Parallelism

Both support EP, TP, ETP, and EP+TP; both declare `_supports_pp`, but pipeline parallelism is [not yet available in this release](../../parallelism/pipeline-parallelism.md). SMPO also supports CP (and EP+CP); DPO does not — its chosen/rejected forward needs global log-prob sums over full sequences, incompatible with sequence splitting. Full matrix: [Trainer Compatibility](../../reference/trainer-architecture.md#trainer-compatibility).

## Quick start

```bash
# SMPO on an MoE with EP (torchrun for EP/CP/TP)
torchrun --nproc_per_node=8 scripts/training/preference/smpo.py \
    examples/preference/gptoss/smpo-gptoss-20b-tulu3-prefmix-ep.yaml \
    --expert_parallel_size=8

# DPO
accelerate launch scripts/training/preference/dpo.py \
    examples/preference/qwen3_5/dpo-qwen3.5-9b-tulu3-prefmix.yaml
```
