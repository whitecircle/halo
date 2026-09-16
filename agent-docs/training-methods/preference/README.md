# Preference Optimization

Three methods learn from comparisons and KTO from per-row binary labels. Pick by the shape of the rows you have.

| Row shape | Method | Why |
|---|---|---|
| `prompt` / `chosen` / `rejected` | [SMPO](smpo.md) | No reference model; the margin loss stops once a pair separates |
| `prompt` / `chosen` / `rejected` | [DPO](dpo.md) | Reference-based; 15 TRL loss types, combinable |
| `prompt` / `completion` / `label` | [KTO](kto.md) | Unpaired thumbs-up/down feedback |
| `prompt` / `chosen` / `rejected` → scorer | [Reward modeling](reward-modeling.md) | A Bradley-Terry score head for rejection sampling or RL |

Other shapes: several scored completions per prompt → [Offline GRPO](../grpo/offline-grpo.md); generation during training → [Online GRPO](../grpo/online-grpo.md) or [Async GRPO with Environments](../grpo/async-grpo/README.md); one good completion per prompt → [SFT](../sft.md).

## Dataset format

The three pairwise methods share one format, all fields `list[dict]` messages:

```jsonl
{"prompt": [{"role": "user", "content": "What is the capital of France?"}], "chosen": [{"role": "assistant", "content": "Paris is the capital of France."}], "rejected": [{"role": "assistant", "content": "France is in Europe."}]}
```

`prompt` is the conversation up to divergence; `chosen` and `rejected` are the competing completions. Reward modeling also reads implicit-prompt datasets. KTO's unpaired shape is on its own page. Full contract: [Dataset Formats](../../data/dataset-formats.md#preference-dposmpo).

## Parallelism

All four run EP, TP, ETP and EP+TP. SMPO alone declares CP support: it CP-aggregates its per-sequence log-prob sums, while DPO and KTO run TRL's CP-unaware loss path and the reward head needs the whole sequence to pool. All four also declare `_supports_pp`, but pipeline parallelism is [not yet available in this release](../../parallelism/pipeline-parallelism.md); its shipped gates would take DPO and KTO with precomputed reference log-probs only. Full matrix: [Trainer Compatibility](../../reference/trainer-architecture.md#trainer-compatibility).

## Launch

```bash
torchrun --nproc_per_node=8 scripts/training/preference/smpo.py \
    examples/preference/gptoss/smpo-gptoss-20b-tulu3-prefmix-ep.yaml \
    --expert_parallel_size=8
```

`halo launch smpo <config> --nproc 8` builds the same line; the other methods are `dpo`, `kto` and `preference/rewards`.
