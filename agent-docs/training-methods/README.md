# Training Methods

Every method is YAML-driven, runs through the same distributed parallelism system, and ships example configs under `examples/`.

## Methods at a glance

| Method | Trainer | Dataset columns | When to use |
|---|---|---|---|
| [SFT](sft.md) | `DistributedSFTTrainer` | `prompt` (message list) | Instruction tuning, text or vision-language; also [pre-training](pretraining.md), same trainer and script |
| [DPO](preference/dpo.md) | `DistributedDPOTrainer` | `prompt`, `chosen`, `rejected` | Paired preference data, against a reference model; 15 TRL `loss_type` values, combinable |
| [KTO](preference/kto.md) | `DistributedKTOTrainer` | `prompt`, `completion`, `label` | Unpaired binary feedback — one completion, thumbs up or down |
| [SMPO](preference/smpo.md) | `SmoothMarginPOTrainer` | `prompt`, `chosen`, `rejected` | Preference tuning with no reference model to hold in memory |
| [Offline GRPO](grpo/offline-grpo.md) | `OfflineGRPOTrainer` | `prompt`, `completions`, `rewards` | Completions already generated and scored |
| [Online GRPO (RLVR)](grpo/online-grpo.md) | `DistributedGRPOTrainer` | `prompt`, `answer` | Single-turn verifiable rewards (math, format) against a live vLLM server |
| [Async GRPO with Environments](grpo/async-grpo/README.md) | `DistributedAsyncEnvironmentalGRPOTrainer` | `prompt` (+ `answer` where the environment grades against one) | Multi-turn RL with tools, sandboxes or a judge, on vLLM or SGLang |
| [Reward modeling](preference/reward-modeling.md) | `DistributedRewardTrainer` | `chosen`, `rejected` | A Bradley-Terry scorer to rank completions |
| [Classification](classification.md) | `ClassificationTrainer` | `prompt` or `text_field`, `label` | Single- and multi-label sequence classification |
| [Distillation](distillation/README.md) | `DistributedDistillationTrainer`, `DistributedSelfDistillationTrainer`, `DistributedSDPGTrainer` | `messages` conversations | Compress a teacher, or self-distill from a privileged hint (offline or online SDPG) |
| [Embedding](embedding.md) | `EmbeddingTrainer` | text pairs or triplets, optional `label` / `score` | Retrieval and similarity models, 10 SBERT losses and Matryoshka |

Column types and per-script defaults: [Dataset Formats](../data/dataset-formats.md#required-columns-by-method). Scripts and their flags: [Scripts Reference](../reference/scripts-reference.md). Undecided: [Choosing a Training Method](../getting-started/choosing-a-method.md).

## Vision-language support

SFT, DPO, KTO, SMPO, both off-policy distillation scripts and reward modeling (on families with a sequence-classification head) take images; classification, embedding and the GRPO family are text-only. One script serves both modalities — the model class follows the checkpoint and the data path follows the run, so a text-only dataset on a multimodal checkpoint trains through the text pipeline.

Per-method rules: [Modality support](../reference/trainer-architecture.md#modality-support). `text_only_model: true` loads a multimodal checkpoint through its text-only sibling ([field reference](../reference/configuration-reference.md#distributedarguments-model-from-scratch)).

## Parallelism support

EP, TP and ETP are available on every trainer. Context parallelism is declare-to-enable per trainer (`_supports_cp`): it requires each rank to hold a subsequence, which rules out pooling, concatenated forward passes and `logits_to_keep`, so SFT and SMPO are the only two that use it.

Pipeline parallelism is [not yet available in this release](../parallelism/pipeline-parallelism.md); `_supports_pp` records which trainers would take the axis when the engine lands — SFT, SMPO, reward, classification and offline GRPO, plus DPO and KTO with precomputed reference log probs only (DPO further restricted to `sigmoid`/`hinge`/`ipo`, KTO to `apo_zero_unpaired`).

Full trainer × mode matrix: [Trainer Compatibility](../reference/trainer-architecture.md#trainer-compatibility).

## Experiment tracking

Every script reports to the backend named by `report_to`, logging loss, learning rate and gradient norm automatically. The toolkit callbacks add throughput, MoE expert load, eval-time generations and parameter stats ([Training Callbacks](callbacks.md)).

```yaml
report_to: wandb        # wandb | clearml | tensorboard | none
run_name: my-sft-run    # optional; names the run in the tracker
```
