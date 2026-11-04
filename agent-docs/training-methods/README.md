# Training Methods

Every method is YAML-driven, runs through the same distributed parallelism system, and ships example configs under `examples/`.

## Methods at a glance

| Method | Trainer | Dataset columns | When to use |
|---|---|---|---|
| [SFT](sft.md) | `DistributedSFTTrainer` | `prompt` (message list) | Instruction tuning, text or vision-language; also [pre-training](pretraining.md), same trainer and script, and a release's [MTP head](../optimization/mtp-heads.md) (`mtp_loss_weight`) |
| [DPO](preference/dpo.md) | `DistributedDPOTrainer` | `prompt`, `chosen`, `rejected` | Paired preference data, against a reference model; 15 TRL `loss_type` values, combinable |
| [KTO](preference/kto.md) | `DistributedKTOTrainer` | `prompt`, `completion`, `label` | Unpaired binary feedback — one completion, thumbs up or down |
| [SMPO](preference/smpo.md) | `SMPOTrainer` | `prompt`, `chosen`, `rejected` | Preference tuning with no reference model to hold in memory |
| [Offline GRPO](grpo/offline.md) | `OfflineGRPOTrainer` | `prompt`, `completions`, `rewards` | Completions already generated and scored |
| [Online GRPO (RLVR)](grpo/online.md) | `DistributedOnlineGRPOTrainer` | `prompt`, `answer` | Single-turn verifiable rewards (math, format) against a live vLLM server |
| [Async GRPO with Environments](async-grpo/README.md) | `DistributedAsyncEnvironmentalGRPOTrainer` | `prompt` (+ `answer` where the environment grades against one) | Multi-turn RL with tools, sandboxes or a judge, on vLLM or SGLang |
| [Reward modeling](reward-modeling.md) | `DistributedRewardTrainer` | `chosen`, `rejected` | A Bradley-Terry scorer to rank completions |
| [Classification](classification.md) | `ClassificationTrainer` | `prompt` or `text_field`, `label` | Single- and multi-label sequence classification |
| [Distillation](distillation/README.md) | `DistributedTeacherDistillationTrainer`, `DistributedSelfDistillationTrainer`, `DistributedSDPGTrainer` | SFT conversations (`conversation_field`), or `prompt`, `answer` (online SDPG) | Compress a teacher, or self-distill from a privileged hint (offline or online SDPG) |
| [Embedding](embedding.md) | `EmbeddingTrainer` | text pairs or triplets, optional `label` / `score` | Retrieval and similarity models, 10 SBERT losses and Matryoshka |

Family overviews: [Preference](preference/README.md) · [GRPO](grpo/README.md).

Column types and per-script defaults: [Dataset Formats](../data/dataset-formats.md#required-columns-by-method). Scripts and their flags: [Scripts Reference](../reference/scripts-reference.md). Undecided: [Choosing a method](../../human-docs/choosing-a-method.md).

## Vision-language support

SFT, DPO, KTO, SMPO, both off-policy distillation scripts and reward modeling (on families with a sequence-classification head) take images, PaliGemma only through SFT and distillation; classification, embedding and the GRPO family are text-only. One script serves both modalities — the model class follows the checkpoint and the data path follows the run, so a text-only dataset on a multimodal checkpoint trains through the text pipeline.

Per-method rules: [Modality support](../reference/trainer-architecture.md#modality-support). `text_only_model: true` loads a multimodal checkpoint through its text-only sibling ([field reference](../reference/configuration-reference.md#model-checkpoint-and-from-scratch-flags)).

## Parallelism support

No trainer refuses EP, TP or ETP ([Trainer Compatibility](../reference/trainer-architecture.md#trainer-compatibility)). Context parallelism is declare-to-enable per trainer (`SUPPORTS_CP`): SFT, SMPO and offline GRPO full fine-tuning support it; online and environment GRPO do not.

Pipeline parallelism is declare-to-enable the same way (`PP_REFUSAL = None`): SFT, SMPO, DPO and KTO (precompute-only), offline GRPO, reward and classification support it ([Pipeline Parallelism](../parallelism/pipeline-parallelism.md)).

Full trainer × mode matrix: [Trainer Compatibility](../reference/trainer-architecture.md#trainer-compatibility).

## Experiment tracking

Every script reports to the backend named by `report_to`, logging loss, learning rate and gradient norm automatically. The toolkit callbacks add throughput, MoE expert load, eval-time generations and parameter stats ([Training Callbacks](callbacks.md)).

```yaml
report_to: wandb        # wandb | clearml | tensorboard | none
run_name: my-sft-run    # optional; names the run in the tracker
```
