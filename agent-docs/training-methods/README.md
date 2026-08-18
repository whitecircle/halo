# Training Methods

Every method is YAML-driven and integrates with the distributed parallelism system.

## Methods at a glance

| Method | Description | Script | Config | EP | CP | TP | ETP |
|--------|-------------|--------|--------|:--:|:--:|:--:|:---------:|
| [SFT](sft.md) | Supervised fine-tuning ([text or VLM](sft.md#vision-language-models)) | `scripts/training/sft.py` | [YAML](../reference/configuration-reference.md#yaml-config-examples) | Yes | Yes | Yes | Yes |
| [DPO](preference/dpo.md) | Direct Preference Optimization (+ IPO, SLiC-HF, RPO) | `scripts/training/preference/dpo.py` | [YAML](../reference/configuration-reference.md#yaml-config-examples) | Yes | — | Yes | Yes |
| [KTO](preference/kto.md) | Kahneman-Tversky Optimization (unpaired preference) | `scripts/training/preference/kto.py` | [YAML](../reference/configuration-reference.md#yaml-config-examples) | Yes | — | Yes | Yes |
| [SMPO](preference/smpo.md) | Smooth Margin Preference Optimization (reference-free) | `scripts/training/preference/smpo.py` | [SmoothMarginPOConfig](../reference/configuration-reference.md#smoothmarginpoconfig) | Yes | Yes | Yes | Yes |
| [Offline GRPO](grpo/offline-grpo.md) | Group-relative PO with pre-computed rewards | `scripts/training/offline_grpo.py` | [OfflineGRPOConfig](../reference/configuration-reference.md#offlinegrpoconfig) | Yes | — | Yes | Yes |
| [Online GRPO (RLVR)](grpo/online-grpo.md) | Online GRPO with verifiable rewards (math, format) | `scripts/training/online_grpo/rlvr.py` | [GRPOConfig](../reference/configuration-reference.md#grpoconfig-distributedgrpotrainer-distributedasyncenvironmentalgrpotrainer-trl-docs) | Yes | — | Yes | Yes |
| [Environmental GRPO](grpo/environmental-grpo.md) | Multi-turn RL with async Ray actors and vLLM or SGLang | `scripts/training/environmental_grpo.py` | [AsyncTrainingConfig](../reference/configuration-reference.md#asynctrainingconfig) | Yes | — | Yes | Yes |
| [Reward Modeling](preference/reward-modeling.md) | Bradley-Terry reward model training | `scripts/training/preference/rewards.py` | [YAML](../reference/configuration-reference.md#yaml-config-examples) | Yes | — | Yes | Yes |
| [Classification](classification.md) | Sequence classification (single- and multi-label) | `scripts/training/classification.py` | [ClassificationConfig](../reference/configuration-reference.md#classificationconfig) | Yes | — | Yes | Yes |
| [Distillation](distillation/README.md) | Knowledge distillation: teacher, self, online SDPG | `scripts/training/distillation/teacher_distill.py`, `scripts/training/distillation/self_distill.py`; SDPG runs `scripts/training/online_grpo/rlvr.py` with `--use_sdpg` | [DistillationConfig](../reference/configuration-reference.md#distillationconfig) | Yes | — | Yes | Yes |
| [Embedding](embedding.md) | Embedding fine-tuning (10 SBERT losses, Matryoshka) | `scripts/training/embedding.py` | [EmbeddingConfig](embedding.md#configuration) | Yes | — | Yes | Yes |

See the [Scripts Reference](../reference/scripts-reference.md) for usage, and [Choosing a Training Method](../getting-started/choosing-a-method.md) for a decision flowchart.

## Vision-language support

Where a method supports images at all (roster below), one script handles both modalities. Point it at a VLM checkpoint; images ride embedded in message content or in a separate column. SFT and the distillation scripts inject the `images_field` column into the prompt conversation, DPO / KTO / reward modeling rename it to the `images` column their vision route probes for, and SMPO reads an `images`/`image` column only.

Two verdicts decide a run, plus one override. The **model class** follows the checkpoint: `load_model_for_training` (`src/distributed/loading/vlm_setup.py`) reads a `model_type` registered under `AutoModelForImageTextToText`, or a config carrying `vision_config`, and `is_vlm_model` falls back to a model-id substring only when the config cannot load or its `model_type` is unregistered (remote code) — a registered config decides both ways ([detection rules](../data/dataset-formats.md#sft-vlm)).

The **data path** follows the run: `is_vlm_run` takes the VLM path only when a multimodal checkpoint *also* has image data — an `images_field` the config names, or an image column on the rows. Every VLM-capable script routes through it, so a text-only dataset on a natively-multimodal checkpoint trains through the text pipeline; the start log names the verdict (`modality: vlm|text`).

`text_only_model: true` overrides the model-class verdict on the six scripts that build a causal LM — SFT, DPO, KTO, SMPO and both distillation scripts: the multimodal checkpoint loads through its text-only CausalLM sibling with no vision tower, and image data is then refused rather than pruned unseen. Reward modeling refuses it outright: its loader pins `AutoModelForSequenceClassification`, so the flag could only warn.

Available for **SFT**, **DPO**, **SMPO**, **KTO** (unpaired-only), **off-policy distillation** (`teacher_distill.py`), **offline SDPG self-distillation** (`self_distill.py`), and **reward modeling** — the last only on families that also carry a sequence-classification head ([roster](preference/reward-modeling.md#vision-language)).

Not available for **classification** (the script loads through the text-only path and builds no image pipeline, and [most families ship no VLM score head](classification.md#vision-language)), **multimodal embedding** (only standard sentence-transformers mode is feasible; EP/TP would need a multimodal preloaded module), and **online SDPG** (its privileged hint is tokenized — text-only).

## Parallelism support

Each trainer declares its own CP and PP support (`_supports_cp` / `_supports_pp`); EP and TP are on by default, and ETP rides the EP declaration. CP requires each rank to process a subsequence, so methods that need complete sequences for pooling, concatenated forward passes, or `logits_to_keep` cannot use it — SFT and SMPO are the only two that do. Pipeline parallelism is [not yet available in this release](../parallelism/pipeline-parallelism.md); `_supports_pp` records which trainers take the axis when the engine lands.

Full trainer × mode matrix: [Trainer Compatibility](../reference/trainer-architecture.md#trainer-compatibility).

## Experiment tracking

Every script reports to the backend set by `report_to` — `wandb`, `clearml`, `tensorboard`, or `none`. Loss, learning rate, and gradient norm are logged automatically; the toolkit callbacks add throughput, MoE expert-load distribution, eval-time generations, and parameter stats ([Training Callbacks](callbacks.md)).

```yaml
report_to: wandb        # wandb | clearml | tensorboard | none
run_name: my-sft-run    # optional; names the run in the tracker
```
