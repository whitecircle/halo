# Quickstart

Complete [Installation](installation.md) first, then run everything inside the image — `python`/`torchrun`/`accelerate` are on `PATH`, no prefix.

## 1. First SFT run

A ready-to-run config ships at `examples/sft/qwen3/qwen3-4b-ultrachat.yaml`. The fields that carry the run:

```yaml
model_name_or_path: Qwen/Qwen3-4B-Instruct-2507
attn_implementation: flash_attention_2

dataset:
- HuggingFaceH4/ultrachat_200k@train_sft
conversation_field: messages
assistant_message_template: "<|im_start|>assistant\n"   # must match the model's chat template
test_size: 0.01

per_device_train_batch_size: 2
gradient_accumulation_steps: 8
learning_rate: 2.0e-05
num_train_epochs: 1.0
max_length: 4096
packing: true
gradient_checkpointing: true
gradient_checkpointing_kwargs:
  use_reentrant: false
optim: adamw_torch_fused
lr_scheduler_type: cosine
warmup_steps: 32          # an int is exact steps; a float in [0, 1) is a ratio of total steps

output_dir: checkpoints/sft-qwen3-4b-ultrachat
save_strategy: steps
save_steps: 500
report_to: wandb                  # needs a WANDB_API_KEY; --report_to=none to disable
logging_steps: 1
```

```bash
torchrun --nproc_per_node=8 scripts/training/sft.py examples/sft/qwen3/qwen3-4b-ultrachat.yaml
```

`torchrun` is the default launcher — it lands on FSDP2 even with no parallelism flag. Use `python` for a single GPU; `accelerate launch` stays supported for plain data parallelism ([launcher selection](configuration.md#launcher-selection)). BF16, Liger kernels and grouped GEMM (SM90+, MoE experts) are on by default, and the attention implementation is auto-detected per architecture unless the config pins one, as this one does.

`assistant_message_template` must byte-match the model's rendered assistant-turn prefix or the completion mask misfires and loss stays 0. It has no default — set the ChatML form for Qwen, the Llama-3 header for Llama-3. See [SFT](../training-methods/sft.md).

## 2. Other methods

Same YAML shape, different script and a few method fields. `halo launch <method> <config>` runs any of them by name and picks the launcher (`halo launch --list`).

| Method | Script | Guide |
|---|---|---|
| SMPO (reference-free preference) | `scripts/training/preference/smpo.py` | [SMPO](../training-methods/preference/smpo.md) |
| DPO / KTO | `scripts/training/preference/{dpo,kto}.py` | [DPO](../training-methods/preference/dpo.md) · [KTO](../training-methods/preference/kto.md) |
| Offline GRPO (pre-scored) | `scripts/training/offline_grpo.py` | [Offline GRPO](../training-methods/grpo/offline-grpo.md) |
| Online GRPO (RLVR) | `scripts/training/online_grpo/rlvr.py` | [Online GRPO](../training-methods/grpo/online-grpo.md) |
| Environmental GRPO (multi-turn) | `scripts/training/environmental_grpo.py` | [Environmental GRPO](../training-methods/grpo/environmental-grpo.md) |
| Reward modeling / classification | `scripts/training/preference/rewards.py`, `scripts/training/classification.py` | [Reward](../training-methods/preference/reward-modeling.md) · [Classification](../training-methods/classification.md) |
| Distillation | `scripts/training/distillation/{teacher_distill,self_distill}.py` | [Distillation](../training-methods/distillation/README.md) |
| Embedding | `scripts/training/embedding.py` | [Embedding](../training-methods/embedding.md) |

Pick one with [Choosing a Training Method](choosing-a-method.md); per-method fields are in the [Configuration Reference](../reference/configuration-reference.md).

## 3. LoRA and QLoRA

Add to any config:

```yaml
use_peft: true
lora_r: 64
lora_alpha: 64
lora_target_modules: [q_proj, k_proj, v_proj, o_proj]
# load_in_4bit: true   # QLoRA — base weights 4-bit, adapters BF16
```

Use a 5–10× higher learning rate than full fine-tuning (`5e-5` vs `5e-6`).

> [!WARNING]
> **Parallelism limits**
>
> - Attention LoRA works under FSDP/DDP, CP, EP and ETP; it **raises** at construction under TP, EP+TP and PP.
> - On any MoE model, expert names (`gate_up_proj`, `gate_proj`, `up_proj`, `down_proj`, the `gate_proj_gmm` / `up_proj_gmm` grouped spellings, and the `experts` / `mlp.experts` containers) are peeled out of `lora_target_modules` into native grouped adapters — with a warning, since plain `nn.Linear` MLPs sharing those names (dense prefix layers, shared experts) are then adapted by neither half. `use_dora` and `lora_target_parameters` are rejected on that path, and `expert_tp_size > 1` rejects expert LoRA.
> - A `use_peft: true` that would build no adapter raises rather than silently full-finetuning.
> - QLoRA **raises** under EP/TP/PP/grouped-GEMM-MoE loaders — use it with plain DDP/FSDP, or CP on a **dense** model.
> - Full matrix: [PEFT](../optimization/peft.md).

Merge adapters after training:

```bash
python scripts/after_training/merge_peft_adapters.py \
    --adapter_dir checkpoints/my-lora/checkpoint-final \
    --output_dir checkpoints/my-merged-model
```

## 4. Adding parallelism

`torchrun` is **required** for any `--*_parallel_size` flag — it applies FSDP2 (`fully_shard`) and the parallelism-specific gradient hooks; `accelerate launch` raises on those flags. Both launchers do plain data parallelism.

| Mode | What it splits | Best for |
|---|---|---|
| [DDP/FSDP](../parallelism/data-parallelism.md) | Data batches | Any model |
| [EP](../parallelism/expert-parallelism.md) | MoE experts (DeepEP) | Large MoE |
| [TP](../parallelism/tensor-parallelism.md) | Weight matrices | Large dense, or any model with `tp_plan` |
| [CP](../parallelism/context-parallelism.md) | Input sequences | 32K+ tokens |
| [ETP](../parallelism/expert-tensor-parallelism.md) | Expert FFN weights | MoE expert memory (experimental) |
| [PP](../parallelism/pipeline-parallelism.md) | The layer stack, into stages | Not yet available in this release |

```bash
# EP (MoE only); every other axis is the same line with its own flag
torchrun --nproc_per_node=8 scripts/training/sft.py cfg.yaml --expert_parallel_size=8
```

One flag per axis, combinable where the allowlist allows it — the per-mode command table is on
[Configuration → launcher selection](configuration.md#launcher-selection).

Before you burn a run:

- **Single-node EP** must form one dispatch group per NVLink domain: `ep_size × expert_tp_size` = GPUs in the domain, or `ep_size` = 2. Anything narrower with `ep_size > 2` (ep4 on 8) is rejected at config time — its DeepEP combine barriers race FSDP2's collectives and hang. For 4-way expert sharding on 8 GPUs use `ep4 + expert_tp2`; attention TP does not widen the dispatch group.
- **CP** needs a real Flash Attention impl, `seq_len` and both head counts divisible by `cp_size`, and stays node-local.
- **Combinations are an allowlist.** Each axis alone, plus EP+TP, EP+CP, and EP+ETP. Everything else — TP+CP, ETP+CP, TP+ETP, EP+TP+ETP — is rejected by `ParallelismConfig` before any model loads, with the mechanism; `pipeline_parallel_size > 1` ([not yet available](../parallelism/pipeline-parallelism.md)) is rejected one step earlier, where the CLI arguments are turned into that config. See [Parallelism](../parallelism/README.md).

### Multi-node

```bash
torchrun --nnodes=2 --node_rank=$NODE_RANK --nproc_per_node=8 \
    --master_addr=$MASTER_ADDR --master_port=29500 \
    scripts/training/sft.py cfg.yaml --expert_parallel_size=8
```

One `torchrun` per node with its own `--node_rank` (0 and 1 here); `halo launch` is single-node only. Set `DIST_SHARED_FILESYSTEM=0` when nodes do **not** share a filesystem (default `1`). Interconnect, timeout, and pre-sharding rules: [Launch Recipes](../parallelism/launch-recipes.md).

## 5. Production launch

Run detached and point caches at a **verified** large volume (`df -h` / `findmnt` first — a `/mnt` path is not always a big array):

```bash
D=/mnt   # ← your verified large volume
HF_HOME=$D/hf HF_DATASETS_CACHE=$D/hf/datasets TMPDIR=$D/tmp HALO_DATA_ROOT=$D \
  nohup torchrun --nproc_per_node=8 scripts/training/sft.py \
    examples/sft/qwen3_5/qwen3.5-35b-a3b-ultrachat-ep.yaml &
```

That config declares `expert_parallel_size: 8`, so no CLI flag is needed. `HALO_DATA_ROOT` is the toolkit's own scratch (S3 dataset cache, profiler artifacts). Each var falls back to a default cache or temp location when unset. Secrets (`WANDB_API_KEY` / `HF_TOKEN` / `AWS_*`) come from `.env` and are never auto-loaded — pass `--env-file` under Docker.

The log-writing rank tees stdout and stderr to `<output_dir>/log/run.log` with no shell redirection —
global rank 0 on a shared filesystem, each node's local rank 0 otherwise. `tail -f` it to monitor;
append `> $D/job.log 2>&1` to also capture the other ranks (e.g. a crash off rank 0).

## 6. CLI overrides

Any YAML field can be overridden as `--key=value`; CLI wins, and an unknown or repeated flag raises
before training starts ([rules](configuration.md#cli-overrides)).

```bash
torchrun --nproc_per_node=8 scripts/training/sft.py examples/sft/qwen3/qwen3-4b-ultrachat.yaml \
    --learning_rate=1e-5 --max_length=32000 --output_dir=checkpoints/experiment-v2
```

## 7. Datasets

SFT reads the column named by `conversation_field` (default `prompt`): a `list[dict]` of `role`/`content` turns. Other methods take other columns — full matrix in [Dataset Formats](../data/dataset-formats.md).

Sources: HF Hub (`org/name[:config][@split]`), S3 (`s3://bucket/key` — the **full** URI; a bare name resolves as a local path or Hub repo), a local path, or a list with `dataset_ratio` for mixing. See [S3 Utilities](../data/s3-utilities.md).

## Verifying the setup

```bash
python tests/cpu/environments/test_execution.py                    # CPU, no GPU
torchrun --nproc_per_node=2 \                                      # 2 GPUs
    tests/gpu/trainers/preference/test_smpo_fsdp.py
```
