# Teacher Distillation

Fit a student to a separate, frozen teacher's token-level distribution over a fixed dataset. Use it when a stronger model of the same tokenizer family is available; without one, distill the model against its own hinted forward with [self-distillation](self-distillation.md).

Trainer `DistributedTeacherDistillationTrainer`, script `scripts/training/distillation/teacher_distill.py` (text or VLM). EP, TP and ETP apply to the **student**; CP and PP are rejected, since every loss reads the teacher's whole `[tokens, vocab]` plane and no stage or sequence shard holds it ([matrix](../../reference/trainer-architecture.md#trainer-compatibility)).

Both models sit on every rank — the student with its optimizer states, the teacher weights-only under `torch.no_grad()` in `eval()` mode. Before any teacher weight loads, the script reads the teacher repo's tokenizer and config (at `teacher_model_revision`) and raises unless the two tokenizers map every token to the same id — tokens added to the student only (`added_special_tokens`) count as a mismatch. The two `vocab_size`s may differ only in embedding padding past the tokenizer; both logit rows are then compared over the tokenizer's ids. The trainer repeats the check at construction.

## Configuration

```yaml
model_name_or_path: Qwen/Qwen3.5-9B          # student
teacher_model: Qwen/Qwen3.6-35B-A3B          # teacher, same tokenizer/vocab
attn_implementation: sdpa                    # the shipped recipe's pin; unset takes FA4 on Blackwell

distill_loss: kl_divergence
distill_alpha: 0.5                           # 0.5 = even split with CLM
dataset: allenai/tulu-3-sft-mixture
conversation_field: messages
test_size: 0.02

per_device_train_batch_size: 1
gradient_accumulation_steps: 16
learning_rate: 5.0e-05
max_length: 16384
gradient_checkpointing: true
assistant_message_template: "<|im_start|>assistant\n"   # the model's rendered assistant prefix

use_peft: true                               # cuts student optimizer memory; rejected under TP
lora_r: 16
lora_alpha: 16
output_dir: checkpoints/distill-qwen3.5-9b-from-qwen3.6-35b-a3b
```

`distill_alpha` weighs the distillation term and `1 − distill_alpha` the CLM term (`1.0` drops CLM; below it both terms share one chunked pass over the student logits, so backward builds one gradient plane of them, not one per term), and `distill_temperature` reaches only the four losses that declare it. `apply_hard_labels` scales the distillation term per token by `(1 − student_prob[label]) · teacher_prob[label]`, a detached weight that carries no gradient of its own; `slim`, which weights by its own gold-token rule, refuses it. `teacher_model_revision` pins the teacher repo, since the student's `model_revision` names a commit elsewhere. Over-length conversations are **dropped**, not truncated. Types and defaults: [TeacherDistillationConfig](../../reference/configuration-reference.md#teacherdistillationconfig) and [TeacherDistillScriptArguments](../../reference/configuration-reference.md#teacherdistillscriptarguments).

The teacher loads in the run's own dtype (an fp32 run scored against a bf16 teacher fits rounded targets) and under the run's `trust_remote_code`. Student and teacher load on one padded-workload attention request, decided on both checkpoints (on Blackwell FA2 for a softcapped student, and SDPA where either is remote code or only the teacher carries a softcap, which lands the teacher on FA2), so the two compared forwards take the same backend wherever both models accept it. Each model resolves that request against its own config, so the teacher's per-family kernel limits apply to the teacher (DeepSeek-V4 eager-only, Gemma 4 head_dim 512). `reset_sinks` applies to both models, and a GptOss teacher is refused under `reset_sinks: true` before any weight loads: a frozen teacher never trains back from neutralized sinks, and gpt-oss-20b without them predicts ~6 nats/token (KL) from its pretrained self. Set `reset_sinks: false` so both keep their sinks live and frozen ([GPT-OSS → Attention sinks](../../models/gpt-oss.md#attention-sinks)). The trainer refuses a teacher whose load stamped neutralized sinks too, for a teacher built outside the script. A pinned `flash_attention_4` the load keeps for a softcapped student (Gemma 2 27B) is refused before any weight loads at more than one row per batch, training or evaluation: those padded batches reach FA4's varlen kernels, which a softcap stays off ([Flash Attention](../../optimization/flash-attention.md#model-specific-handling)).

Each term is one mean over the micro-batch's supervised tokens, so a short row's tokens weigh what a long row's do.

### Loss types

| `distill_loss` | What it computes |
|---|---|
| `kl_divergence` | `KL(teacher ‖ student)` at `distill_temperature` |
| `mse` | `MSE(teacher_logits, student_logits)` on raw logits |
| `soft_cross_entropy` | `-sum(teacher_probs · log student_probs)` at `distill_temperature` |
| `cosine_similarity` | `1 - cos(teacher_logits, student_logits)`; tolerant of logit-scale differences |
| `jensen_shannon` | `0.5·KL(P‖M) + 0.5·KL(Q‖M)` with `M` the midpoint — symmetric |
| `slim` | `KL(teacher ‖ student)` weighted per token by `1 - exp(-teacher_prob[label] / student_prob[label])`, a detached weight |

`slim` takes the weighting [SLIM](https://openreview.net/forum?id=2fc5GOPYip)'s text describes — larger where the teacher is more confident in the gold token than the student — not the paper's loss: its Eq. 4 prints the inverse ratio, `1 − exp(−s/t)`, and adds the weighted soft cross-entropy over a top-5% teacher to a unit-weight CE term, while here the full teacher distribution is used and CLM keeps its `1 − distill_alpha` weight.

`distill_temperature` reaches `kl_divergence`, `soft_cross_entropy`, `jensen_shannon` and `slim`; `mse` and `cosine_similarity` take no temperature, so setting it there raises. Every softened divergence is scaled by `distill_temperature²` (Hinton's convention), which holds the distillation term's pull — and its weight against CLM — fixed as the temperature moves.

## Launch

```bash
torchrun --nproc_per_node=8 scripts/training/distillation/teacher_distill.py \
    examples/distillation/qwen3_5/distill-qwen3.5-9b-from-qwen3.6-35b-a3b.yaml
```

`halo launch teacher-distill <config> --nproc 8` builds the same line. That student is dense, so it runs plain FSDP2 data parallel; a MoE student adds `--expert_parallel_size=8`. The teacher is never parallelized.

## Vision-language

One script serves both modalities. The student class follows its checkpoint; the data path follows the run, so a multimodal student distilled on text-only rows takes the text path, a text teacher included. An image run refuses a text teacher, which cannot score its rows. Images ride embedded in messages or in an `images_field` column, as in [VLM SFT](../sft.md#vision-language-models).

An image run maps through the shared `prepare_vlm_dataset` and forces `remove_unused_columns=False` so the `history`/`images` columns reach `VLMDataCollator`. `pixel_values` thread to both forwards, and the two models must share processor geometry as well as vocabulary.

`train_on_completions_only` is honored on both paths via `assistant_message_template`: both loss terms mask on `labels`. Without it, the text path's padded collator keeps the turn-ending EOS label where pad and EOS share an id.

## Testing a setup

```bash
torchrun --nproc_per_node=2 scripts/training/distillation/teacher_distill.py <config> \
    --max_steps=5 --save_strategy=no --report_to=none
```

Covering tests: `pytest tests/cpu/trainers -m cpu`, `tests/gpu/trainers/other/test_distillation.py`, `test_distillation_oss20b.py` and `tests/gpu/trainers/lora/test_lora_teacher_distill.py`.

## What to watch

| Signal | Reading |
|---|---|
| `distillation_loss` | The divergence term; should fall as the student matches the teacher |
| `sft_loss` | Hard-label CE, logged at every alpha — a metric only when `distill_alpha: 1.0` |
| `distillation_coef` | The mean gold-token gate; logged with `apply_hard_labels` |

Failure signatures:

- A tokenizer-mismatch raise before the teacher loads (repeated at construction) — the teacher is from another tokenizer family, or `added_special_tokens` grew the student's tokenizer; a raise naming fewer logit rows than the tokenizer — one model's embedding is smaller than the tokenizer.
- OOM on the first step — both models are resident, beside the student's and the teacher's `[tokens, vocab]` logits ([memory](README.md)). Use PEFT on the student, gradient checkpointing, or a smaller `max_length`.
- Most rows dropped at prep — `max_length` drops over-length conversations rather than truncating them.
