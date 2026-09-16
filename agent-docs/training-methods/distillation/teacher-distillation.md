# Teacher Distillation

Fit a student to a separate, frozen teacher's token-level distribution over a fixed dataset. Use it when a stronger model of the same tokenizer family is available; without one, distill the model against its own hinted forward with [self-distillation](self-distillation.md).

Trainer `DistributedDistillationTrainer`, script `scripts/training/distillation/teacher_distill.py` (text or VLM). EP, TP and ETP apply to the **student**; CP and PP are rejected, since every loss reads the teacher's whole `[tokens, vocab]` plane and no stage or sequence shard holds it ([matrix](../../reference/trainer-architecture.md#trainer-compatibility)).

Both models sit on every rank — the student with its optimizer states, the teacher weights-only under `torch.no_grad()` in `eval()` mode. Their vocabularies must match, or construction raises.

## Configuration

```yaml
model_name_or_path: Qwen/Qwen3.5-9B          # student
teacher_model: Qwen/Qwen3.6-35B-A3B          # teacher, same tokenizer/vocab
attn_implementation: sdpa                    # the script's default under reset_sinks: true

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

| Knob | Default | Effect |
|---|---|---|
| `distill_loss` | `kl_divergence` | The divergence against the teacher; see below |
| `distill_temperature` | `1.0` | Softmax temperature; reaches only the four losses that declare it |
| `distill_alpha` | `1.0` | Weight on the distillation term; `1.0` drops CLM from the loss |
| `use_clm_loss` | `True` | `False` needs `distill_alpha: 1.0`, or alpha would rescale the sole term |
| `apply_hard_labels` | `False` | Scales the loss by `(1 − student_prob[label]) · teacher_prob[label]` |
| `max_length` | `2048` | Over-length conversations are **dropped**, not truncated; `null` → context window |
| `teacher_model_revision` | `None` | Pins the teacher repo; the student's `model_revision` names a commit elsewhere |

The teacher loads in the run's own dtype (an fp32 run scored against a bf16 teacher fits rounded targets) and under the run's `trust_remote_code`. Student and teacher make the same padded-workload attention request, so the two compared forwards cannot split across kernels — but it resolves against the **teacher's** config, so the teacher's per-family kernel limits apply (DeepSeek-V4 eager-only, Gemma 4 head_dim 512).

### Loss types

| `distill_loss` | What it computes |
|---|---|
| `kl_divergence` | `KL(teacher ‖ student)` at `distill_temperature` |
| `mse` | `MSE(teacher_logits, student_logits)` on raw logits |
| `soft_cross_entropy` | `-sum(teacher_probs · log student_probs)` at `distill_temperature` |
| `cosine_similarity` | `1 - cos(teacher_logits, student_logits)`; tolerant of logit-scale differences |
| `jensen_shannon` | `0.5·KL(P‖M) + 0.5·KL(Q‖M)` with `M` the midpoint — symmetric |
| `earth_mover_distance` | Per-token 1-Wasserstein `sum_v \|CDF_s(v) − CDF_t(v)\|` over the vocab axis |
| `alpha_beta_divergence` | Alpha-beta divergence at its fixed `α=1.0`, `β=2.0`; unrelated to `distill_alpha`, and not settable |
| `slim` | Soft cross-entropy kept at the gold token, scaled by `1 - exp(-teacher_prob/student_prob)` |

`distill_temperature` reaches `kl_divergence`, `soft_cross_entropy`, `jensen_shannon` and `slim`; the other four take no temperature, so setting it there changes nothing. Every softened divergence is scaled by `distill_temperature²` (Hinton's convention), which holds the distillation term's pull — and its weight against CLM — fixed as the temperature moves.

`slim` is the exception: its student-dependent coefficient multiplies an offset that does not shrink with `T`, so it is left unscaled and its gradient drifts with temperature. Retune the learning rate when raising `distill_temperature` there.

## Launch

```bash
torchrun --nproc_per_node=8 scripts/training/distillation/teacher_distill.py \
    examples/distillation/qwen3_5/distill-qwen3.5-9b-from-qwen3.6-35b-a3b.yaml
```

`halo launch teacher-distill <config> --nproc 8` builds the same line. That student is dense, so it runs plain FSDP2 data parallel; a MoE student adds `--expert_parallel_size=8`. The teacher is never parallelized.

## Vision-language

One script serves both modalities. The student class follows its checkpoint; the data path follows the run, so a multimodal student distilled on text-only rows takes the text path. Images ride embedded in messages or in an `images_field` column, as in [VLM SFT](../sft.md#vision-language-models).

An image run maps through the shared `prepare_vlm_dataset` and forces `remove_unused_columns=False` so the `history`/`images` columns reach `VLMDataCollator`. `pixel_values` thread to both forwards, and the two models must share processor geometry as well as vocabulary.

`train_on_completions_only` is honored on both paths via `assistant_message_template`: both loss terms mask on `labels`.

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
| `distillation_coef` | The mean gold-token gate; logged with `apply_hard_labels` except under `slim` |

Failure signatures:

- Vocabulary-mismatch raise at construction — the teacher is from another tokenizer family.
- OOM on the first step — both models are resident. Use PEFT on the student, gradient checkpointing, or a smaller `max_length`.
- Most rows dropped at prep — `max_length` drops over-length conversations rather than truncating them.
- `use_clm_loss=False needs distill_alpha=1.0` — set alpha to 1.0, or keep CLM on to weight the two terms.
