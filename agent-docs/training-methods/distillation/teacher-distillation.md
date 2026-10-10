# Teacher Distillation

Fit a student to a separate, frozen teacher's token-level distribution over a fixed dataset. Use it when a stronger model of the same tokenizer family is available; without one, distill the model against its own hinted forward with [self-distillation](self-distillation.md).

Trainer `DistributedDistillationTrainer`, script `scripts/training/distillation/teacher_distill.py` (text or VLM). EP, TP and ETP apply to the **student**; CP and PP are rejected, since every loss reads the teacher's whole `[tokens, vocab]` plane and no stage or sequence shard holds it ([matrix](../../reference/trainer-architecture.md#trainer-compatibility)).

Both models sit on every rank — the student with its optimizer states, the teacher weights-only under `torch.no_grad()` in `eval()` mode. Before any teacher weight loads, the script reads the teacher repo's tokenizer and config (at `teacher_model_revision`) and raises unless the two tokenizers map every token to the same id — tokens added to the student only (`added_special_tokens`) count as a mismatch. The two `vocab_size`s may differ only in embedding padding past the tokenizer; both logit rows are then compared over the tokenizer's ids. The trainer repeats the check at construction.

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
| `distill_temperature` | `1.0` | Softmax temperature; reaches only the five losses that declare it |
| `distill_jsd_beta` | `0.5` | β of `jensen_shannon`, in `[0, 1]`; any other value needs `distill_loss: jensen_shannon` |
| `distill_topk` | `null` | Score the loss on the teacher's top-k tokens plus a tail bin ([Top-k](#top-k)) |
| `distill_alpha` | `1.0` | Weight on the distillation term, `1 − distill_alpha` on CLM; `1.0` drops CLM from the loss |
| `apply_hard_labels` | `False` | Scales the distillation term per token by `(1 − student_prob[label]) · teacher_prob[label]`, a detached weight that carries no gradient of its own; refused at trainer construction under `slim`, which weights by its own gold-token rule |
| `max_length` | `2048` | Over-length conversations are **dropped**, not truncated; `null` → context window |
| `teacher_model_revision` | `None` | Pins the teacher repo; the student's `model_revision` names a commit elsewhere |

The teacher loads in the run's own dtype (an fp32 run scored against a bf16 teacher fits rounded targets) and under the run's `trust_remote_code`. Student and teacher make the same padded-workload attention request, so the two compared forwards cannot split across kernels — but it resolves against the **teacher's** config, so the teacher's per-family kernel limits apply (DeepSeek-V4 eager-only, Gemma 4 head_dim 512).

Each term is one mean over the micro-batch's supervised tokens, so a short row's tokens weigh what a long row's do.

### Loss types

| `distill_loss` | What it computes |
|---|---|
| `kl_divergence` | Forward `KL(teacher ‖ student)` at `distill_temperature`; mode-covering |
| `reverse_kl` | Reverse `KL(student ‖ teacher)` at `distill_temperature`; mode-seeking |
| `mse` | `MSE(teacher_logits, student_logits)` on raw logits |
| `soft_cross_entropy` | `-sum(teacher_probs · log student_probs)` at `distill_temperature` |
| `cosine_similarity` | `1 - cos(teacher_logits, student_logits)`; tolerant of logit-scale differences |
| `jensen_shannon` | Generalized JSD at `distill_jsd_beta` ([below](#generalized-jsd)); symmetric at the default `0.5` |
| `slim` | `KL(teacher ‖ student)` weighted per token by `1 - exp(-teacher_prob[label] / student_prob[label])`, a detached weight |

`slim` takes the weighting [SLIM](https://openreview.net/forum?id=2fc5GOPYip)'s text describes — larger where the teacher is more confident in the gold token than the student — not the paper's loss: its Eq. 4 prints the inverse ratio, `1 − exp(−s/t)`, and adds the weighted soft cross-entropy over a top-5% teacher to a unit-weight CE term, while here the full teacher distribution is used and CLM keeps its `1 − distill_alpha` weight.

`distill_temperature` reaches `kl_divergence`, `reverse_kl`, `soft_cross_entropy`, `jensen_shannon` and `slim`; `mse` and `cosine_similarity` take no temperature, so setting it there changes nothing. Every softened divergence is scaled by `distill_temperature²` (Hinton's convention), which holds the distillation term's pull — and its weight against CLM — fixed as the temperature moves.

Both KL directions are per-token divergences at the dataset's prefixes. Sequence-level reverse KL is an expectation over prefixes the student samples itself, the on-policy objective [online SDPG](online-sdpg.md) trains; off-policy `reverse_kl` applies the mode-seeking pull per token only. Likewise, `kl_divergence` equals the sequence-level forward KL only when the dataset completions were sampled from the teacher.

### Generalized JSD

`JSD_β = β·KL(q ‖ M) + (1 − β)·KL(p ‖ M)` with `M = β·q + (1 − β)·p`, `q` the teacher and `p` the student ([GKD](https://arxiv.org/abs/2306.13649), Eq. 1; TRL's `beta` convention). `β = 0` is exactly `kl_divergence` and `β = 1` exactly `reverse_kl`.

The loss shrinks toward the endpoints, to about `β·KL(q ‖ p)` near 0 and `(1 − β)·KL(p ‖ q)` near 1, then jumps to the full KL at exactly 0 and 1. A β close to an endpoint trains with a proportionally weak pull. For forward or reverse KL, set `kl_divergence` / `reverse_kl` (or β exactly `0` / `1`) rather than a β near it.

### Top-k

`distill_topk: k` scores the loss on the teacher's top-k tokens plus one tail bin holding the rest of each distribution's mass. The config refuses it with any loss but `kl_divergence` and `soft_cross_entropy`, the two weighted by the teacher's probabilities. `k` must be below the vocabulary size; the trainer refuses one that covers it, since `null` is the full-vocab loss.

It approximates the full-vocab objective. Merging tokens into one bin never increases a divergence, so the top-k loss is a lower bound on the full-vocab loss. Both forwards still produce full-vocab logits, so it saves no memory: the tail bin sums the off-support probabilities, which keeps one more fp32 `[tokens, vocab]` copy of the student's log-probs for the backward.

## Launch

```bash
torchrun --nproc_per_node=8 scripts/training/distillation/teacher_distill.py \
    examples/distillation/qwen3_5/distill-qwen3.5-9b-from-qwen3.6-35b-a3b.yaml
```

`halo launch teacher-distill <config> --nproc 8` builds the same line. That student is dense, so it runs plain FSDP2 data parallel; a MoE student adds `--expert_parallel_size=8`. The teacher is never parallelized.

## Vision-language

One script serves both modalities. The student class follows its checkpoint; the data path follows the run, so a multimodal student distilled on text-only rows takes the text path. Images ride embedded in messages or in an `images_field` column, as in [VLM SFT](../sft.md#vision-language-models).

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
- OOM on the first step — both models are resident. Use PEFT on the student, gradient checkpointing, or a smaller `max_length`.
- Most rows dropped at prep — `max_length` drops over-length conversations rather than truncating them.
- `distill_jsd_beta=… only applies to distill_loss: jensen_shannon` or `distill_topk applies only to distill_loss in …` — the knob is set beside a loss that ignores it.
- `distill_topk=… covers the whole vocabulary` at construction — `k` is at least the vocab size; set `distill_topk: null` for the full-vocab loss.
