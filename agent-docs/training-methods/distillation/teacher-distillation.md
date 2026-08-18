# Teacher Distillation

Train a student to match a separate, frozen teacher's token-level output distribution. The teacher (usually larger) scores the same fixed dataset completions as the student; the loss combines a distillation term against the teacher logits with an optional CLM term.

| Aspect | Value |
|--------|-------|
| Trainer | `DistributedDistillationTrainer` |
| Script | `scripts/training/distillation/teacher_distill.py` (text or VLM) |
| Loss | `distill_alpha · L_distill + (1 - distill_alpha) · L_clm` (`distill_alpha=1.0` drops CLM) |
| Parallelism | EP, TP, ETP, EP+TP, EP+ETP; no CP, no PP |

The teacher runs per rank under `torch.no_grad()` in `eval()` mode with frozen params. Only the student is parallelized and carries optimizer states, so each GPU holds the student (weights + optimizer states) plus the teacher (weights only) — use PEFT on the student to cut optimizer memory. Every loss compares full-vocabulary distributions, so a student/teacher `vocab_size` mismatch raises at construction.

Teacher loading: `teacher_model_revision` pins the teacher repo (the student's `model_revision` names a commit in a different repo), and reaches the config fetch as well as the weight fetch so a pinned checkpoint cannot pair with hub-main's config. The teacher loads in the run's own dtype — an fp32 run scored against a bf16 teacher would fit rounded targets — and under the run's own `trust_remote_code`, so a remote-code teacher needs that flag set.

Student and teacher make the **same** padded-workload attention request, so the two compared forwards cannot split across kernels. With the default `reset_sinks: true` that request is `sdpa`, since the auto-detected FA4 would run these padded batches through its slow varlen path. With `reset_sinks: false` neither side requests anything and the resolver auto-selects, rejecting sink-dropping backends that would shift every teacher logprob by nats against the student it supervises.

The teacher's request resolves against the **teacher's** config, so the per-family kernel limits that apply are the teacher's (DeepSeek-V4 eager-only, Gemma4 head_dim-512, fp32 vs FlashAttention).

## Loss types

Set `distill_loss` (default `kl_divergence`). Eight types are defined in `src/trainers/distillation/teacher_losses.py`:

<!-- markdownlint-disable MD056 -- pipes inside a code span are literal to Python-Markdown -->

| `distill_loss` | What it computes |
|-----------|------------------|
| `kl_divergence` | `KL(teacher ‖ student)` at `distill_temperature` |
| `mse` | `MSE(teacher_logits, student_logits)` on raw logits |
| `soft_cross_entropy` | `-sum(teacher_probs · log student_probs)` at `distill_temperature` |
| `cosine_similarity` | `1 - cos(teacher_logits, student_logits)` — tolerant of logit-scale differences |
| `jensen_shannon` | `0.5·KL(P‖M) + 0.5·KL(Q‖M)`, `M` the midpoint — symmetric |
| `earth_mover_distance` | per-token 1-Wasserstein `sum_v |CDF_s(v) − CDF_t(v)|` over the vocab axis |
| `alpha_beta_divergence` | generalized alpha-beta divergence (Cichocki et al., 2011) at its fixed `α=1.0`, `β=2.0` — unrelated to the config's `distill_alpha` below, and not settable |
| `slim` | soft cross-entropy kept only at the gold token, scaled there by `1 - exp(-teacher_prob/student_prob)` |

<!-- markdownlint-enable MD056 -->

`distill_temperature` reaches only the four losses that declare it — `kl_divergence`, `soft_cross_entropy`, `jensen_shannon`, `slim`. `mse`, `cosine_similarity`, `earth_mover_distance`, and `alpha_beta_divergence` take no temperature, so setting it there changes nothing.

**All softened divergences are scaled by `distill_temperature²`** (Hinton's convention, shared with the [self-distillation](self-distillation.md) OPD losses). Softening shrinks a divergence and its gradient as `1/T²`, so the factor holds the distillation term's pull on the student — and its weight against the CLM term — fixed as the temperature moves: `distill_alpha` means the same thing at `T = 4` as at `T = 1`.

`slim` is the exception and is **not** rescaled. Its coefficient depends on the student, so it multiplies the soft cross-entropy's teacher-entropy offset, which does not shrink with `T`; rescaling that product grows the gradient ~`T²` instead of holding it fixed. Its gradient still drifts with `T`, so retune the learning rate when raising the temperature on `slim` alone.

`distill_alpha` weights the distillation term against CLM; `use_clm_loss: false` drops CLM entirely (`loss = distill_alpha · L_distill`). `apply_hard_labels: true` multiplies the distillation loss by `(1 - student_prob[label]) · teacher_prob[label]` (ignored for `slim`).

## Dataset format

Standard SFT conversation format under `conversation_field` (default `"messages"`); teacher and student process the same text. See [Dataset Formats](../../data/dataset-formats.md).

```jsonl
{"messages": [{"role": "user", "content": "Explain gravity."}, {"role": "assistant", "content": "Gravity is a fundamental force..."}]}
```

## Quick start

```bash
torchrun --nproc_per_node=8 \
    scripts/training/distillation/teacher_distill.py \
    examples/distillation/qwen3_5/distill-qwen3.5-9b-from-qwen3.6-35b-a3b.yaml
```

The student (Qwen3.5-9B) is dense, so this runs plain FSDP2 data parallel; EP applies to a MoE **student** only (`--expert_parallel_size`) — the teacher is loaded unparallelized.

Core fields of that config (`bf16: true` and `use_liger_kernel: true` are toolkit defaults):

```yaml
model_name_or_path: Qwen/Qwen3.5-9B     # student
teacher_model: Qwen/Qwen3.6-35B-A3B     # teacher (same tokenizer/vocab family)

distill_loss: kl_divergence
distill_temperature: 1.0
distill_alpha: 0.5                       # 0.5 = 50/50 with CLM; 1.0 (default) = distillation only
apply_hard_labels: false
use_clm_loss: true

dataset: allenai/tulu-3-sft-mixture
conversation_field: messages
test_size: 0.02

per_device_train_batch_size: 1
gradient_accumulation_steps: 16
learning_rate: 5.0e-5
max_length: 16384
gradient_checkpointing: true

use_peft: true                           # optional, cuts student optimizer memory; rejected under TP
lora_r: 16
lora_alpha: 16
lora_target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]

output_dir: checkpoints/distill-qwen3.5-9b-from-qwen3.6-35b-a3b
report_to: wandb
```

## Parallelism

EP, TP, and ETP apply to the student, in any supported combination (including EP+TP and the experimental EP+ETP). CP is rejected at construction: it would have to wrap both student and teacher, for little memory benefit since both are already resident. PP is rejected because each microbatch needs a second forward through a distinct frozen network that no single pipeline stage holds. Use gradient checkpointing for long sequences. Full matrix: [Trainer Compatibility](../../reference/trainer-architecture.md#trainer-compatibility).

## Vision-language support

`teacher_distill.py` handles both modalities from one script. The student class follows its checkpoint; the data path follows the run (`is_vlm_run`), so a multimodal student distilled on text-only rows takes the text path. Images ride embedded in messages or in an `images_field` column, as in [VLM SFT](../sft.md#vision-language-models).

The dataset goes through the shared `prepare_vlm_dataset` map + over-length pre-filter, and the script forces `remove_unused_columns=False` so the mapped `history`/`images` columns reach the collator. Teacher and student must share processor geometry for full-vocabulary logit alignment; both load via `AutoModelForImageTextToText` and `pixel_values` threads to both forwards. `train_on_completions_only` is honored on both paths (via `assistant_message_template`): the text path selects the same completion-masking collator as SFT, and both distillation terms mask on the labels.

## Configuration

`DistillScriptArguments` (`src/args/distill_args.py`) requires `teacher_model` and adds `teacher_model_revision` (`None`) and `conversation_field` (`"messages"`) — what the script must load and render. Dataset fields (`dataset`, `dataset_ratio`, `test_size`) come from `CommonScriptArguments`.

`DistillationConfig` (`src/configs/distillation_config.py`) carries the method itself: `distill_loss` (`"kl_divergence"`), `distill_temperature` (`1.0`), `distill_alpha` (`1.0`), `apply_hard_labels` (`False`), `use_clm_loss` (`True`), plus `max_length` (`2048`; over-length conversations are **dropped**, not truncated — `null` resolves to the student's context window) and `dataset_num_proc` (`None`). All are set in the same YAML block as every other training field.

Parallelism flags (`--expert_parallel_size`, `--tensor_parallel_size`, `--expert_tensor_parallel_size`, `--ep_scope`): [ParallelismConfig](../../reference/configuration-reference.md#parallelismconfig).
