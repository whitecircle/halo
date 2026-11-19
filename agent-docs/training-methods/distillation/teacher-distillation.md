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

`distill_alpha` weighs the distillation term and `1 − distill_alpha` the CLM term (`1.0` drops CLM; below it both terms share one chunked pass over the student logits, so backward builds one gradient plane of them, not one per term), and `distill_temperature` reaches only the five losses that declare it. `distill_jsd_beta` sets the β of `jensen_shannon` ([Generalized JSD](#generalized-jsd)) and `distill_topk` scores the loss on the teacher's top-k tokens plus a tail bin ([Top-k](#top-k)); each is refused off its default beside a loss that does not read it. `apply_hard_labels` scales the distillation term per token by `(1 − student_prob[label]) · teacher_prob[label]`, a detached weight that carries no gradient of its own; `slim`, which weights by its own gold-token rule, refuses it. `teacher_model_revision` pins the teacher repo, since the student's `model_revision` names a commit elsewhere. Over-length conversations are **dropped**, not truncated. Types and defaults: [TeacherDistillationConfig](../../reference/configuration-reference.md#teacherdistillationconfig) and [TeacherDistillScriptArguments](../../reference/configuration-reference.md#teacherdistillscriptarguments).

The teacher loads in the run's own dtype (an fp32 run scored against a bf16 teacher fits rounded targets) and under the run's `trust_remote_code`. Student and teacher load on one padded-workload attention request, decided on both checkpoints (on Blackwell FA2 for a softcapped student, and SDPA where either is remote code or only the teacher carries a softcap, which lands the teacher on FA2), so the two compared forwards take the same backend wherever both models accept it. Each model resolves that request against its own config, so the teacher's per-family kernel limits apply to the teacher (DeepSeek-V4 eager-only, Gemma 4 head_dim 512). `reset_sinks` applies to both models, and a GptOss teacher is refused under `reset_sinks: true` before any weight loads: a frozen teacher never trains back from neutralized sinks, and gpt-oss-20b without them predicts ~6 nats/token (KL) from its pretrained self. Set `reset_sinks: false` so both keep their sinks live and frozen ([GPT-OSS → Attention sinks](../../models/gpt-oss.md#attention-sinks)). The trainer refuses a teacher whose load stamped neutralized sinks too, for a teacher built outside the script. A pinned `flash_attention_4` the load keeps for a softcapped student (Gemma 2 27B) is refused before any weight loads at more than one row per batch, training or evaluation: those padded batches reach FA4's varlen kernels, which a softcap stays off ([Flash Attention](../../optimization/flash-attention.md#model-specific-handling)).

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

`distill_temperature` reaches `kl_divergence`, `reverse_kl`, `soft_cross_entropy`, `jensen_shannon` and `slim`; `mse` and `cosine_similarity` take no temperature, so setting it there raises. Every softened divergence is scaled by `distill_temperature²` (Hinton's convention), which holds the distillation term's pull — and its weight against CLM — fixed as the temperature moves.

Both KL directions are per-token divergences at the dataset's prefixes. Sequence-level reverse KL is an expectation over prefixes the student samples itself, the on-policy objective [online SDPG](online-sdpg.md) trains; off-policy `reverse_kl` applies the mode-seeking pull per token only. Likewise, `kl_divergence` equals the sequence-level forward KL only when the dataset completions were sampled from the teacher.

### Generalized JSD

`JSD_β = β·KL(q ‖ M) + (1 − β)·KL(p ‖ M)` with `M = β·q + (1 − β)·p`, `q` the teacher and `p` the student ([GKD](https://arxiv.org/abs/2306.13649), Eq. 1; TRL's `beta` convention). `β = 0` is exactly `kl_divergence` and `β = 1` exactly `reverse_kl`. `distill_jsd_beta` must lie in `[0, 1]`; the SDPG arms take the same loss at `sdpg_jsd_beta`.

The loss shrinks toward the endpoints, to about `β·KL(q ‖ p)` near 0 and `(1 − β)·KL(p ‖ q)` near 1, then jumps to the full KL at exactly 0 and 1. A β close to an endpoint trains with a proportionally weak pull. For forward or reverse KL, set `kl_divergence` / `reverse_kl` (or β exactly `0` / `1`) rather than a β near it.

### Top-k

`distill_topk: k` scores the loss on the teacher's top-k tokens plus one tail bin holding the rest of each distribution's mass. The config refuses it with any loss but `kl_divergence` and `soft_cross_entropy`, the two weighted by the teacher's probabilities alone. `k` must be below the compared vocabulary width: the script refuses one that covers it before any teacher weight loads, and the trainer again at construction, since `null` is the full-vocab loss.

It approximates the full-vocab objective. Merging tokens into one bin never increases a divergence, so the top-k loss is a lower bound on the full-vocab loss. The teacher's top k are picked from its full-vocab logits, so it saves no memory. The tail bin sums the off-support probabilities rather than taking one minus the support mass, which keeps its gradient where a student's support mass rounds to 1 in fp32. The sum builds an extra fp32 copy of each side's log-probs, one token chunk at a time.

### Chunked loss

`use_chunked_distillation_loss: true` (default off) computes every term without either model's `[tokens, vocab]` logits. Both models run to their last hidden state and keep only the supervised positions. Each token chunk of those positions (sized as the divergence's own chunks) is projected through both output embeddings, scored and dropped under a checkpoint, so the backward projects the chunk again. The student's backbone and the chunk loop run inside the student's own forward call, as the chunked GRPO scorer's do, so FSDP2 gathers the output embedding once for the forward and once for the backward, and a rank with no supervised position still runs its backward. Under TP a dense student's output embedding (a tied one included) is sharded on the vocabulary: the projection reads it whole, and its gradient returns to each rank as that rank's slice.

A LoRA adapter on `lm_head` trains through it: each chunk adds `lora_B(lora_A(dropout(x))) · scaling` to the base weight's logits, the dropout drawn over the whole hidden state as PEFT's own forward draws it. The family's head transform applies as on the [chunked GRPO path](../../models/adding-a-model.md#declare-the-head-transform). Refused at construction: a DoRA or other variant adapter on `lm_head`, a family whose head transform does not verify, a composite whose own forward builds its image features into the text model's inputs (Nemotron 3.5 Super VL; every other composite's backbone takes `pixel_values`), and `compute_metrics` or `preprocess_logits_for_metrics`, which read evaluation logits this path never builds. A step raises on an `lm_head` adapter merged into its weight while adapters are disabled, which PEFT's forward would unmerge first.

The default path holds about three bf16 `[tokens, vocab]` planes beyond the models (both models' logits and the student logits' gradient), about 22 GiB at the shipped recipe's 16,384 tokens and 248,320 vocabulary. With the flag off, construction weighs them as `rows × max_length × vocab × 3 planes` in the run's precision (6 B a logit in bf16, 12 B in fp32), at the larger of the two models' vocabularies, against the device memory free once the teacher is placed, and warns past half of it, naming the flag. It only warns, because rows are padded only to the longest row in the batch: `max_length` (the context window when unset) is an upper limit the data may never reach. On one RTX 5070 Ti (Qwen3.5-0.8B from Qwen3.5-2B, LoRA on `lm_head`) the chunked path fit 16,384-token rows where the default path stopped at 4,096. With every position supervised it ran 5–7% fewer tokens/s; with half of each row supervised, as `train_on_completions_only` leaves a prompt, it ran about 35% more, since prompt positions are never projected.

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

## Quantization-aware distillation

`lowp_precision` and the other `lowp_*` knobs apply to the **student** as under [SFT](../sft.md): it loads through the same loader, which fake-quantizes its dense MLP and MoE expert GEMMs over bf16/fp32 masters ([Low-Precision Kernels](../../optimization/low-precision-moe-kernels.md#usage)). The teacher loads through `load_frozen_auxiliary_model`, which takes no parallelism config and never applies them, so it scores in the run's dtype. The trainer refuses a teacher that holds a converted module (`reject_lowp_teacher`).

Pointing `teacher_model` at the student's own checkpoint fits the quantized student to the full-precision model's token distribution rather than to the dataset's labels:

```yaml
model_name_or_path: <moe-checkpoint>
teacher_model: <moe-checkpoint>              # the student's own weights, unquantized
lowp_precision: fp4                          # nvfp4; mxfp4 and fp8 apply too
distill_loss: kl_divergence
distill_alpha: 1.0                           # distillation only; the CE stays logged as sft_loss
```

With both models on one checkpoint and one attention backend, step 0's `distillation_loss` is the divergence the quantization alone causes. Native expert-LoRA adapters on the student compute in bf16, as under SFT. The teacher is not parallelized, so every rank holds a full copy beside its share of the EP student, and `moe_balancing: aux_loss` stays refused (this trainer adds no router aux loss).

## Testing a setup

```bash
torchrun --nproc_per_node=2 scripts/training/distillation/teacher_distill.py <config> \
    --max_steps=5 --save_strategy=no --report_to=none
```

Covering tests: `pytest tests/cpu/trainers -m cpu` (the chunked loss against the default path in `test_teacher_distillation_chunked_loss.py`), `tests/gpu/trainers/other/test_distillation.py`, `test_distillation_oss20b.py`, `test_lowp_teacher_distill.py` (QAD) and `tests/gpu/trainers/lora/test_lora_teacher_distill.py`.

## What to watch

| Signal | Reading |
|---|---|
| `distillation_loss` | The divergence term; should fall as the student matches the teacher |
| `sft_loss` | Hard-label CE, logged at every alpha — a metric only when `distill_alpha: 1.0` |
| `distillation_coef` | The mean gold-token gate; logged with `apply_hard_labels` |

Failure signatures:

- A tokenizer-mismatch raise before the teacher loads (repeated at construction) — the teacher is from another tokenizer family, or `added_special_tokens` grew the student's tokenizer; a raise naming fewer logit rows than the tokenizer — one model's embedding is smaller than the tokenizer.
- OOM on the first step — both models are resident, beside the student's and the teacher's `[tokens, vocab]` logits ([memory](README.md)). Set `use_chunked_distillation_loss: true` ([Chunked loss](#chunked-loss)), use PEFT on the student and gradient checkpointing, or lower `max_length`.
- Most rows dropped at prep — `max_length` drops over-length conversations rather than truncating them.
