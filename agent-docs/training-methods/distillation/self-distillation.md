# Self-Distillation

One model plays both roles: the student sees the prompt, the teacher sees the same prompt plus a hint revealing the gold answer. Training distills the student toward the teacher on the shared response tokens, on top of an SFT objective.

Use it when no stronger teacher exists but the dataset carries gold answers. With a separate teacher use [teacher distillation](teacher-distillation.md); with a verifier and a rollout budget use [online SDPG](online-sdpg.md).

Trainer `DistributedSelfDistillationTrainer` (extends `DistributedSFTTrainer`), script `scripts/training/distillation/self_distill.py` (text or VLM). EP, TP and ETP apply; CP and PP are rejected — the teacher forward uses a second, longer sequence through the whole model ([matrix](../../reference/trainer-architecture.md#trainer-compatibility)).

The loss is `L_sft + beta(k)·L_OPD + alpha·L_ref`. The teacher forward reuses the trainable model in `eval()` under `torch.no_grad()`, so no second model is held.

## Configuration

The config class is TRL's `SFTConfig`, so `max_length` defaults to `1024` — set it explicitly.

```yaml
model_name_or_path: Qwen/Qwen3.5-9B
dataset: open-r1/OpenR1-Math-220k:default   # raw SFT conversations + a gold-answer column
conversation_field: messages
sdpg_answer_field: answer
test_size: 0.02

sdpg_loss: reverse_kl
sdpg_beta_base: 1.0
sdpg_beta_warmup_steps: 50                  # ramp OPD in after SFT settles
sdpg_beta_decay_steps: 100                  # phase OPD out near the end
opd_exclude_eos: true
reference_kl_coef: 0.0

train_on_completions_only: true
assistant_message_template: "<|im_start|>assistant\n"   # required; must byte-match the template
per_device_train_batch_size: 1
gradient_accumulation_steps: 16
learning_rate: 1.0e-05
max_length: 16384
gradient_checkpointing: true
output_dir: checkpoints/self-distill-qwen3.5-9b
```

| Knob | Default | Effect |
|---|---|---|
| `sdpg_hint_template` | `\n[Hint] The correct answer is: {answer}. ...\n` | Appended to the last user turn, teacher forward only. Fills `{answer}` and `{solution}`; any other placeholder is refused at parse time |
| `sdpg_answer_field` / `privileged_solution_field` | `answer` / `solution` | Columns filling `{answer}` and `{solution}`; a slot the template names needs its column while `sdpg_beta_base > 0`, and a row with a blank named slot gets no hint (counted in a prep-time warning) |
| `sdpg_loss` | `reverse_kl` | OPD loss; or `forward_kl`, `unnormalized_kl`, `jensen_shannon` |
| `sdpg_jsd_beta` | `0.5` | β of `jensen_shannon`, in `[0, 1]` ([generalized JSD](teacher-distillation.md#generalized-jsd)); any other value needs `sdpg_loss: jensen_shannon` |
| `sdpg_temperature` | `1.0` | OPD softmax temperature, finite and `> 0`; every OPD loss scales by `T²` |
| `sdpg_beta_base` | `1.0` | Base OPD coefficient, finite and `>= 0`; `0` builds no teacher branch and skips the teacher forward |
| `sdpg_beta_warmup_steps` / `sdpg_beta_decay_steps` | `0` / `0` | `beta(k) = base · min(1, k/T_warm) · min(1, (T−k)/T_decay)`; each `>= 0` |
| `opd_exclude_eos` | `True` | Drops EOS/stop tokens from OPD but not from SFT |
| `reference_kl_coef` | `0.0` | Alpha on a frozen-reference KL anchor, finite and `>= 0`; `0` loads no reference. Under EP, ETP or TP the reference is a whole dense replica on every rank, experts included (a warning says so): its log-probs match the policy's up to kernel numerics, but budget for its memory |
| `reference_kl_loss` | `unnormalized_kl` | The anchor's divergence; or `reverse_kl`, `forward_kl`; evaluated at `sdpg_temperature`. No `jensen_shannon`: the anchor has no β |
| `reference_model_name_or_path` | `None` | The anchor model; defaults to the student's init weights. Loaded like a preference reference, the run's added special tokens included. A separate repo takes its own main, not the policy's `model_revision`, and its tokenizer must map every token to the policy repo's id, refused before its weights load |
| `confidence_field` / `confidence_power` | `None` / `4.0` | Per-sample weight `conf**p` (`p` finite and `> 0`), divided by its mean over the train split, so it weights rows at any batch size. Every confidence must lie in `[0, 1]` |
| `confidence_weight_opd` | `True` | Applies that weight to OPD as well as SFT |

With `reference_kl_coef <= 0`, a non-default `reference_model_name_or_path` or `reference_kl_loss` raises rather than being ignored; so do `confidence_power` and `confidence_weight_opd` without a `confidence_field`, and a `confidence_field` over a presharded dataset (each rank would divide by its own shard's mean).

OPD pairs the k-th supervised token of the student branch with the k-th of the teacher's, so `train_on_completions_only` must stay on while `sdpg_beta_base > 0`: with the prompt supervised, the hint shifts every pair after it onto a different token. `assistant_message_template` has no default and is required whenever `train_on_completions_only` is on — the collator refuses that pair at construction; on the text path it also refuses a marker the chat template never renders, through the same probe `select_data_collator` runs.

Neither branch is ever truncated — the teacher is systematically longer, so right-truncation would cut response tokens the student keeps. On the text path a row over `max_length` raises, naming the branch, as does a row whose two branches do not supervise the same token sequence; a prep-time audit collates every row once, so those raises are world-uniform instead of hanging the peers of one rank. Size `max_length` with headroom for the hint.

The dataset stays raw: the collator tokenizes the student and the hinted teacher branch at collation time. Inherited SFT knobs that cannot reach it are refused, not ignored — `packing`, `padding_free`, `eval_packing`, `completion_only_loss`, `assistant_only_loss`, `dataset_text_field`, `dataset_kwargs`, `train_on_last_assistant_only`, `generate_eval_examples`, `num_eval_examples`, and a `loss_type` other than `nll` / `chunked_nll` (the trainer computes its own cross-entropy).

## Launch

```bash
torchrun --nproc_per_node=8 scripts/training/distillation/self_distill.py \
    examples/distillation/qwen3_5/self-distill-qwen3.5-9b.yaml
```

`halo launch self-distill <config> --nproc 8` builds the same line. That model is dense, so it runs plain FSDP2 data parallel; the MoE recipes (`gptoss`, `gemma4`) pin `expert_parallel_size: 8` themselves.

## Vision-language

The data path follows the run, so a multimodal student trained on text-only rows takes the text path. An image run maps raw conversations into `history`/`images` through the shared `prepare_vlm_dataset`, keeping the privileged columns, and collates with `SelfDistillVLMDataCollator`.

A VLM run drops at prep the rows whose hinted teacher text alone is over `max_length`, and its audit refuses a misaligned row there, world-uniform; image tokens, which neither counts, still raise at collation when they push a branch over. Keep `max_length` headroom for the hint. The hint is text-only and does not change the image grid, so the student's image features are shared with the teacher forward — cached and replayed for `lfm2_vl` to skip a vision-tower re-encode.

## Testing a setup

```bash
torchrun --nproc_per_node=2 scripts/training/distillation/self_distill.py <config> \
    --max_steps=5 --save_strategy=no --report_to=none
```

Covering tests: `pytest tests/cpu/trainers tests/cpu/data/test_self_distill_collator.py -m cpu`, `tests/gpu/trainers/other/test_self_distillation_text.py`, `test_self_distillation_vlm.py` and `tests/gpu/trainers/lora/test_lora_self_distill.py`.

## What to watch

| Signal | Reading |
|---|---|
| `sft_loss` | The hard-label term; the run's backbone |
| `opd_loss` | Student-to-teacher divergence; falls as the hint stops changing the distribution |
| `opd_beta` | The schedule's current coefficient — check it is not pinned at 0 |
| `reference_kl` | Logged only with `reference_kl_coef > 0`; drift from the anchor |

Failure signatures:

- An over-length raise naming the student or teacher branch — raise `max_length`; the hint needs headroom.
- A raise that a row's student and teacher branches supervise different token sequences — the labels reach into the prompt, or the chat template renders an assistant turn differently once the hint is in the user turn.
- Every row masked, loss flat — `assistant_message_template` does not match the rendered prefix.
- A column the template names (`{answer}`, `{solution}`) unset or missing while `sdpg_beta_base > 0` — the run raises rather than distilling toward a teacher told the answer is nothing.
- A collator that builds no `teacher_*` branch while `sdpg_beta_base != 0`, or a trainer built with `reference_kl_coef > 0` and no `reference_model` — both raise at construction rather than dropping the term from the loss.
