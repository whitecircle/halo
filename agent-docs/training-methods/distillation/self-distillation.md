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

The hint (`sdpg_hint_template`) is appended to the last user turn for the teacher forward only, its `{answer}` and `{solution}` slots filled from the `sdpg_answer_field` and `privileged_solution_field` columns; a row with a blank named slot gets no hint (counted in a prep-time warning). The OPD coefficient follows `beta(k) = base · min(1, k/T_warm) · min(1, (T−k)/T_decay)`, all three losses scale by `T²` at `sdpg_temperature`, and `sdpg_beta_base: 0` builds no teacher branch and skips the teacher forward. `opd_exclude_eos` keeps EOS/stop tokens out of OPD but not out of SFT.

With `attn_implementation` unset, a softcapped student (Gemma 2) at more than one row per batch, training or evaluation, takes FA2 on Blackwell, since its padded batches reach FA4's varlen kernels, which a softcap stays off; a pinned `flash_attention_4` the load keeps there (Gemma 2 27B; the resolver moves a head_dim-256 pin to FA2) is refused after the load ([Flash Attention](../../optimization/flash-attention.md#model-specific-handling)).

`reference_kl_coef` adds a KL anchor to `reference_model_name_or_path` (the student's init weights when unset), loaded on the student's attention request like a preference reference, the run's added special tokens included. A separate repo takes its own main, not the policy's `model_revision`, and its tokenizer must map every token to the policy repo's id, refused before its weights load. Under EP, ETP or TP the reference is a whole dense replica on every rank, experts included (a warning says so): its log-probs match the policy's up to kernel numerics, but budget for its memory. `confidence_field` weights each row by `conf**confidence_power`, divided by its mean over the train split so it weights rows at any batch size; `confidence_weight_opd` applies it to OPD as well as SFT. Types, defaults and bounds: [SelfDistillScriptArguments](../../reference/configuration-reference.md#selfdistillscriptarguments).

With `reference_kl_coef <= 0`, a non-default `reference_model_name_or_path` or `reference_kl_loss` raises rather than being ignored; so do `confidence_power` and `confidence_weight_opd` without a `confidence_field`, and a `confidence_field` over a presharded dataset (each rank would divide by its own shard's mean).

OPD pairs the k-th supervised token of the student branch with the k-th of the teacher's, so `train_on_completions_only` must stay on while `sdpg_beta_base > 0`: with the prompt supervised, the hint shifts every pair after it onto a different token. `assistant_message_template` has no default and is required whenever `train_on_completions_only` is on — the collator refuses that pair at construction; on the text path it also refuses a marker the chat template never renders, through the same probe `select_data_collator` runs.

Neither branch is ever truncated — the teacher is systematically longer, so right-truncation would cut response tokens the student keeps. On the text path a row over `max_length` raises, naming the branch, as does a row whose two branches do not supervise the same token sequence; both are checked when the row is encoded at prep, so those raises are world-uniform instead of hanging the peers of one rank. Size `max_length` with headroom for the hint.

On the text path each row's student and hinted teacher branches are rendered, tokenized and labelled once at prep (`encode_self_distill_row`, cached under the collator's render knobs and the encoder's source), and collation only pads them; the VLM collator tokenizes at collation time. Inherited SFT knobs that cannot reach it are refused, not ignored — `packing`, `padding_free`, `eval_packing`, `completion_only_loss`, `assistant_only_loss`, `dataset_text_field`, `dataset_kwargs`, `chat_template_path` (applied after the rows are rendered), `train_on_last_assistant_only`, `generate_eval_examples`, `num_eval_examples`, and a `loss_type` other than `nll` / `chunked_nll` (the trainer computes its own cross-entropy).

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
