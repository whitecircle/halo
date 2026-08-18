# Self-Distillation

One model is both student and a privileged teacher. The student sees the prompt only; the teacher additionally sees a hint that reveals the gold answer. Training distills the student toward the teacher's distribution on the shared response tokens, on top of an SFT objective. This is an offline approximation of [SDPG](online-sdpg.md): no generation, no verifier — it scores the fixed dataset response tokens with static confidence weights standing in for SDPG's verifier-gated advantages.

| Aspect | Value |
|--------|-------|
| Trainer | `DistributedSelfDistillationTrainer` (extends `DistributedSFTTrainer`) |
| Script | `scripts/training/distillation/self_distill.py` (text or VLM) |
| Loss | `L_sft + beta(k) · L_OPD + alpha · L_ref` |
| Parallelism | EP, TP, ETP, EP+TP, EP+ETP; no CP, no PP |

## Privileged context

The privileged hint is appended to the last user turn for the teacher forward only. Default template:

```text
\n[Hint] The correct answer is: {answer}. Do NOT state that you were given the answer.\n
```

`{answer}` and `{solution}` fill from `sdpg_answer_field` (default `answer`) and `privileged_solution_field` (default `solution`). `SelfDistillTextCollator` (`src/data/collators/self_distill.py`) tokenizes the student conversation and the hinted teacher conversation at collation time, so the dataset is the raw SFT conversation plus the privileged field — no offline preprocessing. The assistant response tokens are byte-identical across the two branches, the invariant OPD row alignment relies on.

Neither branch is ever truncated — the teacher is systematically longer, so right-truncation would cut trailing response tokens the student keeps. A sequence over `max_length` raises, naming the branch; size `max_length` with headroom for the hint, or drop over-long rows before training. The teacher forward reuses the same trainable model under `torch.no_grad()`, so no second model is held in memory.

## Loss

- `L_sft` — token-mean SFT cross-entropy on the response tokens. With `confidence_field` set it is per-sample weighted by `confidence**confidence_power`, mean-normalized across the batch to preserve the effective learning rate.
- `L_OPD` — the on-policy-distillation term: the full-vocabulary student→teacher KL on the shared response tokens, with the teacher detached. The default `sdpg_loss: reverse_kl` is the exact `D_KL(p ‖ SG[q])`; `forward_kl` and `unnormalized_kl` are also available. All three scale by `sdpg_temperature²`, so the OPD gradient magnitude stays comparable across temperatures. With `opd_exclude_eos: true` (default) the EOS/stop tokens are dropped from OPD but not from SFT, so SFT's hard `P(EOS)→1` is not diluted by the softer teacher.
- `beta(k)` — the SDPG warmup→decay schedule `sdpg_beta_base · min(1, k/T_warm) · min(1, (T-k)/T_decay)`, constant `sdpg_beta_base` when `sdpg_beta_warmup_steps`/`sdpg_beta_decay_steps` are `0` (their default).
- `L_ref` — optional unnormalized-KL regularization to a frozen reference model, active only when `reference_kl_coef > 0` (the reference defaults to the student's init weights).

Student and teacher sequences differ in length, so the response rows are gathered per sample by their label masks and aligned positionally. Losses and schedule live in `src/trainers/distillation/losses.py`.

## Quick start

```bash
torchrun --nproc_per_node=8 \
    scripts/training/distillation/self_distill.py \
    examples/distillation/qwen3_5/self-distill-qwen3.5-9b.yaml
```

Qwen3.5-9B is dense, so this runs plain FSDP2 data parallel; a MoE student takes `--expert_parallel_size` (EP applies to the student only).

Core fields of that config:

```yaml
model_name_or_path: Qwen/Qwen3.5-9B

dataset: open-r1/OpenR1-Math-220k:default  # raw SFT conversations + an answer field
conversation_field: messages
sdpg_answer_field: answer

sdpg_beta_base: 1.0
sdpg_beta_warmup_steps: 50                 # ramp OPD in after SFT settles
sdpg_beta_decay_steps: 100                 # phase OPD out near the end

train_on_completions_only: true
assistant_message_template: "<|im_start|>assistant\n"   # required — the marker your model's chat template renders

learning_rate: 1.0e-5
max_length: 16384                          # R1 traces run long; the collator fails loud on over-length rows
gradient_checkpointing: true
```

The config is TRL's `SFTConfig`, so `max_length` defaults to `1024` — set it explicitly. `assistant_message_template` has no default (`src/args/mixins.py`) and is required whenever `train_on_completions_only` is on — the collator refuses the pair at construction, before the first batch — and it must byte-match the model's rendered assistant-turn prefix, since a mismatched marker would mask every row.

## Parallelism

EP, TP, and ETP apply. CP is rejected (both on the trainer and at config-build time in the script) because the privileged teacher uses a separate, longer sequence that sequence-sharded attention cannot reconstruct; PP is rejected because that second forward would give the stage boundary a different activation shape. Use gradient checkpointing for long sequences. Full matrix: [Trainer Compatibility](../../reference/trainer-architecture.md#trainer-compatibility).

## Vision-language support

`self_distill.py` handles both modalities, routing the data path on the run (`is_vlm_run`): a multimodal student distilled on text-only rows takes the text path. On an image-declaring run the script maps raw conversations into `history`/`images` columns through the shared `prepare_vlm_dataset` (keeping the privileged fields) and uses `SelfDistillVLMDataCollator` (`src/data/collators/vlm.py`).

The over-length pre-filter runs on the student text and does not see the hint, so keep `max_length` headroom for it — the teacher branch is never truncated and fails loud past `max_length`. The teacher's text-only hint does not change the image grid, so the student's image features are shared with the teacher forward (cached and replayed for `lfm2_vl` to skip a vision-tower re-encode).

## Configuration

`SelfDistillationArguments` (`src/args/self_distill_args.py`) extends `SFTScriptArguments`. Its tracking project defaults to `self-distillation` rather than inheriting SFT's `sft-tuning`, so set `project_name` explicitly to keep a run's history alongside earlier ones. Fields beyond the SFT set:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `sdpg_hint_template` | (see above) | Template appended to the last user turn for the teacher forward |
| `sdpg_answer_field` | `"answer"` | Dataset field with the ground-truth answer |
| `privileged_solution_field` | `"solution"` | Optional dataset field for `{solution}` |
| `sdpg_loss` | `"reverse_kl"` | OPD loss: `reverse_kl`, `forward_kl`, `unnormalized_kl` |
| `sdpg_temperature` | `1.0` | OPD softmax temperature |
| `sdpg_beta_base` | `1.0` | Base OPD coefficient |
| `sdpg_beta_warmup_steps` | `0` | Steps to ramp beta 0→`sdpg_beta_base` |
| `sdpg_beta_decay_steps` | `0` | Final steps over which beta decays to 0 |
| `reference_kl_coef` | `0.0` | Alpha for KL regularization to a frozen reference; 0 loads no reference |
| `reference_kl_loss` | `"unnormalized_kl"` | Reference regularizer: `unnormalized_kl`, `reverse_kl`, `forward_kl` |
| `reference_model_name_or_path` | `None` | Frozen reference; defaults to the student's init weights when `reference_kl_coef > 0` |
| `confidence_field` | `None` | Per-sample confidence in [0, 1]; weights SFT (and OPD) by `confidence**confidence_power` |
| `confidence_power` | `4.0` | Exponent `p` in `conf**p` |
| `confidence_weight_opd` | `True` | Apply the confidence weight to OPD too |
| `opd_exclude_eos` | `True` | Drop EOS/stop tokens from OPD (not from SFT) |

Inherited SFT knobs the script refuses rather than silently ignores: `generate_eval_examples` and
`num_eval_examples` — the generation callback needs a tokenized `generate` split and self-distillation
keeps the dataset raw, so **any non-default value** of either raises — plus
`train_on_last_assistant_only` (the SelfDistill collators mask all assistant turns) and
`packing` / `padding_free` (both branches are tokenized at collation and always right-padded).
