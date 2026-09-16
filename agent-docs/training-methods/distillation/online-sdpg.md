# Online SDPG

Self-Distilled Policy Gradient ([arXiv:2606.04036](https://arxiv.org/abs/2606.04036)) is [Online GRPO (RLVR)](../grpo/online-grpo.md) plus a privileged-teacher distillation term. Each step, the same policy is re-run under `no_grad` in eval mode with a hint revealing the gold answer, and the student is distilled toward that teacher's next-token distribution: `L = L_GRPO + beta(k)·L_OPD`. `DistributedSDPGTrainer` (`src/trainers/distillation/sdpg.py`) subclasses the online trainer, so vLLM server rollouts, NCCL weight sync, EP/TP/ETP and the LoRA rules carry over unchanged.

Use it when a verifiable answer exists and the policy solves too few prompts for GRPO's group signal alone. At `sdpg_beta_base: 0` the trainer is plain online GRPO. For a fixed dataset instead of live rollouts use [self-distillation](self-distillation.md).

## Configuration

Every knob lives on the RLVR script arguments and is read only with `use_sdpg: true`; setting one away from its default with the gate off is refused before the model loads. No shipped YAML enables it — start from `examples/grpo/online/rlvr-online-grpo-template.yaml`, which carries the block commented out.

| Knob | Default | Effect |
|---|---|---|
| `use_sdpg` | `false` | Swap in `DistributedSDPGTrainer` |
| `sdpg_beta_base` | `1.0` | Base OPD coefficient; `0` drops the term |
| `sdpg_beta_warmup_steps` | `0` | Steps to ramp beta from 0 to `sdpg_beta_base` |
| `sdpg_beta_decay_steps` | `0` | Final steps over which beta decays to 0 |
| `sdpg_loss` | `reverse_kl` | OPD loss: `reverse_kl`, `forward_kl` or `unnormalized_kl` |
| `sdpg_temperature` | `1.0` | OPD softmax temperature |
| `sdpg_hint_template` | `\n[Hint] The correct answer is: {answer}. Do NOT state that you were given the answer.\n` | Appended to the prompt for the teacher forward only |
| `opd_positive_advantage_only` | `true` | Restrict OPD to rows with a positive advantage; `false` distills every completion row |

The hint is tokenized and appended to each rollout's prompt ids, so the term is text-only. It reads the pinned `answer` column that `process_for_rlvr` normalizes `answer_field` into — a train dataset without that column raises at construction whenever `sdpg_beta_base` is non-zero.

## Launch

```bash
# vLLM on GPU 7 and the trainer on 0-6, the compose defaults (server setup: Online GRPO)
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 torchrun --nproc_per_node=7 \
    scripts/training/online_grpo/rlvr.py \
    examples/grpo/online/rlvr-online-grpo-template.yaml \
    --use_sdpg=true --sdpg_beta_base=1.0
```

From Python, construct the trainer exactly as the online one plus the SDPG kwargs:

```python
trainer = DistributedSDPGTrainer(
    model=model,
    reward_funcs=[accuracy_reward],      # src/rewards/verifiable.py
    args=grpo_config,                      # use_vllm=True, vllm_mode="server"
    train_dataset=train_dataset,           # needs an "answer" column
    processing_class=tokenizer,
    parallelism_config=parallelism_config,
    sdpg_loss="reverse_kl",
    sdpg_beta_base=1.0,
)
```

## Testing a setup

Run a three-step smoke against a live server before the real run: take one of the smoke configs under `examples/grpo/online/` and add `--use_sdpg=true`. The term is covered by `pytest tests/cpu/trainers/test_sdpg_trainer.py tests/cpu/trainers/test_distillation_shared_losses.py -m cpu` and, end to end, by `tests/gpu/trainers/grpo/test_online_grpo_vllm_e2e.py --mode sdpg` plus the `--trainer sdpg` rows of the MoE and dense suites.

## What to watch

Two keys on top of the online-GRPO metrics: `opd_beta` (the live coefficient, prefixed `eval_` under evaluation) and `opd_loss`. Treat a finite `opd_loss` with a non-zero `opd_beta` as healthy.

- **`opd_loss` is exactly 0** — under `opd_positive_advantage_only: true`, no rollout in the batch earned a positive advantage, so the gate masked every token. Expected on easy or fully-failed batches; persistent zeros mean the verifier never separates a group.
- **OOM the online trainer did not hit** — OPD adds two full-vocabulary forwards per micro-batch, neither trimmed with `logits_to_keep`, both logit planes upcast to fp32, and the teacher pass runs the longer prompt+hint+completion sequence. `use_chunked_grpo_logprobs` bounds the GRPO half only; size the batch against the OPD peak.
- **Missing-answer warning** — a row whose answer is blank gets no hint, so it distills toward an unprivileged teacher. Fix the `answer_field` mapping.

`use_liger_kernel` is cleared with a warning (`disable_trl_liger`): TRL's fused GRPO-Liger loss replaces the trainer's own and would silently drop the OPD term. Model-level Liger kernels still apply at load.

## Related pages

- [Online GRPO (RLVR)](../grpo/online-grpo.md) · [Self-Distillation](self-distillation.md) · [Distillation Overview](README.md)
