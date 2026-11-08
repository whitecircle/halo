# Online SDPG

Self-Distilled Policy Gradient ([arXiv:2606.04036](https://arxiv.org/abs/2606.04036)) is [Online GRPO (RLVR)](../grpo/online.md) plus a privileged-teacher distillation term. Each step, the same policy is re-run under `no_grad` in eval mode with a hint revealing the gold answer, and the student is distilled toward that teacher's next-token distribution: `L = L_GRPO + beta(k)·L_OPD`. `DistributedSDPGTrainer` (`halo/trainers/distillation/sdpg.py`) subclasses the online trainer, so vLLM server rollouts, NCCL weight sync, EP/TP/ETP, the LoRA rules and `balance_rank_tokens` ([Rank balance](../async-grpo/performance.md#rank-balance)) carry over.

Use it when a verifiable answer exists and the policy solves too few prompts for GRPO's group signal alone. At `sdpg_beta_base: 0` the trainer is plain online GRPO. For a fixed dataset instead of live rollouts use [self-distillation](self-distillation.md). The engine, the families and the axes online GRPO takes carry over (vLLM only, CP and PP refused, [Online GRPO](../grpo/online.md)); `packed_microbatch_tokens` does not ([Refusals](#refusals)).

## Configuration

Every knob lives on the RLVR script arguments and is read only with `use_sdpg: true`; setting one away from its default with the gate off is refused before the model loads. An out-of-range value is refused at parse time, gate on or off. No shipped YAML enables it — start from `examples/grpo/online/rlvr-online-grpo-template.yaml`, which carries the block commented out.

The hint (`sdpg_hint_template`) is appended to the rendered generation prompt for the teacher forward only; `{answer}` is the one placeholder it fills. `opd_positive_advantage_only` restricts OPD to rows with a positive advantage; `false` distills every completion row. Types, defaults and bounds: [SDPGArguments](../../reference/configuration-reference.md#sdpgarguments) and [OnlineGRPOScriptArguments](../../reference/configuration-reference.md#onlinegrposcriptarguments).

OPD runs on the tokens the GRPO loss trains, so tool-output tokens are excluded. The hint is tokenized and appended to each rollout's prompt ids, so the term is text-only. It reads the pinned `answer` column that `process_for_rlvr` normalizes `answer_field` into — a train dataset without that column raises at construction whenever `sdpg_beta_base` is non-zero and the template names `{answer}`, and a loss batch without the teacher prompts raises rather than training plain GRPO.

## Refusals

- **`packed_microbatch_tokens`**: a packed slot cannot pair the student's per-row logits with the teacher pass ([Packed micro-batches](../async-grpo/performance.md#packed-micro-batches)).

## Launch

```bash
# vLLM on GPU 7 and the trainer on 0-6, the compose defaults (server setup: Online GRPO)
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 torchrun --nproc_per_node=7 \
    scripts/training/grpo/online.py \
    examples/grpo/online/rlvr-online-grpo-template.yaml \
    --use_sdpg=true --sdpg_beta_base=1.0
```

From Python, construct the trainer exactly as the online one plus the SDPG kwargs:

```python
trainer = DistributedSDPGTrainer(
    model=model,
    reward_funcs=[accuracy_reward],
    args=grpo_config,                      # use_vllm=True, vllm_mode="server"
    train_dataset=train_dataset,           # needs an "answer" column
    processing_class=tokenizer,
    parallelism_config=parallelism_config,
    sdpg_answer_field="answer",            # required, like opd_positive_advantage_only
    opd_positive_advantage_only=True,
    sdpg_loss="reverse_kl",
    sdpg_beta_base=1.0,
)
```

## Testing a setup

Run a three-step smoke against a live server before the real run: take one of the smoke configs under `examples/grpo/online/` and add `--use_sdpg=true`. The term is covered by `pytest tests/cpu/trainers/test_sdpg_trainer.py tests/cpu/trainers/test_distillation_divergences.py -m cpu` and, end to end, by the dense and MoE `--trainer sdpg` rows of `tests/gpu/trainers/grpo/test_online_grpo_vllm_e2e.py`.

## What to watch

Two keys on top of the online-GRPO metrics: `opd_beta` (the live coefficient, prefixed `eval_` under evaluation) and `opd_loss`. Treat a finite `opd_loss` with a non-zero `opd_beta` as healthy.

Under evaluation `opd_loss` reads the split's own rows alone, but the rest is TRL's online eval: generating the final round's padding prompts, the GRPO term and its reward, length, KL and clamp metrics count them like real prompts ([Eval padding](../grpo/online.md#what-to-watch)).

- **`opd_loss` is exactly 0** — under `opd_positive_advantage_only: true`, no rollout in the batch earned a positive advantage, so the gate masked every token. Expected on easy or fully-failed batches; persistent zeros mean the verifier never separates a group.
- **OOM the online trainer did not hit** — OPD reads the student's `[B, C, V]` completion logits off the GRPO loss forward and keeps that plane alive until backward (where the forward takes `logits_to_keep`, the prompt rows stay out of it; the divergence itself chunks, [memory](README.md)). With `use_chunked_grpo_logprobs` the no-grad teacher forward on the longer prompt+hint+completion sequence yields hidden states only, and the head projects them in token chunks inside the divergence, so no teacher plane exists. Without it the teacher forward still builds its `[B, C, V]` plane, as the full-logits scorer holds the student's anyway (the head transform of a family is verified only for the chunked scorer). Size the batch against the student plane.
- **Missing-answer warning** — a row whose answer is blank gets no hint, so it distills toward an unprivileged teacher. Fix the `answer_field` mapping.

TRL's `use_liger_kernel` is off, as on every online GRPO run ([Forced and refused settings](../grpo/online.md#forced-and-refused-settings)).

## Related pages

- [Online GRPO (RLVR)](../grpo/online.md) · [Self-Distillation](self-distillation.md) · [Distillation Overview](README.md)
- [Configuration Reference](../../reference/configuration-reference.md#onlinegrposcriptarguments) — the RLVR script arguments the `sdpg_*` fields extend
