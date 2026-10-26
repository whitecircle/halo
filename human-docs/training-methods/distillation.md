# Distillation

Distillation fits a student to a teacher's next-token distribution instead of to hard labels, which
carries more signal per token than plain [SFT](sft.md). Halo has three variants. They differ in where
the teacher comes from and whether the data is fixed or generated live.

| Variant | Teacher | Needs a second model | Needs a rollout server | `halo launch` |
| --- | --- | :--: | :--: | --- |
| Teacher distillation | a separate frozen model | yes | no | `teacher-distill` |
| Self-distillation | the student, shown the answer | no | no | `self-distill` |
| Online SDPG | the student, shown the answer | no | yes (vLLM) | `rlvr --use_sdpg=true` |

Expert, tensor and expert-tensor parallelism apply to the student in all three; none takes context
parallelism. The MoE recipes are under `examples/distillation/gptoss/` and
`examples/distillation/gemma4/`.

## Teacher distillation

Data is the SFT conversation format. Both models sit on every rank, the teacher as weights only, so
memory is the binding constraint and the shipped recipe puts LoRA on the student. Before any teacher
weight loads, the script raises unless both tokenizers map every token to the same id: in practice
teacher and student share a tokenizer family.

```bash
halo launch teacher-distill \
    examples/distillation/qwen3_5/distill-qwen3.5-9b-from-qwen3.6-35b-a3b.yaml -n 8
```

`distill_alpha` weights the divergence against the cross-entropy term, and `distill_loss` picks the
divergence; `cosine_similarity` tolerates a teacher whose logit scale differs.

## Self-distillation

No second model and no generation. The same weights run twice per step: once on the prompt, and once on
the prompt plus a hint that reveals the gold answer. The hinted forward is the teacher, and the student
is pulled toward it on top of an ordinary SFT loss. Use it when nothing stronger than your own model is
available but the dataset has answers.

```bash
halo launch self-distill examples/distillation/qwen3_5/self-distill-qwen3.5-9b.yaml -n 8
```

- `max_length` needs headroom for the hint: neither branch is truncated, so an over-length row raises.
- `assistant_message_template` must byte-match what the chat template renders, and
  `train_on_completions_only` must stay on while the distillation term is.
- `reference_kl_coef: 0` loads no reference model. Raising it costs a second resident model, unsharded
  under expert or tensor parallelism.

## Online SDPG

SDPG is [Online GRPO](online-grpo.md) plus the same answer-hinted teacher term. Reach for it when the
policy solves too few prompts for GRPO's group signal to work on its own. It needs everything the online
trainer does: a running [rollout server](../rollout-servers.md) and the `{"prompt", "answer"}` dataset.
No shipped config turns it on; start from the template, which carries the block commented out:

```bash
# trainer on GPUs 0-6, the vLLM server holding GPU 7
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 halo launch rlvr \
    examples/grpo/online/rlvr-online-grpo-template.yaml -n 7 \
    --use_sdpg=true --sdpg_beta_base=1.0
```

SDPG adds two full-vocabulary forwards per micro-batch, so it runs out of memory at batch sizes Online
GRPO survives.

## What to watch

- **Teacher distillation:** `distillation_loss` should fall while `sft_loss` stays sane. An OOM on the
  first step means both models did not fit; cut `max_length` or move the student to LoRA.
- **Self-distillation and SDPG:** watch `opd_loss` with the live coefficient `opd_beta`. A coefficient
  pinned at zero means the schedule never ramped.
- **SDPG:** an `opd_loss` of exactly zero means no rollout in the batch earned a positive advantage.
  That is expected now and then, and a broken verifier if it persists.

Every knob, loss and refusal:
[Teacher](../../agent-docs/training-methods/distillation/teacher-distillation.md) ↗ ·
[Self](../../agent-docs/training-methods/distillation/self-distillation.md) ↗ ·
[Online SDPG](../../agent-docs/training-methods/distillation/online-sdpg.md) ↗.
