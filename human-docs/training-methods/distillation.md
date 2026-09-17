# Distillation

Distillation fits a student to a teacher's next-token distribution instead of to hard labels, which
carries more signal per token than plain [SFT](sft.md). Halo has three variants, and what separates
them is where the teacher comes from and whether the data is fixed or generated live.

| Variant | Teacher | Needs a second model | Needs a rollout server | `halo launch` |
| --- | --- | :--: | :--: | --- |
| Teacher distillation | a separate frozen model | yes | no | `teacher-distill` |
| Self-distillation | the student, shown the answer | no | no | `self-distill` |
| Online SDPG | the student, shown the answer | no | yes (vLLM) | `rlvr --use_sdpg=true` |

Expert, tensor and expert-tensor parallelism apply to the student in all three. None of them takes
context parallelism: the two offline variants need a second whole-model forward a sequence split
cannot reproduce, and SDPG inherits Online GRPO's refusal.

## Teacher distillation

Both models sit on every rank — the student with its optimizer state, the teacher weights-only under
`no_grad`. Construction raises unless the two vocabulary sizes match, so teacher and student should
come from the same family: equal sizes over a different token map pass the check and train on
misaligned targets. Memory is the binding constraint: LoRA on the student is the usual
answer, and the shipped recipe uses it.

Data is the SFT conversation format, under `conversation_field` (default `messages` for this script).
From `examples/distillation/qwen3_5/distill-qwen3.5-9b-from-qwen3.6-35b-a3b.yaml`:

```yaml
model_name_or_path: Qwen/Qwen3.5-9B          # student
teacher_model: Qwen/Qwen3.6-35B-A3B          # teacher, same tokenizer family
distill_loss: kl_divergence
distill_alpha: 0.5                           # even split between distillation and hard-label CE
dataset: allenai/tulu-3-sft-mixture
assistant_message_template: "<|im_start|>assistant\n"
max_length: 16384
learning_rate: 5.0e-05
use_peft: true
lora_r: 16
```

`distill_alpha` weights the divergence term against the cross-entropy term; `1.0` drops
cross-entropy entirely. `distill_loss` takes eight losses — `kl_divergence` is the default and
the one to start from; `cosine_similarity` tolerates teachers whose logit scale differs. Over-length
conversations are dropped, not truncated.

```bash
halo launch teacher-distill \
    examples/distillation/qwen3_5/distill-qwen3.5-9b-from-qwen3.6-35b-a3b.yaml -n 8
```

## Self-distillation

No second model and no generation. The same weights run twice per step: once on the prompt, once on
the prompt plus a hint that reveals the gold answer. The hinted forward is the teacher, and the
student is pulled toward it on the shared response tokens, on top of an ordinary SFT loss. Use it
when nothing stronger than your own model is available but the dataset has answers.

From `examples/distillation/qwen3_5/self-distill-qwen3.5-9b.yaml`:

```yaml
model_name_or_path: Qwen/Qwen3.5-9B
dataset: open-r1/OpenR1-Math-220k:default
conversation_field: messages
sdpg_hint_template: "\n[Hint] The correct answer is: {answer}. Do NOT state that you were given the answer.\n"
sdpg_loss: reverse_kl
sdpg_beta_warmup_steps: 50
sdpg_beta_decay_steps: 100
opd_exclude_eos: true
reference_kl_coef: 0.0
assistant_message_template: "<|im_start|>assistant\n"
max_length: 16384
learning_rate: 1.0e-5
```

The hint is filled from the column named by `sdpg_answer_field` (default `answer`). The warmup and
decay steps ramp the distillation coefficient in after the SFT term has settled and phase it back
out near the end. `reference_kl_coef: 0` means no reference model is loaded at all; raise it only if
you want an anchor against the starting weights, which costs a second resident model.

Two things bite here. `max_length` must have headroom for the hint, because neither branch is ever
truncated — the teacher branch is systematically longer and an over-length row raises. And
`assistant_message_template` has to byte-match what the chat template renders, or every token is
masked and the loss goes flat.

```bash
halo launch self-distill examples/distillation/qwen3_5/self-distill-qwen3.5-9b.yaml -n 8
```

The MoE recipes (`examples/distillation/gptoss/`, `examples/distillation/gemma4/`) pin
`expert_parallel_size: 8` themselves.

## Online SDPG

SDPG is [Online GRPO](online-grpo.md) plus the same privileged-teacher term: the policy generates
rollouts against a vLLM server, earns verifiable rewards, and is additionally distilled toward its
own answer-hinted forward. Reach for it when the policy solves too few prompts for GRPO's group
signal to say anything on its own.

Everything the online trainer needs applies unchanged — a running
[rollout server](../rollout-servers.md), NCCL weight sync, and the `{"prompt", "answer"}`
dataset. No shipped config turns SDPG on; start from
`examples/grpo/online/rlvr-online-grpo-template.yaml`, which carries the block commented out, and
flip the gate:

```bash
# trainer on GPUs 0-6, the vLLM server holding GPU 7
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 halo launch rlvr \
    examples/grpo/online/rlvr-online-grpo-template.yaml -n 7 \
    --use_sdpg=true --sdpg_beta_base=1.0
```

Setting any `sdpg_*` knob without `use_sdpg: true` is refused rather than ignored. A train dataset
with no `answer` column raises, since the hint has nothing to reveal.

## What to watch

On teacher distillation, `distillation_loss` should fall while `sft_loss` stays sane; an OOM on the
first step means both models did not fit, so cut `max_length` or move the student to LoRA. On the two
self-distillation variants, watch `opd_loss` together with the live coefficient (`beta`, or
`opd_beta` online) — a coefficient pinned at zero means the schedule never ramped. An `opd_loss` of
exactly zero under SDPG means no rollout in the batch earned a positive advantage, which is expected
occasionally and a broken verifier if it persists.

SDPG runs two extra full-vocabulary forwards per micro-batch, so it OOMs at batch sizes Online GRPO
survives. Size the batch against that peak.

## Go deeper

- [SFT](sft.md) · [Online GRPO](online-grpo.md) · [Async GRPO](async-grpo-environments.md)
- [Distillation reference](../../agent-docs/training-methods/distillation/README.md) ↗ ·
  [Online SDPG](../../agent-docs/training-methods/distillation/online-sdpg.md) ↗
