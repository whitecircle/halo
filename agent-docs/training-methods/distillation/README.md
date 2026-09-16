# Distillation

Fit a student to a teacher's token-level output distribution. The three methods differ in where the teacher comes from and whether training is on- or off-policy.

| Aspect | [Teacher](teacher-distillation.md) | [Self](self-distillation.md) | [Online SDPG](online-sdpg.md) |
|---|---|---|---|
| Teacher | A separate frozen model | The student, given a hint | The student, given a hint |
| Policy | Off-policy, fixed dataset | Off-policy, fixed responses | On-policy, student rollouts |
| Generation | None | None | vLLM rollouts |
| Objective | `distill_alpha·L_distill + (1−distill_alpha)·L_clm` | `L_sft + beta(k)·L_OPD + alpha·L_ref` | `L_GRPO + beta(k)·L_OPD` |
| Trainer | `DistributedDistillationTrainer` | `DistributedSelfDistillationTrainer` | `DistributedSDPGTrainer` |
| Script | `distillation/teacher_distill.py` | `distillation/self_distill.py` | `online_grpo/rlvr.py --use_sdpg` |
| Modality | Text + VLM | Text + VLM | Text only |

Which to pick:

- A separate, stronger teacher is available → [teacher distillation](teacher-distillation.md).
- One model, gold answers in the dataset, no generation budget → [self-distillation](self-distillation.md).
- One model, a verifier, and a vLLM rollout budget → [online SDPG](online-sdpg.md).

All three take EP, TP, ETP, EP+TP and EP+ETP on the student. None takes CP or PP: each needs a second whole-model forward that neither the sequence split nor the stage split can reproduce.

## Dataset

Teacher distillation and self-distillation read the SFT conversation format under `conversation_field` — default `messages` for teacher distillation, `prompt` for self-distillation, so set it to match your data.

```jsonl
{"messages": [{"role": "user", "content": "Explain gravity."}, {"role": "assistant", "content": "Gravity is a fundamental force..."}]}
```

Self-distillation also reads a gold-answer column (`sdpg_answer_field`, default `answer`) to build the teacher's hint. Online SDPG reads the online-GRPO `{"prompt", "answer"}` format, where the answer feeds both the verifier reward and the hint.
