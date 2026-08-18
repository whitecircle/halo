# Distillation

Transfer a teacher model's token-level output distribution into a student. Three methods differ in where the teacher comes from and whether training is on- or off-policy.

| Aspect | [Teacher distillation](teacher-distillation.md) | [Self-distillation](self-distillation.md) | [Online SDPG](online-sdpg.md) |
|--------|------|------|------|
| Teacher | A separate, frozen (usually larger) model | The student itself, given a privileged hint | The student itself, given a privileged hint |
| Policy | Off-policy (scores a fixed dataset) | Off-policy (scores fixed response tokens) | On-policy (student rollouts) |
| Generation | None | None | vLLM rollouts (like online GRPO) |
| Objective | `distill_alpha·L_distill + (1-distill_alpha)·L_clm` | `L_sft + beta(k)·L_OPD + alpha·L_ref` | `L_GRPO + beta(k)·L_OPD` |
| Trainer | `DistributedDistillationTrainer` | `DistributedSelfDistillationTrainer` | `DistributedSDPGTrainer` |
| Script | `scripts/training/distillation/teacher_distill.py` | `scripts/training/distillation/self_distill.py` | `scripts/training/online_grpo/rlvr.py --use_sdpg=true` |
| Parallelism | EP, TP, ETP, EP+TP, EP+ETP | EP, TP, ETP, EP+TP, EP+ETP | EP, TP, ETP, EP+TP, EP+ETP |
| Modality | Text + VLM | Text + VLM | Text only |

None of the three supports CP or PP: each needs a second forward the sequence split or the stage split cannot reproduce.

`DistributedSelfDistillationTrainer` subclasses `DistributedSFTTrainer` (reusing SFT data handling), `DistributedSDPGTrainer` subclasses `DistributedGRPOTrainer` (reusing the rollout loop), and `DistributedDistillationTrainer` extends `DistributedTrainerMixin` directly. The OPD reverse-KL losses and the `beta(k)` warmup→decay schedule shared by self-distillation and online SDPG live in `src/trainers/distillation/losses.py`; the eight off-policy teacher losses are in `teacher_losses.py`.

## Dataset format

Teacher distillation and self-distillation read the standard SFT conversation format under `conversation_field` (default `"messages"` for teacher distillation, `"prompt"` for self-distillation — set it to match your data):

```jsonl
{"messages": [{"role": "user", "content": "Explain gravity."}, {"role": "assistant", "content": "Gravity is a fundamental force..."}]}
```

Self-distillation additionally reads a privileged answer field (`sdpg_answer_field`, default `answer`) used to build the teacher's hint. Online SDPG reads the online-GRPO format `{"prompt": [...], "answer": "..."}` — the answer feeds both the verifier reward (via `answer_field`) and the hint.

## Which to pick

- A separate, stronger teacher is available → [teacher distillation](teacher-distillation.md).
- One model, a dataset with gold answers, no generation budget → [self-distillation](self-distillation.md) (offline SDPG approximation).
- One model, a verifier, and a vLLM rollout budget → [online SDPG](online-sdpg.md) (faithful on-policy SDPG).
