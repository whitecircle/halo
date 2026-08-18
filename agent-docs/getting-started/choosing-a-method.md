# Choosing a Training Method

The data you have picks the method: conversation turns → SFT, preference pairs → SMPO/DPO/reward,
unpaired thumbs-up/down → KTO, pre-scored completions → offline GRPO, live generation → online or
environmental GRPO.

## Method comparison

| Method | Input | Reference model | Online generation | Script |
|---|---|:--:|:--:|---|
| [SFT](../training-methods/sft.md) | Conversation turns | No | No | `scripts/training/sft.py` |
| [SMPO](../training-methods/preference/smpo.md) | prompt + chosen + rejected | No | No | `scripts/training/preference/smpo.py` |
| [DPO](../training-methods/preference/dpo.md) | prompt + chosen + rejected | Yes | No | `scripts/training/preference/dpo.py` |
| [KTO](../training-methods/preference/kto.md) | prompt + completion + bool label | Yes | No | `scripts/training/preference/kto.py` |
| [Offline GRPO](../training-methods/grpo/offline-grpo.md) | prompt + completions + rewards | No | No | `scripts/training/offline_grpo.py` |
| [Online GRPO (RLVR)](../training-methods/grpo/online-grpo.md) | prompt + ground-truth answer | No | vLLM | `scripts/training/online_grpo/rlvr.py` |
| [Environmental GRPO](../training-methods/grpo/environmental-grpo.md) | prompt (+ expected answer per env) | No | vLLM/SGLang + Ray | `scripts/training/environmental_grpo.py` |
| [Distillation](../training-methods/distillation/README.md) | Conversation turns | Teacher | No | `scripts/training/distillation/teacher_distill.py` |
| [Reward modeling](../training-methods/preference/reward-modeling.md) | chosen + rejected | No | No | `scripts/training/preference/rewards.py` |
| [Classification](../training-methods/classification.md) | prompt + label | No | No | `scripts/training/classification.py` |
| [Embedding](../training-methods/embedding.md) | Sentence pairs / triplets | No | No | `scripts/training/embedding.py` |

Every trainer supports EP/TP/ETP. CP is SFT and SMPO only. Pipeline parallelism is
[not yet available in this release](../parallelism/pipeline-parallelism.md). Full matrix:
[Trainer Compatibility](../reference/trainer-architecture.md#trainer-compatibility).

## Picking between close alternatives

- **SMPO vs DPO** — SMPO is reference-free (trains one model, not two) and its bounded dynamic-margin loss plus built-in SFT anchor hold steadier than DPO, which keeps pushing chosen and rejected apart past separation. Choose DPO when you want the KL constraint to a specific reference checkpoint.
- **KTO vs DPO/SMPO** — KTO takes unpaired thumbs-up/down feedback, so use it when you never collected pairs.
- **Reward modeling vs preference training** — a Bradley-Terry scorer for rejection sampling or as a GRPO reward signal, not an aligned policy.
- **Offline vs online GRPO** — offline replays completions you already scored (no vLLM server, variable group sizes); online generates and scores live.
- **Environmental GRPO** — the only multi-turn option: the model acts, receives observations, and learns from trajectory rewards, with async Ray-actor rollouts and NCCL weight sync to vLLM or SGLang (`rollout_backend`). The environment comes from `environment_type`; see the [Environments overview](../training-methods/grpo/environments/README.md) for the built-in list and [Custom Environments](../training-methods/grpo/environments/custom-environments.md) to add one.
- **Teacher vs self distillation** — `teacher_distill.py` transfers from a larger model; two privileged-context self-distillation variants use one model as both student and teacher, offline (`self_distill.py`) and on-policy SDPG (`rlvr.py --use_sdpg=true`). See the [Distillation guide](../training-methods/distillation/README.md).
