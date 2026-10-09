# Choosing a method

Pick a method by the data you have and what you want the model to do. Every
method uses the same YAML format and parallelism stack, so switching is a config
change. The last column is the name you pass to `halo launch`
([Quickstart](quickstart.md)).

| Method | Use when | Data | `halo launch` |
| --- | --- | --- | --- |
| SFT | Teach format, behavior or a domain from example conversations | conversations (`prompt`) | `sft` |
| SFT (VLM) | Same, with images. `sft` takes the VLM path when a multimodal checkpoint meets image data | image conversations | `sft` |
| Classification | Predict a sequence-level label (safety, topic, toxicity) | conversation (`prompt`) or raw text, + `label` | `classification` |
| Reward modeling | Train a Bradley-Terry scorer to use later | `chosen` / `rejected` pairs | `rewards` |
| DPO | Align on preference pairs, held close to a reference model by a KL term | `chosen`, `rejected` (+ optional `prompt`) | `dpo` |
| SMPO | Align on preference pairs without a reference model | `chosen`, `rejected` (+ optional `prompt`) | `smpo` |
| KTO | Your feedback is unpaired thumbs-up / thumbs-down, not pairs | `prompt`, `completion`, `label` | `kto` |
| [Offline GRPO](training-methods/offline-grpo.md) | You already have scored completions; no live generation | `prompt`, `completions`, `rewards` | `offline-grpo` |
| [Online GRPO (RLVR)](training-methods/online-grpo.md) | The model generates and earns rule-based, verifiable rewards (math, format) | `prompt`, `answer` | `rlvr` |
| [Async GRPO with Environments](training-methods/async-grpo-environments.md) | Multi-turn, tool-using, agentic trajectories | `prompt`, plus `answer` where the environment grades one | `environmental-grpo` |
| Teacher distillation | Compress a separate, larger teacher into a smaller student | conversations | `teacher-distill` |
| Self-distillation | Improve the model with a privileged answer hint, no second model | conversations + answer | `self-distill` |
| Online SDPG | On-policy self-distillation | `prompt`, `answer` | `rlvr --use_sdpg=true` |
| Embedding | Tune for retrieval, similarity or clustering | pairs, triplets or scored pairs | `embedding` |

## Similar methods compared

- **DPO, SMPO or KTO.** All three train on preference data. DPO scores against a
  frozen reference model, a second copy in memory unless you train LoRA or
  precompute the reference log-probs. SMPO needs no reference: its bounded loss
  stops pushing a pair once it is well separated, and an SFT term protects
  generation quality. KTO takes unpaired thumbs-up / thumbs-down labels.
- **Preference training or reward modeling.** Same pairs, different output.
  DPO, SMPO and KTO change the policy. `rewards` trains a scorer you use later,
  for rejection sampling or as a GRPO reward.
- **Offline, online or async GRPO.** Offline GRPO trains on pre-scored
  completions and generates nothing. Online GRPO (RLVR) generates single-turn
  completions and scores them with verifiable rewards. Async GRPO with
  environments adds multi-turn tool use and overlaps rollouts with training.
- **Teacher or self-distillation.** `teacher-distill` learns from a separate,
  larger model. Self-distillation uses the same model, given the answer as a
  hint, as its own teacher: offline (`self-distill`) or on-policy
  (`rlvr --use_sdpg=true`).

## Pretraining

Pretraining runs `sft` on raw text that you tokenize offline first
(`halo run prepare-dataset --mode text`). To train from random weights, set
`init_from_scratch: true`.

From-scratch runs support data parallelism only (one GPU, DDP or FSDP2) and
reject EP, CP, TP and ETP. To use those, create the random-init checkpoint in a
separate step and train from it. See
[Pretraining](../agent-docs/training-methods/pretraining.md) ↗.

## Extra infrastructure

Online GRPO, online SDPG and async GRPO with environments generate with a
separate vLLM server ([Rollout servers](rollout-servers.md)). Every other method
trains without vLLM or Ray.

Async GRPO also runs Ray rollout actors. With two or more servers it overlaps
rollouts with training through a prefetch queue. It can generate with SGLang
instead (`rollout_backend: sglang`), with the limits listed in the
[Supported matrix](supported-matrix.md#rollout-engines).

Every hyperparameter, per method, is in the
[training methods reference](../agent-docs/training-methods/README.md) ↗.
