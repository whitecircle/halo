# Choosing a Method

Pick by the data you have and the goal. Every method shares one config, collator factory, and parallelism stack, so switching is a config change, not a rewrite. The last column is the `halo launch` name; see [Quickstart](quickstart.md) to run one.

| Method | Use when | Data | `halo launch` |
| --- | --- | --- | --- |
| SFT | Teach format, behavior, or a domain from example conversations | conversations (`prompt`) | `sft` |
| SFT (VLM) | Same, with images — `sft` auto-detects a multimodal model | image conversations | `sft` |
| Classification | Produce a sequence-level label (safety, topic, toxicity) | text + `label` | `classification` |
| Reward modeling | Train a Bradley-Terry scorer to consume later | `chosen` / `rejected` pairs | `rewards` |
| DPO | Align on preference pairs with a KL leash to a reference model | `prompt`, `chosen`, `rejected` | `dpo` |
| SMPO | Reference-free preference alignment, one model, dynamic margin | `prompt`, `chosen`, `rejected` | `smpo` |
| KTO | Feedback is unpaired thumbs-up / thumbs-down, not pairs | `prompt`, `completion`, `label` | `kto` |
| Offline GRPO | You already have completions with reward scores; no live generation | `prompt`, `completions`, `rewards` | `offline-grpo` |
| Online GRPO (RLVR) | Model generates and earns rule-based verifiable rewards (math, format) | `prompt`, `answer` | `rlvr` |
| Environmental GRPO | Multi-turn, tool-using, agentic trajectories | `prompt`, `answer` | `environmental-grpo` |
| Teacher distillation | Compress a separate larger teacher into a smaller student | conversations | `teacher-distill` |
| Self-distillation | Self-improve with a privileged answer hint, no second model | conversations + answer | `self-distill` |
| Online SDPG | On-policy form of self-distillation | `prompt`, `answer` | `rlvr --use_sdpg=true` |
| Embedding | Fine-tune for retrieval, similarity, or clustering | pairs / triplets / scored pairs | `embedding` |

Pretraining runs the `sft` path on raw text. From-scratch adds `--init_from_scratch` and is dense FSDP only; EP/CP/TP/ETP are rejected there, so materialize the random-init checkpoint outside the job if you need parallelism. See the `agent-docs` [Pretraining](../agent-docs/training-methods/pretraining.md) ↗ guide.

## Which of these do I pick

- **DPO vs SMPO vs KTO** — all three train on preference data. DPO keeps a KL leash to a frozen reference, so two models sit in memory. SMPO drops the reference: one model, less memory, and its bounded dynamic-margin loss stops updating a pair once it is well separated where DPO's sigmoid keeps pushing, with an SFT anchor holding generation quality. KTO takes unpaired thumbs-up/down labels instead of pairs.
- **Preference training vs reward modeling** — same pairs, different output: DPO/SMPO/KTO move the policy; `rewards` produces a scorer you consume later (rejection sampling, or as a GRPO reward).
- **Offline vs online vs environmental GRPO** — offline trains from pre-scored completions with no generation; online (RLVR) generates single-turn completions with verifiable rewards; environmental adds multi-turn tool-use environments and async rollouts.
- **Teacher vs self distillation** — teacher-distill transfers from a separate larger model; self-distill uses the same model as its own answer-hinted teacher, offline (`self-distill`) or on-policy (`rlvr --use_sdpg=true`).

## Extra infrastructure

Online GRPO, online SDPG, and environmental GRPO generate with a separate vLLM server. Environmental GRPO also runs Ray rollout actors asynchronously — rollouts overlap training through a prefetch queue — and can serve rollouts from SGLang instead (`rollout_backend: sglang`, with the restrictions in the [Supported Matrix](supported-matrix.md#rollout-engines)). Every other method trains without vLLM or Ray. See the `agent-docs` guides for [Online GRPO](../agent-docs/training-methods/grpo/online-grpo.md) ↗ and [Environmental GRPO](../agent-docs/training-methods/grpo/environmental-grpo.md) ↗.

The full per-method reference, with every hyperparameter, is in the `agent-docs` [training-methods reference](../agent-docs/training-methods/sft.md) ↗.
