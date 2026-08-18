# GRPO Variant Comparison

| Aspect | Offline GRPO | RLVR Online GRPO | Environmental GRPO |
|--------|-------------|-------------------|-------------------|
| Trainer | `OfflineGRPOTrainer` | `DistributedGRPOTrainer` | `DistributedAsyncEnvironmentalGRPOTrainer` |
| Script | `offline_grpo.py` | `online_grpo/rlvr.py` | `environmental_grpo.py` |
| Generation | None (pre-collected) | Online (vLLM) | Online (vLLM or SGLang, + Ray) |
| Rewards | Pre-computed | Rule-based (`\boxed{}`, regex) | Environment-defined: task success, verifiable answers, or an LLM judge |
| Turns | Single | Single | Multi-turn (single-turn envs exist) |
| Config | `OfflineGRPOConfig` | TRL `GRPOConfig` | TRL `GRPOConfig` + `EnvironmentConfig` + `AsyncTrainingConfig` |
| Infrastructure | Training only | Training + vLLM | Training + rollout engine + Ray |
| GPUs | 1–8+ | 2+ (1 for vLLM) | 2+ (1+ for the rollout server) |
| Best for | Large pre-scored datasets (no generation cost) | Single-turn verifiable answers (math, structured output) | Tool use, code generation, environment feedback |

All three extend `DistributedTrainerMixin` and support EP, TP, pure ETP, EP+TP and EP+ETP; **none supports CP** (GRPO needs whole sequences per rank for its global log-prob sums). Offline GRPO is the only one that declares `_supports_pp` — [pipeline parallelism itself is not yet available in this release](../../parallelism/pipeline-parallelism.md).

Environmental GRPO is the only variant with a rollout-engine choice (`rollout_backend: vllm` default, or `sglang` — refused under expert distribution and for every MoE family but GptOss, see [Rollout backend](environmental-grpo.md#rollout-backend)). RLVR online GRPO is vLLM-only by construction: it drives TRL's vLLM generation path with the vendored NCCL client and rejects in-process and colocate modes.

The online and environmental trainers take TRL's `GRPOConfig` directly, so their objective is tuned through `loss_type` / `epsilon` / `scale_rewards` / `beta`. For a verifiable-reward task (pass-rate in `[0, 1]`, small group) the tuned setting is DAPO + clip-higher + batch-std scaling — see [GRPO objective for verifiable rewards](online-grpo.md#grpo-objective-for-verifiable-rewards). Offline GRPO has its own loss and advantage machinery ([Offline GRPO](offline-grpo.md)).

## Related pages

- [GRPO Family Overview](index.md) — launch commands per variant
- [Environments Overview](environments/index.md) — the environment registry and registry names
- [SMPO](../preference/smpo.md) — pairwise preference alternative
