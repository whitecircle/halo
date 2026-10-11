# GRPO

Group Relative Policy Optimization scores a prompt's completions against each other, not against a learned value model. For prompt `x` with completions `{y1..yk}` and rewards `{r1..rk}`, the advantage is `A(yi) = normalize(ri - mean(r))` and the loss is `L = -E[A(yi) * log pi(yi|x)]`.

| | Offline GRPO | Online GRPO (RLVR) | Async GRPO with Environments |
|---|---|---|---|
| Trainer | `OfflineGRPOTrainer` | `DistributedGRPOTrainer` | `DistributedAsyncEnvironmentalGRPOTrainer` |
| Script | `offline_grpo.py` | `online_grpo/rlvr.py` | `environmental_grpo.py` |
| Generation | None, pre-collected | Online, vLLM only | Online, vLLM or SGLang (`rollout_backend`), driven by Ray actors |
| Rewards | Pre-scored in the dataset | [`rewards:` terms](rewards/README.md): `accuracy`, `format` and the [external terms](rewards/README.md#common-options) | [`rewards:` terms](rewards/README.md): the `environment` grade plus the [external terms](rewards/README.md#common-options) |
| Turns | Single | Single | Multi-turn (single-turn environments exist) |
| Config | `OfflineGRPOConfig` | TRL `GRPOConfig` | TRL `GRPOConfig` + `EnvironmentConfig` + `AsyncTrainingConfig` |
| Infrastructure | Training only | Training + one vLLM server | Training + rollout servers + Ray |
| CP | Full fine-tuning (CP, EP+CP) | No | No |
| Best for | Large pre-scored datasets, no generation cost | Single-turn verifiable answers: math, structured output | Tool use, code generation, environment feedback |

All three extend `DistributedTrainerMixin` and run under EP, TP, pure ETP, EP+TP and EP+ETP. Online GRPO runs only against a vLLM server ([Server mode only](online.md#server-mode-only)).

## Launch

Offline GRPO needs no server ([Launch](offline.md#launch)). Online GRPO and Async GRPO generate against servers on GPUs the trainer does not use; by compose default vLLM takes GPU 7 and the trainer the other seven ([Online launch](online.md#launch), [Async quickstart](../async-grpo/README.md#quickstart)).

## Pages

- [Offline GRPO](offline.md)
- [Online GRPO (RLVR)](online.md)
- [Reward Terms](rewards/README.md): the `rewards:` list the online and environment arms read.
- [Async GRPO with Environments](../async-grpo/README.md) and its [environments](../async-grpo/environments/README.md)

Pairwise preference alternatives: [SMPO](../preference/smpo.md) · [DPO](../preference/dpo.md). All methods: [Training Methods Overview](../README.md).
