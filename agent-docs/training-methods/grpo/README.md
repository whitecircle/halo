# GRPO

Group Relative Policy Optimization scores a prompt's completions against each other instead of against a learned value model: for prompt `x` with completions `{y1..yk}` and rewards `{r1..rk}`, the advantage is `A(yi) = normalize(ri - mean(r))` and the loss is `L = -E[A(yi) * log pi(yi|x)]`. Three variants ship, differing in where the completions and their rewards come from.

| | Offline GRPO | Online GRPO (RLVR) | Async GRPO with Environments |
|---|---|---|---|
| Trainer | `OfflineGRPOTrainer` | `DistributedGRPOTrainer` | `DistributedAsyncEnvironmentalGRPOTrainer` |
| Script | `offline_grpo.py` | `online_grpo/rlvr.py` | `environmental_grpo.py` |
| Generation | None, pre-collected | Online, vLLM | Online, vLLM or SGLang, driven by Ray actors |
| Rewards | Pre-scored in the dataset | The config's `rewards:` terms: `accuracy` (strict `\boxed{}` match), `format`, `judge`, `reward_model` | The config's `rewards:` terms: the `environment` grade plus `judge` and `reward_model` |
| Turns | Single | Single | Multi-turn (single-turn environments exist) |
| Config | `OfflineGRPOConfig` | TRL `GRPOConfig` | TRL `GRPOConfig` + `EnvironmentConfig` + `AsyncTrainingConfig` |
| Infrastructure | Training only | Training + one vLLM server | Training + rollout servers + Ray |
| Best for | Large pre-scored datasets, no generation cost | Single-turn verifiable answers: math, structured output | Tool use, code generation, environment feedback, policy compliance |

All three extend `DistributedTrainerMixin` and run under EP, TP, pure ETP, EP+TP and EP+ETP; none supports CP, since GRPO needs whole sequences per rank for its log-prob sums, and Offline GRPO alone declares `_supports_pp` — [pipeline parallelism](../../parallelism/pipeline-parallelism.md) is not yet available in this release.

Async GRPO is the only variant with an engine choice, `rollout_backend: vllm | sglang`. Online GRPO is vLLM-only by construction: it drives TRL's vLLM server path through the vendored NCCL client and rejects in-process and colocate generation.

## Launch

Offline GRPO needs no server.

```bash
torchrun --nproc_per_node=8 scripts/training/offline_grpo.py \
    examples/grpo/offline/qwen3_5/offline-grpo-qwen3.6-35b-a3b-gsm8k.yaml --expert_parallel_size=8
```

The other two generate against a server, so the trainer takes the remaining GPUs.

```bash
docker compose -f docker-compose.vllm.yml up vllm-server   # vLLM on GPU 7, the compose default
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 torchrun --nproc_per_node=7 \
    scripts/training/online_grpo/rlvr.py examples/grpo/online/rlvr-online-grpo-template.yaml
```

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 torchrun --nproc_per_node=7 \
    scripts/training/environmental_grpo.py \
    examples/grpo/environmental/environmental-grpo-template.yaml
```

## Related pages

- [Offline GRPO](offline-grpo.md) · [Online GRPO (RLVR)](online-grpo.md) · [Async GRPO with Environments](async-grpo/README.md) · [Reward Terms](rewards.md)
- [Environments](environments/README.md) — the registry, tools, rewards and dataset formats
- [SMPO](../preference/smpo.md) · [DPO](../preference/dpo.md) — pairwise preference alternatives
- [Training Methods Overview](../README.md)
