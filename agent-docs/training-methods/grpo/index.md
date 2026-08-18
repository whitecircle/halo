# GRPO Family: Group Relative Policy Optimization

GRPO computes advantages from groups of completions per prompt rather than pairwise comparisons: for prompt `x` with completions `{y1..yk}` and rewards `{r1..rk}`, the advantage is `A(yi) = normalize(ri - mean(r))` and the loss is `L = -E[A(yi) * log pi(yi|x)]`. Three variants ship; the [comparison page](grpo-comparison.md) is the side-by-side matrix.

**[Offline GRPO](offline-grpo.md)** (`OfflineGRPOTrainer`) — you already have a dataset with multiple pre-scored completions per prompt. No generation during training, so the cheapest variant.

```bash
torchrun --nproc_per_node=8 scripts/training/offline_grpo.py \
    examples/grpo/offline/qwen3_5/offline-grpo-qwen3.6-35b-a3b-gsm8k.yaml --expert_parallel_size=8
```

**[RLVR online GRPO](online-grpo.md)** (`DistributedGRPOTrainer`) — math reasoning and other deterministically verifiable answers; rewards computed locally with no network calls.

```bash
docker compose -f docker-compose.vllm.yml up vllm-server   # vLLM on GPU 0
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 torchrun --nproc_per_node=7 \
    scripts/training/online_grpo/rlvr.py examples/grpo/online/rlvr-online-grpo-template.yaml
```

**[Environmental GRPO](environmental-grpo.md)** (`DistributedAsyncEnvironmentalGRPOTrainer`) — multi-turn tool-use environments (code execution, search, file ops) before a final reward.

```bash
docker compose -f docker-compose.vllm.yml up vllm-server   # vLLM on GPU 0
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 torchrun --nproc_per_node=7 \
    scripts/training/environmental_grpo.py \
    examples/grpo/environmental/environmental-grpo-template.yaml --expert_parallel_size=7
```

## Related pages

- [SMPO](../preference/smpo.md) — pairwise preference alternative
- [DPO](../preference/dpo.md) — KL-constrained preference optimization
- [Training Methods Overview](../index.md)
