# Laguna 2.1 Cookbook

Poolside's [Laguna S 2.1](https://huggingface.co/poolside/Laguna-S-2.1) and
[Laguna XS 2.1](https://huggingface.co/poolside/Laguna-XS-2.1) route to 256 experts plus a shared expert
through a sigmoid router with a correction bias. S has 48 layers and picks ten experts per token; XS has
40 layers and picks eight.

## Support

| FSDP | EP | CP | TP | ETP | EP+CP | EP+TP | LoRA | Online RL |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Yes | Yes | No | No | partial | No | No | Yes | vLLM |

`partial`: pure ETP and EP+ETP train, save and resume on a tiny Laguna in the GPU tests. The released
checkpoints have not run under ETP.

- **Checkpoints:** `poolside/Laguna-S-2.1` and `poolside/Laguna-XS-2.1`. The recipes load them through
  the hub's remote code at a pinned revision.
- **GPUs:** S runs at EP4 on four GPUs, 64 experts per GPU, or at EP8 on eight. XS runs on one GPU.

## Recipes

Run these in the [training container](README.md#start-the-training-container).

### Full fine-tune

Laguna S on four GPUs:

```bash
halo launch sft examples/sft/laguna/laguna-s-2.1-ultrachat-ep.yaml -n 4 \
  --output_dir=/data/checkpoints/laguna-s-2.1-sft
```

EP4 on eight GPUs is rejected at startup. On eight GPUs, launch with `-n 8` and add
`--expert_parallel_size=8`.

Laguna XS on one GPU, where the `flash_adamw` optimizer keeps the full fine-tune within one B300:

```bash
halo launch sft examples/sft/laguna/laguna-xs-2.1-ultrachat.yaml \
  --output_dir=/data/checkpoints/laguna-xs-2.1-sft
```

### Other layouts

- Pure ETP2 on Laguna S (`partial`, see above): add `--expert_parallel_size=1 --expert_tensor_parallel_size=2`
  to the four-GPU command.

### LoRA

```bash
halo launch sft examples/sft/laguna/laguna-s-2.1-ultrachat-ep.yaml -n 4 \
  --use_peft=true --learning_rate=1e-4 \
  --lora_target_modules=q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
  --output_dir=/data/checkpoints/laguna-s-2.1-lora
```

These targets adapt attention and the experts.

### GRPO

Read [Before a GRPO run](README.md#before-a-grpo-run) first. Laguna S has about 118B parameters, so the
trainer keeps the SFT recipe's shape: EP4 with `flash_adamw` on four GPUs. This setup serves the SFT
checkpoint from vLLM on GPUs 0–3 and trains on GPUs 4–7. On the host:

```bash
VLLM_MODEL=/data/checkpoints/laguna-s-2.1-sft VLLM_CUDA_DEVICES=0,1,2,3 VLLM_TP=4 VLLM_ENABLE_R3=1 \
VLLM_TOOL_PARSER=poolside_v1 \
  docker compose -f docker-compose.vllm.yml up -d vllm-server
```

In the training container, copy the template, set `dataset`, `environment_type` and `rewards`, then
launch:

```bash
cp examples/grpo/environmental/environmental-grpo-template.yaml laguna-grpo.yaml
CUDA_VISIBLE_DEVICES=4,5,6,7 halo launch environmental-grpo laguna-grpo.yaml -n 4 \
  --model_name_or_path=/data/checkpoints/laguna-s-2.1-sft \
  --trust_remote_code=true --attn_implementation=sdpa \
  --expert_parallel_size=4 --optim=flash_adamw \
  --routing_replay=rollout --beta=0.0 \
  --output_dir=/data/checkpoints/laguna-s-2.1-grpo
```

## Settings that matter

- **Attention.** Keep `attn_implementation: sdpa`. The pinned modeling code passes the sliding window
  only through the attention mask, which flash attention ignores, so a flash label would silently run
  the sliding-window layers as full attention. Halo does not switch to SDPA for you. SDPA rules out
  `padding_free`, so the recipes use `packing`.
- **Pad and EOS tokens.** `pad_token: "〈|PAD|〉"` and `eos_token: "〈|EOS|〉"` use the CJK angle
  brackets U+3008 and U+3009. Typing ASCII `<` and `>` instead adds new tokens without any error.
- **Remote code.** The recipes set `trust_remote_code: true` with a pinned `model_revision`, the
  revisions Halo validated against.
- **Router balancing.** `auto` picks `aux_loss`, but the released checkpoints set
  `router_aux_loss_coef: 0.0`, so the term stays off unless you set a coefficient in
  `model_init_kwargs`. `bias_update` trains the router's own correction bias, which ships in the
  checkpoint.
- **Tool parser.** vLLM needs `VLLM_TOOL_PARSER=poolside_v1`. The default `hermes` leaves Laguna's tool
  calls as text, so every episode scores zero.

## Limits

- No CP or TP: Laguna attention has no wrapper for either.
- No SGLang weight sync. SGLang 0.5.17's Laguna loader expects every expert weight in every update,
  which a chunked update can't provide. Online RL runs on vLLM.
- vLLM 0.26.0 drops the exported correction bias. A model trained with `bias_update` serves on the
  pretrained bias there.

## Export and serve

Gathered saves write the hub's per-expert layout. Smoke-test with `AutoModelForCausalLM`, passing
`trust_remote_code=True` to both `from_pretrained` calls ([snippet](README.md#smoke-test-a-checkpoint)).
Serve from the [host](README.md#serve-from-the-host) with vLLM on port 8000:

```bash
VLLM_MODEL=/data/checkpoints/laguna-s-2.1-sft VLLM_CUDA_DEVICES=0,1 VLLM_TP=2 \
VLLM_TOOL_PARSER=poolside_v1 \
  docker compose -f docker-compose.vllm.yml up -d vllm-server
```

## Reference

- [Laguna model notes](../../agent-docs/models/laguna.md) ↗: routing, the hub key spellings, why CP and
  TP are missing
- [Async GRPO with Environments](../training-methods/async-grpo-environments.md)
- Model cards: [Laguna S 2.1](https://huggingface.co/poolside/Laguna-S-2.1),
  [Laguna XS 2.1](https://huggingface.co/poolside/Laguna-XS-2.1)
