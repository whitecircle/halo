# Gemma 4 MoE Cookbook

[Gemma 4 26B-A4B](https://huggingface.co/google/gemma-4-26B-A4B-it) has 128 routed experts and picks
eight per token. The checkpoint takes text and images; the recipes train on text. Its router sits
outside the expert block, and its global attention layers use a 512-wide head that no flash kernel
supports.

## Support

| FSDP | EP | CP | TP | ETP | EP+CP | EP+TP | LoRA | Online RL |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Yes | Yes | No | No | Yes | No | No | Yes | vLLM, SGLang |

- **Checkpoint:** `google/gemma-4-26B-A4B-it`.
- **GPUs:** eight at EP8, 16 experts per GPU, at 32,768 tokens.

## Recipes

Run these in the [training container](README.md#start-the-training-container).

### Full fine-tune

```bash
halo launch sft examples/sft/gemma4/gemma4-26b-a4b-ultrachat-ep.yaml -n 8 \
  --output_dir=/data/checkpoints/gemma-4-26b-a4b-sft
```

### Other layouts

Add one of these to the full fine-tune command:

- Pure ETP8, to shard every expert instead of placing whole experts: `--expert_parallel_size=1 --expert_tensor_parallel_size=8`
- A 4-way expert split: `--expert_parallel_size=4 --expert_tensor_parallel_size=2`

### LoRA

```bash
halo launch sft examples/sft/gemma4/gemma4-26b-a4b-ultrachat-ep.yaml -n 8 \
  --use_peft=true --learning_rate=1e-4 \
  --lora_target_modules=q_proj,k_proj,v_proj,o_proj \
  --output_dir=/data/checkpoints/gemma-4-26b-a4b-lora
```

These targets adapt the language model only. The vision tower's projections share the names, but PEFT
can't adapt their wrapper class, so Halo skips them with a warning. `all-linear` adapts the tower too.

### GRPO

The shipped recipes train on code contests. Read [Before a GRPO run](README.md#before-a-grpo-run) first.
This one is a full fine-tune at EP4 with two vLLM servers on GPUs 4–7.

On the host, start the servers on the SFT checkpoint:

```bash
cp jinja-templates/gemma4/gemma4-reasoning-effort.jinja "$HALO_SCRATCH/"
export VLLM_MODEL=/data/checkpoints/gemma-4-26b-a4b-sft
export VLLM_CHAT_TEMPLATE=/data/gemma4-reasoning-effort.jinja
export VLLM_TOOL_PARSER=gemma4 VLLM_REASONING_PARSER=gemma4 VLLM_USE_V2_MODEL_RUNNER=0
VLLM_CUDA_DEVICES=4,5 VLLM_TP=2 VLLM_PORT=8000 \
  docker compose -p gemma4-rollout-0 -f docker-compose.vllm.yml up -d vllm-server
VLLM_CUDA_DEVICES=6,7 VLLM_TP=2 VLLM_PORT=8001 \
  docker compose -p gemma4-rollout-1 -f docker-compose.vllm.yml up -d vllm-server
```

In the training container, copy the recipe, set its `dataset`, and launch on GPUs 0–3:

```bash
cp examples/grpo/environmental/gemma4/vllm/gemma4-26b-a4b-code-contests-full-ep4.yaml gemma4-grpo.yaml
CUDA_VISIBLE_DEVICES=0,1,2,3 DIST_NCCL_TIMEOUT_MINUTES=60 \
  halo launch environmental-grpo gemma4-grpo.yaml -n 4 \
  --model_name_or_path=/data/checkpoints/gemma-4-26b-a4b-sft \
  --output_dir=/data/checkpoints/gemma-4-26b-a4b-grpo
```

The other configs in `examples/grpo/environmental/gemma4/vllm/` switch to LoRA or to ep1. The `sglang/`
configs (ep1) run on SGLang: serve them on ports 30000 and 30001 with `SGLANG_CHAT_TEMPLATE` on the same
file and `SGLANG_REASONING_PARSER=gemma4`.

## Settings that matter

- **Attention.** Keep `attn_implementation: sdpa`. Halo redirects any flash label to SDPA and builds the
  model with FlexAttention on the sliding-window layers and matmul attention on short global layers. At
  16,384 tokens, full SFT runs 3.9 times as fast as on plain SDPA (EP2 on two B300s).
- **Batch size.** Keep `per_device_train_batch_size: 1` and scale with `gradient_accumulation_steps`.
  Packing builds a dense mask over the whole batch, and its memory grows with the square of
  `batch size × max_length`.
- **Precision.** `fp32_experts: true` is the recipe default. `fp32_non_ep_params` is refused under EP,
  and `fp32_router` does nothing, because the router sits outside the EP wrapper.
- **No router balancing.** Keep `moe_balancing: none`. Gemma 4 has no auxiliary loss, and an explicit
  bias mode raises.
- **RL.** The environment GRPO recipes set `use_chunked_grpo_logprobs: true` for the 262k-token
  vocabulary. vLLM needs `VLLM_TOOL_PARSER=gemma4`; `hermes` leaves the tool calls as text, so every
  episode scores zero.

## Limits

- No CP or TP: Gemma 4 attention has no wrapper for either.
- `padding_free` is refused, since no variable-length kernel serves this attention. Use `packing`.
- No routing replay on either engine. Don't set `VLLM_ENABLE_R3` or `SGLANG_ENABLE_R3`.
- No router balancing of any kind.

## Export and serve

Gathered saves write the fused expert pair that both engines read. Smoke-test with
`AutoModelForImageTextToText` and `AutoProcessor`, which take images too
([snippet](README.md#smoke-test-a-checkpoint)). Serve from the [host](README.md#serve-from-the-host)
with SGLang on port 30000:

```bash
SGLANG_MODEL="$HALO_SCRATCH/checkpoints/gemma-4-26b-a4b-sft" \
SGLANG_CUDA_DEVICES=0,1,2,3 SGLANG_TP=4 \
  docker compose -f docker-compose.sglang.yml up -d sglang-server
```

SGLang weight sync needs this repo's SGLang image, which patches the Gemma 4 router.

## Reference

- [Gemma 4 model notes](../../agent-docs/models/gemma4.md) ↗: the attention path and its benchmarks, the
  precision flags, why CP and TP are missing
- [Async GRPO with Environments](../training-methods/async-grpo-environments.md)
- [Model card](https://huggingface.co/google/gemma-4-26B-A4B-it)
