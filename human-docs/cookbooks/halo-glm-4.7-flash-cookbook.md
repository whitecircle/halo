# GLM-4.7-Flash Cookbook

[GLM-4.7-Flash](https://huggingface.co/zai-org/GLM-4.7-Flash) is a GLM-4 MoE Lite model: 64 routed
experts plus a shared expert, four picked per token by a sigmoid router. Its attention compresses
queries and keys through low-rank projections (MLA).

## Support

| FSDP | EP | CP | TP | ETP | EP+CP | EP+TP | LoRA | Online RL |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes | vLLM, SGLang |

- **Checkpoint:** `zai-org/GLM-4.7-Flash`.
- **GPUs:** eight at EP8, eight experts per GPU, at 30,720 tokens.

## Recipes

Run these in the [training container](README.md#start-the-training-container).

### Full fine-tune

```bash
halo launch sft examples/sft/glm4/glm-4.7-flash-ultrachat-ep.yaml -n 8 \
  --output_dir=/data/checkpoints/glm-4.7-flash-sft
```

### Other layouts

Add one of these to the full fine-tune command:

- EP8 + CP2, for long sequences: `--context_parallel_size=2 --packing=false`. EP+CP needs EP to fill
  the node, so EP8 is the only EP size that pairs with CP on eight GPUs.
- EP8 + TP2, when attention memory is the limit: `--tensor_parallel_size=2`
- Pure ETP8, to shard every expert instead of placing whole experts: `--expert_parallel_size=1 --expert_tensor_parallel_size=8`
- A 4-way expert split: `--expert_parallel_size=4 --expert_tensor_parallel_size=2`

### LoRA

```bash
halo launch sft examples/sft/glm4/glm-4.7-flash-ultrachat-ep.yaml -n 8 \
  --use_peft=true --learning_rate=1e-4 \
  --lora_target_modules=q_a_proj,q_b_proj,kv_a_proj_with_mqa,kv_b_proj,o_proj \
  --output_dir=/data/checkpoints/glm-4.7-flash-lora
```

### GRPO

Read [Before a GRPO run](README.md#before-a-grpo-run) first. This setup serves the SFT checkpoint from
vLLM on GPUs 0–3 and trains on GPUs 4–7.

RL uses `jinja-templates/glm/glm-native.jinja`, the upstream template with tool calls, because the SFT
template renders no tools. The server must serve the same file. On the host:

```bash
cp jinja-templates/glm/glm-native.jinja "$HALO_SCRATCH/"
VLLM_MODEL=/data/checkpoints/glm-4.7-flash-sft VLLM_CHAT_TEMPLATE=/data/glm-native.jinja \
VLLM_CUDA_DEVICES=0,1,2,3 VLLM_TP=4 VLLM_ENABLE_R3=1 \
VLLM_TOOL_PARSER=glm47 VLLM_ATTENTION_BACKEND=CUTLASS_MLA \
  docker compose -f docker-compose.vllm.yml up -d vllm-server
```

In the training container, copy the template and set `dataset`, `environment_type` and `rewards`. The
template's generation prompt opens `<think>`, so rollouts start in thinking mode while SFT trained the
non-thinking render. To keep rollouts non-thinking, also add
`rollout_chat_template_kwargs: {enable_thinking: false}` to the copy. Then launch:

```bash
cp examples/grpo/environmental/environmental-grpo-template.yaml glm47-grpo.yaml
CUDA_VISIBLE_DEVICES=4,5,6,7 halo launch environmental-grpo glm47-grpo.yaml -n 4 \
  --model_name_or_path=/data/checkpoints/glm-4.7-flash-sft \
  --chat_template=jinja-templates/glm/glm-native.jinja --force_chat_template=true \
  --routing_replay=rollout --beta=0.0 \
  --output_dir=/data/checkpoints/glm-4.7-flash-grpo
```

Add `--expert_parallel_size=4` to shard the experts across the four trainer GPUs. For SGLang, start the
server with `SGLANG_ATTENTION_BACKEND=triton`, `SGLANG_CHAT_TEMPLATE` on the same file and
`SGLANG_ENABLE_R3=1`, and add `--rollout_backend=sglang --rollout_server_url=http://localhost:30000` to
the trainer.

## Settings that matter

- **Attention.** The recipe pins `flash_attention_2`. FA4's backward produces NaN gradients on GLM's
  attention shape, so Halo replaces an FA4 pick with SDPA. CP refuses an SDPA label, so keep FA2 under
  CP.
- **Router balancing.** The recipe sets `moe_balancing: bias_update`, which `auto` also picks. It updates
  the router's own `e_score_correction_bias`, so the trained bias ships in every checkpoint and the
  served model routes as it trained. The router has no auxiliary loss to use instead.
- **Chat templates.** SFT uses `glm-chat.jinja` for multi-turn data. `glm-instruct.jinja` raises on more
  than one assistant turn. Keep `assistant_message_template: "<|assistant|>"`: the role token alone
  matches both thinking and non-thinking turns.
- **Tool parser.** vLLM needs `VLLM_TOOL_PARSER=glm47`. The default `hermes` can't read GLM tool calls,
  so every episode scores zero.

## Limits

- On Blackwell, both engines need an MLA attention backend that accepts GLM's head size:
  `VLLM_ATTENTION_BACKEND=CUTLASS_MLA` or `SGLANG_ATTENTION_BACKEND=triton`. With SGLang's default the
  server exits at startup.
- SGLang weight sync needs this repo's SGLang image. It patches the GLM-4 router, which upstream caches
  on first use.

## Export and serve

Gathered saves write the hub's per-expert layout, which both engines read. A checkpoint that skipped the
EP save, such as a `merge-peft-adapters` merge over a plain load, stays fused and needs
`halo run unfuse-moe-experts` before serving.

Smoke-test with `AutoModelForCausalLM` ([snippet](README.md#smoke-test-a-checkpoint)). Serve from the
[host](README.md#serve-from-the-host) with SGLang on port 30000:

```bash
SGLANG_MODEL="$HALO_SCRATCH/checkpoints/glm-4.7-flash-sft" SGLANG_ATTENTION_BACKEND=triton \
  docker compose -f docker-compose.sglang.yml up -d
```

## Reference

- [GLM-4 model notes](../../agent-docs/models/glm4.md) ↗: the CP and TP plans for MLA, chat templates,
  serving
- [Async GRPO with Environments](../training-methods/async-grpo-environments.md)
- [Model card](https://huggingface.co/zai-org/GLM-4.7-Flash)
