# LFM-2 MoE Cookbook

Liquid AI's [LFM2.5-8B-A1B](https://huggingface.co/LiquidAI/LFM2.5-8B-A1B) and
[LFM2-24B-A2B](https://huggingface.co/LiquidAI/LFM2-24B-A2B) mix short-convolution layers with
full-attention layers. A sigmoid router with a selection bias picks four experts per token, out of 32
on the 8B and 64 on the 24B.

## Support

| FSDP | EP | CP | TP | ETP | EP+CP | EP+TP | LoRA | Online RL |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Yes | Yes | No | Yes | Yes | No | Yes | Yes | vLLM, SGLang |

- **Checkpoints:** `LiquidAI/LFM2.5-8B-A1B` and `LiquidAI/LFM2-24B-A2B`.
- **GPUs:** the 8B runs at EP2 on two GPUs, 16 experts per GPU. The 24B runs at EP4 on four GPUs or EP8 on
  eight.

## Recipes

Run these in the [training container](README.md#start-the-training-container).

### Full fine-tune

LFM2.5-8B-A1B on two GPUs:

```bash
halo launch sft examples/sft/lfm2/lfm2.5-8b-a1b-ultrachat-ep2.yaml -n 2 \
  --output_dir=/data/checkpoints/lfm2.5-8b-a1b-sft
```

The recipe sets `report_to: none`. To log to Weights & Biases, copy it and change that line to
`report_to: wandb`; this field can't be overridden on the command line.

LFM2-24B-A2B on four GPUs:

```bash
halo launch sft examples/sft/lfm2/lfm2.5-8b-a1b-ultrachat-ep2.yaml -n 4 \
  --model_name_or_path=LiquidAI/LFM2-24B-A2B --expert_parallel_size=4 \
  --output_dir=/data/checkpoints/lfm2-24b-a2b-sft
```

On eight GPUs, change these to `-n 8` and `--expert_parallel_size=8`. EP4 on eight GPUs is rejected at
startup.

### Other layouts

Add one of these to the two-GPU command:

- EP2 + TP2, when attention memory is the limit: `--tensor_parallel_size=2`. TP shards only the
  full-attention layers (6 of 24 on the 8B), so per-GPU memory drops much less than half.
- Pure ETP2, to shard every expert instead of placing whole experts: `--expert_parallel_size=1 --expert_tensor_parallel_size=2`

### LoRA

```bash
halo launch sft examples/sft/lfm2/lfm2.5-8b-a1b-ultrachat-ep2.yaml -n 2 \
  --use_peft=true --learning_rate=1e-4 \
  --lora_target_modules=q_proj,k_proj,v_proj,out_proj \
  --output_dir=/data/checkpoints/lfm2.5-8b-a1b-lora
```

`out_proj` is also the output projection of every short-convolution block (18 of 24 layers on the 8B),
so this adapter trains both kinds of layer.

### GRPO

Read [Before a GRPO run](README.md#before-a-grpo-run) first. This setup serves the SFT checkpoint from
vLLM on GPUs 0–1 and trains on GPUs 2–3. On the host:

```bash
VLLM_MODEL=/data/checkpoints/lfm2.5-8b-a1b-sft VLLM_CUDA_DEVICES=0,1 VLLM_TP=2 VLLM_ENABLE_R3=1 \
VLLM_TOOL_PARSER=lfm2 \
  docker compose -f docker-compose.vllm.yml up -d vllm-server
```

In the training container, copy the template, set `dataset`, `environment_type` and `rewards`, then
launch:

```bash
cp examples/grpo/environmental/environmental-grpo-template.yaml lfm2-grpo.yaml
CUDA_VISIBLE_DEVICES=2,3 halo launch environmental-grpo lfm2-grpo.yaml -n 2 \
  --model_name_or_path=/data/checkpoints/lfm2.5-8b-a1b-sft \
  --routing_replay=rollout --beta=0.0 \
  --output_dir=/data/checkpoints/lfm2.5-8b-a1b-grpo
```

Add `--expert_parallel_size=2` to shard the experts across the two trainer GPUs. SGLang 0.5.17 also
serves and weight-syncs LFM-2: add `--rollout_backend=sglang --rollout_server_url=http://localhost:30000`
and leave out `--routing_replay=rollout`, because SGLang's routing capture is not verified for LFM-2.

## Settings that matter

- **Router balancing.** The recipe sets `moe_balancing: bias_update`, which `auto` also picks under EP.
  Sigmoid routing can collapse onto a few experts during SFT. The updates land in the router's own
  `expert_bias`, so the trained bias ships in every checkpoint and the served model routes as it
  trained. LFM-2 has no router auxiliary loss. Without EP or grouped GEMM nothing carries the bias, so
  freeze the router instead with `freeze_layers_patterns: ["*.feed_forward.gate.weight"]`.
- **Tool parser.** vLLM needs `VLLM_TOOL_PARSER=lfm2`. The default `hermes` leaves LFM-2's tool calls as
  text, so every episode scores zero.

## Limits

- No CP: the short-convolution layers can't split the sequence across ranks.

## Export and serve

Gathered saves write the hub's per-expert layout (`experts.{i}.w1/w3/w2`), which both engines read.
Smoke-test with `AutoModelForCausalLM` ([snippet](README.md#smoke-test-a-checkpoint)). Serve from the
[host](README.md#serve-from-the-host) with SGLang on port 30000:

```bash
SGLANG_MODEL="$HALO_SCRATCH/checkpoints/lfm2.5-8b-a1b-sft" \
  docker compose -f docker-compose.sglang.yml up -d
```

## Reference

- [LFM-2 model notes](../../agent-docs/models/lfm2.md) ↗: the EP wrapper, packing across the conv
  layers, balancing
- [Async GRPO with Environments](../training-methods/async-grpo-environments.md)
- Model cards: [LFM2.5-8B-A1B](https://huggingface.co/LiquidAI/LFM2.5-8B-A1B),
  [LFM2-24B-A2B](https://huggingface.co/LiquidAI/LFM2-24B-A2B)
