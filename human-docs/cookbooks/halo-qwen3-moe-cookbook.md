# Qwen3 MoE Cookbook

[Qwen3-30B-A3B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507) has 128 routed
experts and picks eight per token. Qwen3 MoE has the broadest parallelism coverage in Halo, which makes it
the reference MoE family.

## Support

| FSDP | EP | CP | TP | ETP | EP+CP | EP+TP | LoRA | Online RL |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes | vLLM, SGLang |

- **Checkpoint:** `Qwen/Qwen3-30B-A3B-Instruct-2507`, pinned below. The larger
  `Qwen/Qwen3-235B-A22B-Instruct-2507` trains at `--expert_parallel_size=8`.
- **GPUs:** four at EP4, 32 experts per GPU.

## Recipes

Run these in the [training container](README.md#start-the-training-container).

### Full fine-tune

No Qwen3 MoE SFT recipe ships, so save this as `qwen3-moe-sft.yaml` in the repo root:

```yaml
model_name_or_path: Qwen/Qwen3-30B-A3B-Instruct-2507
model_revision: 0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe
moe_balancing: aux_loss

dataset:
- HuggingFaceH4/ultrachat_200k@train_sft
conversation_field: messages
test_size: 0.01
assistant_message_template: "<|im_start|>assistant\n"
pad_token: <|endoftext|>
eos_token: <|im_end|>

expert_parallel_size: 4
fp32_router: true

packing: true
max_length: 8192
per_device_train_batch_size: 1
per_device_eval_batch_size: 1
gradient_accumulation_steps: 8
gradient_checkpointing: true

optim: adamw_torch_fused
learning_rate: 5.0e-06
lr_scheduler_type: cosine
warmup_steps: 32
num_train_epochs: 1.0

save_strategy: steps
save_steps: 1000
save_total_limit: 1
save_only_model: true
eval_strategy: steps
eval_steps: 300
logging_steps: 1
report_to: wandb
remove_unused_columns: false
output_dir: /data/checkpoints/qwen3-30b-a3b-sft
```

```bash
halo launch sft qwen3-moe-sft.yaml -n 4
```

### Other layouts

Each of these runs on the same four GPUs. Add one to the launch command:

- EP4 + CP2, for long sequences: `--context_parallel_size=2 --packing=false`
- EP4 + TP2, when attention memory is the limit: `--tensor_parallel_size=2`
- Pure ETP4, to shard every expert instead of placing whole experts: `--expert_parallel_size=1 --expert_tensor_parallel_size=4`

On eight GPUs, launch with `-n 8 --expert_parallel_size=8`. EP4 on eight GPUs is rejected at startup.

### LoRA

```bash
halo launch sft qwen3-moe-sft.yaml -n 4 \
  --use_peft=true --learning_rate=1e-4 \
  --lora_target_modules=q_proj,k_proj,v_proj,o_proj \
  --output_dir=/data/checkpoints/qwen3-30b-a3b-lora
```

Add `gate_proj,up_proj,down_proj` to adapt the experts as well. Under EP they train as grouped expert
adapters.

### GRPO

Read [Before a GRPO run](README.md#before-a-grpo-run) first. This setup serves the SFT checkpoint from
vLLM on GPUs 0–1 and trains on GPUs 2–3.

On the host:

```bash
VLLM_MODEL=/data/checkpoints/qwen3-30b-a3b-sft VLLM_CUDA_DEVICES=0,1 VLLM_TP=2 VLLM_ENABLE_R3=1 \
  docker compose -f docker-compose.vllm.yml up -d vllm-server
```

In the training container, copy the template, set `dataset`, `environment_type` and `rewards`, then
launch:

```bash
cp examples/grpo/environmental/environmental-grpo-template.yaml qwen3-moe-grpo.yaml
CUDA_VISIBLE_DEVICES=2,3 halo launch environmental-grpo qwen3-moe-grpo.yaml -n 2 \
  --model_name_or_path=/data/checkpoints/qwen3-30b-a3b-sft \
  --routing_replay=rollout --beta=0.0 \
  --output_dir=/data/checkpoints/qwen3-30b-a3b-grpo
```

Add `--expert_parallel_size=2` to shard the experts across the two trainer GPUs. On SGLang, add
`--rollout_backend=sglang --rollout_server_url=http://localhost:30000` and start the server with
`SGLANG_ENABLE_R3=1`.

## Settings that matter

- **Attention.** Leave `attn_implementation` unset. Halo picks FA4 on Blackwell and FA3 (or FA2) on
  Hopper.
- **Router balancing.** `aux_loss` uses the checkpoint's `router_aux_loss_coef` (0.001). The router has no
  bias slot, so `bias_update` raises. `bias_update_transient` balances training only, and exported
  checkpoints serve without the bias.
- **Tool parser.** Qwen3 tool calls parse with vLLM's default `hermes` parser.

## Limits

Qwen3 MoE has no family-specific limits. The general ones apply: LoRA is refused under TP and EP+TP,
and expert adapters are refused once `expert_tensor_parallel_size > 1`.

## Export and serve

The gathered save writes the hub's per-expert layout, which transformers, vLLM and SGLang read
directly. Smoke-test it with `AutoModelForCausalLM` ([snippet](README.md#smoke-test-a-checkpoint)).
Serve it from the [host](README.md#serve-from-the-host) with SGLang on port 30000:

```bash
SGLANG_MODEL="$HALO_SCRATCH/checkpoints/qwen3-30b-a3b-sft" \
  docker compose -f docker-compose.sglang.yml up -d
```

## Reference

- [Qwen3 model notes](../../agent-docs/models/qwen3.md) ↗: the EP, CP and TP wrappers, balancing
- [Async GRPO with Environments](../training-methods/async-grpo-environments.md)
- [Model card](https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507)
