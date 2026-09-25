# Halo / Qwen3 MoE cookbook

Fine-tune [Qwen3 30B A3B Instruct 2507](https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507) with Halo.

The model has 128 routed experts and selects eight experts for each token. Halo can distribute the experts with EP, shard each expert with ETP, and shard attention with TP.

## Halo support

| FSDP | EP | CP | TP | ETP | EP+CP | EP+TP | LoRA |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes |

Halo uses DeepEP for token dispatch and grouped GEMM for the expert projections. Qwen3 MoE also supports Ulysses CP for long sequences.

This recipe starts with four NVIDIA B300 GPUs. EP4 places 32 experts on each GPU.

## Start the training container

Start the [cookbook container](README.md#start-the-training-container) and run the commands
below inside it, except the server commands marked for the host.

## Train all weights with EP4

Create `qwen3-moe-sft.yaml`.

```yaml
model_name_or_path: Qwen/Qwen3-30B-A3B-Instruct-2507
model_revision: 0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe
model_init_kwargs:
  output_router_logits: true
  router_aux_loss_coef: 0.001
moe_balancing: aux_loss

dataset:
- HuggingFaceH4/ultrachat_200k@train_sft
conversation_field: messages
test_size: 0.01
train_on_completions_only: true
assistant_message_template: "<|im_start|>assistant\n"
pad_token: <|endoftext|>
eos_token: <|im_end|>

expert_parallel_size: 4
save_sharded_ep: false
use_grouped_gemm: true
fp32_router: true
fp32_experts: false

use_liger_kernel: true
packing: true
max_length: 8192
bf16: true

per_device_train_batch_size: 1
per_device_eval_batch_size: 1
gradient_accumulation_steps: 8
num_train_epochs: 1.0
gradient_checkpointing: true

optim: adamw_torch_fused
learning_rate: 5.0e-06
lr_scheduler_type: cosine
warmup_steps: 32
max_grad_norm: 1.0

save_strategy: steps
save_steps: 1000
eval_strategy: steps
eval_steps: 300
save_total_limit: 1
save_only_model: true
output_dir: /data/checkpoints/qwen3-30b-a3b-ultrachat-ep4

logging_steps: 1
logging_first_step: true
report_to: wandb
remove_unused_columns: false
dataloader_num_workers: 2

use_peft: false
```

Launch four processes.

```bash
halo launch sft qwen3-moe-sft.yaml -n 4
```

Halo gathers the expert weights when it saves because `save_sharded_ep` is false.

Leave `attn_implementation` unset — Halo auto-selects FA4 on Blackwell and FA3 (FA2 if
FA3 is absent) on Hopper.

## Add CP, TP, or ETP

Every layout below stays on the same four ranks, where EP4 is one dispatch group. The
same config on eight GPUs is rejected at config time
([rules](../parallelism.md#rules-that-save-you-a-wasted-run)).

Use CP for longer sequences. EP4 and CP2 use the same four ranks. Disable packing, since
the collator rejects it when CP splits the sequence.

```yaml
expert_parallel_size: 4
context_parallel_size: 2
packing: false
```

```bash
halo launch sft qwen3-moe-sft.yaml -n 4
```

Use TP when attention memory is the limit. EP4 and TP2 also use the same four ranks.

```yaml
expert_parallel_size: 4
tensor_parallel_size: 2
```

```bash
halo launch sft qwen3-moe-sft.yaml -n 4
```

Use pure ETP when each expert needs more sharding.

```yaml
expert_parallel_size: 1
expert_tensor_parallel_size: 4
```

Do not enable TP and ETP together. Do not combine LoRA with TP.

## Run inference

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

path = "/data/checkpoints/qwen3-30b-a3b-ultrachat-ep4"
tokenizer = AutoTokenizer.from_pretrained(path)
model = AutoModelForCausalLM.from_pretrained(
    path,
    dtype=torch.bfloat16,
    device_map="auto",
)

messages = [{"role": "user", "content": "Explain expert parallelism in five sentences."}]
inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt",
).to(model.device)

output = model.generate(**inputs, max_new_tokens=256, do_sample=True, temperature=0.2)
print(tokenizer.decode(output[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True))
```

Serve the gathered checkpoint with SGLang on port 30000, from the host
([server setup](README.md#serve-from-the-host)).

```bash
SGLANG_MODEL="$HALO_SCRATCH/checkpoints/qwen3-30b-a3b-ultrachat-ep4" \
  docker compose -f docker-compose.sglang.yml up
```

## Train a LoRA adapter

Add this block to the SFT configuration.

```yaml
use_peft: true
lora_r: 16
lora_alpha: 32
lora_dropout: 0.05
lora_target_modules:
- q_proj
- k_proj
- v_proj
- o_proj

learning_rate: 1.0e-04
output_dir: /data/checkpoints/qwen3-30b-a3b-ultrachat-lora
```

Keep EP enabled if the base model needs expert sharding. Keep TP disabled for LoRA.

## Continue with GRPO

Copy `examples/grpo/environmental/environmental-grpo-template.yaml` to
`qwen3-moe-grpo.yaml`, set `model_name_or_path` to the gathered SFT checkpoint's `/data`
path and the environment and reward fields for your task, and add:

```yaml
rollout_backend: vllm
rollout_server_url: http://localhost:8000
train_on_sampled_tokens: true
routing_replay: rollout
beta: 0.0
output_dir: /data/checkpoints/qwen3-30b-a3b-grpo
```

Rollouts run on vLLM (the config default). SGLang 0.5.17 also serves and weight-syncs
Qwen3 MoE (`rollout_backend: sglang`, port 30000), with expert distribution.

Start the server on the host ([server setup](README.md#serve-from-the-host)), on GPUs the
trainer will not use:

```bash
VLLM_MODEL=/data/checkpoints/qwen3-30b-a3b-ultrachat-ep4 \
VLLM_CUDA_DEVICES=0,1 VLLM_TP=2 VLLM_ENABLE_R3=1 \
  docker compose -f docker-compose.vllm.yml up vllm-server
```

Launch the trainer in the training container on the remaining GPUs; they cannot share
one. `expert_parallel_size` may match the trainer's GPU count.

```bash
CUDA_VISIBLE_DEVICES=2,3 halo launch environmental-grpo qwen3-moe-grpo.yaml -n 2
```

Full setup:
[Async GRPO with Environments](../../agent-docs/training-methods/grpo/async-grpo/README.md) ↗.
