# Halo / Qwen3.5 and Qwen3.6 MoE cookbook

Fine-tune [Qwen3.6 35B A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) with Halo.

The same recipe covers Qwen3.5 35B A3B. Both checkpoints use the Qwen3.5 MoE model family in Transformers. They have 256 routed experts, select eight experts for each token, and combine full-attention layers with GatedDeltaNet linear-attention layers.

## Halo support

| FSDP | EP | CP | TP | ETP | EP+CP | EP+TP | LoRA |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Yes | Yes | No | Yes | Yes | No | Yes | Yes |

CP is not supported because the recurrent linear-attention layers cannot use a Ulysses sequence split. TP is limited to two GPUs because the released checkpoints have two KV heads.

| Checkpoint | Revision | Experts | Active experts |
|---|---|---:|---:|
| `Qwen/Qwen3.5-35B-A3B` | `59d61f3ce65a6d9863b86d2e96597125219dc754` | 256 | 8 |
| `Qwen/Qwen3.6-35B-A3B` | `995ad96eacd98c81ed38be0c5b274b04031597b0` | 256 | 8 |

Qwen has not published an official Qwen3.7 checkpoint. Add it only after its model class and checkpoint format are verified.

This recipe starts with eight NVIDIA B300 GPUs; EP8 places 32 experts on each GPU.

## Start the training container

```bash
git clone --recurse-submodules https://github.com/whitecircle/halo
cd halo
docker pull public.ecr.aws/whitecircle/halo:blackwell
```

Export `HF_TOKEN` and `WANDB_API_KEY` in the host shell.

```bash
mkdir -p /mnt/hf /mnt/checkpoints /mnt/tmp
docker run --rm -it \
  --name halo-qwen36 \
  --gpus all \
  --network host \
  --ipc=host \
  --shm-size=128g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -e HF_TOKEN \
  -e WANDB_API_KEY \
  -e HF_HOME=/mnt/hf \
  -e HF_DATASETS_CACHE=/mnt/hf/datasets \
  -e TMPDIR=/mnt/tmp \
  -e PYTHONPATH=/workspace \
  -e CUDA_DEVICE_MAX_CONNECTIONS=1 \
  -v "$(pwd)":/workspace \
  -v /mnt/hf:/mnt/hf \
  -v /mnt/tmp:/mnt/tmp \
  -v /mnt/checkpoints:/mnt/checkpoints \
  -w /workspace \
  public.ecr.aws/whitecircle/halo:blackwell bash
```

Run all remaining commands inside this container.

## Train all weights with EP8

Create `qwen3.6-sft.yaml`.

```yaml
model_name_or_path: Qwen/Qwen3.6-35B-A3B
model_revision: 995ad96eacd98c81ed38be0c5b274b04031597b0
moe_balancing: bias_update_transient

dataset:
- HuggingFaceH4/ultrachat_200k@train_sft
conversation_field: messages
test_size: 0.01
train_on_completions_only: true
assistant_message_template: "<|im_start|>assistant\n"
pad_token: <|endoftext|>
eos_token: <|im_end|>
chat_template: jinja_templates/qwen3-multiturn.jinja
force_chat_template: true

expert_parallel_size: 8
save_sharded_ep: false
use_grouped_gemm: true
fp32_router: true
fp32_experts: true
fp32_non_ep_params: true
fp32_output_conversion: false

attn_implementation: flash_attention_2
use_liger_kernel: true
packing: true
padding_free: false
max_length: 33000
bf16: true

per_device_train_batch_size: 1
per_device_eval_batch_size: 1
gradient_accumulation_steps: 8
num_train_epochs: 1.0
gradient_checkpointing: true
gradient_checkpointing_kwargs:
  use_reentrant: false

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
output_dir: /mnt/checkpoints/qwen3.6-35b-a3b-ultrachat-ep8

logging_steps: 1
logging_first_step: true
report_to: wandb
remove_unused_columns: false
dataloader_num_workers: 2

use_peft: false
```

Launch eight processes.

```bash
halo launch sft qwen3.6-sft.yaml -n 8
```

Keep Flash Attention 2. Flash Attention 4's backward emits NaN gradients on this
architecture, so the loader demotes an FA4 selection to SDPA for the `qwen3_5*` model
types. Keep `padding_free: false` — the multimodal RoPE crashes on the varlen path.

`moe_balancing: bias_update_transient` is the working choice here, and what the
shipped config sets. `aux_loss` is refused: the multimodal forward reads
`output_router_logits` from kwargs, never from the config, so the coefficient
would never reach the loss. The `_transient` spelling is a deliberate trade-off.
The architecture has no exportable bias slot, so the bias balances routing during
training but every exported checkpoint serves without it, and near-tied top-k
picks can flip between trainer and server. Plain `bias_update` raises here for
exactly that reason.

To train Qwen3.5, replace the model name, revision, and output directory with the values in the checkpoint table.

## Add TP or ETP

On one eight-GPU node `expert_parallel_size` must be 8, 2, or 1 — an intermediate size
such as 4 forms two four-rank DeepEP dispatch groups whose combine barriers race FSDP2,
and [`ParallelismConfig`](../parallelism.md) rejects it at config time. Attention TP does
not change that: it leaves the dispatch-group width alone.

Use EP8 with TP2 when attention memory is the limit.

```yaml
expert_parallel_size: 8
tensor_parallel_size: 2
```

TP cannot exceed two for these checkpoints.

Use pure ETP to shard every expert across GPUs.

```yaml
expert_parallel_size: 1
expert_tensor_parallel_size: 8
```

Do not enable CP. Do not enable TP and ETP together.

## Run inference

```python
import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer

path = "/mnt/checkpoints/qwen3.6-35b-a3b-ultrachat-ep8"
tokenizer = AutoTokenizer.from_pretrained(path)
model = AutoModelForImageTextToText.from_pretrained(
    path,
    dtype=torch.bfloat16,
    device_map="auto",
)

messages = [{"role": "user", "content": "Write a short plan to investigate a training loss spike."}]
inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt",
).to(model.device)
output = model.generate(**inputs, max_new_tokens=512, do_sample=True, temperature=0.2)
print(tokenizer.decode(output[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True))
```

Serve the gathered checkpoint with Halo's vLLM image on the host, not inside the
training container; it listens on port 8000, and vLLM 0.26.0's expert loader reads
the gathered save's fused layout directly. SGLang 0.5.17 cannot load it: outside
GPT-OSS its loaders want per-expert expert names, and this family's gather emits the
fused pair. The compose service mounts only the HuggingFace cache, so add
`- /mnt/checkpoints:/mnt/checkpoints:ro` under the `vllm-server` `volumes:` to serve
a checkpoint from disk.

```bash
docker pull public.ecr.aws/whitecircle/halo:vllm-0.26.0
docker tag public.ecr.aws/whitecircle/halo:vllm-0.26.0 vllm-server:0.26.0

VLLM_MODEL=/mnt/checkpoints/qwen3.6-35b-a3b-ultrachat-ep8 \
VLLM_CUDA_DEVICES=0,1,2,3 VLLM_TP=4 \
VLLM_TOOL_PARSER=qwen3_xml \
  docker compose -f docker-compose.vllm.yml up vllm-server
```

## Train a LoRA adapter

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
output_dir: /mnt/checkpoints/qwen3.6-35b-a3b-ultrachat-lora
```

Keep TP disabled for LoRA.

## Continue with GRPO

Start from one of the shipped configs under `examples/grpo/environmental/qwen3_5/vllm/`, or
from `examples/grpo/environmental/environmental-grpo-template.yaml`. Replace the model
path with the gathered SFT checkpoint.

SGLang can weight-sync only GPT-OSS among the MoE families, so Qwen3.5/3.6 rollouts run
on vLLM (`rollout_backend: vllm`, the config default). Start the server on separate GPUs.

Run the server on the host, not inside the training container; the commands below retag
the pulled image to the name the compose file expects. Its service mounts only the
HuggingFace cache, so add `- /mnt/checkpoints:/mnt/checkpoints:ro` under the
`vllm-server` `volumes:` to serve a checkpoint from disk.

```bash
docker pull public.ecr.aws/whitecircle/halo:vllm-0.26.0
docker tag public.ecr.aws/whitecircle/halo:vllm-0.26.0 vllm-server:0.26.0

VLLM_MODEL=/mnt/checkpoints/qwen3.6-35b-a3b-ultrachat-ep8 \
VLLM_CUDA_DEVICES=0,1,2,3 VLLM_TP=4 \
VLLM_TOOL_PARSER=qwen3_xml \
VLLM_REASONING_PARSER=qwen3 \
  docker compose -f docker-compose.vllm.yml up vllm-server
```

That command already passes `--moe-backend triton`, which is required: Blackwell's
auto-selected MoE backends repack expert weights at load and silently corrupt every
weight sync. `VLLM_TOOL_PARSER=qwen3_xml` matters as much for tool-using
environments. The `hermes` default cannot parse this family's XML tool calls, so calls
stay plain text, every episode ends unsolved, and training runs on a flat zero
gradient. The shipped configs set `rollout_max_thinking_tokens`, which needs the
reasoning parser above **and** `VLLM_USE_V2_MODEL_RUNNER=0` in the server
environment; Model Runner V2 rejects thinking budgets with a 400 on every request.
To serve `routing_replay: rollout`, also add `--enable-return-routed-experts` to the
server's `command:` block — the compose file exposes no variable for it.

```yaml
rollout_server_url: http://localhost:8000
train_on_sampled_tokens: true
routing_replay: rollout
```

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 halo launch environmental-grpo qwen3.6-grpo.yaml -n 4
```

`CUDA_VISIBLE_DEVICES` fences the trainer off the server — they cannot share a GPU.
Size `expert_parallel_size` to the trainer's GPU count, not the node's: the SFT value
assumes the whole node. Full setup:
[Environmental GRPO](../../agent-docs/training-methods/grpo/environmental-grpo.md) ↗.
