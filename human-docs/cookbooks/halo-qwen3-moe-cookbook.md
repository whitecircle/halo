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

```bash
git clone --recurse-submodules https://github.com/whitecircle/halo
cd halo
docker pull public.ecr.aws/whitecircle/halo:blackwell
```

Export `HF_TOKEN` and `WANDB_API_KEY` in the host shell.

```bash
mkdir -p /mnt/hf /mnt/checkpoints /mnt/tmp
docker run --rm -it \
  --name halo-qwen3-moe \
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
output_dir: /mnt/checkpoints/qwen3-30b-a3b-ultrachat-ep4

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

Every layout below stays on the same four ranks. Size EP to the whole job (`ep4` here) or
to 2. An intermediate size such as `ep4` on an eight-GPU node forms two four-rank DeepEP
dispatch groups whose combine barriers race FSDP2, and
[`ParallelismConfig`](../parallelism.md) rejects it at config time. EP+CP narrows that
further: the EP group has to fill the NVLink domain.

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

path = "/mnt/checkpoints/qwen3-30b-a3b-ultrachat-ep4"
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

Serve the gathered checkpoint with Halo's SGLang image, which listens on port 30000. Run
it on the host, not inside the training container: pull the prebuilt image and point
`SGLANG_IMAGE` at it (no retag needed), or build the compose file's local tag once with
`make build-sglang`.

```bash
docker pull public.ecr.aws/whitecircle/halo:sglang-0.5.17

SGLANG_IMAGE=public.ecr.aws/whitecircle/halo:sglang-0.5.17 \
SGLANG_MODEL=/mnt/checkpoints/qwen3-30b-a3b-ultrachat-ep4 \
SGLANG_MODEL_DIR=/mnt/checkpoints \
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
output_dir: /mnt/checkpoints/qwen3-30b-a3b-ultrachat-lora
```

Keep EP enabled if the base model needs expert sharding. Keep TP disabled for LoRA.

## Continue with GRPO

Use `examples/grpo/environmental/environmental-grpo-template.yaml` as the starting point.
Set `model_name_or_path` to the gathered SFT checkpoint.

Rollouts run on vLLM (`rollout_backend: vllm`, the config default). SGLang is
refused at construction for Qwen3 MoE: its 0.5.17 loader maps per-expert names
only and silently drops the fused expert keys weight sync would ship, so the
served policy would never update, and the gate fails the run before the engine
is touched. vLLM has no family or expert-distribution restriction here and keeps
the trainer's NVLink.

Run the server on the host, not inside the training container, on GPUs the
trainer will not use. Pull the prebuilt server image, retag it to the name the
compose file expects, and add `- /mnt/checkpoints:/mnt/checkpoints:ro` under the
`vllm-server` `volumes:`, since its service mounts only the HuggingFace cache.

```bash
docker pull public.ecr.aws/whitecircle/halo:vllm-0.26.0
docker tag public.ecr.aws/whitecircle/halo:vllm-0.26.0 vllm-server:0.26.0

VLLM_MODEL=/mnt/checkpoints/qwen3-30b-a3b-ultrachat-ep4 \
VLLM_CUDA_DEVICES=0,1 VLLM_TP=2 \
  docker compose -f docker-compose.vllm.yml up vllm-server
```

That command already passes the required `--moe-backend triton`; Blackwell's
auto-selected MoE backends repack expert weights at load and silently corrupt
every weight sync. To serve `routing_replay: rollout`, also add
`--enable-return-routed-experts` to the server's `command:` block, since the
compose file exposes no variable for it.

```yaml
rollout_backend: vllm
rollout_server_url: http://localhost:8000
train_on_sampled_tokens: true
routing_replay: rollout
```

Launch the trainer on the remaining GPUs; `expert_parallel_size` may match the
trainer's GPU count.

```bash
CUDA_VISIBLE_DEVICES=2,3 halo launch environmental-grpo qwen3-moe-grpo.yaml -n 2
```

`CUDA_VISIBLE_DEVICES` fences the trainer off the server; they cannot share a
GPU. Full setup:
[Environmental GRPO](../../agent-docs/training-methods/grpo/environmental-grpo.md) ↗.
