# Halo / LFM2 MoE cookbook

Fine-tune [LiquidAI LFM2.5 8B A1B](https://huggingface.co/LiquidAI/LFM2.5-8B-A1B) with Halo.

The same recipe covers [LFM2 24B A2B](https://huggingface.co/LiquidAI/LFM2-24B-A2B). Change the checkpoint and EP size.

LFM2 uses short-convolution layers and full-attention layers. Its MoE router uses sigmoid scores and expert-selection bias.

## Halo support

| FSDP | EP | CP | TP | ETP | EP+CP | EP+TP | LoRA |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Yes | Yes | No | Yes | Yes | No | Yes | Yes |

Halo uses DeepEP for token dispatch. It uses grouped GEMM for the fused expert projections.

CP is not supported. The short-convolution layers operate across the sequence axis and cannot use a Ulysses split.

## Select a checkpoint

| Checkpoint | Routed experts | Active experts | Suggested start |
|---|---:|---:|---|
| LFM2.5-8B-A1B | 32 | 4 | Two GPUs with EP2 |
| LFM2-24B-A2B | 64 | 4 | Four or eight GPUs |

This cookbook uses LFM2.5-8B-A1B. No LFM2 SFT config ships with Halo; the path is
exercised by `tests/gpu/trainers/sft/test_sft_lfm2_moe.py`, which runs LFM2-24B-A2B in
both plain FSDP and EP modes.

Start with two NVIDIA B300 GPUs for LFM2.5-8B-A1B and four or eight GPUs for
LFM2-24B-A2B; the GRPO continuation needs four, two for the rollout server and two for the
trainer.

## Start the training container

```bash
git clone --recurse-submodules https://github.com/whitecircle/halo
cd halo
docker pull public.ecr.aws/whitecircle/halo:blackwell
```

Export `HF_TOKEN` and `WANDB_API_KEY` in the host shell.

```bash
# D = the host's large scratch volume. /mnt is not guaranteed large — verify with `df -h`
# and point D (or HALO_SCRATCH) at the real big disk.
D=${HALO_SCRATCH:-/mnt}
mkdir -p "$D/hf" "$D/checkpoints" "$D/tmp"
docker run --rm -it \
  --name halo-lfm2 \
  --gpus all \
  --network host \
  --ipc=host \
  --shm-size=128g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -e HF_TOKEN \
  -e WANDB_API_KEY \
  -e HF_HOME=/data/hf \
  -e HF_DATASETS_CACHE=/data/hf/datasets \
  -e TMPDIR=/data/tmp \
  -e HALO_DATA_ROOT=/data \
  -e PYTHONPATH=/workspace \
  -e CUDA_DEVICE_MAX_CONNECTIONS=1 \
  -v "$(pwd)":/workspace \
  -v "$D":/data \
  -w /workspace \
  public.ecr.aws/whitecircle/halo:blackwell bash
```

Run all remaining commands inside this container.

## Train all weights with EP2

Create `lfm2-sft.yaml`. It trains on the supervised split of
[UltraChat 200K](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k). LFM2 uses
a ChatML-style template, and completion masking trains only the assistant turns.

```yaml
model_name_or_path: LiquidAI/LFM2.5-8B-A1B
model_revision: 5673e0de372b64331504de73bbbc33b0dde71903
moe_balancing: bias_update

dataset:
- HuggingFaceH4/ultrachat_200k@train_sft
conversation_field: messages
test_size: 0.01
train_on_completions_only: true
assistant_message_template: "<|im_start|>assistant\n"
pad_token: "<|pad|>"
eos_token: "<|im_end|>"

expert_parallel_size: 2
save_sharded_ep: false
use_grouped_gemm: true
fp32_router: true
fp32_experts: false

attn_implementation: flash_attention_2
use_liger_kernel: false
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
output_dir: /data/checkpoints/lfm2.5-8b-a1b-ultrachat-ep2

logging_steps: 1
logging_first_step: true
report_to: wandb
remove_unused_columns: false
dataloader_num_workers: 2

use_peft: false
```

Launch two processes.

```bash
halo launch sft lfm2-sft.yaml -n 2
```

Point the in-container steps below at this checkpoint.

```bash
export LFM2_CHECKPOINT=/data/checkpoints/lfm2.5-8b-a1b-ultrachat-ep2
```

Each rank owns 16 experts. Halo gathers the fused expert weights when it saves.

LFM2 has no native router auxiliary loss. `moe_balancing: bias_update` corrects expert selection during training.

## Scale to LFM2-24B-A2B

Change the model fields.

```yaml
model_name_or_path: LiquidAI/LFM2-24B-A2B
model_revision: 8ef52d16709b3c707176a469a005addc0460ea36
expert_parallel_size: 4
output_dir: /data/checkpoints/lfm2-24b-a2b-ultrachat-ep4
```

Launch four processes.

```bash
halo launch sft lfm2-sft.yaml -n 4
```

Point the in-container steps at the 24B checkpoint instead.

```bash
export LFM2_CHECKPOINT=/data/checkpoints/lfm2-24b-a2b-ultrachat-ep4
```

On an eight-GPU node, raise `expert_parallel_size` to 8 when expert memory is the main
limit, and launch eight processes to match. On a single node the working EP sizes are the
whole job, 2, or 1: an intermediate size such as 4 on eight GPUs forms two four-rank DeepEP
dispatch groups whose combine barriers race FSDP2, and
[`ParallelismConfig`](../parallelism.md) rejects it at config time. Keep CP disabled for
all released LFM2 MoE checkpoints.

## Add TP or ETP

Use EP2 with TP2 when the attention weights need more sharding. Both dimensions use the same two ranks.

```yaml
expert_parallel_size: 2
tensor_parallel_size: 2
```

```bash
halo launch sft lfm2-sft.yaml -n 2
```

TP shards only the full-attention layers; the short-convolution layers stay replicated, so
the per-rank footprint falls by considerably less than `1/tp_size`.

Use pure ETP when each local expert is too large. This mode keeps all experts and shards each expert across two GPUs.

```yaml
expert_parallel_size: 1
expert_tensor_parallel_size: 2
```

Do not combine attention TP with ETP. Do not combine LoRA with TP.

## Run inference

Load the gathered checkpoint with Transformers.

```python
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

path = os.environ["LFM2_CHECKPOINT"]
tokenizer = AutoTokenizer.from_pretrained(path)
model = AutoModelForCausalLM.from_pretrained(
    path,
    dtype=torch.bfloat16,
    device_map="auto",
)

messages = [{"role": "user", "content": "Summarize this support ticket and list the next actions."}]
inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt",
).to(model.device)

output = model.generate(
    **inputs,
    max_new_tokens=512,
    do_sample=True,
    temperature=0.1,
    top_k=50,
    repetition_penalty=1.05,
)
reply = tokenizer.decode(output[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
print(reply)
```

Serve the gathered checkpoint with Halo's SGLang image on the host, not inside the
training container; it listens on port 30000. Point `SGLANG_IMAGE` at the prebuilt image
(no retag needed), or build the compose file's local tag once with `make build-sglang`.

```bash
docker pull public.ecr.aws/whitecircle/halo:sglang-0.5.17

SGLANG_IMAGE=public.ecr.aws/whitecircle/halo:sglang-0.5.17 \
SGLANG_MODEL=/data/checkpoints/lfm2.5-8b-a1b-ultrachat-ep2 \
SGLANG_MODEL_DIR=/data/checkpoints \
  docker compose -f docker-compose.sglang.yml up
```

## Train an attention LoRA adapter

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
- out_proj

learning_rate: 1.0e-04
output_dir: /data/checkpoints/lfm2.5-8b-a1b-ultrachat-lora
```

Keep EP enabled if the base model needs expert sharding. Keep TP disabled for LoRA.

## Continue with GRPO

Start from the SFT checkpoint. Copy `examples/grpo/environmental/environmental-grpo-template.yaml`,
set `model_name_or_path` to that checkpoint, and set the environment and reward fields
for your task.

SGLang can weight-sync only GPT-OSS among the MoE families, so LFM2 rollouts run on vLLM
(`rollout_backend: vllm`, the config default). Start the server on separate GPUs.

Run the server on the host, not inside the training container. Pull the prebuilt server
image and retag it to the name the compose file expects. Its service mounts only the
HuggingFace cache, so add `- /data/checkpoints:/data/checkpoints:ro` under the
`vllm-server` `volumes:` to serve a checkpoint from disk.

```bash
docker pull public.ecr.aws/whitecircle/halo:vllm-0.26.0
docker tag public.ecr.aws/whitecircle/halo:vllm-0.26.0 vllm-server:0.26.0

VLLM_MODEL=/data/checkpoints/lfm2.5-8b-a1b-ultrachat-ep2 \
VLLM_CUDA_DEVICES=0,1 VLLM_TP=2 \
  docker compose -f docker-compose.vllm.yml up vllm-server
```

That command already passes `--moe-backend triton`, which is required: Blackwell's
auto-selected MoE backends repack expert weights at load and silently corrupt every
weight sync. To serve `routing_replay: rollout`, also add `--enable-return-routed-experts` to the
server's `command:` block, since the compose file exposes no variable for it.

```yaml
rollout_server_url: http://localhost:8000
train_on_sampled_tokens: true
routing_replay: rollout
```

Then launch the trainer.

```bash
CUDA_VISIBLE_DEVICES=2,3 halo launch environmental-grpo lfm2-grpo.yaml -n 2
```

`CUDA_VISIBLE_DEVICES` fences the trainer off the server — they cannot share a GPU.
Size `expert_parallel_size` to the trainer's GPU count, not the node's: the SFT value
assumes the whole node.

## Sources

- [LFM2.5-8B-A1B model card](https://huggingface.co/LiquidAI/LFM2.5-8B-A1B)
- [LFM2-24B-A2B model card](https://huggingface.co/LiquidAI/LFM2-24B-A2B)
- [Halo LFM2 model notes](../../agent-docs/models/lfm2.md) ↗
- [Environmental GRPO](../../agent-docs/training-methods/grpo/environmental-grpo.md) ↗
