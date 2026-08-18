# Halo / Gemma 4 MoE cookbook

Fine-tune [Gemma 4 26B A4B IT](https://huggingface.co/google/gemma-4-26B-A4B-it) with Halo.

Gemma 4 26B A4B has 128 routed experts and selects eight experts for each token. The checkpoint can process text and images. This recipe uses text data.

## Halo support

| FSDP | EP | CP | TP | ETP | EP+CP | EP+TP | LoRA |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Yes | Yes | No | No | Yes | No | No | Yes |

Halo uses DeepEP for token dispatch and grouped GEMM for the expert projections. CP and TP are not supported for this model family.

This recipe starts with eight NVIDIA B300 GPUs. EP8 places 16 experts on each GPU.

## Start the training container

```bash
git clone --recurse-submodules https://github.com/whitecircle/halo
cd halo
docker pull public.ecr.aws/whitecircle/halo:blackwell
```

Export `HF_TOKEN` and `WANDB_API_KEY` in the host shell. Downloading the Gemma checkpoint requires accepting Google's license on Hugging Face.

```bash
mkdir -p /mnt/hf /mnt/checkpoints /mnt/tmp
docker run --rm -it \
  --name halo-gemma4 \
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

Create `gemma4-sft.yaml`, or start from the shipped
`examples/sft/gemma4/gemma4-26b-a4b-ultrachat-ep.yaml`.

```yaml
model_name_or_path: google/gemma-4-26B-A4B-it
moe_balancing: none

dataset:
- HuggingFaceH4/ultrachat_200k@train_sft
conversation_field: messages
test_size: 0.01
train_on_completions_only: true
assistant_message_template: "<|turn>model\n"
pad_token: <pad>

expert_parallel_size: 8
save_sharded_ep: false
use_grouped_gemm: true
fp32_experts: true
fp32_output_conversion: false

attn_implementation: sdpa
use_liger_kernel: true
packing: true
max_length: 32768
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
output_dir: /mnt/checkpoints/gemma-4-26b-a4b-ultrachat-ep8

logging_steps: 1
logging_first_step: true
report_to: wandb
remove_unused_columns: false
dataloader_num_workers: 2

use_peft: false
```

Launch eight processes.

```bash
halo launch sft gemma4-sft.yaml -n 8
```

Keep `attn_implementation: sdpa`. It is the supported attention backend for this checkpoint.

Do not enable `fp32_non_ep_params`. The router is outside the EP wrapper, and DeepEP combine expects BF16 data. Gemma 4 has no supported auxiliary-loss or bias-update balancing path, so keep `moe_balancing: none`.

Keep `per_device_train_batch_size` at 1 with `packing: true` and scale through
`gradient_accumulation_steps`: Gemma 4 never reaches a varlen kernel, so packing builds a
dense mask over the flattened batch and its memory grows with the square of
`batch size × max_length`.

## Use ETP

Use pure ETP when the expert weights need more sharding. It shards only the expert FFN,
leaving attention and the sibling router replicated, and it divides the data-parallel
width: eight ranks at ETP8 train one batch stream.

```yaml
expert_parallel_size: 1
expert_tensor_parallel_size: 8
```

Do not enable TP or CP. On one eight-GPU node `expert_parallel_size` must also be 8, 2,
or 1. An intermediate size such as 4 forms two four-rank DeepEP dispatch groups whose
combine barriers race FSDP2, and [`ParallelismConfig`](../parallelism.md) rejects it at
config time.

## Run text inference

```python
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

path = "/mnt/checkpoints/gemma-4-26b-a4b-ultrachat-ep8"
processor = AutoProcessor.from_pretrained(path)
model = AutoModelForImageTextToText.from_pretrained(
    path,
    dtype=torch.bfloat16,
    device_map="auto",
)

messages = [{"role": "user", "content": [{"type": "text", "text": "Summarize the main failure modes of an MoE router."}]}]
inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device)

output = model.generate(**inputs, max_new_tokens=256, do_sample=True, temperature=0.2)
print(processor.decode(output[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True))
```

Use the same model class and processor for image-and-text inference.

Serve the gathered checkpoint with Halo's vLLM image, which listens on port
8000: the toolkit writes Gemma 4 exports in the config schema vLLM 0.26.0's
model code reads, and its expert loader takes the gathered save's fused layout
directly. SGLang 0.5.17 cannot load it, because outside GPT-OSS its loaders want
per-expert expert names and this family's gather emits the fused pair. Run the
server on the host, not inside the training container, and add
`- /mnt/checkpoints:/mnt/checkpoints:ro` under the `vllm-server` `volumes:` to
serve a checkpoint from disk, since the compose service otherwise mounts only
the HuggingFace cache.

```bash
docker pull public.ecr.aws/whitecircle/halo:vllm-0.26.0
docker tag public.ecr.aws/whitecircle/halo:vllm-0.26.0 vllm-server:0.26.0

VLLM_MODEL=/mnt/checkpoints/gemma-4-26b-a4b-ultrachat-ep8 \
VLLM_CUDA_DEVICES=0,1,2,3 VLLM_TP=4 \
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
output_dir: /mnt/checkpoints/gemma-4-26b-a4b-ultrachat-lora
```

Keep TP disabled for LoRA.

## Continue with GRPO

Start from `examples/grpo/environmental/gemma4/vllm/gemma4-26b-a4b-code-contests-lora-ep1.yaml`,
or from `examples/grpo/environmental/environmental-grpo-template.yaml`. Set
`model_name_or_path` to the gathered checkpoint.

SGLang can weight-sync only GPT-OSS among the MoE families, so Gemma 4 rollouts run on
vLLM (`rollout_backend: vllm`, the config default). Start the server on separate GPUs.

Run the server on the host, not inside the training container. Pull the prebuilt server
image, retag it to the name the compose file expects, and add
`- /mnt/checkpoints:/mnt/checkpoints:ro` under the `vllm-server` `volumes:` to serve a
checkpoint from disk, since its service mounts only the HuggingFace cache.

```bash
docker pull public.ecr.aws/whitecircle/halo:vllm-0.26.0
docker tag public.ecr.aws/whitecircle/halo:vllm-0.26.0 vllm-server:0.26.0

VLLM_MODEL=/mnt/checkpoints/gemma-4-26b-a4b-ultrachat-ep8 \
VLLM_CUDA_DEVICES=0,1,2,3 VLLM_TP=4 \
  docker compose -f docker-compose.vllm.yml up vllm-server
```

That command already passes the required `--moe-backend triton`; Blackwell's
auto-selected MoE backends repack expert weights at load and silently corrupt every
weight sync.

```yaml
rollout_server_url: http://localhost:8000
train_on_sampled_tokens: true
routing_replay: none
```

Leave `routing_replay` at `none`. Both replay modes are rejected for Gemma 4: its router
sits outside the EP wrapper, so gate weights cannot be re-derived at a forced selection.

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 halo launch environmental-grpo gemma4-grpo.yaml -n 4
```

`CUDA_VISIBLE_DEVICES` fences the trainer off the server; they cannot share a GPU.
Size `expert_parallel_size` to the trainer's GPU count, not the node's: the SFT value
assumes the whole node. Full setup:
[Environmental GRPO](../../agent-docs/training-methods/grpo/environmental-grpo.md) ↗.
