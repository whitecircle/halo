# Fine-tune Gemma 4 MoE with Halo

Fine-tune [Gemma 4 26B A4B IT](https://huggingface.co/google/gemma-4-26B-A4B-it)
with expert parallelism, then serve the Hugging Face checkpoint or continue with GRPO.

Gemma 4 26B A4B has 128 routed experts and selects eight experts for each token. The checkpoint can process text and images. This recipe uses text data.

## Halo support

| FSDP | EP | CP | TP | ETP | EP+CP | EP+TP | LoRA |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Yes | Yes | No | No | Yes | No | No | Yes |

Halo uses DeepEP for token dispatch and grouped GEMM for the expert projections. CP and TP are not supported for this model family.

This recipe uses eight NVIDIA B300 GPUs. EP8 places 16 experts on each GPU. On
H100 or H200, use the Hopper image. The training config does not change.

## Start the training container

```bash
git clone --recurse-submodules https://github.com/whitecircle/halo
cd halo
export HALO_IMAGE=public.ecr.aws/whitecircle/halo:blackwell
# On H100 or H200, use public.ecr.aws/whitecircle/halo:hopper.
docker pull "$HALO_IMAGE"
```

Export `HF_TOKEN` and `WANDB_API_KEY` in the host shell. Downloading the Gemma checkpoint requires accepting Google's license on Hugging Face.

```bash
export HALO_SCRATCH=/path/to/storage
mkdir -p "$HALO_SCRATCH/hf" "$HALO_SCRATCH/checkpoints" "$HALO_SCRATCH/tmp"
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
  -e HF_HOME=/data/hf \
  -e HF_DATASETS_CACHE=/data/hf/datasets \
  -e TMPDIR=/data/tmp \
  -e HALO_DATA_ROOT=/data \
  -e PYTHONPATH=/workspace \
  -e CUDA_DEVICE_MAX_CONNECTIONS=1 \
  -v "$(pwd)":/workspace \
  -v "$HALO_SCRATCH":/data \
  -w /workspace \
  "$HALO_IMAGE" bash
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
output_dir: /data/checkpoints/gemma-4-26b-a4b-ultrachat-ep8

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

With eight training GPUs, `expert_parallel_size` must be 8, 2, or 1. A value of 4
forms two four-rank DeepEP dispatch groups whose
combine barriers race FSDP2, and [`ParallelismConfig`](../parallelism.md) rejects it at
config time.

## Run text inference

```python
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

path = "/data/checkpoints/gemma-4-26b-a4b-ultrachat-ep8"
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

Serve the gathered checkpoint with Halo's SGLang image. Run this command on the host.
The compose service mounts `SGLANG_MODEL_DIR` at the same path inside the container.

```bash
docker pull public.ecr.aws/whitecircle/halo:sglang-0.5.17

SGLANG_IMAGE=public.ecr.aws/whitecircle/halo:sglang-0.5.17 \
SGLANG_MODEL="$HALO_SCRATCH/checkpoints/gemma-4-26b-a4b-ultrachat-ep8" \
SGLANG_MODEL_DIR="$HALO_SCRATCH/checkpoints" \
SGLANG_CUDA_DEVICES=0,1,2,3 SGLANG_TP=4 \
  docker compose -f docker-compose.sglang.yml up sglang-server
```

## Train a LoRA adapter

These targets adapt the language model only. The vision and audio towers' projections share the
`q_proj`…`o_proj` names but are `Gemma4ClippableLinear`, a wrapper PEFT cannot adapt, so they are
excluded from injection with a warning naming the count. `all-linear` adapts the towers too.

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
output_dir: /data/checkpoints/gemma-4-26b-a4b-ultrachat-lora
```

Keep TP disabled for LoRA.

## Train with GRPO

Use the shipped SGLang LoRA recipe. It trains from the base checkpoint by default.
To continue from the SFT checkpoint, set `model_name_or_path` to its `/data` path.
Set `SGLANG_MODEL` to the matching `$HALO_SCRATCH` path on the host.

```bash
cp \
  examples/grpo/environmental/gemma4/sglang/gemma4-26b-a4b-code-contests-lora-ep1.yaml \
  gemma4-grpo.yaml
```

The config contains a placeholder dataset. Prepare a HardTests pool as described in
[Code Contests](../../agent-docs/training-methods/grpo/environments/code-contests.md#dataset) ↗,
then replace `your-org/code-contests-hardtests-rl:medium` in `gemma4-grpo.yaml`.

The shipped config uses two SGLang servers on four GPUs. Start both servers on the
host. The trainer will use GPUs 0–3.

```bash
docker pull public.ecr.aws/whitecircle/halo:sglang-0.5.17

export SGLANG_IMAGE=public.ecr.aws/whitecircle/halo:sglang-0.5.17
export SGLANG_MODEL=google/gemma-4-26B-A4B-it
cp jinja-templates/gemma4/gemma4-reasoning-effort.jinja "$HALO_SCRATCH/"
export SGLANG_MODEL_DIR="$HALO_SCRATCH"
export SGLANG_CHAT_TEMPLATE="$HALO_SCRATCH/gemma4-reasoning-effort.jinja"

SGLANG_CUDA_DEVICES=4,5 SGLANG_TP=2 SGLANG_PORT=30000 \
  docker compose -p gemma4-rollout-0 -f docker-compose.sglang.yml up -d sglang-server

SGLANG_CUDA_DEVICES=6,7 SGLANG_TP=2 SGLANG_PORT=30001 \
  docker compose -p gemma4-rollout-1 -f docker-compose.sglang.yml up -d sglang-server

curl --fail http://localhost:30000/health
curl --fail http://localhost:30001/health
```

The compose file sets `--moe-runner-backend triton`, which keeps the expert weights in
the layout required for synchronization. Do not set `SGLANG_ENABLE_R3`. Gemma 4 does
not support routing replay.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 DIST_NCCL_TIMEOUT_MINUTES=60 \
  halo launch environmental-grpo gemma4-grpo.yaml -n 4
```

`CUDA_VISIBLE_DEVICES` fences the trainer off the server; they cannot share a GPU.
The shipped SGLang recipe uses `expert_parallel_size: 1`. For a shipped EP4 recipe,
use a config under `examples/grpo/environmental/gemma4/vllm/`. Full setup:
[Async GRPO with Environments](../../agent-docs/training-methods/grpo/async-grpo/README.md) ↗.
