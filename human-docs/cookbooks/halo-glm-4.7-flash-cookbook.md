# Halo / GLM-4.7-Flash cookbook

Fine-tune [Z.ai GLM-4.7-Flash](https://huggingface.co/zai-org/GLM-4.7-Flash) with Halo.

GLM-4.7-Flash uses the GLM-4 MoE Lite architecture. It has 64 routed experts and selects four per token.

This recipe uses supervised fine-tuning, expert parallelism, and a 30,720-token sequence length.

## Halo support

| FSDP | EP | CP | TP | ETP | EP+CP | EP+TP | LoRA |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes |

Halo preserves the sigmoid router, the group-limited top-k rule, and the shared experts.

The EP path uses DeepEP and grouped GEMM. The CP and TP paths support GLM's compressed MLA attention.

This recipe uses eight NVIDIA B300 GPUs for EP8, with BF16 inputs and FP32 router and
expert compute.

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
  --name halo-glm47 \
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

## Train all weights with EP8

Start from the checked-in configuration.

```bash
cp examples/sft/glm4/glm-4.7-flash-ultrachat-ep.yaml glm47-sft.yaml
```

The copy reads as below. It trains on the supervised split of
[UltraChat 200K](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k) and renders
multi-turn data with Halo's `glm-chat.jinja` template, which preserves GLM's native role
markers.

```yaml
model_name_or_path: zai-org/GLM-4.7-Flash
moe_balancing: bias_update

dataset:
- HuggingFaceH4/ultrachat_200k@train_sft
conversation_field: messages
test_size: 0.01

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
max_length: 30720
bf16: true

per_device_train_batch_size: 2
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
output_dir: /data/checkpoints/glm-4.7-flash-ultrachat-ep8

logging_steps: 1
logging_first_step: true
report_to: wandb
remove_unused_columns: false
dataloader_num_workers: 2

use_peft: false

assistant_message_template: "<|assistant|>"
train_on_completions_only: true
pad_token: <|endoftext|>
eos_token: <|endoftext|>
chat_template: jinja-templates/glm/glm-chat.jinja
force_chat_template: true
```

Launch eight processes.

```bash
halo launch sft glm47-sft.yaml -n 8
```

Each rank owns eight routed experts. Halo gathers the expert weights when it saves.

Flash Attention 4's backward pass emits NaN gradients on GLM's MLA shape, so the loader
demotes an FA4 selection to SDPA for this family. Flash Attention 2 is the stable choice
and what the shipped config sets.

## Add CP, TP, or ETP

On one eight-GPU node `expert_parallel_size` must be 8, 2, or 1 — an intermediate size
such as 4 forms two four-rank DeepEP dispatch groups whose combine barriers race FSDP2,
and [`ParallelismConfig`](../parallelism.md) rejects it at config time. EP+CP narrows that
further: the EP group has to fill the NVLink domain, so EP8 is the only EP size that pairs
with CP here.

Use CP when the sequence length causes attention memory pressure.

```yaml
context_parallel_size: 2
```

EP8 and CP2 use the same eight ranks.

```bash
halo launch sft glm47-sft.yaml -n 8
```

Use TP when the compressed attention weights need more sharding.

```yaml
tensor_parallel_size: 2
```

EP8 and TP2 also use the same eight ranks.

```bash
halo launch sft glm47-sft.yaml -n 8
```

Use pure ETP when each local expert is too large. This mode keeps all experts and shards each expert across eight GPUs.

```yaml
expert_parallel_size: 1
expert_tensor_parallel_size: 8
```

Do not combine attention TP with ETP. Do not combine LoRA with TP.

## Run inference

Load the gathered checkpoint with Transformers.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

path = "/data/checkpoints/glm-4.7-flash-ultrachat-ep8"
tokenizer = AutoTokenizer.from_pretrained(path)
model = AutoModelForCausalLM.from_pretrained(
    path,
    dtype=torch.bfloat16,
    device_map="auto",
)

messages = [{"role": "user", "content": "Explain how to diagnose a failed distributed training step."}]
inputs = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device)

output = model.generate(**inputs, max_new_tokens=512, do_sample=False)
reply = tokenizer.decode(output[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)
print(reply)
```

Serve the gathered checkpoint with Halo's SGLang image on the host, not inside the
training container; it listens on port 30000. Point `SGLANG_IMAGE` at the prebuilt image
(no retag needed), or build the compose file's local tag once with `make build-sglang`. On
Blackwell, first add `--attention-backend triton` to the compose file's `command:` block:
flashinfer's MLA kernel rejects GLM-4's head config on SM100+, and the compose file
exposes no variable for the flag.

```bash
docker pull public.ecr.aws/whitecircle/halo:sglang-0.5.17

SGLANG_IMAGE=public.ecr.aws/whitecircle/halo:sglang-0.5.17 \
SGLANG_MODEL=/data/checkpoints/glm-4.7-flash-ultrachat-ep8 \
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
- q_a_proj
- q_b_proj
- kv_a_proj_with_mqa
- kv_b_proj
- o_proj

learning_rate: 1.0e-04
output_dir: /data/checkpoints/glm-4.7-flash-ultrachat-lora
```

Keep EP enabled if the base model needs expert sharding. Keep TP disabled for LoRA.

## Continue with GRPO

Start from the SFT checkpoint. Copy `examples/grpo/environmental/environmental-grpo-template.yaml`,
set `model_name_or_path` to that checkpoint, and set the environment and reward fields
for your task.

SGLang can weight-sync only GPT-OSS among the MoE families, so GLM-4.7 rollouts run on
vLLM (`rollout_backend: vllm`, the config default). Start the server on separate GPUs.

Run the server on the host, not inside the training container: pull the prebuilt server
image and retag it to the name the compose file expects. Its service mounts only the
HuggingFace cache, so add `- /data/checkpoints:/data/checkpoints:ro` under the
`vllm-server` `volumes:` to serve a checkpoint from disk.

```bash
docker pull public.ecr.aws/whitecircle/halo:vllm-0.26.0
docker tag public.ecr.aws/whitecircle/halo:vllm-0.26.0 vllm-server:0.26.0

VLLM_MODEL=/data/checkpoints/glm-4.7-flash-ultrachat-ep8 \
VLLM_CUDA_DEVICES=0,1,2,3 VLLM_TP=4 \
VLLM_TOOL_PARSER=glm47 \
  docker compose -f docker-compose.vllm.yml up vllm-server
```

`VLLM_TOOL_PARSER` is not optional here. The compose default is `hermes`, which cannot
read GLM-4's tool-call format: the server returns no `tool_calls`, every episode scores
zero reward, and the run trains on a flat zero gradient without erroring.

That command already passes `--moe-backend triton`, which is required: Blackwell's
auto-selected MoE backends repack expert weights at load and silently corrupt every
weight sync. On Blackwell, GLM-4's MLA attention additionally needs
`VLLM_ATTENTION_BACKEND=CUTLASS_MLA` (a compose variable; SGLang:
`--attention-backend triton`). Serving `routing_replay: rollout` needs
`--enable-return-routed-experts`, added to the server's `command:` block, since the
compose file exposes no variable for that one.

```yaml
rollout_server_url: http://localhost:8000
train_on_sampled_tokens: true
routing_replay: rollout
```

Then launch the trainer.

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 halo launch environmental-grpo glm47-grpo.yaml -n 4
```

`CUDA_VISIBLE_DEVICES` fences the trainer off the server — they cannot share a GPU.
Size `expert_parallel_size` to the trainer's GPU count, not the node's: the SFT value
assumes the whole node.

## Sources

- [GLM-4.7-Flash model card](https://huggingface.co/zai-org/GLM-4.7-Flash)
- [Halo GLM-4 model notes](../../agent-docs/models/glm4.md) ↗
- Halo GLM-4 SFT configuration: `examples/sft/glm4/glm-4.7-flash-ultrachat-ep.yaml`
- [Environmental GRPO](../../agent-docs/training-methods/grpo/environmental-grpo.md) ↗
