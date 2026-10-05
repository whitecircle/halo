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

This recipe uses eight NVIDIA B300 GPUs for EP8. It keeps FP32 master weights for the
router, experts and dense layers; the router matmul runs in FP32, the rest in BF16.

## Start the training container

Start the [cookbook container](README.md#start-the-training-container) and run the commands
below inside it, except the server commands marked for the host.

## Train all weights with EP8

Start from the checked-in configuration.

```bash
cp examples/sft/glm4/glm-4.7-flash-ultrachat-ep.yaml glm47-sft.yaml
```

In the copy, set `output_dir: /data/checkpoints/glm-4.7-flash-ultrachat-ep8` so the
checkpoint lands on the scratch volume. Its settings are below, defaults spelled out. It trains on the
supervised split of [UltraChat 200K](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k) and renders
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

On one eight-GPU node pure EP is 8, 2 or 1; for a 4-way expert split use `ep4 + etp2`
([rules](../parallelism.md#rules-that-save-you-a-wasted-run)). EP+CP also needs the EP
group to fill the NVLink domain, so EP8 is the only EP size that pairs with CP here.

Use CP when the sequence length causes attention memory pressure.

```yaml
context_parallel_size: 2
packing: false
```

EP8 and CP2 use the same eight ranks. The collator rejects packing under CP.

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

Serve the gathered checkpoint with SGLang on port 30000, from the host
([server setup](README.md#serve-from-the-host)). On Blackwell set
`SGLANG_ATTENTION_BACKEND=triton`: the engine's default backend has no kernel for GLM-4's
MLA head size and the server exits at start.

```bash
SGLANG_MODEL="$HALO_SCRATCH/checkpoints/glm-4.7-flash-ultrachat-ep8" \
SGLANG_ATTENTION_BACKEND=triton \
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

Copy `examples/grpo/environmental/environmental-grpo-template.yaml` to `glm47-grpo.yaml`,
set `model_name_or_path` to the SFT checkpoint's `/data` path and the environment and
reward fields for your task, and set the keys below, editing the template's own line where
it already has the key (a repeated key fails to parse):

```yaml
rollout_server_url: http://localhost:8000
train_on_sampled_tokens: true
routing_replay: rollout
chat_template: jinja-templates/glm/glm-native.jinja
force_chat_template: true
beta: 0.0
output_dir: /data/checkpoints/glm-4.7-flash-grpo
```

The SFT's `glm-chat.jinja` renders no tools, so GRPO switches to `glm-native.jinja`, the
upstream template with its tool-call and observation turns
([chat templates](../../agent-docs/models/glm4.md#chat-templates) ↗). Its generation prompt
opens `<think>`, so rollouts start in thinking mode where the SFT trained the non-thinking
render; add `rollout_chat_template_kwargs: {enable_thinking: false}` to keep them
non-thinking. The server must serve the same file. On the host
([server setup](README.md#serve-from-the-host)), copy it onto the scratch volume:

```bash
cp jinja-templates/glm/glm-native.jinja "$HALO_SCRATCH/"
```

Rollouts run on vLLM (`rollout_backend: vllm`, the config default). Start the server on
GPUs the trainer will not use:

```bash
VLLM_MODEL=/data/checkpoints/glm-4.7-flash-ultrachat-ep8 \
VLLM_CHAT_TEMPLATE=/data/glm-native.jinja \
VLLM_CUDA_DEVICES=0,1,2,3 VLLM_TP=4 VLLM_ENABLE_R3=1 \
VLLM_TOOL_PARSER=glm47 VLLM_ATTENTION_BACKEND=CUTLASS_MLA \
  docker compose -f docker-compose.vllm.yml up vllm-server
```

`VLLM_TOOL_PARSER=glm47` is required: the compose default `hermes` cannot read GLM-4's
tool-call format, so every episode scores zero and the run trains on a flat zero gradient
without erroring. `VLLM_ATTENTION_BACKEND=CUTLASS_MLA` is required on Blackwell, whose
auto-selected MLA kernel rejects GLM-4's head config.

SGLang 0.5.17 also serves and weight-syncs this family. For it, set
`rollout_backend: sglang` and `rollout_server_url: http://localhost:30000`, and start:

```bash
SGLANG_MODEL="$HALO_SCRATCH/checkpoints/glm-4.7-flash-ultrachat-ep8" \
SGLANG_CHAT_TEMPLATE="$HALO_SCRATCH/glm-native.jinja" \
SGLANG_CUDA_DEVICES=0,1,2,3 SGLANG_TP=4 SGLANG_ENABLE_R3=1 \
SGLANG_ATTENTION_BACKEND=triton \
  docker compose -f docker-compose.sglang.yml up sglang-server
```

Then launch the trainer in the training container.

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 halo launch environmental-grpo glm47-grpo.yaml -n 4
```

`CUDA_VISIBLE_DEVICES` fences the trainer off the server — they cannot share a GPU.
The template leaves expert parallelism off; to shard the experts, add
`expert_parallel_size` matching the trainer's GPU count (4 here).

## Sources

- [GLM-4.7-Flash model card](https://huggingface.co/zai-org/GLM-4.7-Flash)
- [Halo GLM-4 model notes](../../agent-docs/models/glm4.md) ↗
- Halo GLM-4 SFT configuration: `examples/sft/glm4/glm-4.7-flash-ultrachat-ep.yaml`
- [Async GRPO with Environments](../../agent-docs/training-methods/grpo/async-grpo/README.md) ↗
