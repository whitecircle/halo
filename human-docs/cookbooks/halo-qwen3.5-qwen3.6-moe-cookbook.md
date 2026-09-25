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

This recipe starts with eight NVIDIA B300 GPUs; EP8 places 32 experts on each GPU.

## Start the training container

Start the [cookbook container](README.md#start-the-training-container) and run the commands
below inside it, except the server commands marked for the host.

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
chat_template: jinja-templates/qwen3/qwen3-multiturn.jinja
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
output_dir: /data/checkpoints/qwen3.6-35b-a3b-ultrachat-ep8

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

On one eight-GPU node pure EP is 8, 2 or 1, with or without attention TP; for a 4-way
expert split use `ep4 + etp2` ([rules](../parallelism.md#rules-that-save-you-a-wasted-run)).

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

path = "/data/checkpoints/qwen3.6-35b-a3b-ultrachat-ep8"
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

Serve the gathered checkpoint with vLLM on port 8000, from the host
([server setup](README.md#serve-from-the-host)); vLLM 0.26.0's expert loader reads the
gathered save's fused layout directly. SGLang 0.5.17 registers the multimodal
`Qwen3_5MoeForConditionalGeneration` as well as the text-only `Qwen3_5MoeForCausalLM`,
so it serves the hub checkpoint and a `text_only_model` export alike; vLLM takes the
latter only after `scripts/after_training/reattach_vision_tower.py`.

```bash
VLLM_MODEL=/data/checkpoints/qwen3.6-35b-a3b-ultrachat-ep8 \
VLLM_CUDA_DEVICES=0,1,2,3 VLLM_TP=4 VLLM_TOOL_PARSER=qwen3_xml \
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
output_dir: /data/checkpoints/qwen3.6-35b-a3b-ultrachat-lora
```

Keep TP disabled for LoRA.

## Continue with GRPO

Copy the shipped EP4 code-contests recipe, set `model_name_or_path` to the SFT
checkpoint's `/data` path, and point `output_dir` at `/data/checkpoints/` too; the recipe
writes under the repo checkout.

```bash
cp examples/grpo/environmental/qwen3_5/vllm/qwen3.6-35b-a3b-code-contests-full-ep4.yaml \
  qwen3.6-grpo.yaml
```

Its dataset is a placeholder: prepare a HardTests pool as described in
[Code Contests](../../agent-docs/training-methods/grpo/environments/code-contests.md#dataset) ↗,
then replace `your-org/code-contests-hardtests-rl:medium`. The other configs under
`examples/grpo/environmental/qwen3_5/vllm/` change the environment, adapter or EP size.
The full-finetune ep1 code-contests recipe is a curriculum: run `-stage1-codeforces`,
`-stage2-hard` and `-stage3-extra-hard` in order, each from the previous stage's checkpoint.

The recipe pins `chat_template: jinja-templates/qwen3/qwen3.6-reasoning-effort.jinja` and
sends a thinking budget, so its two servers must serve that same file, with the `qwen3_xml`
tool parser, the `qwen3` reasoning parser and `VLLM_USE_V2_MODEL_RUNNER=0`. Start them on
the host ([server setup](README.md#serve-from-the-host)), on GPUs 4–7:

```bash
cp jinja-templates/qwen3/qwen3.6-reasoning-effort.jinja "$HALO_SCRATCH/"
export VLLM_MODEL=/data/checkpoints/qwen3.6-35b-a3b-ultrachat-ep8
export VLLM_CHAT_TEMPLATE=/data/qwen3.6-reasoning-effort.jinja
export VLLM_TOOL_PARSER=qwen3_xml VLLM_REASONING_PARSER=qwen3 VLLM_USE_V2_MODEL_RUNNER=0

VLLM_CUDA_DEVICES=4,5 VLLM_TP=2 VLLM_PORT=8000 \
  docker compose -p qwen36-rollout-0 -f docker-compose.vllm.yml up -d vllm-server
VLLM_CUDA_DEVICES=6,7 VLLM_TP=2 VLLM_PORT=8001 \
  docker compose -p qwen36-rollout-1 -f docker-compose.vllm.yml up -d vllm-server
```

The `hermes` default cannot parse this family's XML tool calls, so without `qwen3_xml`
every episode ends unsolved and training runs on a flat zero gradient. Model Runner V2
answers the thinking budget with a 400 on every request.

Launch the trainer in the training container on GPUs 0–3.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 DIST_NCCL_TIMEOUT_MINUTES=60 \
  halo launch environmental-grpo qwen3.6-grpo.yaml -n 4
```

SGLang 0.5.17 also serves and weight-syncs this family, from the ep1 configs under
`examples/grpo/environmental/qwen3_5/sglang/` (ports 30000 and 30001). Serve them with
`SGLANG_CHAT_TEMPLATE="$HALO_SCRATCH/qwen3.6-reasoning-effort.jinja"` and
`SGLANG_REASONING_PARSER=qwen3`.
`rollout_max_thinking_tokens`, `rollout_thinking_budget_scope: episode` and
`carry_reasoning` are vLLM-only ([Supported Matrix](../supported-matrix.md#rollout-engines)).
Full setup:
[Async GRPO with Environments](../../agent-docs/training-methods/grpo/async-grpo/README.md) ↗.
