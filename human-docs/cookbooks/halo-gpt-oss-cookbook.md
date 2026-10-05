# Halo / GPT-OSS cookbook

Fine-tune [GPT-OSS 20B](https://huggingface.co/openai/gpt-oss-20b) with Halo.

The same recipe supports GPT-OSS 120B. Use a BF16 checkpoint for EP training.

## Halo support

| FSDP | EP | CP | TP | ETP | EP+CP | EP+TP | LoRA |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes |

Halo supports the GPT-OSS expert layout and attention sinks. It gathers EP, TP, and ETP shards into a standard Hugging Face checkpoint.

## Select a checkpoint

| Model | Training checkpoint | Suggested start |
|---|---|---|
| GPT-OSS 20B | `unsloth/gpt-oss-20b-BF16` | Eight GPUs with EP8 |
| GPT-OSS 120B | `unsloth/gpt-oss-120b-BF16` | Eight B300 GPUs with EP8; multi-node EP on H200 |

The native OpenAI checkpoints store the experts in MXFP4. Halo EP requires dequantized floating-point expert weights. Use the BF16 checkpoint for training.

This recipe uses eight NVIDIA B300 GPUs for EP8.

## Start the training container

Start the [cookbook container](README.md#start-the-training-container) and run the commands
below inside it, except the server commands marked for the host.

## Train all weights with EP8

Create `gpt-oss-20b-sft.yaml`.

```yaml
model_name_or_path: unsloth/gpt-oss-20b-BF16
model_init_kwargs:
  output_router_logits: true
  router_aux_loss_coef: 0.001
moe_balancing: aux_loss

dataset:
- HuggingFaceH4/ultrachat_200k@train_sft
conversation_field: messages
test_size: 0.01
chat_template: jinja-templates/gpt-oss/gpt-oss-multiturn.jinja
force_chat_template: true
assistant_message_template: <|start|>assistant<|channel|>final<|message|>
train_on_completions_only: true

expert_parallel_size: 8
save_sharded_ep: false
use_grouped_gemm: true
max_concurrent_loading: 2
fp32_output_conversion: false

use_liger_kernel: true
packing: true
max_length: 8192
bf16: true

per_device_train_batch_size: 2
per_device_eval_batch_size: 1
gradient_accumulation_steps: 4
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
output_dir: /data/checkpoints/gpt-oss-20b-ultrachat-ep8

logging_steps: 5
logging_first_step: true
report_to: wandb
remove_unused_columns: false
dataloader_num_workers: 2

use_peft: false
```

Two templates ship for GPT-OSS. `gpt-oss-multiturn.jinja` is the SFT choice: it renders
the `<|channel|>final` marker on every assistant turn, so completion-only masking trains
every turn. Under `gpt-oss-harmony.jinja` that marker matches nothing and the run trains
zero tokens at a loss near zero. RL uses harmony, where the training render must
byte-match the server's. Both need `force_chat_template: true`.

Launch eight processes.

```bash
halo launch sft gpt-oss-20b-sft.yaml -n 8
```

Halo selects the installed Flash Attention backend. SFT neutralizes the attention sinks by
default (`reset_sinks: true`) and exports them that way; a later stage with
`reset_sinks: false`, such as GRPO, runs the sinks as saved. If GRPO should keep the
pretrained sinks, set `reset_sinks: false` here too (FA4 on Blackwell; the CP variant below
then does not apply).

## Change the parallelism layout

On one eight-GPU node pure EP is 8, 2 or 1; for a 4-way expert split use `ep4 + etp2`
([rules](../parallelism.md#rules-that-save-you-a-wasted-run)).

Use CP2 with EP8 for long sequences. EP+CP requires the EP group to fill the NVLink
domain, so EP8 is the only EP size that pairs with CP here.

```yaml
expert_parallel_size: 8
context_parallel_size: 2
packing: false
```

Use EP8 with TP2 when attention weights need more sharding.

```yaml
expert_parallel_size: 8
tensor_parallel_size: 2
```

Use pure ETP8 when expert weight size is the main memory limit.

```yaml
expert_parallel_size: 1
expert_tensor_parallel_size: 8
```

Expert compute drops to the per-expert loop at `expert_tensor_parallel_size > 1`:
ETP de-interleaves GPT-OSS's GLU halves and stores the shards where the loop reads
them, not in the grouped-GEMM layout.

Do not combine attention TP with ETP.

## Run inference

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

path = "/data/checkpoints/gpt-oss-20b-ultrachat-ep8"
tokenizer = AutoTokenizer.from_pretrained(path)
model = AutoModelForCausalLM.from_pretrained(
    path,
    dtype=torch.bfloat16,
    device_map="auto",
)

messages = [{"role": "user", "content": "Write a short plan to diagnose an unstable training loss."}]
inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt",
).to(model.device)

output = model.generate(**inputs, max_new_tokens=512, do_sample=True, temperature=0.2)
print(tokenizer.decode(output[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True))
```

Serve the gathered checkpoint with SGLang on port 30000, from the host
([server setup](README.md#serve-from-the-host)).

```bash
SGLANG_MODEL="$HALO_SCRATCH/checkpoints/gpt-oss-20b-ultrachat-ep8" \
  docker compose -f docker-compose.sglang.yml up
```

## Train a LoRA adapter

Add this block to the EP8 configuration.

```yaml
use_peft: true
lora_r: 64
lora_alpha: 128
lora_dropout: 0.05
lora_task_type: CAUSAL_LM
lora_target_modules:
- q_proj
- k_proj
- v_proj
- o_proj
- gate_up_proj
- down_proj
lora_modules_to_save:
- embed_tokens
- lm_head
- router

learning_rate: 1.0e-04
output_dir: /data/checkpoints/gpt-oss-20b-ultrachat-lora
```

Halo sends the expert targets to its grouped LoRA path. Keep TP disabled for LoRA.

## Continue with GRPO

Copy `examples/grpo/environmental/environmental-grpo-template.yaml` to `gpt-oss-grpo.yaml`,
point `model_name_or_path` at the SFT checkpoint's `/data` path, set the environment and
reward fields for your task, and set the keys below, editing the template's own line where
it already has the key (a repeated key fails to parse). The shipped GPT-OSS configs under
`examples/grpo/environmental/gptoss/sglang/` (full and LoRA, ep1) and `.../vllm/` (full and
LoRA, ep1 and ep4) are already wired for their engine but list two servers; start the
ones their header names instead of the single server below.

```yaml
rollout_backend: sglang
rollout_server_url: http://localhost:30000
train_on_sampled_tokens: true
routing_replay: rollout
rollout_stop_tokens: ["<|call|>"]
chat_template: jinja-templates/gpt-oss/gpt-oss-harmony.jinja
force_chat_template: true
attn_implementation: flash_attention_4
reset_sinks: false
moe_balancing: none
beta: 0.0
output_dir: /data/checkpoints/gpt-oss-20b-grpo
fsdp_reshard_after_backward: false
```

`rollout_stop_tokens` matters because `<|call|>` is not an eos here: without it the
model generates past its tool call and hallucinates the result for most of the turn.
`fsdp_reshard_after_backward: false` is optional: it leaves one FSDP2 re-gather per
optimizer step instead of one per grad-accumulation microstep, for one unsharded bf16
parameter copy per GPU (fine at 20B).
`reset_sinks: false` keeps the checkpoint's sinks live and frozen so the trainer's log
probabilities match the served policy. Live sinks need a sink-carrying attention
implementation (FA4 on Blackwell); FA2 and SDPA are rejected and CP is unavailable
([sink handling](../../agent-docs/models/gpt-oss.md#attention-sinks) ↗). `beta: 0.0`
is required too: the reference model a nonzero `beta` builds cannot carry live sinks.

Serve the same harmony file. On the host ([server setup](README.md#serve-from-the-host)),
copy it onto the scratch volume, then start the server on GPUs the trainer will not use:

```bash
cp jinja-templates/gpt-oss/gpt-oss-harmony.jinja "$HALO_SCRATCH/"
```

```bash
SGLANG_MODEL="$HALO_SCRATCH/checkpoints/gpt-oss-20b-ultrachat-ep8" \
SGLANG_CHAT_TEMPLATE="$HALO_SCRATCH/gpt-oss-harmony.jinja" \
SGLANG_REASONING_PARSER=gpt-oss SGLANG_ENABLE_R3=1 \
SGLANG_CUDA_DEVICES=0,1,2,3 SGLANG_TP=4 \
  docker compose -f docker-compose.sglang.yml up sglang-server
```

The compose default `--tool-call-parser auto` picks the harmony parser off the template.
Launch the trainer in the training container on the remaining GPUs; they cannot share one.

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 halo launch environmental-grpo gpt-oss-grpo.yaml -n 4
```

vLLM (`rollout_backend: vllm`, `rollout_server_url: http://localhost:8000`) is the engine
the shipped ep4 configs target, and the only one for `rollout_max_thinking_tokens` and
`carry_reasoning` ([Supported Matrix](../supported-matrix.md#rollout-engines)).
`rollout_thinking_budget_scope: episode` is out of reach on GPT-OSS: SGLang refuses the
scope for every model, and on vLLM it counts a turn's reasoning up to a one-token close,
while GPT-OSS's close is five tokens.
GPT-OSS tool calls arrive as plain text that the default `hermes` parser cannot read, so
vLLM needs the bundled text tool parser, and a thinking budget needs the bundled reasoning
parser with Model Runner V1 (V2 answers `thinking_token_budget` with a 400). Set
`rollout_reasoning_end_token: "<|start|>assistant<|channel|>final<|message|>"` too, as the shipped
vLLM configs do: it is the opener the budget forces, and naming it keeps those forced tokens out of the loss:

```bash
VLLM_MODEL=/data/checkpoints/gpt-oss-20b-ultrachat-ep8 \
VLLM_CHAT_TEMPLATE=/data/gpt-oss-harmony.jinja \
VLLM_CUDA_DEVICES=0,1,2,3 VLLM_TP=4 VLLM_ENABLE_R3=1 \
VLLM_TOOL_PARSER_PLUGIN=/opt/gpt_oss_text_tool_parser.py VLLM_TOOL_PARSER=gpt_oss_text \
VLLM_REASONING_PARSER_PLUGIN=/opt/gpt_oss_reasoning_parser.py VLLM_REASONING_PARSER=openai_gptoss \
VLLM_USE_V2_MODEL_RUNNER=0 \
  docker compose -f docker-compose.vllm.yml up vllm-server
```

The trainer may size `expert_parallel_size` to its own GPU count; the shipped ep4
configs assume four trainer GPUs.

## Sources

- [GPT-OSS 20B model card](https://huggingface.co/openai/gpt-oss-20b)
- [GPT-OSS 120B model card](https://huggingface.co/openai/gpt-oss-120b)
- [Halo GPT-OSS model notes](../../agent-docs/models/gpt-oss.md) ↗
- Halo GPT-OSS SFT example: `examples/sft/gptoss/gptoss-20b-multinode-ep.yaml`
- [Async GRPO with Environments](../../agent-docs/training-methods/grpo/async-grpo/README.md) ↗
