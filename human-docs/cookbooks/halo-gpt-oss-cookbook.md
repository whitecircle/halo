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
| GPT-OSS 120B | `unsloth/gpt-oss-120b-BF16` | Multi-node EP |

The native OpenAI checkpoints store the experts in MXFP4. Halo EP requires dequantized floating-point expert weights. Use the BF16 checkpoint for training.

This recipe uses eight NVIDIA B300 GPUs for EP8.

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
  --name halo-gpt-oss \
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
chat_template: jinja-templates/gpt-oss/gpt-oss-harmony.jinja
force_chat_template: true
assistant_message_template: <|start|>assistant<|channel|>final<|message|>
train_on_completions_only: true

expert_parallel_size: 8
save_sharded_ep: false
use_grouped_gemm: false
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

Launch eight processes.

```bash
halo launch sft gpt-oss-20b-sft.yaml -n 8
```

Halo selects the installed Flash Attention backend. SFT resets the attention sinks by default.

## Change the parallelism layout

On one eight-GPU node, `expert_parallel_size` must be 8, 2, or 1 — an intermediate
size such as 4 forms two four-rank DeepEP dispatch groups whose combine barriers race
FSDP2, and [`ParallelismConfig`](../parallelism.md) rejects it at config time.

Use CP2 with EP8 for long sequences. EP+CP requires the EP group to fill the NVLink
domain, so EP8 is the only EP size that pairs with CP here.

```yaml
expert_parallel_size: 8
context_parallel_size: 2
packing: false
```

Use EP8 with TP2 when attention weights need more sharding. Attention TP leaves the
dispatch-group width alone, so the same EP sizes apply — 8, 2, or 1.

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
GPT-OSS stores its GLU halves interleaved, and once they are TP-sharded they can no
longer be read as contiguous grouped-GEMM operands.

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

Serve the gathered checkpoint with Halo's SGLang image on the host, not inside the
training container; it listens on port 30000. Point `SGLANG_IMAGE` at the prebuilt image
(no retag needed), or build the compose file's local tag once with `make build-sglang`.

```bash
docker pull public.ecr.aws/whitecircle/halo:sglang-0.5.17

SGLANG_IMAGE=public.ecr.aws/whitecircle/halo:sglang-0.5.17 \
SGLANG_MODEL=/data/checkpoints/gpt-oss-20b-ultrachat-ep8 \
SGLANG_MODEL_DIR=/data/checkpoints \
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

Start from `examples/grpo/environmental/environmental-grpo-template.yaml`, or from one of
the shipped GPT-OSS configs: `examples/grpo/environmental/gptoss/sglang/` (full and LoRA,
ep1) or `examples/grpo/environmental/gptoss/vllm/` (full and LoRA, ep1 and ep4). Point
`model_name_or_path` at the SFT checkpoint and set the environment and reward fields
for your task.

vLLM (`rollout_backend: vllm`) is the config default and runs the faster step. GPT-OSS is
also the only MoE family SGLang can weight-sync, because it loads experts in the
checkpoint-fused layout that only the GPT-OSS layer gathers, and the shipped
`sglang/gptoss-20b-code-contests-lora-ep1.yaml` is already wired for that engine. Two
constraints come with SGLang: the trainer must run at `expert_parallel_size: 1` (SGLang
refuses expert distribution), and `rollout_max_thinking_tokens` stays unset. That field is
vLLM-only; steer reasoning with the environment's `reasoning_effort` instead.

Serve from the prebuilt NCCL-aligned image, since upstream SGLang images ship a different
NCCL and cannot form the weight-sync group. Run it on the host, not inside the training
container, on GPUs the trainer will not use.

```bash
docker pull public.ecr.aws/whitecircle/halo:sglang-0.5.17

SGLANG_IMAGE=public.ecr.aws/whitecircle/halo:sglang-0.5.17 \
SGLANG_MODEL=/data/checkpoints/gpt-oss-20b-ultrachat-ep8 \
SGLANG_MODEL_DIR=/data/checkpoints \
SGLANG_CUDA_DEVICES=0,1,2,3 SGLANG_TP=4 \
  docker compose -f docker-compose.sglang.yml up sglang-server
```

The compose default `--tool-call-parser auto` resolves GPT-OSS to the harmony parser from
its chat template. A server started with no parser at all leaves tool calls as plain text:
every episode ends unsolved, and training runs to completion on a flat zero gradient. Set
`SGLANG_REASONING_PARSER=gpt-oss` as well. When serving `routing_replay: rollout`, add
`SGLANG_ENABLE_R3=1`; the triton MoE runner the capture hook needs is already the compose
file's default (`SGLANG_MOE_RUNNER_BACKEND=triton`).

Save the config as `gpt-oss-grpo.yaml` (or start from the shipped
`examples/grpo/environmental/gptoss/sglang/gptoss-20b-code-contests-lora-ep1.yaml`):

```yaml
rollout_backend: sglang
rollout_server_url: http://localhost:30000
train_on_sampled_tokens: true
routing_replay: rollout
rollout_stop_tokens: ["<|call|>"]
reset_sinks: false
moe_balancing: none
fsdp_reshard_after_backward: false
```

`rollout_stop_tokens` matters because `<|call|>` is not an eos here: without it the
model generates past its tool call and hallucinates the result for most of the turn.
`fsdp_reshard_after_backward: false` is required for performance on this engine. Its sync
forces process-global socket NCCL, and without the flag FSDP2 re-gathers the whole model
over loopback TCP once per grad-accum microstep.
`reset_sinks: false` keeps the pretrained sinks live and frozen so the trainer's log
probabilities match the served policy. Live sinks restrict the attention backend to a
sink-carrying implementation: FA4 on Blackwell, or `flex_attention`, `eager`, or an FA3
build exposing `s_aux` on Hopper (the shipped Hopper FA3 does not).
FA2 and SDPA are rejected, and CP is unavailable in this mode.

Launch the trainer on the remaining GPUs with the same five NCCL socket variables the
server's compose file sets, because the weight-sync group crosses the container boundary
and NCCL transport state is process-global. Expect a slower step than on vLLM for the same
reason: socket NCCL also costs the trainer NVLink between its own ranks.

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 \
NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1 NCCL_IB_DISABLE=1 NCCL_NET=Socket NCCL_NET_PLUGIN=none \
  halo launch environmental-grpo gpt-oss-grpo.yaml -n 4
```

`CUDA_VISIBLE_DEVICES` fences the trainer off the server — they cannot share a GPU.
The trainer keeps `expert_parallel_size: 1` on this engine, whatever its GPU count.

vLLM (`rollout_backend: vllm`, the config default) is the other engine, and it is required
for the expert-distributed ep4 configs and for `rollout_max_thinking_tokens`. Pull the
prebuilt server image and retag it to the name the compose file expects:

```bash
docker pull public.ecr.aws/whitecircle/halo:vllm-0.26.0
docker tag public.ecr.aws/whitecircle/halo:vllm-0.26.0 vllm-server:0.26.0
```

Its service mounts only the HuggingFace cache, so add
`- /data/checkpoints:/data/checkpoints:ro` under the `vllm-server` `volumes:` to serve a
checkpoint from disk. Two parsers are required. The Halo image runs GPT-OSS with harmony
off, so tool calls arrive as plain text that the default `hermes` parser cannot read — use
the bundled text tool parser instead. `rollout_max_thinking_tokens` needs the bundled
reasoning parser:

```bash
VLLM_MODEL=/data/checkpoints/gpt-oss-20b-ultrachat-ep8 \
VLLM_CUDA_DEVICES=0,1,2,3 VLLM_TP=4 \
VLLM_TOOL_PARSER_PLUGIN=/opt/gpt_oss_text_tool_parser.py \
VLLM_TOOL_PARSER=gpt_oss_text \
VLLM_REASONING_PARSER_PLUGIN=/opt/gpt_oss_reasoning_parser.py \
VLLM_REASONING_PARSER=openai_gptoss \
VLLM_USE_V2_MODEL_RUNNER=0 \
  docker compose -f docker-compose.vllm.yml up vllm-server
```

`VLLM_USE_V2_MODEL_RUNNER=0` pairs with the reasoning parser: Model Runner V2 rejects
`thinking_token_budget`, so with it on, every request carrying
`rollout_max_thinking_tokens` comes back 400.

That command already passes `--moe-backend triton`, which is required: Blackwell's
auto-selected MoE backends repack expert weights at load and silently corrupt every
weight sync. To serve `routing_replay: rollout`, also add `--enable-return-routed-experts`
to the server's `command:` block, since the compose file exposes no variable for it. If SFT
overrode the chat template, point `VLLM_CHAT_TEMPLATE` at the same `.jinja` so the
server-side render matches training. The trainer config then sets `rollout_backend: vllm`
and `rollout_server_url: http://localhost:8000`, and launches the same way minus the NCCL
socket variables. It may size `expert_parallel_size` to the trainer's GPU count; the
shipped ep4 configs assume four trainer GPUs.

## Sources

- [GPT-OSS 20B model card](https://huggingface.co/openai/gpt-oss-20b)
- [GPT-OSS 120B model card](https://huggingface.co/openai/gpt-oss-120b)
- [Halo GPT-OSS model notes](../../agent-docs/models/gpt-oss.md) ↗
- Halo GPT-OSS SFT example: `examples/sft/gptoss/gptoss-20b-multinode-ep.yaml`
- [Environmental GRPO](../../agent-docs/training-methods/grpo/environmental-grpo.md) ↗
