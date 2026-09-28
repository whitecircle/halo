# Halo / Poolside Laguna 2.1 cookbook

Fine-tune [Poolside Laguna S 2.1](https://huggingface.co/poolside/Laguna-S-2.1) with Halo.

The same recipe supports [Laguna XS 2.1](https://huggingface.co/poolside/Laguna-XS-2.1).

## Halo support

| FSDP | EP | CP | TP | ETP | EP+CP | EP+TP | LoRA |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Yes | Yes | No | No | partial | No | No | Yes |

Halo supports Laguna's sigmoid router, correction bias, shared expert, fused expert weights, and standard Hugging Face checkpoint layout.

ETP is mechanically reachable — the experts use the shared fused-GLU storage, so the generic sharding path handles `expert_tensor_parallel_size > 1` — but on Laguna its only GPU test is a tiny-model LoRA row at ETP2. Nothing rejects it; it is a validation gap, not a limit.

## Select a checkpoint

| Model | Checkpoint | Suggested start |
|---|---|---|
| Laguna S 2.1 | `poolside/Laguna-S-2.1` | Four GPUs with EP4 |
| Laguna XS 2.1 | `poolside/Laguna-XS-2.1` | One B300, single process |

This cookbook uses Laguna S 2.1 on four NVIDIA B300 GPUs; EP4 places 64 of the 256 routed experts on each GPU.

## Start the training container

Start the [cookbook container](README.md#start-the-training-container) and run the commands
below inside it, except the server commands marked for the host.

## Train Laguna S 2.1 with EP4

Create `laguna-s-2.1-sft.yaml`.

```yaml
model_name_or_path: poolside/Laguna-S-2.1
model_revision: e80da38da3ed4c4e56888cc1ba39582946a164ba
trust_remote_code: true

dataset:
- HuggingFaceH4/ultrachat_200k@train_sft
conversation_field: messages
test_size: 0.01
train_on_completions_only: true
assistant_message_template: "<assistant>"
pad_token: "〈|PAD|〉"
eos_token: "〈|EOS|〉"

expert_parallel_size: 4
save_sharded_ep: false
use_grouped_gemm: true

attn_implementation: sdpa
packing: true
max_length: 2048
bf16: true

per_device_train_batch_size: 1
per_device_eval_batch_size: 1
gradient_accumulation_steps: 8
num_train_epochs: 1.0
gradient_checkpointing: true

optim: flash_adamw
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
output_dir: /data/checkpoints/laguna-s-2.1-ultrachat-ep4

logging_steps: 1
logging_first_step: true
report_to: wandb
remove_unused_columns: false
dataloader_num_workers: 2

use_peft: false
```

Launch four processes.

```bash
halo launch sft laguna-s-2.1-sft.yaml -n 4
```

`ep_size=4` is one dispatch group on four GPUs. The same config on eight GPUs makes two
racy four-rank groups, which [`ParallelismConfig`](../parallelism.md) rejects at config
time — run this recipe on exactly four.

Two settings are load-bearing. `attn_implementation: sdpa` is required: the pinned hub
revision has no Flash-Attention path. And `pad_token` / `eos_token` really do use the CJK angle
brackets U+3008/U+3009 — substituting ASCII `<`/`>` silently adds new tokens instead of
resolving the existing ones.

The shipped equivalent is `examples/sft/laguna/laguna-s-2.1-ultrachat-ep.yaml`.

## Train Laguna XS 2.1 on one GPU

Copy the config to `laguna-xs-2.1-sft.yaml`, change the model and output directory, and
drop `expert_parallel_size`.

```yaml
model_name_or_path: poolside/Laguna-XS-2.1
model_revision: 205dc65dd4bda946c50da6b7522b215734fa107b
output_dir: /data/checkpoints/laguna-xs-2.1-ultrachat
```

Launch one process.

```bash
halo launch sft laguna-xs-2.1-sft.yaml -n 1
```

The shipped equivalent is `examples/sft/laguna/laguna-xs-2.1-ultrachat.yaml`.

## Use ETP

Pure ETP shards each expert instead of distributing whole experts. On Laguna only a
tiny-model LoRA row tests it — validate a short run before committing to it.

```yaml
expert_parallel_size: 1
expert_tensor_parallel_size: 2
```

Leave `use_grouped_gemm: true`. Laguna stores its GLU halves contiguously, which the ETP
split handles; only GPT-OSS, whose halves are interleaved, has to fall back to the
per-expert loop under ETP.

Use EP4 as the default full-model recipe. Use ETP when expert tensor size is the main memory limit. Do not enable TP or CP.

## Run inference

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

path = "/data/checkpoints/laguna-s-2.1-ultrachat-ep4"
tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    path,
    trust_remote_code=True,
    dtype=torch.bfloat16,
    device_map="auto",
)

messages = [{"role": "user", "content": "Write a short incident response plan for a failed deployment."}]
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
SGLANG_MODEL="$HALO_SCRATCH/checkpoints/laguna-s-2.1-ultrachat-ep4" \
  docker compose -f docker-compose.sglang.yml up
```

## Train an expert LoRA adapter

Add this block to the EP4 configuration.

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
- gate_proj
- up_proj
- down_proj

learning_rate: 1.0e-04
output_dir: /data/checkpoints/laguna-s-2.1-ultrachat-lora
```

Keep TP disabled for LoRA.

## Continue with GRPO

Copy `examples/grpo/environmental/environmental-grpo-template.yaml` to `laguna-grpo.yaml`,
set `model_name_or_path` to the gathered checkpoint's `/data` path and the environment and
reward fields for your task, and set the keys below, editing the template's own line where
it already has the key (a repeated key fails to parse):

```yaml
trust_remote_code: true
attn_implementation: sdpa
rollout_server_url: http://localhost:8000
train_on_sampled_tokens: true
routing_replay: rollout
beta: 0.0
output_dir: /data/checkpoints/laguna-s-2.1-grpo
```

Laguna rollouts run on vLLM (`rollout_backend: vllm`, the config default). SGLang 0.5.17
refuses the weight sync for the family: its loader asserts every routed-expert tensor of
every sparse layer in each `load_weights` call, which the chunked online update cannot
satisfy. Start the server on the host ([server setup](README.md#serve-from-the-host)), on
GPUs the trainer will not use:

```bash
VLLM_MODEL=/data/checkpoints/laguna-s-2.1-ultrachat-ep4 \
VLLM_CUDA_DEVICES=0,1 VLLM_TP=2 VLLM_ENABLE_R3=1 VLLM_TOOL_PARSER=poolside_v1 \
  docker compose -f docker-compose.vllm.yml up vllm-server
```

`VLLM_TOOL_PARSER=poolside_v1` reads Laguna's `<tool_call>name<arg_key>…` calls; the compose
default `hermes` leaves them as text, so a native-tool environment scores every episode
zero without erroring.

```bash
CUDA_VISIBLE_DEVICES=2,3 halo launch environmental-grpo laguna-grpo.yaml -n 2
```

`CUDA_VISIBLE_DEVICES` fences the trainer off the server — they cannot share a GPU.
The template leaves expert parallelism off; to shard the experts, add
`expert_parallel_size` matching the trainer's GPU count (2 here). Full setup:
[Async GRPO with Environments](../../agent-docs/training-methods/grpo/async-grpo/README.md) ↗.

## Sources

- [Laguna S 2.1 model card](https://huggingface.co/poolside/Laguna-S-2.1)
- [Laguna XS 2.1 model card](https://huggingface.co/poolside/Laguna-XS-2.1)
- [Halo Laguna model notes](../../agent-docs/models/laguna.md) ↗
- Halo Laguna examples: `examples/sft/laguna/laguna-s-2.1-ultrachat-ep.yaml`,
  `examples/sft/laguna/laguna-xs-2.1-ultrachat.yaml`
