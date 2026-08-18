# Halo / Poolside Laguna 2.1 cookbook

Fine-tune [Poolside Laguna S 2.1](https://huggingface.co/poolside/Laguna-S-2.1) with Halo.

The same recipe supports [Laguna XS 2.1](https://huggingface.co/poolside/Laguna-XS-2.1).

## Halo support

| FSDP | EP | CP | TP | ETP | EP+CP | EP+TP | LoRA |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Yes | Yes | No | No | untested | No | No | Yes |

Halo supports Laguna's sigmoid router, correction bias, shared expert, fused expert weights, and standard Hugging Face checkpoint layout. CPU parity coverage lives in `tests/cpu/parallelism/test_laguna_ep.py`: an fp64 forward match against the library block, the per-expert gather layout, and the shared-expert naming.

ETP is mechanically reachable — the experts use the shared fused-GLU storage, so the generic sharding path handles `expert_tensor_parallel_size > 1` — but no GPU test covers it on Laguna. Nothing rejects it; it is a validation gap, not a limit.

## Select a checkpoint

| Model | Checkpoint | Suggested start |
|---|---|---|
| Laguna S 2.1 | `poolside/Laguna-S-2.1` | Four GPUs with EP4 |
| Laguna XS 2.1 | `poolside/Laguna-XS-2.1` | One B300, single process |

This cookbook uses Laguna S 2.1 on four NVIDIA B300 GPUs; EP4 places 64 of the 256 routed experts on each GPU.

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
  --name halo-laguna \
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
use_liger_kernel: false
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

Two settings are load-bearing. `attn_implementation: sdpa` and `use_liger_kernel: false`
are required: the pinned hub revision provides neither a Flash-Attention path nor
Liger-patchable module names. And `pad_token` / `eos_token` really do use the CJK angle
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

Pure ETP shards each expert instead of distributing whole experts. It is reachable on
Laguna but has no GPU test — validate a short run before committing to it.

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

Serve the gathered checkpoint with Halo's SGLang image on the host, not inside the
training container; it listens on port 30000. Pull the prebuilt image and point
`SGLANG_IMAGE` at it (no retag needed), or build the compose file's local tag once with
`make build-sglang`.

```bash
docker pull public.ecr.aws/whitecircle/halo:sglang-0.5.17

SGLANG_IMAGE=public.ecr.aws/whitecircle/halo:sglang-0.5.17 \
SGLANG_MODEL=/data/checkpoints/laguna-s-2.1-ultrachat-ep4 \
SGLANG_MODEL_DIR=/data/checkpoints \
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

Use `examples/grpo/environmental/environmental-grpo-template.yaml` as the starting point.
Set `model_name_or_path` to the gathered checkpoint.

SGLang can weight-sync only GPT-OSS among the MoE families, so Laguna rollouts run on
vLLM (`rollout_backend: vllm`, the config default). Start the server on separate GPUs.

Run the server on the host, not inside the training container; the commands below retag
the pulled image to the name the compose file expects. Its service mounts only the
HuggingFace cache, so add `- /data/checkpoints:/data/checkpoints:ro` under the
`vllm-server` `volumes:` to serve a checkpoint from disk.

```bash
docker pull public.ecr.aws/whitecircle/halo:vllm-0.26.0
docker tag public.ecr.aws/whitecircle/halo:vllm-0.26.0 vllm-server:0.26.0

VLLM_MODEL=/data/checkpoints/laguna-s-2.1-ultrachat-ep4 \
VLLM_CUDA_DEVICES=0,1 VLLM_TP=2 \
  docker compose -f docker-compose.vllm.yml up vllm-server
```

That command already passes `--moe-backend triton`, which is required: Blackwell's
auto-selected MoE backends repack expert weights at load and silently corrupt every
weight sync. To serve `routing_replay: rollout`, also add `--enable-return-routed-experts` to the
server's `command:` block — the compose file exposes no variable for it.

Save the config as `laguna-grpo.yaml`:

```yaml
rollout_server_url: http://localhost:8000
train_on_sampled_tokens: true
routing_replay: rollout
```

```bash
CUDA_VISIBLE_DEVICES=2,3 halo launch environmental-grpo laguna-grpo.yaml -n 2
```

`CUDA_VISIBLE_DEVICES` fences the trainer off the server — they cannot share a GPU.
Size `expert_parallel_size` to the trainer's GPU count, not the node's: the SFT value
assumes the whole node. Full setup:
[Environmental GRPO](../../agent-docs/training-methods/grpo/environmental-grpo.md) ↗.

## Sources

- [Laguna S 2.1 model card](https://huggingface.co/poolside/Laguna-S-2.1)
- [Laguna XS 2.1 model card](https://huggingface.co/poolside/Laguna-XS-2.1)
- [Halo Laguna model notes](../../agent-docs/models/laguna.md) ↗
- Halo Laguna examples: `examples/sft/laguna/laguna-s-2.1-ultrachat-ep.yaml`,
  `examples/sft/laguna/laguna-xs-2.1-ultrachat.yaml`
