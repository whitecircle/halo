# Halo / ZAYA1 cookbook

Fine-tune [ZAYA1 8B](https://huggingface.co/Zyphra/ZAYA1-8B) with Halo.

ZAYA1 8B has 16 routed experts and selects one expert for each token. Halo supports the model's hybrid attention and MoE layers.

## Halo support

| FSDP | EP | CP | TP | ETP | EP+CP | EP+TP | LoRA |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Yes* | Yes* | No | No | Yes* | No | No | Yes |

`*` Gradient checkpointing is unavailable in every mode: FSDP2, EP, and ETP alike. The toolkit clears `ZayaPreTrainedModel.supports_gradient_checkpointing` at load, because recompute through CCA's `nn.Conv1d` pair faults in cuDNN on the CUDA 13.2 image; the launcher then raises rather than failing in the first backward. Halo supports neither CP nor TP here. The convolution-enhanced attention runs a `Conv1d` over the sequence axis and shifts each token's value one step, which breaks a Ulysses split, and it replaces QKV with `q_proj` / `k_proj` / `v_proj_current` / `v_proj_delayed` plus that conv stack, for which no DTensor sharding primitive exists.

ZAYA1 is a native transformers family: hub `main` loads directly, with no
revision pin and no `trust_remote_code`.

This recipe starts with one NVIDIA B300 GPU. The ETP variant needs two.

## Start the training container

```bash
git clone --recurse-submodules https://github.com/whitecircle/halo
cd halo
docker pull public.ecr.aws/whitecircle/halo:blackwell
```

Export `HF_TOKEN` and `WANDB_API_KEY` in the host shell.

```bash
mkdir -p /mnt/hf /mnt/checkpoints /mnt/tmp
docker run --rm -it \
  --name halo-zaya1 \
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

## Train all weights

Create `zaya1-sft.yaml`.

```yaml
model_name_or_path: Zyphra/ZAYA1-8B
model_init_kwargs:
  output_router_logits: false
moe_balancing: bias_update
router_balancing_rate: 0.001

dataset:
- HuggingFaceH4/ultrachat_200k@train_sft
conversation_field: messages
test_size: 0.01
dataset_num_proc: 16
train_on_completions_only: true
train_on_last_assistant_only: true
assistant_message_template: "<|im_start|>assistant\n<think>\n"
pad_token: <pad>
eos_token: <|im_end|>

use_grouped_gemm: true

attn_implementation: flash_attention_2
use_liger_kernel: true
packing: true
padding_free: false
max_length: 4096
bf16: true

per_device_train_batch_size: 1
per_device_eval_batch_size: 1
gradient_accumulation_steps: 16
num_train_epochs: 1.0
gradient_checkpointing: false

optim: adamw_torch_fused
learning_rate: 5.0e-06
lr_scheduler_type: cosine
warmup_ratio: 0.03
max_grad_norm: 1.0

save_strategy: steps
save_steps: 1000
eval_strategy: steps
eval_steps: 500
save_total_limit: 2
save_only_model: true
output_dir: /mnt/checkpoints/zaya1-8b-ultrachat

logging_steps: 1
logging_first_step: true
report_to: wandb
remove_unused_columns: false
dataloader_num_workers: 2
generate_eval_examples: false
seed: 42

use_peft: false
```

Launch one process.

```bash
halo launch sft zaya1-sft.yaml -n 1
```

Keep `gradient_checkpointing: false`. The toolkit clears
`ZayaPreTrainedModel.supports_gradient_checkpointing` at load, so every mode
raises at `gradient_checkpointing_enable` — FSDP2, EP, and ETP alike.

The shipped equivalents are `examples/sft/zaya/zaya-1-8b-ultrachat.yaml` and, for the EP
path, `examples/sft/zaya/zaya-1-8b-ultrachat-ep.yaml` (EP8, 16 routed experts → 2 per
rank, launched with `-n 8`).

## Use ETP

Use pure ETP to shard every expert across two GPUs. `expert_tensor_parallel_size` must
divide the process count, so this needs two processes; with `-n 1` the config is rejected
before the model loads.

```yaml
expert_parallel_size: 1
expert_tensor_parallel_size: 2
gradient_checkpointing: false
```

```bash
halo launch sft zaya1-sft.yaml -n 2
```

Larger ETP sizes are valid when the process count and the expert FFN width both divide.

## Run inference

```python
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer

path = "/mnt/checkpoints/zaya1-8b-ultrachat"
tokenizer = AutoTokenizer.from_pretrained(path)
model = AutoModelForCausalLM.from_pretrained(
    path,
    dtype=torch.bfloat16,
    device_map="auto",
)

messages = [{"role": "user", "content": "Give three checks for an imbalanced MoE router."}]
inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt",
).to(model.device)
output = model.generate(**inputs, max_new_tokens=256, do_sample=True, temperature=0.2)
print(tokenizer.decode(output[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True))
```

Neither pinned rollout engine serves ZAYA1 natively. vLLM 0.26.0 ships no native
Zaya implementation, and an exported checkpoint resolves only through its generic
transformers backend, whose generation quality is unverified, so inference stays
on transformers for now.

## Train a LoRA adapter

Use the ZAYA1 projection names in the LoRA configuration.

```yaml
use_peft: true
lora_r: 16
lora_alpha: 32
lora_dropout: 0.05
lora_target_modules:
- q_proj
- k_proj
- v_proj_current
- v_proj_delayed
- o_proj

learning_rate: 1.0e-04
output_dir: /mnt/checkpoints/zaya1-8b-ultrachat-lora
gradient_checkpointing: false
```

## No online RL for ZAYA1

Online and environmental GRPO are refused at construction for this family. Both
need weight sync into a rollout server, and the Zaya EP layer declares
`_supports_weight_sync = False`: vLLM 0.26.0 ships no native Zaya
implementation, so there is no served model for the stream to land in. SGLang
cannot take it either: its weight sync accepts only GPT-OSS among the MoE
families. Offline GRPO, which needs no rollout server, remains available:
[Offline GRPO](../../agent-docs/training-methods/grpo/offline-grpo.md) ↗.

## Sources

- [ZAYA1 8B model card](https://huggingface.co/Zyphra/ZAYA1-8B)
- [Halo Zaya model notes](../../agent-docs/models/zaya.md) ↗
- Halo Zaya examples: `examples/sft/zaya/zaya-1-8b-ultrachat.yaml`,
  `examples/sft/zaya/zaya-1-8b-ultrachat-ep.yaml`
