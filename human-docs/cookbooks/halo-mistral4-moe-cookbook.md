# Halo / Mistral 4 MoE cookbook

Fine-tune [Mistral Small 4 119B A6B](https://huggingface.co/mistralai/Mistral-Small-4-119B-2603) with Halo.

The model has 128 routed experts and selects four experts for each token. It also has one shared expert and a Pixtral vision encoder. This recipe uses text data and keeps the multimodal model wrapper intact.

## Halo support

| FSDP | EP | CP | TP | ETP | EP+CP | EP+TP | LoRA |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes |

Halo uses DeepEP for token dispatch and grouped GEMM for the expert projections, and it preserves Mistral 4's group-top-k router and shared expert. CP and selective TP support the MLA attention layers.

This recipe starts with eight NVIDIA B300 GPUs. EP8 places 16 routed experts on each GPU.

## Start the training container

```bash
git clone --recurse-submodules https://github.com/whitecircle/halo
cd halo
docker pull public.ecr.aws/whitecircle/halo:blackwell
```

Export `HF_TOKEN` and `WANDB_API_KEY` in the host shell.

```bash
mkdir -p /mnt/hf /mnt/models /mnt/checkpoints /mnt/tmp
docker run --rm -it \
  --name halo-mistral4 \
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
  -v /mnt/models:/mnt/models \
  -v /mnt/tmp:/mnt/tmp \
  -v /mnt/checkpoints:/mnt/checkpoints \
  -w /workspace \
  public.ecr.aws/whitecircle/halo:blackwell bash
```

Run the training commands inside this container.

## Convert the checkpoint to BF16

The public checkpoint stores its expert weights in FP8, so convert it once before EP training. The source and destination checkpoints need about 500 GB of disk space in total.

```bash
hf download mistralai/Mistral-Small-4-119B-2603 \
  --local-dir /mnt/models/mistral-small-4-119b-fp8

python scripts/before_training/convert_mistral4_bf16.py \
  --model_id /mnt/models/mistral-small-4-119b-fp8 \
  --output_dir /mnt/models/mistral-small-4-119b-bf16
```

The converter streams one input shard at a time and writes a standard BF16 Hugging Face checkpoint.

## Train all weights with EP8

Create `mistral4-sft.yaml`, or start from `examples/sft/mistral4/mistral-small-4-119b-ultrachat-ep.yaml` and change `model_name_or_path` and `output_dir`.

```yaml
model_name_or_path: /mnt/models/mistral-small-4-119b-bf16
moe_balancing: bias_update_transient

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
max_length: 32000
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
output_dir: /mnt/checkpoints/mistral-small-4-119b-ultrachat-ep8

logging_steps: 1
logging_first_step: true
report_to: wandb
remove_unused_columns: false
dataloader_num_workers: 2

use_peft: false

assistant_message_template: "[/INST]"
train_on_completions_only: true
pad_token: <pad>
eos_token: </s>
chat_template: jinja_templates/mistral4-multiturn.jinja
force_chat_template: true
```

Launch eight processes.

```bash
halo launch sft mistral4-sft.yaml -n 8
```

Bias-update balancing changes expert selection without adding an auxiliary loss. Mistral 4 needs the `_transient` spelling: the router has no exportable bias slot, so the bias balances training-time routing only and every exported checkpoint serves without it (near-tied top-k picks can flip vs training). Plain `bias_update` raises. Halo gathers the expert weights when it saves because `save_sharded_ep` is false.

Keep `flash_attention_2` — it is what the shipped config pins and what this recipe was validated with.

## Add CP, TP, or ETP

Use CP2 with EP8 for longer sequences. Disable packing when CP splits the sequence.

```yaml
expert_parallel_size: 8
context_parallel_size: 2
packing: false
```

Use TP2 with EP8 when attention memory is the limit. TP shards the MLA expansion and output projections. The compression projections stay replicated.

```yaml
expert_parallel_size: 8
tensor_parallel_size: 2
```

Use pure ETP8 to shard the experts without DeepEP dispatch. Every rank then keeps all 128
experts at one-eighth width instead of 16 full experts.

```yaml
expert_parallel_size: 1
expert_tensor_parallel_size: 8
```

All three layouts use eight processes.

```bash
halo launch sft mistral4-sft.yaml -n 8
```

Do not combine attention TP with ETP. Do not combine LoRA with TP.

## Run multimodal inference

```python
import torch
from transformers import AutoProcessor, Mistral3ForConditionalGeneration

path = "/mnt/checkpoints/mistral-small-4-119b-ultrachat-ep8"
processor = AutoProcessor.from_pretrained(path)
model = Mistral3ForConditionalGeneration.from_pretrained(
    path,
    dtype=torch.bfloat16,
    device_map="auto",
)

messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Describe the image and list the safety risks."},
            {
                "type": "image_url",
                "image_url": {
                    "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg"
                },
            },
        ],
    }
]
inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device)

output = model.generate(**inputs, max_new_tokens=512, do_sample=True, temperature=0.2)
print(processor.decode(output[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True))
```

## Serving

Neither pinned rollout engine serves a toolkit Mistral 4 export. vLLM 0.26.0
registers no `mistral4` class: the public `mistralai/Mistral-Small-4-*` repos
serve only because vLLM detects their Mistral-native `params.json` layout, and a
toolkit export is plain HF-format with no such path. The one remaining route,
vLLM's generic transformers backend (`--model-impl transformers`), is neither
pinned nor verified here, and SGLang 0.5.17 has no verified path for the family
either. Run inference from transformers (above), or serve the pretrained hub
repo directly.

## Train a LoRA adapter

Add this block to the EP8 configuration.

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
output_dir: /mnt/checkpoints/mistral-small-4-119b-ultrachat-lora
```

Keep EP enabled for expert sharding. Keep TP disabled for LoRA.

## Continue with GRPO

Online and environmental GRPO are refused at trainer construction for this
family. `EPMistral4MoELayer` declares `_supports_weight_sync = False`: vLLM
0.26.0 registers no `mistral4` class, so there is no served model for the weight
stream to land in, and SGLang can weight-sync only GPT-OSS among the MoE
families. Offline GRPO trains on pre-generated scored completions and needs no
rollout server, so it remains available:
[Offline GRPO](../../agent-docs/training-methods/grpo/offline-grpo.md) ↗.
