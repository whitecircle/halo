# Halo / Command A+ cookbook

Fine-tune [Cohere Command A+](https://huggingface.co/CohereLabs/command-a-plus-05-2026-bf16) with Halo.

This recipe uses the BF16 checkpoint and text-only UltraChat data. The same model can also train on image-text data.

Command A+ has 128 routed experts. The router selects eight experts and also runs four shared experts.

Validation status: EP8 is validated on the 200B+ checkpoint with a real 8× B300 run. The
CP, TP, and ETP wrappers pass the tiny-model 8-GPU parallelism matrix (cp8, tp8, ep8+tp2,
ep2+etp4), but no full-scale run has confirmed them yet; the layout sections below say so
where it matters.

## Halo support

| FSDP | EP | CP | TP | ETP | EP+CP | EP+TP | LoRA |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Yes | Yes | untested | untested | untested | untested | untested | untested |

Halo wraps the Cohere2 MoE blocks inside the vision model. It keeps the native Hugging Face checkpoint format.

The EP path preserves the sigmoid router and averaged shared expert. The CP path preserves each layer's positional-encoding rule.

The EP8 recipe assumes eight NVIDIA B300 GPUs; the BF16 model needs at least four B200 GPUs for inference.

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
  --name halo-command-a-plus \
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

## Train all weights with EP8

The config below uses the supervised split of [UltraChat 200K](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k). The processor reads the text conversations through the VLM data path; keep packing disabled for this model.

Create `command-a-plus-sft.yaml`.

```yaml
model_name_or_path: CohereLabs/command-a-plus-05-2026-bf16
model_revision: 5fb6fde5fd12ff89356aae552e11883bc49f069b
moe_balancing: bias_update_transient
router_balancing_rate: 1.0e-3

dataset:
- HuggingFaceH4/ultrachat_200k@train_sft
conversation_field: messages
test_size: 0.01
train_on_completions_only: true
assistant_message_template: "<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>"

expert_parallel_size: 8
save_sharded_ep: false
use_grouped_gemm: true
fp32_router: true
fp32_experts: false

attn_implementation: flash_attention_2
use_liger_kernel: false
packing: false
# 1024 peaks at 255.8 GiB of the B300's ~268 GiB. At 4096 DeepEP's elastic buffer has no
# room left to grow and hits a CUDA OOM a few steps in.
max_length: 1024
bf16: true

per_device_train_batch_size: 1
per_device_eval_batch_size: 1
gradient_accumulation_steps: 16
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
output_dir: /mnt/checkpoints/command-a-plus-ultrachat-ep8

logging_steps: 1
logging_first_step: true
report_to: wandb
remove_unused_columns: false
dataloader_num_workers: 2

use_peft: false
```

Launch eight processes.

```bash
halo launch sft command-a-plus-sft.yaml -n 8
```

Each rank owns 16 routed experts. Halo replicates the shared experts and averages their output with the routed output.

The model has no usable router auxiliary loss and no exportable bias slot, so plain `bias_update` raises. `moe_balancing: bias_update_transient` (with `router_balancing_rate: 1.0e-3`, as the shipped config sets) balances expert selection during training; exported checkpoints serve without the bias, so near-tied top-k picks can flip between trainer and server. Drop the line for unbalanced-but-serve-exact training.

## Add CP, TP, or ETP

Only EP8 is validated at full scale (see the note at the top); validate any
layout below with a short run before committing GPU-days to it.

Every layout below stays on the same eight ranks. On a single node the working EP sizes are
the whole job, 2, or 1 — an intermediate size such as 4 forms two four-rank DeepEP dispatch
groups whose combine barriers race FSDP2, and `ParallelismConfig` rejects it at config time.
EP+CP narrows that further: the EP group has to fill the NVLink domain, so the CP variant
keeps `ep_size` at 8. Attention TP leaves the dispatch-group width alone, so EP+TP keeps
the same 8, 2, or 1. Pure ETP is the exception — it turns EP off entirely.

Use CP when the sequence length causes attention memory pressure.

```yaml
context_parallel_size: 2
```

EP8 and CP2 use the same eight ranks. Keep the process count at eight.

```bash
halo launch sft command-a-plus-sft.yaml -n 8
```

Use TP when the dense attention weights need more sharding.

```yaml
tensor_parallel_size: 2
```

EP8 and TP2 also use the same eight ranks.

```bash
halo launch sft command-a-plus-sft.yaml -n 8
```

Use pure ETP when each local expert is too large. This mode keeps all experts and shards each expert across eight GPUs.

```yaml
expert_parallel_size: 1
expert_tensor_parallel_size: 8
```

Do not combine LoRA with TP. Validate one topology before a long run.

## Run text inference

Load the gathered checkpoint with Transformers.

```python
import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer

path = "/mnt/checkpoints/command-a-plus-ultrachat-ep8"
tokenizer = AutoTokenizer.from_pretrained(path)
model = AutoModelForImageTextToText.from_pretrained(
    path,
    dtype=torch.bfloat16,
    device_map="auto",
)

messages = [{"role": "user", "content": "Summarize the causes of a failed database migration."}]
inputs = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
).to(model.device)

output = model.generate(
    **inputs,
    max_new_tokens=512,
    do_sample=True,
    temperature=0.6,
    top_p=0.95,
)
reply = tokenizer.decode(output[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
print(reply)
```

Use `AutoProcessor` when the dataset or request contains images.

Serve the gathered checkpoint with Halo's vLLM image on the host, not inside the
training container; it listens on port 8000. vLLM 0.26.0 registers the family, and
its loader reads the gathered save's fused expert pair directly, with nothing to
unfuse. The compose service mounts only the HuggingFace cache, so add
`- /mnt/checkpoints:/mnt/checkpoints:ro` under the `vllm-server` `volumes:` to serve
a checkpoint from disk.

```bash
docker pull public.ecr.aws/whitecircle/halo:vllm-0.26.0
docker tag public.ecr.aws/whitecircle/halo:vllm-0.26.0 vllm-server:0.26.0

VLLM_MODEL=/mnt/checkpoints/command-a-plus-ultrachat-ep8 \
VLLM_CUDA_DEVICES=0,1,2,3 VLLM_TP=4 \
  docker compose -f docker-compose.vllm.yml up vllm-server
```

## Train an attention LoRA adapter

Add this block to the SFT configuration.

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
output_dir: /mnt/checkpoints/command-a-plus-ultrachat-lora
```

Keep EP enabled if the base model needs expert sharding. Keep TP disabled for LoRA.
Like the non-EP layouts, LoRA has no full-scale validation run for this family —
validate before a long run.

## Continue with GRPO

Offline GRPO works from the SFT checkpoint today: it trains on pre-generated
completions and needs no rollout server, so the standard offline configuration applies
unchanged. `packing` is an SFT-only field; the GRPO configs declare none, so a
`packing:` key there fails to parse.

Online and environmental GRPO are refused at construction for this family:
`EPCohere2MoELayer` declares `_supports_weight_sync = False`, because no NCCL weight
sync has been validated against a serving engine for Cohere2 MoE. vLLM 0.26.0 ships the
architecture — including a fused `experts.gate_up_proj` load path — so the missing
piece is a validated end-to-end sync, not engine support. Until that run exists,
continue with offline methods.

## Sources

- [Command A+ model card](https://huggingface.co/CohereLabs/command-a-plus-05-2026-bf16)
- [Halo Cohere2 MoE model notes](../../agent-docs/models/cohere2-moe.md) ↗
- Halo Cohere2 SFT example: `examples/sft/cohere2_moe/command-a-plus-ultrachat-ep.yaml`
