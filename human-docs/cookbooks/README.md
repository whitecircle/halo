# Model Cookbooks

Each cookbook takes one MoE family from an empty host to a trained, served checkpoint. The
[Supported Matrix](../supported-matrix.md) is the authority: where a cookbook disagrees, the matrix wins.

| Cookbook | Recipe model | GPUs | RL engines |
| --- | --- | --- | --- |
| [GPT-OSS](halo-gpt-oss-cookbook.md) | `unsloth/gpt-oss-20b-BF16` | 8 | vLLM, SGLang |
| [Qwen3 MoE](halo-qwen3-moe-cookbook.md) | `Qwen/Qwen3-30B-A3B-Instruct-2507` | 4 | vLLM, SGLang |
| [Qwen3.5 / Qwen3.6 MoE](halo-qwen3.5-qwen3.6-moe-cookbook.md) | `Qwen/Qwen3.5-35B-A3B`, `Qwen/Qwen3.6-35B-A3B` | 8 | vLLM, SGLang |
| [GLM-4.7-Flash](halo-glm-4.7-flash-cookbook.md) | `zai-org/GLM-4.7-Flash` | 8 | vLLM, SGLang |
| [Gemma 4 MoE](halo-gemma4-moe-cookbook.md) | `google/gemma-4-26B-A4B-it` | 8 | vLLM, SGLang |
| [Mistral 4 MoE](halo-mistral4-moe-cookbook.md) | `mistralai/Mistral-Small-4-119B-2603` | 8 | No |
| [Laguna 2.1](halo-laguna-2.1-cookbook.md) | `poolside/Laguna-S-2.1`, `poolside/Laguna-XS-2.1` | 8, or 1 for XS | vLLM |
| [LFM-2 MoE](halo-lfm2-moe-cookbook.md) | `LiquidAI/LFM2.5-8B-A1B`, `LiquidAI/LFM2-24B-A2B` | 2, or 4 for 24B | vLLM, SGLang |
| [ZAYA1](halo-zaya1-cookbook.md) | `Zyphra/ZAYA1-8B` | 1 or 8 | No |
| [Command A+](halo-command-a-plus-cookbook.md) | `CohereLabs/command-a-plus-05-2026-bf16` | 8 | No |
| [Nemotron 3.5 Super VL](halo-nemotron-3.5-super-cookbook.md) | `nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16` | 8 | No |

Every cookbook has the same sections: Support, Recipes, Settings that matter, Limits, Export and serve,
and Reference. The GPU counts are the ones the recipes are written for, B300s unless a page says otherwise. On
H100 or H200, use the Hopper image. Those GPUs have far less memory than a B300, so a recipe may need
more GPUs or a shorter `max_length` there.

## What ran on 1.2.0

Short runs on the full checkpoints covered these SFT recipes: Gemma 4, GLM-4.7-Flash, Ling 2.0, both ZAYA1
recipes, Qwen3.5-35B-A3B at 33,000 tokens, and on two 8-GPU machines gpt-oss-20b and the two pipeline
recipes. They also covered gpt-oss-20b's classification, self-distillation, embedding, offline GRPO, DPO, SMPO
and reward recipes and Gemma 4's DPO, SMPO and reward recipes. All four Nemotron 3.5 Super VL recipes ran on the
text tower, and its image recipe on the composite. The Nemotron 3.5 Lightning recipe ran for 20 steps. Both
Qwen3.6-35B-A3B MTP recipes ran, the joint one for 5 steps.

The Laguna S, LFM2.5-8B, Mistral Small 4, DeepSeek-V4 LoRA and Command A+ recipes have not run at their current
shape on the full-size model. Their memory fit is computed (Command A+'s comes from an earlier shape) and their
speed unmeasured, so try a short run first.

## Start the training container

Every training, inference and conversion command runs inside the training container: start it as in
[Installation → Start a container](../installation.md#5-start-a-container), with two changes. Mount your
large volume at `/data` (`-v "$HALO_SCRATCH":/data`, with `HF_HOME=/data/hf`,
`HF_DATASETS_CACHE=/data/hf/datasets`, `TMPDIR=/data/tmp` and `HALO_DATA_ROOT=/data`), and add
`--network host` so the trainer reaches the rollout servers. `/mnt` is not always large, so check with
`df -h` first. Load-bearing flags: [Docker](../../agent-docs/infrastructure/docker.md#standard-training-launch) ↗.

Inside the container, `/data` is `$HALO_SCRATCH`. A run that writes `/data/checkpoints/<run>` leaves its
output in `$HALO_SCRATCH/checkpoints/<run>` on the host.

The recipes override config fields on the command line, so the shipped YAML files stay untouched. Write
each override as `--field=value`, and separate list items with commas
(`--lora_target_modules=q_proj,v_proj`). Fields such as `dataset` and `rewards` can't be set this way:
copy the YAML and edit them there. Edit the existing line rather than adding a second one, because a
repeated key fails to parse.

Every layout in these cookbooks passes Halo's startup checks at the GPU count shown. The rule people hit
most often, EP on one eight-GPU node, is in [Parallelism](../parallelism.md#rules-that-save-you-a-wasted-run).

## Smoke-test a checkpoint

A gathered save (`save_sharded_ep: false`, the default) is a standard Hugging Face checkpoint. Each
cookbook names the model class to load it with. The rest is the same for every family:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

path = "/data/checkpoints/<run>"
tokenizer = AutoTokenizer.from_pretrained(path)
model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, device_map="auto")

messages = [{"role": "user", "content": "Explain expert parallelism in three sentences."}]
inputs = tokenizer.apply_chat_template(
    messages, add_generation_prompt=True, return_dict=True, return_tensors="pt"
).to(model.device)
output = model.generate(**inputs, max_new_tokens=256)
print(tokenizer.decode(output[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True))
```

A LoRA run saves an adapter, not a full checkpoint. Load it with
`PeftModel.from_pretrained(base, path)` or merge it first;
`AutoModelForCausalLM.from_pretrained` does not load an adapter directory under
PEFT 0.18.1. A run with `merge_expert_lora_on_save: true` saves the merged checkpoint instead.

## Serve from the host

Start rollout and serving containers on the host from the repo root, not inside the training container,
and give them GPUs the trainer does not use. Run this setup once in each host shell that starts a
server:

```bash
cd halo
export HALO_SCRATCH=/path/to/large/volume   # the volume the training container mounts
docker pull public.ecr.aws/whitecircle/halo:sglang-0.5.22
docker pull public.ecr.aws/whitecircle/halo:vllm-0.32.0
export SGLANG_IMAGE=public.ecr.aws/whitecircle/halo:sglang-0.5.22
export VLLM_IMAGE=public.ecr.aws/whitecircle/halo:vllm-0.32.0
export SGLANG_MODEL_DIR="$HALO_SCRATCH"   # SGLang mounts it read-only at the same path
export HF_HOME="$HALO_SCRATCH/hf"         # both servers share the training container's hub cache
```

Both servers mount `jinja-templates/` at `/jinja-templates`, the path a chat template takes. SGLang takes
host paths for checkpoints (`$HALO_SCRATCH/checkpoints/<run>`). The vLLM service mounts only the Hugging
Face cache for them. Add `- ${HALO_SCRATCH}:/data:ro` under the `vllm-server` service's `volumes:` in
`docker-compose.vllm.yml`, then give vLLM the same `/data` paths the trainer uses. vLLM answers a
request for any other model name with a 404, so the served path must match the trainer's `model_name`,
which defaults to its `model_name_or_path`.

The compose files already pass the MoE backend that weight sync needs (`--moe-backend triton` and
`--moe-runner-backend triton`), so leave it alone. A run with `routing_replay: rollout` (experimental) also needs
`VLLM_ENABLE_R3=1` or `SGLANG_ENABLE_R3=1` set where you run compose. Plain serving works on the upstream
SGLang 0.5.22 image, but weight sync needs this repo's server images. Parsers, ports and the other variables
are in [Rollout Servers](../rollout-servers.md).

## Before a GRPO run

GRPO in these cookbooks is async GRPO with environments. Ray actors run episodes against a rollout
server, and the trainer pushes new weights into that server after each step.

- **Split the GPUs.** The server and the trainer can't share a GPU. Fence the trainer with
  `CUDA_VISIBLE_DEVICES` and give the server the rest.
- **Match the NCCL transport.** Both ends must use the same transport, or the weight sync hangs at its
  first collective. On a host with no RDMA fabric, add
  `-e NCCL_IB_DISABLE=1 -e NCCL_NET=Socket -e NCCL_SOCKET_IFNAME=^docker,veth` to the training container,
  which matches the compose defaults. On an EFA host, put both ends on the EFA overlay instead: a trainer
  left at `Socket` sends every collective over TCP and breaks DeepEP
  ([Servers on other nodes](../../agent-docs/infrastructure/rollout-servers.md#servers-on-other-nodes-efa) ↗).
- **Code-contests recipes.** Their `dataset` is a placeholder under `your-org/`. Build the problem pool
  first ([Building a pool](../../agent-docs/training-methods/async-grpo/environments/tasks/code-contests/datasets.md#building-a-pool) ↗).
  Their `audit` judge vetoes or prices hacks and tool misuse
  ([Rewards](../training-methods/async-grpo-environments.md#rewards)). It calls OpenRouter with
  `OPENROUTER_API_KEY` from `.env`, and a judge outage trains nothing. Remove that reward term to train
  without a judge.
- **Sandbox.** Code environments run programs the policy writes. The default `local` sandbox lets them
  read the container's environment, `.env` secrets included. Set `sandbox_backend` under
  `environment_kwargs` to `remote`, which also needs a server URL in `sandbox_url` or `HALO_SANDBOX_URL`,
  or to `bubblewrap`, which needs extra container rights
  ([Async GRPO with Environments](../training-methods/async-grpo-environments.md)).

Families without a shipped GRPO recipe start from
`examples/grpo/environmental/environmental-grpo-template.yaml`. Copy it, set `dataset`,
`environment_type` and `rewards` for your task, and pass the rest as overrides. Each cookbook gives the
exact server and trainer commands.
