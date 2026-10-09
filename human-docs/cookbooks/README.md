# Model Cookbooks

Each cookbook takes one MoE family from an empty host to a trained, served checkpoint. The
[Supported Matrix](../supported-matrix.md) is the authority: where a cookbook disagrees, the matrix wins.

| Cookbook | Recipe model | GPUs | Online RL |
| --- | --- | --- | --- |
| [GPT-OSS](halo-gpt-oss-cookbook.md) | `unsloth/gpt-oss-20b-BF16` | 8 | vLLM, SGLang |
| [Qwen3 MoE](halo-qwen3-moe-cookbook.md) | `Qwen/Qwen3-30B-A3B-Instruct-2507` | 4 | vLLM, SGLang |
| [Qwen3.5 / Qwen3.6 MoE](halo-qwen3.5-qwen3.6-moe-cookbook.md) | `Qwen/Qwen3.5-35B-A3B`, `Qwen/Qwen3.6-35B-A3B` | 8 | vLLM, SGLang |
| [GLM-4.7-Flash](halo-glm-4.7-flash-cookbook.md) | `zai-org/GLM-4.7-Flash` | 8 | vLLM, SGLang |
| [Gemma 4 MoE](halo-gemma4-moe-cookbook.md) | `google/gemma-4-26B-A4B-it` | 8 | vLLM, SGLang |
| [Mistral 4 MoE](halo-mistral4-moe-cookbook.md) | `mistralai/Mistral-Small-4-119B-2603` | 8 | No |
| [Laguna 2.1](halo-laguna-2.1-cookbook.md) | `poolside/Laguna-S-2.1`, `poolside/Laguna-XS-2.1` | 4, or 1 for XS | vLLM |
| [LFM-2 MoE](halo-lfm2-moe-cookbook.md) | `LiquidAI/LFM2.5-8B-A1B`, `LiquidAI/LFM2-24B-A2B` | 2, or 4 for 24B | vLLM, SGLang |
| [ZAYA1](halo-zaya1-cookbook.md) | `Zyphra/ZAYA1-8B` | 1 or 8 | No |
| [Command A+](halo-command-a-plus-cookbook.md) | `CohereLabs/command-a-plus-05-2026-bf16` | 8 | No |

Every cookbook has the same sections: Support, Recipes, Settings that matter, Limits, Export and serve,
and Reference. The GPU counts are the ones the recipes run on, B300s unless a page says otherwise. On
H100 or H200, use the Hopper image. Those GPUs have far less memory than a B300, so a recipe may need
more GPUs or a shorter `max_length` there.

## Start the training container

Every training, inference and conversion command runs inside this container. Point `HALO_SCRATCH` at a
large volume first. `/mnt` is not always large, so check with `df -h`.

```bash
git clone https://github.com/whitecircle/halo
cd halo
cp .env.example .env    # fill in HF_TOKEN, WANDB_API_KEY, and OPENROUTER_API_KEY for code-contests GRPO
export HALO_IMAGE=public.ecr.aws/whitecircle/halo:blackwell   # :hopper on H100 / H200
export HALO_SCRATCH=/path/to/large/volume
docker pull "$HALO_IMAGE"
mkdir -p "$HALO_SCRATCH/hf" "$HALO_SCRATCH/checkpoints" "$HALO_SCRATCH/tmp"
docker run --rm -it --gpus all --network host --ipc=host --shm-size=128g \
  --ulimit memlock=-1 --ulimit stack=67108864 --env-file .env \
  -e HF_HOME=/data/hf -e HF_DATASETS_CACHE=/data/hf/datasets \
  -e TMPDIR=/data/tmp -e HALO_DATA_ROOT=/data \
  -v "$(pwd)":/workspace -v "$HALO_SCRATCH":/data -w /workspace \
  "$HALO_IMAGE" bash
```

Inside the container, `/data` is `$HALO_SCRATCH`. A run that writes `/data/checkpoints/<run>` leaves its
output in `$HALO_SCRATCH/checkpoints/<run>` on the host.

The recipes override config fields on the command line, so the shipped YAML files stay untouched. Write
each override as `--field=value`, and separate list items with commas
(`--lora_target_modules=q_proj,v_proj`). Fields such as `dataset` and `rewards` can't be set this way:
copy the YAML and edit them there. Edit the existing line rather than adding a second one, because a
repeated key fails to parse.

Every layout in these cookbooks passes Halo's startup checks at the GPU count shown. The rule people hit
most often: on one eight-GPU node, pure EP must be 8, 2 or 1, and a 4-way expert split is `ep4 + etp2`.
The full rules are in [Parallelism](../parallelism.md).

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

## Serve from the host

Start rollout and serving containers on the host from the repo root, not inside the training container,
and give them GPUs the trainer does not use. Run this setup once in each host shell that starts a
server:

```bash
cd halo
export HALO_SCRATCH=/path/to/large/volume   # the volume the training container mounts
docker pull public.ecr.aws/whitecircle/halo:sglang-0.5.17
docker pull public.ecr.aws/whitecircle/halo:vllm-0.26.0
docker tag public.ecr.aws/whitecircle/halo:vllm-0.26.0 vllm-server:0.26.0
export SGLANG_IMAGE=public.ecr.aws/whitecircle/halo:sglang-0.5.17
export SGLANG_MODEL_DIR="$HALO_SCRATCH"   # SGLang mounts it read-only at the same path
export HF_HOME="$HALO_SCRATCH/hf"         # both servers share the training container's hub cache
```

SGLang takes host paths (`$HALO_SCRATCH/checkpoints/<run>`). The vLLM service mounts only the Hugging Face
cache. Add `- ${HALO_SCRATCH}:/data:ro` under the `vllm-server` service's `volumes:` in
`docker-compose.vllm.yml`, then give vLLM the same `/data` paths the trainer uses. vLLM answers a
request for any other model name with a 404, so the served path must match the trainer's `model_name`,
which defaults to its `model_name_or_path`.

The compose files already pass the MoE backend that weight sync needs (`--moe-backend triton` and
`--moe-runner-backend triton`), so leave it alone. A run with `routing_replay: rollout` also needs
`VLLM_ENABLE_R3=1` or `SGLANG_ENABLE_R3=1` set where you run compose. Plain serving works on any SGLang
0.5.17 image, but weight sync needs this repo's server images. Parsers, ports and the other variables
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
  first ([Code Contests](../../agent-docs/training-methods/grpo/environments/code-contests.md#dataset) ↗). Their
  `audit` judge vetoes a solved episode's credit on six checks for hacks and tool misuse. It calls
  OpenRouter with `OPENROUTER_API_KEY` from `.env`. Remove that reward term to train without a judge.
- **Sandbox.** Code environments run programs the policy writes. The default `local` sandbox lets them
  read the container's environment, `.env` secrets included. Set `sandbox_backend` under
  `environment_kwargs` to `remote`, which also needs a server URL in `sandbox_url` or `HALO_SANDBOX_URL`,
  or to `bubblewrap`, which needs extra container rights
  ([Async GRPO with Environments](../training-methods/async-grpo-environments.md)).

Families without a shipped GRPO recipe start from
`examples/grpo/environmental/environmental-grpo-template.yaml`. Copy it, set `dataset`,
`environment_type` and `rewards` for your task, and pass the rest as overrides. Each cookbook gives the
exact server and trainer commands.
