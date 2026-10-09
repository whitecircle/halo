# Rollout Servers

On-policy RL generates with the model while it trains. Halo runs generation in a
**separate inference container**, vLLM or SGLang. The trainer sends it requests
over HTTP and pushes fresh weights into it over NCCL. The training process never
imports either engine, so you start, stop and scale the server yourself.

[Online GRPO (RLVR)](training-methods/online-grpo.md) and
[Async GRPO with Environments](training-methods/async-grpo-environments.md) need
a running server before you launch. No other method uses one.

**The server must own GPUs that no trainer rank uses.** A process cannot
NCCL-broadcast to itself. Split the GPUs with `CUDA_VISIBLE_DEVICES` on both
sides; an overlap hangs at group formation.

## Start a vLLM server

Build the image from `Dockerfile.vllm`, or pull it:

```bash
make build-vllm
# or: docker pull public.ecr.aws/whitecircle/halo:vllm-0.26.0
#     docker tag public.ecr.aws/whitecircle/halo:vllm-0.26.0 vllm-server:0.26.0

VLLM_MODEL=Qwen/Qwen3-30B-A3B VLLM_CUDA_DEVICES=6,7 VLLM_TP=2 \
    docker compose -f docker-compose.vllm.yml up -d vllm-server

curl -s localhost:8000/health
```

This server takes GPUs 6 and 7, so start the trainer with
`TRAINER_CUDA_DEVICES=0,1,2,3,4,5`; its default `0,1,2,3,4,5,6` overlaps GPU 6.

`docker-compose.vllm.yml` already passes the flags RL depends on. These are the
variables you normally set:

| Variable | Default | What it does |
| --- | --- | --- |
| `VLLM_MODEL` | `Qwen/Qwen3-0.6B` | Hub id or local checkpoint path |
| `VLLM_CUDA_DEVICES` | `7` | the server's GPUs; must not overlap `TRAINER_CUDA_DEVICES` |
| `VLLM_PORT` | `8000` | host port for the API and the health check |
| `VLLM_TP` | `1` | tensor-parallel size, matching the device list |
| `VLLM_GPU_MEM` | `0.85` | fraction of each GPU for weights and KV cache; `0.9` on dedicated GPUs, at most `0.80` with `isr_engine_reference` |
| `VLLM_MOE_BACKEND` | `triton` | leave it; any other backend breaks weight sync |
| `VLLM_TOOL_PARSER` | `hermes` | tool-call parser, set per family (below) |
| `VLLM_REASONING_PARSER` | unset | required if the run sends a thinking budget |
| `VLLM_CHAT_TEMPLATE` | unset | the same `.jinja` file as the trainer's `chat_template:`, visible in this container |

### Tool parsers

A wrong tool parser fails silently: tool calls come back as plain text, no tool
runs, and every episode scores zero. Set it per family:

- `qwen3_xml`: Qwen3.5/3.6 and Qwen3-Coder.
- `gpt_oss_text`: GPT-OSS, loaded with `VLLM_TOOL_PARSER_PLUGIN`.
- `glm45` / `glm47`: GLM-4.5 / GLM-4.7.
- `gemma4`, `lfm2`, `poolside_v1`, `step3p5`: Gemma 4, LFM-2, Laguna and
  Step-3.7 Flash.
- `hermes` (default): most others, Qwen3 included.
- No parser (`VLLM_TOOL_CALLING_FLAGS=`): ReAct environments, which send no tool
  schema.

The reasoning parsers and attention backends a few families need are in the
[reference](../agent-docs/infrastructure/rollout-servers.md#vllm) ↗.

### Serve bf16 weights

Serve bf16 weights, not a quantized checkpoint: a weight update cannot reach the
transformed tensors a quantized engine stores. For GPT-OSS that means a bf16
conversion of the release, not the stock MXFP4 one.

The trainers send every sampling field with each request, so the run config, not
the checkpoint's `generation_config.json`, decides how rollouts sample.

## SGLang instead

Async GRPO can use SGLang with `rollout_backend: sglang`. Online GRPO is
vLLM-only.

```bash
make build-sglang      # or pull public.ecr.aws/whitecircle/halo:sglang-0.5.17 and set SGLANG_IMAGE to it
SGLANG_MODEL=Qwen/Qwen3-30B-A3B SGLANG_CUDA_DEVICES=7 \
    docker compose -f docker-compose.sglang.yml up -d
```

The variables mirror vLLM's: `SGLANG_MODEL` (required), `SGLANG_PORT` (`30000`),
`SGLANG_CUDA_DEVICES`, `SGLANG_TP`, `SGLANG_GPU_MEM`, `SGLANG_TOOL_PARSER`
(`auto`, read from the chat template) and `SGLANG_MOE_RUNNER_BACKEND`
(`triton`, for the same reason as vLLM's). The compose file also sets
`NCCL_CUMEM_ENABLE=1`; a container you start by hand needs it too.

Weight sync needs **this repo's** SGLang image, not upstream's. It matches NCCL
with the training image and patches the GLM-4 and Gemma 4 routers so a synced
router weight takes effect.

Which families each engine can update online differs. The trainer refuses an
unsupported pair at construction and names the reason; the lists are in the
[Supported Matrix](supported-matrix.md#rollout-engines). Use vLLM unless you
need SGLang.

## Where the server lives

On one machine, give the server the spare GPUs and the trainer the rest. Run
both containers with `network_mode: host`: the server dials back to the
trainer's group port and then to a random NCCL port, and a bridge network
publishes neither.

![Separate inference node: the trainer and its Ray actors on node 1, the rollout server on node 2 joined to the
trainer's NCCL group on port 51216 while actors post to it over HTTP, with the port, Ray and EFA settings listed
below](../agent-docs/assets/diagrams/multi_node_separate_inference.png)

The server can also run on its own node. The trainer binds the weight-sync
group and the server's workers dial back to it; the Ray actors reach the server
over HTTP.

![Dedicated rollout nodes: one training node, two inference nodes with one rollout_server_configs entry each and its
own group port, and a GPU-less actor tier joined through ray_address posting round-robin across the
servers](../agent-docs/assets/diagrams/multi_node_dedicated_rollout.png)

At scale, each inference node gets its own `rollout_server_configs` entry and
`group_port`, and the actors round-robin across the servers. The environment
actors can run on GPU-less nodes joined to the same Ray cluster.

### Across nodes

Put the sync on the fabric, not on TCP. On EFA hosts, add the overlay after the
base compose file and start the trainer with `make ... EFA=1`:

```bash
docker compose -f docker-compose.vllm.yml -f docker-compose.vllm.efa.yml up -d vllm-server
```

SGLang has the same pair (`docker-compose.sglang.efa.yml`). Both ends must run
images built from this repo: a mismatched fabric stack forms the group and then
hangs on the first collective. Check the transport before training:

```bash
halo run weight-sync-transport --server-url http://<server>:8000 --backend vllm --expect efa
```

## How the trainer finds it

| Config | Method | Meaning |
| --- | --- | --- |
| `rollout_server_url` | async GRPO | one server; default `http://localhost:8000` |
| `rollout_server_configs` | async GRPO | a list of `{url, group_port}`, one per server; `enable_prefetch` needs two or more |
| `vllm_server_host` / `vllm_server_port` | online GRPO | the address the **trainer dials**, not a bind address |
| `vllm_group_port` | both | the weight-sync port, bound on the **trainer** host (default `51216`); a server entry without its own `group_port` takes `vllm_group_port` plus its index |

- **Group ports are unique across servers**, even on different hosts, because
  all of them bind on the trainer.
- **Name the trainer's address** if its routable address is not on the
  default-route NIC: `VLLM_GROUP_HOST` (or `SGLANG_GROUP_HOST`) in the trainer's
  environment, or a `group_host` entry per server.
- **Keep the group port private.** It listens on the advertised address only and
  accepts unauthenticated connections until the trainer closes its
  communicator, so only trusted hosts should reach it.
- **NAT is refused.** A trainer the server reaches through NAT or a port mapping
  is rejected. `HALO_WEIGHT_SYNC_BIND_ALL=1` listens on every interface instead;
  set it only where no untrusted host can reach the port.

## What weight sync does

After an optimizer step the trainer streams the whole model into the running
engine over NCCL. The engine keeps serving with the new weights, with no restart
and no checkpoint on disk.

The server pauses for the push. vLLM freezes in-flight requests and resumes
them; SGLang drops them and the rollout actor re-issues the turn. With several
servers, every push pauses all of them.

MoE models must be served with `--moe-backend triton` (`--moe-runner-backend
triton` on SGLang). The backends the engine picks on its own repack expert
weights after loading, so an update writes into the wrong layout and the served
model quietly diverges from the trainer. Nothing errors; the trainer's
engine-versus-policy log-ratio drifts negative. The compose files default to
`triton`.

## When it goes wrong

| Sign | Cause and fix |
| --- | --- |
| 400 on every rollout, zero tokens back | The server cannot take a thinking budget. Set `VLLM_REASONING_PARSER` **and** `VLLM_USE_V2_MODEL_RUNNER=0`, or drop `rollout_max_thinking_tokens`. Code contests also send a budget whenever `reasoning_effort` is set, as the shipped recipes do. |
| The run finishes with a flat zero reward | Wrong tool-call parser on a tool-using environment; the calls came back as text. A missing parser returns 400 instead. |
| `Could not bind the weight-transfer group port` | Another process holds the port, two servers share one, or an outbound connection took it. Give each server its own `group_port`, reserve it or pick one outside `32768–60999`, and remove stale containers. |
| `/health` is 200 but nothing generates | A trainer died mid-sync and left the engine paused. Resume it, or restart the container if the push had started. |

Group formation and first-collective hangs are in
[Troubleshooting](troubleshooting.md#rl-runs-vllm--sglang).

For throughput, run one engine per GPU rather than two half-memory engines, give
each server container its own CPU cores, and leave prefix caching on for
multi-turn environments.

## Go deeper

- [Rollout Servers](../agent-docs/infrastructure/rollout-servers.md) ↗: every
  flag, the sync internals, measured transport rates and the full
  troubleshooting table.
- [Online GRPO (RLVR)](training-methods/online-grpo.md) ·
  [Async GRPO with Environments](training-methods/async-grpo-environments.md)
- [Supported Matrix](supported-matrix.md#rollout-engines) ·
  [Clusters and Multi-Node](clusters.md) ·
  [Troubleshooting](troubleshooting.md)
