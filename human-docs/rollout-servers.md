# Rollout Servers

On-policy RL generates with the model while it trains. Halo runs generation in a
**separate inference container**, vLLM or SGLang. The trainer sends it requests
over HTTP and pushes fresh weights into it over NCCL. The training process never
imports either engine, so you start, stop and scale the server yourself.

[Online GRPO (RLVR)](training-methods/online-grpo.md) and
[Async GRPO with Environments](training-methods/async-grpo-environments.md) need
a running server before you launch. No other method uses one.

**The server must own GPUs that no trainer rank uses.** NCCL needs a distinct
device for every member of the sync group. Split the GPUs with `CUDA_VISIBLE_DEVICES` on both
sides; an overlap fails group formation at its 120 s deadline.

## Start a vLLM server

Build the image from `Dockerfile.vllm`, or pull it:

```bash
make build-vllm        # or pull public.ecr.aws/whitecircle/halo:vllm-0.32.0 and set VLLM_IMAGE to it

VLLM_MODEL=Qwen/Qwen3-30B-A3B VLLM_CUDA_DEVICES=6,7 VLLM_TP=2 \
    docker compose -f docker-compose.vllm.yml up -d vllm-server

curl -s localhost:8000/health
```

This server takes GPUs 6 and 7, so start the trainer with
`TRAINER_CUDA_DEVICES=0,1,2,3,4,5`; its default `0,1,2,3,4,5,6` overlaps GPU 6.
The compose file also needs `HALO_SCRATCH`, the trainer's large scratch volume,
even to start the server alone; set it once in the repo-root `.env`. Serve from
the server image of the same Halo release as the trainer, which refuses a vLLM
server older than 0.32.0.

`docker-compose.vllm.yml` already passes the flags RL depends on. You normally
set `VLLM_MODEL`, `VLLM_CUDA_DEVICES`, `VLLM_TP` to match the device list,
`VLLM_GPU_MEM`, `VLLM_TOOL_PARSER` (below), `VLLM_REASONING_PARSER` when the run
caps reasoning, and `VLLM_CHAT_TEMPLATE` to the trainer's `chat_template:` file, which
the compose file mounts under `/jinja-templates/`.
Every variable and its default:
[Rollout Servers → vLLM](../agent-docs/infrastructure/rollout-servers.md#vllm) ↗.

### Tool parsers

A wrong tool parser leaves tool calls as plain text, so no tool runs and every
episode scores alike. A run on a tool-using environment stops by its second round
holding an unparsed call. Set the parser per family:

- `qwen3_xml`: Qwen3.5/3.6 and Qwen3-Coder.
- `gpt_oss_text`: GPT-OSS, loaded with `VLLM_TOOL_PARSER_PLUGIN`.
- `glm47`: GLM-4.7-Flash.
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
make build-sglang      # or pull public.ecr.aws/whitecircle/halo:sglang-0.5.22 and set SGLANG_IMAGE to it
SGLANG_MODEL=Qwen/Qwen3-30B-A3B SGLANG_MODEL_DIR=/path/to/checkpoints SGLANG_CUDA_DEVICES=7 \
    docker compose -f docker-compose.sglang.yml up -d
```

The variables mirror vLLM's (`SGLANG_MODEL`, `SGLANG_PORT`, `SGLANG_CUDA_DEVICES`,
`SGLANG_TP`, `SGLANG_GPU_MEM`), with two differences:

- `SGLANG_MODEL_DIR` is required: the host directory holding a local
  `SGLANG_MODEL`, any existing one for a hub id.
- `SGLANG_TOOL_PARSER` reads the parser off the chat template by default. On a
  template it cannot read (the Qwen3 2507 releases, Ling-mini-2.0) that turns the
  parser off, which the trainer refuses on a tool-using environment. Set
  `SGLANG_TOOL_PARSER=qwen25` for Qwen3, `hermes` for Ling
  ([Tool parsers on SGLang](../agent-docs/infrastructure/rollout-servers.md#tool-parsers-on-sglang) ↗).

The compose file also sets `NCCL_CUMEM_ENABLE=1`; a container you start by hand
needs it too
([Rollout Servers → SGLang](../agent-docs/infrastructure/rollout-servers.md#sglang) ↗).

Weight sync needs **this repo's** SGLang image, and the trainer refuses any other.
It matches NCCL with the training image and patches the Gemma 4 router so a synced
router weight takes effect.

Which families each engine can update online differs. The scripts refuse an
unsupported pair at startup and name the reason; the lists are in the
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
halo run weight-sync-transport --server_url http://<server>:8000 --backend vllm --expect efa
```

## How the trainer finds it

Async GRPO names one server with `rollout_server_url`, or several with
`rollout_server_configs` (a list of `{url, group_port}`; two or more turn prefetch
on by default). Online GRPO dials `vllm_server_host` / `vllm_server_port`. The
weight-sync port binds on the **trainer** host: `rollout_group_port` for async
GRPO (a server entry without its own `group_port` takes it plus the entry's
index) and `vllm_group_port` for online GRPO
([fields and defaults](../agent-docs/reference/configuration-reference.md#asyncrolloutconfig) ↗).

- **Group ports are unique across servers**, even on different hosts, because
  all of them bind on the trainer. `rollout_sync_topology: single_group` instead
  forms one group over every server, sends each chunk once, and takes the one
  `rollout_group_port`
  ([how it fails](../agent-docs/infrastructure/rollout-servers.md#one-group-over-every-server) ↗).
- **Name the trainer's address** when the server cannot dial back to the one on
  the route to it (asymmetric routing on a multi-homed host): `VLLM_GROUP_HOST`
  (or `SGLANG_GROUP_HOST`) in the trainer's environment, or a `group_host` entry
  per server. The trainer must be able to bind it.
- **Keep the group port private.** It accepts unauthenticated connections until
  the trainer closes its communicator, so only trusted hosts should reach it.
- **NAT or a port mapping is refused.** `HALO_WEIGHT_SYNC_BIND_ALL=1` listens on
  every interface instead; set it only where no untrusted host can reach the port.

## What weight sync does

After an optimizer step the trainer streams the whole model into the running
engine over NCCL: no restart, no checkpoint on disk. Every server pauses for the
push. vLLM freezes in-flight requests and resumes them; SGLang drops them and the
rollout actor re-issues the turn.

A LoRA run on SGLang can send only the adapter instead: `rollout_adapter_sync: true`
pushes the frozen base once, then loads the small adapter after each step without
pausing the server. Launch the server with `SGLANG_LORA_MAX_RANK` set to at least
`lora_r`; the trainer refuses a model or server it cannot serve this way and names
the reason ([Adapter sync](../agent-docs/infrastructure/rollout-servers.md#adapter-sync)).

MoE models must be served with `--moe-backend triton` (`--moe-runner-backend
triton` on SGLang), the compose default. The backends the engine picks on its
own repack expert weights after loading, so an update would write into the wrong
layout with no error. The trainer refuses a server that reports one of them.

## When it goes wrong

| Sign | Cause and fix |
| --- | --- |
| 400 on every rollout, zero tokens back | The server cannot take a thinking budget. Set `VLLM_REASONING_PARSER`, or drop the run's reasoning cap, and leave `VLLM_USE_V2_MODEL_RUNNER` unset. |
| `… the server parsed no call` by the second round | Wrong tool-call parser on a tool-using environment; the calls came back as text. A missing parser is refused at startup. If the parser is right and the model only writes malformed calls so far, raise `tool_parser_watch_rounds` or set it to `null`. |
| `No token carried a policy gradient …` | Every group tied or dropped: read the shares in the message. A launch that has not trained yet stops at `zero_gradient_round_limit` rounds in a row, so raise it for a very sparse reward; past its first trained round a streak only warns. |
| `… does not report the toolkit's image`, or another `cannot serve this run` at startup | A hand-run or upstream server. Serve from this repo's image and restate the flags the message names. |
| `Could not bind the weight-transfer group port` | Another process holds the port, two servers share one, or an outbound connection took it. Give each server its own `group_port`, reserve it or pick one outside `32768–60999`, and remove stale containers. |
| `/health` is 200 but nothing generates | A trainer died mid-sync and left the engine paused. Resume it, or restart the container if the push had started. |
| A server restarted during the run | Between syncs the trainer reconnects, warns and pushes the current weights. A server that fails mid-push stays paused and logs `RESTART the … server`: restart the container. |
| A run looks stuck after its server stopped | A slow failure, not a hang. A killed server raised 253–258 s after the fault on both engines, naming the server, at `rollout_connection_timeout: 180`. A frozen SGLang server (`docker pause`) took about 20 minutes at two trainer ranks. |

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
