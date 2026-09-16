# Rollout Servers

On-policy RL needs the model to generate while it trains. Halo does that in a **separate inference container** —
vLLM or SGLang — that the trainer talks to over HTTP and pushes fresh weights into over NCCL. The training process
never imports either engine, so the two stacks stay independent and you start, stop and scale them yourself.

[Online GRPO (RLVR)](training-methods/online-grpo.md) and
[Async GRPO with Environments](training-methods/async-grpo-environments.md) both need one running before you launch.
Every other method trains without it.

One rule has no exception: **the server must own GPUs no trainer rank uses.** A process cannot NCCL-broadcast to
itself, so the split is made with `CUDA_VISIBLE_DEVICES` on both sides, and an overlap hangs at group formation.

## Start a vLLM server

The image is built from `Dockerfile.vllm`, or pulled:

```bash
make build-vllm
# or: docker pull public.ecr.aws/whitecircle/halo:vllm-0.26.0
#     docker tag public.ecr.aws/whitecircle/halo:vllm-0.26.0 vllm-server:0.26.0

VLLM_MODEL=Qwen/Qwen3-30B-A3B VLLM_CUDA_DEVICES=6,7 VLLM_TP=2 \
    docker compose -f docker-compose.vllm.yml up -d vllm-server

curl -s localhost:8000/health
```

`docker-compose.vllm.yml` already passes the flags RL depends on, so these are the variables you normally set:

| Variable | Default | What it does |
| --- | --- | --- |
| `VLLM_MODEL` | `Qwen/Qwen3-0.6B` | Hub id or a local checkpoint path |
| `VLLM_CUDA_DEVICES` | `7` | The server's GPUs. Must not overlap `TRAINER_CUDA_DEVICES` |
| `VLLM_PORT` | `8000` | Bound on the host; it drives the serve command, the health check and the banner |
| `VLLM_TP` | `1` | Tensor-parallel size, paired with the device list |
| `VLLM_GPU_MEM` | `0.85` | Fraction of each GPU for weights and KV cache; raise to `0.9` on dedicated GPUs |
| `VLLM_MOE_BACKEND` | `triton` | Leave it. Any other backend repacks expert weights and breaks the sync |
| `VLLM_TOOL_PARSER` | `hermes` | Per family (below); needed by every tool-calling environment |
| `VLLM_REASONING_PARSER` | unset | Required if the run sends a thinking budget |
| `VLLM_CHAT_TEMPLATE` | unset | The same `.jinja` file the trainer's `chat_template:` names, visible in this container |

Tool parsers are per model family, and a wrong one is silent: the calls come back as ordinary text, no tool ever runs,
and every episode scores zero. Qwen3.5/3.6 need `qwen3_xml` (hermes does not parse their XML calls); GPT-OSS needs the
bundled `gpt_oss_text` parser, loaded with `VLLM_TOOL_PARSER_PLUGIN`; the shipped Gemma 4 recipes serve on the default
`hermes`, as most families do. ReAct environments send no tool schema at all and want **no** parser. The
reasoning-parser and attention-backend variables a few families need are on
[Rollout Servers](../agent-docs/infrastructure/rollout-servers.md) ↗.

Serve **bf16 weights**, not a quantized checkpoint — a quantized engine stores transformed tensors that an in-place
weight update cannot reach. For GPT-OSS that means a BF16 conversion of the release, not the stock MXFP4 one.

## SGLang instead

Async GRPO can serve rollouts from SGLang with `rollout_backend: sglang`; Online GRPO is vLLM-only.

```bash
make build-sglang      # or pull public.ecr.aws/whitecircle/halo:sglang-0.5.17 and set SGLANG_IMAGE to it
SGLANG_MODEL=Qwen/Qwen3-30B-A3B SGLANG_CUDA_DEVICES=7 \
    docker compose -f docker-compose.sglang.yml up -d
```

The variables mirror vLLM's — `SGLANG_MODEL` (required), `SGLANG_PORT` (`30000`), `SGLANG_CUDA_DEVICES`, `SGLANG_TP`,
`SGLANG_GPU_MEM`, `SGLANG_TOOL_PARSER` (`auto`, read off the chat template), `SGLANG_MOE_RUNNER_BACKEND` (`triton`,
the same repack gate) — plus `NCCL_CUMEM_ENABLE=1`, which the compose file sets and a hand-run container must too.

Weight sync needs **this repo's** SGLang image, not the upstream one: it aligns NCCL with the training image and
patches two loaders an online update has to reach. Use vLLM unless you need SGLang specifically. Which families each
engine can take an online update for differs, and the trainer refuses the pair at construction, naming the family and
the loader reason — the current lists are in [Supported Matrix](supported-matrix.md#rollout-engines).

## Where the server lives

On one machine, give the server the spare GPUs and the trainer the rest; both containers run with
`network_mode: host`, because the NCCL rendezvous uses an ephemeral port a bridge network would hide.

![Separate inference node: the trainer and its Ray actors on node 1, the rollout server on node 2 joined to the
trainer's NCCL group on port 51216 while actors post to it over HTTP, with the port, Ray and EFA settings listed
below](../agent-docs/assets/diagrams/multi_node_separate_inference.png)

The server can equally sit on its own node: the trainer binds the weight-sync store and the server's workers dial back
to it, while the Ray actors reach it over HTTP.

![Dedicated rollout nodes: one training node, two inference nodes with one rollout_server_configs entry each and its
own group port, and a GPU-less actor tier joined through ray_address posting round-robin across the
servers](../agent-docs/assets/diagrams/multi_node_dedicated_rollout.png)

At scale each inference node gets its own `rollout_server_configs` entry and `group_port`, and the environment actors
can run on GPU-less nodes joined to the same Ray cluster.

Across nodes, put the sync on the fabric rather than on TCP. On EFA hosts, layer the overlay after the base compose
file (`docker compose -f docker-compose.vllm.yml -f docker-compose.vllm.efa.yml up -d vllm-server`) and start the
trainer with `make ... EFA=1`. Both ends must be images built from this repo, since a mismatched fabric userspace
forms the group and then hangs on the first collective. Verify before training:

```bash
python scripts/profiling/weight_sync_transport.py --server-url http://<server>:8000 --backend vllm --expect efa
```

## How the trainer finds it

| Config | Method | Meaning |
| --- | --- | --- |
| `rollout_server_url` | async GRPO | one server, default `http://localhost:8000` |
| `rollout_server_configs` | async GRPO | a list of `{url, group_port}`, one per server; two or more make prefetch possible |
| `vllm_server_host` / `vllm_server_port` | online GRPO | the address the **trainer dials**, not a bind address |
| `vllm_group_port` | both | the weight-sync port, bound on the **trainer** host, one per server |

Group ports must be unique across servers even on different hosts, since all of them are bound on the trainer. If the
trainer's routable address is not on its default-route NIC, name it in the trainer's environment with
`VLLM_GROUP_HOST` (or `SGLANG_GROUP_HOST`), or per server with a `group_host` entry.

## What weight sync is

After an optimizer step the trainer streams the **whole model** into the running engine over NCCL and the engine
keeps serving with the new weights — no restart, no checkpoint on disk. The server is paused for the push: vLLM
freezes in-flight requests and resumes them, SGLang drops them and the rollout actor re-issues the turn.

It has one hard requirement: MoE models must be served with `--moe-backend triton` (`--moe-runner-backend triton` on
SGLang). The backends the engine picks automatically on Blackwell repack expert weights after loading, so an update
writes the canonical layout into a repacked buffer and the served model quietly stops matching the trainer. Nothing
errors — the trainer's engine-versus-policy log-ratio drifts negative. The compose files default to `triton`,
so leave it alone.

## When it goes wrong

| Sign | Cause and fix |
| --- | --- |
| 400 on every rollout, zero tokens back | A thinking budget the server cannot take: set `VLLM_REASONING_PARSER` **and** `VLLM_USE_V2_MODEL_RUNNER=0`, or drop `rollout_max_thinking_tokens` |
| The run finishes with a flat zero reward | Wrong tool-call parser on a tool-using environment: the calls came back as text (a missing one 400s instead) |
| `Could not bind the weight-transfer group port` | Another process holds it, or two servers share one — give each its own `group_port` and remove stale containers |
| Group formation stalls with the server never joining | A bridge network hides the rendezvous port: `network_mode: host` on both containers |
| The first collective hangs with both sides idle | The two containers drive different NCCL transports — pin `NCCL_SOCKET_IFNAME` off Docker's `veth` interfaces (the compose default) and use the same images and fabric recipe on both ends |
| `/health` is 200 but nothing generates | A trainer died mid-sync and left the engine paused. Resume it, or restart the container if the push had already started |

For throughput, run one engine per GPU rather than two half-memory engines, give each server container its own
cores, and leave prefix caching on for multi-turn environments.

## Go deeper

- [Rollout Servers](../agent-docs/infrastructure/rollout-servers.md) ↗ — every flag, the sync internals, measured
  transport rates, the full troubleshooting table.
- [Online GRPO (RLVR)](training-methods/online-grpo.md) ·
  [Async GRPO with Environments](training-methods/async-grpo-environments.md)
- [Supported Matrix](supported-matrix.md#rollout-engines) · [Clusters and Multi-Node](clusters.md) ·
  [Troubleshooting](troubleshooting.md)
