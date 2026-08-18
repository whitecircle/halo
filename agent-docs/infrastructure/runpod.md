# Multi-Node EP Training on RunPod InfiniBand Clusters

RunPod multi-node pods with InfiniBand provide the interconnect for cross-node EP training. There is no
RunPod tooling in the repo — this is a manual runbook.

| Topology | `--ep_scope` | Inter-node traffic | Use |
|----------|--------------|--------------------|-----|
| Node-local EP | `node` | Gradients only (NCCL AllReduce over IB) | Default — EP stays on NVLink |
| Cross-node EP | `global` | Expert all-to-all (RDMA) + gradients | Maximum expert distribution for large MoE; higher comm overhead |

`data_parallel_size = (world_size / pp_size) / max(cp_size, tp_size, expert_tp_size)`. With only EP active it reduces to
`world_size` — EP is orthogonal to DP. Theory: [Multi-Node Parallelism](../parallelism/multi-node.md).

## Prerequisites

- 2+ RunPod pods with InfiniBand, same data center / IB fabric, 8 GPUs each (H100/H200 SXM).
- A persistent volume mounted at `/workspace` (500 GB+) for weights, HF cache, checkpoints.
- The training image. Set the pod's Container Image to `public.ecr.aws/whitecircle/halo:hopper`
  (anonymous, no registry credentials), or push your own build to your own registry
  ([Docker Guide](docker.md), [AWS Auth](aws-auth.md)).
- HuggingFace token for gated models, optionally a WandB key.

The `halo:hopper` image ships everything the run needs, including the Mellanox NCCL defaults
([Docker](docker.md)). Override the CMD with `sleep infinity` — the default CMD prints a
banner and exits, which stops a RunPod pod.

## Network setup

Designate one pod as Node 0 and take its **IB-routable** IP as `MASTER_ADDR` (not the public SSH IP):

```bash
hostname -I | awk '{print $1}'        # IB-routable IP
ibstat                                # expect State: Active
ib_write_bw -d mlx5_0                 # server on Node 0
ib_write_bw -d mlx5_0 <NODE0_IP>      # client on Node 1 — expect close to the port's line rate
                                      # (HDR 200 Gb/s ≈ 24 GB/s); an order of magnitude below it
                                      # means the fabric is not what the pod advertises
```

Install diagnostics if missing: `apt-get install -y infiniband-diags perftest libibverbs-dev rdmacm-utils`
— `ibstat`/`perfquery` come from `infiniband-diags`, `ib_write_bw` from `perftest`. The training image
installs none of them; it ships the NCCL fabric transports, not the fabric diagnostics.

Set only the environment-specific variables; the image covers the rest:

| Variable | Value | Purpose |
|----------|-------|---------|
| `NCCL_SOCKET_IFNAME` | `eth0` / `ib0` / `ibp*` | Override if NCCL auto-detects the wrong interface |
| `DIST_NCCL_TIMEOUT_MINUTES` | `30` (default) | NCCL collective watchdog in minutes; raise for very large cross-node collectives |
| `DIST_SHARED_FILESYSTEM` | `0` | RunPod pods have independent storage |
| `HF_HOME` | `/workspace/hf_cache` | HuggingFace cache on the volume |

`init_distributed` passes `DIST_NCCL_TIMEOUT_MINUTES` as the `timeout=` of `init_process_group` and pins it
as the default for every later `dist.new_group()`, so EP/CP/TP subgroups and the gathered-checkpoint
all_gather inherit it. A bare `NCCL_TIMEOUT` does nothing — PyTorch does not read it. It does not bound
DeepEP's own dispatch/combine barrier ([DeepEP](deepep.md)).

## Per-pod setup

```bash
cd /workspace
git clone https://github.com/whitecircle/halo.git code && cd code
hf auth login --token $HF_TOKEN
wandb login $WANDB_API_KEY  # optional

python -c "import torch; print(torch.__version__, torch.version.cuda)"
python -c "import deep_ep; print('DeepEP OK')"
python -c "from flash_attn_interface import flash_attn_func; print('FA3 OK')"

# Pre-download the model on every pod to avoid download races during torchrun init
export HF_HOME=/workspace/hf_cache
python -c "from huggingface_hub import snapshot_download; snapshot_download('your-org/model-name')"
```

## EP topologies (2 nodes × 8 GPUs)

| Setup | EP | CP | TP | Scope | DP | Flags |
|-------|----|----|-----|-------|----|-------|
| Node-local EP (default) | 8 | 1 | 1 | `node` | 16 | `--expert_parallel_size=8 --ep_scope=node` |
| Long sequence (8K+) | 8 | 8 | 1 | `node` | 2 | `+ --context_parallel_size=8` |
| Cross-node EP | 16 | 1 | 1 | `global` | 16 | `--expert_parallel_size=16 --ep_scope=global` |
| EP+TP (max memory efficiency) | 16 | 1 | 8 | `global` | 2 | `+ --tensor_parallel_size=8` |

- **Node-local EP**: each node runs its own EP group over NVLink; only gradient AllReduce crosses nodes.
  Simplest to debug.
- **Cross-node EP**: one EP group spans all 16 GPUs; the expert all-to-all uses the NCCL Gin backend over IB
  RDMA. Each GPU holds fewer experts → more memory for activations.
- **EP+TP**: TP groups are contiguous rank blocks, so `tp_size` must **divide** the NVLink domain. Cross-node EP+TP must be a single
  global EP group (`ep_size == world`) — a multi-domain multi-group EP+TP shape is rejected at config time.

CLI arguments come from `DistributedArguments`: `--expert_parallel_size` (default `1`), `--ep_scope`
(`auto` | `node` | `global`, default `auto`), `--context_parallel_size`, `--tensor_parallel_size`,
`--save_sharded_ep`, `--fp32_router`.

## Launch

Run on each pod simultaneously (use `tmux`). Both pods must use identical `--expert_parallel_size`,
`--ep_scope`, `--master_addr`, `--master_port`, `--nnodes`; only `--node_rank` differs.

```bash
#!/bin/bash
set -e
export DIST_NCCL_TIMEOUT_MINUTES=30
export HF_HOME=/workspace/hf_cache
export DIST_SHARED_FILESYSTEM=0       # RunPod pods have independent storage

NODE_RANK=${1:?Usage: bash launch.sh <node_rank> <master_ip>}
MASTER_ADDR=${2:?Usage: bash launch.sh <node_rank> <master_ip>}
MASTER_PORT=${3:-29500}

cd /workspace/code
torchrun \
    --nnodes=2 --node_rank=$NODE_RANK --nproc_per_node=8 \
    --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
    scripts/training/sft.py examples/sft/gptoss/gptoss-20b-multinode-ep.yaml \
    --expert_parallel_size=8 --ep_scope=node \
    --output_dir=/workspace/checkpoints/run-001
```

Run `bash launch.sh 0 <master_ip>` on Node 0 and `bash launch.sh 1 <master_ip>` on Node 1. Swap the
parallelism flags from the topology table; add `--resume_from_checkpoint=true` to resume.

The YAML sets `save_only_model: true` (warm restart, smaller checkpoints); `save_on_each_node` needs
no key, since `DIST_SHARED_FILESYSTEM=0` above auto-forces it.

## Checkpoints and resume

Formats, save modes, and resume mechanics are owned by
[Checkpoints & Resume](../reference/checkpoints.md). The RunPod-specific parts:

- **Gathered (default)** is the multi-node choice. It all-gathers expert weights and writes sharded
  safetensors loadable with `from_pretrained()`, and every pod's save rank writes its own copy.
- **`--save_sharded_ep` raises at trainer construction on a multi-node non-shared FS.** Per-rank shards are
  keyed by global rank and scatter across the pods' local disks (the index and non-expert params land only
  on the rank-0 pod), so no single pod holds a mergeable checkpoint and
  `scripts/after_training/merge_ep_shards.py` takes one input dir. Where it does run (single node, or a shared FS) it buys write bandwidth — every rank writes its own
  slice — at the price of a checkpoint that must be merged before resume or serving, and it saves no host
  memory.
- **Resume needs no rsync.** With `DIST_SHARED_FILESYSTEM=0` the mixin forces `save_on_each_node=True`, so
  each pod's local rank 0 writes a complete checkpoint to its own disk; re-run `torchrun` on every pod with
  `--resume_from_checkpoint=true`. To resume *trained* EP/CP weights (not just trainer state), point
  `model_name_or_path` at the gathered checkpoint directory —
  [why](../reference/checkpoints.md#resume-by-parallelism-mode).
- `WANDB_RUN_ID` is hashed from `output_dir` + launch timestamp and broadcast from rank 0, so it is
  consistent across a run's ranks but not across re-runs. Export the same `WANDB_RUN_ID` on every pod
  to resume into one WandB run.

## Monitoring

```bash
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv -l 5
watch -n 2 "perfquery -x | grep -E 'XmtData|RcvData'"   # IB throughput counters
```

Only rank 0 prints training output. For comm issues set `NCCL_DEBUG=INFO` and `NCCL_DEBUG_SUBSYS=ALL`, then
look for `NCCL INFO NET/IB` (connection up) and `NCCL WARN` (fallbacks/timeouts).

## Troubleshooting

**NCCL timeout / pods can't connect.** Use the IB-routable `MASTER_ADDR`. Test the control port from Node 1:
`python -c "import socket; s=socket.socket(); s.settimeout(5); s.connect(('$MASTER_ADDR', 29500)); print('OK')"`.
Confirm both pods are on the same IB fabric; raise `DIST_NCCL_TIMEOUT_MINUTES=60` for large collectives.

**`No module named 'deep_ep'`.** The image includes it — a missing module means an outdated image; re-pull the
current tag, or rebuild and push your own. An in-container rebuild needs the full CUDA-13 build env ([DeepEP](deepep.md)).

**InfiniBand not active** (`ibstat` shows `State: Down`). Verify the pod was provisioned with IB; install
`infiniband-diags`; contact RunPod support. Without IB, fall back to node-local EP (`--ep_scope=node`) — only
gradient sync crosses the network.

**Host OOM during gathered save.** Every gathered save streams — the EP path one MoE layer at a
time, the dense/CP/TP paths one decoder layer at a time
([Checkpoints](../reference/checkpoints.md#expert-parallelism-ep)) — so each pod's save rank peaks
at the replicated non-expert params plus one pending shard (`save_max_shard_size`, default `5GB`),
not a model's worth. Lower `save_max_shard_size` if that peak is still too high; `--save_sharded_ep`
does not help — it is MoE-only and holds more host memory per node, not less.

**`ProcessGroup not initialized` / mismatch.** Ensure both pods use identical parallelism args and start
within the torchrun rendezvous timeout (10 min default). Kill stale processes:
`pkill -f torchrun; pkill -f sft.py`.

**Cross-node EP slow.** Check RDMA bandwidth (`ib_write_bw -d mlx5_0 <REMOTE_IP>`). Cross-node dispatch is
per-operation latency-bound, not bandwidth-bound ([DeepEP](deepep.md#expert-parallelism-over-aws-efa)):
switch to node-local EP so only gradient sync crosses the network, or raise batch size / gradient
accumulation to amortize comm.

**Resume "no checkpoint found".** Confirm `checkpoint-*` and its `trainer_state.json` exist on **both** pods
at the same path.

## Related pages

- [Multi-Node Parallelism](../parallelism/multi-node.md) · [Expert Parallelism](../parallelism/expert-parallelism.md)
- [Checkpoints & Resume](../reference/checkpoints.md)
- [DeepEP Installation](deepep.md) · [Docker Guide](docker.md)
