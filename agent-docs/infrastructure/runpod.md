# Multi-Node EP Training on RunPod InfiniBand Clusters

A manual runbook for multi-node EP on RunPod pods with InfiniBand; the repo ships no RunPod tooling.

| Topology | `--ep_scope` | Inter-node traffic | Use |
|----------|--------------|--------------------|-----|
| Node-local EP | `node` | Gradients only (NCCL AllReduce over IB) | Default — EP stays on NVLink |
| Cross-node EP | `global` | Expert all-to-all (RDMA) + gradients | Maximum expert distribution for large MoE; higher comm overhead |

With only EP active, `data_parallel_size = world_size` — EP is orthogonal to DP
([Stacking the dimensions](../parallelism/README.md#stacking-the-dimensions)). Theory:
[Multi-Node Parallelism](../parallelism/multi-node.md).

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

`DIST_NCCL_TIMEOUT_MINUTES` is the one watchdog setting ([Launch Recipes → Environment variables](../parallelism/launch-recipes.md#environment-variables) lists the groups it covers and the DeepEP barrier it does not).
A bare `NCCL_TIMEOUT` does nothing — PyTorch does not read it.

## Per-pod setup

```bash
cd /workspace
git clone https://github.com/whitecircle/halo.git code && cd code
hf auth login --token $HF_TOKEN
wandb login $WANDB_API_KEY  # optional

python -c "import torch; print(torch.__version__, torch.version.cuda)"
python -c "import deep_ep; print('DeepEP OK')"
python -c "from flash_attn_interface import flash_attn_func; print('FA3 OK')"

# Optional: pre-fetch the model on every pod (with DIST_SHARED_FILESYSTEM=0 each pod's local rank 0
# fetches it at load otherwise)
export HF_HOME=/workspace/hf_cache
python -c "from huggingface_hub import snapshot_download; snapshot_download('openai/gpt-oss-20b', ignore_patterns=['*/*'])"
```

## EP topologies (2 nodes × 8 GPUs)

| Setup | EP | CP | TP | Scope | DP | Flags |
|-------|----|----|-----|-------|----|-------|
| Node-local EP (default) | 8 | 1 | 1 | `node` | 16 | `--expert_parallel_size=8 --ep_scope=node --use_grouped_gemm=true` |
| Long sequence (8K+) | 8 | 8 | 1 | `node` | 2 | `+ --context_parallel_size=8 --packing=false` (CP refuses packing) |
| Cross-node EP | 16 | 1 | 1 | `global` | 16 | `--expert_parallel_size=16 --ep_scope=global` |
| EP+TP (max memory efficiency) | 16 | 1 | 8 | `global` | 2 | `+ --tensor_parallel_size=8` |

- **Node-local EP**: each node runs its own EP group over NVLink; only gradient AllReduce crosses nodes.
  Simplest to debug.
- **Cross-node EP**: one EP group spans all 16 GPUs; the expert all-to-all uses the NCCL Gin backend over IB
  RDMA. Each GPU holds fewer experts → more memory for activations.
- **EP+TP**: a single global EP group across nodes; the rules: [Topology rules](../parallelism/README.md#topology-rules).

The flags and their defaults: [Configuration reference → ParallelismConfig](../reference/configuration-reference.md#parallelismconfig).

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
    --expert_parallel_size=8 --ep_scope=node --use_grouped_gemm=true \
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
  on the rank-0 pod).

    No single pod holds a mergeable checkpoint, and `scripts/after_training/merge_ep_shards.py` takes one
    input dir. Where it does run (single node, or a shared FS) it buys write bandwidth, since every rank
    writes its own slice. The price is a checkpoint that must be merged before resume or serving, and it
    saves no host memory.

- **Resume needs no rsync.** With `DIST_SHARED_FILESYSTEM=0` the mixin forces `save_on_each_node=True`, so
  each pod's local rank 0 writes a complete checkpoint to its own disk; re-run `torchrun` on every pod with
  `--resume_from_checkpoint=true`.

    Trained EP/CP weights reload with no manual step: the training scripts load the policy from the
    resumed checkpoint, so leave `model_name_or_path` at the base, which reference and teacher models
    still resolve ([why](../reference/checkpoints.md#resume-by-parallelism-mode)).

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

Pods that never rendezvous, InfiniBand down, a resume that finds no checkpoint on one pod, and the
rest: [Troubleshooting](../reference/troubleshooting.md). Cross-node dispatch is per-operation
latency-bound, so a slow cross-node EP run is a case for node-local EP or a larger batch
([DeepEP](deepep.md#expert-parallelism-over-aws-efa)).

## Related pages

- [Multi-Node Parallelism](../parallelism/multi-node.md) · [Expert Parallelism](../parallelism/expert-parallelism.md)
- [Checkpoints & Resume](../reference/checkpoints.md)
- [DeepEP Installation](deepep.md) · [Docker Guide](docker.md)
