# Multi-Node Parallelism

Multi-node training spans EP, CP, TP, and DP across machines connected by InfiniBand/RDMA.
`ParallelismConfig` (`src/distributed/parallelism_config.py`) validates every combination against
the [supported-axis allowlist](README.md#supported-combinations); this page adds the locality rules —
which dimensions must stay NVLink-local and which may span domains.

Runnable launches, SLURM, and troubleshooting live in [Launch Recipes](launch-recipes.md).

## Scope rules

| Dimension | Scope | Constraint |
|-----------|-------|------------|
| EP | node-local (`ep_scope="node"`) or cross-domain (`"global"`, needs RDMA) | — |
| TP | always NVLink-local | `tp_size <= nvlink_domain_size`, divides it evenly |
| ETP | always NVLink-local | `expert_tp_size` divides the domain |
| CP | always NVLink-local | `cp_size <= nvlink_domain_size` and divides it (Ulysses all-to-all is bandwidth-bound) |
| PP | — | not yet available in this release ([Pipeline Parallelism](pipeline-parallelism.md)) |
| DP | automatic | `stage_world_size / max(tp_size, cp_size, expert_tp_size)` |

EP is **orthogonal** to DP — it routes tokens to experts and returns them to their originating
ranks, so EP never reduces DP.

**NVLink domain vs OS node:** on a standard cluster the NVLink domain is the OS node (≤8 GPUs), so
`nvlink_domain_size == gpus_per_node` and "node-local" means the OS node. On GB200/GB300 NVL72 the
fabric spans the rack (up to 72 GPUs across ~18 OS nodes); set `NVLINK_DOMAIN_SIZE` to unlock
NVLink-wide TP/CP/EP — see [NVL72](#gb200gb300-nvl72-multi-node-nvlink).

| Term | Definition |
|------|------------|
| `ep_size` | Number of distinct expert subsets |
| `ep_group_size` | Full EP process group = `ep_size * expert_tp_size` (auto-computed) |
| `stage_world_size` | `world_size / pp_size` — one pipeline stage's rank block |
| `data_parallel_size` | `stage_world_size / max(tp_size, cp_size, expert_tp_size)`; drives DataLoader sharding |
| `nvlink_domain_size` | GPUs reachable over NVLink (= `gpus_per_node`, or `NVLINK_DOMAIN_SIZE` on NVL72) |

Gradient-sync scope follows the mode — the world, the EP group, or the PP stage plus one reduce along
the chain for the grad norm ([below](#gradient-synchronization)).

## Node-local vs cross-node EP

| Mode | EP groups | Communication |
|------|-----------|---------------|
| Node-local (`ep_scope="node"`) | Within each NVLink domain (consecutive ranks) | NVLink |
| Cross-node (`ep_scope="global"`) | Span every NVLink domain, contiguous block per domain | NVLink + RDMA |

Cross-node EP uses the **column-block** layout (`src/distributed/group_layout.py`): each group spans
every domain with a contiguous device block per domain, and the group-relative rank is
`rdma_rank * members_per_domain + nvl_rank` — what DeepEP's internode transport expects. Contiguity
is mandatory because DeepEP's intranode P2P kernel addresses CUDA IPC peers by device offset.

```text
world=16 (2 domains of 8), ep_group_size=8:
  EP group 0: [0, 1, 2, 3,  8,  9, 10, 11]
  EP group 1: [4, 5, 6, 7, 12, 13, 14, 15]
Single global group (ep_group_size == 16): [0 .. 15]
```

A group that cannot split into equal contiguous per-domain blocks is rejected at config time. The
reachable widths follow from that rule: `ep_group_size = num_domains × d`, where `d` (the members
per domain) divides `nvlink_domain_size` — on 4 domains of 8 that admits 4, 8, 16 and 32 only, and
with many small domains the narrowest cross-node group is `num_domains` wide. When
`ep_group_size < stage_world_size` the EP groups are data-parallel replicas (`num_ep_groups > 1`);
that case — and node-local EP across domains — routes through the deferred sync below. The **single
global group** (`ep_group_size == stage_world_size`) needs no cross-replica deferral: every
collective already spans the same ranks.

### Deferred cross-replica sync

**Every multi-EP-group topology defers**, single-node ones included. `EPConfig.defer_grad_sync` is
`num_ep_groups > 1` — or more than one PP rank block — minus the FSDP-managed-expert case
(`fsdp_shard_ep1_experts` at `ep_group_size == 1`, where FSDP2's reduce-scatter over the DTensor
experts is already the sole sync and deferring on top would double-sync it). It carries no
node-count term: single-domain `ep2` (four groups on 8 GPUs) and `ep2+tp2` defer exactly as
node-local `ep8×2` does. Under a deferred topology **no in-backward EP grad hook registers at all** —
expert and router alike — and every average runs in one post-backward sweep
(`DistributedTrainerMixin._sync_deferred_expert_grads`).

Two things keep the hooks out of it. A post-accumulate hook fires only where a grad accumulated, so a
rank whose dispatch delivered no tokens for a layer never enters the collective its replicas are
already waiting in. And an in-backward all-reduce of different membership races each group's
intra-group DeepEP combine — rank-inconsistent collective order deadlocks. The sweep contributes
every param structurally instead (a missing grad is zero-filled). The hook that survives on a
single-group EP run carries no collective: it only divides by `world_size / expert_tp_size`.

`EPConfig.is_deferred_dp` sits on top and is still **multi-node** — `num_ep_groups > 1`,
`ep_group_size > 1`, more than one NVLink domain, no expert-TP. (Attention TP needs no term of its
own: `ParallelismConfig` rejects multi-domain multi-group EP+TP at config time.) It does not gate the
sweep. It decides that FSDP shards the non-expert params over `process_group` (the EP group) instead
of the DP world, which is what adds the third leg below.

The sweep lands every grad at the `/world_size` DP average over its rank block — the whole job, or
the stage's block under PP:

- expert shards: `all_reduce(SUM)` over the `expert_replica_group`, then
  `/(world_size / expert_tp_size)`. The ETP factor drops out because expert-TP partners hold slices
  of one expert and consume the same batch, so they are not DP replicas; at `expert_tp_size == 1`
  the divisor is just `world_size`. With one EP group per rank block (a PP stage) there is no replica
  group and only the divide runs;
- router, replicated EP submodules and plain non-EP params: `all_reduce(AVG)` over the DP scope —
  the world, or the stage's rank block under PP;
- non-expert FSDP shards, **`is_deferred_dp` only**: `all_reduce(AVG)` over the replica group, since
  the reduce-scatter averaged them within the EP group alone. Everywhere else it already spanned the
  full DP scope, so they are left untouched.

`ep_group_size`-way FSDP × `num_ep_groups`-way replica average = `world_size`-way DP. The sweep is
enforced once per optimizer step — its expert `SUM / world_size` leg is not idempotent, so a second
pass raises rather than halving every expert gradient.

Single-domain multi-group EP with dispatch groups wider than 2 ranks stays hard-blocked rather than
deferred: that rejection is about FSDP2's DP-wide reduce-scatter racing the narrower DeepEP combine,
which moving the cross-replica average out of the backward does not address.

## Configuration matrix

**Single node (8 GPUs):**

| Config | EP | CP | ETP | DP | Note |
|--------|----|----|-----|----|------|
| EP only | 8 | 1 | 1 | 8 | Full DP |
| EP only | 2 | 1 | 1 | 8 | 2-rank dispatch groups |
| EP+CP orthogonal | 8 | 8 | 1 | 1 | Same batch, seq split 8-way |
| EP+CP | 8 | 4 | 1 | 2 | 2 CP groups |
| Pure ETP | 1 | 1 | 8 | 1 | Experts replicated, FFN sharded 8-way |

**2 nodes × 8 GPUs (16 total):**

| Config | EP | CP | TP | ETP | PP | EP scope | DP |
|--------|----|----|----|-----|----|----------|----|
| Node-local EP+CP | 8 | 8 | 1 | 1 | 1 | node | 2 |
| Cross-node EP | 16 | 1 | 1 | 1 | 1 | global | 16 |
| Node-local EP | 8 | 1 | 1 | 1 | 1 | node | 16 |
| Pure ETP per node | 1 | 1 | 1 | 8 | 1 | node | 2 |
| EP+TP | 16 | 1 | 2 | 1 | 1 | global | 8 |
| EP+ETP per domain | 2 | 1 | 1 | 8 | 1 | global | 2 |

Every `PP > 1` column value is rejected at config time — pipeline parallelism is
[not yet available in this release](pipeline-parallelism.md).

Three shapes are narrower than they look; all are rejected at config time, not at runtime:

- **EP+TP across domains must be a SINGLE global EP group** (`ep_size == stage_world_size`,
  `ep_scope=global`). Cross-domain multi-group EP needs FSDP to shard non-expert params over the EP
  group (`is_deferred_dp`), while EP+TP shards them over the `(dp, tp)` mesh — the two contracts
  cannot both hold. `ep8+tp2` on 16 GPUs is rejected; `ep16+tp2` is the working shape.
- **EP+CP requires `ep_group_size == nvlink_domain_size`** — on 8-GPU nodes, `ep_size=8` exactly.
  `ep2+cp2` and `ep4+cp2` are rejected, as is cross-domain EP under CP (`ep_scope=global`).
- **EP+ETP across domains** needs a single dispatch group covering the job *and* exactly one ETP
  group per domain: `expert_tp_size == nvlink_domain_size` and `ep_size == domain count`, which
  keeps the ETP all-reduce on NVLink. On 2×8 that leaves `ep2+etp8`. Anything narrower — `ep2+etp4`,
  `ep4+etp2` — has `ep_group_size` below the world and is refused one rule earlier, by the
  multi-dispatch-group check (`world_size // ep_group_size > 1`): expert-TP keeps `is_deferred_dp`
  off, so FSDP2's DP-wide reduce-scatter would race the narrower DeepEP combine across domains. Both
  raise at either `ep_scope`.

> [!WARNING]
> **Single-domain pure EP needs a single dispatch group**
>
> Across domains `ep4` node-scope on 2×8 is accepted and routes through the deferred sweep above;
> on one domain the same shape is rejected before model loading —
> [Expert Parallelism](expert-parallelism.md#single-domain-multi-group-ep-races-and-hangs).

## Gradient synchronization

FSDP2 (`fully_shard`) syncs non-expert gradients; EP params go in `ignored_params`.

| Mode | Non-expert FSDP group | Expert gradients |
|------|----------------------|------------------|
| DP / CP / single-group EP / single-domain EP+CP | All ranks (1D, `world_size`) | The DeepEP combine already sums the group; the in-backward hook only rescales |
| Multi-group EP (no TP) — across domains (node-local EP, EP+CP) **or within one** (`ep_size=2`) | The **EP group** above one domain (`is_deferred_dp`), all ranks otherwise | One post-backward cross-replica sweep |
| EP+TP (DP>1) | `(dp, tp)` mesh — per-TP-position DP groups | The combine, plus the sweep once the world holds several EP groups |

**EP+TP groups:** TP params are DTensors with different shards per TP rank, so the world group would
mix gradient slices from different weight positions. Ranks sharing a TP position form a DP group:
`[0,8], [1,9], …` for 2 nodes × 8 GPUs at TP=8.

**CP:** syncs across all ranks whenever `cp_size > 1`, even at `data_parallel_size == 1` — each rank
holds partial gradients from its sequence chunk. FSDP's `all_reduce(SUM) / world_size` yields the
correct mean because `world_size = num_batches × cp_size`.

**HSDP (`--use_hsdp`):** the default 1D full-shard path sends every shard collective over RDMA. `--use_hsdp`
switches to a 2D `(dp_replicate, dp_shard)` mesh that shards within each NVLink domain and
replicates across domains, so only one gradient all-reduce crosses RDMA per step. See
[Data Parallelism → HSDP](data-parallelism.md#hsdp-hybrid-sharded-data-parallel). Rejected with EP,
TP, EP+TP, Expert-TP, and PP.

**Multi-EP-group expert grad norm:** with `num_ep_groups > 1`, expert grad norms sum within each EP
group, then average across groups (the `expert_replica_group` all-reduce divided by
`num_ep_groups`). Non-expert DTensor shards (TP) and full-world FSDP shards are reduced over their
own tiling group and counted once. The global norm is `sqrt(expert² + non_expert²)`, computed in
`_compute_global_grad_norm` (`src/trainers/mixins/grad_sync.py`).

## EP+TP mode

TP (DTensor) shards attention within each NVLink domain; EP (DeepEP) distributes experts; FSDP2
syncs DP across domains. Rules (`_validate_tp`): `tp_size <= nvlink_domain_size` and divides it;
`ep_size` is a multiple of `tp_size`; cross-domain EP under TP must be a **single** EP group
spanning the job; DP = `stage_world_size / tp_size`. Mechanism and load path:
[tensor-parallelism.md](tensor-parallelism.md#eptp-mode).

![Multi-node EP+TP: each node runs a node-local TP group (attention via DTensor over NVLink) while the experts form a single global EP group spanning both nodes (DeepEP all-to-all over InfiniBand); FSDP2 full-shard syncs the DP=2 non-expert gradients across nodes over InfiniBand](../assets/diagrams/ep_multi_node_layout.png){ .diagram-narrow }

Valid shapes: single node — `ep8/tp8` (DP 1) or `ep8/tp4` (DP 2, one 8-rank EP group spanning two TP
groups); 2×8 — `ep16` global with `tp8` (DP 2) or `tp4` (DP 4). At DP=1 there is no inter-node FSDP
sync.

## ParallelismConfig

```python
from src.distributed.parallelism_config import ParallelismConfig

config = ParallelismConfig(ep_size=8, tp_size=8, ep_scope="node")
```

| Parameter | Description | Default |
|-----------|-------------|---------|
| `ep_size` | Expert distribution size | 1 |
| `cp_size` | Context parallel size | 1 |
| `tp_size` | Tensor parallel size (attention) | 1 |
| `expert_tp_size` | Expert FFN TP size (MoE-only) | 1 |
| `pp_size` | Pipeline stages — only `1` accepted in this release ([Pipeline Parallelism](pipeline-parallelism.md)) | 1 |
| `ep_scope` | `"node"`, `"global"`, or `"auto"` (node-local when the EP group fits one domain) | `"auto"` |
| `world_size` / `gpus_per_node` / `nvlink_domain_size` | Override the detected topology | 0 (auto) |

The dataclass and the `--ep_scope` CLI flag share that default, so a hand-built config behaves like a
YAML-built one. Derived fields: `data_parallel_size`,
`num_ep_groups`, `requires_rdma` (True only when the EP group spans more than one NVLink domain).
`create_ep_config()` / `create_cp_config()` build the process groups.

## DeepEP inter-node communication

`DeepEPDispatcher` (`src/distributed/expert_parallel/dispatcher.py`) owns one lazy
`deep_ep.ElasticBuffer` per EP layer and selects the transport from the EP topology: intra-node uses
the NVLink (non-Gin) path; inter-node (`requires_rdma=True`) uses the NCCL Gin backend
(GPU-Initiated Networking / RDMA, e.g. AWS EFA). The dispatcher sets `EP_DISABLE_GIN` (`0`
inter-node, `1` intra-node) at buffer construction; an explicit env value is honored.

NVLink carries the all-to-all several times faster than any RDMA fabric ([measured busbw by
interconnect](../reference/gpu-training-theory.md#interconnect-tiers)), so prefer `ep_scope="node"`
unless you need maximum expert distribution — cross-node EP buys per-GPU memory (GPT-OSS 20B on
2 nodes × 8 B300: `ep16` global holds 2 experts per GPU against `ep8` node-local's 4).

## RDMA fabrics

Cross-node EP and inter-node FSDP gradient sync ride the node's RDMA NIC. The toolkit sets **no**
fabric env vars in code — NCCL (and libfabric, on EFA) read them from the process environment, so
they belong in the launcher. The NGC base ships the NCCL transport plugins for both AWS EFA
(libfabric + `aws-ofi-nccl`) and IB (HPC-X) in the `ldconfig` cache, and auto-selects: its
`/etc/shinit_v2` sets `NCCL_NET_PLUGIN=ofi` when it detects EFA hardware, else leaves it unset so
NCCL loads the HPC-X IB plugin. It fires only for a shell that sources it, so an EFA job sets the
variable itself rather than relying on it — that is what every `launcher-configs/skypilot/aws/**` task does. The
image also bakes `NCCL_IB_HCA=mlx5`.

| Fabric | Verify (on the host) | Launch env (beyond image defaults) |
|--------|--------|------------------------------------|
| InfiniBand / RoCE (Mellanox) | `ibstat` → `State: Active` | none; set `NCCL_IB_HCA` only for a non-default HCA |
| AWS EFA | `fi_info -p efa` | `NCCL_NET_PLUGIN=ofi`, `FI_PROVIDER=efa`, `FI_EFA_USE_DEVICE_RDMA=1`, `NCCL_PROTO=simple` |

- **InfiniBand and RoCE** share the NCCL IB path and run on the baked defaults.
- **AWS EFA is libfabric, not Mellanox.** The base bundles `aws-ofi-nccl` 1.17.3, which exports no
  `ncclGin`; the image builds a GIN-capable plugin over it and exposes it as `libnccl-gin.so`. EFA
  is unreliable with NCCL's LL/LL128 protocols, hence `NCCL_PROTO=simple`. The **host** supplies the
  EFA kernel driver and `/dev/infiniband`; pass them into the container.
- **Cross-node EP over EFA** additionally needs proxy GIN — `NCCL_GIN_TYPE=2` plus
  `--device /dev/gdrdrv`. It is bound by proxy-GIN per-operation latency, not bandwidth, so a
  narrower dispatch group beats a wider one; use it for MoE too large for one node.
  Measured EFA collective ceilings and the full GIN prerequisites:
  [DeepEP → AWS EFA](../infrastructure/deepep.md#expert-parallelism-over-aws-efa).
- **Multi-homed nodes** — set `NCCL_SOCKET_IFNAME` to the fast NIC (the ENA interface on AWS, `ib0`
  on IB clusters) when the default route is the management network.

## Verified topologies

What is validated by real multi-node runs, and what is covered only by simulation. Everything else
on this page is enforced in code and covered by one-host simulations (a `ParallelismConfig` built
with `gpus_per_node` / `nvlink_domain_size` below the world, so the domain-crossing branches execute
over NVLink) — which proves the rank math and the gradient algebra, not the fabric.

| Shape | Validated | Where |
|---|---|---|
| 2-node RDMA topologies (Blackwell) — cross-node EP `ep8`/`ep16`, the 8192 tokens/rank Gin dispatch ceiling, node-local `ep8×2` matching single-group `ep16` | ✅ | [DeepEP → AWS EFA](../infrastructure/deepep.md#expert-parallelism-over-aws-efa) |
| 2-node, 2/4/8 GPUs per node — plain FSDP2 DP, HSDP, node-local multi-group EP (deferred sync), cross-node EP, EP+TP, EP+CP, pure ETP, cross-node EP+ETP | ✅ | real multi-node runs over an EFA fabric |
| **Wider layouts** (4-node, 8-node, the 512-GPU layouts in [Large-Scale Scenarios](large-scale-scenarios.md)) | ❌ | rank math only — `ParallelismConfig` is exercised at world 8/16/32, no recorded run |
| **NVL72 / MNNVL rack-wide domains** | ❌ | simulated domain sizes only; see the warning below and [Scale & Limits](../reference/scale-and-limitations.md) |
| **InfiniBand/RoCE as a multi-node fabric** | ❌ | the recorded multi-node runs used EFA; the IB path is config guidance, not a measurement |
| **Multi-node weight sync for online / environmental GRPO** (`*_GROUP_HOST` group formation across hosts) | ❌ | stated at [Rollout Servers](../infrastructure/rollout-servers.md) |
| **Cross-node gathered EP save** on a shared filesystem | ❌ | hand-run recipe in `tests/gpu/parallelism/ep/test_ep_save_reload_roundtrip.py`; not exercised multi-node |

## GB200/GB300 NVL72 (multi-node NVLink)

An NVL72 rack spans the NVLink/NVSwitch fabric across up to 72 GPUs on ~18 OS nodes (4 GPUs each) as
a single NVLink domain with direct GPU-to-GPU P2P across OS-node boundaries (MNNVL, 130 TB/s
aggregate). TP, EP, and CP can run NVLink-wide with no RDMA.

Here the NVLink domain exceeds the OS node, so the two units diverge: `gpus_per_node`
(`LOCAL_WORLD_SIZE`, 4) keys filesystem coordination, while `nvlink_domain_size` keys node-local
TP/CP/EP/ETP grouping, EP's locality unit and `requires_rdma`. Set it to the NVLink partition size
(72 for a full rack, or the clique size; on Kubernetes the `nvidia.com/gpu.clique` partition):

```bash
export NVLINK_DOMAIN_SIZE=72
```

The toolkit then builds node-local EP/CP groups as contiguous blocks across the domain and sets
`requires_rdma=False` for EP within the domain. Leaving it unset caps node-local TP/CP/EP at
`gpus_per_node=4`, and warns if it detects an NVLink fabric anyway.

**The domain is the tiling unit, so every node-local width must divide it.** `tp_size`, `cp_size`,
`expert_tp_size` and a node-scope `ep_group_size` are each required to divide `nvlink_domain_size`
exactly (`_validate_tp`, `_validate_cp_locality`, `_validate_expert_tp`, `_validate_ep_group`) — groups are
contiguous rank blocks, and a non-dividing width straddles a domain boundary while `requires_rdma`
still reports False. At `NVLINK_DOMAIN_SIZE=72` the legal widths are the divisors of 72
(1, 2, 3, 4, 6, 8, 9, 12, 18, 24, 36, 72); 16, 32 and 64 are rejected at config time. Pure EP is
narrower still: on a single 72-GPU domain the racy-topology gate leaves only `ep_size=2` and
`ep_size=72` ([Recommended cell](large-scale-scenarios.md#recommended-cell)). On a rack whose
GPU count is not a power of two, declare the largest power-of-two divisor your model can use —
`NVLINK_DOMAIN_SIZE=64` over 16 of the rack's 18 compute trays — and leave the remainder idle.

Ranks disagreeing on `NVLINK_DOMAIN_SIZE` (a per-node drift builds different groups on different
ranks) are rejected by `ParallelismConfig` itself, before any fabric read.
`validate_nvlink_domain_against_fabric` (`src/distributed/nvlink.py`) then cross-checks
the declaration against hardware at config time. Both verdicts are per **domain**, taken from the
gathered clique ids and node widths so every rank reaches the same one: a domain block spanning more
than one fabric clique **raises**, and so does a block whose ranks all report no fabric while the
declared domain exceeds their node — the case no clique comparison can see, since every fabric-less
GPU reports the same sentinel (a mixed job, an NVL72 rack beside plain NVL8 trays, has both kinds).
A clique wider than the declared domain **warns**: node-local parallelism is capped below the
available NVLink width. If any rank cannot read its clique the check no-ops, keeping it rank-uniform.

Each rank reads its clique id out of process through `nvidia-smi -q -i <pci_bus_id>`, never through
NVML's Python binding — the binding's versioned out-param struct segfaults inside the C call on a
driver mismatch, killing every rank inside `ParallelismConfig.__post_init__`. Keying on the PCI bus
id rather than the device index keeps `CUDA_VISIBLE_DEVICES` masking from making a rank read another
GPU's clique.

**Prerequisites (must already run on the rack):**

| Component | Why |
|-----------|-----|
| NVIDIA Fabric Manager | NVSwitch routing / NVLink partition |
| IMEX service + channels (`/dev/nvidia-caps-imex-channels/`) | GPU-memory export/import for cross-OS-image NVLink P2P; jobs fail without it |
| NCCL ≥ 2.25.2 | MNNVL support; leave `NCCL_MNNVL_ENABLE` at its default (on). A domain-local EP group is classified intranode, so the dispatcher sets `EP_DISABLE_GIN=1` even when the domain spans OS nodes — export `EP_DISABLE_GIN=0` to A/B the Gin backend over MNNVL |
| NVSHMEM (`nvidia-nvshmem-cu13`) | DeepEP device linking only — not the V2 transport |

The first two are **enforced, not advisory**. Whenever `nvlink_domain_size > gpus_per_node`, every
rank of a live multi-rank job runs `check_mnnvl_prerequisites`, and the job raises (`ValueError`,
naming the first failing rank) if any rank has no IMEX channels, reports an NVLink fabric
registration other than `COMPLETED` (`nvidia-smi` "Fabric State"), or sees no fabric clique at all.
Without them the declaration promises cross-OS-node NVLink P2P that fails deep inside the first
collective instead. At or below the threshold nothing is checked; outside a live job the same
verdict only logs.

> [!WARNING]
> **DeepEP on NVL72**
>
> NVLink-wide EP across OS-node boundaries depends on the NCCL Gin / NVSHMEM build targeting
> MNNVL on Blackwell. Validate before relying on rack-wide EP all-to-all.

Each compute tray is an OS node with 4 GPUs: launch with `--nproc_per_node=4` and ~18 nodes per
rack. With SLURM use the `topology/block` plugin with one block per NVL72 domain and `--segment` to
keep a job in one rack.

| Goal | Layout |
|------|--------|
| Large MoE | Wide node-scope EP at a divisor of the declared domain — `ep_size=72` at `NVLINK_DOMAIN_SIZE=72`, or `ep_size=64` at `NVLINK_DOMAIN_SIZE=64` when the expert count is a power of two |
| Long sequences | CP up to the domain width, at a divisor of it |
| Dense / attention-heavy | TP at a divisor of the domain — 8, 9, 12, 18, 24 or 36 at domain 72; `tp_size=16` is rejected |
| Multi-rack | TP/EP inside each rack, DP across racks ([PP](pipeline-parallelism.md), the axis designed to cross racks, is not yet available) |
