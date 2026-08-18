# Parallelism

Skip this page if the model already trains under plain data parallelism — `-n 8`
gives you FSDP2 sharding across 8 GPUs on its own.

Reach for the modes below when something doesn't fit: dense weights → TP,
MoE experts → EP or ETP, long sequences → CP.

| Need | Mode | Flag |
| --- | --- | --- |
| MoE experts sharded across GPUs | EP | `--expert_parallel_size=N` |
| Long context (sequence split across GPUs) | CP | `--context_parallel_size=N` |
| Dense weights sharded (attention + FFN) | TP | `--tensor_parallel_size=N` |
| MoE expert FFNs sharded | ETP | `--expert_tensor_parallel_size=N` |
| MoE + long context | EP + CP | both flags |
| MoE + attention sharding | EP + TP | both flags |
| Multi-node DP with less cross-node traffic | HSDP | `--use_hsdp` |

Data-parallel size is what's left over:
`data_parallel_size = world_size / max(cp_size, tp_size, expert_tp_size)`
(EP is orthogonal — EP ranks are also data-parallel ranks). The divisor is a
max rather than a product because no two of those three axes may exceed 1 in
the same run, so they never compound.

## Rules that save you a wasted run

Halo validates the layout at startup and rejects invalid shapes with an
explanation before touching the GPUs. The rules people actually hit:

- **TP+CP, TP+ETP, ETP+CP, and EP+TP+ETP are unsupported.** Attention TP and
  expert TP never combine — pick EP+TP *or* EP+ETP, never both.
- **Single-node EP must be one dispatch group**: `ep_size` equal to the GPU
  count, or 2. Something in between (say EP=4 on 8 GPUs) is rejected — that
  shape deadlocks the MoE routing collectives against FSDP2. For a 4-way expert
  split on 8 GPUs, `ep4 + etp2` is the validated shape: the gate keys on
  `ep_size × expert_tp_size`, which is 8 there, so it stays one dispatch group.
  `ep4 + tp2` is **not** — attention TP leaves that product at 4 and lands back
  on the same rejection.
- **TP and node-local EP can't leave the NVLink domain** — the node on a
  typical 8-GPU host, the whole rack on NVL72. EP *can* span domains with
  `--ep_scope=global` on a proper RDMA fabric — see [Clusters](clusters.md).
- **EP+CP needs node-local EP filling the whole domain**: `ep_scope` must stay
  node, and `ep_size × expert_tp_size` must equal the NVLink domain size
  (`ep8` on an 8-GPU node). Global-scope EP with CP is rejected.
- **HSDP is for the plain data-parallel path.** `--use_hsdp` shards within the
  NVLink domain and replicates across domains, so only the gradient all-reduce
  crosses the fabric ([Clusters](clusters.md)). It is rejected with EP, TP, and
  ETP, and does nothing on a single-domain job.
- **`ep_size=1` MoE experts are FSDP-sharded by default**
  (`fsdp_shard_ep1_experts`, default `true`), which makes the reduce-scatter
  their only gradient sync and frees memory that otherwise grows with the DP
  size. Turning it off is rejected under TP or CP — those paths shard the
  replicated experts unconditionally.
- **LoRA doesn't combine with TP**, and **QLoRA only runs on DDP/FSDP/CP.** EP
  and TP reject QLoRA outright, and so does the grouped-GEMM MoE path — a MoE
  model rejects it even under plain FSDP unless you set
  `use_grouped_gemm: false`. Online and environmental GRPO reject it in every
  mode: vLLM weight sync ships raw parameter storage, and packed 4-bit tensors
  corrupt the served policy.
- **CP only works for SFT and SMPO.** The other trainers need full-sequence
  quantities that don't survive sequence splitting.

Pipeline parallelism is not yet available: the `pipeline_parallel_size` knob
parses, but any value above 1 is rejected at config time in this release.

Per-model coverage (which family supports which mode) is the
[Supported Matrix](supported-matrix.md). The deep dives:
[Expert](../agent-docs/parallelism/expert-parallelism.md) ↗ ·
[Context](../agent-docs/parallelism/context-parallelism.md) ↗ ·
[Tensor](../agent-docs/parallelism/tensor-parallelism.md) ↗ ·
[Expert-Tensor](../agent-docs/parallelism/expert-tensor-parallelism.md) ↗ ·
[Multi-Node](../agent-docs/parallelism/multi-node.md) ↗.
