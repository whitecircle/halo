---
name: parallelism
description: >-
  Recommend a VALID ParallelismConfig and REJECT unsupported EP/CP/TP/ETP/PP combos
  BEFORE a GPU run. Auto-fire whenever the user: asks how to configure or pick a
  parallelism mode (expert/context/tensor/expert-tensor/pipeline parallelism,
  EP/CP/TP/ETP/PP, DDP/FSDP/HSDP); sets or asks about --expert_parallel_size /
  --context_parallel_size / --tensor_parallel_size / --expert_tensor_parallel_size /
  --pipeline_parallel_size / --pipeline_split / --pipeline_schedule / ep_scope / ep_size /
  cp_size / tp_size / expert_tp_size / pp_size / nvlink_domain_size; asks "which parallelism
  for this MoE / dense model / seq len / 8-GPU node / multi-node" or how to compute
  data_parallel_size; or hits a parallelism error/hang (DeepEP combine deadlock,
  NCCL hang, "must divide", "node-local", "not supported", ep4 "DeepEP dispatch groups" rejection,
  ParallelismConfig raise). Wrong advice wastes multi-GPU runs — this is the gate.
allowed-tools:
  - Read
  - Grep
  - Glob
---

# Parallelism configuration gate

Recommend a **valid** `ParallelismConfig` (`halo/distributed/parallelism_config.py`) and **reject**
unsupported combos before the user burns GPU hours. Every rule lives once in the docs; read the owner
section before a verdict on any non-trivial combo:

- allowed axis sets, their DP size and each rejection's mechanism —
  [Supported combinations](../../agent-docs/parallelism/README.md#supported-combinations);
- locality, EP+TP / EP+CP / EP+ETP and model-shape rules —
  [Topology rules](../../agent-docs/parallelism/README.md#topology-rules);
- `data_parallel_size` — [Stacking the dimensions](../../agent-docs/parallelism/README.md#stacking-the-dimensions);
- trainer × CP/PP — [Trainer Compatibility](../../agent-docs/reference/trainer-architecture.md#trainer-compatibility);
- family × axis — [Compatibility matrix](../../agent-docs/models/README.md#compatibility-matrix);
- PP stage layout, families and refusals —
  [Pipeline Parallelism](../../agent-docs/parallelism/pipeline-parallelism.md#rejected-combinations);
- per-axis knob refusals — each axis page's *Limitations*.

The code is the **ultimate** authority: `halo/distributed/parallelism_rules.py` (`reject_unsupported_axes`,
then `reject_invalid_composition` and its `_reject_*`), and `parallelism_config_from_args`
(`halo/configs/parallelism_arguments.py`: `lowp_*` or `init_from_scratch` outside SFT, a trainer's
`SUPPORTS_CP` / `PP_REFUSAL`). When a doc, this skill or memory disagrees, read those files before
advising.

## How to pick a mode

1. **Dense (non-MoE) model?** EP/ETP do nothing — use **TP** (memory/large hidden),
   **CP** (long seq) or **PP** across nodes (step 5), else plain FSDP2 DDP (`dp_size = world_size`).
2. **MoE model?** Start with **EP** (orthogonal to DP, keeps full DP). For finer
   expert sharding add **ETP** (EP+ETP); EP+TP shards attention only, and TP+ETP is rejected.
   For long sequences on MoE use **EP+CP** (node-local EP only).
3. **Long sequence, OOM on activations?** add **CP** (Ulysses, node-local). CP
   reduces DP by `cp_size`.
4. **Weights/attention too big for one GPU?** **TP** (node-local DTensor). Plain FSDP2/DDP
   loads the whole model onto every GPU before sharding, so it cannot take a dense model past
   one GPU ([Data Parallelism](../../agent-docs/parallelism/data-parallelism.md#fsdp2-strategy-by-mode)).
   TP reduces DP by `tp_size`.
5. **Model too large for one node?** **PP** — the outermost axis, composing with **EP or pure
   ETP** only. Every stage is a whole number of NVLink domains of at least two ranks, so a run
   needs at least two domains; only the families with a PP verdict in the compatibility matrix
   split. A per-node (`DIST_OUTPUT_SHARED_FILESYSTEM=0`) save exports through
   `scripts/after_training/merge_pp_stages.py`.
6. Compute `dp_size` and sanity-check it's >= 1 and the global batch is what you intend.

## Worked dp_size examples (8 GPUs, 1 node, domain=8)

| Config | ep_group_size | dp_divisor | dp_size | Valid? |
|--------|:--:|:--:|:--:|--------|
| DDP | 1 | 1 | 8 | yes |
| ep_size=8 | 8 | 1 | 8 | yes (single intra-node group) |
| ep_size=2 | 2 | 1 | 8 | yes (2-rank groups) |
| ep_size=4 | 4 | 1 | 8 | **NO** — multi-group >2-rank pure EP, rejected at config time (use 2 or 8) |
| tp_size=8 | 1 | 8 | 1 | yes |
| cp_size=2 | 1 | 2 | 4 | yes |
| expert_tp_size=2 (ep_size=1) | 2 | 2 | 4 | yes (pure ETP) |
| ep_size=8 + cp_size=8 | 8 | 8 | 1 | yes — canonical EP+CP (`cp_size == ep_group_size == domain`) |
| ep_size=8 + cp_size=2 | 8 | 2 | 4 | yes — `cp_size` only has to divide the domain |
| ep_size=4 + cp_size=2 | 4 | 2 | 4 | **NO** — node-local EP+CP requires `ep_group_size == domain` |
| ep_size=8 + tp_size=8 | 8 | 8 | 1 | yes (ep_size multiple of tp_size, node-local) |
| ep_size=2 + expert_tp_size=2 | 4 | 2 | 4 | yes (EP+ETP, node-local) |
| ep_size=4 + expert_tp_size=2 | 8 | 2 | 4 | yes — the 4-way expert split on 8 GPUs |
| tp_size=4 + cp_size=2 | — | — | — | **NO** — TP+CP rejected |
| tp_size=2 + expert_tp_size=2 | — | — | — | **NO** — TP+ETP rejected (`ep_size=1`, so the axis set is `{tp, etp}`) |
| expert_tp_size=2 + cp_size=2 | 2 | 2 | — | **NO** — ETP+CP rejected |
| pp_size=2 | — | — | — | **NO** — a 4-rank stage is not a whole NVLink domain (PP needs 2+ nodes here) |

## Reject these

- TP+CP, TP+ETP (with or without EP), ETP+CP —
  [Supported combinations](../../agent-docs/parallelism/README.md#supported-combinations).
- `ep_size > 2` with `ep_group_size` short of a single NVLink domain (`ep4` or `ep4+tp2` on 8) —
  [Single-domain multi-group EP](../../agent-docs/parallelism/expert-parallelism.md#single-domain-multi-group-ep-races-and-hangs).
- A size not dividing the domain, EP+CP that is not node-local filling the domain, multi-group EP+TP
  or EP+ETP across domains, indivisible heads or experts —
  [Topology rules](../../agent-docs/parallelism/README.md#topology-rules).
- Every PP pairing outside EP or pure ETP, and every PP config, model and trainer refusal —
  [Pipeline → Rejected combinations](../../agent-docs/parallelism/pipeline-parallelism.md#rejected-combinations).
- CP or PP with a trainer that does not declare it —
  [Trainer Compatibility](../../agent-docs/reference/trainer-architecture.md#trainer-compatibility).
- HSDP or an FSDP shaping knob outside its modes —
  [Data Parallelism → Limitations](../../agent-docs/parallelism/data-parallelism.md#limitations).
- LoRA with TP, expert LoRA with ETP, offline GRPO adapters under CP —
  [TP](../../agent-docs/parallelism/tensor-parallelism.md#limitations),
  [ETP](../../agent-docs/parallelism/expert-tensor-parallelism.md#limitations),
  [CP](../../agent-docs/parallelism/context-parallelism.md#limitations) limitations.

When rejecting, cite the exact reason and offer the nearest valid alternative
(e.g. "ep_size=4 on 8 GPUs is rejected at config time → use ep_size=2, ep_size=8 (the whole domain), or
ep4+etp2; EP+TP does not help — attention TP leaves `ep_group_size` at 4"). Related skill: `data`
(sharded dataset ↔ `data_parallel_size`).
