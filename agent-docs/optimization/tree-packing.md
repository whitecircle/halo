# Tree Packing

Tree packing stores a prefix that several rows share once. Rows that start identically — the same
instructions and document, then a different question and answer — become branches of one tree inside a
packed row. A branch attends the shared prefix and itself, never a sibling, so every example gets the
logits, language-model loss and gradients it gets when packed flat, at fewer tokens.

![Three rows that share a prefix, packed flat with the prefix repeated per row and packed as a tree with the prefix once and one branch per row, each branch continuing the prefix's positions; in the tree row a token attends the prefix and its own branch, never a sibling](../assets/diagrams/tree_packing_layout.png)

SFT only (`scripts/training/sft.py`), raw (not pre-processed) text datasets: GPT-OSS, Qwen3 and
Qwen3.5/3.6 (dense and MoE) on `flash_attention_4` or `flash_attention_2`, and GLM-5 Next on its SDPA
backend.

```yaml
packing: true
tree_packing: true
tree_share_markers:
- "\nQuestions:\n"
tree_share_requires: "\nDocument:\n"
```

`tree_share_markers` is required: rows whose message before the final turn is identical up to and including
one of its strings are examined together. `tree_share_requires` shares a row only if its text occurs before
the marker, which leaves rows that put the long part *after* the marker alone instead of sharing a short
header. Types and defaults of every `tree_*` option: [SFTScriptArguments](../reference/configuration-reference.md#sftscriptarguments).

## What is shared

The markers only decide which rows are *examined* together. Each split is ordered by a digest of the
text up to the marker, so rows over one document become adjacent. Every row is then rendered and
tokenized whole, exactly as the flat path does it, and a run of adjacent rows shares what its token
sequences have in common. A row whose tokenization differs one token into the "shared" text shares one
token less; nothing is assumed from the text.

- Sharing stops one token before a row's first trained token, so the token a completion is predicted
  from sits in the branch with it. Two rows with one prompt and different answers share all but the
  last prompt token.
- Rows may share at several depths: the document, then the start of what follows it.
- A group longer than `max_length` splits into several trees that each repeat the prefix. A row
  longer than `max_length` on its own is dropped, as on the flat path.
- A row with no marker, or with the marker more than once, shares only with an exact duplicate.
- A shared span under 16 tokens is repeated per row, not split off.

The digest order replaces the dataset's row order, in both splits. The trainer still shuffles packed
rows every epoch.

Measured on Halo 1.2.0. The gain is the share of tokens that sit in shared prefixes: a dataset where half the rows share a long
prefix with others packs into about 1.4× fewer rows. A tree row costs about what a flat row costs where
sliding-window layers dominate (GPT-OSS). Qwen3-30B-A3B, where every layer is a full layer in the
`split` form, pays 10% per step, and Qwen3.5-35B-A3B on FA2 (`gather`) about 7%. That cost grows with the
shared-prefix length and fan-out: on a corpus sharing 6k-token prefixes about 1.8× deep it is 1.41× per step,
still about 1.3× more examples per second. Building the trees adds dataset maps on top of tokenization.

## Attention

The collator (`halo/data/collators/tree_packing.py`) concatenates the batch's packed rows, recovers every
tree from two per-token columns (position along the token's own path, depth of its node) and puts the
index tensors each layer needs under the `tree_attention` batch key (`halo/models/attention/tree_plan.py`).
`halo/models/attention/tree.py` installs the readers on the family's own modeling module; nothing else in
the model changes. A flash softmax layer runs the stock packed path on a batch with no branch.

- **Sliding-window layers** gather, per tree node, the last `sliding_window - 1` prefix tokens ahead
  of the node's own. FlashAttention aligns the causal mask bottom-right when a sequence has more
  keys than queries, which is "the prefix, then yourself causally".
- **Full layers, `split`** (default): the stock packed call runs over every node, and one
  rectangular pass per parent lets its children attend the prefix. The two partial softmaxes merge
  through their log-sum-exps. No key is copied per branch, and tokens outside any branch are
  bit-identical to flat packing. Needs `flash_attention_4`, the one backend that returns the
  log-sum-exp with a gradient.
- **Full layers, `gather`**: one key sequence per node, the whole prefix ahead of it. Runs on FA2 as
  well; the gathered keys grow with fan-out × prefix length, so a layer whose row gathers more than
  2^19 keys (`GATHER_KEYS_PER_CALL` in `halo/models/attention/tree_plan.py`) attends in several calls of
  whole nodes. That bounds the GQA backward, which expands key and value gradients to every query head.
- **GatedDeltaNet and KDA layers** (Qwen3.5/3.6, GLM-5 Next): the causal convolution runs each node
  behind the last `linear_conv_kernel_dim - 1` tokens of its path, and the delta-rule recurrence runs
  one tree depth at a time, each node starting from its parent's final state. Each scan input is
  permuted into depth order once and every depth reads a view of it, so the backward is one inverse
  permutation per input instead of a gather's zero-filled, scattered gradient per depth. The plan
  carries each depth's varlen bounds as host integers as well, which reach fla as `cu_seqlens_cpu`, so
  building its chunk indices costs no host sync per depth and layer.
- **DeepSeek sparse attention layers** (GLM-5 Next) read the plan in their own forward
  (`halo/models/attention/tree_sparse.py`), never at the attention interface:
  - **Pools.** The indexer's pools of `index_kpool` tokens run along each token's own path: a pool is
    a token and its predecessors there. The pool straddling a branch point is one pool per branch.
  - **Selection.** A query's candidates are the pools whose last token precedes it on its path, plus
    the incomplete pool it sits in. The indexer takes no gradient, so it is scored a chunk of queries
    at a time; the stock form holds `[T, heads, T / index_kpool]` fp32 scores. A tie at the cut goes to
    the earlier pool on the path.
  - **Attention.** It runs on the MLA latent, with the key projection folded into the query and the
    value projection applied after, so a query reads only its selected latent rows. Under autocast it
    reads the folded query and the latent in the autocast dtype and keeps its fp32 scores outside
    autocast. The fold reads `kv_b_proj`'s weight and takes no dropout, so an adapter on `kv_b_proj`,
    a quantized load and a training-time `attention_dropout` are refused before the model loads: from
    the config (its `quantization_config` included), the run's `load_in_4bit`/`load_in_8bit`, and from
    `lora_target_modules` where it names `kv_b_proj` (or `self_attn.kv_b_proj`) or is `all-linear`.
    The first tree batch checks the loaded layers again, which catches a regex target.
  - **Plain documents.** The plan's `sparse` part (per token: path predecessor, subtree end, path
    position, tree start) comes with every batch, so plain documents in one packed row stay apart in
    these layers too.

`tree_full_attention` left unset takes `split` where the model's backend runs it and `gather` elsewhere: on
`flash_attention_2`, and on FA4's head_dim-256 kernel, which takes no log-sum-exp gradient. Qwen3.5/3.6 takes
`gather`: on Blackwell an unset `attn_implementation` resolves to `flash_attention_2`, its packed-row default
([Qwen3.5](../models/qwen3_5.md#configs)), and its GatedDeltaNet layers need the `fla` and `causal_conv1d` kernels
the images ship. On Hopper, pin `flash_attention_2`: auto-selection picks FA3, which tree packing refuses. The
backend is checked before any weight loads, against the one the loader resolves for the config, dtype and GPU, and
again on the loaded model. GLM-5 Next is held to none of these backends and refuses `tree_full_attention`:
under a plan neither of its layer types reaches the attention interface. Its KDA layers need the same `fla` and
`causal_conv1d` kernels.

The `split` form and the state-passing recurrence add kernel variants to the
[warm-up](flash-attention.md), compiled on every rank at startup of a multi-rank run.

## Equivalence

Every example of a tree row is compared with the same example run completely alone. In fp32 on eager
attention the per-example loss agrees to about 1e-5 and the argmax at every trained position is the
same. In bf16 the difference is at the level of the backend's own noise — the yardstick column runs the
*unpacked* examples through eager instead of the flash kernel:

| Model | Backend, form | Same argmax at trained positions | Yardstick |
|---|---|---|---|
| Qwen3-30B-A3B | FA4, `gather` | 100% (bit-identical) | 99.4% |
| Qwen3-30B-A3B | FA4, `split` | 99.5% | 99.4% |
| Qwen3.5-35B-A3B | FA2, `gather` | 98.1% | 98.5% |

Gradients are compared in fp32: in bf16, two attention backends already disagree on input-embedding
gradients by more than a defect would move them. With the whole model in fp32, Triton dots in IEEE
precision (`TRITON_F32_DEFAULT=ieee`, `FLA_TRIL_PRECISION=ieee`) and an exact torch varlen kernel
standing in for flash attention, the examples of real trees agree with their runs alone in loss and in
every argmax, and their input-embedding gradients to rounding; detaching the delta-rule state passed
between depths moves the shared tokens' gradients by orders of magnitude more. The residual is fp32
rounding of equivalent kernels, not the tree: each example run alone through the tree path, with no cut
and no sharing, moves its gradients just as much, while cutting a path into nodes and sharing a prefix
each add a few percent of that. Against a float64 recurrence, fla's delta rule cut in two with the state
handed over, and the causal convolution split at a `seq_idx` boundary, are as exact as one call.

A gather split into several calls gives the one-call forward bit for bit in fp32. Two bf16 forwards of
the same row already differ, so chunked and one-call bf16 runs agree only to that level.

A tree row read as one causal sequence agrees at 69% on Qwen3.5-35B-A3B. That never runs: a
truncated, split or reordered tree row raises at collation, and a model whose layers would not read
the `tree_attention` key is refused at startup.

`tests/cpu/data/test_tree_rows.py` holds the data path to this (tree and flat packing of one dataset
yield the same examples and labels). `tests/cpu/models/test_tree_attention_oracle.py` and
`tests/gpu/kernels/test_fa4_tree_attention.py` hold the forward and gradients of every admitted flash
family. `tests/cpu/models/test_tree_sparse_attention.py` holds GLM-5 Next's: every example of a tree row
against its run alone, the selection against the stock indexer on every path, and the latent attention
against the dense masked softmax; `tests/gpu/kernels/test_tree_sparse_attention.py` its CUDA kernels. Both CPU
suites train a tree row of every admitted family under gradient checkpointing, reentrant and not, to the
logits and gradients of the run without it.
`tests/gpu/data/test_tree_packing_e2e.py` trains Qwen3-0.6B both ways through `scripts/training/sft.py` under
FSDP2 on two ranks, each step one whole epoch, and holds the tree run's per-step loss and gradient norm to the flat
run's.

## What differs from flat packing

- **Steps per epoch.** A packed row holds more examples, so an epoch has fewer optimizer steps and
  each step more trained tokens. The learning-rate schedule spans fewer steps.
- **Counters.** `num_input_tokens_seen` and tokens/s count each stored token once. Throughput in
  stored tokens/s is that of flat packing; examples/s is higher by the sharing factor. The MFU
  attention term costs each node's own tokens at its path's length, so a shared prefix is costed once.
- **`eval_loss`.** Each eval step's loss is token-weighted over that step's rows and the steps are
  averaged, so the value depends on which rows share a step. A tree run and a flat run of one model
  report different numbers on a small eval split while every example's loss is the same.
- **Router balancing.** `bias_update` counts a token once per example it belongs to, the load flat
  packing sees. The `aux_loss` term is computed over stored tokens, so a shared token weighs once
  in that regularizer instead of once per example; the language-model loss is unaffected. The MoE load
  metrics (`enable_moe_metrics`) count stored tokens too.
- **Disk.** The token caches hold four columns, about 10 bytes per token against flat packing's 5.

## Limits

`tree_packing` raises unless `packing: true` with `packing_strategy: bfd` (the other strategies cut a
row below a tree), `eval_packing` is left on and `train_on_completions_only` is on. It is refused
under context, tensor and pipeline parallelism, with `torch_compile`, with `train_sinks`, on a
pre-processed dataset and on the VLM path. The other `tree_*` options raise when set without it.
Expert parallelism and FSDP2 data parallelism never slice a row and run tree rows unchanged. Pure ETP and
EP+ETP are admitted but unverified on tree rows.

Families are admitted one by one (`TREE_FAMILIES` in `halo/models/attention/tree.py`), each by passing
the per-example oracle test; every other family is refused before the model loads. Backends are admitted
the same way, by the GPU kernel test: `flash_attention_3` is refused. GLM-5 Next additionally refuses an
adapter on `kv_b_proj`, a quantized load and a training-time `attention_dropout`, before the model loads (see
[Attention](#attention)).
