# Qwen3.8-Flash-Next

`model_type` `qwen4_exp` (`Qwen4ExpForConditionalGeneration`, the release `Qwen/Qwen3.8-Flash-Next`) wrapping the text tower `qwen4_exp_text` (`Qwen4ExpForCausalLM`). `text_only_model: true` loads the text tower and drops the vision tower and the `mtp.*` head; the multimodal class loads too: its forward is tiny-model verified on image and packed text rows, and no test trains it. The release has 48 decoder layers: 36 GatedDeltaNet `linear_attention` layers and 12 sparse-attention (QSA) layers, which the hub spells `full_attention` and transformers maps to `indexed_attention`. Four hyper-connection streams carry the residual; every layer's MoE block routes 512 experts at top-10 beside a sigmoid-gated shared expert; the second layer adds a per-layer embedding (PLE) looked up in a hashed n-gram table of 320,001,536 rows × 160 (95 GiB in bf16, `split_ngram_parts: 128` shards on the hub).

**Experimental.** GPU tests cover tiny models only: EP against the reference model, pure ETP, the gathered-save round trip, deterministic expert gradients and LoRA merged-save resume. EP+ETP, admitted through the inherited Qwen3.5 layer, is untested, and no real-weight or full-depth run has trained the family. Per-axis support: [Compatibility matrix](README.md#compatibility-matrix). Methods: SFT, full fine-tuning and LoRA. EP runs with lazy loading; TP, CP, PP and RL weight sync are refused. At `ep_size: 1` (plain FSDP2) every GPU loads the whole model, about 246 GiB at full size: tiny and test models only.

## EP layer

`EPQwen4ExpMoELayer` (`halo/distributed/expert_parallel/families/qwen4_exp.py`) subclasses [Qwen3-Next](qwen3-next.md#ep-layer)'s layer: Qwen3.5-MoE's fused `gate_up_proj [E, 2M, H]` / `down_proj [E, H, M]` experts and replicated gated shared expert, under a router that renormalizes the top-k weights only under `norm_topk_prob`. The release stores the experts fused under the block's own names, so the spec keeps the default `FUSED` layout and the gathered save writes `experts.gate_up_proj` / `experts.down_proj` as the hub does. Hidden 2560 is a multiple of 256, so DeepEP pads nothing.

## N-gram table

The table never enters the module state. The family patches (`halo/models/qwen4_exp/forwards.py`, applied by `apply_family_patches` on every policy, trainer-path and frozen reference or teacher load) stamp the config the model builds on, and a stamped `Qwen4ExpTextNGramEmbedding` holds a `HostNGramTable` (`halo/models/qwen4_exp/ngram_table.py`) in place of its `nn.Embedding`: no parameter, buffer or state-dict entry, so FSDP2, the optimizer, the fp32 masters, PEFT, the coverage gate and the lazy planner never see it, and it trains frozen (the stock class trains it). A build without the stamp keeps the stock class and holds the table as a parameter: a checkpoint tool's `from_pretrained`. The sentence-transformers embedding load, which reads its backbone config itself, refuses a backbone with external tables.

`finalize_loaded_model` attaches the table from the local directory the weights were read from: the snapshot of the commit every rank agreed on, which the policy loaders, the trainer-path load and a frozen reference or teacher load that agreed its source name for a Hub source (the lazy EP loader reads it directly). A model loaded outside them whose config records a Hub id is refused, since transformers records no revision to pair the table with the weights. The parts are zero-copy views of the `ngram_embedding.shard_*` tensors, so every rank on a node shares one page-cache copy, and local rank 0 asks the kernel to read the table files ahead. A lookup gathers its rows on the CPU (one thread per rank) and copies them, about 5 KB per token, to the device through pinned memory; the reentrant recompute of the PLE layer reads the first pass's rows back. Part count, row count, width and dtype are checked against the config; a model whose config names no checkpoint, or one without the table, is refused, and so is a random-weight build (`init_from_scratch`, refused before the build), whose config still names the pretrained checkpoint.

The n-gram embedding copies `eos_token_id` when it is built and restarts its history after it. A run's `eos_token` that would record another id is refused (`reject_stale_eos_copies`, `halo/model_loading/tokenizer_setup.py`): training would restart at the copy while every export records the new id.

## Checkpoints and serving

Every gathered save (EP gathered, FSDP2, CP, TP) writes the table as table-only files (`model-table00-*.safetensors`) under the hub's `ngram_embedding.shard_{i}.weight` keys, beside the streamed weights; a model with the table always gets an index, and its exported `config.json` keeps `split_ngram_parts` and carries no run stamp ([Checkpoints](../reference/checkpoints.md#external-tables)). One process or DDP at `use_grouped_gemm: false`, which no parallel saver claims, streams the same way rather than through transformers' `save_pretrained`, whose state dict holds no table (the family patch refuses a direct call); accelerate-managed FSDP, whose saves stay with `save_pretrained`, is refused at trainer construction unless the run trains adapters. The first save of a process writes the table once from the attached views; each later save hard-links the previous save's files, and a resume attached from a step checkpoint links on its first save. Full fine-tuning resumes from the step checkpoint and attaches the table from it; LoRA resumes onto the base. `save_sharded_ep: true` is refused: `merge_ep_shards.py` has no model to read the table from.

An export loads in stock transformers 5.20 and carries no `mtp.*` head. vLLM 0.32 registers `Qwen4ExpForCausalLM` and reads the table as its `shard_{i}` keys; SGLang 0.5.22 registers only the multimodal class, so run `reattach_vision_tower.py` first, which also streams the MTP head back. Attention and GatedDeltaNet adapters merge with `merge_peft_adapters.py`, which loads the stock class (the table as a parameter, about 360 GB of host RAM); expert adapters fold with `merge_expert_lora_on_save: true`.

## Attention and rows

The class declares no flash or flex support, so the load takes SDPA. transformers 5.20's QSA scores every query in a Python loop and its model forward builds a `[B, 1, T, T]` mask, which leaves rows past a few thousand tokens impractical; `patch_qwen4_exp_document_forwards` (`halo/models/qwen4_exp/forwards.py`, applied with the n-gram patch) replaces both for every uncached forward:

- The model forward hands its layers the 2-D padding mask in place of the mask mapping, so no `[T, T]` mask exists.
- A QSA layer selects each query's key blocks per 128-query tile against each document's pooled blocks, and re-scores a query whose k-th and next block tie in transformers' own order, so it attends the keys transformers' indexer gives it, up to a near-tie the two matmul shapes round apart. It attends per tile with masked SDPA over K/V from the document's first key to the tile end, saves only Q/K/V and the block ids, and recomputes each tile in the backward (`halo/models/qwen4_exp/sparse_attention.py`). Memory is linear in the row; compute is dense causal per document on masked SDPA tiles, not a flash kernel.
- The PLE n-gram history restarts at every document start as well as after `eos_token_id`, and its dilated conv runs as masked shifted multiply-adds whose taps stop at the document start, off cuDNN (`halo/models/qwen4_exp/documents.py`). An adapter on the PLE `conv1d` is refused.

Documents come from `seq_idx`, which the packing collator emits for this family beside GatedDeltaNet's `cu_seq_lens`, so each document of a packed row computes what it computes alone. A row whose `position_ids` restart at 0 without `seq_idx` (and without a padding mask) is refused. A call with a cache or `use_cache=True`, as in generation, keeps transformers' forwards; one that leaves `use_cache` unset runs per document and returns no cache.

Under gradient checkpointing the recompute reuses the forward's QSA selection and n-gram rows; each QSA layer holds its selection, 1 KB per token, until its backward (about 400 MB at 32k tokens across the 12 layers). `max_length` must be set; the config's 262,144 exceeds the DeepEP token ceiling. The tiled kernel's GPU speed is unmeasured.

## Recipes

`examples/sft/qwen4_exp/qwen3.8-flash-next-ep8-lora.yaml` (16k packed rows) and `-ep8-full.yaml` (8k, B300) train the text tower at ep8 on one node with the fused loss. LoRA sets `moe_balancing: none` (the router is frozen), full fine-tuning `bias_update_transient`. Both pin Halo's `qwen3-multiturn.jinja`: the hub template is Qwen3.8-27B's, with its [reasoning-effort caveats](qwen3_5.md#qwen38-27b). Both leave `eos_token` unset: the PLE n-gram history restarts after the checkpoint's `eos_token_id` (`<|endoftext|>`), and a run's `eos_token` that changes it is refused ([N-gram table](#n-gram-table)).

## LoRA

PEFT adapts attention (`q_proj`, `k_proj`, `v_proj`, `o_proj`) and GatedDeltaNet (`in_proj_qkv`, `in_proj_z`, `in_proj_a`, `in_proj_b`, `out_proj`); routed experts take Halo's native expert LoRA under EP ([PEFT](../optimization/peft.md)). An adapter on `indexer.index_qk_proj` never gets a gradient: the indexer's top-k is a boolean selection, in transformers as here. List targets explicitly, as the LoRA recipe does.

## Balancing

The router has no bias slot. On the family's own head `moe_balancing: auto` resolves to `aux_loss`. The Liger spec declares the head's in-head aux loss, so under the fused loss `output_router_logits` is refused and `auto` resolves to `none` with a warning; pick `bias_update_transient` (trainer-only, no export carries the bias) or `none` explicitly. `bias_update` raises ([Callbacks](../training-methods/callbacks.md#moe-balancing-modes)).

## Refused modes

- **CP**: 36 of 48 layers are `linear_attention`, whose conv and recurrent scan run along the sequence axis.
- **TP**: `Qwen4ExpTextAttention` is not in `TP_SHARDABLE_ATTENTION_CLASSES`, so a TP run would shard no layer.
- **PP**: `Qwen4ExpTextModel` has no `PPFamilySpec`.
- **`padding_free`, tree packing**: SDPA reads no `cu_seqlens`, and tree packing admits only the families it is verified on.
- **Packed rows without `seq_idx`**: the model forward refuses them ([above](#attention-and-rows)).
- **`init_from_scratch`** and an **`eos_token`** that changes `eos_token_id` ([N-gram table](#n-gram-table)), a direct **`save_pretrained`** and a full fine-tune under **accelerate-managed FSDP** ([Checkpoints and serving](#checkpoints-and-serving)).
- **RL weight sync**: the spec's `weight_sync_refusal`; no online update into either pinned engine is validated, and the table is outside the model the sync streams.
