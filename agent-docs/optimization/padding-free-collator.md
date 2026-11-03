# Padding-Free Flash Attention Collator

Concatenates variable-length sequences into one flattened tensor and emits `cu_seq_lens` so a varlen Flash Attention kernel skips padding compute. These collators are SFT-only: `select_data_collator` (`halo/data/collators/factory.py`) picks them when `padding_free: true`. Mutually exclusive with `packing`. Incompatible with Context and Pipeline Parallelism.

SMPO also supports `padding_free` (not under PP), but flattens the batch itself in `_forward_padding_free` (`halo/trainers/preference/smpo.py`) rather than using these collators. Padded rows pay full GEMM cost; the M-dimension argument is in [GPU Training Theory §2](../reference/gpu-training-theory.md#2-the-roofline--arithmetic-intensity-and-the-ridge-point).

SMPO's flattened row isolates its documents too: attention through `position_ids`, and LFM-2 and GatedDeltaNet through `seq_idx` / `cu_seq_lens` segment markers built by the same gate these collators use (`halo/models/segment_markers.py`).

Implementations live in `halo/data/collators/packing.py`:

- `DataCollatorWithFlattening` — padding-free, no completion masking.
- `DataCollatorWithFlatteningAndCompletionMask` — padding-free + completion-only masking (train only on assistant responses).

## Configuration

`padding_free`, `packing`, `packing_strategy` and `eval_packing` come from TRL's [`SFTConfig`](../reference/configuration-reference.md#trl-trainer-configs); the completion-masking fields (`train_on_completions_only`, `assistant_message_template`) from [`SFTScriptArguments`](../reference/configuration-reference.md#sftscriptarguments):

```yaml
padding_free: true
attn_implementation: "flash_attention_2"   # a varlen FA backend (FA2, FA3 or FA4)
packing: false                              # mutually exclusive with padding_free
train_on_completions_only: true
assistant_message_template: "<|im_start|>assistant\n"
```

`select_data_collator` gates on the model's **resolved** `_attn_implementation`, not on a family list.
`padding_free` is **rejected** outside `flash_attention_2` / `_3` / `_4`: the `cu_seq_lens` it exists to emit
go unread there, so it buys nothing at all. The SFT script runs the same gate (`reject_dense_attention_rows`)
before the model load, against the backend the loader will resolve for the checkpoint.

That catches every model the loader routes off Flash Attention: DeepSeek-V4 (eager-only), Gemma 4
(`head_dim=512`), the families whose upstream class declares no flash support (Ling 3.0, GLM-5 Next,
Step-3.7 Flash, Inkling, Qwen3.8-Flash-Next), PaliGemma and PaliGemma 2 (prefix LMs, whose `token_type_ids`
mask no flash kernel reads), and, outside SFT, a pinned `flash_attention_4` on a head_dim-256 model under `full_determinism`,
which falls back to SDPA (SFT's multi-document rows load FA2 there, pinned or not). The full map
is in [Flash Attention](flash-attention.md#choosing-a-backend). A softcapped model (Gemma 2) runs both on FA2 on
Blackwell, since a softcap stays off FA4's varlen kernels; a pinned `flash_attention_4` the load keeps (Gemma 2 27B)
is refused there.

Use packing or padded batches there: packing raises tokens per row on any kernel, so off a varlen backend it
only **warns**. gpt-oss, the Bailing family (Ling 2.0, Ring, Ling 3.0) and GLM-5 Next are the exceptions. Their
forwards never pass the packed `position_ids` into mask construction, so the row would run as one dense causal
sequence with documents attending across each other, and packing off a varlen backend is **rejected** there.
DeepSeek-V4 rejects packing too, on every backend: its compressors cut KV windows at row indices, so a later packed
document reads the row's first windows. A model whose linear-attention layers no segment marker reaches (Ring's
Lightning Attention-2) refuses packing and `padding_free` on every backend, flash included, unless the run sets
`allow_cross_document_linear_attention: true`; Ling 3.0 (no flash class, and its SDPA mask attends across documents)
and GLM-5 Next's flat rows stay refused either way
([Document isolation](../data/collators.md#document-isolation-under-packing)).

`padding_free` is also rejected on an M-RoPE decoder (Qwen3.5/3.6, text towers included, and the Qwen-VL text
models) on every backend, by the SFT script before the model load, the SFT collator factory, the SFT trainer (for
a collator handed to it) and the SMPO trainer alike (`reject_mrope_padding_free`): the flattened M-RoPE row faults
the flash varlen kernels. Turn `padding_free` off there; under SFT, `packing` serves the same rows (SMPO has no
`packing` field and pads instead).

Both collators emit `position_ids` that reset per document; whether a family's forward turns those into an
isolating mask is per-family, and the exceptions are tabulated in
[Document isolation under packing](../data/collators.md#document-isolation-under-packing). Where it does, the
only difference is cost: a non-varlen backend materializes a dense `[L, L]` mask instead of consuming
`cu_seq_lens`.

**Both** collators flatten the mini-batch into a single row — padding-free by
construction, packing via `flatten_packed_batch` — so `L` is the whole batch's token count either way (the
summed real tokens for padding-free, up to `per_device_train_batch_size × max_length` for packing), and at
equal tokens per step the two masks are the same size.

Packing's mask is bounded by `max_length` only at `per_device_train_batch_size: 1`, so keep the batch at 1
there and scale with `gradient_accumulation_steps`.

The collator emits `input_ids`, `labels`, and `position_ids` (reset per sequence) shaped `[1, total_tokens]`, plus the varlen kwargs `cu_seq_lens_q`/`cu_seq_lens_k` (1-D int32, length `num_seqs + 1`) and `max_length_q`/`max_length_k` (Python ints). The varlen FA kernel reads `cu_seq_lens` to attend only within each sequence. The varlen kwargs are gated by the collator's `emit_varlen_always=True`. `seq_idx` joins them under `return_seq_idx`, which `select_data_collator` sets for the families whose conv/linear-attention mixers read it (LFM-2, GatedDeltaNet).

## Benchmark results

Throughput is **real (non-padding) tokens/s/GPU** — `attention_mask.sum()`, not padded element count — set via `include_num_input_tokens_seen="non_padding"` so all three modes share one real-token basis.

**Model:** Qwen3-30B-A3B (128 experts, top_k=8), FA2. **Hardware:** 2× B300 (SM103), EP=2, Blackwell image. **Data:** max_length=4096, avg ≈ 1024 tokens (~75% padding waste). **Setup:** batch_size=2/GPU, GC on, 8 steps / 3 warmup.

| Mode | tokens/s/GPU (real) | Step time | vs Standard |
|------|:-------------------:|:---------:|:-----------:|
| Standard | 922 | 2.08s | 1.0× |
| Packing | 13,341 | 0.59s | **14.5×** |
| Padding-Free | 4,178 | 0.46s | 4.5× |

Peak memory is 117–120 GiB across all three (weights + optimizer states dominate; packing's is ~3 GiB above the other two). At ~75% padding waste, packing wins by filling each max_length block with ~4 real sequences and cutting the step count. Padding-free strips the padding via varlen FA, so it processes only real tokens per step at lower memory than packing, while keeping per-sample boundaries (no cross-sequence attention).

The table uses `--attn_implementation flash_attention_2`. FA4 is also valid: FA4 + packing + EP is safe (`warm_attention_kernels` compiles the FA4 kernels behind a barrier at load — see [Flash Attention](flash-attention.md#known-issues)). With `--attn_implementation` unset (the default), the benchmark auto-detects FA4 on Blackwell (`tests/common/benchmark_args.py`).

## Parallelism compatibility

| Collator | EP | TP | CP | PP |
|----------|-----|-----|-----|-----|
| `DataCollatorWithFlattening` (+ `…AndCompletionMask`) | Yes | Yes | **No** | **No** |
| `DataCollatorWithPacking` (+ `DataCollatorForCompletionOnlyLMWithPacking`) | Yes | Yes | **No** | Yes |
| `DataCollatorForCausalLMWithPadding` | Yes | Yes | Yes | Yes |

CP requires fixed-length sequences for collective synchronization: padding-free (variable-length output) and packing (the Ulysses CP attention path has no per-document boundaries, so packed documents would attend across each other) are both rejected by `select_data_collator` when `use_context_parallel=True` — use the standard padded collator with `pad_to_multiple_of=cp_size` ([Context Parallelism](../parallelism/context-parallelism.md)).

PP rejects padding-free: its flattened width varies every step, while the pipeline's point-to-point buffers freeze after the first. Packing runs under PP at one packed row per microbatch, each stage rebuilding the segment markers its conv / linear-attention layers read, except where such layers read no boundary at all: GLM-5 Next and Inkling in the PP roster ([PP packing](../parallelism/pipeline-parallelism.md#packing)).

## When to use each

- **avg > 80% of max_length** — any collator (all within ~1%); use standard.
- **avg << max_length** — packing (14.5× at ~75% waste; packing's own cross-sequence padding overhead is 1–5%).
- **Need CP** — standard padding only; both packing and padding-free are rejected.
- **Want to skip padding FLOPS without cross-sequence boundaries** — padding-free.

## Running benchmarks

```bash
# All 3 collator modes (variable-length, ~75% padding waste) — table above
torchrun --nproc_per_node=2 \
    tests/gpu/profiling/benchmark_collators.py --model qwen3-30b-a3b --ep 2 --seq 4096 \
    --batch_size 2 --attn_implementation flash_attention_2

# Higher padding waste (avg 10% of max_length)
torchrun --nproc_per_node=2 \
    tests/gpu/profiling/benchmark_collators.py --model qwen3-30b-a3b --ep 2 --seq 4096 --avg_ratio 0.1

# Single mode
torchrun --nproc_per_node=2 \
    tests/gpu/profiling/benchmark_collators.py --model qwen3-30b-a3b --ep 2 --mode packing

# Fixed-length comparison (collators equivalent)
torchrun --nproc_per_node=2 \
    tests/gpu/profiling/benchmark_sft_ep.py --model qwen3-30b-a3b --ep 2 --seq 16384
```
