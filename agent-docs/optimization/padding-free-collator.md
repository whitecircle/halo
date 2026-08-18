# Padding-Free Flash Attention Collator

Concatenates variable-length sequences into one flattened tensor and emits `cu_seq_lens` so a varlen Flash Attention kernel skips padding compute. These collators are SFT-only — `select_data_collator` (`src/data/collators/factory.py`) picks them when `padding_free: true`. SMPO also supports `padding_free`, but flattens the batch itself in `_forward_padding_free` (`src/trainers/preference/smpo.py`) rather than using these collators. Mutually exclusive with `packing`. Incompatible with Context Parallelism. Padded rows pay full GEMM cost — the M-dimension argument is in [GPU Training Theory §2](../reference/gpu-training-theory.md#2-the-roofline-arithmetic-intensity-and-the-ridge-point).

Implementations live in `src/data/collators/packing.py`:

- `DataCollatorWithFlattening` — padding-free, no completion masking.
- `DataCollatorWithFlatteningAndCompletionMask` — padding-free + completion-only masking (train only on assistant responses).

## Configuration

`padding_free`, `packing`, `packing_strategy`, and `eval_packing` come from TRL's `SFTConfig`; the completion-masking fields (`train_on_completions_only`, `assistant_message_template`) come from `ConversationRenderArguments` in `src/args/mixins.py`:

```yaml
padding_free: true
attn_implementation: "flash_attention_2"   # a varlen FA backend (FA2, FA3 or FA4)
packing: false                              # mutually exclusive with padding_free
train_on_completions_only: true
assistant_message_template: "<|im_start|>assistant\n"
```

| Option | Default | Description |
|--------|---------|-------------|
| `padding_free` | `false` | Enable padding-free varlen-FA training |
| `packing` | `false` | Enable traditional sequence packing |
| `packing_strategy` | `"bfd"` | `"bfd"` (best-fit-decreasing, drops overflow), `"bfd_split"` (carries overflow into later examples) or `"wrapped"` (concatenate-and-chunk) |
| `eval_packing` | `null` | Packing for eval (defaults to `packing`) |

`select_data_collator` gates on the model's **resolved** `_attn_implementation`, not on a family list.
`padding_free` is **rejected** outside `flash_attention_2` / `_3` / `_4`: the `cu_seq_lens` it exists to emit
go unread there, so it buys nothing at all. That catches every model the loader routes off Flash Attention —
DeepSeek-V4 (eager-only), Gemma 4 (`head_dim=512`), the families whose upstream class declares no flash
support (Bailing/Ling, GLM-5 Next, Step-3.7 Flash, Inkling), and Qwen3.5/3.6 and GLM-4 MoE Lite when FA4
falls them back to SDPA; the full map is in [Flash Attention](flash-attention.md#choosing-a-backend). Use
packing or padded batches there: packing raises tokens per row on any kernel, so off a varlen backend it
only **warns**. gpt-oss is the exception — its forward never passes the packed
`position_ids` into mask construction, so the row would run as one dense causal sequence with documents
attending across each other, and packing off a varlen backend is **rejected** there.

Both collators emit `position_ids` that reset per document; whether a family's forward turns those into an
isolating mask is per-family, and the exceptions are tabulated in
[Document isolation under packing](../data/collators.md#document-isolation-under-packing). Where it does, the
only difference is cost: a non-varlen backend materializes a dense `[L, L]` mask instead of consuming
`cu_seq_lens`. Outside pipeline parallelism **both** collators flatten the mini-batch into a single row —
padding-free by construction, packing via `flatten_packed_batch` — so `L` is the whole batch's token count
either way (the summed real tokens for padding-free, up to `per_device_train_batch_size × max_length` for
packing), and at equal tokens per step the two masks are the same size.

Packing's mask is bounded by `max_length` only at `per_device_train_batch_size: 1`, so keep the batch at 1
there and scale with `gradient_accumulation_steps`.

The collator emits `input_ids`, `labels`, and `position_ids` (reset per sequence) shaped `[1, total_tokens]`, plus the varlen kwargs `cu_seq_lens_q`/`cu_seq_lens_k` (1-D int32, length `num_seqs + 1`) and `max_length_q`/`max_length_k` (Python ints). The varlen FA kernel reads `cu_seq_lens` to attend only within each sequence. The varlen kwargs are gated by the collator's `return_flash_attn_kwargs=True`. `seq_idx` joins them under `return_seq_idx`, which `select_data_collator` sets for the families whose conv/linear-attention mixers read it (LFM-2, GatedDeltaNet).

## Benchmark results

Throughput is **real (non-padding) tokens/s/GPU** — `attention_mask.sum()`, not padded element count — set via `include_num_input_tokens_seen="non_padding"` so all three modes share one real-token basis.

**Model:** Qwen3-30B-A3B (128 experts, top_k=8), FA2. **Hardware:** 2× B300 (SM103), EP=2. **Data:** max_length=4096, avg ≈ 1024 tokens (~75% padding waste). **Setup:** batch_size=2/GPU, GC on, 8 steps / 3 warmup.

| Mode | tokens/s/GPU (real) | Step time | vs Standard |
|------|:-------------------:|:---------:|:-----------:|
| Standard | 1,092 | 1.83s | 1.0× |
| Packing | 10,072 | 0.78s | **9.2×** |
| Padding-Free | 2,547 | 0.79s | 2.3× |

Peak memory is ~115.8 GB across all three (weights + optimizer states dominate). At ~75% padding waste, packing wins by filling each max_length block with ~4 real sequences and cutting the step count. Padding-free strips the padding via varlen FA, so it processes only real tokens per step at lower memory than packing, while keeping per-sample boundaries (no cross-sequence attention).

The table was measured with `--attn_implementation flash_attention_2`. FA4 is also valid: FA4 + packing + EP is safe (DeepEP V2's long-timeout dispatch barrier absorbs FA4's first-use JIT compile — see [Flash Attention](flash-attention.md#known-issues)). With `--attn_implementation` unset (the default), the benchmark auto-detects FA4 on Blackwell (`tests/common/benchmark_args.py`).

## Parallelism compatibility

| Collator | EP | TP | CP | PP ([not yet available](../parallelism/pipeline-parallelism.md)) |
|----------|-----|-----|-----|-----|
| `DataCollatorWithFlattening` (+ `…AndCompletionMask`) | Yes | Yes | **No** | **No** |
| `DataCollatorWithPacking` (+ `DataCollatorForCompletionOnlyLMWithPacking`) | Yes | Yes | **No** | Yes |
| `DataCollatorForCausalLMWithPadding` | Yes | Yes | Yes | Yes |

CP requires fixed-length sequences for collective synchronization: padding-free (variable-length output) and packing (the Ulysses CP attention path has no per-document boundaries, so packed documents would attend across each other) are both rejected by `select_data_collator` when `use_context_parallel=True` — use the standard padded collator with `pad_to_multiple_of=cp_size`.

Pipeline parallelism ([not yet available in this release](../parallelism/pipeline-parallelism.md)) will take packing but not padding-free: its shipped collator seam keeps packed rows padded to `max_length` as a fixed shape, while padding-free's flattened width varies every step. See [Context Parallelism](../parallelism/context-parallelism.md).

## When to use each

- **avg > 80% of max_length** — any collator (all within ~1%); use standard.
- **avg << max_length** — packing (9.2× at ~75% waste; packing's own cross-sequence padding overhead is 1–5%).
- **Need CP** — standard padding only; both packing and padding-free are rejected.
- **Need PP** — packing or standard padding; padding-free is rejected.
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
