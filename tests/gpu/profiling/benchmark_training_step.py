#!/usr/bin/env python
"""One training step of a real causal LM on one GPU, timed by phase beside its memory: activations across batch and
sequence with gradient checkpointing off and on (``activations``), what the trainable scope costs (``trainable``:
full, LoRA, two layers at either end), and padded against packed rows holding the same real tokens (``padding``).

    python tests/gpu/profiling/benchmark_training_step.py --experiment activations --output activations.json
    python tests/gpu/profiling/benchmark_training_step.py --experiment trainable --shapes 1x4096 --output trainable.json
    python tests/gpu/profiling/benchmark_training_step.py --experiment padding --gc on --output padding.json
"""

import argparse

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoConfig, AutoModelForCausalLM

from halo.model_loading.dtype import configure_float32_matmul_precision
from halo.model_loading.finalization import finalize_loaded_model
from halo.model_loading.liger.dispatch import LIGER_FLAGS, apply_liger_kernel
from halo.model_loading.model_preparation import apply_family_patches
from halo.models.introspection.module_tree import backbone_with_layers, decoder_layers
from halo.optimizers.adamw_bf16 import AdamWBF16
from tests.common.benchmark_args import resolve_benchmark_attn
from tests.common.memory import cleanup_memory
from tests.common.models import MODEL_CONFIGS
from tests.gpu.profiling.measurement import gpu_state, guarded, run_record, summarize, write_record

EXPERIMENTS = ("activations", "trainable", "padding")
SCOPES = ("full", "lora", "last_two", "first_two")
LORA_RANK = 16
LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj")
SEED = 0
DEFAULT_SHAPES = ("1x1024", "1x2048", "1x4096", "1x8192", "1x16384", "1x32768", "4x4096", "16x1024")
# 16 documents of 1024 tokens: 16 rows padded to 4096, against the same tokens packed (4 rows' worth of 4096).
PADDING_DOCS, PADDING_DOC_LEN, PADDING_WIDTH = 16, 1024, 4096
# Largest relative gap between the packed and the documents-alone loss that still reads as the same computation.
PACKED_LOSS_TOLERANCE = 1e-3
FLCE = "fused_linear_cross_entropy"


def _load_model(model_path: str, attn_implementation: str | None, liger: str):
    """The checkpoint through the family patches and ``finalize_loaded_model`` on GPU 0, as the loaders build it.
    ``liger``: ``off``; ``flce``, Liger's fused linear cross-entropy alone, so a labelled forward never builds the
    logits; or ``defaults``, the family's default Liger kernels as an SFT run applies them."""
    config = AutoConfig.from_pretrained(model_path)
    if liger == "flce":
        apply_liger_kernel(config, liger_kernel_config={flag: flag == FLCE for flag in LIGER_FLAGS})
    elif liger == "defaults":
        apply_liger_kernel(config)
    attn = apply_family_patches(config, resolve_benchmark_attn(model_path, attn_implementation))
    model = AutoModelForCausalLM.from_pretrained(
        model_path, config=config, dtype=torch.bfloat16, attn_implementation=attn, device_map={"": 0}
    )
    finalize_loaded_model(model)
    configure_float32_matmul_precision()
    model.train()
    return model, attn


def _set_checkpointing(model, enabled: bool) -> None:
    if enabled:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": True})
    else:
        model.gradient_checkpointing_disable()


def _apply_scope(model, scope: str):
    """Trainable parameters for ``scope``; the input embedding and the head, one tensor when tied, stay frozen outside
    ``full``."""
    if scope == "lora":
        lora = LoraConfig(r=LORA_RANK, lora_alpha=2 * LORA_RANK, lora_dropout=0.0, target_modules=list(LORA_TARGETS))
        return get_peft_model(model, lora)
    model.requires_grad_(scope == "full")
    layers = decoder_layers(backbone_with_layers(model))
    for layer in {"last_two": layers[-2:], "first_two": layers[:2]}.get(scope, ()):
        layer.requires_grad_(True)
    return model


def _tokens(shape: tuple[int, int], vocab: int) -> torch.Tensor:
    generator = torch.Generator(device="cuda").manual_seed(SEED)
    return torch.randint(0, vocab, shape, device="cuda", generator=generator)


def _full_rows(batch: int, seq: int, vocab: int) -> dict:
    ids = _tokens((batch, seq), vocab)
    return {"input_ids": ids, "labels": ids}


def _padding_batches(vocab: int) -> dict[str, dict]:
    """The same documents padded (one per row, right-padded), packed and alone (one per unpadded row), so all three
    predict the same tokens. Packed rows reach the model flattened into one row with ``position_ids`` restarting per
    document and each first token unlabelled: the varlen form flash attention isolates documents in, which transformers
    infers from position ids only for a single row."""
    docs = _tokens((PADDING_DOCS, PADDING_DOC_LEN), vocab)
    padded = torch.zeros(PADDING_DOCS, PADDING_WIDTH, dtype=docs.dtype, device="cuda")
    padded[:, :PADDING_DOC_LEN] = docs
    mask = torch.zeros_like(padded)
    mask[:, :PADDING_DOC_LEN] = 1
    packed = docs.reshape(1, -1)
    packed_labels = packed.clone()
    packed_labels[:, ::PADDING_DOC_LEN] = -100
    positions = torch.arange(PADDING_DOC_LEN, device="cuda").repeat(1, PADDING_DOCS)
    return {
        "padded": {"input_ids": padded, "attention_mask": mask, "labels": padded.masked_fill(mask == 0, -100)},
        "packed": {"input_ids": packed, "position_ids": positions, "labels": packed_labels},
        "alone": {"input_ids": docs, "labels": docs},
    }


def _tensor_bytes(tensors) -> int:
    return sum(t.numel() * t.element_size() for t in tensors)


def _backbone_saved(model, batch: dict) -> dict:
    """``{"bytes": n}``: what the decoder stack alone holds for its backward after one forward, its output included."""
    inputs = {key: value for key, value in batch.items() if key != "labels"}
    backbone = backbone_with_layers(model)
    cleanup_memory()
    before = torch.cuda.memory_allocated()
    hidden = backbone(**inputs).last_hidden_state
    saved = torch.cuda.memory_allocated() - before
    del hidden
    return {"bytes": saved}


def _measure(model, batch: dict, optimizer, *, repeats: int, warmup: int) -> dict:
    """``warmup`` then ``repeats`` steps: per-phase device seconds from CUDA events, the allocator's peak over the
    measured steps, and the bytes held after the forward, of gradients, and of optimizer state."""
    params = [p for p in model.parameters() if p.requires_grad]
    phases: dict[str, list[float]] = {"forward": [], "backward": []}
    if optimizer is not None:
        phases["optimizer"] = []
    held: dict[str, int] = {}
    try:
        for step in range(warmup + repeats):
            if step == warmup:
                cleanup_memory()
                torch.cuda.reset_peak_memory_stats()
                held["before_step"] = torch.cuda.memory_allocated()
            marks = [torch.cuda.Event(enable_timing=True) for _ in range(4)]
            marks[0].record()
            out = model(**batch)
            marks[1].record()
            after_forward = torch.cuda.memory_allocated()
            out.loss.backward()
            marks[2].record()
            grad_bytes = _tensor_bytes(p.grad for p in params if p.grad is not None)
            if optimizer is not None:
                optimizer.step()
            marks[3].record()
            model.zero_grad(set_to_none=True)
            loss = out.loss.item()
            held["logits_materialized"] = out.logits is not None
            del out
            marks[3].synchronize()
            if step < warmup:
                continue
            held["after_forward"], held["gradients"] = after_forward, grad_bytes
            # One event pair per phase; without an optimizer the last pair goes unread.
            for name, start, end in zip(phases, marks, marks[1:], strict=False):
                phases[name].append(start.elapsed_time(end) / 1e3)
    finally:
        model.zero_grad(set_to_none=True)
    step_seconds = [sum(parts) for parts in zip(*phases.values(), strict=True)]
    state_bytes = 0
    if optimizer is not None:
        state_bytes = _tensor_bytes(
            t for state in optimizer.state.values() for t in state.values() if torch.is_tensor(t)
        )
    return {
        "seconds": phases,
        "seconds_summary": {name: summarize(samples) for name, samples in phases.items()},
        "step_seconds": step_seconds,
        "step_seconds_summary": summarize(step_seconds),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "allocated_before_step_bytes": held["before_step"],
        "allocated_after_forward_bytes": held["after_forward"],
        "saved_by_forward_bytes": held["after_forward"] - held["before_step"],
        "gradient_bytes": held["gradients"],
        "optimizer_state_bytes": state_bytes,
        "last_loss": loss,
        "logits_materialized": held["logits_materialized"],
        "state": gpu_state(),
    }


def _guarded(fn, *args, **kwargs) -> dict:
    """:func:`guarded`, the allocator emptied after it either way."""
    try:
        return guarded(fn, *args, **kwargs)
    finally:
        cleanup_memory()


def _model_facts(model, attn: str) -> dict:
    config = model.config
    return {
        "attn_implementation": attn,
        "parameters": sum(p.numel() for p in model.parameters()),
        "num_hidden_layers": config.num_hidden_layers,
        "hidden_size": config.hidden_size,
        "intermediate_size": config.intermediate_size,
        "vocab_size": config.vocab_size,
        "tie_word_embeddings": config.tie_word_embeddings,
    }


def _report(record: dict, row: dict, output: str) -> None:
    record["rows"].append(row)
    write_record(output, record)
    if "oom" in row:
        print(f"  {row}  OOM", flush=True)
        return
    label = {key: row[key] for key in ("scope", "layout", "gradient_checkpointing", "batch", "seq") if key in row}
    print(
        f"  {label}  step {row['step_seconds_summary']['median'] * 1e3:8.1f} ms  "
        f"peak {row['peak_allocated_bytes'] / 2**30:7.2f} GiB  saved {row['saved_by_forward_bytes'] / 2**30:7.2f} GiB",
        flush=True,
    )


def _activations(args, model_path: str, record: dict) -> None:
    model, attn = _load_model(model_path, args.attn_implementation, args.liger)
    record["model"] = _model_facts(model, attn)
    vocab = model.config.vocab_size
    for gc in args.gc:
        _set_checkpointing(model, gc == "on")
        for shape in args.shapes:
            batch_size, seq = (int(part) for part in shape.split("x"))
            batch = _full_rows(batch_size, seq, vocab)
            row = {"gradient_checkpointing": gc, "batch": batch_size, "seq": seq, "tokens": batch_size * seq}
            row["logits_bf16_bytes"] = 2 * batch_size * seq * vocab
            row["backbone_saved"] = _guarded(_backbone_saved, model, batch)
            row.update(_guarded(_measure, model, batch, None, repeats=args.repeats, warmup=args.warmup))
            _report(record, row, args.output)


def _trainable(args, model_path: str, record: dict) -> None:
    batch_size, seq = (int(part) for part in args.shapes[0].split("x"))
    for gc in args.gc:
        for scope in args.scopes:
            model, attn = _load_model(model_path, args.attn_implementation, args.liger)
            record["model"] = _model_facts(model, attn)
            model = _apply_scope(model, scope)
            _set_checkpointing(model, gc == "on")
            named = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
            optimizer = AdamWBF16(named, lr=1e-5)
            batch = _full_rows(batch_size, seq, model.config.vocab_size)
            row = {
                "scope": scope,
                "gradient_checkpointing": gc,
                "batch": batch_size,
                "seq": seq,
                "trainable_parameters": sum(p.numel() for _, p in named),
                "trainable_dtypes": sorted({str(p.dtype) for _, p in named}),
            }
            row.update(_guarded(_measure, model, batch, optimizer, repeats=args.repeats, warmup=args.warmup))
            _report(record, row, args.output)
            del model, optimizer
            cleanup_memory()


@torch.no_grad()
def _losses(model, batches: dict[str, dict]) -> dict[str, float]:
    return {layout: model(**batch).loss.item() for layout, batch in batches.items()}


def _padding(args, model_path: str, record: dict) -> None:
    model, attn = _load_model(model_path, args.attn_implementation, args.liger)
    record["model"] = _model_facts(model, attn)
    batches = _padding_batches(model.config.vocab_size)
    losses = record["no_grad_loss_by_layout"] = _losses(model, batches)
    # Packed rows predict the same tokens as the documents alone only when attention stops at each document.
    if abs(losses["packed"] - losses["alone"]) > PACKED_LOSS_TOLERANCE * abs(losses["alone"]):
        raise RuntimeError(
            f"packed loss {losses['packed']:.6f} differs from the documents run alone ({losses['alone']:.6f}): "
            "attention crossed document boundaries, so the padding comparison would not compare the same work. Pass "
            "--attn_implementation flash_attention_2, flash_attention_3 or flash_attention_4 (unset picks this GPU's "
            "flash kernel), which isolates each document by its position_ids."
        )
    optimizer = AdamWBF16(list(model.named_parameters()), lr=1e-5)
    real_tokens = PADDING_DOCS * PADDING_DOC_LEN
    for gc in args.gc:
        _set_checkpointing(model, gc == "on")
        for layout in ("padded", "packed"):
            batch = batches[layout]
            row = {
                "layout": layout,
                "gradient_checkpointing": gc,
                "model_input_shape": list(batch["input_ids"].shape),
                "real_tokens": real_tokens,
                "gemm_tokens": batch["input_ids"].numel(),
            }
            row.update(_guarded(_measure, model, batch, optimizer, repeats=args.repeats, warmup=args.warmup))
            if "oom" not in row:
                forward_backward = [
                    f + b for f, b in zip(row["seconds"]["forward"], row["seconds"]["backward"], strict=True)
                ]
                row["real_tokens_per_s_forward_backward"] = real_tokens / summarize(forward_backward)["median"]
                row["real_tokens_per_s_step"] = real_tokens / row["step_seconds_summary"]["median"]
            _report(record, row, args.output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiment", required=True, choices=EXPERIMENTS)
    parser.add_argument("--output", required=True, help="JSON file, rewritten after each row")
    parser.add_argument(
        "--model",
        default="qwen3-4b",
        choices=[key for key, entry in MODEL_CONFIGS.items() if entry["num_experts"] == 0],
        help="A dense row of the shared roster: the script loads a plain causal LM",
    )
    parser.add_argument("--attn_implementation", default=None, help="Unset: the production pick for this GPU")
    parser.add_argument(
        "--liger",
        choices=("off", "flce", "defaults"),
        default="off",
        help="flce: Liger's fused linear cross-entropy alone; defaults: the family's default Liger kernels",
    )
    parser.add_argument("--shapes", nargs="+", default=DEFAULT_SHAPES, help="BATCHxSEQ rows; trainable uses the first")
    parser.add_argument("--gc", nargs="+", choices=("off", "on"), default=("off", "on"), help="Gradient checkpointing")
    parser.add_argument("--scopes", nargs="+", choices=SCOPES, default=SCOPES)
    parser.add_argument("--repeats", type=int, default=7, help="Measured steps; each is one sample")
    parser.add_argument("--warmup", type=int, default=3, help="Unmeasured steps first, past any kernel compile")
    args = parser.parse_args()

    torch.cuda.set_device(0)
    torch.manual_seed(SEED)
    model_path = MODEL_CONFIGS[args.model]["hf_name"]
    record = {
        **run_record(),
        "experiment": args.experiment,
        "model_path": model_path,
        "liger": args.liger,
        "rows": [],
    }
    {"activations": _activations, "trainable": _trainable, "padding": _padding}[args.experiment](
        args, model_path, record
    )


if __name__ == "__main__":
    main()
