#!/usr/bin/env python
"""
VLM Model Parallelism Patching Test.

Validates that Context Parallelism (CP) and Tensor Parallelism (TP) patching
work correctly for Vision-Language Models (VLMs), specifically Qwen3-VL-2B.

Sub-tests, all on one text-only batch shared by every rank:
1. Model loading: Load VLM, verify architecture (has language model + vision encoder), and take the
   unpatched forward loss as the baseline
2. CP patching: Apply CP patching; the CP loss must match the baseline
3. TP patching: Apply TP via DTensor; the TP loss must match the baseline and agree across ranks

Run with 2 GPUs:
    torchrun --nproc_per_node=2 \
        tests/gpu/parallelism/tp/test_vlm_parallelism.py

Requirements:
    - 2x GPUs with >=16GB memory each
    - Model: Qwen/Qwen3-VL-2B-Instruct (auto-downloaded)
"""

import math

import torch
import torch.distributed as dist
from transformers import AutoModelForImageTextToText

from src.distributed.context_parallel.config import CPConfig
from src.distributed.context_parallel.wrapper import UlyssesCPModelWrapper, patch_model_for_cp
from src.distributed.mesh import create_dp_tp_mesh
from src.distributed.tensor_parallel.parallelize_attention import apply_tp_to_attention_only
from tests.common.distributed import ensure_model_downloaded, world_mean, world_spread
from tests.common.harness import gpu_test_main
from tests.common.models import QWEN3_VL_2B
from tests.common.tolerances import TOL
from tests.common.utils import cleanup_memory, gpu_mem_gb, log, log_all

MODEL_NAME = QWEN3_VL_2B
CP_SIZE = 2
TP_SIZE = 2
MAX_SEQ_LENGTH = 64
SEED = 42


def count_parameters(model: torch.nn.Module) -> tuple[int, int]:
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def create_text_only_input(device: str) -> dict[str, torch.Tensor]:
    """The one text-only batch (no images) every sub-test runs, identical on every rank.

    Uses random token IDs in a safe range to avoid special tokens. Broadcast from rank 0, so the CP
    and TP losses compare against the baseline on the same tokens.
    """
    torch.manual_seed(SEED)
    input_ids = torch.randint(100, 30000, (1, MAX_SEQ_LENGTH), device=device)
    dist.broadcast(input_ids, src=0)
    return {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
        "labels": input_ids.clone(),
    }


def find_attention_layers(model: torch.nn.Module) -> list[tuple[str, str]]:
    """Find all attention layers in the model and return (path, class_name) pairs."""
    attn_layers = []
    for name, module in model.named_modules():
        cls_name = type(module).__name__
        if "Attention" in cls_name or "attention" in cls_name:
            attn_layers.append((name, cls_name))
    return attn_layers


def find_vision_components(model: torch.nn.Module) -> dict[str, bool]:
    """Detect vision encoder and language model components in a VLM."""
    components = {
        "has_visual": False,
        "has_language_model": False,
        "has_lm_head": False,
        "visual_class": None,
        "language_model_class": None,
    }

    # transformers >=5 nests the vision tower and text backbone one level below the top-level
    # wrapper, so walk recursively and take the top-most (fewest-dotted) match.
    def _topmost(predicate) -> str | None:
        matches = [(name, type(m).__name__) for name, m in model.named_modules() if name and predicate(name.lower())]
        if not matches:
            return None
        return min(matches, key=lambda nc: nc[0].count("."))[1]

    components["visual_class"] = _topmost(lambda n: "visual" in n or "vision" in n)
    components["has_visual"] = components["visual_class"] is not None
    # Prefer an explicit language_model submodule; fall back to a non-vision backbone module.
    components["language_model_class"] = _topmost(lambda n: "language_model" in n) or _topmost(
        lambda n: "model" in n and "visual" not in n and "vision" not in n
    )
    components["has_language_model"] = components["language_model_class"] is not None
    components["has_lm_head"] = any(name and "lm_head" in name.lower() for name, _ in model.named_modules())

    return components


def test_model_loading(inputs: dict[str, torch.Tensor], local_rank: int) -> tuple[dict[str, bool], float]:
    """
    Test 1: Load VLM model, verify architecture, and return the unpatched forward loss.

    Verifies the model has both a vision encoder and a language model backbone.
    """
    checks = {}

    log("\n" + "=" * 70)
    log("SUB-TEST 1: VLM Model Loading and Architecture Verification")
    log("=" * 70)

    log(f"\n  Loading model: {MODEL_NAME}")
    log(f"  GPU memory before load: {gpu_mem_gb():.1f}GB")

    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_NAME,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
        device_map={"": local_rank},
    )

    log(f"  GPU memory after load: {gpu_mem_gb():.1f}GB")

    total_params, trainable_params = count_parameters(model)
    log(f"  Total parameters: {total_params:,}")
    log(f"  Trainable parameters: {trainable_params:,}")

    components = find_vision_components(model)

    checks["has_visual"] = components["has_visual"]
    checks["has_language_model"] = components["has_language_model"]

    log("\n  Architecture analysis:")
    log(f"  Has vision encoder: {'PASS' if components['has_visual'] else 'FAIL'} ({components['visual_class']})")
    log(
        f"  Has language model: {'PASS' if components['has_language_model'] else 'FAIL'}"
        f" ({components['language_model_class']})"
    )
    log(f"  Has lm_head: {components['has_lm_head']}")

    log(f"\n  Model type: {model.config.model_type}")
    if hasattr(model.config, "text_config"):
        text_cfg = model.config.text_config
        log(f"  Text model type: {getattr(text_cfg, 'model_type', 'N/A')}")
        log(f"  Hidden size: {getattr(text_cfg, 'hidden_size', 'N/A')}")
        log(f"  Num attention heads: {getattr(text_cfg, 'num_attention_heads', 'N/A')}")
        log(f"  Num KV heads: {getattr(text_cfg, 'num_key_value_heads', 'N/A')}")
        log(f"  Num hidden layers: {getattr(text_cfg, 'num_hidden_layers', 'N/A')}")
    if hasattr(model.config, "vision_config"):
        vis_cfg = model.config.vision_config
        log(f"  Vision model type: {getattr(vis_cfg, 'model_type', 'N/A')}")

    attn_layers = find_attention_layers(model)
    attn_types = {cls for _, cls in attn_layers}
    log(f"\n  Attention layer types: {attn_types}")
    log(f"  Total attention layers: {len(attn_layers)}")

    log("\n  Running baseline forward pass (text-only)...")
    model.eval()
    with torch.no_grad():
        loss_value = model(**inputs).loss.item()

    checks["baseline_forward"] = math.isfinite(loss_value)
    log(f"  Baseline forward loss: {loss_value:.6f} {'PASS' if checks['baseline_forward'] else 'FAIL'}")

    del model
    cleanup_memory()

    return checks, loss_value


def test_cp_patching(inputs: dict[str, torch.Tensor], local_rank: int, baseline_loss: float) -> dict[str, bool]:
    """
    Test 2: Apply CP patching; the CP forward loss must match the unpatched baseline.

    Loads the VLM model, patches it for context parallelism (Ulysses attention),
    and runs the shared batch through it.
    """
    checks = {}

    log("\n" + "=" * 70)
    log(f"SUB-TEST 2: VLM Context Parallelism Patching (CP={CP_SIZE})")
    log("=" * 70)

    log("\n  Loading model for CP patching...")
    log(f"  GPU memory before load: {gpu_mem_gb():.1f}GB")

    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_NAME,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
        device_map={"": local_rank},
    )

    log(f"  GPU memory after load: {gpu_mem_gb():.1f}GB")

    log(f"\n  Creating CP config (cp_size={CP_SIZE})...")
    cp_config = CPConfig(
        cp_size=CP_SIZE,
        world_size=dist.get_world_size(),
        gpus_per_node=dist.get_world_size(),  # Single-node test
    )
    log(f"    cp_rank={cp_config.cp_rank}, cp_group_idx={cp_config.cp_group_idx}")

    log("\n  Applying CP patching (Ulysses attention)...")
    cp_model = patch_model_for_cp(model, cp_config)
    log(f"  Wrapper type: {type(cp_model).__name__}")

    is_wrapped = isinstance(cp_model, UlyssesCPModelWrapper)
    checks["cp_wrapped"] = is_wrapped
    log(f"  Model wrapped as UlyssesCPModelWrapper: {'PASS' if is_wrapped else 'FAIL'}")

    log("\n  Running CP forward pass (text-only)...")
    cp_model.eval()

    # Full-length input: the wrapper shards the sequence, so it must divide cp_size.
    assert MAX_SEQ_LENGTH % CP_SIZE == 0, f"seq_length {MAX_SEQ_LENGTH} must be divisible by cp_size {CP_SIZE}"

    with torch.no_grad():
        loss_value = cp_model(**inputs).loss.item()

    checks["cp_forward"] = math.isfinite(loss_value)
    log_all(f"  CP forward loss: {loss_value:.6f} {'PASS' if checks['cp_forward'] else 'FAIL'}")

    # Same weights, same batch, no step taken: CP's global normalization equals the baseline mean.
    cp_loss_avg = world_mean(loss_value)
    cp_diff = abs(cp_loss_avg - baseline_loss)
    checks["cp_loss_matches_baseline"] = cp_diff < TOL.parallel_vs_baseline_loss_abs
    log(
        f"  CP loss (avg) {cp_loss_avg:.6f} vs baseline {baseline_loss:.6f}: |diff|={cp_diff:.2e} "
        f"(tol {TOL.parallel_vs_baseline_loss_abs}) {'PASS' if checks['cp_loss_matches_baseline'] else 'FAIL'}"
    )

    log(f"\n  Sub-test 2 result: {'PASS' if all(checks.values()) else 'FAIL'}")

    del cp_model, model
    cleanup_memory()

    return checks


def test_tp_patching(inputs: dict[str, torch.Tensor], local_rank: int, baseline_loss: float) -> dict[str, bool]:
    """
    Test 3: Apply TP via DTensor; the TP forward loss must match the unpatched baseline.

    Loads the VLM model on CPU, applies selective TP to attention layers
    via apply_tp_to_attention_only, moves to GPU, and runs the shared batch through it.
    """
    checks = {}

    log("\n" + "=" * 70)
    log(f"SUB-TEST 3: VLM Tensor Parallelism Patching (TP={TP_SIZE})")
    log("=" * 70)

    # CPU first: DTensor parallelization requires it.
    log("\n  Loading model on CPU for TP patching...")
    log(f"  GPU memory before load: {gpu_mem_gb():.1f}GB")

    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_NAME,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
        device_map="cpu",
    )

    log(f"\n  Creating TP mesh (tp_size={TP_SIZE})...")
    tp_mesh = create_dp_tp_mesh(tp_size=TP_SIZE)
    log(f"    mesh={tp_mesh}")

    log("\n  Applying TP to attention layers...")
    # For VLMs, TP applies to the language model's attention layers only.
    num_tp_modules = apply_tp_to_attention_only(model, tp_mesh)
    checks["tp_applied"] = num_tp_modules > 0
    log(f"  TP applied to {num_tp_modules} modules: {'PASS' if num_tp_modules > 0 else 'FAIL'}")

    log("\n  Moving TP-patched model to GPU...")
    model = model.to(f"cuda:{local_rank}")
    log(f"  GPU memory after TP: {gpu_mem_gb():.1f}GB")

    log("\n  Running TP forward pass (text-only)...")
    model.eval()

    with torch.no_grad():
        loss_value = model(**inputs).loss.item()

    checks["tp_forward"] = math.isfinite(loss_value)
    log_all(f"  TP forward loss: {loss_value:.6f} {'PASS' if checks['tp_forward'] else 'FAIL'}")

    # TP is a sharded rearrangement of one computation, so every rank must agree.
    spread = world_spread(loss_value)
    checks["tp_losses_consistent"] = spread < TOL.ep_identical_batch_rank_spread_abs
    log(f"  TP loss consistency (spread={spread:.2e}): {'PASS' if checks['tp_losses_consistent'] else 'FAIL'}")

    tp_diff = abs(loss_value - baseline_loss)
    checks["tp_loss_matches_baseline"] = tp_diff < TOL.parallel_vs_baseline_loss_abs
    log(
        f"  TP loss {loss_value:.6f} vs baseline {baseline_loss:.6f}: |diff|={tp_diff:.2e} "
        f"(tol {TOL.parallel_vs_baseline_loss_abs}) {'PASS' if checks['tp_loss_matches_baseline'] else 'FAIL'}"
    )

    log(f"\n  Sub-test 3 result: {'PASS' if all(checks.values()) else 'FAIL'}")

    del model
    cleanup_memory()

    return checks


@gpu_test_main(exact_world_size=2, prefix="vlm_parallelism")
def run(ctx):
    log(f"\n{'#' * 70}")
    log("  VLM Parallelism Patching Test")
    log(f"  World size: {ctx.world_size}, CP size: {CP_SIZE}, TP size: {TP_SIZE}")
    log(f"  Model: {MODEL_NAME}")
    log(f"  GPU: {torch.cuda.get_device_name(ctx.local_rank)}")
    log(f"  GPU memory: {torch.cuda.get_device_properties(ctx.local_rank).total_memory / 1e9:.1f}GB")
    log(f"{'#' * 70}")

    log("\nEnsuring model is downloaded...")
    ensure_model_downloaded(MODEL_NAME, ctx.rank)

    inputs = create_text_only_input(f"cuda:{ctx.local_rank}")

    checks, baseline_loss = test_model_loading(inputs, ctx.local_rank)
    ctx.barrier()
    cleanup_memory()

    checks.update(test_cp_patching(inputs, ctx.local_rank, baseline_loss))
    ctx.barrier()
    cleanup_memory()

    checks.update(test_tp_patching(inputs, ctx.local_rank, baseline_loss))

    return {"checks": checks}


if __name__ == "__main__":
    run()
