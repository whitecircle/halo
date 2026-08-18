#!/usr/bin/env python
"""
VLM Model Parallelism Patching Test.

Validates that Context Parallelism (CP) and Tensor Parallelism (TP) patching
work correctly for Vision-Language Models (VLMs), specifically Qwen3-VL-2B.

Sub-tests:
1. Model loading: Load VLM, verify architecture (has language model + vision encoder)
2. CP patching: Apply CP patching, verify forward pass works with dummy text input
3. TP patching: Apply TP via DTensor, verify forward pass works

Run with 2 GPUs:
    torchrun --nproc_per_node=2 \
        tests/gpu/parallelism/tp/test_vlm_parallelism.py

Requirements:
    - 2x GPUs with >=16GB memory each
    - Model: Qwen/Qwen3-VL-2B-Instruct (auto-downloaded)
"""

import contextlib
import os
import sys
import traceback

import torch
import torch.distributed as dist
from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor

from src.distributed.runtime import barrier
from tests.common.distributed import init_distributed, teardown_distributed
from tests.common.models import QWEN3_VL_2B
from tests.common.utils import cleanup_memory, gpu_mem_gb, log

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


def create_text_only_input(
    local_rank: int,
    seq_length: int = MAX_SEQ_LENGTH,
) -> dict[str, torch.Tensor]:
    """Create a simple text-only input (no images) for VLM forward pass.

    Uses random token IDs in a safe range to avoid special tokens.

    Args:
        local_rank: GPU device index.
        seq_length: Sequence length for the input.

    Returns:
        Dict with input_ids, attention_mask, and labels tensors.
    """
    device = f"cuda:{local_rank}"
    torch.manual_seed(SEED)
    input_ids = torch.randint(100, 30000, (1, seq_length), device=device)
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
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


def test_model_loading() -> tuple[bool, dict]:
    """
    Test 1: Load VLM model, verify architecture.

    Verifies the model has both a vision encoder and a language model backbone.

    Returns:
        (passed, metrics_dict)
    """
    dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    checks = {}

    log("\n" + "=" * 70)
    log("SUB-TEST 1: VLM Model Loading and Architecture Verification")
    log("=" * 70)

    log(f"\n  Loading model: {MODEL_NAME}")
    log(f"  GPU memory before load: {gpu_mem_gb():.1f}GB")

    try:
        model = AutoModelForImageTextToText.from_pretrained(
            MODEL_NAME,
            dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
            device_map={"": local_rank},
        )
    except Exception as e:
        log(f"\n  FATAL: Model loading failed: {e}")
        traceback.print_exc()
        return False, {"error": f"Model loading failed: {e}"}

    log(f"  GPU memory after load: {gpu_mem_gb():.1f}GB")

    total_params, trainable_params = count_parameters(model)
    log(f"  Total parameters: {total_params:,}")
    log(f"  Trainable parameters: {trainable_params:,}")

    checks["model_loaded"] = True
    log("  Model loaded: PASS")

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
    inputs = create_text_only_input(local_rank)

    try:
        with torch.no_grad():
            outputs = model(**inputs)
            loss = outputs.loss
            loss_value = loss.item()

        loss_ok = not (torch.isnan(torch.tensor(loss_value)) or torch.isinf(torch.tensor(loss_value)))
        checks["baseline_forward"] = loss_ok
        log(f"  Baseline forward loss: {loss_value:.6f} {'PASS' if loss_ok else 'FAIL'}")
    except Exception as e:
        log(f"  Baseline forward FAILED: {e}")
        traceback.print_exc()
        checks["baseline_forward"] = False
        loss_value = float("nan")

    all_passed = all(checks.values())
    log(f"\n  Sub-test 1 result: {'PASS' if all_passed else 'FAIL'}")

    del model
    if "outputs" in dir():
        del outputs
    cleanup_memory()

    return all_passed, {"checks": checks, "components": components, "loss": loss_value}


def test_cp_patching() -> tuple[bool, dict]:
    """
    Test 2: Apply CP patching, verify forward pass works.

    Loads the VLM model, patches it for context parallelism (Ulysses attention),
    and verifies that a forward pass produces a finite loss.

    Returns:
        (passed, metrics_dict)
    """
    dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    checks = {}

    log("\n" + "=" * 70)
    log(f"SUB-TEST 2: VLM Context Parallelism Patching (CP={CP_SIZE})")
    log("=" * 70)

    from src.distributed.context_parallel.config import CPConfig
    from src.distributed.context_parallel.wrapper import patch_model_for_cp

    log("\n  Loading model for CP patching...")
    log(f"  GPU memory before load: {gpu_mem_gb():.1f}GB")

    try:
        model = AutoModelForImageTextToText.from_pretrained(
            MODEL_NAME,
            dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
            device_map={"": local_rank},
        )
    except Exception as e:
        log(f"\n  FATAL: Model loading failed: {e}")
        traceback.print_exc()
        return False, {"error": f"Model loading failed: {e}"}

    log(f"  GPU memory after load: {gpu_mem_gb():.1f}GB")
    checks["model_loaded"] = True

    log(f"\n  Creating CP config (cp_size={CP_SIZE})...")
    try:
        cp_config = CPConfig(
            cp_size=CP_SIZE,
            world_size=dist.get_world_size(),
            gpus_per_node=dist.get_world_size(),  # Single-node test
        )
        checks["cp_config_created"] = True
        log("  CP config created: PASS")
        log(f"    cp_rank={cp_config.cp_rank}, cp_group_idx={cp_config.cp_group_idx}")
    except Exception as e:
        log(f"  CP config creation FAILED: {e}")
        traceback.print_exc()
        checks["cp_config_created"] = False
        cleanup_memory()
        return False, {"checks": checks, "error": str(e)}

    log("\n  Applying CP patching (Ulysses attention)...")
    try:
        cp_model = patch_model_for_cp(model, cp_config)
        checks["cp_patching"] = True
        log("  CP patching applied: PASS")
        log(f"  Wrapper type: {type(cp_model).__name__}")

        is_wrapped = type(cp_model).__name__ == "UlyssesCPModelWrapper"
        checks["cp_wrapped"] = is_wrapped
        log(f"  Model wrapped as UlyssesCPModelWrapper: {'PASS' if is_wrapped else 'FAIL'}")

        if is_wrapped:
            log(f"    cp_size={cp_model.cp_size}, cp_rank={cp_model.cp_rank}")

    except Exception as e:
        log(f"  CP patching FAILED: {e}")
        traceback.print_exc()
        checks["cp_patching"] = False
        checks["cp_wrapped"] = False
        log("  NOTE: CP patching failure may indicate the model's attention architecture")
        log("  is not yet supported for Ulysses CP. This is an expected limitation.")

        cleanup_memory()
        return False, {"checks": checks, "error": str(e)}

    log("\n  Running CP forward pass (text-only)...")
    cp_model.eval()

    # Full-length input: the wrapper shards the sequence, so it must divide cp_size.
    seq_length = MAX_SEQ_LENGTH
    assert seq_length % CP_SIZE == 0, f"seq_length {seq_length} must be divisible by cp_size {CP_SIZE}"

    device = f"cuda:{local_rank}"
    torch.manual_seed(SEED)
    input_ids = torch.randint(100, 30000, (1, seq_length), device=device)
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()

    dist.broadcast(input_ids, src=0)
    dist.broadcast(attention_mask, src=0)
    dist.broadcast(labels, src=0)

    try:
        with torch.no_grad():
            outputs = cp_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss if hasattr(outputs, "loss") else outputs.get("loss", None)
            if loss is not None:
                loss_value = loss.item()
            else:
                loss_value = float("nan")
                log("  WARNING: No loss returned from CP forward pass")

        loss_ok = not (torch.isnan(torch.tensor(loss_value)) or torch.isinf(torch.tensor(loss_value)))
        checks["cp_forward"] = loss_ok
        log(f"  CP forward loss: {loss_value:.6f} {'PASS' if loss_ok else 'FAIL'}")

        loss_tensor = torch.tensor([loss_value], device=device)
        all_losses = [torch.zeros(1, device=device) for _ in range(dist.get_world_size())]
        dist.all_gather(all_losses, loss_tensor)
        all_losses_values = [t.item() for t in all_losses]
        log(f"  CP losses per rank: {[f'{l:.6f}' for l in all_losses_values]}")

        # Per-rank losses differ (different chunks); only finiteness is asserted here.
        all_finite = all(not (torch.isnan(torch.tensor(l)) or torch.isinf(torch.tensor(l))) for l in all_losses_values)
        checks["cp_all_ranks_finite"] = all_finite
        log(f"  All ranks finite: {'PASS' if all_finite else 'FAIL'}")

    except Exception as e:
        log(f"  CP forward pass FAILED: {e}")
        traceback.print_exc()
        checks["cp_forward"] = False
        checks["cp_all_ranks_finite"] = False
        loss_value = float("nan")

    all_passed = all(checks.values())
    log(f"\n  Sub-test 2 result: {'PASS' if all_passed else 'FAIL'}")

    del cp_model, model
    if "outputs" in dir():
        del outputs
    cleanup_memory()

    return all_passed, {"checks": checks, "loss": loss_value}


def test_tp_patching() -> tuple[bool, dict]:
    """
    Test 3: Apply TP via DTensor, verify forward pass works.

    Loads the VLM model on CPU, applies selective TP to attention layers
    via apply_tp_to_attention_only, moves to GPU, and verifies forward pass.

    Returns:
        (passed, metrics_dict)
    """
    dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    checks = {}

    log("\n" + "=" * 70)
    log(f"SUB-TEST 3: VLM Tensor Parallelism Patching (TP={TP_SIZE})")
    log("=" * 70)

    from src.distributed.mesh import create_dp_tp_mesh
    from src.distributed.tensor_parallel.parallelize_attention import apply_tp_to_attention_only

    # CPU first: DTensor parallelization requires it.
    log("\n  Loading model on CPU for TP patching...")
    log(f"  GPU memory before load: {gpu_mem_gb():.1f}GB")

    try:
        model = AutoModelForImageTextToText.from_pretrained(
            MODEL_NAME,
            dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
            device_map="cpu",
        )
    except Exception as e:
        log(f"\n  FATAL: Model loading failed: {e}")
        traceback.print_exc()
        return False, {"error": f"Model loading failed: {e}"}

    checks["model_loaded"] = True
    log("  Model loaded on CPU: PASS")

    log(f"\n  Creating TP mesh (tp_size={TP_SIZE})...")
    try:
        tp_mesh = create_dp_tp_mesh(tp_size=TP_SIZE)
        checks["tp_mesh_created"] = True
        log("  TP mesh created: PASS")
        log(f"    mesh={tp_mesh}")
    except Exception as e:
        log(f"  TP mesh creation FAILED: {e}")
        traceback.print_exc()
        checks["tp_mesh_created"] = False
        cleanup_memory()
        return False, {"checks": checks, "error": str(e)}

    log("\n  Applying TP to attention layers...")
    try:
        # For VLMs, TP applies to the language model's attention layers only.
        num_tp_modules = apply_tp_to_attention_only(model, tp_mesh)
        checks["tp_applied"] = num_tp_modules > 0
        log(f"  TP applied to {num_tp_modules} modules: {'PASS' if num_tp_modules > 0 else 'FAIL'}")
    except Exception as e:
        log(f"  TP application FAILED: {e}")
        traceback.print_exc()
        checks["tp_applied"] = False

        # Most likely cause: the attention class is absent from TP_SHARDABLE_ATTENTION_CLASSES.
        attn_layers = find_attention_layers(model)
        attn_types = {cls for _, cls in attn_layers}
        log(f"  Model attention classes: {attn_types}")
        log(
            "  NOTE: These may need to be added to TP_SHARDABLE_ATTENTION_CLASSES in src/distributed/tensor_parallel/module_types.py"
        )

        cleanup_memory()
        return False, {"checks": checks, "error": str(e)}

    log("\n  Moving TP-patched model to GPU...")
    try:
        model = model.to(f"cuda:{local_rank}")
        checks["tp_model_on_gpu"] = True
        log("  Model on GPU: PASS")
        log(f"  GPU memory after TP: {gpu_mem_gb():.1f}GB")
    except Exception as e:
        log(f"  Moving to GPU FAILED: {e}")
        traceback.print_exc()
        checks["tp_model_on_gpu"] = False
        cleanup_memory()
        return False, {"checks": checks, "error": str(e)}

    log("\n  Running TP forward pass (text-only)...")
    model.eval()

    device = f"cuda:{local_rank}"
    torch.manual_seed(SEED)
    input_ids = torch.randint(100, 30000, (1, MAX_SEQ_LENGTH), device=device)
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()

    dist.broadcast(input_ids, src=0)
    dist.broadcast(attention_mask, src=0)
    dist.broadcast(labels, src=0)

    try:
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss
            loss_value = loss.item()

        loss_ok = not (torch.isnan(torch.tensor(loss_value)) or torch.isinf(torch.tensor(loss_value)))
        checks["tp_forward"] = loss_ok
        log(f"  TP forward loss: {loss_value:.6f} {'PASS' if loss_ok else 'FAIL'}")

        # TP is a sharded rearrangement of one computation, so every rank must agree.
        loss_tensor = torch.tensor([loss_value], device=device)
        all_losses = [torch.zeros(1, device=device) for _ in range(dist.get_world_size())]
        dist.all_gather(all_losses, loss_tensor)
        all_losses_values = [t.item() for t in all_losses]
        log(f"  TP losses per rank: {[f'{l:.6f}' for l in all_losses_values]}")

        if len(all_losses_values) >= 2:
            max_diff = max(abs(all_losses_values[i] - all_losses_values[0]) for i in range(1, len(all_losses_values)))
            losses_match = max_diff < 1e-2
            checks["tp_losses_consistent"] = losses_match
            log(f"  TP loss consistency (max_diff={max_diff:.2e}): {'PASS' if losses_match else 'FAIL'}")
        else:
            checks["tp_losses_consistent"] = True

    except Exception as e:
        log(f"  TP forward pass FAILED: {e}")
        traceback.print_exc()
        checks["tp_forward"] = False
        checks["tp_losses_consistent"] = False
        loss_value = float("nan")

    all_passed = all(checks.values())
    log(f"\n  Sub-test 3 result: {'PASS' if all_passed else 'FAIL'}")

    del model
    if "outputs" in dir():
        del outputs
    cleanup_memory()

    return all_passed, {"checks": checks, "loss": loss_value, "num_tp_modules": num_tp_modules}


def main() -> int:
    rank, world_size, local_rank = init_distributed()

    log(f"\n{'#' * 70}")
    log("  VLM Parallelism Patching Test")
    log(f"  World size: {world_size}, CP size: {CP_SIZE}, TP size: {TP_SIZE}")
    log(f"  Model: {MODEL_NAME}")
    log(f"  GPU: {torch.cuda.get_device_name(local_rank)}")
    log(f"  GPU memory: {torch.cuda.get_device_properties(local_rank).total_memory / 1e9:.1f}GB")
    log(f"{'#' * 70}")

    if world_size != 2:
        log(f"\nERROR: This test requires exactly 2 GPUs, got {world_size}")
        teardown_distributed()
        return 1

    results = {}
    all_passed = True

    try:
        log("\nEnsuring model is downloaded...")
        if rank == 0:
            AutoConfig.from_pretrained(MODEL_NAME, trust_remote_code=True)
            # Processor download may fail for some VLMs; the tokenizer is sufficient here.
            with contextlib.suppress(Exception):
                AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)

        barrier()

        from accelerate import PartialState

        PartialState()

        t1_passed, t1_metrics = test_model_loading()
        results["1_model_loading"] = {"passed": t1_passed, "metrics": t1_metrics}
        if not t1_passed:
            all_passed = False

        barrier()
        cleanup_memory()

        t2_passed, t2_metrics = test_cp_patching()
        results["2_cp_patching"] = {"passed": t2_passed, "metrics": t2_metrics}
        if not t2_passed:
            all_passed = False

        barrier()
        cleanup_memory()

        t3_passed, t3_metrics = test_tp_patching()
        results["3_tp_patching"] = {"passed": t3_passed, "metrics": t3_metrics}
        if not t3_passed:
            all_passed = False

    except Exception as e:
        log(f"\nFATAL ERROR: {e}")
        traceback.print_exc()
        all_passed = False

    barrier()

    log(f"\n{'#' * 70}")
    log("  FINAL RESULTS")
    log(f"{'#' * 70}")

    for test_name, result in results.items():
        status = "PASS" if result["passed"] else "FAIL"
        log(f"  {test_name}: {status}")

    if all_passed:
        log("\n  ALL TESTS PASSED")
    else:
        log("\n  SOME TESTS FAILED")

    log(f"{'#' * 70}\n")

    barrier()
    teardown_distributed()

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
