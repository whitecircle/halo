#!/usr/bin/env python
"""ZAYA1-8B end-to-end load + forward + backward smoke test (single GPU).

What this validates:
  1. ``load_distributed_model`` resolves the native ``ZayaForCausalLM``, applies the toolkit's
     Zaya patches, and materialises the checkpoint on GPU with the hub's fused 3D
     ``gate_up_proj`` / ``down_proj`` expert parameters.
  2. ``ZayaModel.forward`` runs without error: CCA + ResidualScaling +
     Router (with EDA cross-layer state) + Experts all yield finite
     activations.
  3. ``loss.backward()`` populates gradients on every leaf parameter
     (router, experts, CCA projections, residual scaling, embeddings,
     final norm) and the gradients are finite.
  4. One AdamW optimizer step moves the parameters (the L2 distance
     between pre- and post-step weights is strictly positive).
  5. Gradient checkpointing is REFUSED (``apply_zaya_patches`` clears
     ``supports_gradient_checkpointing``: the recompute faults in cuDNN on the CCA Conv1d pair).

Single-GPU, no torchrun. Plain DDP / FSDP / EP / TP / CP wrappers are
exercised by separate tests; this one isolates the modeling + patch path.

Memory budget (B300, 275 GB HBM):
  weights bf16: ~17 GB
  AdamW master + state (fp32): ~70 GB (Param + m + v at fp32, then bf16 copy)
  fwd activations (8-token batch): a few hundred MB
  Total: ~90 GB, comfortably below the 275 GB ceiling.

Run:
    docker run --rm --gpus '"device=0"' \\
        -v $(pwd):/workspace -w /workspace \\
        halo:blackwell \\
        python tests/gpu/trainers/sft/test_zaya_load_forward_backward.py
"""

import sys

import torch
from transformers import AutoConfig

from src.distributed.expert_parallel.layers.zaya import EPZayaMoELayer
from src.distributed.loading.model_loading import load_distributed_model
from src.distributed.parallelism_config import ParallelismConfig
from src.env import env_str
from tests.common.models import ZAYA_8B
from tests.common.utils import log

MODEL = env_str("HALO_TEST_ZAYA_MODEL", ZAYA_8B)
# Use a small batch so the test stays fast and keeps headroom on smaller GPUs.
BATCH = 1
SEQ = 16
# One live parameter path per module family the optimizer step must move — attention, router,
# residual scaling, norm. Every one is asserted to match below: a renamed module would otherwise
# shrink the sample silently instead of failing.
SAMPLE_PARAM_SUFFIXES = (
    "self_attn.qkv_proj.q_proj.weight",
    "mlp.gate.down_proj.weight",
    "post_mlp_residual_scale.hidden_states_bias",
    "input_layernorm.weight",
)


def main() -> int:
    if not torch.cuda.is_available():
        log("SKIP: no CUDA")  # the sentinel the launcher skips on; a bare exit 0 reads as a PASS
        return 0

    from accelerate import PartialState

    # Production always has accelerate state up before the loader (TrainingArguments initializes
    # it); the loader's hub-download path logs through accelerate's logger, which raises otherwise.
    PartialState()

    log("=" * 70)
    log("  ZAYA1-8B load + fwd + bwd smoke test")
    log(f"  Model: {MODEL}")
    log(f"  GPU:   {torch.cuda.get_device_name(0)}")
    log(f"  Batch / Seq: {BATCH} / {SEQ}")
    log("=" * 70)

    # ─── 1. AutoConfig resolves to the native ZayaConfig ────────────────────
    log("\n[1/6] Loading config...")
    cfg = AutoConfig.from_pretrained(MODEL)
    assert cfg.__class__.__name__ == "ZayaConfig", f"Expected ZayaConfig, got {type(cfg).__name__}"
    assert type(cfg).__module__.startswith("transformers."), (
        f"ZayaConfig came from {type(cfg).__module__} — the native class must win over remote code"
    )
    log(f"  ✓ Config class: {type(cfg).__name__} ({type(cfg).__module__})")
    log(
        f"  ✓ Layers: {cfg.num_hidden_layers}, Experts: {cfg.num_experts}, "
        f"Hidden: {cfg.hidden_size}, Vocab: {cfg.vocab_size}"
    )

    # ─── 2. Load model through the toolkit loader (bf16) ────────────────────
    # Not a bare from_pretrained: the toolkit loader is what applies the Zaya patches (load
    # recording, the GC refusal asserted in step 6, flash position_ids), so this exercises the path
    # every training run takes.
    log("\n[2/6] Loading model weights (bf16, single GPU)...")
    model, tokenizer = load_distributed_model(
        model_name_or_path=MODEL,
        parallelism_config=ParallelismConfig(),
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    n_params = sum(p.numel() for p in model.parameters())
    mem_after_load = torch.cuda.memory_allocated() / 1e9
    log(f"  ✓ Model loaded: {n_params / 1e9:.2f}B params")
    log(f"  ✓ HBM after load: {mem_after_load:.2f} GB")
    assert n_params > 8_000_000_000, f"Suspiciously few params: {n_params}"

    # The production ep1 path wraps every ZayaSparseMoeBlock into EPZayaMoELayer (grouped-GEMM),
    # which stores the fused experts in matmul convention: gate_up [E, H, 2M], down [E, M, H].
    first_block = None
    for module in model.modules():
        if isinstance(module, EPZayaMoELayer):
            first_block = module
            break
    assert first_block is not None, "No EPZayaMoELayer found — ep1 grouped-GEMM wrapping failed"
    gate_up = first_block.gate_up_proj
    down = first_block.down_proj
    log(f"  ✓ Expert gate_up_proj shape: {tuple(gate_up.shape)} (E×H×2M)")
    log(f"  ✓ Expert down_proj shape:    {tuple(down.shape)} (E×M×H)")
    assert gate_up.dim() == 3 and gate_up.shape[0] == cfg.num_experts
    assert down.dim() == 3 and down.shape[0] == cfg.num_experts

    # ─── 3. Forward pass ────────────────────────────────────────────────────
    log("\n[3/6] Forward pass...")
    # Use a real prompt rendered through the bundled chat template — better than
    # arbitrary token ids since CCA's Conv1d cares about contiguous structure.
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Say hi briefly."}],
        add_generation_prompt=True,
        tokenize=False,
        enable_thinking=False,
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(0)
    if inputs.input_ids.shape[1] > SEQ * 4:
        # Truncate so the activation memory stays bounded.
        inputs.input_ids = inputs.input_ids[:, : SEQ * 4]
        if "attention_mask" in inputs:
            inputs.attention_mask = inputs.attention_mask[:, : SEQ * 4]
    model.train()
    out = model(**inputs, labels=inputs.input_ids.clone(), use_cache=False)
    assert out.loss is not None, "No loss returned"
    assert torch.isfinite(out.loss), f"Non-finite loss: {out.loss.item()}"
    assert torch.isfinite(out.logits).all(), "Non-finite logits"
    log(f"  ✓ Forward OK — loss {out.loss.item():.4f}, logits {tuple(out.logits.shape)} dtype {out.logits.dtype}")
    log(f"  ✓ HBM peak after fwd: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

    # ─── 4. Backward pass ───────────────────────────────────────────────────
    log("\n[4/6] Backward pass...")
    out.loss.backward()
    grad_stats = {
        "total_grad_params": 0,
        "missing_grad_params": 0,
        "nonfinite_grad_params": 0,
        "max_abs_grad": 0.0,
    }
    sample_names: list[str] = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        grad_stats["total_grad_params"] += 1
        if p.grad is None:
            grad_stats["missing_grad_params"] += 1
            continue
        if not torch.isfinite(p.grad).all():
            grad_stats["nonfinite_grad_params"] += 1
        m = p.grad.detach().abs().max().item()
        grad_stats["max_abs_grad"] = max(grad_stats["max_abs_grad"], m)
        if len(sample_names) < 6 and (
            "gate.router_mlp" in name
            or "experts.gate_up" in name
            or "experts.down" in name
            or "self_attn.qkv_proj.q_proj" in name
            or "residual_scale" in name
            or "input_layernorm" in name
        ):
            sample_names.append(f"    {name}: max|grad|={m:.3e}")
    log(
        f"  ✓ {grad_stats['total_grad_params'] - grad_stats['missing_grad_params']}/"
        f"{grad_stats['total_grad_params']} params got grads"
    )
    log(f"  ✓ max |grad| across model: {grad_stats['max_abs_grad']:.3e}")
    for s in sample_names:
        log(s)
    assert grad_stats["missing_grad_params"] == 0, f"{grad_stats['missing_grad_params']} params missed gradients"
    assert grad_stats["nonfinite_grad_params"] == 0, (
        f"{grad_stats['nonfinite_grad_params']} params got non-finite gradients"
    )
    assert grad_stats["max_abs_grad"] > 0, "Gradients are identically zero — backward did nothing"

    # ─── 5. Optimizer step moves the parameters ─────────────────────────────
    # Use lr=1.0 (a *visibility* lr, not a training one) — at lr=1e-4 with
    # bf16 parameters the Adam updates on slow-moving weights (embeddings,
    # router) underflow below bf16's ~7e-3 epsilon and round to zero on the
    # weight tensor even though they're computed correctly internally. This
    # would be a false negative for "did the step do anything".
    log("\n[5/6] Optimizer step (AdamW, lr=1.0 — visibility check)...")

    snapshot = {
        name: p.detach().clone()
        for name, p in model.named_parameters()
        if p.requires_grad and name.endswith(SAMPLE_PARAM_SUFFIXES)
    }
    for suffix in SAMPLE_PARAM_SUFFIXES:
        assert any(name.endswith(suffix) for name in snapshot), (
            f"no trainable parameter ends in {suffix!r} — the modeling renamed it and this sample went stale"
        )
    opt = torch.optim.AdamW(model.parameters(), lr=1.0)
    opt.step()
    moved = 0
    after_map = dict(model.named_parameters())
    for name, before in snapshot.items():
        delta = (after_map[name].detach() - before).abs().max().item()
        if delta > 0:
            moved += 1
    log(f"  ✓ {moved}/{len(snapshot)} sampled params moved after step")
    assert moved == len(snapshot), f"AdamW step did not move {len(snapshot) - moved} of {len(snapshot)} sampled params"

    # ─── 6. Gradient checkpointing is refused ───────────────────────────────
    # The patch flips the class attribute so transformers raises up front, instead of the run
    # faulting mid-backward in cuDNN on the CCA Conv1d pair.
    log("\n[6/6] Gradient checkpointing refusal...")
    try:
        model.gradient_checkpointing_enable()
    except ValueError as exc:
        log(f"  ✓ Refused: {exc}")
    else:
        raise AssertionError(
            "gradient_checkpointing_enable() was accepted — apply_zaya_patches did not reach this model"
        )

    log("\n" + "=" * 70)
    log("  ✓ ALL CHECKS PASSED")
    log("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
