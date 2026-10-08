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

Each stage builds on the one before, so a failed stage ends the run with the checks so far. FSDP2
(``test_zaya_fsdp.py``) and EP (``tests/gpu/parallelism/ep/test_zaya_ep*.py``) are exercised by
separate tests; this one isolates the modeling + patch path.

Memory budget (B300, 288 GB HBM):
  weights bf16: ~17 GB
  AdamW master + state (fp32): ~70 GB (Param + m + v at fp32, then bf16 copy)
  fwd activations (8-token batch): a few hundred MB
  Total: ~90 GB, comfortably below the 288 GB ceiling.

Run:
    torchrun --nproc_per_node=1 tests/gpu/trainers/sft/test_zaya_load_forward_backward.py
"""

import torch
from transformers import AutoConfig

from src.distributed.expert_parallel.layers.zaya import EPZayaMoELayer
from src.distributed.loading.model_loading import load_distributed_model
from src.distributed.parallelism_config import ParallelismConfig
from src.env import env_str
from tests.common.harness import gpu_test_main
from tests.common.models import ZAYA_8B
from tests.common.utils import log

MODEL = env_str("HALO_TEST_ZAYA_MODEL", ZAYA_8B)
# Truncation bound on the prompt, so the activation memory stays bounded.
MAX_PROMPT_TOKENS = 64
# More parameters than any truncated or partially-loaded ZAYA1-8B checkpoint holds.
MIN_PARAMS = 8_000_000_000
# One live parameter path per module family the optimizer step must move — attention, router,
# residual scaling, norm. Every one must be found: a renamed module would otherwise shrink the sample
# silently instead of failing.
SAMPLE_PARAM_SUFFIXES = (
    "self_attn.qkv_proj.q_proj.weight",
    "mlp.gate.down_proj.weight",
    "post_mlp_residual_scale.hidden_states_bias",
    "input_layernorm.weight",
)


@gpu_test_main(exact_world_size=1, prefix="zaya_load_forward_backward")
def run(ctx):
    log(f"  ZAYA1-8B load + fwd + bwd smoke test: {MODEL} on {torch.cuda.get_device_name(ctx.local_rank)}")
    checks: dict[str, bool] = {}

    # ─── 1. AutoConfig resolves to the native ZayaConfig ────────────────────
    cfg = AutoConfig.from_pretrained(MODEL)
    checks["native_config"] = type(cfg).__name__ == "ZayaConfig" and type(cfg).__module__.startswith("transformers.")
    log(f"  Config class: {type(cfg).__name__} ({type(cfg).__module__})")
    if not checks["native_config"]:
        return {"checks": checks}

    # ─── 2. Load through the toolkit loader (bf16) ──────────────────────────
    # Not a bare from_pretrained: the toolkit loader is what applies the Zaya patches (load
    # recording, the GC refusal asserted in stage 6, flash position_ids), so this exercises the path
    # every training run takes.
    model, tokenizer = load_distributed_model(
        model_name_or_path=MODEL,
        parallelism_config=ParallelismConfig(),
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    n_params = sum(p.numel() for p in model.parameters())
    log(f"  Model loaded: {n_params / 1e9:.2f}B params, HBM {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    checks["all_params_loaded"] = n_params > MIN_PARAMS
    # The production ep1 path wraps every ZayaSparseMoeBlock into EPZayaMoELayer (grouped-GEMM), which
    # stores the fused experts in matmul convention: gate_up [E, H, 2M], down [E, M, H].
    block = next((module for module in model.modules() if isinstance(module, EPZayaMoELayer)), None)
    checks["experts_wrapped_fused"] = block is not None and all(
        weight.dim() == 3 and weight.shape[0] == cfg.num_experts for weight in (block.gate_up_proj, block.down_proj)
    )
    if not (checks["all_params_loaded"] and checks["experts_wrapped_fused"]):
        return {"checks": checks}

    # ─── 3. Forward pass ────────────────────────────────────────────────────
    # A real prompt through the bundled chat template rather than arbitrary token ids, since CCA's
    # Conv1d cares about contiguous structure.
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Say hi briefly."}],
        add_generation_prompt=True,
        tokenize=False,
        enable_thinking=False,
    )
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids[:, :MAX_PROMPT_TOKENS].to(ctx.device)
    model.train()
    out = model(input_ids=input_ids, labels=input_ids.clone(), use_cache=False)
    checks["forward_finite"] = (
        out.loss is not None and bool(torch.isfinite(out.loss)) and bool(torch.isfinite(out.logits).all())
    )
    log(f"  Forward: loss {out.loss.item() if out.loss is not None else None}, logits {tuple(out.logits.shape)}")
    if not checks["forward_finite"]:
        return {"checks": checks}

    # ─── 4. Backward pass ───────────────────────────────────────────────────
    out.loss.backward()
    trainable = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    missing = [name for name, p in trainable if p.grad is None]
    nonfinite = [name for name, p in trainable if p.grad is not None and not torch.isfinite(p.grad).all()]
    max_abs_grad = max((p.grad.detach().abs().max().item() for _, p in trainable if p.grad is not None), default=0.0)
    log(
        f"  Backward: {len(trainable) - len(missing)}/{len(trainable)} params got grads, max |grad| {max_abs_grad:.3e}"
    )
    checks["every_param_gets_a_grad"] = not missing
    checks["grads_finite"] = not nonfinite
    checks["grads_nonzero"] = max_abs_grad > 0
    if missing or nonfinite:
        log(f"  Missing grads: {missing[:5]}; non-finite grads: {nonfinite[:5]}")

    # ─── 5. Optimizer step moves the parameters ─────────────────────────────
    # lr=1.0 is a visibility lr, not a training one: at 1e-4 the Adam updates on slow-moving bf16
    # weights (embeddings, router) round below bf16's epsilon, a false negative for "did it move".
    snapshot = {name: p.detach().clone() for name, p in trainable if name.endswith(SAMPLE_PARAM_SUFFIXES)}
    checks["sampled_params_present"] = all(any(name.endswith(s) for name in snapshot) for s in SAMPLE_PARAM_SUFFIXES)
    torch.optim.AdamW(model.parameters(), lr=1.0).step()
    after = dict(model.named_parameters())
    moved = sum((after[name].detach() - before).abs().max().item() > 0 for name, before in snapshot.items())
    checks["optimizer_step_moves_params"] = bool(snapshot) and moved == len(snapshot)
    log(f"  Optimizer step moved {moved}/{len(snapshot)} sampled params")

    # ─── 6. Gradient checkpointing is refused ───────────────────────────────
    # The patch flips the class attribute so transformers raises up front, instead of the run
    # faulting mid-backward in cuDNN on the CCA Conv1d pair.
    try:
        model.gradient_checkpointing_enable()
    except ValueError as exc:
        checks["gradient_checkpointing_refused"] = True
        log(f"  Gradient checkpointing refused: {exc}")
    else:
        checks["gradient_checkpointing_refused"] = False
        log("  gradient_checkpointing_enable() was accepted: apply_zaya_patches did not reach this model")
    return {"checks": checks}


if __name__ == "__main__":
    run()
