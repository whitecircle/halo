#!/usr/bin/env python
"""CP correctness for Qwen3.5 / Qwen3.6 full-attention layers.

Qwen3.5 attention has two distinguishing features:

1. **Double-width q_proj** — output is split into a query and a post-attention
   sigmoid gate (``attn_output * sigmoid(gate)`` before ``o_proj``). The Qwen3
   Ulysses wrapper would corrupt this.
2. **Linear-attention layers** (``Qwen3_5MoeGatedDeltaNet``) interleaved with
   full attention — Conv1d + recurrent scan over the sequence axis, which
   Ulysses cannot shard without cross-rank halo exchange.

This test exercises (1) on a synthetic Qwen3.5 MoE model configured with
``layer_types=["full_attention"] * num_hidden_layers`` (no GatedDeltaNet
layers). It also verifies that (2) is precisely rejected by validation when
hybrid configs are loaded — a separate negative test in the same file.

It further pins the CP router-aux contract (no extra ``cp_size`` factor on the
aux term — the FSDP average over CP ranks makes it the mean of per-chunk means):
the aux component, isolated by differencing a real-coef forward against a
zero-coef forward on the same batch, must match between CP and the reference.

Run (2 GPUs):

    torchrun --nproc_per_node=2 \\
        tests/gpu/parallelism/cp/test_qwen3_5_cp_correctness.py
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.distributed as dist
from transformers.models.qwen3_5_moe import (
    Qwen3_5MoeForCausalLM,
    Qwen3_5MoeTextConfig,
)
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeGatedDeltaNet,
)

from src.distributed.context_parallel.layers.qwen3_5 import Qwen3_5MoeUlyssesAttention
from src.distributed.context_parallel.validation import (
    UlyssesConfigError,
    validate_model_for_ulysses,
)
from src.distributed.context_parallel.wrapper import UlyssesCPModelWrapper
from src.distributed.parallelism_config import ParallelismConfig
from tests.common.distributed import shared_scratch_dir
from tests.common.harness import gpu_test_main
from tests.common.utils import cleanup_memory, log, log_all

# Small enough to fit comfortably on 2x B300 with plenty of headroom, but still
# exercising the head split (num_q_heads=8 → 4 per CP rank, num_kv_heads=2 → 1
# per CP rank). num_hidden_layers is intentionally small (2) — we only need to
# verify the wrapper math, not stress the model.
TINY_TEXT_CONFIG = {
    "vocab_size": 512,
    "hidden_size": 256,
    "intermediate_size": 512,
    "moe_intermediate_size": 64,
    "shared_expert_intermediate_size": 128,
    "num_experts": 8,
    "num_experts_per_tok": 2,
    "num_hidden_layers": 2,
    "num_attention_heads": 8,
    "num_key_value_heads": 2,
    "head_dim": 32,
    "hidden_act": "silu",
    "max_position_embeddings": 512,
    "norm_topk_prob": True,
    "rms_norm_eps": 1e-6,
    "tie_word_embeddings": False,
    "attention_bias": False,
    "rope_parameters": {"rope_type": "default", "rope_theta": 10000.0},
    # Loud aux component for the phase-3 check (the 1e-3 default would drown in bf16 noise).
    "router_aux_loss_coef": 0.1,
}


# bf16 ulp at loss-scale ~6 is ~5e-2; one all-to-all per layer plus the
# sigmoid-gate multiply add ordering noise, and the cross-entropy
# normalization differs slightly between the reference (single-mean over all
# active tokens) and the CP wrapper (per-rank globally-normalized sum). The
# tolerance must absorb that noise yet still catch gate corruption, which is
# order-of-magnitude: a Qwen3-style split dropping the gate lands ~10× off.
LOSS_TOLERANCE = 8e-2

# Two forwards per model (real coef vs coef=0) cancel the CE term and the CP-vs-reference
# normalization gap bitwise-modulo kernel jitter, leaving coef·aux ≈ 0.1·2. The residual is bf16
# forward noise plus the chunked-vs-full router-statistic covariance (≲1e-2 together), so a quarter
# of LOSS_TOLERANCE holds — while a dropped or doubled ÷cp_size scales the CP component ×2 / ×0.5,
# a miss of ≥ AUX_COMPONENT_MIN / 2 = 2.5× this bound (~5× at the expected component of 0.2).
AUX_COMPONENT_TOLERANCE = LOSS_TOLERANCE / 4
# Anti-vacuity floor: a ~0 component would let the match above pass with the aux term dropped
# entirely. Near-uniform random-init routing puts aux ≈ top_k = 2, so coef·aux ≈ 0.2 clears 2×.
AUX_COMPONENT_MIN = 5 * AUX_COMPONENT_TOLERANCE


def build_synthetic_all_full_attention_ckpt(out_dir: Path, seed: int = 0) -> Path:
    """Build a tiny Qwen3.5 MoE checkpoint with no GatedDeltaNet layers.

    ``Qwen3_5MoeForCausalLM`` accepts ``Qwen3_5MoeTextConfig`` directly (it's
    the text-only causal-LM head), so we skip the VLM wrapping config and pass
    the text config straight to the constructor.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    n = TINY_TEXT_CONFIG["num_hidden_layers"]
    config = Qwen3_5MoeTextConfig(
        **TINY_TEXT_CONFIG,
        layer_types=["full_attention"] * n,  # disable linear-attention
    )
    model = Qwen3_5MoeForCausalLM(config).to(torch.bfloat16)
    # Sanity: no GatedDeltaNet modules should be instantiated.
    for module in model.modules():
        assert not isinstance(module, Qwen3_5MoeGatedDeltaNet), (
            "Synthetic checkpoint must not contain Qwen3_5MoeGatedDeltaNet (layer_types should be all full_attention)"
        )
    model.save_pretrained(out_dir, safe_serialization=True)
    return out_dir


def make_inputs(vocab_size: int, batch: int, seq: int, device: str):
    torch.manual_seed(123)
    ids = torch.randint(0, vocab_size, (batch, seq), device=device)
    labels = ids.clone()
    # Mask every fourth token so every CP rank receives some active labels.
    labels[:, ::4] = -100
    return ids, labels


def reference_forward(checkpoint_dir: str, ids: torch.Tensor, labels: torch.Tensor, device: str) -> float:
    """Single-GPU reference: load with no parallelism, run forward."""
    log("  Loading reference (no parallelism) on rank 0...")
    model = Qwen3_5MoeForCausalLM.from_pretrained(
        checkpoint_dir,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    ).to(device)
    model.eval()
    with torch.no_grad():
        out = model(input_ids=ids, labels=labels, use_cache=False)
    loss = out.loss.item()
    log(f"  Reference loss: {loss:.6f}")
    del model
    cleanup_memory()
    return loss


def cp_forward(checkpoint_dir: str, ids: torch.Tensor, labels: torch.Tensor, rank: int, local_rank: int) -> float:
    """CP=2 forward: load model directly, patch via UlyssesCPModelWrapper.

    Bypasses ``load_distributed_model`` (which tries to load a tokenizer the
    synthetic checkpoint doesn't ship). The full pipeline still goes through
    :func:`validate_model_for_ulysses` and :func:`patch_attention_for_ulysses`
    via :class:`UlyssesCPModelWrapper`.
    """
    pc = ParallelismConfig(cp_size=2)
    log(f"  Loading CP model — {pc.mode_string}")

    device = f"cuda:{local_rank}"
    base_model = Qwen3_5MoeForCausalLM.from_pretrained(
        checkpoint_dir,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    ).to(device)

    cp_config = pc.create_cp_config()
    model = UlyssesCPModelWrapper(base_model, cp_config)

    # Confirm the wrapper landed on every full-attention layer.
    # ``Qwen3_5MoeAttention`` instances still appear as submodules because
    # ``UlyssesAttentionBase`` stores the original as ``original_attention`` —
    # we check that each decoder layer's ``self_attn`` is the wrapper.
    wrappers = [m for m in model.modules() if isinstance(m, Qwen3_5MoeUlyssesAttention)]
    log(f"  Qwen3_5MoeUlyssesAttention layers: {len(wrappers)}")
    assert len(wrappers) == TINY_TEXT_CONFIG["num_hidden_layers"], (
        f"Expected {TINY_TEXT_CONFIG['num_hidden_layers']} CP-wrapped layers, got {len(wrappers)}"
    )
    # Each decoder layer's self_attn attribute should be the wrapper.
    for layer in base_model.model.layers:
        assert isinstance(layer.self_attn, Qwen3_5MoeUlyssesAttention), (
            f"Layer {layer.layer_type} self_attn is {type(layer.self_attn).__name__}, "
            f"expected Qwen3_5MoeUlyssesAttention"
        )

    model.eval()
    with torch.no_grad():
        out = model(input_ids=ids, labels=labels, use_cache=False)
    loss = out.loss.item()
    log_all(f"  CP loss (rank {rank}): {loss:.6f}")
    del model, base_model
    cleanup_memory()
    return loss


def reference_aux_component(checkpoint_dir: str, ids: torch.Tensor, labels: torch.Tensor, device: str) -> float:
    """Reference aux component: loss with the checkpoint coef minus loss with the coef zeroed."""
    model = Qwen3_5MoeForCausalLM.from_pretrained(
        checkpoint_dir,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    ).to(device)
    model.eval()
    with torch.no_grad():
        with_aux = model(input_ids=ids, labels=labels, use_cache=False, output_router_logits=True).loss.item()
        # The forward reads the init-latched attribute; zero the config too so no read path survives.
        model.router_aux_loss_coef = 0.0
        model.config.router_aux_loss_coef = 0.0
        without_aux = model(input_ids=ids, labels=labels, use_cache=False, output_router_logits=True).loss.item()
    component = with_aux - without_aux
    log(f"  Reference aux component: {component:.6f}")
    del model
    cleanup_memory()
    return component


def cp_aux_component(
    checkpoint_dir: str, ids: torch.Tensor, labels: torch.Tensor, rank: int, local_rank: int
) -> float:
    """Per-rank CP aux component by the same subtraction; the wrapper reads the coef per forward."""
    pc = ParallelismConfig(cp_size=2)
    device = f"cuda:{local_rank}"
    base_model = Qwen3_5MoeForCausalLM.from_pretrained(
        checkpoint_dir,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    ).to(device)
    model = UlyssesCPModelWrapper(base_model, pc.create_cp_config())
    model.eval()
    with torch.no_grad():
        with_aux = model(input_ids=ids, labels=labels, use_cache=False, output_router_logits=True).loss.item()
        base_model.config.router_aux_loss_coef = 0.0
        without_aux = model(input_ids=ids, labels=labels, use_cache=False, output_router_logits=True).loss.item()
    component = with_aux - without_aux
    log_all(f"  CP aux component (rank {rank}): {component:.6f}")
    del model, base_model
    cleanup_memory()
    return component


def negative_test_linear_attention_rejection(local_rank: int) -> bool:
    """Verify validation rejects hybrid Qwen3.5 configs with linear_attention."""
    log("\n[4/4] Negative test: hybrid layer_types rejection")
    # Default __post_init__ generates a hybrid pattern (linear + full).
    config = Qwen3_5MoeTextConfig(**TINY_TEXT_CONFIG)
    model = Qwen3_5MoeForCausalLM(config).to(torch.bfloat16).to(f"cuda:{local_rank}")
    # Mark as flash for validation; we never run forward, just validate config + class scan.
    model.config._attn_implementation = "flash_attention_2"

    try:
        validate_model_for_ulysses(model, cp_size=2)
    except UlyssesConfigError as e:
        msg = str(e)
        log(f"  ✓ Got expected UlyssesConfigError: {msg[:200]}...")
        cleanup_memory()
        ok = ("linear_attention" in msg) or ("GatedDeltaNet" in msg)
        if not ok:
            log("  FAIL: error message did not mention linear_attention / GatedDeltaNet")
        return ok
    except Exception as e:
        log(f"  FAIL: expected UlyssesConfigError, got {type(e).__name__}: {e}")
        cleanup_memory()
        return False
    log("  FAIL: validation did not raise on hybrid layer_types")
    cleanup_memory()
    return False


def run(ctx) -> dict:
    device = f"cuda:{ctx.local_rank}"

    log("=" * 70)
    log("  Qwen3.5 CP CORRECTNESS — full-attention + double-width q_proj gate")
    log(f"  GPU: {torch.cuda.get_device_name(ctx.local_rank)}")
    log("=" * 70)

    ckpt_dir = shared_scratch_dir("qwen3_5_tiny_full")

    if ctx.rank == 0:
        log(f"\nBuilding synthetic Qwen3.5 MoE checkpoint at {ckpt_dir}")
        build_synthetic_all_full_attention_ckpt(Path(ckpt_dir))
    ctx.barrier()

    ids, labels = make_inputs(
        TINY_TEXT_CONFIG["vocab_size"],
        batch=2,
        seq=64,
        device=device,
    )
    dist.broadcast(ids, src=0)
    dist.broadcast(labels, src=0)

    # --- Phase 1: reference forward (rank 0 only) ---
    ref_loss_t = torch.zeros(1, dtype=torch.float64, device=device)
    if ctx.rank == 0:
        log("\n[1/4] Reference forward (no parallelism)")
        ref_loss_t[0] = reference_forward(ckpt_dir, ids, labels, device)
    ctx.barrier()
    dist.broadcast(ref_loss_t, src=0)
    ref_loss = ref_loss_t.item()

    # --- Phase 2: CP=2 forward ---
    log("\n[2/4] CP forward (cp_size=2) — Qwen3_5MoeUlyssesAttention")
    cp_loss = cp_forward(ckpt_dir, ids, labels, ctx.rank, ctx.local_rank)

    # Each CP rank computes loss on its sequence chunk → per-rank values differ.
    # Aggregate by averaging the per-token losses across CP ranks (since each
    # rank has the same number of label tokens after the every-fourth mask
    # alignment matches the chunk boundary).
    all_losses = [torch.zeros(1, device=device, dtype=torch.float64) for _ in range(ctx.world_size)]
    dist.all_gather(all_losses, torch.tensor([cp_loss], device=device, dtype=torch.float64))
    per_rank = [l.item() for l in all_losses]
    # The CP trainer aggregates per-token loss across CP ranks; for raw
    # comparison without the trainer's aggregation, take the mean — the
    # interleaved mask leaves each rank with the same active-token count.
    cp_mean = sum(per_rank) / len(per_rank)
    ref_vs_cp = abs(cp_mean - ref_loss)

    if ctx.rank == 0:
        log("\n" + "=" * 70)
        log(f"  Reference loss:         {ref_loss:.6f}")
        log(f"  CP loss (per rank):     {[f'{l:.6f}' for l in per_rank]}")
        log(f"  CP loss (mean):         {cp_mean:.6f}")
        log(f"  |CP mean - reference|:  {ref_vs_cp:.3e}")
        log(f"  Tolerance:              {LOSS_TOLERANCE:.3e}")
        log("=" * 70)

    checks = {"cp_matches_reference": ref_vs_cp <= LOSS_TOLERANCE}
    if not checks["cp_matches_reference"] and ctx.rank == 0:
        log(
            f"  FAIL: CP loss disagrees with reference "
            f"(|{cp_mean:.6f} - {ref_loss:.6f}| > {LOSS_TOLERANCE:.3e}). "
            f"This may indicate the double-width q_proj gate was corrupted."
        )

    # --- Phase 3: aux component — the router aux term takes no extra cp_size factor ---
    log("\n[3/4] Aux component: CP mean of chunk means vs reference (coef c vs 0)")
    ref_aux_t = torch.zeros(1, dtype=torch.float64, device=device)
    if ctx.rank == 0:
        ref_aux_t[0] = reference_aux_component(ckpt_dir, ids, labels, device)
    ctx.barrier()
    dist.broadcast(ref_aux_t, src=0)
    ref_aux = ref_aux_t.item()

    # Equal-length unpadded rows only: each chunk then holds the same token count, so the mean of
    # per-chunk aux means is comparable to the full-batch mean without any count-weighting bias.
    local_aux = cp_aux_component(ckpt_dir, ids, labels, ctx.rank, ctx.local_rank)
    aux_parts = [torch.zeros(1, device=device, dtype=torch.float64) for _ in range(ctx.world_size)]
    dist.all_gather(aux_parts, torch.tensor([local_aux], device=device, dtype=torch.float64))
    cp_aux = sum(t.item() for t in aux_parts) / ctx.world_size
    aux_delta = abs(cp_aux - ref_aux)

    # The eval-path wrapper all-reduces the aux term to the CP-group mean, so the per-rank
    # components must agree — a chunk-local aux (rank-varying) has the same MEAN and would pass the
    # average comparison below while still biasing eval_loss through the one-representative gather.
    aux_spread = max(abs(t.item() - aux_parts[0].item()) for t in aux_parts)
    checks["cp_aux_component_rank_uniform"] = aux_spread <= 1e-6

    if ctx.rank == 0:
        log(
            f"  reference={ref_aux:.6f}  cp_mean={cp_aux:.6f}  |Δ|={aux_delta:.3e}  "
            f"tol={AUX_COMPONENT_TOLERANCE:.1e}  floor={AUX_COMPONENT_MIN:.1e}"
        )
        if aux_delta > AUX_COMPONENT_TOLERANCE:
            log(
                f"  FAIL: CP aux component off by {cp_aux / max(ref_aux, 1e-12):.2f}x — a dropped or "
                f"doubled cp_size factor on the aux term lands at 2x / 0.5x."
            )

    checks["aux_component_nonvacuous"] = ref_aux > AUX_COMPONENT_MIN
    checks["cp_aux_component_matches_reference"] = aux_delta <= AUX_COMPONENT_TOLERANCE

    # --- Phase 4: negative test — hybrid config must be rejected ---
    if ctx.rank == 0:
        checks["hybrid_layer_types_rejected"] = negative_test_linear_attention_rejection(ctx.local_rank)

    return {"checks": ctx.broadcast_checks(checks)}


main = gpu_test_main(exact_world_size=2, prefix="qwen3_5_cp_correctness")(run)

if __name__ == "__main__":
    main()
