#!/usr/bin/env python
"""CP correctness for Bailing MoE / Ling 2.0 attention.

Bailing wraps plain softmax GQA in transformers-v4-era remote code, so its Ulysses wrapper has to
reproduce four family spellings the other wrappers do not share: a fused ``query_key_value``
projection, an output projection named ``dense``, Q/K RMSNorm before a PARTIAL rotary
(``partial_rotary_factor=0.5``), and a softmax scale the module never stores. It also returns three
values to the decoder layer instead of two.

Four phases:

1. Validation gates — the model's sdpa label is accepted (the family cannot carry a flash label
   under transformers v5, and CP calls flash itself), while a Lightning-Attention-2 layer, whose
   file reuses the same full-attention class names, is rejected instead of silently half-patched.
2. Attention-module equivalence — the CP-wrapped module over sequence shards vs the unwrapped
   module over the full sequence. This is what catches a wrong RoPE split, a dropped Q/K norm, or a
   wrong scale; the loss comparison alone cannot separate those from noise on a random model.
3. Full-model CP loss vs the single-rank reference.
4. CP backward: grads finite, and their CP-rank average matching the reference grads.

Run (2 GPUs):

    torchrun --nproc_per_node=2 \\
        tests/gpu/parallelism/cp/test_bailing_cp_correctness.py
"""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM

from src.distributed.context_parallel.layers.bailing import BailingMoeV2UlyssesAttention
from src.distributed.context_parallel.validation import (
    UlyssesConfigError,
    validate_model_for_ulysses,
)
from src.distributed.context_parallel.wrapper import UlyssesCPModelWrapper
from src.distributed.parallelism_config import ParallelismConfig
from src.models.patches.remote_code_compat import apply_remote_code_compat_shims
from tests.common.harness import gpu_test_main
from tests.common.models import BAILING_MOE_LING_MINI
from tests.common.utils import cleanup_memory, log, log_all

# Shrunk from the hub config so every family field the wrapper reads (partial_rotary_factor,
# use_qk_norm, head_dim, sigmoid group-limited routing) keeps its real value. Heads: 8 Q → 4 per CP
# rank, 4 KV → 2 per rank. Layer 0 stays dense (first_k_dense_replace), layer 1 is MoE.
TINY_OVERRIDES = {
    "vocab_size": 512,
    "pad_token_id": 0,
    "eos_token_id": 1,
    "hidden_size": 256,
    "intermediate_size": 512,
    "moe_intermediate_size": 128,
    "moe_shared_expert_intermediate_size": 128,
    "num_hidden_layers": 2,
    "num_attention_heads": 8,
    "num_key_value_heads": 4,
    "head_dim": 64,
    "num_experts": 8,
    "num_experts_per_tok": 2,
    "n_group": 2,
    "topk_group": 1,
    "first_k_dense_replace": 1,
}

BATCH, SEQ = 2, 64

# Phase 2 compares one bf16 attention module against itself: CP runs flash over the gathered
# sequence, the reference runs the family's sdpa kernel over the full one, so the gap is bf16
# accumulation-order noise on activations whose RMS is ~1. A wrong wrapper is not subtle here —
# dropping the Q/K norm or rotating the full head dim instead of half moves the output by ~1e0.
ATTN_ATOL = 3e-2
ATTN_COSINE_MIN = 0.9995

# Loss: CP's per-rank sum normalization vs the reference mean, on top of the same bf16 noise.
LOSS_TOLERANCE = 5e-2

# Gradients (CP-rank average vs reference), matching tests/gpu/parallelism/cp/test_cp_train_correctness.py.
GRAD_COSINE_MIN = 0.99
GRAD_NORM_RTOL = 0.10


def build_tiny_model(device: str) -> nn.Module:
    """Tiny Bailing V2 model, weight-identical on every rank.

    ``attn_implementation="sdpa"``: transformers v5 refuses every flash label for this remote code
    (it declares only the v4 ``_supports_flash_attn_2``), which is exactly why the wrapper waives the
    label gate — CP calls flash through its own probe either way.
    """
    config = AutoConfig.from_pretrained(BAILING_MOE_LING_MINI, trust_remote_code=True)
    for key, value in TINY_OVERRIDES.items():
        setattr(config, key, value)

    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_config(config, trust_remote_code=True, attn_implementation="sdpa")
    model = model.to(device=device, dtype=torch.bfloat16)

    # Same seed gives the same init, but nothing in the test may depend on that: a per-rank
    # difference would show up as a wrapper "error" that is really a weight mismatch.
    for tensor in list(model.parameters()) + list(model.buffers()):
        if tensor.is_floating_point():
            dist.broadcast(tensor.data, src=0)
    return model


def make_inputs(vocab_size: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Token ids + labels with a quarter of the positions masked (every rank keeps active labels)."""
    torch.manual_seed(123)
    ids = torch.randint(0, vocab_size, (BATCH, SEQ), device=device)
    labels = ids.clone()
    labels[:, ::4] = -100
    return ids, labels


def causal_lm_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Shifted causal-LM cross entropy in fp32 — the objective the CP wrapper reproduces per chunk."""
    shift_logits = logits[:, :-1].float().reshape(-1, logits.size(-1))
    shift_labels = labels[:, 1:].reshape(-1)
    return F.cross_entropy(shift_logits, shift_labels, ignore_index=-100)


def cosine_similarity_flat(a: torch.Tensor, b: torch.Tensor) -> float:
    """Cosine between two flattened tensors in fp32 (1.0 for a degenerate pair)."""
    a, b = a.float().flatten(), b.float().flatten()
    denom = a.norm() * b.norm()
    if denom < 1e-12:
        return 1.0
    return (torch.dot(a, b) / denom).item()


def check_validation_gates(model: nn.Module) -> bool:
    """The sdpa label is accepted; a Lightning-Attention-2 layer is rejected by name."""
    ok = True

    try:
        validate_model_for_ulysses(model, cp_size=2)
        log(f"  sdpa label accepted (attn_implementation={model.config._attn_implementation!r})")
    except UlyssesConfigError as e:
        log(f"  FAIL: validation rejected the shipped Bailing label: {e}")
        ok = False

    # The linear sibling (Ring-mini-linear-2.0) names its full-attention classes exactly as Ling 2.0
    # does, so without the rejection CP would wrap those and leave the recurrent layers scanning each
    # rank's chunk alone. Injected by class name — the key the validator and patcher both match on.
    linear_cls = type("BailingMoeV2LinearAttention", (nn.Module,), {})
    model.model.layers[0].linear_attn_probe = linear_cls()
    try:
        validate_model_for_ulysses(model, cp_size=2)
        log("  FAIL: a BailingMoeV2LinearAttention layer was NOT rejected")
        ok = False
    except UlyssesConfigError as e:
        if "BailingMoeV2LinearAttention" not in str(e):
            log(f"  FAIL: rejection did not name the linear-attention class: {e}")
            ok = False
        else:
            log("  Lightning-Attention-2 layer rejected by name")
    finally:
        del model.model.layers[0].linear_attn_probe

    return ok


def check_attention_equivalence(model: nn.Module, cp_group, rank: int, cp_size: int, device: str) -> bool:
    """CP-wrapped attention over shards vs the unwrapped module over the full sequence."""
    attn = model.model.layers[0].attention
    config = model.config

    torch.manual_seed(7)
    hidden = torch.randn(BATCH, SEQ, config.hidden_size, device=device, dtype=torch.bfloat16)
    dist.broadcast(hidden, src=0)
    position_ids = torch.arange(SEQ, device=device).unsqueeze(0).expand(BATCH, -1)
    cos, sin = model.model.rotary_emb(hidden, position_ids)

    with torch.no_grad():
        reference = attn(hidden_states=hidden, position_embeddings=(cos, sin), use_cache=False)[0]

    wrapper = BailingMoeV2UlyssesAttention(attn, cp_group, cp_size)

    # The seams that the loss comparison cannot isolate on a random model.
    structural = {
        "scaling": wrapper.scaling == config.head_dim**-0.5,
        "use_qk_norm": wrapper.use_qk_norm == config.use_qk_norm,
        "output_projection": wrapper._output_projection is attn.dense,
        "rotary_dim": cos.shape[-1] == int(config.head_dim * config.partial_rotary_factor),
    }
    for name, passed in structural.items():
        if not passed:
            log(f"  FAIL: wrapper {name} does not match the config")

    chunk = SEQ // cp_size
    start = rank * chunk
    with torch.no_grad():
        local_out = wrapper(
            hidden[:, start : start + chunk],
            (cos[:, start : start + chunk], sin[:, start : start + chunk]),
            position_ids=position_ids[:, start : start + chunk],
            use_cache=False,
        )[0]

    gathered = [torch.empty_like(local_out) for _ in range(cp_size)]
    dist.all_gather(gathered, local_out.contiguous())
    cp_out = torch.cat(gathered, dim=1)

    max_abs = (cp_out.float() - reference.float()).abs().max().item()
    cosine = cosine_similarity_flat(cp_out, reference)
    log(f"  attention: max|Δ|={max_abs:.3e} (tol {ATTN_ATOL:.1e})  cosine={cosine:.6f} (min {ATTN_COSINE_MIN})")

    return all(structural.values()) and max_abs <= ATTN_ATOL and cosine >= ATTN_COSINE_MIN


def check_wrapped_layers(model: nn.Module) -> bool:
    """Every attention module is the Bailing wrapper (a partial patch would silently attend chunks)."""
    wrapped = [
        layer.attention for layer in model.model.layers if isinstance(layer.attention, BailingMoeV2UlyssesAttention)
    ]
    ok = len(wrapped) == model.config.num_hidden_layers
    log(f"  wrapped attention layers: {len(wrapped)}/{model.config.num_hidden_layers}")
    return ok


def run(ctx):
    apply_remote_code_compat_shims()
    device = str(ctx.device)

    log("=" * 70)
    log("  Bailing MoE / Ling 2.0 — Ulysses CP correctness")
    log(f"  GPU: {torch.cuda.get_device_name(ctx.local_rank)}")
    log("=" * 70)

    checks: dict[str, bool] = {}
    metrics: dict[str, float] = {}

    # Rank 0 first: the remote code (config.json + modeling_bailing_moe_v2.py) is fetched once.
    if ctx.rank == 0:
        AutoConfig.from_pretrained(BAILING_MOE_LING_MINI, trust_remote_code=True)
    ctx.barrier()

    model = build_tiny_model(device)
    ids, labels = make_inputs(model.config.vocab_size, device)

    log("[1/4] Validation gates")
    checks["validation_gates"] = check_validation_gates(model)

    log("[2/4] Attention-module equivalence")
    pc = ParallelismConfig(cp_size=2)
    cp_config = pc.create_cp_config()
    cp_group = cp_config.process_group
    checks["attention_equivalence"] = check_attention_equivalence(model, cp_group, ctx.rank, cp_size=2, device=device)

    log("[3/4] Full-model loss")
    model.train()
    reference_loss = causal_lm_loss(model(input_ids=ids, use_cache=False).logits, labels)
    reference_loss.backward()
    reference_grads = {name: p.grad.detach().clone() for name, p in model.named_parameters() if p.grad is not None}
    reference_value = reference_loss.item()
    model.zero_grad(set_to_none=True)

    cp_model = UlyssesCPModelWrapper(model, cp_config)
    checks["all_attention_layers_wrapped"] = check_wrapped_layers(model)

    cp_loss = cp_model(input_ids=ids, labels=labels, use_cache=False).loss
    log_all(f"  cp_loss={cp_loss.item():.6f}")

    losses = [torch.zeros(1, dtype=torch.float64, device=device) for _ in range(ctx.world_size)]
    dist.all_gather(losses, torch.tensor([cp_loss.item()], dtype=torch.float64, device=device))
    cp_mean = sum(t.item() for t in losses) / ctx.world_size
    delta = abs(cp_mean - reference_value)
    log(f"  reference={reference_value:.6f}  cp_mean={cp_mean:.6f}  |Δ|={delta:.3e}  tol={LOSS_TOLERANCE:.1e}")
    metrics["reference_loss"] = reference_value
    metrics["cp_mean_loss"] = cp_mean
    metrics["loss_delta"] = delta
    checks["cp_loss_matches_reference"] = delta <= LOSS_TOLERANCE

    log("[4/4] CP backward")
    cp_loss.backward()
    # No FSDP here, so the CP-rank average that FSDP would do is done by hand.
    for param in model.parameters():
        if param.grad is not None:
            dist.all_reduce(param.grad, op=dist.ReduceOp.AVG)

    min_cosine, worst_name = 1.0, ""
    cp_sq, ref_sq = 0.0, 0.0
    all_finite = True
    for name, param in model.named_parameters():
        if param.grad is None or name not in reference_grads:
            continue
        all_finite &= bool(torch.isfinite(param.grad).all().item())
        cosine = cosine_similarity_flat(param.grad, reference_grads[name])
        if cosine < min_cosine:
            min_cosine, worst_name = cosine, name
        cp_sq += param.grad.float().pow(2).sum().item()
        ref_sq += reference_grads[name].float().pow(2).sum().item()

    norm_ratio = (cp_sq**0.5) / max(ref_sq**0.5, 1e-12)
    log(f"  grads: finite={all_finite}  min cosine={min_cosine:.6f} ({worst_name})  norm ratio={norm_ratio:.4f}")
    metrics["grad_min_cosine"] = min_cosine
    metrics["grad_norm_ratio"] = norm_ratio
    checks["cp_grads_match_reference"] = (
        all_finite and min_cosine >= GRAD_COSINE_MIN and abs(norm_ratio - 1.0) <= GRAD_NORM_RTOL
    )

    del cp_model, model
    cleanup_memory()

    return {"checks": checks, "metrics": metrics}


main = gpu_test_main(min_world_size=2, exact_world_size=2, prefix="bailing_cp_correctness")(run)

if __name__ == "__main__":
    main()
