#!/usr/bin/env python
"""End-to-end EP / CP / TP / EP+CP / EP+TP / ETP / EP+ETP test for Cohere2 MoE (Command A+).

Validates EP (Cohere2MoeSparseMoeBlock → DeepEP), CP (Cohere2MoeAttention → Ulysses, RoPE on
sliding layers only), and TP (selective DTensor attention) integrations for the text backbone of
``CohereLabs/command-a-plus-05-2026-bf16``. Each invocation runs **one** parallelism mode
determined by ``--mode``; the manifest's args_matrix chains them. Single-node EP+CP pins
``ep_size`` to the 8-GPU domain with ``cp_size`` dividing it (EP is orthogonal to DP).

Build path: instead of downloading the 200B+ VLM checkpoint, the test materializes a tiny
synthetic ``Cohere2MoeForCausalLM`` exercising the same code paths (interleaved sliding/NoPE
attention, parallel-residual blocks, sigmoid top-k-then-renorm routing, averaged shared expert,
tied embeddings, logit scaling). The synthetic checkpoint is written to ``--checkpoint-dir``
(rank 0) so ``load_distributed_model`` and the family's ``from_pretrained`` route (lazy loading is
declared off) are exercised end-to-end.

Run (8 GPUs):

    torchrun --nproc_per_node=8 \
        tests/gpu/parallelism/test_cohere2_moe_all_parallelism.py --mode ep --ep 8
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

import torch
import torch.distributed as dist
from huggingface_hub import snapshot_download
from torch.distributed.tensor import DTensor
from transformers import AutoTokenizer
from transformers.models.cohere2_moe import Cohere2MoeConfig, Cohere2MoeForCausalLM

from src.distributed.expert_parallel.saving import save_ep_model
from src.distributed.loading.model_loading import load_distributed_model
from src.distributed.parallelism_config import ParallelismConfig
from tests.common.distributed import cleanup_dirs, shared_scratch_dir, world_spread
from tests.common.ep_reference import ep_layers, random_token_batch
from tests.common.harness import gpu_test_main
from tests.common.models import COMMAND_A_PLUS
from tests.common.tiny_models import TOKENIZER_FILE_PREFIXES
from tests.common.tolerances import TOL
from tests.common.utils import cleanup_memory, gpu_mem_gb, log

# 8 Q / 8 KV heads so cp8 and tp8 shard heads evenly; hidden 256 keeps the DeepEP transport pad
# (multiple of 256) exact; 4 layers keep the sliding/full interleave present on every pp-free mode.
TINY_CONFIG_KWARGS = {
    "vocab_size": 512,
    "hidden_size": 256,
    "intermediate_size": 64,
    "num_hidden_layers": 4,
    "num_attention_heads": 8,
    "num_key_value_heads": 8,
    "head_dim": 32,
    "num_experts": 16,
    "num_experts_per_tok": 4,
    "num_shared_experts": 2,
    "shared_expert_combination_strategy": "average",
    "expert_selection_fn": "sigmoid",
    "norm_topk_prob": True,
    "layer_types": ["sliding_attention", "full_attention", "sliding_attention", "full_attention"],
    "mlp_layer_types": ["sparse", "sparse", "sparse", "sparse"],
    "sliding_window": 64,
    "max_position_embeddings": 4096,
    "logit_scale": 0.0625,
    "tie_word_embeddings": True,
}


def build_synthetic_checkpoint(out_dir: Path, seed: int = 0) -> Path:
    """Build a tiny synthetic Cohere2 MoE checkpoint at ``out_dir`` and return the path.

    Tokenizer files come from the real Command A+ snapshot so ``AutoTokenizer.from_pretrained``
    works without the 200B weights; the synthetic config bumps vocab_size to the real tokenizer's
    vocab so token ids stay in range.
    """
    tokenizer_dir = Path(
        snapshot_download(COMMAND_A_PLUS, allow_patterns=[f"{prefix}*" for prefix in TOKENIZER_FILE_PREFIXES])
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    for src in tokenizer_dir.iterdir():
        if src.is_file() and src.name.startswith(TOKENIZER_FILE_PREFIXES):
            shutil.copy2(src, out_dir / src.name)

    torch.manual_seed(seed)
    vocab = len(AutoTokenizer.from_pretrained(out_dir))
    config = Cohere2MoeConfig(**{**TINY_CONFIG_KWARGS, "vocab_size": vocab})
    model = Cohere2MoeForCausalLM(config).to(torch.bfloat16)
    model.save_pretrained(out_dir, safe_serialization=True)
    return out_dir


def find_attention_classes(model: torch.nn.Module) -> dict[str, int]:
    counts: dict[str, int] = {}
    for _, mod in model.named_modules():
        cls = type(mod).__name__
        if "Attention" in cls or "Ulysses" in cls:
            counts[cls] = counts.get(cls, 0) + 1
    return counts


def run_mode(
    *,
    mode: str,
    ep: int,
    cp: int,
    tp: int,
    expert_tp: int,
    checkpoint_dir: str,
    reload_dir: str,
    rank: int,
    world_size: int,
    local_rank: int,
) -> tuple[dict[str, bool], dict]:
    """Load Cohere2 MoE with the requested parallelism and run forward + backward."""
    pc = ParallelismConfig(
        ep_size=ep,
        cp_size=cp,
        tp_size=tp,
        expert_tp_size=expert_tp,
        ep_fp32_router=False,
        ep_fp32_experts=False,
        max_concurrent_loading=0,  # All ranks in parallel — tiny model fits easily.
    )

    log(f"\n=== mode={mode}  ep={ep}  cp={cp}  tp={tp}  etp={expert_tp} ===")
    log(f"  GPU memory before load: {gpu_mem_gb():.2f}GB")
    log(f"  world_size={world_size}  data_parallel_size={pc.data_parallel_size}")

    # CP requires a real flash implementation; the other modes run eager for determinism.
    attn_implementation = "flash_attention_2" if pc.is_cp_mode else "eager"
    model, tokenizer = load_distributed_model(
        model_name_or_path=checkpoint_dir,
        parallelism_config=pc,
        dtype=torch.bfloat16,
        attn_implementation=attn_implementation,
        use_liger_kernel=False,  # no cohere2_moe Liger applier
    )
    log(f"  GPU memory after load: {gpu_mem_gb():.2f}GB")

    checks: dict[str, bool] = {}
    moe_layers = ep_layers(model)
    attn_classes = find_attention_classes(model)

    # ── Sanity ─────────────────────────────────────────────────────────────
    # ``use_grouped_gemm`` defaults on, so the EP wrappers exist in EVERY mode; the shard widths
    # are what prove the mode under test engaged.
    expected_layers = TINY_CONFIG_KWARGS["num_hidden_layers"]
    checks["ep_layers_present"] = len(moe_layers) == expected_layers
    log(f"  EP layers found: {len(moe_layers)} (expected {expected_layers})")
    if moe_layers:
        first = moe_layers[0]
        log(f"  experts_per_rank={first.experts_per_rank}, range=[{first.expert_start}, {first.expert_end})")
        checks["correct_ep_layer_class"] = type(first).__name__ == "EPCohere2MoELayer"
        checks["ep_shard_width"] = first.experts_per_rank == TINY_CONFIG_KWARGS["num_experts"] // ep
        checks["etp_shard_width"] = first.expert_tp_size == expert_tp
        checks["average_combination_scaled"] = all(layer._output_scale == 0.5 for layer in moe_layers)
        log(f"  expert_tp_size={first.expert_tp_size} (expected {expert_tp})")

    if pc.is_cp_mode:
        checks["cp_attention_patched"] = "Cohere2MoeUlyssesAttention" in attn_classes
        # RoPE only on sliding layers: the wrapper must mirror the interleave, not blanket-apply.
        cp_wrappers = [m for _, m in model.named_modules() if type(m).__name__ == "Cohere2MoeUlyssesAttention"]
        checks["cp_rope_follows_interleave"] = [w._use_rope for w in cp_wrappers] == [
            t == "sliding_attention" for t in TINY_CONFIG_KWARGS["layer_types"]
        ]
        log(f"  Attention classes after CP: {attn_classes}")
    if pc.is_tp_mode:
        checks["tp_engaged"] = any(isinstance(p.data, DTensor) for p in model.parameters())
        log(f"  Attention classes after TP: {attn_classes}; DTensor params present: {checks['tp_engaged']}")

    # ── Forward + backward ────────────────────────────────────────────────
    ids, labels = random_token_batch(model.config.vocab_size, batch=2, seq=64, device=f"cuda:{local_rank}")
    dist.broadcast(ids, src=0)
    dist.broadcast(labels, src=0)

    model.train()
    out = model(input_ids=ids, labels=labels, use_cache=False)

    loss = out.loss
    log(f"  Forward loss: {loss.item():.6f}")
    checks["loss_finite"] = torch.isfinite(loss).item()

    spread = world_spread(loss.item())
    if pc.is_cp_mode:
        # Each CP rank computes loss on its sequence chunk → values differ; require finiteness
        # (the spread is inf when any rank is non-finite).
        checks["losses_finite_across_ranks"] = math.isfinite(spread)
    else:
        checks["losses_consistent_across_ranks"] = spread < TOL.ep_identical_batch_rank_spread_abs
        log(f"  Cross-rank max loss diff: {spread:.6e}")

    loss.backward()
    cleanup_memory()

    if moe_layers:
        gate_grad = moe_layers[0].gate.weight.grad
        gate_grad_norm = float(gate_grad.norm()) if gate_grad is not None else float("nan")
        checks["ep_router_grad_present"] = (
            gate_grad is not None and torch.isfinite(gate_grad).all().item() and gate_grad_norm > 0.0
        )
        log(f"  Router (gate) grad norm: {gate_grad_norm:.6e}")

    # ── EP checkpoint roundtrip (pure EP only) ─────────────────────────────
    saved_loss = float("nan")
    if pc.is_ep_mode and not (pc.is_cp_mode or pc.is_tp_mode):
        save_dir = Path(reload_dir)
        dist.barrier()
        if rank == 0:
            shutil.rmtree(save_dir, ignore_errors=True)
        dist.barrier()

        log(f"  Saving EP model to {save_dir}")
        save_ep_model(model, str(save_dir), tokenizer=tokenizer, sharded=False)
        dist.barrier()

        if rank == 0:
            files = sorted(p.name for p in save_dir.iterdir())
            checks["ckpt_files_present"] = any(f.startswith("model") and f.endswith(".safetensors") for f in files)

        # Logits, not loss: random-init CE sits at ln(vocab) whatever the experts contain, so only
        # the logits distinguish a correct gather from a zeroed one. Collective — runs on all ranks.
        model.eval()
        with torch.no_grad():
            pre_save_out = model(input_ids=ids, labels=labels, use_cache=False)
            pre_save_loss = pre_save_out.loss.item()
            pre_save_logits = pre_save_out.logits.float().cpu()

        del model
        cleanup_memory()

        reloaded, _ = load_distributed_model(
            model_name_or_path=str(save_dir),
            parallelism_config=pc,
            dtype=torch.bfloat16,
            attn_implementation="eager",
            use_liger_kernel=False,
        )
        reloaded.eval()
        with torch.no_grad():
            out2 = reloaded(input_ids=ids, labels=labels, use_cache=False)
        saved_loss = out2.loss.item()
        log(f"  Reloaded forward loss: {saved_loss:.6f} (pre-save eval loss: {pre_save_loss:.6f})")
        checks["reload_loss_finite"] = torch.isfinite(out2.loss).item()
        logit_delta = (out2.logits.float().cpu() - pre_save_logits).abs().max().item()
        log(f"  Max |Δlogit| across the save/reload roundtrip: {logit_delta:.3e}")
        checks["reload_logits_match"] = logit_delta < 1e-2
        del reloaded
        cleanup_memory()

    metrics = {
        "loss": loss.item() if torch.is_tensor(loss) else loss,
        "reload_loss": saved_loss,
        "ep_layer_count": len(moe_layers),
        "attention_classes": attn_classes,
    }
    return checks, metrics


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--mode",
        required=True,
        choices=["ep", "cp", "tp", "ep_cp", "ep_tp", "etp", "ep_etp"],
        help="Which parallelism configuration to test.",
    )
    p.add_argument("--ep", type=int, default=1)
    p.add_argument("--cp", type=int, default=1)
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--etp", type=int, default=1, help="Expert TP size (only used when mode includes ETP).")
    p.add_argument("--checkpoint-dir", default=None, help="Directory holding the synthetic checkpoint.")
    return p.parse_args()


@gpu_test_main(min_world_size=2, prefix="cohere2_moe_all_parallelism")
def run(ctx):
    args = parse_args()
    ckpt_dir = args.checkpoint_dir or shared_scratch_dir("cohere2_moe_tiny")
    reload_dir = shared_scratch_dir(f"cohere2_moe_reload_{args.mode}")
    if ctx.rank == 0:
        ctx.on_teardown(lambda: cleanup_dirs(reload_dir))
        if args.checkpoint_dir is None:
            ctx.on_teardown(lambda: cleanup_dirs(ckpt_dir))
        log(f"Building synthetic Cohere2 MoE checkpoint at {ckpt_dir}")
        build_synthetic_checkpoint(Path(ckpt_dir))
    ctx.barrier()

    checks, metrics = run_mode(
        mode=args.mode,
        ep=args.ep,
        cp=args.cp,
        tp=args.tp,
        expert_tp=args.etp,
        checkpoint_dir=ckpt_dir,
        reload_dir=reload_dir,
        rank=ctx.rank,
        world_size=ctx.world_size,
        local_rank=ctx.local_rank,
    )
    log(f"Metrics ({args.mode}): {json.dumps(metrics, indent=2, default=str)}")
    # A peer can still be reading reload_dir, which rank 0's teardown removes.
    ctx.barrier()
    return {"checks": checks, "metrics": metrics}


if __name__ == "__main__":
    run()
