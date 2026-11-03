#!/usr/bin/env python
"""One parallelism mode of a tiny Mistral4 or Cohere2 MoE on 8 GPUs: it engages, runs with every rank agreeing, and
under EP survives a gathered save and reload."""

import argparse
import math
import shutil
from collections import Counter
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path

import torch
import torch.distributed as dist
from liger_kernel.transformers import LigerRMSNorm
from torch.distributed.tensor import DTensor

from halo.checkpoint.parallel.ep import save_ep_gathered_checkpoint
from halo.checkpoint.parallel.save import save_tokenizer
from halo.distributed.parallelism_config import ParallelismConfig
from halo.model_loading.parallel.loader import load_distributed_model
from halo.models.moe.config_fields import require_expert_count
from tests.common.checkpoint_io import save_context
from tests.common.distributed import cleanup_dirs, shared_scratch_dir, world_spread
from tests.common.ep_reference import ep_layers, random_token_batch
from tests.common.harness import gpu_test_main
from tests.common.memory import cleanup_memory
from tests.common.models import COMMAND_A_PLUS, MISTRAL3_119B_MOE
from tests.common.rank_log import log
from tests.common.rosters import ep_layer_class_by_model_type
from tests.common.tiny_models import TINY_MOE_FAMILIES, copy_release_tokenizer, shared_tiny_family_checkpoint
from tests.common.tolerances import TOL

SEED = 0
# Random-init cross-entropy sits at ln(vocab) whatever the experts hold, so only logits tell a correct gather from a
# zeroed one.
RELOAD_LOGIT_TOL = 1e-2


@dataclass(frozen=True)
class MatrixFamily:
    """``overrides`` is the 8-way geometry that tp8, cp8, ep8 and etp4 divide; ``attn_implementation`` the backend
    outside CP, which needs flash."""

    release: str
    attn_implementation: str
    ulysses_attention: str
    overrides: dict = field(default_factory=dict)


FAMILIES = {
    # MLA + MoE + YARN + llama-4 scaling + a shared expert + group routing; the tiny config is already 8-way.
    "mistral4": MatrixFamily(MISTRAL3_119B_MOE, "flash_attention_2", "Mistral4UlyssesAttention"),
    # Interleaved sliding/NoPE attention, parallel residual, sigmoid top-k, the averaged shared expert.
    "cohere2_moe": MatrixFamily(
        COMMAND_A_PLUS,
        "eager",
        "Cohere2MoeUlyssesAttention",
        {
            "num_hidden_layers": 4,
            "num_attention_heads": 8,
            "num_key_value_heads": 8,
            "head_dim": 32,
            "num_experts": 16,
            "num_experts_per_tok": 4,
            "layer_types": ["sliding_attention", "full_attention"] * 2,
            "mlp_layer_types": ["sparse"] * 4,
        },
    ),
}


def _cohere2_moe_checks(model, layers: list, pc: ParallelismConfig) -> dict[str, bool]:
    """The "average" shared-expert combination halves the sum, and CP's attention applies RoPE on sliding layers
    alone."""
    checks = {"average_combination_scaled": all(layer._output_scale == 0.5 for layer in layers)}
    if pc.is_cp_mode:
        wrappers = [m for m in model.modules() if type(m).__name__ == FAMILIES["cohere2_moe"].ulysses_attention]
        checks["cp_rope_follows_interleave"] = [w._use_rope for w in wrappers] == [
            kind == "sliding_attention" for kind in model.config.layer_types
        ]
    return checks


FAMILY_CHECKS = {"cohere2_moe": _cohere2_moe_checks}


def _mode_checks(model, pc: ParallelismConfig, args, family: MatrixFamily) -> dict[str, bool]:
    """EP layers exist in every mode (grouped GEMM installs them at ``ep_size`` 1), so shard widths, Ulysses attention,
    TP DTensors and the Liger norm prove it."""
    layers = ep_layers(model)
    attention = Counter(type(m).__name__ for m in model.modules() if "Attention" in type(m).__name__)
    log(f"  {len(layers)} EP layers, attention classes {dict(attention)}")
    ep_cls = ep_layer_class_by_model_type()[args.family]
    checks = {
        "ep_layers_present": len(layers) == model.config.num_hidden_layers,
        "ep_layers_are_the_family_class": all(type(layer) is ep_cls for layer in layers),
    }
    if layers:
        first = layers[0]
        checks["ep_shard_width"] = first.experts_per_rank == require_expert_count(model.config) // args.ep
        checks["etp_shard_width"] = first.expert_tp_size == args.etp
    if pc.is_cp_mode:
        checks["cp_attention_patched"] = attention[family.ulysses_attention] > 0
    if pc.is_tp_mode:
        # Selective TP keeps the attention class and turns its projections into DTensors.
        checks["tp_engaged"] = any(isinstance(p.data, DTensor) for p in model.parameters())
    if args.liger:
        # Patched by subclassing the family's norm and binding Liger's forward, so the name stays.
        norm = type(model.model.norm)
        checks["liger_rmsnorm_patched"] = (
            getattr(norm, "_halo_liger_patched_role", None) == "rms_norm" and norm.forward is LigerRMSNorm.forward
        )
    if args.family in FAMILY_CHECKS:
        checks |= FAMILY_CHECKS[args.family](model, layers, pc)
    return checks


def _reload_checks(model, tokenizer, pc, reload_dir: str, batch, load) -> dict[str, bool]:
    """COLLECTIVE. The gathered EP save reloads through ``load`` to the same eval logits."""
    save_dir = Path(reload_dir)
    dist.barrier()
    if dist.get_rank() == 0:
        shutil.rmtree(save_dir, ignore_errors=True)
    dist.barrier()
    save_ctx = save_context(model, pc, tokenizer=tokenizer)
    save_tokenizer(save_ctx, str(save_dir))
    save_ep_gathered_checkpoint(save_ctx, str(save_dir))
    dist.barrier()
    files = sorted(p.name for p in save_dir.iterdir())
    model.eval()
    with torch.no_grad():
        expected = model(input_ids=batch[0], labels=batch[1], use_cache=False).logits.float().cpu()
    reloaded, _ = load(model_name_or_path=str(save_dir), use_liger_kernel=False)
    reloaded.eval()
    with torch.no_grad():
        out = reloaded(input_ids=batch[0], labels=batch[1], use_cache=False)
    logit_delta = (out.logits.float().cpu() - expected).abs().max().item()
    log(f"  saved {files}; reloaded loss {out.loss.item():.6f}, max |Δlogit| {logit_delta:.3e}")
    checks = {
        "ckpt_files_present": "config.json" in files
        and any(name.startswith("model") and name.endswith(".safetensors") for name in files),
        "reload_loss_finite": bool(torch.isfinite(out.loss)),
        "reload_logits_match": logit_delta < RELOAD_LOGIT_TOL,
    }
    del reloaded
    cleanup_memory()
    return checks


@gpu_test_main(min_world_size=2, prefix="moe_all_parallelism")
def run(ctx):
    p = argparse.ArgumentParser()
    p.add_argument("--family", choices=sorted(FAMILIES), required=True)
    p.add_argument("--mode", choices=["ep", "cp", "tp", "ep_cp", "ep_tp", "etp", "ep_etp"], required=True)
    p.add_argument("--ep", type=int, default=1)
    p.add_argument("--cp", type=int, default=1)
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--etp", type=int, default=1)
    p.add_argument("--liger", action="store_true", help="Load with the Liger kernels.")
    args = p.parse_args()
    family = FAMILIES[args.family]

    ckpt_dir = shared_tiny_family_checkpoint(
        ctx, TINY_MOE_FAMILIES[args.family], f"{args.family}_tiny", None, SEED, overrides=family.overrides
    )
    reload_dir = shared_scratch_dir(f"{args.family}_reload_{args.mode}")
    if ctx.rank == 0:
        copy_release_tokenizer(family.release, Path(ckpt_dir))
        ctx.on_teardown(lambda: cleanup_dirs(reload_dir))
    ctx.barrier()

    pc = ParallelismConfig(
        ep_size=args.ep, cp_size=args.cp, tp_size=args.tp, expert_tp_size=args.etp, max_concurrent_loading=0
    )
    log(f"=== {args.family} --mode {args.mode}: {pc.summary()}, data_parallel_size={pc.data_parallel_size}")
    load = partial(
        load_distributed_model,
        parallelism_config=pc,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2" if pc.is_cp_mode else family.attn_implementation,
    )
    model, tokenizer = load(model_name_or_path=ckpt_dir, use_liger_kernel=args.liger)
    checks = _mode_checks(model, pc, args, family)

    # CP's wrapper shards the full sequence itself.
    batch = random_token_batch(model.config.vocab_size, batch=2, seq=64, device=ctx.device)
    model.train()
    loss = model(input_ids=batch[0], labels=batch[1], use_cache=False).loss
    spread = world_spread(loss.item())
    log(f"  loss {loss.item():.6f}, cross-rank spread {spread:.3e}")
    checks["loss_finite"] = bool(torch.isfinite(loss))
    if pc.is_cp_mode:
        # Each CP rank scores its own chunk; the spread is inf when any rank's loss is not finite.
        checks["losses_finite_across_ranks"] = math.isfinite(spread)
    else:
        checks["losses_consistent_across_ranks"] = TOL.identical_batch_ranks_agree(spread)
    loss.backward()
    layers = ep_layers(model)
    if layers:
        # Zero means no router signal reached the gate.
        gate_grad = layers[0].gate.weight.grad
        checks["ep_router_grad_present"] = (
            gate_grad is not None and bool(torch.isfinite(gate_grad).all()) and gate_grad.norm().item() > 0.0
        )
    metrics = {"loss": loss.item(), "ep_layer_count": len(layers)}
    del loss, layers
    cleanup_memory()

    if pc.has_ep_group and not (pc.is_cp_mode or pc.is_tp_mode):
        checks |= _reload_checks(model, tokenizer, pc, reload_dir, batch, load)
    # A peer can still be reading reload_dir, which rank 0's teardown removes.
    ctx.barrier()
    return {"checks": checks, "metrics": metrics}


if __name__ == "__main__":
    run()
