#!/usr/bin/env python
"""EP+CP gathered save then reload, through the CP wrapper: every expert tensor full-shape under a canonical key, loss
reproduced.

A save walking the wrapper misclassifies experts as non-EP, because ``named_modules()`` is wrapper-prefixed and
``named_parameters()`` inner, and the checkpoint silently reloads with random-init experts.
"""

import json
import shutil
from pathlib import Path

import torch
import torch.distributed as dist
from safetensors import safe_open
from transformers import AutoConfig, AutoTokenizer

from halo.checkpoint.parallel.ep import save_ep_gathered_checkpoint
from halo.checkpoint.parallel.save import save_tokenizer
from halo.distributed.context_parallel.ulysses_model import UlyssesModel
from halo.distributed.expert_parallel.moe_layer.layer import EPMoELayerBase
from halo.distributed.parallelism_config import ParallelismConfig
from halo.env import env_flag, env_str
from halo.model_loading.parallel.loader import load_distributed_model
from tests.common.checkpoint_io import save_context
from tests.common.distributed import shared_scratch_dir
from tests.common.harness import gpu_test_main, skip_unless_local_checkpoint
from tests.common.memory import cleanup_memory
from tests.common.models import GPT_OSS_20B_PATCHED
from tests.common.rank_log import log
from tests.common.tolerances import TOL

MODEL = env_str("HALO_TEST_EP_CP_RT_MODEL", GPT_OSS_20B_PATCHED)
ATTN = env_str("HALO_TEST_EP_CP_RT_ATTN", "flash_attention_2")
EP_SIZE = 2
CP_SIZE = 2
SEQ_LEN = 128  # Must be divisible by cp_size.
# Same weights, input and partition on both sides, so only bf16 serialization noise; dropped experts shift it by >>1.
LOSS_TOL = TOL.resume_loss_abs


def _fixed_batch(tokenizer, device):
    text = (
        "User: What is 17 plus 25?\nAssistant: The answer is 42. "
        "Expert routing must survive an EP+CP checkpoint save and reload intact. " * 4
    )
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=SEQ_LEN, padding="max_length")
    ids = enc["input_ids"].to(device)
    dist.broadcast(ids, src=0)
    return ids, ids.clone()


def _per_rank_cp_loss(model, ids, labels) -> float:
    """COLLECTIVE. Under no_grad the wrapper returns the group mean on every rank."""
    model.eval()
    with torch.no_grad():
        out = model(input_ids=ids, labels=labels, use_cache=False)
    return out["loss"].item()


def _checkpoint_keys_and_shapes(save_dir: str) -> dict[str, list[int]]:
    index_path = Path(save_dir) / "model.safetensors.index.json"
    if index_path.exists():
        weight_map = json.loads(index_path.read_text())["weight_map"]
        files = {Path(save_dir) / fname for fname in weight_map.values()}
    else:
        files = {Path(save_dir) / "model.safetensors"}
    shapes: dict[str, list[int]] = {}
    for file in files:
        with safe_open(str(file), framework="pt") as f:
            for key in list(f.keys()):
                shapes[key] = list(f.get_slice(key).get_shape())
    return shapes


def _assert_checkpoint_layout(save_dir: str, num_experts: int) -> list[str]:
    problems = []
    shapes = _checkpoint_keys_and_shapes(save_dir)
    if not shapes:
        return [f"no tensors found in {save_dir}"]

    wrapper_prefixed = [k for k in shapes if k.startswith("model.model.")]
    if wrapper_prefixed:
        problems.append(f"{len(wrapper_prefixed)} wrapper-prefixed keys, e.g. {wrapper_prefixed[0]}")

    cp_keys = [k for k in shapes if ".original_attention." in k]
    if cp_keys:
        problems.append(f"{len(cp_keys)} unremapped CP keys, e.g. {cp_keys[0]}")

    expert_keys = [k for k in shapes if ".experts." in k and ("proj" in k or "weight" in k)]
    if not expert_keys:
        problems.append("no expert weight keys in checkpoint")
    partial = [k for k in expert_keys if shapes[k][0] != num_experts]
    if partial:
        problems.append(
            f"{len(partial)} expert tensors with partial expert dim (expected {num_experts}), "
            f"e.g. {partial[0]} {shapes[partial[0]]}"
        )
    return problems


@gpu_test_main(exact_world_size=EP_SIZE, prefix="ep_cp_roundtrip")
def run(ctx):
    log(f"\n{'=' * 70}\nEP+CP SAVE/RELOAD ROUNDTRIP: {MODEL} (EP={EP_SIZE}, CP={CP_SIZE}, attn={ATTN})\n{'=' * 70}")

    save_dir = shared_scratch_dir(f"ep_cp_rt_ckpt_{Path(MODEL.rstrip('/')).name}")
    if ctx.rank == 0 and not env_flag("HALO_TEST_EP_CP_RT_KEEP"):
        ctx.on_teardown(lambda: shutil.rmtree(save_dir, ignore_errors=True))
    pc = ParallelismConfig(ep_size=EP_SIZE, cp_size=CP_SIZE)
    num_experts = getattr(AutoConfig.from_pretrained(MODEL, trust_remote_code=True), "num_local_experts", None)
    checks: dict[str, bool] = {}

    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    log("--- Loading model (EP+CP) ---")
    model, _ = load_distributed_model(
        model_name_or_path=MODEL,
        parallelism_config=pc,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation=ATTN,
        use_liger_kernel=False,
    )
    checks["cp_wrapped"] = isinstance(model, UlyssesModel)
    checks["ep_patched"] = any(isinstance(m, EPMoELayerBase) for m in model.modules())

    ids, labels = _fixed_batch(tokenizer, ctx.device)
    ref_loss = _per_rank_cp_loss(model, ids, labels)
    log(f"[rank {ctx.rank}] reference CP loss (pre-save): {ref_loss:.6f}")
    checks["ref_loss_finite"] = bool(torch.isfinite(torch.tensor(ref_loss)))

    ctx.barrier()
    if ctx.rank == 0:
        shutil.rmtree(save_dir, ignore_errors=True)
    ctx.barrier()
    log(f"--- Saving (gathered, through the CP wrapper) → {save_dir} ---")
    # The ladder's call: the CP wrapper under a CP config, which remaps the CP attention keys.
    save_ctx = save_context(model, pc, tokenizer=tokenizer)
    save_tokenizer(save_ctx, save_dir)
    save_ep_gathered_checkpoint(save_ctx, save_dir)
    ctx.barrier()

    # Rank 0 alone reads the written checkpoint; the other ranks leave the key out.
    if ctx.rank == 0:
        problems = _assert_checkpoint_layout(save_dir, num_experts)
        checks["checkpoint_layout_ok"] = not problems
        for p in problems:
            log(f"  LAYOUT PROBLEM: {p}")

    del model
    cleanup_memory()

    log("--- Reloading from saved checkpoint (EP+CP) ---")
    reloaded, _ = load_distributed_model(
        model_name_or_path=save_dir,
        parallelism_config=pc,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation=ATTN,
        use_liger_kernel=False,
    )
    rl_loss = _per_rank_cp_loss(reloaded, ids, labels)
    delta = abs(rl_loss - ref_loss)
    log(f"[rank {ctx.rank}] reloaded CP loss: {rl_loss:.6f}  |Δ| = {delta:.6e} (tol {LOSS_TOL})")
    checks["reload_loss_finite"] = bool(torch.isfinite(torch.tensor(rl_loss)))
    # Every rank checks its own copy; the harness fails the run on any.
    checks["reload_loss_matches"] = delta < LOSS_TOL
    return {"checks": checks}


if __name__ == "__main__":
    skip_unless_local_checkpoint(MODEL, "HALO_TEST_EP_CP_RT_MODEL")
    run()
