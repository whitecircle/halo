#!/usr/bin/env python
"""EP+CP gathered-checkpoint save→reload round-trip correctness.

Under EP+CP the trainer holds a ``UlyssesCPModelWrapper`` whose ``named_modules()`` walk yields
wrapper-prefixed EP-layer paths while ``named_parameters()`` yields inner names. A save that walks
the wrapper misclassifies every expert param as non-EP: the rank-local expert shard is written
under the canonical key, gathered full experts land under doubled bogus keys, and the checkpoint
silently reloads with random-init experts. This test saves through the CP wrapper (the exact
object ``save_ep_checkpoint`` passes), asserts every expert tensor in the checkpoint has the
FULL ``[E, ...]`` shape under a canonical key, and asserts the reloaded per-rank CP loss matches
the pre-save loss.

Usage (2 GPUs — ep_size=2, cp_size=2, dp=1):
    torchrun --nproc_per_node=2 \
        tests/gpu/parallelism/combined/test_ep_cp_save_reload_roundtrip.py

Env:
    HALO_TEST_EP_CP_RT_MODEL  local checkpoint dir (default: $HALO_DATA_ROOT/models/gpt-oss-20b-BF16-patched)
    HALO_TEST_EP_CP_RT_ATTN   attn_implementation (default: flash_attention_2)
"""

import json
import os
import shutil
import sys
from pathlib import Path

import torch
import torch.distributed as dist
from safetensors import safe_open
from transformers import AutoConfig, AutoTokenizer

from src.distributed.context_parallel.wrapper import UlyssesCPModelWrapper
from src.distributed.expert_parallel.base_layer import EPMoELayerBase
from src.distributed.expert_parallel.saving import save_ep_model
from src.distributed.loading.model_loading import load_distributed_model
from src.distributed.parallelism_config import ParallelismConfig
from src.env import env_flag, env_str
from tests.common.distributed import shared_scratch_dir
from tests.common.harness import gpu_test_main
from tests.common.models import GPT_OSS_20B_PATCHED
from tests.common.tolerances import TOL
from tests.common.utils import cleanup_memory, log

MODEL = env_str("HALO_TEST_EP_CP_RT_MODEL", GPT_OSS_20B_PATCHED)
ATTN = env_str("HALO_TEST_EP_CP_RT_ATTN", "flash_attention_2")
EP_SIZE = 2
CP_SIZE = 2
SEQ_LEN = 128  # must be divisible by cp_size
# Same weights + same input + same EP+CP partition on both sides of the round-trip, so the loss
# must match to bf16 serialization noise. Dropped/reinitialized experts shift it by >>1.
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
    """The CP group's loss: under no_grad the wrapper returns the group mean on every rank. Its forward
    is collective, so every rank calls."""
    model.eval()
    with torch.no_grad():
        out = model(input_ids=ids, labels=labels, use_cache=False)
    return out["loss"].item()


def _checkpoint_keys_and_shapes(save_dir: str) -> dict[str, list[int]]:
    """key → shape for every tensor in the saved checkpoint (index or single-file)."""
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
    """Layout problems in the saved checkpoint (empty == OK)."""
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
    checks["cp_wrapped"] = isinstance(model, UlyssesCPModelWrapper)
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
    # The exact call save_ep_checkpoint makes: the CP WRAPPER with cp_key_remap=True.
    save_ep_model(model, save_dir, tokenizer=tokenizer, sharded=False, cp_key_remap=True)
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
    # A local-checkpoint test declines to run before the harness starts: the launcher reports a
    # ``SKIP:`` line with exit 0 and no result line as a skip.
    if not os.path.isdir(MODEL):
        log(f"SKIP: local model path missing: {MODEL} (set HALO_TEST_EP_CP_RT_MODEL to a present checkpoint)")
        sys.exit(0)
    run()
