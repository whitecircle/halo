#!/usr/bin/env python
"""EP gathered save then reload: mismatched keys silently leave experts at random init, which a finite loss can't show.

Guards the Gemma4 layout, where a base gather doubles the prefix to ``...experts.experts.gate_up_proj``.

Cross-node EP=16 (2x8), the cross-node gather and non-shared-FS save, by hand:
    HALO_TEST_EP_RT_MODEL=$HALO_DATA_ROOT/models/gpt-oss-20b-BF16-patched \\
    HALO_TEST_EP_RT_ATTN=flash_attention_2 HALO_TEST_EP=16 \\
        torchrun --nnodes=2 --node_rank=$NODE_RANK --nproc_per_node=8 ... \\
        tests/gpu/parallelism/ep/test_ep_save_reload_roundtrip.py
"""

import json
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import AutoTokenizer

from halo.checkpoint.parallel.ep import save_ep_gathered_checkpoint
from halo.checkpoint.parallel.save import save_tokenizer
from halo.distributed.expert_parallel.moe_layer.layer import EPMoELayerBase
from halo.distributed.parallelism_config import ParallelismConfig
from halo.env import env_flag, env_int, env_str
from halo.model_loading.parallel.loader import load_distributed_model
from halo.runtime.collectives import barrier
from halo.runtime.ranks import is_global_main_process
from tests.common.checkpoint_io import fixed_batch_loss, fixed_text_batch, save_context
from tests.common.distributed import shared_scratch_dir
from tests.common.harness import gpu_test_main, skip_unless_local_checkpoint
from tests.common.memory import cleanup_memory
from tests.common.models import GEMMA4_26B_A4B_PATCHED
from tests.common.rank_log import log
from tests.common.tolerances import TOL

MODEL = env_str("HALO_TEST_EP_RT_MODEL", GEMMA4_26B_A4B_PATCHED)
ATTN = env_str("HALO_TEST_EP_RT_ATTN", "sdpa")
EP_SIZE = env_int("HALO_TEST_EP", None)
# ep_size=1 + HALO_TEST_EP_RT_EXPERT_TP=2 is pure ETP.
EXPERT_TP = env_int("HALO_TEST_EP_RT_EXPERT_TP", 1)
# "auto" stays node-local within one NVLink domain and resolves to "global" (cross-node, RDMA) past it, so ep_size=16
# on 2x8 validates the cross-node save without a config change.
EP_SCOPE = env_str("HALO_TEST_EP_RT_SCOPE", "auto")
# bf16 round-trip noise; a dropped expert axis shifts the loss by >>1.
LOSS_TOL = TOL.resume_loss_abs
FIXED_TEXT = (
    "User: What is 17 plus 25?\nAssistant: The answer is 42. "
    "Expert routing must survive a checkpoint save and reload intact."
)


def _assert_index_keys(save_dir: str) -> list[str]:
    """Returns the problems (empty == OK)."""
    problems = []
    index_path = Path(save_dir) / "model.safetensors.index.json"
    if not index_path.exists():
        # Single-shard saves write model.safetensors with no index.
        single = Path(save_dir) / "model.safetensors"
        if not single.exists():
            return [f"no model.safetensors[.index.json] in {save_dir}"]
        with safe_open(str(single), framework="pt") as f:
            keys = list(f.keys())
    else:
        keys = list(json.loads(index_path.read_text())["weight_map"].keys())

    doubled = [k for k in keys if ".experts.experts." in k]
    if doubled:
        problems.append(f"{len(doubled)} doubled-prefix keys, e.g. {doubled[0]}")

    expert_keys = [k for k in keys if "expert" in k and ("gate" in k or "down_proj" in k or "up_proj" in k)]
    if not expert_keys:
        problems.append("no expert weight keys found in saved index")
    return problems


@gpu_test_main(prefix="ep_roundtrip")
def run(ctx):
    ep_size = EP_SIZE if EP_SIZE is not None else ctx.world_size

    log(f"\n{'=' * 70}\nEP SAVE/RELOAD ROUNDTRIP: {MODEL} (EP={ep_size}, attn={ATTN})\n{'=' * 70}")
    if ctx.world_size % ep_size != 0:
        raise ValueError(f"world_size={ctx.world_size} not divisible by ep_size={ep_size}")

    # Unique per model and run: concurrent runs sharing TMPDIR must not collide.
    save_dir = shared_scratch_dir(f"ep_rt_ckpt_{Path(MODEL.rstrip('/')).name}")
    if is_global_main_process() and not env_flag("HALO_TEST_EP_RT_KEEP"):
        ctx.on_teardown(lambda: shutil.rmtree(save_dir, ignore_errors=True))
    pc = ParallelismConfig(ep_size=ep_size, expert_tp_size=EXPERT_TP, ep_scope=EP_SCOPE)
    checks: dict[str, bool] = {}

    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    log("--- Loading model (EP) ---")
    model, _ = load_distributed_model(
        model_name_or_path=MODEL,
        parallelism_config=pc,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation=ATTN,
        use_liger_kernel=False,
    )
    ep_layers = [m for m in model.modules() if isinstance(m, EPMoELayerBase)]
    checks["ep_patched"] = len(ep_layers) > 0
    log(f"EP layers: {len(ep_layers)}")

    ids, labels = fixed_text_batch(tokenizer, ctx.device, FIXED_TEXT)
    ref_loss = fixed_batch_loss(model, ids, labels)
    log(f"Reference loss (pre-save): {ref_loss:.6f}")
    checks["ref_loss_finite"] = bool(torch.isfinite(torch.tensor(ref_loss)))

    barrier()
    if ctx.rank == 0:
        shutil.rmtree(save_dir, ignore_errors=True)
    barrier()
    log(f"--- Saving (gathered) → {save_dir} ---")
    save_ctx = save_context(model, pc, tokenizer=tokenizer)
    save_tokenizer(save_ctx, save_dir)
    save_ep_gathered_checkpoint(save_ctx, save_dir)
    barrier()

    if is_global_main_process():
        problems = _assert_index_keys(save_dir)
        checks["index_keys_ok"] = not problems
        for p in problems:
            log(f"  KEY PROBLEM: {p}")

    del model
    cleanup_memory()

    log("--- Reloading from saved checkpoint ---")
    reloaded, _ = load_distributed_model(
        model_name_or_path=save_dir,
        parallelism_config=pc,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation=ATTN,
        use_liger_kernel=False,
    )
    rl_loss = fixed_batch_loss(reloaded, ids, labels)
    log(f"Reloaded loss (post-reload): {rl_loss:.6f}")
    delta = abs(rl_loss - ref_loss)
    log(f"|reloaded - reference| = {delta:.6e}  (tol {LOSS_TOL})")
    checks["reload_loss_finite"] = bool(torch.isfinite(torch.tensor(rl_loss)))
    checks["reload_loss_matches"] = delta < LOSS_TOL
    # With several EP groups a peer group can still be reading the checkpoint rank 0's teardown removes.
    barrier()
    return {"checks": checks}


if __name__ == "__main__":
    skip_unless_local_checkpoint(MODEL, "HALO_TEST_EP_RT_MODEL")
    run()
