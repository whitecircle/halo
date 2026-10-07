#!/usr/bin/env python
"""EP gathered-checkpoint save→reload round-trip correctness.

The gathered EP save (``save_sharded_ep: false``, the production default) must
write a checkpoint whose keys match exactly what the model expects on reload —
otherwise expert weights are silently dropped and reload leaves them at random
init. A *finite* reload loss does not prove correctness (uninitialized experts
still give a finite loss); the decisive check is that the reloaded forward loss
matches the pre-save loss, plus a key-layout assertion on the saved index.

This guards the Gemma4 layout class, where the EP wrapper replaces the experts
*container* (``model.layers.{i}.experts``) rather than an encapsulating
``*MoeBlock``: a base ``experts.``-prefixed gather doubles the prefix to
``...experts.experts.gate_up_proj``, which never reloads.

Usage:
    HALO_TEST_EP_RT_MODEL=$HALO_DATA_ROOT/models/gemma-4-26B-A4B-it-patched HALO_TEST_EP_RT_ATTN=sdpa \
        torchrun --nproc_per_node=2 \
        tests/gpu/parallelism/ep/test_ep_save_reload_roundtrip.py

    # Cross-node EP=16 (2×8): each node's local rank 0 writes a complete gathered
    # checkpoint, so the round-trip exercises the cross-node gather + non-shared-FS save.
    HALO_TEST_EP_RT_MODEL=$HALO_DATA_ROOT/models/gpt-oss-20b-BF16-patched \
    HALO_TEST_EP_RT_ATTN=flash_attention_2 HALO_TEST_EP=16 \
        torchrun --nnodes=2 --node_rank=$NODE_RANK --nproc_per_node=8 ... \
        tests/gpu/parallelism/ep/test_ep_save_reload_roundtrip.py

Env:
    HALO_TEST_EP_RT_MODEL  local checkpoint dir (default: Gemma4-26B-A4B patched)
    HALO_TEST_EP_RT_ATTN   attn_implementation (default: sdpa; gpt-oss → flash_attention_2)
    HALO_TEST_EP EP size (default: world_size)
    HALO_TEST_EP_RT_SCOPE  ep_scope: "auto" (default), "node", or "global" (cross-node EP)
"""

import json
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import AutoTokenizer

from src.distributed.checkpoint.ep_save import save_ep_model
from src.distributed.expert_parallel.base_layer import EPMoELayerBase
from src.distributed.loading.model_loading import load_distributed_model
from src.distributed.parallelism_config import ParallelismConfig
from src.distributed.runtime import barrier, is_global_main_process
from src.env import env_flag, env_int, env_str
from tests.common.checkpoint_io import fixed_batch_loss, fixed_text_batch
from tests.common.distributed import shared_scratch_dir
from tests.common.harness import gpu_test_main, skip_unless_local_checkpoint
from tests.common.models import GEMMA4_26B_A4B_PATCHED
from tests.common.tolerances import TOL
from tests.common.utils import cleanup_memory, log

MODEL = env_str("HALO_TEST_EP_RT_MODEL", GEMMA4_26B_A4B_PATCHED)
ATTN = env_str("HALO_TEST_EP_RT_ATTN", "sdpa")
EP_SIZE = env_int("HALO_TEST_EP", None)
# Expert-TP (ETP / EP+ETP): experts are reconstructed across the expert-TP group
# on save. 1 = pure EP. ep_size=1 + HALO_TEST_EP_RT_EXPERT_TP=2 is pure ETP.
EXPERT_TP = env_int("HALO_TEST_EP_RT_EXPERT_TP", 1)
# EP scope. "auto" keeps the group node-local when it fits one NVLink domain and
# resolves to "global" (cross-node, RDMA) when ep_group_size exceeds the domain — so the
# same test validates the cross-node gathered-save path (ep_size=16 on 2×8) without a
# config change. Single-node usage (ep_group_size <= domain) stays node-local.
EP_SCOPE = env_str("HALO_TEST_EP_RT_SCOPE", "auto")
# Reloaded-vs-reference loss must match to bf16 round-trip noise. A dropped expert
# axis shifts the loss by >>1, so this tolerance is comfortably discriminating.
LOSS_TOL = TOL.resume_loss_abs
# The sequence the fixed-batch loss is scored on before the save and after the reload.
FIXED_TEXT = (
    "User: What is 17 plus 25?\nAssistant: The answer is 42. "
    "Expert routing must survive a checkpoint save and reload intact."
)


def _assert_index_keys(save_dir: str) -> list[str]:
    """Inspect the saved index: no doubled ``experts.experts.`` keys; fused/individual
    expert keys present. Returns the list of problems (empty == OK)."""
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

    # Unique per model and run: concurrent runs sharing TMPDIR must not collide. Every node's writer
    # saves under the same path on its own filesystem.
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
    save_ep_model(model, save_dir, tokenizer=tokenizer, sharded=False)
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
