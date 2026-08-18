#!/usr/bin/env python
"""ZAYA1-8B EP checkpoint save correctness test.

Guards the production gathered-save path (``save_sharded_ep: false``) for Zaya. The canonical
``Zyphra/ZAYA1`` checkpoint — and every loader that reads it — stores experts FUSED
(``mlp.experts.gate_up_proj`` ``[E, 2M, H]`` / ``mlp.experts.down_proj`` ``[E, H, M]``), so the base
fused gather is the layout the save must emit; the pre-5.14 per-expert spelling
(``experts.local_experts.{i}.linear_fc{1,2}.weight``) is read by nothing and would make the loader
silently skip every expert.

This test loads ZAYA1-8B under EP (no training), saves via
``save_ep_model(sharded=False)``, and on rank 0 verifies:
  1. The saved checkpoint carries the fused expert tensors, two per MoE layer, with the FULL expert
     count on dim 0 (the EP gather reassembled every rank's slice) and no per-expert key at all.
  2. Those tensors are bit-identical to the original base checkpoint's — i.e. EP load (slice) → save
     (gather) is a loss-less round-trip. A router (non-expert) tensor is checked too.

The reload is verified against the on-disk base rather than by re-loading under
EP in-process: DeepEP does not support recreating its buffers within a single
process (see ``merge_ep_shards.py`` / ``load_ep_model``), so a second EP load
would deadlock in ``create_ep_buffers``.

Run (2 GPUs, EP=2):
    docker run --rm --gpus '"device=4,5"' --ipc=host --ulimit memlock=-1 \\
        --ulimit stack=67108864 \\
        -v $(pwd):/workspace \\
        -v /root/.cache/huggingface:/root/.cache/huggingface \\
        -w /workspace -e HF_HOME=/root/.cache/huggingface \\
        halo:blackwell \\
        torchrun --nproc_per_node=2 \\
            tests/gpu/parallelism/ep/test_zaya_ep_save_roundtrip.py
"""

import json
import os
import re

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open

from src.distributed.expert_parallel.layers.zaya import EPZayaMoELayer
from src.distributed.expert_parallel.saving import save_ep_model
from src.distributed.loading.model_loading import load_distributed_model
from src.distributed.parallelism_config import ParallelismConfig
from src.env import env_str
from tests.common.harness import gpu_test_main
from tests.common.models import ZAYA_8B
from tests.common.utils import log

MODEL = env_str("HALO_TEST_ZAYA_MODEL", ZAYA_8B)

_FUSED_RE = re.compile(r"\.experts\.(gate_up_proj|down_proj)$")
_PER_EXPERT_RE = re.compile(r"\.experts\.local_experts\.\d+\.linear_fc[12]\.weight$")


class ShardedReader:
    """Read individual tensors from a (possibly sharded) safetensors checkpoint."""

    def __init__(self, checkpoint_dir: str):
        index_path = os.path.join(checkpoint_dir, "model.safetensors.index.json")
        if os.path.exists(index_path):
            with open(index_path) as f:
                self._weight_map = json.load(f)["weight_map"]
        else:
            self._weight_map = None  # single-file checkpoint
        self._dir = checkpoint_dir

    def keys(self) -> list[str]:
        if self._weight_map is not None:
            return list(self._weight_map)
        with safe_open(os.path.join(self._dir, "model.safetensors"), framework="pt") as f:
            return list(f.keys())

    def get(self, key: str) -> torch.Tensor:
        filename = self._weight_map[key] if self._weight_map is not None else "model.safetensors"
        with safe_open(os.path.join(self._dir, filename), framework="pt") as f:
            return f.get_tensor(key)


def _verify_saved_checkpoint(save_dir: str, n_experts: int) -> dict[str, bool]:
    """Rank 0: the saved layout is the canonical fused one, bit-identical to the base checkpoint."""
    saved = ShardedReader(save_dir)
    saved_keys = saved.keys()
    fused = [k for k in saved_keys if _FUSED_RE.search(k)]
    per_expert = [k for k in saved_keys if _PER_EXPERT_RE.search(k)]
    # The MoE container the wrapper replaced (``model.layers.N.mlp``), read off the saved
    # keys rather than assumed, so a renamed container fails the count below instead of
    # silently matching nothing.
    containers = sorted(
        {k.rsplit(".experts.", 1)[0] for k in fused},
        key=lambda c: int(re.search(r"layers\.(\d+)\.", c).group(1)),
    )
    log(f"  fused expert tensors:         {len(fused)} over {len(containers)} MoE layers")
    log(f"  per-expert linear_fc tensors: {len(per_expert)} (must be 0)")

    checks = {
        "fused_experts_saved": bool(fused),
        "no_per_expert_keys": not per_expert,
        "two_fused_tensors_per_moe_layer": len(fused) == len(containers) * 2,
    }
    if not all(checks.values()):
        return checks
    log(f"  ✓ {len(containers)} MoE layers × (gate_up_proj, down_proj), all fused (canonical)")

    # Compare saved weights to the original base checkpoint (ground truth).
    base_dir = MODEL if os.path.isdir(MODEL) else snapshot_download(MODEL, allow_patterns=["*.safetensors*"])
    base = ShardedReader(base_dir)
    experts_gathered, experts_match, routers_match = [], [], []
    for container in {containers[0], containers[-1]}:
        for name in ("gate_up_proj", "down_proj"):
            key = f"{container}.experts.{name}"
            tensor = saved.get(key)
            # Every rank's slice, not just rank 0's: an ungathered save has E/world here.
            if tensor.shape[0] != n_experts:
                log(f"  {key} holds {tensor.shape[0]} experts, expected all {n_experts}")
            experts_gathered.append(tensor.shape[0] == n_experts)
            experts_match.append(torch.equal(tensor, base.get(key)))
        # A non-expert (router) tensor must also round-trip exactly.
        router_key = f"{container}.gate.down_proj.weight"
        routers_match.append(torch.equal(saved.get(router_key), base.get(router_key)))

    checks["all_experts_gathered"] = all(experts_gathered)
    checks["expert_weights_match_base"] = all(experts_match)
    checks["router_weights_match_base"] = all(routers_match)
    log(f"  ✓ {len(experts_match) + len(routers_match)} sampled tensors compared against base")
    return checks


def run(ctx) -> dict:
    save_dir = os.path.join(ctx.output_dir, "saved")

    log("=" * 70)
    log("  ZAYA1-8B EP save correctness test")
    log(f"  Model: {MODEL}  World/EP: {ctx.world_size}")
    log("=" * 70)

    pc = ParallelismConfig(ep_size=ctx.world_size, use_grouped_gemm=True)

    # ── [1/3] Load under EP ────────────────────────────────────────
    log("\n[1/3] Loading ZAYA1-8B under EP...")
    model, tok = load_distributed_model(
        model_name_or_path=MODEL,
        parallelism_config=pc,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        use_liger_kernel=True,
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    ep_layers = [m for m in model.modules() if isinstance(m, EPZayaMoELayer)]
    assert ep_layers, "No EPZayaMoELayer found — patching failed"
    n_experts = ep_layers[0].num_experts
    log(f"  ✓ {len(ep_layers)} EP layers, {n_experts} experts each")

    # ── [2/3] Save (gathered, fused) ───────────────────────────────
    log("\n[2/3] Saving via save_ep_model(sharded=False)...")
    save_ep_model(model, save_dir, tokenizer=tok, sharded=False)
    del model
    torch.cuda.empty_cache()
    ctx.barrier()

    # ── [3/3] Verify layout + weight equality on rank 0 ────────────
    checks: dict[str, bool] = {}
    if ctx.rank == 0:
        log("\n[3/3] Verifying saved checkpoint (rank 0)...")
        checks = _verify_saved_checkpoint(save_dir, n_experts)
    return {"checks": ctx.broadcast_checks(checks)}


main = gpu_test_main(min_world_size=1, prefix="test_zaya_ep_rt")(run)

if __name__ == "__main__":
    main()
