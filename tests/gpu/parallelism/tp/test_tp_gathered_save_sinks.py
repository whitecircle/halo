#!/usr/bin/env python
"""The TP gathered save writes every hand-sliced GptOss sink once, at full head width.

Sinks can't be DTensors (the forward concatenates them with sharded logits), so ``_tp_chunks`` gathers them after the
parameter walk, which must drop them: it would emit this rank's slice under the same key. ``max_shard_size`` forces a
multi-part save, since a single part has no index for the ``total_size`` assertion.
"""

import json
import os
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file
from torch.distributed.device_mesh import init_device_mesh
from transformers import GptOssConfig, GptOssForCausalLM

# A hand-built model reaches save_tp_checkpoint without the load path that registers the EP export roster.
from halo.checkpoint.format import SAFETENSORS_INDEX_FILE
from halo.checkpoint.parallel.save import save_tp_checkpoint
from halo.distributed.tensor_parallel.parallelize_attention import apply_tp_to_attention_only
from halo.distributed.tensor_parallel.plain_params import hand_sliced_suffixes
from tests.common.checkpoint_io import save_context, weight_files
from tests.common.distributed import shared_scratch_dir
from tests.common.harness import gpu_test_main
from tests.common.models import TINY_GPTOSS_CONFIG
from tests.common.rank_log import log

TP_SIZE = 2
NUM_LAYERS = TINY_GPTOSS_CONFIG["num_hidden_layers"]
NUM_HEADS = TINY_GPTOSS_CONFIG["num_attention_heads"]
MAX_SHARD_SIZE = "4MB"
DISK_CHECKS = (
    "every_key_written_once",
    "sinks_saved_whole",
    "index_total_size_matches_shards",
    "reloads_with_the_saved_sinks",
)


def _tiny_gpt_oss(device: torch.device) -> GptOssForCausalLM:
    torch.manual_seed(1234)
    config = GptOssConfig(**TINY_GPTOSS_CONFIG, pad_token_id=0, eos_token_id=1)
    return GptOssForCausalLM(config).to(device=device, dtype=torch.bfloat16)


def _full_sinks(layer_idx: int, device: torch.device) -> torch.Tensor:
    """Unique per (layer, head), so a truncated, wrong-layer or wrong-rank write is visible."""
    start = layer_idx * NUM_HEADS + 1
    return torch.arange(start, start + NUM_HEADS, device=device, dtype=torch.bfloat16)


def _shard_contents(save_dir: str) -> dict[str, dict[str, torch.Tensor]]:
    """File by file: ``load_full_state_dict`` collapses a key claimed by two parts, the corruption under test."""
    return {name: load_file(os.path.join(save_dir, name)) for name in weight_files(save_dir)}


def _checkpoint_problems(save_dir: str, expected: dict[str, torch.Tensor]) -> dict[str, list[str]]:
    shards = _shard_contents(save_dir)
    owners: dict[str, list[str]] = {}
    for part, tensors in shards.items():
        for key in tensors:
            owners.setdefault(key, []).append(part)

    sinks: list[str] = []
    for key, want in expected.items():
        parts = owners.get(key, [])
        if len(parts) != 1:
            sinks.append(f"{key}: claimed by {parts or 'no shard file'}, and exactly one part must own it")
            continue
        got = shards[parts[0]][key]
        if tuple(got.shape) != (NUM_HEADS,):
            sinks.append(
                f"{key}: {tuple(got.shape)} on disk, expected all {NUM_HEADS} heads — a TP rank's "
                f"head slice was written under the full-tensor key"
            )
        elif not torch.equal(got.float(), want):
            sinks.append(f"{key}: saved {got.tolist()} != pre-shard {want.tolist()}")

    index: list[str] = []
    if len(shards) < 2:
        index.append(
            f"the save wrote {len(shards)} part(s) and therefore no index: MAX_SHARD_SIZE "
            f"({MAX_SHARD_SIZE}) no longer splits this model, so total_size goes unchecked"
        )
    else:
        declared = json.loads(Path(save_dir, SAFETENSORS_INDEX_FILE).read_text())
        total_size = declared["metadata"]["total_size"]
        weight_map = declared["weight_map"]
        stored_bytes = sum(t.numel() * t.element_size() for tensors in shards.values() for t in tensors.values())
        # Both sums: against the parts, total_size misses a file the index never named; against its entries, a key
        # written twice.
        named_bytes = sum(
            shards[part][key].numel() * shards[part][key].element_size() for key, part in weight_map.items()
        )
        if total_size != stored_bytes:
            index.append(
                f"index total_size {total_size} != the {stored_bytes} B actually stored across {len(shards)} parts"
            )
        if total_size != named_bytes:
            index.append(
                f"index total_size {total_size} is inflated over the {named_bytes} B of the "
                f"{len(weight_map)} keys it names — a key claimed by two parts is counted twice"
            )
        if set(weight_map) != set(owners):
            index.append(f"the index names {len(weight_map)} keys but the parts hold {len(owners)}")

    duplicates = sorted((key, parts) for key, parts in owners.items() if len(parts) > 1)
    return {
        "every_key_written_once": [f"{key} claimed by {parts}" for key, parts in duplicates],
        "sinks_saved_whole": sinks,
        "index_total_size_matches_shards": index,
        "reloads_with_the_saved_sinks": _reload_problems(save_dir, expected),
    }


def _reload_problems(save_dir: str, expected: dict[str, torch.Tensor]) -> list[str]:
    reloaded = GptOssForCausalLM.from_pretrained(save_dir, dtype=torch.bfloat16, attn_implementation="eager")
    problems = []
    for index, layer in enumerate(reloaded.model.layers):
        key = f"model.layers.{index}.self_attn.sinks"
        got = layer.self_attn.sinks.detach().float().cpu()
        if not torch.equal(got, expected[key]):
            problems.append(f"{key}: reloaded {got.tolist()} != pre-save {expected[key].tolist()}")
    return problems


def _disk_checks(save_dir: str, expected: dict[str, torch.Tensor], save_error: str | None) -> dict[str, bool]:
    """Rank 0's verdict, never a raise, which its peers would wait out at the broadcast until the watchdog fires."""
    if save_error is not None:
        log(f"save_tp_checkpoint raised, so there is no artifact to inspect: {save_error}")
        return dict.fromkeys(DISK_CHECKS, False)
    try:
        problems = _checkpoint_problems(save_dir, expected)
    except Exception as exc:
        log(f"reading back the checkpoint in {save_dir} failed: {type(exc).__name__}: {exc}")
        return dict.fromkeys(DISK_CHECKS, False)
    for name in DISK_CHECKS:
        for problem in problems[name]:
            log(f"  {name}: {problem}")
    return {name: not problems[name] for name in DISK_CHECKS}


def run(ctx) -> dict:
    save_dir = shared_scratch_dir("halo_tp_gathered_save_sinks")
    if ctx.rank == 0:
        shutil.rmtree(save_dir, ignore_errors=True)
        ctx.on_teardown(lambda: shutil.rmtree(save_dir, ignore_errors=True))

    model = _tiny_gpt_oss(ctx.device)
    expected = {}
    for index, layer in enumerate(model.model.layers):
        full = _full_sinks(index, ctx.device)
        with torch.no_grad():
            layer.self_attn.sinks.copy_(full)
        expected[f"model.layers.{index}.self_attn.sinks"] = full.float().cpu()

    mesh = init_device_mesh("cuda", (TP_SIZE,), mesh_dim_names=("tp",))
    model._device_mesh = mesh
    patched = apply_tp_to_attention_only(model, mesh)

    # Anti-vacuity: with nothing for _tp_chunks to exclude, every assertion below holds for the wrong reason.
    checks = {
        "tp_sliced_the_sinks_by_hand": patched == NUM_LAYERS
        and {layer.self_attn.sinks.shape[0] for layer in model.model.layers} == {NUM_HEADS // TP_SIZE}
        and hand_sliced_suffixes(model) == ("self_attn.sinks",)
    }

    ctx.barrier()  # Rank 0's wipe of the shared scratch dir must land before the save writes into it.

    save_error = None
    try:
        save_tp_checkpoint(save_context(model, max_shard_size=MAX_SHARD_SIZE), save_dir)
    except Exception as exc:
        # The writer defers a rank-local failure to a collective, so every rank lands here together.
        save_error = f"{type(exc).__name__}: {exc}"
    checks["save_completed"] = save_error is None

    ctx.barrier()
    if ctx.rank == 0:
        checks.update(_disk_checks(save_dir, expected, save_error))
    checks = ctx.broadcast_checks(checks)

    if all(checks.values()):
        log(f"TP gathered save wrote {NUM_LAYERS} whole {NUM_HEADS}-head sinks, once each")
    return {"checks": checks}


main = gpu_test_main(exact_world_size=TP_SIZE, prefix="tp_gathered_save_sinks")(run)

if __name__ == "__main__":
    main()
