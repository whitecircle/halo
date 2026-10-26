#!/usr/bin/env python
"""``merge_pp_stage_parts``: the export step of a pipeline-parallel save on a non-shared filesystem.

Each node's directory holds the whole-model index, the parts of the stages that node ran and the wrapper
parts every node writes. Merged, the directories must load through plain ``from_pretrained`` as the unsplit
model; a stage whose parts no directory holds must fail the merge instead of leaving a checkpoint that
loads around the gap. Each node also holds its own ranks' optimizer shards and RNG states and an identical
optimizer meta; the merge carries all of them, so the merged directory resumes exactly on the save's topology.

Run: python tests/cpu/checkpoint/test_pp_stage_merge.py
"""

import os
import shutil

import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from halo.checkpoint.format import (
    OPTIMIZER_META_FILE,
    PP_STAGE_PART_RE,
    SAFETENSORS_INDEX_FILE,
    optimizer_shard_filename,
    pp_stage_part_prefix,
)
from halo.checkpoint.pp_stage_merge import merge_pp_stage_parts
from halo.checkpoint.writing import StreamingShardWriter, write_merged_index
from tests.common.models import TINY_QWEN3_CONFIG

PP_SIZE = 2
WRAPPER_KEYS = ("lm_head.weight",)
RANKS_PER_NODE = 2
# The payload ``write_optimizer_meta`` writes, identical on every node of one save.
OPTIMIZER_META = {
    "num_ranks": PP_SIZE * RANKS_PER_NODE,
    "fingerprint": {"world_size": PP_SIZE * RANKS_PER_NODE, "pp_size": PP_SIZE, "optimizer_class": "AdamW"},
    "pp_stage_partition": [[0, 4], [4, 8]],
}


def _stage_of(key: str, first_stage_layers: int) -> int:
    if key.startswith("model.layers."):
        return int(int(key.split(".")[2]) >= first_stage_layers)
    return 0 if key.startswith("model.embed_tokens") else 1


def _write_node_directories(root: str, state: dict[str, torch.Tensor], config: Qwen3Config) -> list[str]:
    """One directory per stage-node as the per-node save leaves them: the stage's own parts (several, under
    a small shard cap), the wrapper part on every node, the whole index and the config on each."""
    first_stage_layers = config.num_hidden_layers // 2
    weight_map, total = {}, 0
    node_dirs = [os.path.join(root, f"node{rank}") for rank in range(PP_SIZE)]
    for rank, node_dir in enumerate(node_dirs):
        os.makedirs(node_dir)
        config.save_pretrained(node_dir)
        stage = StreamingShardWriter(
            node_dir, "64KB", enabled=True, filename_prefix=pp_stage_part_prefix(rank, PP_SIZE)
        )
        wrapper = StreamingShardWriter(node_dir, "64KB", enabled=True, filename_prefix="model-wrapper")
        for key, tensor in state.items():
            if key in WRAPPER_KEYS:
                wrapper.add(key, tensor)
            elif _stage_of(key, first_stage_layers) == rank:
                stage.add(key, tensor)
        for writer in (stage, wrapper):
            fragment, written = writer.close()
            weight_map.update(fragment)
            total += written
    for node_dir in node_dirs:
        write_merged_index(node_dir, weight_map, {"total_size": total})
    return node_dirs


def _write_resume_state(node_dirs: list[str], meta: dict = OPTIMIZER_META) -> None:
    """Each node's own ranks' optimizer shards and RNG states, and the meta every node writes."""
    for node, node_dir in enumerate(node_dirs):
        for rank in range(node * RANKS_PER_NODE, (node + 1) * RANKS_PER_NODE):
            torch.save(
                {"state": {0: {"exp_avg": torch.full((4,), float(rank))}}},
                os.path.join(node_dir, optimizer_shard_filename(rank)),
            )
            torch.save({"python": rank}, os.path.join(node_dir, f"rng_state_{rank}.pth"))
        torch.save(meta, os.path.join(node_dir, OPTIMIZER_META_FILE))


@pytest.fixture
def saved(tmp_path) -> tuple[list[str], dict[str, torch.Tensor]]:
    torch.manual_seed(0)
    config = Qwen3Config(**TINY_QWEN3_CONFIG)
    state = {key: tensor.clone() for key, tensor in Qwen3ForCausalLM(config).state_dict().items()}
    return _write_node_directories(str(tmp_path / "save"), state, config), state


def test_merged_node_directories_load_as_the_unsplit_model(saved, tmp_path):
    node_dirs, state = saved
    stage_parts = [sorted(f for f in os.listdir(d) if f.startswith("model-pp")) for d in node_dirs]
    assert all(len(parts) > 1 for parts in stage_parts), f"premise: each stage spans several parts: {stage_parts}"
    assert not set(stage_parts[0]) & set(stage_parts[1]), "premise: no node holds another stage's part"

    merged = str(tmp_path / "merged")
    merge_pp_stage_parts(node_dirs, merged)

    loaded = Qwen3ForCausalLM.from_pretrained(merged).state_dict()
    assert loaded.keys() == state.keys()
    unequal = [key for key, tensor in state.items() if not torch.equal(loaded[key], tensor)]
    assert not unequal, f"merged values differ from the saved model: {unequal[:6]}"


def test_the_merged_directory_carries_every_ranks_resume_state(saved, tmp_path):
    node_dirs, _ = saved
    _write_resume_state(node_dirs)
    merged = tmp_path / "merged"
    merge_pp_stage_parts(node_dirs, str(merged))

    ranks = range(PP_SIZE * RANKS_PER_NODE)
    assert sorted(f for f in os.listdir(merged) if f.startswith("optimizer_shard_")) == [
        optimizer_shard_filename(rank) for rank in ranks
    ]
    assert sorted(f for f in os.listdir(merged) if f.startswith("rng_state")) == [
        f"rng_state_{rank}.pth" for rank in ranks
    ]
    assert torch.load(merged / OPTIMIZER_META_FILE, weights_only=True) == OPTIMIZER_META
    shard = torch.load(merged / optimizer_shard_filename(3), weights_only=True)
    assert torch.equal(shard["state"][0]["exp_avg"], torch.full((4,), 3.0)), "rank 3's shard is not its own"


def test_a_weights_only_save_merges_without_resume_state(saved, tmp_path):
    node_dirs, _ = saved
    merged = tmp_path / "merged"
    merge_pp_stage_parts(node_dirs, str(merged))
    assert not [f for f in os.listdir(merged) if f.startswith(("optimizer", "rng_state"))]


def test_a_rank_whose_optimizer_shard_no_node_holds_fails_the_merge(saved, tmp_path):
    node_dirs, _ = saved
    _write_resume_state(node_dirs)
    os.remove(os.path.join(node_dirs[1], optimizer_shard_filename(3)))
    merged = tmp_path / "merged"
    with pytest.raises(ValueError, match=r"optimizer shards of ranks \[3\]"):
        merge_pp_stage_parts(node_dirs, str(merged))
    assert not merged.exists()


def test_node_directories_with_different_optimizer_metas_are_refused(saved, tmp_path):
    node_dirs, _ = saved
    _write_resume_state(node_dirs)
    torch.save(
        {**OPTIMIZER_META, "pp_stage_partition": [[0, 3], [3, 8]]}, os.path.join(node_dirs[1], OPTIMIZER_META_FILE)
    )
    with pytest.raises(ValueError, match=f"different {OPTIMIZER_META_FILE}"):
        merge_pp_stage_parts(node_dirs, str(tmp_path / "merged"))


def test_a_per_rank_file_held_at_different_sizes_is_refused(saved, tmp_path):
    node_dirs, _ = saved
    _write_resume_state(node_dirs)
    torch.save({"python": 0, "torn": list(range(64))}, os.path.join(node_dirs[1], "rng_state_0.pth"))
    with pytest.raises(ValueError, match=r"rng_state_0\.pth differs in size"):
        merge_pp_stage_parts(node_dirs, str(tmp_path / "merged"))


def test_a_stage_part_no_node_holds_fails_the_merge(saved, tmp_path):
    """Negative control: the index names a stage-1 part that no node directory carries."""
    node_dirs, _ = saved
    missing = sorted(f for f in os.listdir(node_dirs[1]) if f.startswith("model-pp00001"))[0]
    os.remove(os.path.join(node_dirs[1], missing))

    merged = tmp_path / "merged"
    with pytest.raises(ValueError, match=rf"No node directory holds {missing} \(stage 1\)"):
        merge_pp_stage_parts(node_dirs, str(merged))
    assert not (merged / SAFETENSORS_INDEX_FILE).exists()


def test_only_one_node_directory_is_refused(saved, tmp_path):
    node_dirs, _ = saved
    with pytest.raises(ValueError, match=r"model-pp00001-of-00002-00001\.safetensors \(stage 1\)"):
        merge_pp_stage_parts(node_dirs[:1], str(tmp_path / "merged"))


def test_directories_of_different_saves_are_refused(saved, tmp_path):
    node_dirs, state = saved
    other = _write_node_directories(
        str(tmp_path / "other"), {key: tensor[:1] for key, tensor in state.items()}, Qwen3Config(**TINY_QWEN3_CONFIG)
    )
    shutil.rmtree(node_dirs[1])
    shutil.copytree(other[1], node_dirs[1])
    with pytest.raises(ValueError, match="different safetensors indexes"):
        merge_pp_stage_parts(node_dirs, str(tmp_path / "merged"))


def test_the_stage_part_pattern_reads_back_the_builders_names():
    match = PP_STAGE_PART_RE.match(f"{pp_stage_part_prefix(3, 12)}-00001.safetensors")
    assert match is not None and (int(match[1]), int(match[2])) == (3, 12)
    assert PP_STAGE_PART_RE.match("model-wrapper-00001.safetensors") is None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
