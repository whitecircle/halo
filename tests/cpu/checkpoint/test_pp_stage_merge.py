#!/usr/bin/env python
"""``merge_pp_stage_parts``: the export step of a pipeline-parallel save on a non-shared filesystem.

Each node's directory holds the whole-model index, the parts of the stages that node ran and the wrapper
parts every node writes. Merged, the directories must load through plain ``from_pretrained`` as the unsplit
model; a stage whose parts no directory holds must fail the merge instead of leaving a checkpoint that
loads around the gap.

Run: python tests/cpu/checkpoint/test_pp_stage_merge.py
"""

import os
import shutil

import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from src.checkpoint.format import SAFETENSORS_INDEX_FILE
from src.checkpoint.pp_stage_merge import merge_pp_stage_parts
from src.checkpoint.writing import StreamingShardWriter, write_merged_index
from tests.common.models import TINY_QWEN3_CONFIG

PP_SIZE = 2
WRAPPER_KEYS = ("lm_head.weight",)


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
            node_dir, "64KB", enabled=True, filename_prefix=f"model-pp{rank:05d}-of-{PP_SIZE:05d}"
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


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
