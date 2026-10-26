"""Every checkpoint reader resolves "where do this directory's tensors live" the same way.

The cascade (sharded index → single ``model.safetensors`` → legacy ``pytorch_model.bin``) has one
implementation, ``resolve_checkpoint_weights``; the resume key readers, ``load_full_state_dict``,
the lazy loaders' ``require_safetensors`` and the standalone tools' ``checkpoint_shard_files``
are policy on top of it. These tests fail when one of them re-forks the walk — a divergent
precedence (a stale single file shadowing the index), a lost leg, or an index read that starts
opening shards.
"""

import json
import os

import pytest
import torch
from safetensors.torch import save_file

from halo.checkpoint.format import LEGACY_WEIGHTS_FILE, WHOLE_MODEL_WEIGHT_FILES
from halo.checkpoint.reading import (
    StreamingCheckpointReader,
    checkpoint_shard_files,
    has_whole_model_weight_file,
    load_full_state_dict,
    read_checkpoint_config,
    resolve_checkpoint_weights,
)


def _lazy_index(directory: str):
    layout = resolve_checkpoint_weights(directory).require_safetensors("checkpoint")
    return layout.weight_map, list(layout.shard_files)


def _write_sharded(directory, shards: dict[str, dict[str, torch.Tensor]]) -> None:
    weight_map = {}
    for name, tensors in shards.items():
        save_file(tensors, os.path.join(directory, name))
        weight_map.update(dict.fromkeys(tensors, name))
    with open(os.path.join(directory, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": {"total_size": 0}, "weight_map": weight_map}, f)


def test_readers_agree_on_a_sharded_index(tmp_path):
    _write_sharded(
        tmp_path,
        {
            "model-00001-of-00002.safetensors": {"a.weight": torch.ones(2, 2)},
            "model-00002-of-00002.safetensors": {"b.weight": torch.full((3,), 2.0)},
        },
    )
    directory = str(tmp_path)
    expected = {"a.weight", "b.weight"}

    assert resolve_checkpoint_weights(directory).key_set() == expected
    assert set(load_full_state_dict(directory)) == expected
    assert set(_lazy_index(directory)[0]) == expected
    assert {os.path.basename(p) for p in checkpoint_shard_files(directory)} == {
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
    }
    with StreamingCheckpointReader(directory, ["b.weight"]) as reader:
        assert torch.equal(reader.get("b.weight"), torch.full((3,), 2.0))


def test_readers_agree_on_a_single_file(tmp_path):
    save_file({"a.weight": torch.ones(2, 2)}, os.path.join(tmp_path, "model.safetensors"))
    directory = str(tmp_path)

    assert resolve_checkpoint_weights(directory).key_set() == {"a.weight"}
    assert set(load_full_state_dict(directory)) == {"a.weight"}
    weight_map, shard_files = _lazy_index(directory)
    assert weight_map == {"a.weight": "model.safetensors"}
    assert shard_files == ["model.safetensors"]
    assert checkpoint_shard_files(directory) == [os.path.join(directory, "model.safetensors")]


def test_an_index_is_read_as_the_map_itself(tmp_path):
    """The index answers a key query on its own — no shard is opened, so keys resolve even when the
    shards are absent (the fast path a 100B+ resume gate depends on)."""
    with open(os.path.join(tmp_path, "model.safetensors.index.json"), "w") as f:
        json.dump({"weight_map": {"a.weight": "model-00001-of-00001.safetensors"}}, f)
    directory = str(tmp_path)

    assert resolve_checkpoint_weights(directory).key_set() == {"a.weight"}
    assert _lazy_index(directory) == (
        {"a.weight": "model-00001-of-00001.safetensors"},
        ["model-00001-of-00001.safetensors"],
    )


def test_the_index_wins_over_a_stale_single_file(tmp_path):
    """Precedence is one decision: a leftover ``model.safetensors`` beside a fresh index must not
    change what any reader reports, or a resume and an export would disagree on the same directory.
    """
    _write_sharded(tmp_path, {"model-00001-of-00001.safetensors": {"fresh.weight": torch.ones(2)}})
    save_file({"stale.weight": torch.zeros(2)}, os.path.join(tmp_path, "model.safetensors"))
    directory = str(tmp_path)

    assert resolve_checkpoint_weights(directory).key_set() == {"fresh.weight"}
    assert set(load_full_state_dict(directory)) == {"fresh.weight"}
    assert set(_lazy_index(directory)[0]) == {"fresh.weight"}
    assert checkpoint_shard_files(directory) == [os.path.join(directory, "model-00001-of-00001.safetensors")]


def test_the_legacy_bin_is_the_last_leg_and_only_the_resume_readers_take_it(tmp_path):
    torch.save({"a.weight": torch.ones(2, 2)}, os.path.join(tmp_path, "pytorch_model.bin"))
    directory = str(tmp_path)

    layout = resolve_checkpoint_weights(directory)
    assert layout.shard_files == () and layout.legacy_bin == "pytorch_model.bin"
    assert resolve_checkpoint_weights(directory).key_set() == {"a.weight"}
    assert set(load_full_state_dict(directory)) == {"a.weight"}
    # The lazy loaders and the standalone tools are safetensors-only: a .bin reads as absent.
    with pytest.raises(FileNotFoundError):
        _lazy_index(directory)
    with pytest.raises(FileNotFoundError):
        checkpoint_shard_files(directory)


@pytest.mark.parametrize("filename", WHOLE_MODEL_WEIGHT_FILES)
def test_every_whole_model_filename_answers_both_probes_the_same_way(tmp_path, filename):
    """The stats-only probe and its safetensors-only mode are ONE list of filenames.

    The lazy gate asks the narrower question (a legacy ``.bin`` is a whole-model checkpoint it cannot
    read) but must ask it over the SAME list: over its own copy of the names, a new whole-model
    filename lands in one probe alone and silently routes every lazy load down the eager fallback.
    """
    (tmp_path / filename).write_bytes(b"")
    directory = str(tmp_path)

    assert has_whole_model_weight_file(directory)
    assert has_whole_model_weight_file(directory, safetensors_only=True) is (filename != LEGACY_WEIGHTS_FILE), (
        f"the lazy gate must accept every whole-model filename but {LEGACY_WEIGHTS_FILE}"
    )


def test_an_empty_directory_is_absent_not_an_error_for_the_resume_readers(tmp_path):
    directory = str(tmp_path)

    assert resolve_checkpoint_weights(directory).shard_files == ()
    assert resolve_checkpoint_weights(directory).key_set() == set()
    assert load_full_state_dict(directory) is None


def test_absent_shard_files_names_only_the_missing_index_shards(tmp_path):
    _write_sharded(
        tmp_path,
        {
            "model-00001-of-00003.safetensors": {"a.weight": torch.ones(1)},
            "model-00002-of-00003.safetensors": {"b.weight": torch.ones(1)},
            "model-00003-of-00003.safetensors": {"c.weight": torch.ones(1)},
        },
    )
    for name in ("model-00003-of-00003.safetensors", "model-00002-of-00003.safetensors"):
        os.remove(tmp_path / name)
    layout = resolve_checkpoint_weights(str(tmp_path))

    assert layout.absent_shard_files() == ["model-00002-of-00003.safetensors", "model-00003-of-00003.safetensors"]
    assert layout.absent_shard_files(["a.weight", "c.weight"]) == ["model-00003-of-00003.safetensors"]
    assert layout.absent_shard_files(["a.weight", "not.in.index"]) == []


def test_require_safetensors_refuses_a_bin_only_layout_and_passes_a_safetensors_one(tmp_path):
    bin_only, single = tmp_path / "bin", tmp_path / "single"
    bin_only.mkdir(), single.mkdir()
    torch.save({"a.weight": torch.ones(1)}, bin_only / LEGACY_WEIGHTS_FILE)
    save_file({"a.weight": torch.ones(1)}, single / "model.safetensors")

    with pytest.raises(FileNotFoundError, match="No safetensors checkpoint found"):
        resolve_checkpoint_weights(str(bin_only)).require_safetensors("checkpoint")
    layout = resolve_checkpoint_weights(str(single))
    assert layout.require_safetensors("checkpoint") is layout


def test_read_checkpoint_config_tolerates_only_an_absent_file(tmp_path):
    assert read_checkpoint_config(str(tmp_path), missing_ok=True) is None
    with pytest.raises(FileNotFoundError):
        read_checkpoint_config(str(tmp_path))
    (tmp_path / "config.json").write_text("{")
    with pytest.raises(json.JSONDecodeError):
        read_checkpoint_config(str(tmp_path), missing_ok=True)
    (tmp_path / "config.json").write_text('{"model_type": "qwen3"}')
    assert read_checkpoint_config(str(tmp_path)) == {"model_type": "qwen3"}


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
