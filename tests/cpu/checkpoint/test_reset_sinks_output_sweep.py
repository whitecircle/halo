#!/usr/bin/env python
"""``reset_sinks`` hands back a directory holding only the checkpoint it wrote, rewritten shard by shard.
Every rewritten shard is staged and verified at dtype min before anything replaces its target."""

import json
import logging
import os
import time

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from transformers import GptOssConfig, GptOssForCausalLM

import halo.checkpoint.tools.sink_reset as reset_sinks_mod
from halo.checkpoint.format import REFERENCE_CACHE_DIR_NAME, RUN_PLAN_FILE, SAFETENSORS_METADATA
from halo.checkpoint.reading import checkpoint_shard_files
from halo.checkpoint.tools import offline_export
from halo.checkpoint.tools.sink_reset import reset_sinks
from tests.common.checkpoint_io import weight_files

INDEX = "model.safetensors.index.json"
SINGLE = "model.safetensors"
# The round-trip assert reads these, so a fixture change cannot silently turn it vacuous.
VOCAB_SIZE, HIDDEN_SIZE = 128, 64


def _tiny_gptoss():
    """gpt-oss is the family this tool exists for."""
    torch.manual_seed(0)
    config = GptOssConfig(
        vocab_size=VOCAB_SIZE,
        hidden_size=HIDDEN_SIZE,
        intermediate_size=64,
        num_hidden_layers=2,
        num_local_experts=2,
        num_experts_per_tok=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        sliding_window=32,
        tie_word_embeddings=False,
    )
    return GptOssForCausalLM(config).to(torch.bfloat16)


def _build_source(source_dir, *, sharded: bool) -> dict[str, torch.Tensor]:
    model = _tiny_gptoss()
    model.save_pretrained(str(source_dir), max_shard_size="64KB" if sharded else "5GB")
    assert os.path.isfile(os.path.join(source_dir, INDEX)) == sharded, "the source did not land in the shape asked for"
    sinks = {name: param.detach().clone() for name, param in model.named_parameters() if ".sinks" in name}
    assert sinks, "premise: the fixture carries sinks"
    assert not all(torch.all(t == torch.finfo(t.dtype).min) for t in sinks.values()), (
        "premise: the source sinks are live values, so resetting them is observable"
    )
    return sinks


def _plant_previous_sharded_run(out_dir) -> str:
    """Complete on purpose: only the back-dated modification time says which layout is the leftover."""
    os.makedirs(out_dir, exist_ok=True)
    shards = ("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors")
    for shard in shards:
        with open(os.path.join(out_dir, shard), "wb") as handle:
            handle.write(b"stale-shard")
    with open(os.path.join(out_dir, INDEX), "w") as handle:
        json.dump({"metadata": {}, "weight_map": {f"stale.{i}.weight": s for i, s in enumerate(shards)}}, handle)
    stamp = time.time() - 10
    for name in (*shards, INDEX):
        os.utime(os.path.join(out_dir, name), (stamp, stamp))
    return str(out_dir)


def _weight_files(directory) -> set[str]:
    return set(weight_files(str(directory), include_index=True))  # The sweep owns the index too.


def _sinks_across_shards(directory) -> dict[str, torch.Tensor]:
    values: dict[str, torch.Tensor] = {}
    for shard in checkpoint_shard_files(str(directory)):
        for key, tensor in load_file(shard).items():
            if ".sinks" in key:
                values[key] = tensor
    return values


def _assert_sinks_reset(directory, source_sinks: dict[str, torch.Tensor]) -> None:
    written = _sinks_across_shards(directory)
    assert set(source_sinks) <= set(written), "the reset checkpoint dropped sink tensors"
    for name in source_sinks:
        tensor = written[name]
        assert torch.all(tensor == torch.finfo(tensor.dtype).min), f"{name} was not reset to dtype min"


def _sink_shards(directory, sinks) -> set[str]:
    return {os.path.basename(s) for s in checkpoint_shard_files(str(directory)) if set(load_file(s)) & set(sinks)}


def _shard_bytes(directory) -> dict[str, bytes]:
    return {os.path.basename(shard): open(shard, "rb").read() for shard in checkpoint_shard_files(str(directory))}


def _staged_leftovers(directory) -> list[str]:
    return sorted(name for name in os.listdir(directory) if name.startswith("."))


def test_a_single_file_reset_sweeps_the_previous_runs_leftovers(tmp_path):
    """It would win the index-first lookup over the file just written."""
    source, out = tmp_path / "src", tmp_path / "out"
    sinks = _build_source(source, sharded=False)
    _plant_previous_sharded_run(out)

    assert reset_sinks(str(source), str(out)) == len(sinks)

    assert _weight_files(out) == {SINGLE}, "the previous sharded run outlived the reset"
    _assert_sinks_reset(out, sinks)
    assert load_file(str(out / SINGLE))["model.embed_tokens.weight"].shape == (VOCAB_SIZE, HIDDEN_SIZE)


def test_the_rewrite_stamps_the_safetensors_format(tmp_path):
    source, out = tmp_path / "src", tmp_path / "out"
    _build_source(source, sharded=False)

    reset_sinks(str(source), str(out))

    with safe_open(os.path.join(out, SINGLE), framework="pt") as handle:
        assert handle.metadata() == SAFETENSORS_METADATA


def test_a_sharded_reset_keeps_the_layout_and_rewrites_only_the_sink_shards(tmp_path, monkeypatch):
    """Only one shard is resident at a time, and no model class is involved."""
    source, out = tmp_path / "src", tmp_path / "out"
    sinks = _build_source(source, sharded=True)
    before = _shard_bytes(source)
    sink_shards = _sink_shards(source, sinks)
    assert sink_shards and sink_shards != set(before), "premise: the source mixes sink and sink-free shards"
    loaded: list[str] = []
    real_load = reset_sinks_mod.load_file
    monkeypatch.setattr(reset_sinks_mod, "load_file", lambda path, **kw: loaded.append(path) or real_load(path, **kw))

    assert reset_sinks(str(source), str(out)) == len(sinks)

    after = _shard_bytes(out)
    assert set(after) == set(before) and _weight_files(out) == {*before, INDEX}
    assert all(after[name] == before[name] for name in set(before) - sink_shards), "a sink-free shard was rewritten"
    assert json.loads((out / INDEX).read_text()) == json.loads((source / INDEX).read_text())
    read_back = {os.path.basename(path) for path in loaded}
    assert {name for name in read_back if not name.startswith(".")} == sink_shards, f"shards loaded: {loaded}"
    _assert_sinks_reset(out, sinks)


def test_a_sharded_reset_sweeps_the_previous_runs_files(tmp_path):
    source, out = tmp_path / "src", tmp_path / "out"
    sinks = _build_source(source, sharded=True)
    _plant_previous_sharded_run(out)
    (out / SINGLE).write_bytes(b"stale-single")

    assert reset_sinks(str(source), str(out)) == len(sinks)

    assert _weight_files(out) == _weight_files(source), "a previous run's weight files survived the reset"
    _assert_sinks_reset(out, sinks)


def test_an_interrupted_write_leaves_the_target_and_no_stage(tmp_path, monkeypatch):
    """An unrecognized leftover beside the weights rides along into every later aux-file copy."""
    source = tmp_path / "src"
    _build_source(source, sharded=False)
    before = sorted(os.listdir(source))
    original = (source / SINGLE).read_bytes()

    def interrupted(tensors, filename, metadata=None):
        with open(filename, "wb") as handle:
            handle.write(b"partial")
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(reset_sinks_mod, "save_safetensors_file", interrupted)
    with pytest.raises(OSError, match="No space left"):
        reset_sinks(str(source), in_place=True)

    assert sorted(os.listdir(source)) == before, "the interrupted write left a staged file behind"
    assert (source / SINGLE).read_bytes() == original


def test_an_in_place_run_deletes_nothing_it_did_not_write(tmp_path):
    """A weight file outside the checkpoint's own layout belongs to whoever put it there."""
    source = tmp_path / "src"
    sinks = _build_source(source, sharded=False)
    stray = source / "model-00009-of-00009.safetensors"
    stray.write_bytes(b"not-ours")

    assert reset_sinks(str(source), in_place=True) == len(sinks)

    assert stray.read_bytes() == b"not-ours"
    _assert_sinks_reset(source, sinks)


def test_a_write_that_kept_live_sinks_never_replaces_the_source(tmp_path, monkeypatch):
    """A torn or buggy write must not ship live sinks as a reported success over the only copy."""
    source = tmp_path / "src"
    sinks = _build_source(source, sharded=True)
    sink_shard_count = len(_sink_shards(source, sinks))
    assert sink_shard_count > 1, "premise: the reset stages more than one shard"
    before = {p.name: p.read_bytes() for p in source.iterdir() if p.is_file()}
    real_save_file = reset_sinks_mod.save_safetensors_file
    saves: list[str] = []

    def corrupting_save_file(state_dict, path, **kwargs):
        # A torn writer on the last staged shard, after the others verified.
        saves.append(str(path))
        last = len(saves) == sink_shard_count
        live = {key: (torch.zeros_like(t) if ".sinks" in key and last else t) for key, t in state_dict.items()}
        real_save_file(live, path, **kwargs)

    monkeypatch.setattr(reset_sinks_mod, "save_safetensors_file", corrupting_save_file)

    with pytest.raises(RuntimeError, match="failed verification"):
        reset_sinks(str(source), in_place=True)

    after = {p.name: p.read_bytes() for p in source.iterdir() if p.is_file() and not p.name.startswith(".")}
    assert after == before, "a failed verification must leave the in-place source untouched"
    assert not _staged_leftovers(source), "the unverified staged shards must be cleaned up"


def test_a_failed_publish_names_the_verified_shards_still_to_place(tmp_path, monkeypatch):
    """Once one shard is renamed, the staged ones are the only reset copies of the rest."""
    source = tmp_path / "src"
    sinks = _build_source(source, sharded=True)
    assert len(_sink_shards(source, sinks)) > 1, "premise: more than one shard is published"
    real_replace, moved = os.replace, []

    def failing_replace(src, dst):
        if moved:
            raise OSError("Read-only file system")
        moved.append(str(dst))
        return real_replace(src, dst)

    monkeypatch.setattr(os, "replace", failing_replace)
    with pytest.raises(RuntimeError, match="rename each over its name by hand"):
        reset_sinks(str(source), in_place=True)
    monkeypatch.undo()

    staged = _staged_leftovers(source)
    assert staged, "the verified staged shards must survive a failed publish"
    for name in staged:
        for key, tensor in load_file(str(source / name)).items():
            if key in sinks:
                assert torch.all(tensor == torch.finfo(tensor.dtype).min), f"{key} was not reset in {name}"


def test_a_dry_run_neither_writes_nor_sweeps(tmp_path):
    """Sweeping would delete files for a run that writes nothing."""
    source, out = tmp_path / "src", tmp_path / "out"
    sinks = _build_source(source, sharded=False)
    _plant_previous_sharded_run(out)
    planted = set(os.listdir(out))

    assert reset_sinks(str(source), str(out), dry_run=True) == len(sinks)

    assert set(os.listdir(out)) == planted, "a dry run wrote to (or swept) the output directory"
    unchanged = load_file(os.path.join(source, SINGLE))
    for name, tensor in sinks.items():
        assert torch.equal(unchanged[name], tensor), f"a dry run rewrote {name}"


@pytest.mark.parametrize("sharded", [False, True], ids=["single_file", "sharded"])
def test_a_dry_run_needs_no_output_dir(tmp_path, sharded):
    """The destination requirement exists to stop a writing run defaulting onto its own input."""
    source = tmp_path / "src"
    sinks = _build_source(source, sharded=sharded)

    assert reset_sinks(str(source), dry_run=True) == len(sinks)

    unchanged = _sinks_across_shards(source)
    for name, tensor in sinks.items():
        assert torch.equal(unchanged[name], tensor), f"a dry run rewrote {name}"


def test_a_writing_run_still_requires_an_output_dir(tmp_path):
    source = tmp_path / "src"
    _build_source(source, sharded=False)

    with pytest.raises(ValueError, match="--output_dir is required"):
        reset_sinks(str(source))


@pytest.mark.parametrize("sharded", [False, True], ids=["single_file", "sharded"])
def test_the_reset_preflights_its_resident_shard(tmp_path, monkeypatch, caplog, sharded):
    """Silence means the tool stopped calling the helper."""
    monkeypatch.setattr(offline_export, "available_host_ram_bytes", lambda: 1)
    source, out = tmp_path / "src", tmp_path / "out"
    sinks = _build_source(source, sharded=sharded)

    assert reset_sinks(str(source), str(out)) == len(sinks)

    warned = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any(m.startswith("reset_sinks: estimated peak host RAM") for m in warned), warned
    _assert_sinks_reset(out, sinks)


def test_a_sink_family_whose_keys_miss_the_spelling_raises(tmp_path):
    """Otherwise it copies through, reported as reset, with its sinks untouched."""
    source, out = tmp_path / "src", tmp_path / "out"
    _build_source(source, sharded=False)
    renamed = {key.replace(".sinks", ".sink_bias"): tensor for key, tensor in load_file(str(source / SINGLE)).items()}
    save_file(renamed, str(source / SINGLE), metadata=SAFETENSORS_METADATA)

    with pytest.raises(RuntimeError, match="none of its tensors is named like a sink"):
        reset_sinks(str(source), str(out))

    assert not out.exists(), "a checkpoint whose sinks were never reset was written anyway"


def _build_sink_free_source(source_dir) -> None:
    _build_source(source_dir, sharded=False)
    stripped = {key: tensor for key, tensor in load_file(str(source_dir / SINGLE)).items() if ".sinks" not in key}
    save_file(stripped, str(source_dir / SINGLE), metadata=SAFETENSORS_METADATA)
    config = json.loads((source_dir / "config.json").read_text())
    config["model_type"] = "qwen3"
    (source_dir / "config.json").write_text(json.dumps(config))


def test_a_sink_free_family_is_copied_through(tmp_path):
    source, out = tmp_path / "src", tmp_path / "out"
    _build_sink_free_source(source)

    assert reset_sinks(str(source), str(out)) == 0

    assert (out / SINGLE).read_bytes() == (source / SINGLE).read_bytes()


def test_the_copy_through_leaves_the_vendor_weight_dumps_and_the_run_internal_entries_behind(tmp_path):
    """gpt-oss ships ``original/`` and ``metal/``; an ``output_dir`` root also holds the run's ``_*`` entries."""
    source, out = tmp_path / "src", tmp_path / "out"
    _build_sink_free_source(source)
    for dump in ("original", "metal"):
        (source / dump).mkdir()
        (source / dump / SINGLE).write_bytes(b"vendor-dump")
    (source / RUN_PLAN_FILE).write_text("{}")
    (source / REFERENCE_CACHE_DIR_NAME).mkdir()
    (source / REFERENCE_CACHE_DIR_NAME / "merged.values").write_bytes(b"scores")
    (source / "2_Dense").mkdir()
    (source / "2_Dense" / "config.json").write_text("{}")

    assert reset_sinks(str(source), str(out)) == 0

    left_behind = ("original", "metal", RUN_PLAN_FILE, REFERENCE_CACHE_DIR_NAME)
    assert not any((out / name).exists() for name in left_behind), sorted(os.listdir(out))
    assert (out / "2_Dense" / "config.json").read_text() == "{}"
    assert (out / SINGLE).read_bytes() == (source / SINGLE).read_bytes()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
