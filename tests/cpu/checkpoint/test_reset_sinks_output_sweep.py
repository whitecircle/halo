#!/usr/bin/env python
"""``reset_sinks`` must hand back a directory that holds ONLY the checkpoint it just wrote.

The tool has two write paths — a direct safetensors rewrite for a single-file checkpoint and a
``from_pretrained`` round-trip for a sharded one — and both land in ``--output_dir``. When that
directory already holds a previous run's weight files, the leftovers outlive the save: the
safetensors path writes one file and deletes nothing, and ``save_pretrained`` deletes only the shards
its own ``-00001-of-00002`` numbering regex matches, so a stale index survives it. Either way
``from_pretrained`` and the toolkit's index-first readers can resolve the OLD weights — or shards
that no longer exist — from a directory the tool reported as successfully reset.

The sweep therefore sits at the dispatcher, after whichever branch ran, and only when the output
directory is not the input: rewriting a checkpoint where it stands is this tool's documented default,
and there the directory's other weight files are not the tool's to delete.

The other way this tool hands back a directory it should not: the ``from_pretrained`` branch applies
the trainers' own sink policy, so a layout that walk does not recognize must RAISE there instead of
saving a checkpoint whose sinks are still live under a name that says they are off.

    python tests/cpu/checkpoint/test_reset_sinks_output_sweep.py
"""

import json
import os
import time

import pytest
import torch
from accelerate import PartialState
from safetensors.torch import load_file, save_file
from transformers import GptOssConfig, GptOssForCausalLM

PartialState()  # the script's model loading logs through accelerate's logger

import scripts.after_training.reset_sinks as reset_sinks_mod
from scripts.after_training.reset_sinks import reset_sinks
from src.checkpoint.tool_io import STAGING_SUFFIX, checkpoint_shard_files
from tests.common.checkpoint_io import weight_files

INDEX = "model.safetensors.index.json"
SINGLE = "model.safetensors"
# The round-trip assert reads these, so a fixture change cannot silently turn it vacuous.
VOCAB_SIZE, HIDDEN_SIZE = 128, 64


def _tiny_gptoss():
    """A real sink-carrying model: gpt-oss is the family this tool exists for."""
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
    """Write a source checkpoint; return its sink tensors so the reset can be shown to have happened."""
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
    """A COMPLETE previous sharded save in the output directory: index plus every shard it names.

    Complete on purpose. Both that layout and the single file this run writes describe a whole
    checkpoint, so only the modification time separates them — back-dated here to say which one is
    the leftover. Returns the directory for convenience.
    """
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


def _staging_dir(target):
    """Where a staged publish builds its output — the shared suffix, so a rename cannot silently
    leave this test asserting on a path nothing writes."""
    return target.parent / f"{target.name}{STAGING_SUFFIX}"


def _weight_files(directory) -> set[str]:
    return set(weight_files(str(directory), include_index=True))  # the sweep owns the index too


def _assert_sinks_reset(directory, source_sinks: dict[str, torch.Tensor]) -> None:
    written = load_file(os.path.join(directory, SINGLE))
    assert set(source_sinks) <= set(written), "the reset checkpoint dropped sink tensors"
    for name in source_sinks:
        tensor = written[name]
        assert torch.all(tensor == torch.finfo(tensor.dtype).min), f"{name} was not reset to dtype min"
    # Not a stub and not the planted leftovers: a real weight survived the round trip.
    assert written["model.embed_tokens.weight"].shape == (VOCAB_SIZE, HIDDEN_SIZE)


def test_the_single_file_branch_sweeps_the_previous_runs_leftovers(tmp_path):
    """The branch that clears nothing on its own: it copies the source directory over and writes one
    ``model.safetensors``, so a complete previous sharded save survives it untouched — index, shards
    and all — and would win the lookup over the file just written."""
    source, out = tmp_path / "src", tmp_path / "out"
    sinks = _build_source(source, sharded=False)
    _plant_previous_sharded_run(out)

    assert reset_sinks(str(source), str(out)) == len(sinks)

    assert _weight_files(out) == {SINGLE}, "the previous sharded run outlived the reset"
    _assert_sinks_reset(out, sinks)


def test_the_sharded_branch_sweeps_the_previous_runs_index(tmp_path):
    """The ``from_pretrained`` branch: ``save_pretrained`` removes the numbered shards it did not
    write but leaves the index naming them, so index-first readers resolve files that are gone."""
    source, out = tmp_path / "src", tmp_path / "out"
    sinks = _build_source(source, sharded=True)
    _plant_previous_sharded_run(out)

    assert reset_sinks(str(source), str(out)) == len(sinks)

    assert _weight_files(out) == {SINGLE}, "a previous run's index/shards survived the reset"
    _assert_sinks_reset(out, sinks)


def test_an_in_place_run_sweeps_nothing(tmp_path):
    """In place is this tool's default, and there the directory is not its to prune: the other weight
    files belong to whoever put them there, and the reset still has to happen."""
    source = tmp_path / "src"
    sinks = _build_source(source, sharded=False)
    _plant_previous_sharded_run(source)

    assert reset_sinks(str(source), in_place=True) == len(sinks)

    assert _weight_files(source) == {
        SINGLE,
        INDEX,
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
    }
    _assert_sinks_reset(source, sinks)


def _sinks_across_shards(directory) -> dict[str, torch.Tensor]:
    values: dict[str, torch.Tensor] = {}
    for shard in checkpoint_shard_files(str(directory)):
        for key, tensor in load_file(shard).items():
            if ".sinks" in key:
                values[key] = tensor
    return values


def test_an_in_place_sharded_reset_is_staged_verified_and_swapped(tmp_path, monkeypatch):
    """The sharded (from_pretrained) branch's default is in place over the only copy, so WHERE the
    reset is written before it lands is the contract, not just the end state: a save straight into
    the target reaches the same final bytes and loses the checkpoint on any failure along the way.
    Recorded at the swap seam — the staged copy must already be reset while the target still holds
    the original sinks."""
    source = tmp_path / "src"
    sinks = _build_source(source, sharded=True)
    real_swap = reset_sinks_mod._swap_staged_checkpoint
    observed: dict = {}

    def recording_swap(staging_dir, output_dir):
        observed["staging"] = str(staging_dir)
        observed["staged_sinks"] = _sinks_across_shards(staging_dir)
        observed["target_sinks"] = _sinks_across_shards(output_dir)
        return real_swap(staging_dir, output_dir)

    monkeypatch.setattr(reset_sinks_mod, "_swap_staged_checkpoint", recording_swap)

    assert reset_sinks(str(source), in_place=True) == len(sinks)

    assert observed.get("staging") == str(_staging_dir(source)), "the reset was not written to a staging copy"
    assert set(observed["staged_sinks"]) == set(sinks), "the staged copy dropped sink tensors"
    for name, tensor in observed["staged_sinks"].items():
        assert torch.all(tensor == torch.finfo(tensor.dtype).min), f"{name} was not reset in the staged copy"
    assert all(
        torch.equal(tensor, sinks[name].to(tensor.dtype)) for name, tensor in observed["target_sinks"].items()
    ), "the target was already rewritten before the swap — the staging copy is not what lands"

    written = _sinks_across_shards(source)
    assert set(written) == set(sinks), "the reset checkpoint dropped sink tensors"
    for name, tensor in written.items():
        assert torch.all(tensor == torch.finfo(tensor.dtype).min), f"{name} was not reset to dtype min"
    assert not _staging_dir(source).exists(), "the staging directory must be cleaned up"


def test_a_write_that_kept_live_sinks_never_replaces_the_source(tmp_path, monkeypatch):
    """A torn or buggy write must not ship live sinks as a reported success, over the only copy.
    The staged copy is verified sinks-at-dtype-min BEFORE the swap, and a failure raises with the
    source byte-identical."""
    source = tmp_path / "src"
    _build_source(source, sharded=True)
    before = {p.name: p.read_bytes() for p in source.iterdir() if p.is_file()}
    real_save = reset_sinks_mod.save_full_checkpoint

    def corrupting_save(model, out_dir, **kwargs):
        real_save(model, out_dir, **kwargs)
        # A torn writer: flip every written sink back to a live value after the save reports done.
        for shard in checkpoint_shard_files(out_dir):
            tensors = load_file(shard)
            sink_keys = [key for key in tensors if ".sinks" in key]
            if sink_keys:
                for key in sink_keys:
                    tensors[key] = torch.zeros_like(tensors[key])
                save_file(tensors, shard)

    monkeypatch.setattr(reset_sinks_mod, "save_full_checkpoint", corrupting_save)

    with pytest.raises(RuntimeError, match="failed verification"):
        reset_sinks(str(source), in_place=True)

    after = {p.name: p.read_bytes() for p in source.iterdir() if p.is_file()}
    assert after == before, "a failed verification must leave the in-place source untouched"
    assert not _staging_dir(tmp_path / "src").exists(), "the staging directory must be cleaned up"


def test_single_file_write_that_kept_live_sinks_never_replaces_the_source(tmp_path, monkeypatch):
    """The same guarantee on the single-file branch: the staged temp file is verified BEFORE the rename.

    Verifying after it would, under ``--in_place``, put a checkpoint with live sinks over the only
    copy and then raise — an unusable artifact where the source had been fine.
    """
    source = tmp_path / "src"
    _build_source(source, sharded=False)
    before = {p.name: p.read_bytes() for p in source.iterdir() if p.is_file()}
    real_save_file = reset_sinks_mod.save_file

    def corrupting_save_file(state_dict, path, **kwargs):
        # A torn writer: the staged file lands with live sinks despite the in-memory reset.
        live = {key: (torch.zeros_like(t) if ".sinks" in key else t) for key, t in state_dict.items()}
        real_save_file(live, path, **kwargs)

    monkeypatch.setattr(reset_sinks_mod, "save_file", corrupting_save_file)

    with pytest.raises(RuntimeError, match="failed verification"):
        reset_sinks(str(source), in_place=True)

    after = {p.name: p.read_bytes() for p in source.iterdir() if p.is_file()}
    assert after == before, "a failed verification must leave the in-place source untouched"
    assert not list(source.glob("*.tmp")), "the unverified staging file must be cleaned up"


def test_a_failed_swap_keeps_the_staged_checkpoint(tmp_path, monkeypatch):
    """Staging is only worth anything if the staged copy outlives a failed swap: it holds the ONLY
    complete reset checkpoint once the first file has replaced its counterpart, so a cleanup there
    would leave the target half-old/half-new with nothing to recover from."""
    source = tmp_path / "src"
    _build_source(source, sharded=True)
    real_replace = os.replace
    moved: list[str] = []

    def failing_replace(src, dst):
        # Only the swap's own moves (into the target directory) fail, and only after one landed —
        # that is the state where the staging copy is the only complete checkpoint left.
        if os.path.dirname(str(dst)) == str(source):
            if moved:
                raise OSError("Read-only file system")
            moved.append(str(dst))
        return real_replace(src, dst)

    monkeypatch.setattr(reset_sinks_mod.os, "replace", failing_replace)

    with pytest.raises(RuntimeError, match="move it into place by hand"):
        reset_sinks(str(source), in_place=True)

    staging = _staging_dir(tmp_path / "src")
    assert staging.is_dir(), "the only complete copy of the reset checkpoint must survive the failure"
    for name, tensor in _sinks_across_shards(staging).items():
        assert torch.all(tensor == torch.finfo(tensor.dtype).min), f"{name} was not reset in the staged copy"


def test_a_dry_run_neither_writes_nor_sweeps(tmp_path):
    """``--dry_run`` reports what would change. Sweeping there would delete files for a run that
    writes nothing — and it must not touch the source's sinks either."""
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
    """``--dry_run`` is the documented read-only invocation, so the destination requirement — which
    exists to stop a WRITING run from defaulting onto its own input — must not abort it first.

    Both branches, because both dereference the destination: the dispatcher hands each one an
    ``output_dir`` it never writes under a dry run."""
    source = tmp_path / "src"
    sinks = _build_source(source, sharded=sharded)

    assert reset_sinks(str(source), dry_run=True) == len(sinks)

    unchanged = _sinks_across_shards(source)
    for name, tensor in sinks.items():
        assert torch.equal(unchanged[name], tensor), f"a dry run rewrote {name}"


def test_a_writing_run_still_requires_an_output_dir(tmp_path):
    """Anti-over-widening: the refusal still guards every invocation that writes."""
    source = tmp_path / "src"
    _build_source(source, sharded=False)

    with pytest.raises(ValueError, match="--output_dir is required"):
        reset_sinks(str(source))


def test_the_sharded_branch_honours_max_shard_size(tmp_path):
    """The from_pretrained branch re-saves the whole model; without the toolkit's cap it lands in
    transformers' own default shards (50GB in 5.16), which every single-file reader downstream has
    to hold whole. The flag must reach the staged save and survive the swap."""
    source, out = tmp_path / "src", tmp_path / "out"
    sinks = _build_source(source, sharded=True)

    assert reset_sinks(str(source), str(out), max_shard_size="64KB") == len(sinks)

    written = _weight_files(out)
    assert INDEX in written and SINGLE not in written, f"expected a sharded save at max_shard_size=64KB, got {written}"
    assert len(checkpoint_shard_files(str(out))) > 1
    for name, tensor in _sinks_across_shards(out).items():
        assert torch.all(tensor == torch.finfo(tensor.dtype).min), f"{name} was not reset to dtype min"


@pytest.mark.parametrize("sharded", [False, True], ids=["single_file", "sharded"])
def test_both_branches_preflight_the_full_checkpoint_load(tmp_path, monkeypatch, capsys, sharded):
    """Both branches hold the whole checkpoint in host RAM (``load_file``, or from_pretrained onto the
    CPU), so the shared preflight must warn before either loads — silence means the tool stopped
    calling the helper — and must not abort the reset."""
    from src.checkpoint import tool_io

    monkeypatch.setattr(tool_io, "available_host_ram_bytes", lambda: 1)
    source, out = tmp_path / "src", tmp_path / "out"
    sinks = _build_source(source, sharded=sharded)

    assert reset_sinks(str(source), str(out)) == len(sinks)

    printed = capsys.readouterr().out
    assert "WARNING: reset_sinks" in printed and "RAM" in printed, printed
    _assert_sinks_reset(out, sinks)


def test_an_unrecognized_sink_layout_raises_instead_of_saving_live_sinks(tmp_path, monkeypatch):
    """THE regression the shared policy closes: this tool used to fill ``.sinks`` parameters by name,
    so a layout its own walk could not resolve simply reset nothing — and the tool then wrote, swept
    and reported a "reset" checkpoint whose sinks were untouched. Routed through the trainers'
    ``apply_sinks_policy``, a sinks-carrying model the walk finds no attention layers on is a raise."""
    from src.models.patches import gpt_oss_sinks

    source, out = tmp_path / "src", tmp_path / "out"
    _build_source(source, sharded=True)
    # The one thing an unrecognized layout changes: the decoder-layer list cannot be resolved.
    monkeypatch.setattr(gpt_oss_sinks, "backbone_with_layers", lambda _model: None)

    with pytest.raises(RuntimeError, match="touched no attention layers"):
        reset_sinks(str(source), str(out))

    assert not out.exists(), "a checkpoint whose sinks were never reset was written anyway"


def test_a_hub_source_without_sinks_is_refused_loudly(tmp_path):
    """A Hub id that carries no sinks has no local directory for the pass-through copy: the run must
    say so rather than die in ``copytree`` after loading the whole model."""
    checkpoint_dir, out = tmp_path / "org" / "model", tmp_path / "out"
    with pytest.raises(ValueError, match="no attention sinks"):
        reset_sinks_mod._passthrough_copy(checkpoint_dir, out, dry_run=False)
    assert not out.exists()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
