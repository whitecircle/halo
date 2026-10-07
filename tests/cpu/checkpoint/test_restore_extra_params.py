"""CPU tests for the checkpoint loader's targeted extra-param read.

``read_specific_keys_from_checkpoint`` backs ``CheckpointLoader._restore_extra_trained_params``,
which restores wrapper-level trained params (e.g. a learned adapter table) on the EP/CP skip-reload
path. It must read ONLY the requested keys (never materialize the full frozen-base checkpoint) and
handle single-file, sharded-index, and legacy ``pytorch_model.bin`` layouts.

The restore itself is all-or-nothing: these params are the run's entire trained state, so an
unreadable checkpoint raises instead of resuming them at initialization.
"""

import json
import logging
import os

import pytest
import torch
import torch.nn as nn
from safetensors.torch import save_file

import src.distributed.checkpoint.loader as loader_module
from src.checkpoint.format import read_checkpoint_key_set, read_specific_keys_from_checkpoint
from src.distributed.checkpoint.context import CheckpointLoadContext
from src.distributed.checkpoint.loader import CheckpointLoader

_LOADER_LOGGER = "src.distributed.checkpoint.loader"


def _full_state():
    return {
        "adapter_table": torch.randn(2, 4, 8),
        "temperature": torch.tensor(0.1),
        "model.layers.0.weight": torch.randn(8, 8),
        "model.layers.1.weight": torch.randn(8, 8),
    }


def test_reads_only_requested_keys_single_file(tmp_path):
    state = _full_state()
    save_file(state, os.path.join(tmp_path, "model.safetensors"))

    got = read_specific_keys_from_checkpoint(str(tmp_path), ("adapter_table", "temperature"))

    assert set(got) == {"adapter_table", "temperature"}
    assert torch.equal(got["adapter_table"], state["adapter_table"])
    assert torch.equal(got["temperature"], state["temperature"])
    # The large base weights must NOT be loaded.
    assert "model.layers.0.weight" not in got


def test_reads_from_sharded_index(tmp_path):
    state = _full_state()
    # Extras in shard 1, base weights in shard 2 — the reader must follow the weight_map.
    save_file(
        {"adapter_table": state["adapter_table"], "temperature": state["temperature"]},
        os.path.join(tmp_path, "model-00001-of-00002.safetensors"),
    )
    save_file(
        {
            "model.layers.0.weight": state["model.layers.0.weight"],
            "model.layers.1.weight": state["model.layers.1.weight"],
        },
        os.path.join(tmp_path, "model-00002-of-00002.safetensors"),
    )
    weight_map = {
        "adapter_table": "model-00001-of-00002.safetensors",
        "temperature": "model-00001-of-00002.safetensors",
        "model.layers.0.weight": "model-00002-of-00002.safetensors",
        "model.layers.1.weight": "model-00002-of-00002.safetensors",
    }
    with open(os.path.join(tmp_path, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": {}, "weight_map": weight_map}, f)

    got = read_specific_keys_from_checkpoint(str(tmp_path), ("adapter_table", "temperature"))
    assert set(got) == {"adapter_table", "temperature"}
    assert torch.equal(got["adapter_table"], state["adapter_table"])


def test_missing_key_omitted(tmp_path):
    save_file({"adapter_table": torch.randn(2, 4, 8)}, os.path.join(tmp_path, "model.safetensors"))
    got = read_specific_keys_from_checkpoint(str(tmp_path), ("adapter_table", "temperature"))
    assert set(got) == {"adapter_table"}  # temperature absent → silently omitted (caller warns)


def test_no_checkpoint_returns_empty(tmp_path):
    got = read_specific_keys_from_checkpoint(str(tmp_path), ("adapter_table",))
    assert got == {}


def test_reads_from_pytorch_bin(tmp_path):
    state = _full_state()
    torch.save(state, os.path.join(tmp_path, "pytorch_model.bin"))
    got = read_specific_keys_from_checkpoint(str(tmp_path), ("adapter_table", "temperature"))
    assert set(got) == {"adapter_table", "temperature"}
    assert torch.equal(got["adapter_table"], state["adapter_table"])


def _write_layout(directory, layout, state):
    """Write ``state`` into ``directory`` under one of the three checkpoint layouts."""
    if layout == "index":
        save_file(state, os.path.join(directory, "model-00001-of-00001.safetensors"))
        with open(os.path.join(directory, "model.safetensors.index.json"), "w") as f:
            json.dump({"metadata": {}, "weight_map": dict.fromkeys(state, "model-00001-of-00001.safetensors")}, f)
    elif layout == "single":
        save_file(state, os.path.join(directory, "model.safetensors"))
    else:
        torch.save(state, os.path.join(directory, "pytorch_model.bin"))


@pytest.mark.parametrize("layout", ["index", "single", "bin"])
def test_both_readers_resolve_the_same_layout(tmp_path, layout):
    """The key-set probe and the targeted read must resolve a checkpoint's tensors through one
    cascade: a resume decides coverage from the first (``resume_numel_coverage``) and then restores
    through the second, so a layout either reader alone understands turns a real resume into a
    silent partial one."""
    state = _full_state()
    _write_layout(tmp_path, layout, state)

    assert read_checkpoint_key_set(str(tmp_path)) == set(state)
    got = read_specific_keys_from_checkpoint(str(tmp_path), ("adapter_table", "temperature"))
    assert set(got) == {"adapter_table", "temperature"}
    assert torch.equal(got["adapter_table"], state["adapter_table"])


def test_both_readers_agree_on_layout_precedence(tmp_path):
    """A directory carrying several layouts (an index written beside a stale ``pytorch_model.bin``)
    resolves to exactly one of them, identically for both readers — disagreeing here reads one
    checkpoint's key set and another's tensors."""
    indexed = _full_state()
    stale = {"stale_only": torch.randn(2, 2)}
    _write_layout(tmp_path, "index", indexed)
    _write_layout(tmp_path, "bin", stale)

    keys = read_checkpoint_key_set(str(tmp_path))
    assert keys == set(indexed)
    assert set(read_specific_keys_from_checkpoint(str(tmp_path), keys | set(stale))) == set(indexed)


class _ExtraParamWrapper(nn.Module):
    """A wrapper whose declared extras ARE its trained parameters."""

    def __init__(self):
        super().__init__()
        self.adapter_table = nn.Parameter(torch.zeros(2, 4, 8))
        self.temperature = nn.Parameter(torch.tensor([1.0]))
        self._extra_checkpoint_param_names = ("adapter_table", "temperature")


def _loader(model):
    return CheckpointLoader(
        CheckpointLoadContext(
            model=model,
            optimizer=None,
            lr_scheduler=None,
            parallelism_config=None,
            is_pp_mode=False,
            is_cp_mode=False,
            is_tp_mode=False,
            has_ep_layers=False,
            fsdp_wrapped=False,
            tp_rank=0,
            tp_size=1,
            super_load_from_checkpoint=lambda *a, **k: None,
            super_load_optimizer_and_scheduler=lambda *a, **k: None,
        )
    )


def test_an_unreadable_extra_param_checkpoint_raises(tmp_path):
    """A torn sidecar that warns and continues leaves these params at initialization — and when
    they are the run's ONLY trained state, the run restarts from scratch under a resumed step
    count, LR schedule and dataloader position. Nothing distinguishes that from a working resume
    in the metrics, so it must raise."""
    save_file({"adapter_table": torch.randn(2, 4, 8)}, os.path.join(tmp_path, "model.safetensors"))
    with open(os.path.join(tmp_path, "model.safetensors"), "wb") as fh:
        fh.write(b"truncated, not a safetensors file")
    model = _ExtraParamWrapper()
    before = model.adapter_table.detach().clone()

    with pytest.raises(RuntimeError, match="Resume from a complete checkpoint"):
        _loader(model)._restore_extra_trained_params(str(tmp_path), model)

    assert torch.equal(model.adapter_table.data, before), "a refused restore must not half-apply"


def test_a_readable_extra_param_checkpoint_still_restores(tmp_path, caplog):
    """Anti-over-rejection: the ordinary path must still restore, and an ABSENT key must stay a
    warning — a checkpoint predating a newly declared extra is a legitimate resume, so the raise
    above must fire on an unreadable checkpoint only, never on a merely incomplete one.

    ``temperature`` is declared but absent here, which is exactly that case: the restore
    proceeds, the present key lands, the absent one keeps its initialization, and the run is told."""
    saved = torch.randn(2, 4, 8)
    save_file({"adapter_table": saved}, os.path.join(tmp_path, "model.safetensors"))
    model = _ExtraParamWrapper()
    initialized = model.temperature.detach().clone()

    with caplog.at_level(logging.WARNING, logger=_LOADER_LOGGER):
        _loader(model)._restore_extra_trained_params(str(tmp_path), model)

    assert torch.equal(model.adapter_table.data, saved)
    assert torch.equal(model.temperature.data, initialized), "an absent key must be left alone"
    warnings = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert len(warnings) == 1, warnings
    assert "temperature" in warnings[0], warnings[0]
    assert "adapter_table" not in warnings[0].split("(Restored:")[0], (
        f"the key that WAS restored is named as missing: {warnings[0]}"
    )


@pytest.mark.parametrize("multi_rank", [False, True])
def test_the_restore_branches_on_the_shared_multi_rank_probe(tmp_path, monkeypatch, multi_rank):
    """Which write path the restore takes is decided by ``is_multi_rank_run``, not a local re-spelling.

    Above one rank the params land through the collective every rank must enter; alone they are copied
    straight in, because the collective would raise with no group to reach. Both sides read the same
    probe the rest of the toolkit gates its collectives on, so a world it calls single-rank can never
    be the world this path opens a broadcast for — the deadlock a second spelling drifting from the
    first would produce.
    """
    saved = torch.randn(2, 4, 8)
    save_file({"adapter_table": saved}, os.path.join(tmp_path, "model.safetensors"))
    model = _ExtraParamWrapper()
    initialized = model.adapter_table.detach().clone()

    collective_writes = []
    monkeypatch.setattr(loader_module, "is_multi_rank_run", lambda: multi_rank)
    monkeypatch.setattr(
        loader_module,
        "set_model_state_dict",
        lambda module, state_dict, options: collective_writes.append((module, sorted(state_dict), options)),
    )

    _loader(model)._restore_extra_trained_params(str(tmp_path), model)

    if multi_rank:
        assert [(module, keys) for module, keys, _ in collective_writes] == [(model, ["adapter_table"])]
        assert collective_writes[0][2].broadcast_from_rank0, "peers hold no tensors; rank 0's must be sent"
        assert torch.equal(model.adapter_table.data, initialized), "the collective owns the write, not a local copy"
    else:
        assert not collective_writes, "a lone rank has no peer to broadcast to"
        assert torch.equal(model.adapter_table.data, saved)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
