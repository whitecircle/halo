#!/usr/bin/env python
"""CPU tests for the rank-consistent reference-log-prob precompute cache.

TRL's ``precompute_ref_log_probs`` keys its datasets disk cache on
``Hasher.hash((dataset._fingerprint, hash_module(model)))``. Both inputs diverge across ranks under
distributed training (EP/TP shard the model → per-rank ``hash_module``; datasets' tokenize map emits
a random per-process ``_fingerprint``). TRL writes the cache on the main process only, so with a
divergent key the other ranks block reading a file that was never written and the next collective
expert-parallel forward deadlocks.

``PrecomputeRefLogpsRankConsistentMixin`` pins the cache key to rank 0's on every rank: it must
broadcast both inputs and restore the patched ``hash_module``.

Run: python tests/cpu/trainers/test_precompute_rank_consistent.py
"""

from types import SimpleNamespace

import pytest

from src.trainers.mixins.dataloader import DataParallelDataLoaderMixin
from src.trainers.preference.precompute import (
    PrecomputeRefLogpsRankConsistentMixin,
    _defining_module,
    _rank0_authoritative_module_hash,
)

# Module-global: the mixin patches the module DEFINING _precompute_ref_logps, i.e. this one.
_RANK_HASH = "H1"  # a non-main rank's raw (divergent) parameter hash


def hash_module(model):
    return _RANK_HASH


class _FakeBase:
    """Stands in for TRL's ``DPOTrainer``/``KTOTrainer``: records the cache key it would build."""

    def _precompute_ref_logps(self, dataset, name, batch_size):
        return (dataset._fingerprint, hash_module(self.model))


class _FakeTrainer(PrecomputeRefLogpsRankConsistentMixin, DataParallelDataLoaderMixin, _FakeBase):
    """Same composition the real trainers use — the mixin also pins the sweep to the DP axis."""

    def __init__(self):
        self.model = object()
        self._dataset_presharded = False
        self.accelerator = SimpleNamespace(prepare_data_loader=lambda loader: loader, gather=lambda tensor: tensor)

    def _required_ref_logps_columns(self) -> tuple[str, ...]:
        return ("ref_logps",)

    def get_data_parallel_rank(self) -> int:
        return 0


class _FakeDataset:
    column_names = ["prompt"]  # no reference columns, so the sweep is not short-circuited

    def __init__(self, fingerprint):
        self._fingerprint = fingerprint


def _install_rank0_broadcast(monkeypatch):
    """Simulate the collective: every rank receives rank 0's authoritative value ``"R0"``."""
    import src.trainers.preference.precompute as precompute_mod

    monkeypatch.setattr(precompute_mod, "broadcast_from_rank0", lambda _value: "R0")


def test_control_base_observes_divergent_key():
    """Without the mixin, the base sees this rank's own (divergent) fingerprint + hash."""
    key = _FakeBase._precompute_ref_logps(_FakeTrainer(), _FakeDataset("FP1"), "train", 2)
    assert key == ("FP1", "H1")


def test_mixin_collapses_both_key_inputs_to_rank0(monkeypatch):
    """The mixin must broadcast BOTH the dataset fingerprint AND the model hash to rank 0's value,
    so the cache path is identical on every rank."""
    _install_rank0_broadcast(monkeypatch)
    key = _FakeTrainer()._precompute_ref_logps(_FakeDataset("FP1"), "train", 2)
    # Dropping either broadcast leaves this rank's "FP1"/"H1" — the regression that deadlocks EP/TP.
    assert key == ("R0", "R0")


def test_mixin_pins_dataset_fingerprint_in_place(monkeypatch):
    """The dataset object's fingerprint is rewritten to rank 0's (TRL reads it again downstream)."""
    _install_rank0_broadcast(monkeypatch)
    ds = _FakeDataset("FP1")
    _FakeTrainer()._precompute_ref_logps(ds, "train", 2)
    assert ds._fingerprint == "R0"


def test_hash_module_restored_after_precompute(monkeypatch):
    """The temporary ``hash_module`` patch must be reverted, even across repeated calls."""
    _install_rank0_broadcast(monkeypatch)
    original = hash_module
    _FakeTrainer()._precompute_ref_logps(_FakeDataset("FP1"), "train", 2)
    import tests.cpu.trainers.test_precompute_rank_consistent as this_mod

    assert this_mod.hash_module is original


def test_context_manager_no_hash_module_symbol_is_noop():
    """A module without a ``hash_module`` symbol must not raise (pure-DP TRL versions)."""
    import types

    empty = types.ModuleType("empty")
    with _rank0_authoritative_module_hash(empty):
        pass
    assert not hasattr(empty, "hash_module")


def test_defining_module_resolves_concrete_trainer():
    """``_defining_module`` returns the module that DEFINES ``_precompute_ref_logps`` (the fake base
    here), never the mixin — that is where ``hash_module`` must be patched."""
    module = _defining_module(_FakeTrainer())
    assert module is not None
    assert module.__name__ == __name__  # _FakeBase is defined in this test module
    assert hasattr(module, "hash_module")


def test_distributed_trainers_use_the_mixin():
    """DPO and KTO must route ``_precompute_ref_logps`` through the mixin (the wiring under test).

    DPO layers its own dataset-supplied-columns skip ABOVE the mixin, so identity of the bound
    method is not the invariant — the invariant is (1) the mixin sits between any trainer-level
    override and the concrete TRL trainer in the MRO, so every ``super()`` fall-through passes
    through the rank-consistent wrapper, and (2) ``_defining_module`` still resolves to the TRL
    module that owns ``hash_module`` (a subclass override must never capture that lookup, or the
    hash patch silently no-ops and the EP/TP deadlock returns).
    """
    from src.trainers.preference.dpo import DistributedDPOTrainer
    from src.trainers.preference.kto import DistributedKTOTrainer

    for trainer_cls in (DistributedDPOTrainer, DistributedKTOTrainer):
        assert issubclass(trainer_cls, PrecomputeRefLogpsRankConsistentMixin)
        mro = list(trainer_cls.__mro__)
        concrete = [
            klass
            for klass in mro
            if "_precompute_ref_logps" in klass.__dict__
            and not issubclass(klass, PrecomputeRefLogpsRankConsistentMixin)
        ]
        assert concrete, f"{trainer_cls.__name__} has no concrete TRL _precompute_ref_logps below the mixin"
        assert mro.index(PrecomputeRefLogpsRankConsistentMixin) < mro.index(concrete[0])
        module = _defining_module(trainer_cls.__new__(trainer_cls))
        assert module is not None and hasattr(module, "hash_module")


def test_dpo_skips_sweep_when_ref_columns_present(monkeypatch):
    """Dataset-supplied ref columns must short-circuit the sweep (dataset returned unchanged, no
    broadcast) — the seam PP relies on; a dataset missing a column must fall through INTO the
    rank-consistent mixin (its first act is the fingerprint broadcast)."""
    from accelerate import PartialState

    import src.trainers.preference.precompute as precompute_mod
    from src.trainers.preference.dpo import DistributedDPOTrainer

    PartialState()  # the skip path logs through accelerate's logger, which needs the state

    class _Sentinel(Exception):
        pass

    broadcast_calls = []

    def _record_and_stop(value):
        broadcast_calls.append(value)
        raise _Sentinel  # stop before the mixin chains into TRL's real sweep

    monkeypatch.setattr(precompute_mod, "broadcast_from_rank0", _record_and_stop)
    trainer = DistributedDPOTrainer.__new__(DistributedDPOTrainer)
    trainer._dataset_presharded = False  # set by _init_distributed_config before TRL runs the sweep

    class _ColumnsDataset:
        column_names = ["prompt_ids", "chosen_ids", "rejected_ids", "ref_chosen_logps", "ref_rejected_logps"]

    dataset = _ColumnsDataset()
    assert trainer._precompute_ref_logps(dataset, "train", 2) is dataset
    assert broadcast_calls == []  # skip path: the mixin (and thus the sweep) was never entered

    class _MissingColumnDataset:
        column_names = ["prompt_ids", "chosen_ids", "rejected_ids", "ref_chosen_logps"]
        _fingerprint = "FP1"

    with pytest.raises(_Sentinel):
        trainer._precompute_ref_logps(_MissingColumnDataset(), "train", 2)
    assert broadcast_calls == ["FP1"]  # fall-through entered the mixin, not a silent skip


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
