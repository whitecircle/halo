#!/usr/bin/env python
"""TRL's DPO and KTO constructors never hash the reference under the toolkit's trainers.

With ``precompute_ref_log_probs`` on, TRL's ``__init__`` runs ``hash_module(self.ref_model or
self.model)``: a CPU copy of every tensor on every rank, and ``.numpy()`` on a TP-sharded DTensor,
which raises. The hash keys only the cache of TRL's own sweep, which the DP-axis sweep replaces, so
the toolkit trainers run TRL's construction with the name rebound to a no-op and restore it after.

    python tests/cpu/trainers/test_precompute_reference_hash.py
"""

import sys

import pytest
import torch
from trl import DPOTrainer, KTOTrainer
from trl.trainer.utils import hash_module

from src.trainers.mixins.base import DistributedTrainerMixin
from src.trainers.preference.dpo import DistributedDPOTrainer
from src.trainers.preference.kto import DistributedKTOTrainer

TRAINERS = [(DistributedDPOTrainer, DPOTrainer), (DistributedKTOTrainer, KTOTrainer)]


class _Unhashable(torch.nn.Module):
    def state_dict(self, *args, **kwargs):
        raise AssertionError("TRL hashed the reference")


class _Constructed(Exception):
    pass


@pytest.fixture
def trl_construction(monkeypatch):
    """Stand in for TRL's ``__init__`` with its hash line, run by the toolkit trainer's real ``__init__``."""

    def install(trainer_cls, trl_cls) -> list:
        trl_module = sys.modules[trl_cls.__module__]
        hashes = []

        def trl_init(self, *args, **kwargs):
            hashes.append(trl_module.hash_module(_Unhashable()))
            raise _Constructed

        monkeypatch.setattr(trl_cls, "__init__", trl_init)
        monkeypatch.setattr(DistributedTrainerMixin, "_init_distributed_config", lambda self, kwargs, **_: kwargs)
        monkeypatch.setattr(trainer_cls, "_validate_reference_model", lambda self, ref_model: None)
        return hashes

    return install


@pytest.mark.parametrize(("trainer_cls", "trl_cls"), TRAINERS, ids=lambda v: v.__name__)
def test_trls_construction_never_hashes_the_reference(trl_construction, trainer_cls, trl_cls):
    hashes = trl_construction(trainer_cls, trl_cls)

    with pytest.raises(_Constructed):
        trainer_cls(model=None, args=None)

    assert hashes == [None]
    assert sys.modules[trl_cls.__module__].hash_module is hash_module, "TRL's hash_module was not restored"


@pytest.mark.parametrize(("trainer_cls", "trl_cls"), TRAINERS, ids=lambda v: v.__name__)
def test_a_trl_that_no_longer_binds_hash_module_is_refused(trl_construction, monkeypatch, trainer_cls, trl_cls):
    trl_construction(trainer_cls, trl_cls)
    monkeypatch.setattr(sys.modules[trl_cls.__module__], "hash_module", lambda module: "renamed")

    with pytest.raises(RuntimeError, match="hash_module"):
        trainer_cls(model=None, args=None)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
