#!/usr/bin/env python
"""Trainer ctor position tables derive from the installed TRL signatures and fail loud when a parameter disappears.
Slots are literals off TRL 1.14.2; recomputing ``params.index(...)`` would test the derivation against itself."""

import pytest

from halo.trainers.construction.ctor_slots import (
    get_ctor_arg,
    get_ctor_config,
    get_ctor_model_and_config,
    resolve_ctor_positions,
)
from halo.trainers.grpo.on_policy import GRPO_CTOR_POSITIONS
from halo.trainers.preference.dpo import DistributedDPOTrainer
from halo.trainers.preference.kto import DistributedKTOTrainer
from halo.trainers.reward.bradley_terry import DistributedRewardTrainer
from halo.trainers.sft import _CTOR_POSITIONS as SFT_CTOR_POSITIONS

DPO_CTOR_POSITIONS = DistributedDPOTrainer.CTOR_POSITIONS
KTO_CTOR_POSITIONS = DistributedKTOTrainer.CTOR_POSITIONS
REWARD_CTOR_POSITIONS = DistributedRewardTrainer.CTOR_POSITIONS


def test_sft_slots_match_installed_trl():
    """TRL 1.14.2: ``SFTTrainer.__init__(self, model, args, data_collator, ...)``."""
    assert SFT_CTOR_POSITIONS == {"model": 0, "args": 1, "data_collator": 2}


def test_dpo_slots_match_installed_trl():
    """TRL 1.14.2: ``DPOTrainer.__init__(self, model, ref_model, args, ...)``."""
    assert DPO_CTOR_POSITIONS == {"model": 0, "args": 2, "ref_model": 1}


def test_kto_slots_match_installed_trl():
    """TRL 1.14.2: ``KTOTrainer.__init__(self, model, ref_model, args, ...)``."""
    assert KTO_CTOR_POSITIONS == {"model": 0, "args": 2, "ref_model": 1}


def test_reward_and_grpo_slots_match_installed_trl():
    """TRL 1.14.2: ``RewardTrainer(model, args, ...)``, ``GRPOTrainer(model, reward_funcs, args, ...)``."""
    assert REWARD_CTOR_POSITIONS == {"model": 0, "args": 1}
    assert GRPO_CTOR_POSITIONS == {"model": 0, "args": 2}


def test_ctor_config_reads_the_slot_of_the_forwarded_base():
    """``RewardTrainer`` takes its config one slot earlier than the GRPO/DPO shape; the table decides."""
    config = object()
    assert get_ctor_config((None, config), {}, REWARD_CTOR_POSITIONS) is config
    assert get_ctor_config((None, None, config), {}, GRPO_CTOR_POSITIONS) is config
    assert get_ctor_config((None, config), {}, GRPO_CTOR_POSITIONS) is None


def test_ctor_config_falls_through_an_explicit_none_keyword():
    """``require_vllm_server_mode`` relies on it: ``args=None`` must not mask a positional config."""
    config = object()
    assert get_ctor_config((None, None, config), {"args": None}, GRPO_CTOR_POSITIONS) is config
    assert get_ctor_config((), {"args": config}, GRPO_CTOR_POSITIONS) is config
    assert get_ctor_config((), {}, GRPO_CTOR_POSITIONS) is None


def test_positionals_without_a_slot_table_raise():
    """Positionals can't resolve without a table, and skipping them would silently drop the model and config setup."""
    model, config = object(), object()
    assert get_ctor_model_and_config((), {"model": model, "args": config}, None) == (model, config)
    with pytest.raises(ValueError, match="without a ctor_positions table"):
        get_ctor_model_and_config((model, config), {}, None)


def test_gate_extracts_the_passed_collator():
    collator = object()
    ctor_args: list = [None] * (SFT_CTOR_POSITIONS["data_collator"] + 1)
    ctor_args[SFT_CTOR_POSITIONS["data_collator"]] = collator
    assert get_ctor_arg(tuple(ctor_args), {}, "data_collator", SFT_CTOR_POSITIONS) is collator
    assert get_ctor_arg((), {"data_collator": collator}, "data_collator", SFT_CTOR_POSITIONS) is collator


def test_gate_extracts_the_passed_reference_model():
    ref = object()
    assert get_ctor_arg((None, ref), {}, "ref_model", DPO_CTOR_POSITIONS) is ref
    assert get_ctor_arg((None, ref), {}, "ref_model", KTO_CTOR_POSITIONS) is ref
    assert get_ctor_arg((), {"ref_model": ref}, "ref_model", DPO_CTOR_POSITIONS) is ref


def test_derivation_follows_a_shifted_signature():
    class _Base:
        def __init__(self, model, args=None, data_collator=None):
            pass

    class _Shifted:
        def __init__(self, model, inserted=None, args=None, data_collator=None):
            pass

    assert resolve_ctor_positions(_Base, "data_collator") == {"data_collator": 2}
    shifted = resolve_ctor_positions(_Shifted, "data_collator")
    assert shifted == {"data_collator": 3}

    collator = object()
    assert get_ctor_arg((None, None, None, collator), {}, "data_collator", shifted) is collator
    # A stale hand table (slot 2) would read the neighbouring argument here.
    assert get_ctor_arg((None, None, collator, None), {}, "data_collator", shifted) is None


def test_a_varargs_signature_declares_no_slot():
    class _Real:
        def __init__(self, model, ref_model=None, args=None):
            pass

    class _Shim(_Real):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

    with pytest.raises(ValueError, match="ref_model"):
        resolve_ctor_positions(_Shim, "ref_model")


def test_missing_parameter_fails_loud():
    class _NoCollator:
        def __init__(self, model, args=None):
            pass

    with pytest.raises(ValueError, match="data_collator"):
        resolve_ctor_positions(_NoCollator, "data_collator")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
