#!/usr/bin/env python
"""The trainer ctor position tables are DERIVED from the installed TRL signatures.

A hand-maintained ``{"data_collator": 2}`` silently reads a neighbouring argument the release after
TRL inserts a parameter; ``ctor_positions`` must follow the signature instead, and fail loud when
the parameter disappears from it.

The per-trainer slots below are pinned as LITERALS read off TRL 1.6.0's signatures — recomputing
``params.index(...)`` here would assert the derivation against itself. A TRL bump that moves one of
these parameters fails this test, which is the point: the gates that read them (SFT's CP collator
rejection, DPO/KTO's EP/TP reference rejection, the mixin's positional model/config) would otherwise
gate on the wrong argument.

Run: ``python tests/cpu/trainers/test_ctor_positions_derived.py`` (or ``pytest -m cpu``).
"""

import sys

import pytest

from src.trainers.grpo.mixins.on_policy_init import GRPO_CTOR_POSITIONS
from src.trainers.mixins.validation import ctor_config, ctor_model_and_config, ctor_positions, ctor_value
from src.trainers.preference.dpo import _CTOR_POSITIONS as DPO_CTOR_POSITIONS
from src.trainers.preference.kto import _CTOR_POSITIONS as KTO_CTOR_POSITIONS
from src.trainers.reward.bradley_terry import _CTOR_POSITIONS as REWARD_CTOR_POSITIONS
from src.trainers.sft import _CTOR_POSITIONS as SFT_CTOR_POSITIONS


def test_sft_slots_match_installed_trl():
    """TRL 1.6.0: ``SFTTrainer.__init__(self, model, args, data_collator, ...)``."""
    assert SFT_CTOR_POSITIONS == {"model": 0, "args": 1, "data_collator": 2}


def test_dpo_slots_match_installed_trl():
    """TRL 1.6.0: ``DPOTrainer.__init__(self, model, ref_model, args, ...)``."""
    assert DPO_CTOR_POSITIONS == {"model": 0, "args": 2, "ref_model": 1}


def test_kto_slots_match_installed_trl():
    """TRL 1.6.0: the public ``KTOTrainer.__init__`` is a ``(*args, **kwargs)`` deprecation shim, so
    the slots are ``trl.experimental.kto.kto_trainer.KTOTrainer.__init__(self, model, ref_model,
    args, ...)`` — the signature the shim forwards to."""
    assert KTO_CTOR_POSITIONS == {"model": 0, "args": 2, "ref_model": 1}


def test_reward_and_grpo_slots_match_installed_trl():
    """TRL 1.6.0: ``RewardTrainer(model, args, ...)``, ``GRPOTrainer(model, reward_funcs, args, ...)``."""
    assert REWARD_CTOR_POSITIONS == {"model": 0, "args": 1}
    assert GRPO_CTOR_POSITIONS == {"model": 0, "args": 2}


def test_ctor_config_reads_the_slot_of_the_forwarded_base():
    """``RewardTrainer`` takes its config one slot earlier than the GRPO/DPO shape; the table decides."""
    config = object()
    assert ctor_config((None, config), {}, REWARD_CTOR_POSITIONS) is config
    assert ctor_config((None, None, config), {}, GRPO_CTOR_POSITIONS) is config
    assert ctor_config((None, config), {}, GRPO_CTOR_POSITIONS) is None


def test_ctor_config_falls_through_an_explicit_none_keyword():
    """Load-bearing at ``_require_vllm_server_mode``: ``args=None`` must not mask a positional config."""
    config = object()
    assert ctor_config((None, None, config), {"args": None}, GRPO_CTOR_POSITIONS) is config
    assert ctor_config((), {"args": config}, GRPO_CTOR_POSITIONS) is config
    assert ctor_config((), {}, GRPO_CTOR_POSITIONS) is None


def test_positionals_without_a_slot_table_raise():
    """Keywords resolve without a table; positionals cannot, and skipping them would silently drop the
    setup that reads the model and the config."""
    model, config = object(), object()
    assert ctor_model_and_config((), {"model": model, "args": config}, None) == (model, config)
    with pytest.raises(ValueError, match="without a ctor_positions table"):
        ctor_model_and_config((model, config), {}, None)


def test_gate_extracts_the_passed_collator():
    """The value the CP gate reads must be the collator the caller passed — either way in."""
    collator = object()
    ctor_args: list = [None] * (SFT_CTOR_POSITIONS["data_collator"] + 1)
    ctor_args[SFT_CTOR_POSITIONS["data_collator"]] = collator
    assert ctor_value(tuple(ctor_args), {}, "data_collator", SFT_CTOR_POSITIONS) is collator
    assert ctor_value((), {"data_collator": collator}, "data_collator", SFT_CTOR_POSITIONS) is collator


def test_gate_extracts_the_passed_reference_model():
    """The DPO/KTO EP/TP reference gate reads ``ref_model`` out of the same positional slot."""
    ref = object()
    assert ctor_value((None, ref), {}, "ref_model", DPO_CTOR_POSITIONS) is ref
    assert ctor_value((None, ref), {}, "ref_model", KTO_CTOR_POSITIONS) is ref
    assert ctor_value((), {"ref_model": ref}, "ref_model", DPO_CTOR_POSITIONS) is ref


def test_derivation_follows_a_shifted_signature():
    """Simulate the TRL insertion that breaks a hand table: the derived slot must move with it."""

    class _Base:
        def __init__(self, model, args=None, data_collator=None):
            pass

    class _Shifted:
        def __init__(self, model, inserted=None, args=None, data_collator=None):
            pass

    assert ctor_positions(_Base, "data_collator") == {"data_collator": 2}
    shifted = ctor_positions(_Shifted, "data_collator")
    assert shifted == {"data_collator": 3}

    collator = object()
    assert ctor_value((None, None, None, collator), {}, "data_collator", shifted) is collator
    # A stale hand table (slot 2) would read the neighbouring argument here.
    assert ctor_value((None, None, collator, None), {}, "data_collator", shifted) is None


def test_derivation_looks_through_a_varargs_shim():
    """TRL's KTO shape: the public class forwards ``*args``, so the slots live one MRO hop up. A
    derivation reading only the shim's own signature finds no parameters at all and raises."""

    class _Real:
        def __init__(self, model, ref_model=None, args=None):
            pass

    class _Shim(_Real):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

    assert ctor_positions(_Shim, "ref_model") == {"ref_model": 1}
    ref = object()
    assert ctor_value((None, ref), {}, "ref_model", ctor_positions(_Shim, "ref_model")) is ref


def test_missing_parameter_fails_loud():
    """A name the installed signature does not declare must raise, not gate on a guessed slot."""

    class _NoCollator:
        def __init__(self, model, args=None):
            pass

    with pytest.raises(ValueError, match="data_collator"):
        ctor_positions(_NoCollator, "data_collator")

    # The shim walk must not paper over it either: a shim whose base lacks the name still raises.
    class _Shim(_NoCollator):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

    with pytest.raises(ValueError, match="data_collator"):
        ctor_positions(_Shim, "data_collator")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
