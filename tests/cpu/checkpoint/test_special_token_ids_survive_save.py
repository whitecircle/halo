#!/usr/bin/env python
"""A list-valued config.eos_token_id must survive training into every save.

HF's ``align_special_tokens`` (run by ``Trainer.train``) overwrites ``config.eos_token_id`` with
the TOKENIZER's single id. Gemma 4 ships ``[1, 106]`` and GLM three role enders — the collapsed
config then serves without turn-end stops, and a stage-2 run masking against the exported config
trains ZERO tokens. The mixin snapshots at setup and restores before every config write.

    python tests/cpu/checkpoint/test_special_token_ids_survive_save.py
"""

from types import SimpleNamespace

import pytest

from src.models.loading.config_levels import restore_special_token_ids, snapshot_special_token_ids


class _Config(SimpleNamespace):
    def get_text_config(self):
        return self


def test_collapsed_eos_list_is_restored():
    config = _Config(eos_token_id=[1, 106], bos_token_id=2, pad_token_id=0)
    snapshot = snapshot_special_token_ids(config)

    config.eos_token_id = 1  # what align_special_tokens does when tokenizer.eos != the list
    config.pad_token_id = 106

    restore_special_token_ids(snapshot)
    assert config.eos_token_id == [1, 106]
    assert config.pad_token_id == 0


def test_snapshot_is_a_copy_not_an_alias():
    config = _Config(eos_token_id=[1, 106])
    snapshot = snapshot_special_token_ids(config)
    config.eos_token_id.append(999)  # in-place mutation must not reach the snapshot

    restore_special_token_ids(snapshot)
    assert config.eos_token_id == [1, 106]


def test_composite_config_levels_both_restore():
    text = _Config(eos_token_id=[7, 8])
    wrapper = SimpleNamespace(eos_token_id=None, get_text_config=lambda: text)
    snapshot = snapshot_special_token_ids(wrapper)

    wrapper.eos_token_id = 7
    text.eos_token_id = 7
    restore_special_token_ids(snapshot)
    assert wrapper.eos_token_id is None and text.eos_token_id == [7, 8]


def test_restore_removes_an_id_align_planted_on_the_wrapper():
    """5.14 composite wrappers declare no top-level ids; align_special_tokens PLANTS one there —
    the restore must remove it, or every export ships a wrapper-level single-eos int."""
    text = _Config(eos_token_id=[7, 8], bos_token_id=1, pad_token_id=0)
    wrapper = SimpleNamespace(get_text_config=lambda: text)  # no id fields at all
    snapshot = snapshot_special_token_ids(wrapper)

    wrapper.eos_token_id = 7  # align planting a brand-new wrapper-level attribute
    text.eos_token_id = 7
    restore_special_token_ids(snapshot)
    assert not hasattr(wrapper, "eos_token_id"), "the planted wrapper-level id must be removed"
    assert text.eos_token_id == [7, 8]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
