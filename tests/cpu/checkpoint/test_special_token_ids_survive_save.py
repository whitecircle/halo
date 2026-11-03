#!/usr/bin/env python
"""A list-valued config.eos_token_id survives training though ``align_special_tokens`` overwrites it with one id.
Collapsed, a stage-2 run masking against the exported config trains zero tokens."""

from types import SimpleNamespace

import pytest

from halo.models.config_access import SPECIAL_TOKEN_ID_FIELDS, restore_config_fields, snapshot_config_fields


class _Config(SimpleNamespace):
    def get_text_config(self):
        return self


def test_collapsed_eos_list_is_restored():
    config = _Config(eos_token_id=[1, 106], bos_token_id=2, pad_token_id=0)
    snapshot = snapshot_config_fields(config, SPECIAL_TOKEN_ID_FIELDS)

    config.eos_token_id = 1  # What align_special_tokens does when tokenizer.eos != the list.
    config.pad_token_id = 106

    restore_config_fields(snapshot)
    assert config.eos_token_id == [1, 106]
    assert config.pad_token_id == 0


def test_snapshot_is_a_copy_not_an_alias():
    config = _Config(eos_token_id=[1, 106])
    snapshot = snapshot_config_fields(config, SPECIAL_TOKEN_ID_FIELDS)
    config.eos_token_id.append(999)  # In-place mutation must not reach the snapshot.

    restore_config_fields(snapshot)
    assert config.eos_token_id == [1, 106]


def test_composite_config_levels_both_restore():
    text = _Config(eos_token_id=[7, 8])
    wrapper = SimpleNamespace(eos_token_id=None, get_text_config=lambda: text)
    snapshot = snapshot_config_fields(wrapper, SPECIAL_TOKEN_ID_FIELDS)

    wrapper.eos_token_id = 7
    text.eos_token_id = 7
    restore_config_fields(snapshot)
    assert wrapper.eos_token_id is None and text.eos_token_id == [7, 8]


def test_restore_removes_an_id_align_planted_on_the_wrapper():
    """5.14 composite wrappers declare no top-level ids; a planted one would ship a wrapper-level single-eos int in
    every export."""
    text = _Config(eos_token_id=[7, 8], bos_token_id=1, pad_token_id=0)
    wrapper = SimpleNamespace(get_text_config=lambda: text)  # No id fields at all.
    snapshot = snapshot_config_fields(wrapper, SPECIAL_TOKEN_ID_FIELDS)

    wrapper.eos_token_id = 7
    text.eos_token_id = 7
    restore_config_fields(snapshot)
    assert not hasattr(wrapper, "eos_token_id"), "the planted wrapper-level id must be removed"
    assert text.eos_token_id == [7, 8]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
