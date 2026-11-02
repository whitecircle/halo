#!/usr/bin/env python
"""A packing or padding-free collator handed straight to ``DistributedSFTTrainer`` meets every rule the collator
factory runs on its rows (``reject_multi_document_rows``), read on the model the trainer built, and SMPO's
padding-free row meets the same rules.

    python tests/cpu/trainers/test_sft_trainer_multi_document_rows.py
"""

import types
from unittest import mock

import pytest
from trl import SFTTrainer

from halo.configs.preference import SMPOConfig
from halo.data.collators.packing import DataCollatorWithFlattening, DataCollatorWithPacking
from halo.distributed.parallelism_config import ParallelismConfig
from halo.models import segment_markers
from halo.trainers.preference.smpo import SMPOTrainer
from halo.trainers.sft import DistributedSFTTrainer
from tests.common.tiny_models import (
    TINY_DENSE_ATTENTION_FAMILIES,
    TINY_DENSE_FAMILIES,
    TINY_MOE_FAMILIES,
    tiny_family_model,
)

_KERNELS_MISSING = (("causal_conv1d", lambda: False), ("fla>=0.2.2", lambda: True))
_KERNELS_PRESENT = (("causal_conv1d", lambda: True), ("fla>=0.2.2", lambda: True))


class _Built(Exception):
    """Raised after the trainer's row rules: construction got past them."""


def _model(family, attn_impl: str):
    model = tiny_family_model(family)
    for level in (model.config, model.config.get_text_config()):
        level._attn_implementation = attn_impl
    return model


def _construct_sft(handed, built, collator) -> None:
    """``DistributedSFTTrainer`` handed ``handed`` (a model or its id), its base building ``built``."""

    def _init_cfg(self, kwargs, **_):
        self.parallelism_config = types.SimpleNamespace(is_cp_mode=False)
        return kwargs

    def _base_init(self, *_args, **_kwargs):
        self.model = built

    with (
        mock.patch.object(DistributedSFTTrainer, "_init_distributed_config", _init_cfg),
        mock.patch.object(SFTTrainer, "__init__", _base_init),
        mock.patch.object(DistributedSFTTrainer, "_configure_distributed_modes", side_effect=_Built),
        pytest.raises(_Built),
    ):
        DistributedSFTTrainer(model=handed, data_collator=collator)


def _packing(allow_crossing: bool = False):
    collator = object.__new__(DataCollatorWithPacking)
    collator.allow_cross_document_linear_attention = allow_crossing
    return collator


# case -> (family, backend, collator, segment-aware kernel checks, refusal)
_REFUSED = {
    "glm5_next-dense-packing": (
        TINY_MOE_FAMILIES["glm5_next"],
        "sdpa",
        lambda: _packing(allow_crossing=True),
        _KERNELS_PRESENT,
        "packing=True on attn_implementation='sdpa'",
    ),
    "bailing-dense-packing": (
        TINY_MOE_FAMILIES["bailing_moe"],
        "sdpa",
        _packing,
        _KERNELS_PRESENT,
        "packing=True on attn_implementation='sdpa'",
    ),
    "gated-delta-net-without-kernels": (
        TINY_DENSE_FAMILIES["qwen3_5"],
        "flash_attention_2",
        _packing,
        _KERNELS_MISSING,
        "causal_conv1d unavailable",
    ),
    "fa4-softcap-padding-free": (
        TINY_DENSE_ATTENTION_FAMILIES["gemma2"],
        "flash_attention_4",
        lambda: object.__new__(DataCollatorWithFlattening),
        _KERNELS_PRESENT,
        "attention logit softcapping",
    ),
}


@pytest.mark.parametrize("case", list(_REFUSED))
def test_a_handed_collator_meets_the_factory_rules(case, monkeypatch):
    family, attn_impl, collator, kernels, match = _REFUSED[case]
    monkeypatch.setattr(segment_markers, "GDN_SEGMENT_AWARE_BACKENDS", kernels)
    model = _model(family, attn_impl)
    with pytest.raises(ValueError, match=match):
        _construct_sft(model, model, collator())


def test_a_handed_collator_on_a_flash_backend_constructs(monkeypatch):
    """Anti-vacuity: the same packed rows on a backend that keeps them apart reach the trainer setup."""
    monkeypatch.setattr(segment_markers, "GDN_SEGMENT_AWARE_BACKENDS", _KERNELS_PRESENT)
    model = _model(TINY_MOE_FAMILIES["bailing_moe"], "flash_attention_2")
    _construct_sft(model, model, _packing())


def test_the_rules_read_the_model_built_from_a_model_id(monkeypatch):
    """A model id handed to the constructor carries no config; the rules read the model the base built from it."""
    monkeypatch.setattr(segment_markers, "GDN_SEGMENT_AWARE_BACKENDS", _KERNELS_PRESENT)
    built = _model(TINY_MOE_FAMILIES["bailing_moe"], "sdpa")
    with pytest.raises(ValueError, match="packing=True on attn_implementation='sdpa'"):
        _construct_sft("org/bailing", built, _packing())


def test_smpo_padding_free_meets_the_fa4_varlen_rule(tmp_path, monkeypatch):
    """SMPO's flattened row reaches FA4's varlen kernels like the SFT collator's, so a softcapped model there is
    refused at the policy load."""
    monkeypatch.setattr(segment_markers, "GDN_SEGMENT_AWARE_BACKENDS", _KERNELS_PRESENT)
    with pytest.raises(ValueError, match="attention logit softcapping"):
        SMPOTrainer(
            model=_model(TINY_DENSE_ATTENTION_FAMILIES["gemma2"], "flash_attention_4"),
            args=SMPOConfig(output_dir=str(tmp_path), padding_free=True, bf16=False, report_to="none"),
            processing_class=types.SimpleNamespace(pad_token_id=0, eos_token_id=1),
            parallelism_config=ParallelismConfig(),
            is_vlm=False,
        )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
