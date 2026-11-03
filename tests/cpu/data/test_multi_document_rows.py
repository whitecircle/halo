#!/usr/bin/env python
"""``reject_multi_document_rows`` holds every rule on the rows a model cannot keep apart, or no kernel it runs can
serve, and both callers that see a run's rows before it trains go through it: the collator factory after the load,
and the SFT script before it.

Each rule is shown refusing its rows and passing them once its one precondition is lifted. The SFT script is driven
through ``main()`` up to its distributed init, with the checkpoint probe stubbed.

    python tests/cpu/data/test_multi_document_rows.py
"""

import contextlib
import sys
import types
from unittest import mock

import pytest

from halo.data.collators import collator_rules
from halo.data.collators.collator_rules import reject_multi_document_rows
from halo.data.collators.factory import select_data_collator
from halo.models import segment_markers
from tests.common.repo import load_script_module
from tests.common.tiny_models import (
    TINY_BAILING_RING_FAMILY,
    TINY_DENSE_ATTENTION_FAMILIES,
    TINY_DENSE_FAMILIES,
    TINY_MOE_FAMILIES,
    tiny_family_model,
)

FA2, FA4, SDPA = "flash_attention_2", "flash_attention_4", "sdpa"
# The segment-aware kernel checks as a GPU host without the conv kernel answers them, and as the images answer them.
_KERNELS_MISSING = (("causal_conv1d", lambda: False), ("fla>=0.2.2", lambda: True))
_KERNELS_PRESENT = (("causal_conv1d", lambda: True), ("fla>=0.2.2", lambda: True))


def _config(family, attn_impl: str | None = None):
    config = tiny_family_model(family).config
    if attn_impl is not None:
        for level in (config, config.get_text_config()):
            level._attn_implementation = attn_impl
    return config


# rule -> (family, mode, attn_impl, keyword arguments, refusal match, the keywords that lift its precondition)
_RULES = {
    "dense-packing-glm5_next": (
        TINY_MOE_FAMILIES["glm5_next"],
        "packing",
        SDPA,
        {"allow_cross_document_linear_attention": True},
        "packing=True on attn_implementation='sdpa'",
        {"attn": FA2},
    ),
    "dense-packing-bailing": (
        TINY_MOE_FAMILIES["bailing_moe"],
        "packing",
        SDPA,
        {},
        "packing=True on attn_implementation='sdpa'",
        {"attn": FA2},
    ),
    "padding-free-on-a-dense-kernel": (
        TINY_DENSE_FAMILIES["qwen3"],
        "padding_free",
        SDPA,
        {},
        "padding_free requires a varlen",
        {"attn": FA2},
    ),
    "mrope-padding-free": (
        TINY_DENSE_FAMILIES["qwen3_5"],
        "padding_free",
        FA2,
        {},
        "rotates by M-RoPE",
        {"mode": "packing"},
    ),
    "compressed-kv": (TINY_MOE_FAMILIES["deepseek_v4"], "packing", FA2, {}, "pool KV over windows", {"mode": None}),
    "segment-aware-kernels": (
        TINY_DENSE_FAMILIES["qwen3_5"],
        "packing",
        FA2,
        {"kernels_missing": True},
        "causal_conv1d unavailable",
        {"kernels_missing": False},
    ),
    "unmarked-linear-attention": (
        TINY_BAILING_RING_FAMILY,
        "packing",
        FA2,
        {},
        "allow_cross_document_linear_attention",
        {"allow_cross_document_linear_attention": True},
    ),
    "pipeline-packed-rows": (
        TINY_BAILING_RING_FAMILY,
        "packing",
        FA2,
        {"keeps_packed_rows": True, "allow_cross_document_linear_attention": True},
        "pipeline",
        {"keeps_packed_rows": False},
    ),
    "fa4-varlen-softcap": (
        TINY_DENSE_ATTENTION_FAMILIES["gemma2"],
        None,
        FA4,
        {"varlen_rows": True},
        "attention logit softcapping",
        {"varlen_rows": False},
    ),
}


def _check(family, mode, attn_impl, *, kernels_missing=False, **kwargs) -> None:
    arguments = {
        "allow_cross_document_linear_attention": False,
        "keeps_packed_rows": False,
        "varlen_rows": mode is not None,
        **kwargs,
    }
    backends = _KERNELS_MISSING if kernels_missing else _KERNELS_PRESENT
    with mock.patch.object(segment_markers, "GDN_SEGMENT_AWARE_BACKENDS", backends):
        reject_multi_document_rows(_config(family, attn_impl), mode, attn_impl=attn_impl, **arguments)


@pytest.mark.parametrize("rule", list(_RULES))
def test_each_rule_refuses_its_rows_and_passes_them_without_its_precondition(rule):
    family, mode, attn_impl, kwargs, match, lifted = _RULES[rule]
    with pytest.raises(ValueError, match=match):
        _check(family, mode, attn_impl, **kwargs)
    lifted = dict(lifted)
    attn_impl = lifted.pop("attn", attn_impl)
    mode = lifted.pop("mode", mode)
    _check(family, mode, attn_impl, **{**kwargs, **lifted})


@pytest.mark.parametrize("rule", ["dense-packing-glm5_next", "segment-aware-kernels", "pipeline-packed-rows"])
def test_the_collator_factory_refuses_through_the_one_rule(rule):
    """The factory reads the backend off the loaded model's config and hands every row rule the same arguments."""
    family, mode, attn_impl, kwargs, match, _ = _RULES[rule]
    backends = _KERNELS_MISSING if kwargs.get("kernels_missing") else _KERNELS_PRESENT
    with (
        mock.patch.object(segment_markers, "GDN_SEGMENT_AWARE_BACKENDS", backends),
        pytest.raises(ValueError, match=match),
    ):
        select_data_collator(
            types.SimpleNamespace(pad_token_id=0, eos_token_id=1, padding_side="right"),
            train_on_completions_only=False,
            packing=mode == "packing",
            padding_free=mode == "padding_free",
            model_config=_config(family, attn_impl),
            keeps_packed_rows=kwargs.get("keeps_packed_rows", False),
            allow_cross_document_linear_attention=kwargs.get("allow_cross_document_linear_attention", False),
        )


class _Initialized(Exception):
    """Raised by the stubbed distributed init: ``main()`` got past its pre-load checks."""


def _sft_main(tmp_path, yaml_body: str, checkpoint_config, *, kernels_missing: bool = False) -> None:
    """``sft.py:main()`` from a YAML holding ``yaml_body``, its checkpoint probed as ``checkpoint_config``, up to the
    distributed init."""
    module = load_script_module("scripts/training/sft.py", "halo_test_sft_multi_document_rows")
    config = tmp_path / "config.yaml"
    config.write_text(
        f"model_name_or_path: stub/model\ndataset:\n- dummy/dataset\noutput_dir: {tmp_path / 'out'}\n"
        f"bf16: false\nuse_cpu: true\nmax_length: 512\nper_device_train_batch_size: 1\n{yaml_body}"
    )
    backends = _KERNELS_MISSING if kernels_missing else _KERNELS_PRESENT
    with (
        mock.patch.object(segment_markers, "GDN_SEGMENT_AWARE_BACKENDS", backends),
        mock.patch.object(module, "probe_checkpoint", return_value=(checkpoint_config, False)),
        mock.patch.object(module, "init_training_script", side_effect=_Initialized),
        mock.patch("halo.entrypoints.script.install_log_tee"),
        mock.patch.object(sys, "argv", ["prog", str(config)]),
    ):
        module.main()


@pytest.mark.parametrize(
    ("yaml_body", "family", "attn_impl", "kernels_missing", "match"),
    [
        (
            "packing: true\nattn_implementation: sdpa\nallow_cross_document_linear_attention: true\n",
            TINY_MOE_FAMILIES["glm5_next"],
            None,
            False,
            "packing=True on attn_implementation='sdpa'",
        ),
        ("packing: true\n", TINY_DENSE_FAMILIES["qwen3_5"], None, True, "causal_conv1d unavailable"),
        ("padding_free: true\n", TINY_DENSE_FAMILIES["qwen3_5"], None, False, "rotates by M-RoPE"),
    ],
    ids=["dense-packing", "segment-aware-kernels", "mrope-padding-free"],
)
def test_the_sft_script_refuses_the_rows_before_its_distributed_init(
    tmp_path, yaml_body, family, attn_impl, kernels_missing, match
):
    """The script runs the factory's rules on the probed config before anything loads; without them a run reaches
    its init, then the model load and the packing pass, before the factory refuses."""
    with pytest.raises(ValueError, match=match):
        _sft_main(tmp_path, yaml_body, _config(family, attn_impl), kernels_missing=kernels_missing)


def test_rows_the_rules_accept_reach_the_distributed_init(tmp_path):
    """Anti-vacuity: packed rows of a family whose kernels are in place pass every pre-load rule."""
    with pytest.raises(_Initialized):
        _sft_main(tmp_path, "packing: true\n", _config(TINY_DENSE_FAMILIES["qwen3_5"]))


def test_the_dense_kernel_rule_is_one_predicate():
    """The refusal and the predicate a caller asks before resolving a backend for it agree, row shape by row shape."""
    for family in (TINY_MOE_FAMILIES["glm5_next"], TINY_DENSE_FAMILIES["qwen3"]):
        config = _config(family)
        for padding_free, packing in ((True, False), (False, True), (False, False)):
            refuses = collator_rules.has_dense_kernel_row_refusal(config, padding_free=padding_free, packing=packing)
            with contextlib.nullcontext() if not refuses else pytest.raises(ValueError):
                collator_rules.reject_dense_attention_rows(config, SDPA, padding_free=padding_free, packing=packing)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
