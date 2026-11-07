#!/usr/bin/env python
"""``reject_multi_document_rows`` holds every rule on the rows a model cannot keep apart or no kernel can serve, and
both callers (the collator factory after the load, the SFT script before it) go through it. Each rule refuses its
rows and passes them once its precondition is lifted."""

import contextlib
import sys
import types
from unittest import mock

import pytest
from transformers.models.auto.configuration_auto import CONFIG_MAPPING

from halo.data.collators import collator_rules
from halo.data.collators.collator_rules import reject_multi_document_rows, reject_unpackable_family
from halo.data.collators.factory import select_data_collator
from halo.models import segment_markers
from halo.models.segment_markers import packing_isolation_is_measured
from tests.common.repo import load_script_module
from tests.common.tiny_models import (
    TINY_BAILING_RING_FAMILY,
    TINY_DENSE_ATTENTION_FAMILIES,
    TINY_DENSE_FAMILIES,
    TINY_MOE_FAMILIES,
    TINY_MOE_VLM_FAMILIES,
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


@pytest.mark.parametrize(
    ("family", "kwargs", "refused"),
    [
        (TINY_BAILING_RING_FAMILY, {}, "linear-attention layers"),
        (TINY_MOE_FAMILIES["glm5_next"], {"allow_cross_document_linear_attention": True}, "packing=True"),
        (TINY_MOE_FAMILIES["deepseek_v4"], {"allow_cross_document_linear_attention": True}, "is refused"),
        (TINY_BAILING_RING_FAMILY, {"allow_cross_document_linear_attention": True}, None),
        (TINY_MOE_FAMILIES["gpt_oss"], {}, None),
        (TINY_DENSE_FAMILIES["qwen3_5"], {}, None),
        (TINY_DENSE_FAMILIES["qwen3"], {}, None),
    ],
    ids=["ring", "glm5_next_opted_in", "dsv4_opted_in", "ring_opted_in", "gpt_oss", "qwen3_5", "qwen3"],
)
def test_a_bake_refuses_the_rows_training_would_refuse(family, kwargs, refused):
    """Read off the family alone: a bake holds no backend, and the host baking may lack what training needs."""
    config = _config(family)
    options = {"allow_cross_document_linear_attention": False, **kwargs}
    with mock.patch.object(segment_markers, "GDN_SEGMENT_AWARE_BACKENDS", _KERNELS_MISSING):
        if refused is None:
            reject_unpackable_family(config, flag="--pack_sequences", **options)
        else:
            with pytest.raises(ValueError, match=refused):
                reject_unpackable_family(config, flag="--pack_sequences", **options)


_ROSTER = {**TINY_DENSE_FAMILIES, **TINY_DENSE_ATTENTION_FAMILIES, **TINY_MOE_FAMILIES, **TINY_MOE_VLM_FAMILIES}


@pytest.mark.parametrize("family", list(_ROSTER))
def test_every_family_of_the_roster_is_measured(family):
    assert packing_isolation_is_measured(_config(_ROSTER[family]))


@pytest.mark.parametrize("model_type", ["bart", "opt", "biogpt", "gpt_neo", "moshi"])
@pytest.mark.parametrize("mode", ["packing", "padding_free"])
def test_a_family_outside_the_table_is_warned_about(model_type, mode):
    with mock.patch.object(collator_rules.logger, "warning") as warning:
        collator_rules._warn_unmeasured_multi_document_family(CONFIG_MAPPING[model_type](), multi_document=True)
    assert warning.call_count == 1
    assert "outside the table" in warning.call_args.args[0]


def test_one_document_rows_warn_about_no_family():
    with mock.patch.object(collator_rules.logger, "warning") as warning:
        collator_rules._warn_unmeasured_multi_document_family(CONFIG_MAPPING["bart"](), multi_document=False)
    assert warning.call_count == 0


def test_the_collator_factory_warns_on_a_family_outside_the_table():
    tokenizer = types.SimpleNamespace(pad_token_id=0, eos_token_id=1, padding_side="right")
    config = CONFIG_MAPPING["opt"]()
    config._attn_implementation = FA2
    with mock.patch.object(collator_rules.logger, "warning") as warning:
        select_data_collator(tokenizer, train_on_completions_only=False, packing=True, model_config=config)
    assert any("outside the table" in str(call.args[0]) for call in warning.call_args_list)


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
    ],
    ids=["dense-packing", "segment-aware-kernels"],
)
def test_the_sft_script_refuses_the_rows_before_its_distributed_init(
    tmp_path, yaml_body, family, attn_impl, kernels_missing, match
):
    """The script runs the factory's rules on the probed config before anything loads; without them a run reaches
    its init, then the model load and the packing pass, before the factory refuses."""
    with pytest.raises(ValueError, match=match):
        _sft_main(tmp_path, yaml_body, _config(family, attn_impl), kernels_missing=kernels_missing)


def test_rows_the_rules_accept_reach_the_distributed_init(tmp_path):
    """Packed rows of a family whose kernels are in place pass every pre-load rule."""
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
