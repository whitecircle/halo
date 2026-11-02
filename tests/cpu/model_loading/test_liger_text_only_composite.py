#!/usr/bin/env python
"""A text-only load of a multimodal checkpoint applies Liger to the text sub-config the built model carries.

On a composite config ``AutoModelForCausalLM`` builds the text-only sibling, which holds the text sub-config and runs
the head the fused loss (``fused_linear_cross_entropy``) patches. Liger applied to the composite config resolves the
applier through the wrapper's text fallback, takes the wrapper's own head for the one that runs and keeps the fused
loss off: a Qwen3.8-Flash-Next text-only run would hold the full logits plane while its recipes request the fused
loss. The load of the composite class keeps that wrapper-head guard. Pinned through ``load_distributed_model`` on
the tiny Flash-Next checkpoint (an eager build from the config, and the lazy EP loader's meta shell), and across
every composite ``model_type`` transformers maps that admits a text-only load.

    python tests/cpu/model_loading/test_liger_text_only_composite.py
"""

import json
import logging
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.models.auto.configuration_auto import CONFIG_MAPPING
from transformers.models.auto.modeling_auto import MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES

from halo.model_loading.lazy_safetensors.meta_shell import instantiate_on_meta
from halo.model_loading.liger import dispatch
from halo.model_loading.liger.applier import LigerApplier
from halo.model_loading.model_preparation import resolve_auto_model_class
from halo.model_loading.parallel import loader as model_loading
from halo.model_loading.parallel.model_source import ModelSource
from halo.models.config_access import TOOLKIT_STAMP_PREFIX, config_export_ready, text_config
from tests.common.parallelism import make_parallelism_config
from tests.common.tiny_models import TINY_MOE_VLM_FAMILIES, tiny_family_model

FLCE = "fused_linear_cross_entropy"
FLASH_NEXT_TEXT = "qwen4_exp_text"
WRAPPER_WARNING = "multimodal wrapper"


class _RecordingApplier(LigerApplier):
    """The family's toolkit applier, recording each call's flags instead of patching the modeling classes."""

    def __init__(self, spec):
        super().__init__(spec)
        self.calls: list[dict] = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return []


@pytest.fixture
def flash_next_applier(monkeypatch):
    applier = _RecordingApplier(dispatch._TOOLKIT_LIGER_APPLIERS[FLASH_NEXT_TEXT].spec)
    monkeypatch.setitem(dispatch._TOOLKIT_LIGER_APPLIERS, FLASH_NEXT_TEXT, applier)
    return applier


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    """The tiny Flash-Next release layout: the multimodal wrapper, saved with its vision tower."""
    directory = tmp_path_factory.mktemp("flash_next")
    torch.manual_seed(0)
    tiny_family_model(TINY_MOE_VLM_FAMILIES["qwen4_exp"]).save_pretrained(directory)
    return str(directory)


def _eager_build(source, pc, model_class, model_config, dtype, **kwargs):
    return model_class.from_config(model_config)


def _meta_shell_build(source, pc, model_class, model_config, dtype, **kwargs):
    """The lazy EP loader's build: the shell ``from_pretrained`` reads on the meta device."""
    return instantiate_on_meta(source.local_dir, model_class, model_config, dtype, trust_remote_code=False)


def _load(config, build, *, text_only: bool, liger_kernel_config: dict | None, local_dir: str = "org/model"):
    """``load_distributed_model`` with Liger, the class resolution and the build real, everything a CPU process
    cannot run (or need not, the family patches and kernel warm-ups) stubbed; returns the built model."""
    with (
        patch.object(model_loading, "configure_float32_matmul_precision"),
        patch.object(model_loading, "apply_remote_code_compat_shims"),
        patch.object(model_loading, "resolve_model_source", lambda name, revision, **k: ModelSource(name, None, name)),
        patch.object(model_loading, "AutoConfig", SimpleNamespace(from_pretrained=lambda *a, **k: config)),
        patch.object(model_loading, "resolve_attn_implementation", lambda *a, **k: "sdpa"),
        patch.object(
            model_loading, "apply_family_patches", lambda model_config, attn_implementation: attn_implementation
        ),
        patch.object(model_loading, "AutoTokenizer", SimpleNamespace(from_pretrained=lambda *a, **k: MagicMock())),
        patch.object(model_loading, "_dispatch_model_loading", build),
        patch.object(model_loading, "apply_run_model_policies"),
        patch.object(model_loading, "warm_attention_kernels"),
    ):
        model, _ = model_loading.load_distributed_model(
            local_dir,
            parallelism_config=make_parallelism_config(world_size=1, gpus_per_node=1, use_grouped_gemm=False),
            dtype=torch.float32,
            trust_remote_code=False,
            use_liger_kernel=True,
            liger_kernel_config=liger_kernel_config,
            text_only_model=text_only,
        )
    return model


@pytest.mark.parametrize("build", [_eager_build, _meta_shell_build], ids=["eager", "lazy_ep_meta_shell"])
def test_a_text_only_flash_next_load_takes_the_fused_loss_it_requests(build, checkpoint, flash_next_applier, caplog):
    """The text tower's applier runs with the fused loss on, its stamp sits on the built model's config, where the
    trainer's TRL flag reconciliation reads it, and no export carries it."""
    config = AutoConfig.from_pretrained(checkpoint)
    with caplog.at_level(logging.WARNING, logger=dispatch.__name__):
        model = _load(config, build, text_only=True, liger_kernel_config={FLCE: True}, local_dir=checkpoint)

    assert type(model).__name__ == "Qwen4ExpForCausalLM", "premise: the text-only load builds the text sibling"
    applied = getattr(model.config, dispatch.LIGER_APPLIED_CONFIG_ATTR)
    assert applied[FLCE] is True and applied["cross_entropy"] is False, applied
    assert [call[FLCE] for call in flash_next_applier.calls] == [True]
    assert not [r for r in caplog.records if WRAPPER_WARNING in r.getMessage()], "the wrapper-head refusal fired"

    training_config = SimpleNamespace(use_liger_kernel=True, liger_kernel_config={FLCE: True})
    dispatch.finalize_liger_after_direct_load(training_config, True, model)
    assert training_config.use_liger_kernel is True, "TRL would slice the absent logits of a fused-loss model"

    with config_export_ready(model.config):
        exported = json.loads(model.config.to_json_string())
    assert not [key for key in exported if key.startswith(TOOLKIT_STAMP_PREFIX)], "a run stamp reached config.json"


def test_the_composite_class_load_keeps_the_fused_loss_off(checkpoint, flash_next_applier, caplog):
    """The multimodal class runs its own head, which the text tower's fused loss never patches: the request is
    refused under the wrapper-head warning, so the spec registry is not where the text-only fix may live."""
    config = AutoConfig.from_pretrained(checkpoint)
    with caplog.at_level(logging.WARNING, logger=dispatch.__name__):
        model = _load(config, _eager_build, text_only=False, liger_kernel_config={FLCE: True}, local_dir=checkpoint)

    assert type(model).__name__ == "Qwen4ExpForConditionalGeneration"
    assert getattr(model.config, dispatch.LIGER_APPLIED_CONFIG_ATTR)[FLCE] is False
    assert [call[FLCE] for call in flash_next_applier.calls] == [False]
    assert [r for r in caplog.records if WRAPPER_WARNING in r.getMessage()]


def _composite_config(model_type: str):
    """The tiny roster's config where it carries the family (a stock default config may not build), else the stock
    default."""
    family = TINY_MOE_VLM_FAMILIES.get(model_type)
    return family.build({}).config if family is not None else CONFIG_MAPPING[model_type]()


def _text_only_composite_types() -> list[str]:
    """Every composite ``model_type`` transformers maps to an image-text-to-text class that also admits a text-only
    load (:func:`resolve_auto_model_class` refuses the rest)."""
    admitted = []
    for model_type in sorted(MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES):
        if model_type not in CONFIG_MAPPING:
            continue
        config = _composite_config(model_type)
        if text_config(config) is config:
            continue
        try:
            resolve_auto_model_class(config, text_only=True)
        except ValueError:
            continue
        admitted.append(model_type)
    return admitted


TEXT_ONLY_COMPOSITE_TYPES = _text_only_composite_types()


def test_the_sweep_covers_the_composite_families_the_toolkit_trains():
    assert {"qwen3_5", "qwen3_5_moe", "qwen4_exp"} <= set(TEXT_ONLY_COMPOSITE_TYPES), TEXT_ONLY_COMPOSITE_TYPES


def _applier_resolution(config) -> tuple:
    """The applier Liger resolves for ``config`` and whether it came through the wrapper's text fallback, which
    keeps the fused loss off."""
    resolved, applier = dispatch._resolve_applier(*dispatch._liger_model_types(config))
    return applier, applier is not None and resolved != config.model_type


@pytest.mark.parametrize("model_type", TEXT_ONLY_COMPOSITE_TYPES)
def test_every_text_only_composite_load_applies_liger_to_the_config_its_model_carries(model_type, monkeypatch):
    """Whatever the family, the loader hands Liger the config the text-only build holds."""
    seen: list = []
    monkeypatch.setattr(model_loading, "apply_liger_kernel", lambda config, **kwargs: seen.append(config))
    composite = _composite_config(model_type)

    def meta_build(source, pc, model_class, model_config, dtype, **kwargs):
        with torch.device("meta"):
            return model_class.from_config(model_config)

    model = _load(composite, meta_build, text_only=True, liger_kernel_config=None)

    (liger_config,) = seen
    assert liger_config is text_config(composite)
    assert type(liger_config) is type(model.config), (model_type, type(liger_config), type(model.config))


def test_only_a_family_claimed_through_its_text_tower_alone_changes_its_applier():
    """The text tower's applier differs from the composite spelling's only for a family Liger claims through its
    text tower alone (Flash-Next), whose composite spelling reaches an applier only through the wrapper's text
    fallback; for Qwen3.5 and Qwen3.5-MoE the composite spelling resolves the text tower's applier itself."""
    changed = {}
    for model_type in TEXT_ONLY_COMPOSITE_TYPES:
        composite = _composite_config(model_type)
        composite_spelling, text_tower = _applier_resolution(composite), _applier_resolution(text_config(composite))
        if composite_spelling != text_tower:
            changed[model_type] = (composite_spelling, text_tower)
    assert "qwen4_exp" in changed, "qwen4_exp resolves its applier through its text tower"
    assert not {"qwen3_5", "qwen3_5_moe"} & set(changed), changed
    assert all(composite_spelling[1] and not text_tower[1] for composite_spelling, text_tower in changed.values())


def test_a_load_that_pins_another_class_keeps_the_config_it_read():
    """Only ``AutoModelForCausalLM`` swaps a composite config for its text sub-config; the composite class and a
    plain config pass through."""
    composite = CONFIG_MAPPING["qwen4_exp"]()
    assert model_loading._built_model_config(composite, AutoModelForCausalLM) is text_config(composite)
    assert model_loading._built_model_config(composite, resolve_auto_model_class(composite)) is composite
    plain = text_config(composite)
    assert model_loading._built_model_config(plain, AutoModelForCausalLM) is plain


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
