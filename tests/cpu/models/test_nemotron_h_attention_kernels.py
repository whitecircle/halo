#!/usr/bin/env python
"""Which attention kernel each of Nemotron-H's towers reaches, at the release's geometry, on a recording FA4: the text
tower's causal 16:1 call on one document or across several, and the vision tower's non-causal call on one image or
across several, each warmed at load. The GPU row (``kernels/test_nemotron_attention_kernels.py``) holds their numerics."""

import json
from unittest import mock

import pytest
import torch
from transformers import (
    AutoConfig,
    AutoModelForImageTextToText,
    NemotronH_Omni_Reasoning_V3,
    NemotronH_Omni_Reasoning_V3_Config,
    PreTrainedModel,
)

from halo.models.attention import selection as attention_mod
from halo.models.attention.backend_setup import patch_transformers_flash_varlen_int_seqlen, warm_fa4_kernels
from halo.models.attention.gpt_oss_sinks import SinksPolicy
from halo.models.attention.layout import AttentionCall, attention_calls
from halo.models.attention.selection import (
    declares_flash_attention,
    fa4_dedicated_kernel_refusal,
    fa4_varlen_refusal,
    has_head_dim_beyond_flash,
    resolve_attn_implementation,
)
from halo.models.families.nemotron_h import apply_nemotron_h_patches
from tests.common.attention_stubs import CompileCountingCute, install_compile_counting_cute
from tests.common.models import (
    NEMOTRON_H_RELEASE_TEXT_CONFIG,
    NEMOTRON_H_RELEASE_VISION_CONFIG,
    TINY_NEMOTRON_H_CONFIG,
    TINY_RADIO_VISION_CONFIG,
)
from tests.common.nemotron_h import nemotron_h_release_config

FA4, FA2, FA3 = "flash_attention_4", "flash_attention_2", "flash_attention_3"
_TEXT_HEADS = NEMOTRON_H_RELEASE_TEXT_CONFIG["num_attention_heads"]
_TEXT_KV_HEADS = NEMOTRON_H_RELEASE_TEXT_CONFIG["num_key_value_heads"]
_TEXT_HEAD_DIM = NEMOTRON_H_RELEASE_TEXT_CONFIG["head_dim"]
_VISION_HIDDEN = NEMOTRON_H_RELEASE_VISION_CONFIG["hidden_size"]
_VISION_HEADS = NEMOTRON_H_RELEASE_VISION_CONFIG["num_attention_heads"]
_VISION_HEAD_DIM = _VISION_HIDDEN // _VISION_HEADS
_IMAGE_TOKEN = 18
_TEXT_IDS = (30, 46)
_GRID = (4, 4)


@pytest.fixture(scope="module", autouse=True)
def _patched():
    apply_nemotron_h_patches()


@pytest.fixture
def cute(monkeypatch) -> CompileCountingCute:
    return install_compile_counting_cute(monkeypatch)


@pytest.fixture
def release_config(tmp_path) -> NemotronH_Omni_Reasoning_V3_Config:
    (tmp_path / "config.json").write_text(json.dumps(nemotron_h_release_config()))
    return AutoConfig.from_pretrained(tmp_path)


def _config() -> NemotronH_Omni_Reasoning_V3_Config:
    """The tiny depth at the release's heads on both towers."""
    return NemotronH_Omni_Reasoning_V3_Config(
        text_config={
            **TINY_NEMOTRON_H_CONFIG,
            "num_attention_heads": _TEXT_HEADS,
            "num_key_value_heads": _TEXT_KV_HEADS,
            "head_dim": _TEXT_HEAD_DIM,
        },
        vision_config={
            **TINY_RADIO_VISION_CONFIG,
            "hidden_size": _VISION_HIDDEN,
            "num_attention_heads": _VISION_HEADS,
        },
        vision_hidden_size=_VISION_HIDDEN,
        projector_hidden_size=TINY_NEMOTRON_H_CONFIG["hidden_size"],
        image_token_id=_IMAGE_TOKEN,
    )


def _on_fa4() -> NemotronH_Omni_Reasoning_V3:
    torch.manual_seed(0)
    config = _config()
    config._attn_implementation = FA4
    with mock.patch.object(PreTrainedModel, "_flash_attn_import_error", lambda self, **_kwargs: None):
        model = AutoModelForImageTextToText.from_config(config)
    return model.to(torch.bfloat16).train()


def _image_row(model, images: int) -> dict[str, torch.Tensor]:
    height, width = _GRID
    ids = torch.arange(*_TEXT_IDS).unsqueeze(0)
    ids[0, 1 : 1 + images * height * width // 4] = _IMAGE_TOKEN
    return {
        "input_ids": ids,
        "pixel_values": torch.randn(images * height * width, model.config.vision_config.patch_dim).to(torch.bfloat16),
        "image_grid_hw": torch.tensor([_GRID] * images),
        "use_cache": False,
    }


def _backward(model, batch) -> None:
    model(**batch).logits.float().sum().backward()


def _keys(cute: CompileCountingCute, *, causal: bool) -> set[tuple]:
    return {
        (key.head_ratio, key.qk_dim, key.varlen) for key in cute.calls if key.causal is causal and not key.backward
    }


def test_the_release_takes_fa4_on_both_towers_when_it_is_asked(release_config):
    config = release_config
    assert declares_flash_attention(config), "a composite hands both towers one label, so both must take it"
    assert fa4_dedicated_kernel_refusal(config) is None
    assert fa4_varlen_refusal(config) is None
    assert not has_head_dim_beyond_flash(config)
    resolved = resolve_attn_implementation(config, FA4, torch.bfloat16, sinks_policy=SinksPolicy.NEUTRALIZED)
    assert resolved == FA4
    assert config.vision_config._attn_implementation == FA4
    assert config.text_config._attn_implementation == FA4


@pytest.mark.parametrize(
    ("blackwell", "fa4_imports", "expected"),
    [(True, True, FA4), (True, False, FA2)],
    ids=["fa4", "fa2_without_a_build"],
)
def test_the_auto_detected_backend_lands_on_the_release(release_config, blackwell, fa4_imports, expected):
    """FA2 is what a Blackwell host without the FA4 build falls back to, and the release takes it too."""

    def kernel(attn_impl, *, varlen=True):
        if attn_impl == FA4 and not fa4_imports:
            raise ImportError("no build")
        return lambda *a, **k: None

    with (
        mock.patch.object(attention_mod.torch.cuda, "is_available", lambda: True),
        mock.patch.object(attention_mod, "is_blackwell_gpu", lambda: blackwell),
        mock.patch.object(attention_mod, "is_hopper_gpu", lambda: False),
        mock.patch.object(attention_mod, "flash_kernel", kernel),
    ):
        resolved = resolve_attn_implementation(
            release_config, None, torch.bfloat16, sinks_policy=SinksPolicy.NEUTRALIZED
        )
    assert resolved == expected


def test_an_fp32_run_leaves_the_half_precision_kernels(release_config):
    resolved = resolve_attn_implementation(release_config, FA4, torch.float32, sinks_policy=SinksPolicy.NEUTRALIZED)
    assert resolved == "sdpa"


def test_the_label_reaches_the_module_of_each_tower(cute):
    model = _on_fa4()
    assert model.vision_model.config._attn_implementation == FA4
    assert model.language_model.model.config._attn_implementation == FA4


def test_one_image_runs_the_vision_tower_non_causal_and_the_text_tower_at_its_ratio(cute):
    model = _on_fa4()
    _backward(model, _image_row(model, images=1))
    assert _keys(cute, causal=False) == {(1, _VISION_HEAD_DIM, False)}
    assert _keys(cute, causal=True) == {(_TEXT_HEADS // _TEXT_KV_HEADS, _TEXT_HEAD_DIM, False)}
    assert _TEXT_HEADS // _TEXT_KV_HEADS == 16


def test_two_images_keep_each_to_itself_through_the_varlen_kernel(cute, monkeypatch):
    patch_transformers_flash_varlen_int_seqlen()
    model = _on_fa4()
    _backward(model, _image_row(model, images=2))
    assert _keys(cute, causal=False) == {(1, _VISION_HEAD_DIM, True)}, "the images would attend across each other"


def test_packed_documents_keep_apart_through_the_text_tower_s_varlen_kernel(cute):
    model = _on_fa4()
    position_ids = torch.cat([torch.arange(9), torch.arange(7)]).unsqueeze(0)
    ids = torch.arange(*_TEXT_IDS).unsqueeze(0)
    _backward(model, {"input_ids": ids, "position_ids": position_ids, "use_cache": False})
    assert _keys(cute, causal=True) == {(_TEXT_HEADS // _TEXT_KV_HEADS, _TEXT_HEAD_DIM, True)}
    assert not _keys(cute, causal=False), "a text row reaches no vision call"


def test_a_padded_row_runs_the_text_tower_s_varlen_kernel(cute):
    model = _on_fa4()
    ids = torch.arange(*_TEXT_IDS).unsqueeze(0)
    mask = torch.ones_like(ids)
    mask[0, :5] = 0
    _backward(model, {"input_ids": ids, "attention_mask": mask, "use_cache": False})
    assert _keys(cute, causal=True) == {(_TEXT_HEADS // _TEXT_KV_HEADS, _TEXT_HEAD_DIM, True)}


@pytest.mark.parametrize("images", [1, 2], ids=["one_image", "two_images"])
def test_a_warmed_composite_compiles_nothing_on_its_first_image(cute, images):
    """A compile on one rank mid-forward lets its peers race into the next collective."""
    model = _on_fa4()
    assert attention_calls(model) == {
        AttentionCall(_TEXT_HEADS, _TEXT_KV_HEADS, _TEXT_HEAD_DIM, _TEXT_HEAD_DIM, None, _TEXT_HEAD_DIM**-0.5, None)
    }
    warm_fa4_kernels(model, dtype=torch.bfloat16, device=torch.device("cpu"))
    warmed = set(cute.compiled)
    assert any(key.head_ratio == 16 and key.qk_dim == _TEXT_HEAD_DIM and key.causal for key in warmed)
    assert any(key.head_ratio == 1 and key.qk_dim == _VISION_HEAD_DIM and not key.causal for key in warmed)
    cute.calls.clear()
    _backward(model, _image_row(model, images=images))
    assert cute.calls, "the row reached no kernel"
    assert set(cute.calls) <= warmed, sorted(set(cute.calls) - warmed, key=str)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
