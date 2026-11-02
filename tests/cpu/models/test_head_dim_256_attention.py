#!/usr/bin/env python
"""CPU tests for the attention a head_dim-256 or softcapped model resolves to, packed and padded, per GPU.

The failures these gates close are late or silent. FA4's dedicated head_dim-256 kernel asserts in its first
backward on a softcap (Gemma 2) and on a window (Gemma 3), so FA4 on such a model never trains, and multi-head
latent attention (GLM-4.7-Flash) stays off it on FA2 as a precaution. transformers' SDPA drops
``attn_logit_softcapping`` without a word, so a Gemma 2 run on it trains a different model at a plausible loss.
The padded scripts take the loader's own pick on Blackwell, head_dim-256 checkpoints included.

Run: ``python tests/cpu/models/test_head_dim_256_attention.py`` (or ``pytest -m cpu``).
"""

from __future__ import annotations

import contextlib
import types
from unittest.mock import patch

import pytest
import torch
from transformers import PretrainedConfig
from transformers.models.auto.configuration_auto import CONFIG_MAPPING

from halo.entrypoints import model_setup
from halo.entrypoints.model_setup import default_padded_attn_implementation, default_training_rows_attn_implementation
from halo.models.attention import selection as attention_mod
from halo.models.attention.gpt_oss_sinks import SinksPolicy
from halo.models.attention.layout import has_latent_attention, has_sliding_window_layers

FA2, FA3, FA4, SDPA, EAGER = "flash_attention_2", "flash_attention_3", "flash_attention_4", "sdpa", "eager"
_DETECTED = {"blackwell": FA4, "hopper": FA3}


class _RemoteCodeConfig(PretrainedConfig):
    """A config no Auto mapping knows, as a remote-code checkpoint's is (Bailing's)."""

    model_type = "halo_test_remote_code"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size, self.num_attention_heads, self.num_hidden_layers = 2048, 16, 2


def _config(family: str, **overrides):
    return _RemoteCodeConfig() if family == "remote" else CONFIG_MAPPING[family](**overrides)


@contextlib.contextmanager
def _on(arch: str):
    """A host whose GPU is ``arch`` and whose flash builds all import."""
    with (
        patch.object(attention_mod, "_detect_attention_impl", lambda: _DETECTED[arch]),
        patch.object(attention_mod, "is_blackwell_gpu", lambda: arch == "blackwell"),
        patch.object(attention_mod, "is_hopper_gpu", lambda: arch == "hopper"),
        patch.object(attention_mod, "is_flash_attn_2_available", lambda: True),
        patch.object(attention_mod, "_flash_build_imports", lambda _impl: True),
        patch.object(model_setup, "is_flash_attn_2_available", lambda: True),
        patch.object(attention_mod.logger, "warning"),
        patch.object(attention_mod.logger, "info"),
    ):
        yield


def _resolve(config, request, *, dtype=torch.bfloat16) -> str:
    return attention_mod.resolve_attn_implementation(config, request, dtype, sinks_policy=SinksPolicy.NEUTRALIZED)


def _resolved(family: str, arch: str, collator: str) -> str:
    """What the policy load resolves: a packed SFT row with :func:`default_training_rows_attn_implementation`'s
    request, a padded script's with :func:`default_padded_attn_implementation`'s."""
    config = _config(family)
    unset = types.SimpleNamespace(attn_implementation=None)
    with _on(arch):
        if collator == "padded":
            request = default_padded_attn_implementation(unset, sinks_reset=True, checkpoint_config=config)
        else:
            request = default_training_rows_attn_implementation(
                unset,
                packing=True,
                padding_free=False,
                per_device_train_batch_size=1,
                dtype=torch.bfloat16,
                checkpoint_config=config,
            )
        return _resolve(config, request)


# (family, arch) -> (packed, padded). On Blackwell both shapes take the loader's pick: Gemma 2/3 and GLM-4.7-Flash (MLA)
# leave FA4 for FA2 by head width, Gemma 1 and GPT-J (hd256, no softcap, no window) and the hd128 families keep it,
# PaliGemma's prefix-LM mask reaches no flash kernel, and Gemma 4's 512-wide heads fit none; the head_dim-256
# GatedDeltaNet families take FA2 on packed rows only. A softcapped policy's padded batch requests FA2 on Blackwell,
# which trains its softcap, and SDPA on Hopper, which the softcap turns into FA3; a remote-code checkpoint's requests
# SDPA, as every padded script does on Hopper, where GPT-J's class (no SDPA path) turns it into eager. A remote-code
# config's packed row keeps the loader's pick, which its build may refuse.
_EXPECTED = {
    ("gemma2", "blackwell"): (FA2, FA2),
    ("gemma2", "hopper"): (FA3, FA3),
    ("gemma3_text", "blackwell"): (FA2, FA2),
    ("gemma3_text", "hopper"): (FA3, SDPA),
    ("gemma3", "blackwell"): (FA2, FA2),
    ("gemma3", "hopper"): (FA3, SDPA),
    ("gemma", "blackwell"): (FA4, FA4),
    ("gemma", "hopper"): (FA3, SDPA),
    ("paligemma", "blackwell"): (SDPA, SDPA),
    ("paligemma", "hopper"): (SDPA, SDPA),
    ("gptj", "blackwell"): (FA4, FA4),
    ("gptj", "hopper"): (FA3, EAGER),
    ("qwen3", "blackwell"): (FA4, FA4),
    ("qwen3", "hopper"): (FA3, SDPA),
    ("llama", "blackwell"): (FA4, FA4),
    ("llama", "hopper"): (FA3, SDPA),
    ("gpt_oss", "blackwell"): (FA4, FA4),
    ("gpt_oss", "hopper"): (FA3, SDPA),
    ("qwen3_5", "blackwell"): (FA2, FA4),
    ("qwen3_5", "hopper"): (FA3, SDPA),
    ("qwen3_5_moe", "blackwell"): (FA2, FA4),
    ("qwen3_next", "blackwell"): (FA2, FA4),
    ("glm4_moe_lite", "blackwell"): (FA2, FA2),
    ("glm4_moe_lite", "hopper"): (FA3, SDPA),
    ("gemma4_text", "blackwell"): (SDPA, SDPA),
    ("gemma4_text", "hopper"): (SDPA, SDPA),
    ("remote", "blackwell"): (FA4, SDPA),
    ("remote", "hopper"): (FA3, SDPA),
}


@pytest.mark.parametrize("collator", ["packed", "padded"])
@pytest.mark.parametrize(("family", "arch"), list(_EXPECTED), ids=[f"{f}-{a}" for f, a in _EXPECTED])
def test_the_resolved_backend_per_family_arch_and_collator(family, arch, collator):
    packed, padded = _EXPECTED[(family, arch)]
    assert _resolved(family, arch, collator) == (packed if collator == "packed" else padded)


_REQUESTS = [None, SDPA, FA2, FA3, FA4, "flex_attention", "eager"]


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "fp32"])
@pytest.mark.parametrize("arch", ["blackwell", "hopper"])
@pytest.mark.parametrize("request_", _REQUESTS, ids=lambda r: str(r))
@pytest.mark.parametrize("family", ["gemma2", "vaultgemma"])
def test_a_softcapped_model_never_resolves_to_sdpa(family, request_, arch, dtype):
    """Whatever is requested, pinned or defaulted, on any GPU and in fp32 too: SDPA would train without the
    softcap, and nothing downstream says so."""
    with _on(arch):
        resolved = _resolve(_config(family), request_, dtype=dtype)
    assert resolved != SDPA
    if dtype == torch.float32:
        assert resolved in ("flex_attention", "eager"), "FlashAttention is half-precision only"


def test_a_softcapped_sdpa_request_falls_back_to_flash_then_flex():
    """The order is FA3 on Hopper, FA2, flex, eager: both flash kernels carry the softcap at head_dim 256, the arch's
    own FA3 first; flex's compile path is unusable under EP, so it comes last before eager."""
    config = _config("gemma2")
    with _on("hopper"):
        assert _resolve(config, SDPA) == FA3
        with patch.object(attention_mod, "_flash_build_imports", lambda _impl: False):
            assert _resolve(config, SDPA) == FA2
            with patch.object(attention_mod, "is_flash_attn_2_available", lambda: False):
                assert _resolve(config, SDPA) == "flex_attention"
    with _on("blackwell"):
        assert _resolve(config, SDPA) == FA2, "FA3 is a Hopper build: Blackwell falls back to FA2"


def test_a_softcapped_fallback_past_the_flash_head_cap_skips_flash_attention_2():
    """FA2 caps at head_dim 256, so a softcapped model with wider heads falls back to flex, which carries the
    softcap, and never to an FA2 that would raise at the model build."""
    config = _config("gemma2", head_dim=512)
    with _on("hopper"):
        assert attention_mod.validate_attn_implementation(config, SDPA, SinksPolicy.NEUTRALIZED) == "flex_attention"
        assert _resolve(config, SDPA) == "flex_attention"


def test_an_fp32_softcapped_flash_request_is_redirected_to_flex_by_name():
    """The fp32 redirect names flex for a softcapped model; naming SDPA would log a backend the run never takes."""
    with _on("hopper"), patch.object(attention_mod.logger, "warning") as warn:
        assert _resolve(_config("gemma2"), FA3, dtype=torch.float32) == "flex_attention"
    fp32_warnings = [call.args[0] for call in warn.call_args_list if "dtype=float32" in call.args[0]]
    assert fp32_warnings and "attn_implementation='flex_attention'" in fp32_warnings[0]


def test_the_sdpa_refusal_is_keyed_on_the_softcap_not_the_family():
    """Anti-vacuity: the same Gemma 2 config with the softcap off keeps a pinned SDPA."""
    with _on("hopper"):
        assert _resolve(_config("gemma2", attn_logit_softcapping=None), SDPA) == SDPA


@pytest.mark.parametrize("family", ["gemma2", "gemma3_text"])
def test_fa4_survives_at_head_dim_128(family):
    """Anti-vacuity: the demotion is keyed on FA4's dedicated head width. Gemma 2 and Gemma 3 at 27B have
    128-wide heads, whose general kernel takes the softcap and the window."""
    with _on("blackwell"):
        assert _resolve(_config(family, head_dim=128), None) == FA4


def test_the_fa4_demotion_names_the_kernel_limit():
    """The warning is the only record of why a Gemma run lost FA4; it must name the feature."""
    with _on("blackwell"), patch.object(attention_mod.logger, "warning") as warn:
        assert _resolve(_config("gemma2"), FA4) == FA2
    assert any("attention logit softcapping" in call.args[0] for call in warn.call_args_list)


@pytest.mark.parametrize("request_", [None, FA4], ids=["unset", "pinned_fa4"])
@pytest.mark.parametrize("family", ["gemma2", "vaultgemma", "gemma3_text"])
def test_a_deterministic_run_lands_a_softcapped_or_windowed_head_dim_256_model_on_fa2(family, request_, monkeypatch):
    """The dedicated-kernel refusal runs ahead of the deterministic-backward fallback, so a ``full_determinism`` run
    of Gemma 2, VaultGemma or Gemma 3 at head_dim 256 lands on FA2, which takes the softcap and the window, not on
    the SDPA that fallback names."""
    monkeypatch.setenv("FLASH_ATTENTION_DETERMINISTIC", "1")
    config = _config(family)
    # premise: the deterministic-backward fallback's own condition holds
    assert attention_mod.flash_attention_deterministic() and attention_mod.uses_fa4_dedicated_head_dim(config)
    with _on("blackwell"):
        assert _resolve(config, request_) == FA2


def test_a_window_declared_on_the_config_alone_refuses_the_dedicated_kernel():
    """A head_dim-256 model declaring its window on the config alone (Mistral-style, no ``layer_types``) slides on
    every layer, so FA4's dedicated kernel refuses it and the resolver lands on FA2."""
    config = _config("mistral", head_dim=256, sliding_window=32)
    assert getattr(config, "layer_types", None) is None  # premise
    with _on("blackwell"):
        assert _resolve(config, None) == FA2
        assert _resolve(_config("mistral", head_dim=256, sliding_window=None), None) == FA4


class _MlaConfig(PretrainedConfig):
    """An MLA config declaring its query/key width only as the nope and RoPE slices over a 64-wide head_dim."""

    model_type = "halo_test_mla"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size, self.num_attention_heads, self.num_hidden_layers, self.head_dim = 2048, 16, 2, 64
        self.qk_nope_head_dim, self.qk_rope_head_dim = 192, 64


def test_an_mla_query_key_width_reaches_the_dedicated_kernel_gate():
    """FA4 sees the 256-wide query/key an MLA model builds from its two slices, not its declared head_dim."""
    config = _MlaConfig()
    config.attn_logit_softcapping = 30.0
    assert attention_mod.fa4_dedicated_kernel_refusal(config) == "it takes no attention logit softcapping"
    config.qk_nope_head_dim = 64
    assert attention_mod.fa4_dedicated_kernel_refusal(config) is None


# Multi-head latent attention at a 256-wide query/key head in two families' spellings: GLM-4.7-Flash's release
# widths (192 + 64 query/key, 256 value, the config's defaults) and DeepSeek-V3's classes at the same widths.
_LATENT_256 = {
    "glm4_moe_lite": {},
    "deepseek_v3": {"qk_nope_head_dim": 192, "qk_rope_head_dim": 64, "v_head_dim": 256},
}


@pytest.mark.parametrize("deterministic", ["0", "1"])
@pytest.mark.parametrize("request_", [None, FA4], ids=["unset", "pinned_fa4"])
@pytest.mark.parametrize("family", sorted(_LATENT_256))
def test_multi_head_latent_attention_at_head_dim_256_trains_on_fa2(family, request_, deterministic, monkeypatch):
    """A 256-wide latent attention stays on FA2 as a precaution whatever the family, pinned or not, in a
    deterministic run too (not on the SDPA its fallback names), and CP's kernel refuses FA4 on the same verdict."""
    monkeypatch.setenv("FLASH_ATTENTION_DETERMINISTIC", deterministic)
    config = _config(family, **_LATENT_256[family])
    assert attention_mod.uses_fa4_dedicated_head_dim(config)  # premise
    with _on("blackwell"), patch.object(attention_mod.logger, "warning") as warn:
        assert _resolve(config, request_) == FA2
    assert any("multi-head latent attention" in call.args[0] for call in warn.call_args_list)
    assert attention_mod.fa4_dedicated_kernel_refusal(config) == (
        "multi-head latent attention at this width stays on FA2 as a precaution"
    )
    assert not attention_mod.supports_fa4_backward(config)


@pytest.mark.parametrize("family", sorted(_LATENT_256))
def test_multi_head_latent_attention_keeps_fa4_at_other_widths(family):
    """Anti-vacuity: the latent-attention refusal is the dedicated kernel's; at a 128-wide query/key head (64 + 64,
    Mistral4's) FA4's general kernel trains it."""
    config = _config(family, **{"qk_nope_head_dim": 64, "qk_rope_head_dim": 64, "v_head_dim": 128})
    with _on("blackwell"):
        assert _resolve(config, None) == FA4


def test_a_head_dim_256_model_without_a_latent_keeps_fa4():
    """Anti-vacuity: the refusal reads the latent, not the width; Qwen3.5's 256-wide heads keep FA4."""
    config = _config("qwen3_5_text")
    assert attention_mod.uses_fa4_dedicated_head_dim(config) and not has_latent_attention(config)  # premise
    with _on("blackwell"):
        assert _resolve(config, None) == FA4
    assert attention_mod.supports_fa4_backward(config)


def test_window_layers_are_read_off_layer_types():
    assert has_sliding_window_layers(_config("gemma3_text"))
    assert not has_sliding_window_layers(_config("gemma"))
    assert attention_mod.fa4_dedicated_kernel_refusal(_config("gemma3_text")) == "it takes no sliding-window attention"
    assert attention_mod.fa4_dedicated_kernel_refusal(_config("gemma")) is None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
