#!/usr/bin/env python
"""CPU tests for the attention-backend dispatch in ``src/models/patches/attention.py``.

Two things, both silent when they break:

* the ImportError guards. ``patch_flex_attention_compile`` and ``_disable_gpt_oss_fa_fallback`` are
  load-bearing patches (EP flex-attention deadlock/NaN; Blackwell kernel-hub crash). When a
  transformers bump moves the patched symbol they must WARN loudly, naming the vanished symbol and
  the consequence; a ``logger.debug`` or a bare ``return`` there loses the patch in silence.
* the arch and family gates that pick the kernel — the SM100+ Blackwell predicate every backend
  choice shares, and the per-family kernel limits ``resolve_attn_implementation`` applies on top.

Run: ``python tests/cpu/models/test_attention_patch_import_guards.py`` (or ``pytest -m cpu``).
"""

from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from src import hardware
from src.distributed.context_parallel import base_layer as cp_base_layer
from src.distributed.loading import model_loading
from src.models.patches import attention as attention_mod

# Every ``model_type`` transformers actually registers for an FA4-NaN attention stack. Listed
# literally, NOT splatted from the EP wrappers' ``HF_MODEL_TYPES``: those declare which MoE block a
# wrapper handles for checkpoint merging, which is a different question from which attention stack
# NaNs, and the two sets are in no subset relation (qwen3_next and dense qwen3_5 NaN with no EP
# wrapper; ten wrapped families must keep FA4). Qwen3.6 ships under the 3.5 types — Qwen/Qwen3.6-35B-A3B
# declares ``qwen3_5_moe`` — so there is no separate 3.6 spelling to cover.
FA4_NAN_PRONE_MODEL_TYPES = (
    "qwen3_5",
    "qwen3_5_moe",
    "qwen3_5_moe_text",
    "qwen3_5_text",
    "qwen3_next",
    "glm4_moe_lite",
)


def test_flex_attention_patch_warns_loud_on_vanished_symbol():
    # The symbol is resolved by a module-level try/except at import, so "vanished" IS the module
    # global left at None — poisoning sys.modules would come too late to change anything.
    with (
        patch.object(attention_mod, "WrappedFlexAttention", None),
        patch.object(attention_mod.logger, "warning") as warn,
    ):
        attention_mod.patch_flex_attention_compile("unit test")
    assert warn.call_count == 1
    message = warn.call_args.args[0]
    assert "WrappedFlexAttention" in message
    assert "deadlock" in message


def test_gpt_oss_fa_fallback_guard_warns_loud_on_vanished_symbol():
    config = SimpleNamespace(model_type="gpt_oss")
    with (
        patch.object(attention_mod, "GptOssPreTrainedModel", None),
        patch.object(attention_mod.logger, "warning") as warn,
    ):
        attention_mod._disable_gpt_oss_fa_fallback(config)
    assert warn.call_count == 1
    message = warn.call_args.args[0]
    assert "GptOssPreTrainedModel" in message
    assert "Blackwell" in message


def test_import_guards_resolve_the_symbols_on_this_transformers():
    """Anti-vacuity: on the installed transformers both symbols must actually be present.

    Without this the two tests above would keep passing if a bump made the module-level import fail
    permanently — the patched-to-None state would be the real one, and every run would silently take
    the warn-and-no-op branch that leaves the flex-attention compile wrapper and the GptOss
    kernel-hub fallback ACTIVE.
    """
    assert attention_mod.WrappedFlexAttention is not None
    assert attention_mod.flex_attention is not None
    assert attention_mod.GptOssPreTrainedModel is not None


def test_gpt_oss_fa_fallback_guard_silent_for_other_models():
    with patch.object(attention_mod.logger, "warning") as warn:
        attention_mod._disable_gpt_oss_fa_fallback(SimpleNamespace(model_type="qwen3"))
    warn.assert_not_called()


def _blackwell():
    """Patches making ``_detect_attention_impl`` see a Blackwell (SM100) GPU.

    ``logger.info`` is silenced too: it is accelerate's MultiProcessAdapter, which raises unless a
    ``PartialState`` exists — absent in a bare unit-test process.
    """
    return (
        patch.object(attention_mod.torch.cuda, "is_available", lambda: True),
        patch.object(attention_mod.torch.cuda, "get_device_capability", lambda *a, **k: (10, 0)),
        patch.object(attention_mod.logger, "info"),
    )


def test_detect_attn_impl_warns_when_present_fa4_is_broken():
    """A BROKEN (not absent) FA4 must warn before degrading to FA2.

    An outer ``except Exception: pass`` would swallow everything but a clean ImportError, so a
    Blackwell box whose flash_attn.cute install is broken silently trains every job on FA2 —
    2-3x slower attention — with nothing in the log to say so.
    """
    is_avail, capability, quiet_info = _blackwell()
    with (
        is_avail,
        capability,
        quiet_info,
        patch.object(attention_mod, "ensure_fa4_kernel_cache_env", side_effect=RuntimeError("cutlass dsl broken")),
        patch.object(attention_mod.logger, "warning") as warn,
    ):
        assert attention_mod._detect_attention_impl() == "flash_attention_2"
    assert warn.call_count == 1, "a broken-but-present FA4 must warn, not degrade silently"
    message = warn.call_args.args[0]
    assert "flash_attn.cute" in message
    assert "RuntimeError" in message


def test_detect_attn_impl_stays_quiet_when_fa4_is_merely_absent():
    """Anti-noise counterpart: an ABSENT FA4 is an expected build, not a fault — info, never warn."""
    is_avail, capability, quiet_info = _blackwell()
    with (
        is_avail,
        capability,
        quiet_info,
        patch.object(attention_mod, "ensure_fa4_kernel_cache_env", lambda: None),
        patch.dict(sys.modules, {"flash_attn": None, "flash_attn.cute": None}),
        patch.object(attention_mod.logger, "warning") as warn,
    ):
        assert attention_mod._detect_attention_impl() == "flash_attention_2"
    warn.assert_not_called()


@pytest.mark.parametrize(
    ("capability", "is_blackwell", "is_hopper"),
    [((9, 0), False, True), ((10, 0), True, False), ((10, 3), True, False), ((11, 0), True, False)],
)
def test_arch_predicates_classify_the_capability_range(capability, is_blackwell, is_hopper):
    """SM100 and LATER are the FA4 architecture; a bare ``major == 10`` is not that statement.

    Every backend choice keyed on the GPU generation reads these two predicates, so a comparison
    that stops at major 10 makes them disagree about the same device — the attention dispatch
    picking FA4 while the Context-Parallel resolver, which calls ``flash_attn_func`` itself, falls
    back to FA2 for the same run.
    """
    with (
        patch.object(hardware.torch.cuda, "is_available", lambda: True),
        patch.object(hardware.torch.cuda, "get_device_capability", lambda *a, **k: capability),
    ):
        assert hardware.is_blackwell_gpu() is is_blackwell
        assert hardware.is_hopper_gpu() is is_hopper


@pytest.mark.parametrize("capability", [(10, 0), (10, 3), (11, 0)])
def test_fa4_dispatch_reaches_every_sm100_plus_capability(capability):
    """The production detector takes the FA4 branch on the whole Blackwell-and-later range.

    Probed through ``ensure_fa4_kernel_cache_env``, which only the FA4 branch calls.
    """
    with (
        patch.object(attention_mod.torch.cuda, "is_available", lambda: True),
        patch.object(attention_mod.torch.cuda, "get_device_capability", lambda *a, **k: capability),
        patch.object(attention_mod.logger, "info"),
        patch.object(attention_mod.logger, "warning"),
        patch.object(attention_mod, "ensure_fa4_kernel_cache_env", side_effect=RuntimeError("probe")) as probe,
    ):
        attention_mod._detect_attention_impl()
    assert probe.called, f"capability {capability} never reached the FA4 branch"


@pytest.mark.parametrize("capability", [(10, 0), (10, 3), (11, 0)])
def test_context_parallelism_resolves_fa4_over_the_same_capability_range(capability):
    """CP resolves its own ``flash_attn_func`` and must not carry a second SM-major test.

    ``get_flash_attn_func`` probes FA3 (Hopper) → FA2 → FA4 (Blackwell); an equality test stopping at
    major 10 makes CP silently take a different kernel for the same run on a device the config-level
    detector calls Blackwell. Probed by making ``flash_attn`` unimportable
    (so the FA2 probe cannot short-circuit the walk) and raising from the FA4-only cache setup —
    only ``ImportError`` is caught there, so the RuntimeError escaping proves the branch was taken.
    """
    cp_base_layer.get_flash_attn_func.cache_clear()
    try:
        with (
            patch.object(cp_base_layer.torch.cuda, "is_available", lambda: True),
            patch.object(cp_base_layer.torch.cuda, "get_device_capability", lambda *a, **k: capability),
            patch.dict(sys.modules, {"flash_attn": None, "flash_attn_interface": None}),
            patch.object(cp_base_layer, "ensure_fa4_kernel_cache_env", side_effect=RuntimeError("fa4 probe")),
            pytest.raises(RuntimeError, match="fa4 probe"),
        ):
            cp_base_layer.get_flash_attn_func()
    finally:
        cp_base_layer.get_flash_attn_func.cache_clear()


def test_the_loader_reads_the_shared_blackwell_predicate():
    """The CP flex→flash substitution in the loader is the third arch-keyed site; a behavioural
    probe there would need a full ``load_distributed_model``, so pin that it imports the seam."""
    assert model_loading.is_blackwell_gpu is hardware.is_blackwell_gpu


def test_detect_attn_impl_on_cpu_host_is_silent_fa2():
    """No CUDA is the ordinary CPU/test path — it must not warn (and must not raise)."""
    with (
        patch.object(attention_mod.torch.cuda, "is_available", lambda: False),
        patch.object(attention_mod.logger, "info"),
        patch.object(attention_mod.logger, "warning") as warn,
    ):
        assert attention_mod._detect_attention_impl() == "flash_attention_2"
    warn.assert_not_called()


def test_unmapped_config_never_opens_the_sdpa_sinks_escape_hatch():
    """A sinks config absent from the Auto mapping must not reach sdpa.

    ``validate_attn_implementation`` reads ``_supports_sdpa`` off the resolved model class. When
    the lookup misses, the sinks escape hatch must stay CLOSED — an open one would hand a GptOss
    model sdpa, which drops the sink column from the softmax silently.
    """

    class UnmappedSinksConfig(SimpleNamespace):
        model_type = "gpt_oss"

    config = UnmappedSinksConfig()
    assert attention_mod.AutoModelForCausalLM._model_mapping.get(UnmappedSinksConfig, None) is None
    # sinks_reset=None is the re-validation context: nothing has recorded a reset, so sdpa must be
    # refused on the "sinks were never reset" evidence alone.
    with patch.object(attention_mod.logger, "info"), patch.object(attention_mod.logger, "warning"):
        resolved = attention_mod.validate_attn_implementation(config, "sdpa", sinks_reset=None)
    assert resolved != "sdpa", "unmapped sinks config must not resolve to the sink-dropping sdpa"


@pytest.mark.parametrize("model_type", FA4_NAN_PRONE_MODEL_TYPES)
def test_fa4_is_demoted_to_sdpa_for_every_nan_prone_model_type(model_type):
    """Each spelling of an FA4-backward-NaN family must leave the resolver on sdpa.

    The failure is silent by construction: FA4's forward is finite and only the first backward goes
    non-finite, so a spelling that escapes the demote costs a whole run before anything says why.
    The warning is asserted too — a config can reach sdpa for unrelated reasons (fp32, Gemma4's
    head_dim), and only this warning proves the NaN gate is what moved it.
    """
    with patch.object(attention_mod.logger, "warning") as warn, patch.object(attention_mod.logger, "info"):
        resolved = attention_mod.resolve_attn_implementation(
            SimpleNamespace(model_type=model_type), "flash_attention_4", torch.bfloat16
        )
    assert resolved == "sdpa", f"{model_type} kept FA4, whose backward emits NaN gradients"
    assert any("NaN gradients" in call.args[0] for call in warn.call_args_list)


@pytest.mark.parametrize("model_type", ["llama", "qwen3", "qwen3_moe", "glm4_moe"])
def test_fa4_survives_for_families_outside_the_nan_gate(model_type):
    """Anti-vacuity: the demote is keyed on the family, not applied to everything.

    ``qwen3_moe`` and ``glm4_moe`` are the near misses — a prefix widened to ``qwen3``/``glm4``
    would strip FA4 from two families that run it correctly.
    """
    with patch.object(attention_mod.logger, "warning"), patch.object(attention_mod.logger, "info"):
        resolved = attention_mod.resolve_attn_implementation(
            SimpleNamespace(model_type=model_type), "flash_attention_4", torch.bfloat16
        )
    assert resolved == "flash_attention_4"


def test_fa4_demote_reads_the_multimodal_text_tower_model_type():
    """A VLM wrapper keeps the text stack's ``model_type`` on its text sub-config only.

    The wrapper's own type is the vision-bearing composite, so a demote that read the top level
    alone would hand the NaN-prone text tower FA4.
    """
    config = SimpleNamespace(
        model_type="some_multimodal_wrapper",
        get_text_config=lambda: SimpleNamespace(model_type="qwen3_5_moe_text"),
    )
    with patch.object(attention_mod.logger, "warning"), patch.object(attention_mod.logger, "info"):
        resolved = attention_mod.resolve_attn_implementation(config, "flash_attention_4", torch.bfloat16)
    assert resolved == "sdpa"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
