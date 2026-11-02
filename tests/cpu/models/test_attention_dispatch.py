#!/usr/bin/env python
"""CPU tests for the attention-backend dispatch in ``halo/models/attention/selection.py``.

The arch and model gates that pick the kernel, all silent when they break: the SM100+ Blackwell
predicate every backend choice shares, the per-model kernel limits ``resolve_attn_implementation``
applies on top, and the validator's refusal to hand back a backend the config rejected.

Run: ``python tests/cpu/models/test_attention_dispatch.py`` (or ``pytest -m cpu``).
"""

from __future__ import annotations

import sys
import types
from collections import defaultdict
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from transformers import Glm4MoeLiteConfig, PretrainedConfig, Qwen3_5MoeTextConfig, Qwen3Config
from transformers import modeling_flash_attention_utils as flash_utils
from transformers.models.auto.configuration_auto import CONFIG_MAPPING
from transformers.models.glm4_moe_lite.modeling_glm4_moe_lite import Glm4MoeLiteAttention
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeAttention

from halo import hardware
from halo.distributed.context_parallel import ulysses_attention as cp_base_layer
from halo.distributed.context_parallel.families import Glm4MoeLiteUlyssesAttention, Qwen3_5MoeUlyssesAttention
from halo.model_loading.parallel import loader as model_loading
from halo.models.attention import selection as attention_mod
from halo.models.attention.gpt_oss_sinks import SinksPolicy
from tests.common.models import TINY_GLM4_MOE_LITE_CONFIG
from tests.common.parallelism import make_parallelism_config

# Every spelling of the families FA4 trains on its dedicated head_dim-256 kernel. Qwen3.6 ships under the 3.5
# types (Qwen/Qwen3.6-35B-A3B declares ``qwen3_5_moe``), so there is no separate 3.6 spelling to cover.
FA4_HEAD_DIM_256_MODEL_TYPES = (
    "qwen3_5",
    "qwen3_5_moe",
    "qwen3_5_moe_text",
    "qwen3_5_text",
    "qwen3_next",
)


def _blackwell():
    """Patches making ``_detect_attention_impl`` see a Blackwell (SM100) GPU."""
    return (
        patch.object(attention_mod.torch.cuda, "is_available", lambda: True),
        patch.object(attention_mod.torch.cuda, "get_device_capability", lambda *a, **k: (10, 0)),
    )


def test_detect_attn_impl_warns_when_present_fa4_is_broken():
    """A BROKEN (not absent) FA4 must warn before degrading to FA2.

    An outer ``except Exception: pass`` would swallow everything but a clean ImportError, so a
    Blackwell box whose flash_attn.cute install is broken silently trains every job on FA2 —
    2-3x slower attention — with nothing in the log to say so.
    """
    is_avail, capability = _blackwell()
    with (
        is_avail,
        capability,
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
    is_avail, capability = _blackwell()
    with (
        is_avail,
        capability,
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

    ``_flash_attn_func`` probes FA3 (Hopper) → FA4 (Blackwell) → FA2; an equality
    test stopping at major 10 makes CP silently take a different kernel for the same run on a device
    the config-level detector calls Blackwell. Probed by making ``flash_attn`` unimportable and raising
    from the FA4-only cache setup — only ``ImportError`` is caught there, so the RuntimeError escaping
    proves the branch was taken.
    """
    cp_base_layer._flash_attn_func.cache_clear()
    try:
        with (
            patch.object(cp_base_layer.torch.cuda, "is_available", lambda: True),
            patch.object(cp_base_layer.torch.cuda, "get_device_capability", lambda *a, **k: capability),
            patch.dict(sys.modules, {"flash_attn": None, "flash_attn_interface": None}),
            patch.object(attention_mod, "ensure_fa4_kernel_cache_env", side_effect=RuntimeError("fa4 probe")),
            pytest.raises(RuntimeError, match="fa4 probe"),
        ):
            cp_base_layer._flash_attn_func()
    finally:
        cp_base_layer._flash_attn_func.cache_clear()


@pytest.mark.parametrize(
    ("capability", "allow_fa4", "preferred"),
    [
        ((9, 0), True, "flash_attention_3"),
        ((10, 0), True, "flash_attention_4"),
        ((11, 0), True, "flash_attention_4"),
        ((10, 0), False, "flash_attention_2"),
        ((8, 0), True, "flash_attention_2"),
    ],
)
def test_the_preferred_flash_backend_follows_the_architecture(capability, allow_fa4, preferred):
    with (
        patch.object(hardware.torch.cuda, "is_available", lambda: True),
        patch.object(hardware.torch.cuda, "get_device_capability", lambda *a, **k: capability),
    ):
        assert attention_mod.preferred_flash_impl(allow_fa4=allow_fa4) == preferred


@pytest.mark.parametrize(
    ("hopper", "fa4_trains", "label"), [(True, True, "flash_attention_3"), (False, False, "flash_attention_2")]
)
def test_the_cp_flex_fallback_labels_the_kernel_ulysses_runs(hopper, fa4_trains, label):
    """The loader's CP flex→flash label and ``_flash_attn_func`` read one preference: FA3 on Hopper, and FA2
    for a model FA4 cannot train on Blackwell, where Ulysses refuses FA4."""
    config = Qwen3Config(
        num_hidden_layers=1, hidden_size=64, intermediate_size=128, num_attention_heads=4, num_key_value_heads=2
    )
    pc = make_parallelism_config(world_size=2, gpus_per_node=2, cp_size=2)
    with (
        patch.object(attention_mod, "is_hopper_gpu", lambda: hopper),
        patch.object(attention_mod, "is_blackwell_gpu", lambda: not hopper),
        patch.object(model_loading, "supports_fa4_backward", lambda model_config: fa4_trains),
        patch.object(model_loading, "logger"),
    ):
        resolved = model_loading._resolve_run_attention(
            config,
            pc,
            "flex_attention",
            torch.bfloat16,
            sinks_policy=SinksPolicy.NEUTRALIZED,
            reset_sinks=True,
            init_from_scratch=False,
        )
    assert resolved == label


def _resolve_on_blackwell(config, requested, dtype=torch.bfloat16, **pc_fields) -> str:
    """The loader's attention for ``config`` on a Blackwell host whose FA4 build imports."""
    with (
        patch.object(attention_mod, "_detect_attention_impl", lambda: "flash_attention_4"),
        patch.object(attention_mod, "is_blackwell_gpu", lambda: True),
        patch.object(attention_mod, "is_hopper_gpu", lambda: False),
        patch.object(attention_mod, "logger"),
    ):
        return model_loading._resolve_run_attention(
            config,
            make_parallelism_config(world_size=2, gpus_per_node=2, **pc_fields),
            requested,
            dtype,
            sinks_policy=SinksPolicy.NEUTRALIZED,
            reset_sinks=True,
            init_from_scratch=False,
        )


# GLM-4.7-Flash's MLA at its release widths: 192 + 64 query/key, 256 value, so FA4 would run its head_dim-256 kernel.
_GLM4_FLASH_MLA = {**TINY_GLM4_MOE_LITE_CONFIG, "qk_nope_head_dim": 192, "qk_rope_head_dim": 64, "v_head_dim": 256}
# Qwen3.5/3.6-MoE's attention at its release widths (16 query and 2 KV heads of 256), one layer of it.
_QWEN3_5_MOE_ATTENTION = {"hidden_size": 256, "num_hidden_layers": 1, "layer_types": ["full_attention"]}


def _glm4_flash() -> Glm4MoeLiteConfig:
    return Glm4MoeLiteConfig(**_GLM4_FLASH_MLA)


def _qwen3_5_moe() -> Qwen3_5MoeTextConfig:
    return Qwen3_5MoeTextConfig(**_QWEN3_5_MOE_ATTENTION)


@pytest.mark.parametrize("requested", [None, "flash_attention_4"])
def test_cp_labels_the_flash_kernel_ulysses_runs_where_fa4_cannot_train(requested, monkeypatch):
    """A deterministic run refuses FA4's head_dim-256 kernel, Ulysses then runs FA2 on Blackwell, so under CP the
    label is FA2; the SDPA the resolver picks without CP is a label Ulysses refuses after the whole load."""
    monkeypatch.setenv("FLASH_ATTENTION_DETERMINISTIC", "1")
    assert _resolve_on_blackwell(_qwen3_5_moe(), requested, cp_size=2) == "flash_attention_2"
    assert _resolve_on_blackwell(_qwen3_5_moe(), requested) == "sdpa"


@pytest.mark.parametrize("cp_size", [1, 2])
@pytest.mark.parametrize("requested", [None, "flash_attention_4"])
def test_a_head_dim_256_family_keeps_fa4_outside_a_deterministic_run(requested, cp_size, monkeypatch):
    monkeypatch.delenv("FLASH_ATTENTION_DETERMINISTIC", raising=False)
    assert _resolve_on_blackwell(_qwen3_5_moe(), requested, cp_size=cp_size) == "flash_attention_4"


@pytest.mark.parametrize("deterministic", ["0", "1"])
@pytest.mark.parametrize("cp_size", [1, 2])
@pytest.mark.parametrize("requested", [None, "flash_attention_4"])
def test_glm_4_7_flash_runs_fa2_with_or_without_cp(requested, cp_size, deterministic, monkeypatch):
    """FA4's dedicated kernel does not train multi-head latent attention, so GLM-4.7-Flash's label is FA2 in every
    run, under CP (the kernel Ulysses then runs) or not, never the SDPA a deterministic run falls back to."""
    monkeypatch.setenv("FLASH_ATTENTION_DETERMINISTIC", deterministic)
    with patch.object(attention_mod, "is_flash_attn_2_available", lambda: True):
        assert _resolve_on_blackwell(_glm4_flash(), requested, cp_size=cp_size) == "flash_attention_2"


@pytest.mark.parametrize(
    ("build", "deterministic", "allow_fa4"),
    [
        ("qwen3_5_moe", "1", False),
        ("qwen3_5_moe", "0", True),
        ("glm4_flash", "1", False),
        ("glm4_flash", "0", False),
    ],
)
def test_the_ulysses_wrapper_refuses_fa4_where_the_loader_does(build, deterministic, allow_fa4, monkeypatch):
    """The wrapper picks its own kernel, so it must read the same verdict as the loader's CP label: a deterministic
    run refuses FA4's head_dim-256 kernel, and GLM-4.7-Flash's latent attention never takes it."""
    monkeypatch.setenv("FLASH_ATTENTION_DETERMINISTIC", deterministic)
    if build == "glm4_flash":
        wrapper = Glm4MoeLiteUlyssesAttention(Glm4MoeLiteAttention(_glm4_flash(), layer_idx=0), None, 2)
    else:
        wrapper = Qwen3_5MoeUlyssesAttention(Qwen3_5MoeAttention(_qwen3_5_moe(), layer_idx=0), None, 2)
    assert wrapper._allow_fa4 is allow_fa4


def test_cp_keeps_the_half_precision_refusal_where_fa4_cannot_train(monkeypatch):
    """Only the FA4 refusal takes the CP kernel's label: an fp32 run has no flash kernel at all, under CP or not."""
    monkeypatch.setenv("FLASH_ATTENTION_DETERMINISTIC", "1")
    assert _resolve_on_blackwell(_qwen3_5_moe(), None, torch.float32, cp_size=2) == "sdpa"
    assert _resolve_on_blackwell(_glm4_flash(), None, torch.float32, cp_size=2) == "sdpa"


def test_detect_attn_impl_on_cpu_host_is_silent_fa2():
    """No CUDA is the ordinary CPU/test path — it must not warn (and must not raise)."""
    with (
        patch.object(attention_mod.torch.cuda, "is_available", lambda: False),
        patch.object(attention_mod.logger, "info"),
        patch.object(attention_mod.logger, "warning") as warn,
    ):
        assert attention_mod._detect_attention_impl() == "flash_attention_2"
    warn.assert_not_called()


def _kernel_module(name: str, *entry_points: str) -> types.ModuleType:
    """A stand-in flash build exposing only ``entry_points``."""
    module = types.ModuleType(name)
    for entry_point in entry_points:

        def kernel(q, k, v):
            raise NotImplementedError

        setattr(module, entry_point, kernel)
    return module


def _hopper():
    return (
        patch.object(attention_mod.torch.cuda, "is_available", lambda: True),
        patch.object(attention_mod.torch.cuda, "get_device_capability", lambda *a, **k: (9, 0)),
        patch.object(attention_mod.logger, "info"),
    )


def test_hopper_detects_fa3_through_the_module_transformers_dispatches():
    """transformers runs flash_attention_3 through ``flash_attn_interface``; a Hopper host carrying only
    that build must detect FA3 rather than settle for FA2."""
    interface = _kernel_module("flash_attn_interface", "flash_attn_func", "flash_attn_varlen_func")
    is_avail, capability, quiet_info = _hopper()
    with (
        is_avail,
        capability,
        quiet_info,
        patch.dict(sys.modules, {"flash_attn_interface": interface, "flash_attn_3": None}),
        patch.object(attention_mod.logger, "warning") as warn,
    ):
        assert attention_mod._detect_attention_impl() == "flash_attention_3"
    warn.assert_not_called()


def test_an_fa3_build_without_the_varlen_kernel_reads_as_absent():
    """A build missing the entry point transformers binds is an absent FA3 (info, FA2, no sink support),
    not a broken one."""
    interface = _kernel_module("flash_attn_interface", "flash_attn_func")
    is_avail, capability, quiet_info = _hopper()
    with (
        is_avail,
        capability,
        quiet_info as info,
        patch.dict(sys.modules, {"flash_attn_interface": interface, "flash_attn_3": None}),
        patch.object(attention_mod.logger, "warning") as warn,
    ):
        assert attention_mod._detect_attention_impl() == "flash_attention_2"
        assert attention_mod._attn_impl_handles_sinks("flash_attention_3") is False
    warn.assert_not_called()
    assert any("not installed" in call.args[0] for call in info.call_args_list)


@pytest.mark.parametrize(
    ("attn_impl", "module_name"), [("flash_attention_4", "flash_attn.cute"), ("flash_attention_2", "flash_attn")]
)
def test_flash_kernel_returns_the_builds_own_entry_points(attn_impl, module_name):
    build = _kernel_module(module_name, "flash_attn_func", "flash_attn_varlen_func")
    with (
        patch.dict(sys.modules, {module_name: build}),
        patch.object(attention_mod, "ensure_fa4_kernel_cache_env", lambda: None),
    ):
        assert attention_mod.flash_kernel(attn_impl) is build.flash_attn_varlen_func
        assert attention_mod.flash_kernel(attn_impl, varlen=False) is build.flash_attn_func


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
    # sinks_policy=None is the re-validation context: nothing has recorded a reset, so sdpa must be
    # refused on the "sinks were never reset" evidence alone.
    with patch.object(attention_mod.logger, "info"), patch.object(attention_mod.logger, "warning"):
        resolved = attention_mod.validate_attn_implementation(config, "sdpa", sinks_policy=None)
    assert resolved != "sdpa", "unmapped sinks config must not resolve to the sink-dropping sdpa"


@pytest.mark.parametrize("model_type", FA4_HEAD_DIM_256_MODEL_TYPES)
def test_a_deterministic_run_demotes_fa4_to_sdpa_at_head_dim_256(model_type, monkeypatch):
    """Each spelling of a head_dim-256 family must leave a deterministic run's resolver on sdpa: that kernel's
    deterministic backward is not validated here, so a run that kept FA4 would train on it. The warning is
    asserted too — a config can reach sdpa for unrelated reasons (fp32, Gemma4's head_dim), and only this
    warning proves the determinism gate is what moved it.
    """
    monkeypatch.setenv("FLASH_ATTENTION_DETERMINISTIC", "1")
    with patch.object(attention_mod.logger, "warning") as warn, patch.object(attention_mod.logger, "info"):
        resolved = attention_mod.resolve_attn_implementation(
            CONFIG_MAPPING[model_type](), "flash_attention_4", torch.bfloat16, sinks_policy=SinksPolicy.NEUTRALIZED
        )
    assert resolved == "sdpa", f"{model_type} kept FA4, whose head_dim-256 deterministic backward is not validated"
    assert any("not validated with a deterministic backward" in call.args[0] for call in warn.call_args_list)


@pytest.mark.parametrize("model_type", FA4_HEAD_DIM_256_MODEL_TYPES)
def test_a_head_dim_256_family_keeps_fa4_in_an_ordinary_run(model_type, monkeypatch):
    monkeypatch.delenv("FLASH_ATTENTION_DETERMINISTIC", raising=False)
    with patch.object(attention_mod.logger, "warning"), patch.object(attention_mod.logger, "info"):
        resolved = attention_mod.resolve_attn_implementation(
            CONFIG_MAPPING[model_type](), "flash_attention_4", torch.bfloat16, sinks_policy=SinksPolicy.NEUTRALIZED
        )
    assert resolved == "flash_attention_4"


@pytest.mark.parametrize("value", ["1", "0", "", "true", "True", "yes", "on", "2"])
def test_the_determinism_read_is_the_one_transformers_hands_its_flash_kernels(value, monkeypatch):
    """The FA4 head_dim-256 demote must follow transformers' own read of the variable (only ``"1"`` turns
    ``deterministic`` on): a looser parse demotes FA4 on runs whose kernel never asks for a deterministic
    backward, a stricter one keeps FA4 into that backward's assertion."""
    monkeypatch.setenv("FLASH_ATTENTION_DETERMINISTIC", value)
    flash_kwargs = flash_utils._process_flash_attention_kwargs(
        query_length=4, key_length=4, is_causal=True, supports_mapping=defaultdict(bool, deterministic=True)
    )
    assert attention_mod.flash_attention_deterministic() is flash_kwargs["deterministic"]


@pytest.mark.parametrize("model_type", ["llama", "qwen3", "qwen3_moe", "glm4_moe"])
def test_fa4_survives_a_deterministic_run_at_other_head_widths(model_type, monkeypatch):
    """Anti-vacuity: the demote is keyed on the head widths, not applied to every deterministic run."""
    monkeypatch.setenv("FLASH_ATTENTION_DETERMINISTIC", "1")
    with patch.object(attention_mod.logger, "warning"), patch.object(attention_mod.logger, "info"):
        resolved = attention_mod.resolve_attn_implementation(
            CONFIG_MAPPING[model_type](), "flash_attention_4", torch.bfloat16, sinks_policy=SinksPolicy.NEUTRALIZED
        )
    assert resolved == "flash_attention_4"


@pytest.mark.parametrize(
    ("model_type", "overrides"),
    [("gemma4", {}), ("gemma4_text", {}), ("llama", {"head_dim": 512})],
    ids=["gemma4", "gemma4_text", "llama_head_dim_512"],
)
@pytest.mark.parametrize("requested", ["flash_attention_2", "flash_attention_4"])
def test_heads_past_the_flash_cap_fall_back_to_sdpa(model_type, overrides, requested):
    """No FlashAttention kernel takes a head wider than 256, so the resolver must leave flash for such
    a model whatever its family: Gemma 4's 512-wide global heads (read per-layer, through the
    multimodal wrapper), and any other config declaring the same width."""
    config = CONFIG_MAPPING[model_type](**overrides)
    with patch.object(attention_mod.logger, "warning"), patch.object(attention_mod.logger, "info"):
        assert (
            attention_mod.resolve_attn_implementation(
                config, requested, torch.bfloat16, sinks_policy=SinksPolicy.NEUTRALIZED
            )
            == "sdpa"
        )


@pytest.mark.parametrize("requested", [None, "flash_attention_2", "sdpa"])
def test_an_attention_free_config_resolves_without_a_head_geometry(requested):
    """Mamba declares neither ``head_dim`` nor ``num_attention_heads``: the head-width gate must read
    that as nothing to reject, not raise, and the validator then lands on the eager the config takes."""
    config = CONFIG_MAPPING["mamba"]()
    assert not attention_mod.has_head_dim_beyond_flash(config)
    with patch.object(attention_mod.logger, "warning"), patch.object(attention_mod.logger, "info"):
        assert (
            attention_mod.resolve_attn_implementation(
                config, requested, torch.bfloat16, sinks_policy=SinksPolicy.NEUTRALIZED
            )
            == "eager"
        )


def test_a_head_at_the_flash_cap_keeps_flash():
    """The cap is inclusive: Qwen3-Next's 256-wide heads are within it."""
    config = CONFIG_MAPPING["qwen3_next"]()
    assert attention_mod.resolve_head_dim(config) == attention_mod.FLASH_MAX_HEAD_DIM
    assert not attention_mod.has_head_dim_beyond_flash(config)
    with patch.object(attention_mod.logger, "warning"), patch.object(attention_mod.logger, "info"):
        resolved = attention_mod.resolve_attn_implementation(
            config, "flash_attention_2", torch.bfloat16, sinks_policy=SinksPolicy.NEUTRALIZED
        )
    assert resolved == "flash_attention_2"


def test_a_config_rejecting_every_candidate_raises():
    """The fallback chain ends at eager; a config whose setter refuses that too has no backend left,
    and handing back the value it just rejected would fail later, inside the model build."""

    class _RejectingConfig:
        model_type = "llama"

        @property
        def _attn_implementation(self):
            return None

        @_attn_implementation.setter
        def _attn_implementation(self, value):
            raise ValueError(f"{value} rejected")

    with pytest.raises(ValueError, match="accepts none of the attention implementations"):
        attention_mod.validate_attn_implementation(_RejectingConfig(), "sdpa")


def test_fa4_demote_reads_the_multimodal_text_tower_head_widths(monkeypatch):
    """A VLM wrapper keeps the text stack's head geometry on its text sub-config only, so a demote that read
    the top level alone would hand a deterministic run's text tower FA4."""
    monkeypatch.setenv("FLASH_ATTENTION_DETERMINISTIC", "1")
    config = SimpleNamespace(
        model_type="some_multimodal_wrapper", get_text_config=lambda: CONFIG_MAPPING["qwen3_5_moe_text"]()
    )
    with patch.object(attention_mod.logger, "warning"), patch.object(attention_mod.logger, "info"):
        resolved = attention_mod.resolve_attn_implementation(
            config, "flash_attention_4", torch.bfloat16, sinks_policy=SinksPolicy.NEUTRALIZED
        )
    assert resolved == "sdpa"


def test_a_config_without_a_causal_lm_class_is_judged_by_its_image_text_class(monkeypatch):
    """A composite with no CausalLM sibling (GLM-5 Next, Mistral4) declares ``_supports_flash_attn`` on its
    conditional-generation class only, so the validator must read that class when the CausalLM mapping misses;
    the conservative default would keep a flash backend transformers refuses at model build."""

    class ImageTextOnlyConfig(PretrainedConfig):
        model_type = "image_text_only"

    class ImageTextOnlyModel:
        _supports_flash_attn = False

    monkeypatch.setattr(
        attention_mod,
        "AutoModelForImageTextToText",
        SimpleNamespace(_model_mapping={ImageTextOnlyConfig: ImageTextOnlyModel}),
    )
    with patch.object(attention_mod.logger, "warning"):
        assert attention_mod.validate_attn_implementation(ImageTextOnlyConfig(), "flash_attention_2") == "sdpa"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
