#!/usr/bin/env python
"""Liger's fused MoE kernel never runs routed experts; ``LigerExperts`` computed the input gradient wrong on
Blackwell until the fix that shipped in liger-kernel 0.8.3, which has not been re-measured."""

from __future__ import annotations

import importlib
import json
import logging
import types

import pytest
from liger_kernel.transformers.auto_model import MODEL_TYPE_TO_APPLY_LIGER_FN
from transformers.loss import loss_utils
from transformers.models.auto.configuration_auto import CONFIG_MAPPING

from halo.distributed.parallelism_config import ParallelismConfig
from halo.model_loading.liger import dispatch
from halo.trainers.distributed_trainer import DistributedTrainerMixin
from tests.common.liger_appliers import liger_applier
from tests.common.models import TINY_QWEN3_MOE_CONFIG, TINY_QWEN35_MOE_CONFIG
from tests.common.repo import probe_findings

# Each carries routed experts and, where the family has them, a shared expert and a dense layer, so the fused GLU has
# somewhere to show up.
_TINY_FAMILIES = {
    "qwen3_moe": {**TINY_QWEN3_MOE_CONFIG, "num_hidden_layers": 2},
    "mixtral": {
        "vocab_size": 256,
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "num_local_experts": 4,
        "num_experts_per_tok": 2,
    },
    "qwen3_5_moe_text": {
        **TINY_QWEN35_MOE_CONFIG,
        "layer_types": ["full_attention"] * TINY_QWEN35_MOE_CONFIG["num_hidden_layers"],
    },
    "qwen3_next": {
        "vocab_size": 256,
        "hidden_size": 64,
        "intermediate_size": 128,
        "moe_intermediate_size": 32,
        "shared_expert_intermediate_size": 32,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 16,
        "num_experts": 4,
        "num_experts_per_tok": 2,
        "layer_types": ["full_attention", "full_attention"],
        "mlp_only_layers": [0],
    },
}

_TOOLKIT_GLU_FAMILIES = {"qwen3_5_moe_text", "qwen3_next"}

# Qwen3-MoE, Qwen3.5-MoE and Qwen3-Next have an EP layer class, so only `ep_size: 1` with grouped GEMM off leaves them
# bare; Mixtral has none.
_UNWRAPPED = {
    "qwen3_moe": {"use_grouped_gemm": False},
    "mixtral": {},
    "qwen3_next": {"use_grouped_gemm": False},
    "qwen3_5_moe_text": {"use_grouped_gemm": False},
}

# Run with a `PAYLOAD = <json>` line prepended.
_MODEL_PROBE = """
import inspect
import json

from liger_kernel.transformers.auto_model import MODEL_TYPE_TO_APPLY_LIGER_FN
from liger_kernel.transformers.swiglu import LigerExperts
from transformers import AutoModelForCausalLM
from transformers.models.auto.configuration_auto import CONFIG_MAPPING

from halo.model_loading.liger import dispatch
from tests.common.liger_appliers import liger_applier

model_type, overrides, needs_ep_layers, toolkit_glu = json.loads(PAYLOAD)
config = CONFIG_MAPPING[model_type](**{**overrides, "attn_implementation": "eager"})
applier = liger_applier(model_type)
upstream_calls = []
if model_type in dispatch._TOOLKIT_LIGER_APPLIERS:
    real_upstream = applier.upstream

    def spy(**kwargs):
        upstream_calls.append(kwargs)
        return real_upstream(**kwargs)

    applier.upstream = spy

applied = dispatch.apply_liger_kernel(config, None, needs_ep_layers=needs_ep_layers)
model = AutoModelForCausalLM.from_config(config)

failures = []
experts = [name for name, module in model.named_modules() if name.endswith(".experts")]
if not experts:
    failures.append("the tiny model built no routed experts; the check below would prove nothing")
liger = [name for name, module in model.named_modules() if isinstance(module, LigerExperts)]
if liger:
    failures.append(f"routed experts replaced by LigerExperts: {liger}")
if applied["swiglu"] is not toolkit_glu:
    failures.append(f"effective swiglu={applied['swiglu']}, expected {toolkit_glu}")
if any(call.get("swiglu") for call in upstream_calls):
    failures.append(f"upstream's applier received swiglu on: {upstream_calls}")
fused = [name for name, module in model.named_modules() if getattr(module, "_halo_glu_mul", None) is not None]
if toolkit_glu and not any("shared_expert" in name for name in fused):
    failures.append(f"the shared expert lost the toolkit's fused GLU (fused: {fused})")
if not toolkit_glu and fused:
    failures.append(f"no toolkit spec serves this family, yet {fused} are fused")

# Anti-vacuity: upstream's own applier with swiglu on does put LigerExperts in the experts' place.
upstream = MODEL_TYPE_TO_APPLY_LIGER_FN[model_type]
upstream(**{name: name == "swiglu" for name in inspect.signature(upstream).parameters if name != "model"})
unguarded = AutoModelForCausalLM.from_config(config)
if not any(isinstance(module, LigerExperts) for module in unguarded.modules()):
    failures.append("upstream's swiglu no longer installs LigerExperts on this family; the check is vacuous")
print("FAILURES:" + "|".join(failures))
"""

_REGISTRY_SWEEP = """
import sys
from liger_kernel.transformers.auto_model import MODEL_TYPE_TO_APPLY_LIGER_FN
from liger_kernel.transformers.monkey_patch import apply_liger_kernel_to_qwen3_moe
from liger_kernel.transformers.swiglu import LigerExperts
from transformers.models.auto.configuration_auto import CONFIG_MAPPING

from halo.model_loading.liger import dispatch
from halo.models.moe.config_fields import has_routed_experts


def liger_experts_bindings():
    return sorted(
        f"{module.__name__}.{name}"
        for module in list(sys.modules.values())
        if getattr(module, "__name__", "").startswith("transformers.models.")
        for name, value in list(vars(module).items())
        if value is LigerExperts
    )


failures = []
swept = []
for model_type in sorted({*MODEL_TYPE_TO_APPLY_LIGER_FN, *dispatch._TOOLKIT_LIGER_APPLIERS}):
    if model_type not in CONFIG_MAPPING:
        continue
    config = CONFIG_MAPPING[model_type]()
    if not has_routed_experts(config):
        continue
    dispatch.apply_liger_kernel(config, None, needs_ep_layers=False)
    swept.append(model_type)
    leaked = liger_experts_bindings()
    if leaked:
        failures.append(f"{model_type}: {leaked}")
        break
if "qwen3_moe" not in swept or "qwen3_next" not in swept:
    failures.append(f"the sweep never reached the families it exists for: {swept}")

# Anti-vacuity: the scan does see a swap when one happens.
apply_liger_kernel_to_qwen3_moe(rope=False, fused_linear_cross_entropy=False, rms_norm=False, swiglu=True)
if not liger_experts_bindings():
    failures.append("the scan misses a LigerExperts swap upstream really made")
print("FAILURES:" + "|".join(failures))
"""


# What HF Trainer does at `train()`: upstream's applier alone, on a model nothing patched at load. Run with a `PAYLOAD
# = <json>` line prepended.
_HF_REAPPLICATION_PROBE = """
import json
import sys
from liger_kernel.transformers.swiglu import LigerExperts
from transformers import AutoModelForCausalLM
from transformers.integrations.liger import apply_liger_kernel
from transformers.models.auto.configuration_auto import CONFIG_MAPPING

from tests.cpu.model_loading.test_liger_routed_experts import _TINY_FAMILIES, _trainer_liger_config

model_type, parallelism = json.loads(PAYLOAD)
config = CONFIG_MAPPING[model_type](**_TINY_FAMILIES[model_type])


def liger_experts(model):
    bound = [
        name
        for name, module in model.named_modules()
        if isinstance(module, LigerExperts)
        or getattr(getattr(module, "forward", None), "__func__", None) is LigerExperts.forward
    ]
    modeling = vars(sys.modules[type(model).__module__])
    return bound + [name for name, value in modeling.items() if value is LigerExperts]


failures = []
model = AutoModelForCausalLM.from_config(config)
apply_liger_kernel(model, _trainer_liger_config(config, None, **parallelism))
if liger_experts(model):
    failures.append(f"HF's re-application of the trainer's config installed LigerExperts on {liger_experts(model)}")

# Anti-vacuity: the same call with the config as the trainer found it does install them.
unguarded = AutoModelForCausalLM.from_config(config)
apply_liger_kernel(unguarded, {"fused_linear_cross_entropy": False})
if not liger_experts(unguarded):
    failures.append("HF's re-application no longer installs LigerExperts on this family; the check is vacuous")
print("FAILURES:" + "|".join(failures))
"""


@pytest.fixture(autouse=True)
def _restore_loss_utils():
    """An upstream-resolved application installs the scoped CE patch."""
    original = loss_utils.nn
    yield
    loss_utils.nn = original


class _RecordingApplier:
    def __init__(self):
        self.calls = []

    def __call__(self, rope=True, cross_entropy=False, fused_linear_cross_entropy=True, rms_norm=True, swiglu=True):
        self.calls.append({"swiglu": swiglu, "rms_norm": rms_norm})


class _SwigluOffApplier(_RecordingApplier):
    """``swiglu=False``: it has no SwiGLU patch for the family (GptOss)."""

    def __call__(self, rope=True, cross_entropy=False, fused_linear_cross_entropy=True, rms_norm=True, swiglu=False):
        super().__call__(rope, cross_entropy, fused_linear_cross_entropy, rms_norm, swiglu)


def _stub_upstream(monkeypatch, model_type: str) -> _RecordingApplier:
    """A delegating applier holds upstream's function from import time, so its ``upstream`` is stubbed too and the
    rebound HF classes are guarded for monkeypatch."""
    applier = _RecordingApplier()
    monkeypatch.setitem(MODEL_TYPE_TO_APPLY_LIGER_FN, model_type, applier)
    toolkit = dispatch._TOOLKIT_LIGER_APPLIERS.get(model_type)
    if toolkit is not None:
        # The delegating applier also hands upstream the model instance, which the recorder does not take.
        def upstream(model=None, **flags):
            return applier(**flags)

        monkeypatch.setattr(toolkit, "upstream", upstream)
        module = importlib.import_module(toolkit.spec.modeling_module)
        for name in (*toolkit.spec.rms_norm, *toolkit.spec.gated_rms_norm, *toolkit.spec.glu_mlp):
            monkeypatch.setattr(module, name, getattr(module, name))
    return applier


def _toolkit_takes_over(model_type: str, role: str) -> bool:
    toolkit = dispatch._TOOLKIT_LIGER_APPLIERS.get(model_type)
    return toolkit is not None and role in toolkit.spec.upstream_off


def _moe_config(model_type: str) -> types.SimpleNamespace:
    return types.SimpleNamespace(model_type=model_type, text_config=None, num_experts=64)


def _tiny_config(model_type: str):
    return CONFIG_MAPPING[model_type](**_TINY_FAMILIES[model_type])


class _InitStopped(Exception):
    """Carries control out of the trainer's shared init after the Liger re-sanitization."""


class _InitHost:
    def _snapshot_model_config_state(self, model_config):
        pass

    def _apply_training_args_overrides(self, training_args, model):
        DistributedTrainerMixin._apply_training_args_overrides(self, training_args, model)
        raise _InitStopped


def _trainer_liger_config(model_config, liger_kernel_config, **parallelism) -> dict:
    args = types.SimpleNamespace(use_liger_kernel=True, liger_kernel_config=liger_kernel_config, output_dir="unused")
    kwargs = {
        "parallelism_config": ParallelismConfig(**parallelism),
        "model": types.SimpleNamespace(config=model_config),
        "args": args,
    }
    with pytest.raises(_InitStopped):
        DistributedTrainerMixin._init_distributed_config(_InitHost(), kwargs)
    return args.liger_kernel_config


@pytest.mark.parametrize(
    ("model_type", "needs_ep_layers"),
    [("qwen3_moe", False), ("mixtral", False), ("mixtral", True)],
    ids=["qwen3_moe-ep1-no-grouped-gemm", "mixtral-ep1-no-grouped-gemm", "mixtral-no-ep-layer-class"],
)
def test_an_unwrapped_moe_hands_upstream_no_swiglu(monkeypatch, model_type, needs_ep_layers):
    applier = _stub_upstream(monkeypatch, model_type)
    config = _moe_config(model_type)
    applied = dispatch.apply_liger_kernel(config, None, needs_ep_layers=needs_ep_layers)
    assert applied["swiglu"] is False
    assert applier.calls[-1]["swiglu"] is False, "upstream's applier still received swiglu on"
    assert config._halo_liger_applied_config["swiglu"] is False, "the effective record must say what ran"
    assert applied["rms_norm"] is True, "only swiglu goes; the other kernels still apply"
    # A role the toolkit took over is withheld from upstream.
    assert applier.calls[-1]["rms_norm"] is not _toolkit_takes_over(model_type, "rms_norm")

    # Anti-vacuity: the same family without experts keeps its fused SwiGLU.
    dense = types.SimpleNamespace(model_type=model_type, text_config=None)
    assert dispatch.apply_liger_kernel(dense, None, needs_ep_layers=needs_ep_layers)["swiglu"] is True


def test_an_explicit_swiglu_request_cannot_bring_liger_experts_back(monkeypatch, caplog):
    """Unlike the inert EP swap, this one trains on a wrong gradient, so a request does not override it."""
    applier = _stub_upstream(monkeypatch, "qwen3_moe")
    config = _moe_config("qwen3_moe")
    with caplog.at_level(logging.WARNING, logger="halo.model_loading.liger.dispatch"):
        applied = dispatch.apply_liger_kernel(config, {"swiglu": True}, needs_ep_layers=False)
    assert applied["swiglu"] is False
    assert applier.calls[-1]["swiglu"] is False
    assert dispatch.trl_reapplication_config(config, applied)["swiglu"] is False
    warnings = [record.getMessage() for record in caplog.records if record.levelno >= logging.WARNING]
    assert any("swiglu was explicitly enabled" in m and "LigerExperts" in m for m in warnings), warnings


@pytest.mark.parametrize(
    ("model_type", "parallelism"),
    [*_UNWRAPPED.items(), ("qwen3_5_moe_text", {})],
    ids=[*_UNWRAPPED, "qwen3_5_moe_text-ep-wrapped"],
)
def test_the_trainer_re_sanitization_forces_the_swap_off_too(model_type, parallelism):
    """For Qwen3-Next and Qwen3.5-MoE HF's re-application never consults the toolkit's spec, so the trainer pins the
    flag off, under the EP layer too."""
    config = _trainer_liger_config(_tiny_config(model_type), {"swiglu": True, "rms_norm": True}, **parallelism)
    assert config["swiglu"] is False, "HF's re-application would hand upstream swiglu on the routed experts"
    # HF's re-application would bind upstream's variant over a role the toolkit took over.
    assert config["rms_norm"] is not _toolkit_takes_over(model_type, "rms_norm")


@pytest.mark.parametrize(("model_type", "role"), [("gpt_oss", "rms_norm"), ("gemma4_text", "geglu")])
def test_the_trainer_re_sanitization_turns_the_taken_over_roles_off(model_type, role):
    """Upstream's instance patch would bind its variant over the family's (the llama-cast norm over GptOss's
    Gemma-cast one)."""
    assert role in liger_applier(model_type).spec.upstream_off, "premise: the spec took it over"
    assert _trainer_liger_config(CONFIG_MAPPING[model_type](), {role: True})[role] is False


@pytest.mark.parametrize(("model_type", "parallelism"), list(_UNWRAPPED.items()), ids=list(_UNWRAPPED))
def test_hf_reapplication_of_the_trainer_config_installs_no_liger_experts(model_type, parallelism):
    payload = json.dumps([model_type, parallelism])
    failures = probe_findings(f"PAYLOAD = {payload!r}\n" + _HF_REAPPLICATION_PROBE, "FAILURES:")
    assert not failures, f"{model_type}:\n" + "\n".join(failures)


def test_the_trainer_re_sanitization_keeps_the_soft_ep_gate():
    """Under the EP layer the swap is inert, so an explicit request stands."""
    config = _tiny_config("qwen3_moe")
    assert _trainer_liger_config(config, {"swiglu": True})["swiglu"] is True
    assert _trainer_liger_config(config, None)["swiglu"] is False


def test_an_applier_with_no_swiglu_patch_is_left_alone(monkeypatch, caplog):
    """Forcing it off would warn about ``LigerExperts`` on a family upstream never gives them to."""
    applier = _SwigluOffApplier()
    monkeypatch.setitem(MODEL_TYPE_TO_APPLY_LIGER_FN, "family_with_swiglu_off", applier)
    config = _moe_config("family_with_swiglu_off")
    assert dispatch.liger_routed_expert_overrides(False, config) == {}
    with caplog.at_level(logging.WARNING, logger="halo.model_loading.liger.dispatch"):
        assert dispatch.apply_liger_kernel(config, {"swiglu": True}, needs_ep_layers=False)["swiglu"] is True
    assert applier.calls[-1]["swiglu"] is True
    assert not [record for record in caplog.records if "LigerExperts" in record.getMessage()]


def test_a_toolkit_entry_that_is_not_a_liger_applier_is_refused(monkeypatch):
    """A stand-in has no spec, so both readers refuse it rather than one reading it as withholding nothing."""
    monkeypatch.setitem(dispatch._TOOLKIT_LIGER_APPLIERS, "qwen3_next", _RecordingApplier())
    with pytest.raises(TypeError, match="not a LigerApplier"):
        dispatch.liger_routed_expert_overrides(False, _moe_config("qwen3_next"))
    with pytest.raises(TypeError, match="not a LigerApplier"):
        dispatch.trl_reapplication_config(_moe_config("qwen3_next"), {"swiglu": True})


def test_under_an_ep_layer_the_soft_gate_still_decides(monkeypatch):
    """The wrapper replaces the experts, so upstream's swap is inert there."""
    _stub_upstream(monkeypatch, "qwen3_moe")
    assert dispatch.liger_routed_expert_overrides(True, _moe_config("qwen3_moe")) == {}
    assert dispatch.apply_liger_kernel(_moe_config("qwen3_moe"), None, needs_ep_layers=True)["swiglu"] is False
    explicit = dispatch.apply_liger_kernel(_moe_config("qwen3_moe"), {"swiglu": True}, needs_ep_layers=True)
    assert explicit["swiglu"] is True

    assert dispatch.liger_ep_disables_fused_glu(True, _moe_config("qwen3_5_moe")) is False


@pytest.mark.parametrize("model_type", ["qwen3_5_moe", "qwen3_5_moe_text", "qwen3_next"])
def test_a_delegating_spec_serving_the_shared_expert_hands_upstream_no_swiglu(model_type):
    """Forcing it off would strip the toolkit's shared expert too."""
    applier = liger_applier(model_type)
    assert "swiglu" in applier.defaults, "the toolkit no longer offers swiglu for its shared expert"
    assert applier.hands_upstream("swiglu") is False
    assert applier.hands_upstream("rms_norm") is True, "upstream still owns the norms"
    for needs_ep_layers in (False, True):
        assert dispatch.liger_routed_expert_overrides(needs_ep_layers, _moe_config(model_type)) == {}


@pytest.mark.parametrize("model_type", sorted(_TINY_FAMILIES))
def test_a_built_model_has_no_liger_experts(model_type):
    """Delegating families keep ``swiglu`` on for the fused shared expert while upstream is handed it off."""
    toolkit_glu = model_type in _TOOLKIT_GLU_FAMILIES
    payload = json.dumps([model_type, _TINY_FAMILIES[model_type], False, toolkit_glu])
    failures = probe_findings(f"PAYLOAD = {payload!r}\n" + _MODEL_PROBE, "FAILURES:")
    assert not failures, f"{model_type}:\n" + "\n".join(failures)


def test_no_registered_moe_family_installs_liger_experts():
    """Swept off the registries, so a MoE family upstream adds later is covered."""
    failures = probe_findings(_REGISTRY_SWEEP, "FAILURES:")
    assert not failures, "LigerExperts reached a family's modeling module:\n" + "\n".join(failures)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
