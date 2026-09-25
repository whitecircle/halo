#!/usr/bin/env python
"""Liger's fused MoE kernel never runs routed experts: Halo's EP wrapper or the model's own implementation does.

Upstream liger-kernel's MoE appliers swap a family's routed-experts class for ``LigerExperts``, that
kernel, under ``swiglu``; liger-kernel 0.8.0's ``LigerExperts`` computes the input gradient wrong on
Blackwell (``tests/gpu/kernels/test_liger_routed_experts.py`` measures it). Two rules keep it off:

* a delegating spec that serves its family's dense and shared-expert MLPs (Qwen3.5-MoE, Qwen3-Next)
  withholds ``swiglu`` from upstream, so the shared expert stays fused while the experts stay stock;
* any other resolution that hands ``swiglu`` to upstream on a MoE whose experts Halo does not wrap
  (``ep_size: 1`` with ``use_grouped_gemm: false``, or a family with no EP layer class) gets it
  forced off, an explicit request included.

Both hold at load and in the trainer's re-sanitization of the config HF Trainer re-applies. Under an EP
wrapper the swap is inert, and the soft EP gate still decides.

The whole-model checks run in a subprocess: applying a real applier rebinds the family's classes for
the rest of the process.

    pytest -m cpu tests/cpu/kernels/test_liger_routed_experts.py
"""

from __future__ import annotations

import json
import logging
import types

import pytest
from accelerate import PartialState
from liger_kernel.transformers.auto_model import MODEL_TYPE_TO_APPLY_LIGER_FN
from transformers.loss import loss_utils
from transformers.models.auto.configuration_auto import CONFIG_MAPPING

import src.distributed.expert_parallel.layers.roster  # noqa: F401 — the EP predicates read a filled registry
from src.distributed.parallelism_config import ParallelismConfig
from src.kernels.liger import orchestrator
from src.trainers.mixins.base import DistributedTrainerMixin
from tests.common.models import TINY_QWEN3_MOE_CONFIG, TINY_QWEN35_MOE_CONFIG
from tests.common.utils import probe_findings

PartialState()  # the orchestrator logs through accelerate's rank-aware logger

# One tiny config per family, each carrying routed experts and — where the family has one — a shared
# expert and a dense layer, so the toolkit's fused GLU has somewhere to show up. All-full-attention
# on the hybrid families: their gated-delta-net layers are not what this checks.
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

# The families whose toolkit spec serves `swiglu` on the dense and shared-expert MLPs itself.
_TOOLKIT_GLU_FAMILIES = {"qwen3_5_moe_text", "qwen3_next"}

# Every way Halo leaves a family's routed experts unwrapped, as ParallelismConfig overrides: Qwen3-MoE and
# Qwen3.5-MoE have an EP layer class, so only `ep_size: 1` with grouped GEMM off leaves them bare; Mixtral
# and Qwen3-Next have none, so the grouped-GEMM default leaves them bare too.
_UNWRAPPED = {
    "qwen3_moe": {"use_grouped_gemm": False},
    "mixtral": {},
    "qwen3_next": {},
    "qwen3_5_moe_text": {"use_grouped_gemm": False},
}

# Run with a `PAYLOAD = <json>` line prepended: the family, its config, needs_ep_wrappers, and whether
# a toolkit spec serves its shared expert.
_MODEL_PROBE = """
import inspect
import json

from accelerate import PartialState

PartialState()
from liger_kernel.transformers.auto_model import MODEL_TYPE_TO_APPLY_LIGER_FN
from liger_kernel.transformers.swiglu import LigerExperts
from transformers import AutoModelForCausalLM
from transformers.models.auto.configuration_auto import CONFIG_MAPPING

import src.distributed.expert_parallel.layers.roster
from src.kernels.liger import orchestrator

model_type, overrides, needs_ep_wrappers, toolkit_glu = json.loads(PAYLOAD)
config = CONFIG_MAPPING[model_type](**{**overrides, "attn_implementation": "eager"})
applier = orchestrator.resolve_liger_applier(model_type)
upstream_calls = []
if model_type in orchestrator._TOOLKIT_LIGER_APPLIERS:
    real_upstream = applier.upstream

    def spy(**kwargs):
        upstream_calls.append(kwargs)
        return real_upstream(**kwargs)

    applier.upstream = spy

applied = orchestrator.apply_liger_kernel(config, None, needs_ep_wrappers=needs_ep_wrappers)
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

from accelerate import PartialState

PartialState()
from liger_kernel.transformers.auto_model import MODEL_TYPE_TO_APPLY_LIGER_FN
from liger_kernel.transformers.monkey_patch import apply_liger_kernel_to_qwen3_moe
from liger_kernel.transformers.swiglu import LigerExperts
from transformers.models.auto.configuration_auto import CONFIG_MAPPING

from src.kernels.liger import orchestrator
from src.models.moe_balancing import config_has_experts


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
for model_type in sorted({*MODEL_TYPE_TO_APPLY_LIGER_FN, *orchestrator._TOOLKIT_LIGER_APPLIERS}):
    if model_type not in CONFIG_MAPPING:
        continue
    config = CONFIG_MAPPING[model_type]()
    if not config_has_experts(config):
        continue
    orchestrator.apply_liger_kernel(config, None, needs_ep_wrappers=False)
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


# What HF Trainer does at `train()`: upstream's applier alone, from the config the trainer mixin leaves,
# on a model built outside Halo's loaders (nothing patched it at load). Run with a `PAYLOAD = <json>`
# line prepended: the family and its ParallelismConfig overrides.
_HF_REAPPLICATION_PROBE = """
import json
import sys

from liger_kernel.transformers.swiglu import LigerExperts
from transformers import AutoModelForCausalLM
from transformers.integrations.liger import apply_liger_kernel
from transformers.models.auto.configuration_auto import CONFIG_MAPPING

from tests.cpu.kernels.test_liger_routed_experts import _TINY_FAMILIES, _trainer_liger_config

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
    """An upstream-resolved application installs the scoped CE patch; keep it out of later tests."""
    original = loss_utils.nn
    yield
    loss_utils.nn = original


class _RecordingApplier:
    """Stands in for an upstream MoE applier; records the kwargs it was called with."""

    def __init__(self):
        self.calls = []

    def __call__(self, rope=True, cross_entropy=False, fused_linear_cross_entropy=True, rms_norm=True, swiglu=True):
        self.calls.append({"swiglu": swiglu, "rms_norm": rms_norm})


class _SwigluOffApplier(_RecordingApplier):
    """An upstream applier declaring ``swiglu=False``: it has no SwiGLU patch for the family (GptOss)."""

    def __call__(self, rope=True, cross_entropy=False, fused_linear_cross_entropy=True, rms_norm=True, swiglu=False):
        super().__call__(rope, cross_entropy, fused_linear_cross_entropy, rms_norm, swiglu)


def _moe_config(model_type: str) -> types.SimpleNamespace:
    return types.SimpleNamespace(model_type=model_type, text_config=None, num_experts=64)


def _tiny_config(model_type: str):
    return CONFIG_MAPPING[model_type](**_TINY_FAMILIES[model_type])


class _InitStopped(Exception):
    """Carries control out of the trainer's shared init once it has re-sanitized the Liger config."""


class _InitHost:
    """A bare trainer host for the shared init, stopped just past the Liger re-sanitization."""

    def _configure_mixed_precision(self, kwargs, training_args):
        pass

    def _should_accelerate_manage_fsdp(self):
        raise _InitStopped


def _trainer_liger_config(model_config, liger_kernel_config, **parallelism) -> dict:
    """The ``liger_kernel_config`` the trainer mixin leaves for HF Trainer to re-apply at ``train()``."""
    args = types.SimpleNamespace(use_liger_kernel=True, liger_kernel_config=liger_kernel_config)
    kwargs = {
        "parallelism_config": ParallelismConfig(**parallelism),
        "model": types.SimpleNamespace(config=model_config),
        "args": args,
    }
    with pytest.raises(_InitStopped):
        DistributedTrainerMixin._init_distributed_config(_InitHost(), kwargs)
    return args.liger_kernel_config


@pytest.mark.parametrize(
    ("model_type", "needs_ep_wrappers"),
    [("qwen3_moe", False), ("mixtral", False), ("mixtral", True)],
    ids=["qwen3_moe-ep1-no-grouped-gemm", "mixtral-ep1-no-grouped-gemm", "mixtral-no-ep-layer-class"],
)
def test_an_unwrapped_moe_hands_upstream_no_swiglu(monkeypatch, model_type, needs_ep_wrappers):
    """Where Halo does not wrap the experts, upstream's ``swiglu`` would put ``LigerExperts`` there.

    Qwen3-MoE is unwrapped only at ``ep_size: 1`` with grouped GEMM off; Mixtral has no EP layer class,
    so it is unwrapped under every configuration, the grouped-GEMM default included.
    """
    applier = _RecordingApplier()
    monkeypatch.setitem(MODEL_TYPE_TO_APPLY_LIGER_FN, model_type, applier)
    config = _moe_config(model_type)
    applied = orchestrator.apply_liger_kernel(config, None, needs_ep_wrappers=needs_ep_wrappers)
    assert applied["swiglu"] is False
    assert applier.calls[-1]["swiglu"] is False, "upstream's applier still received swiglu on"
    assert config._halo_liger_applied_config["swiglu"] is False, "the effective record must say what ran"
    assert applier.calls[-1]["rms_norm"] is True, "only swiglu goes; the other kernels still apply"

    # Anti-vacuity: the same family without experts keeps its fused SwiGLU.
    dense = types.SimpleNamespace(model_type=model_type, text_config=None)
    assert orchestrator.apply_liger_kernel(dense, None, needs_ep_wrappers=needs_ep_wrappers)["swiglu"] is True


def test_an_explicit_swiglu_request_cannot_bring_liger_experts_back(monkeypatch, caplog):
    """Unlike the inert EP swap, this one trains on a wrong gradient, so a request does not override it.

    The pinned re-application config keeps it off too, since TRL re-applies upstream's applier alone.
    """
    applier = _RecordingApplier()
    monkeypatch.setitem(MODEL_TYPE_TO_APPLY_LIGER_FN, "qwen3_moe", applier)
    config = _moe_config("qwen3_moe")
    with caplog.at_level(logging.WARNING, logger="src.kernels.liger.orchestrator"):
        applied = orchestrator.apply_liger_kernel(config, {"swiglu": True}, needs_ep_wrappers=False)
    assert applied["swiglu"] is False
    assert applier.calls[-1]["swiglu"] is False
    assert orchestrator.trl_reapplication_config(config, applied)["swiglu"] is False
    warnings = [record.getMessage() for record in caplog.records if record.levelno >= logging.WARNING]
    assert any("swiglu was explicitly enabled" in m and "LigerExperts" in m for m in warnings), warnings


@pytest.mark.parametrize(
    ("model_type", "parallelism"),
    [*_UNWRAPPED.items(), ("qwen3_5_moe_text", {})],
    ids=[*_UNWRAPPED, "qwen3_5_moe_text-ep-wrapped"],
)
def test_the_trainer_re_sanitization_forces_the_swap_off_too(model_type, parallelism):
    """HF Trainer re-applies Liger from ``liger_kernel_config`` at ``train()`` through upstream's applier
    alone, so the trainer mixin sanitizes it a second time: a model loaded outside Halo's loaders, whose
    config asks for ``swiglu``, must not get ``LigerExperts`` there either.

    Qwen3-MoE and Mixtral lose the flag to the routed-experts rule. For Qwen3-Next and Qwen3.5-MoE the
    toolkit's spec takes ``swiglu`` over from upstream, which HF's re-application never consults, so the
    trainer pins it off, under the EP wrapper too, where upstream would still rebind the experts class
    process-wide.
    """
    config = _trainer_liger_config(_tiny_config(model_type), {"swiglu": True, "rms_norm": True}, **parallelism)
    assert config["swiglu"] is False, "HF's re-application would hand upstream swiglu on the routed experts"
    assert config["rms_norm"] is True, "only swiglu goes; the other kernels still apply"


@pytest.mark.parametrize(("model_type", "role"), [("gpt_oss", "rms_norm"), ("gemma4_text", "geglu")])
def test_the_trainer_re_sanitization_turns_the_taken_over_roles_off(model_type, role):
    """Every role a family's spec takes over from upstream goes off for HF's re-application, a request
    included: upstream's instance patch would bind its variant over the family's (the llama-cast norm over
    GptOss's Gemma-cast one, its GeGLU over Gemma 4's dense MLP).
    """
    assert role in orchestrator.resolve_liger_applier(model_type).spec.upstream_off, "premise: the spec took it over"
    assert _trainer_liger_config(CONFIG_MAPPING[model_type](), {role: True})[role] is False


@pytest.mark.parametrize(("model_type", "parallelism"), list(_UNWRAPPED.items()), ids=list(_UNWRAPPED))
def test_hf_reapplication_of_the_trainer_config_installs_no_liger_experts(model_type, parallelism):
    """The whole ``train()``-time path on a real tiny model: trainer re-sanitization, then
    ``transformers.integrations.liger.apply_liger_kernel`` on a model no Halo loader patched.
    """
    payload = json.dumps([model_type, parallelism])
    failures = probe_findings(f"PAYLOAD = {payload!r}\n" + _HF_REAPPLICATION_PROBE, "FAILURES:")
    assert not failures, f"{model_type}:\n" + "\n".join(failures)


def test_the_trainer_re_sanitization_keeps_the_soft_ep_gate():
    """The trainer site folds the same rules the load site does: under the EP wrapper the swap is inert, so
    an explicit request stands, and without one the gate turns it off.
    """
    config = _tiny_config("qwen3_moe")
    assert _trainer_liger_config(config, {"swiglu": True})["swiglu"] is True
    assert _trainer_liger_config(config, None)["swiglu"] is False


def test_an_applier_with_no_swiglu_patch_is_left_alone(monkeypatch, caplog):
    """A declared ``swiglu=False`` says upstream has no SwiGLU patch there, so there is no swap to stop.

    Forcing it off would warn about ``LigerExperts`` on a family upstream never gives them to.
    """
    applier = _SwigluOffApplier()
    monkeypatch.setitem(MODEL_TYPE_TO_APPLY_LIGER_FN, "family_with_swiglu_off", applier)
    config = _moe_config("family_with_swiglu_off")
    assert orchestrator.liger_routed_expert_overrides(False, config) == {}
    with caplog.at_level(logging.WARNING, logger="src.kernels.liger.orchestrator"):
        assert orchestrator.apply_liger_kernel(config, {"swiglu": True}, needs_ep_wrappers=False)["swiglu"] is True
    assert applier.calls[-1]["swiglu"] is True
    assert not [record for record in caplog.records if "LigerExperts" in record.getMessage()]


def test_a_toolkit_entry_that_is_not_a_liger_applier_is_refused(monkeypatch):
    """Which flags a toolkit applier hands upstream, and which roles it takes over, are read off its spec.

    A stand-in has none to read, so both readers refuse it rather than one of them reading it as
    withholding nothing.
    """
    monkeypatch.setitem(orchestrator._TOOLKIT_LIGER_APPLIERS, "qwen3_next", _RecordingApplier())
    with pytest.raises(TypeError, match="not a LigerApplier"):
        orchestrator.liger_routed_expert_overrides(False, _moe_config("qwen3_next"))
    with pytest.raises(TypeError, match="not a LigerApplier"):
        orchestrator.trl_reapplication_config(_moe_config("qwen3_next"), {"swiglu": True})


def test_under_an_ep_wrapper_the_soft_gate_still_decides(monkeypatch):
    """The wrapper replaces the experts, so upstream's swap is inert there and an explicit request stands."""
    monkeypatch.setitem(MODEL_TYPE_TO_APPLY_LIGER_FN, "qwen3_moe", _RecordingApplier())
    assert orchestrator.liger_routed_expert_overrides(True, _moe_config("qwen3_moe")) == {}
    assert orchestrator.apply_liger_kernel(_moe_config("qwen3_moe"), None, needs_ep_wrappers=True)["swiglu"] is False
    explicit = orchestrator.apply_liger_kernel(_moe_config("qwen3_moe"), {"swiglu": True}, needs_ep_wrappers=True)
    assert explicit["swiglu"] is True

    # A delegating spec keeps its shared-expert GLU under the wrapper, as before.
    assert orchestrator.liger_ep_disables_fused_glu(True, _moe_config("qwen3_5_moe")) is False


@pytest.mark.parametrize("model_type", ["qwen3_5_moe", "qwen3_5_moe_text", "qwen3_next"])
def test_a_delegating_spec_serving_the_shared_expert_hands_upstream_no_swiglu(model_type):
    """The toolkit keeps the flag for its own MLPs; forcing it off would strip the shared expert too."""
    applier = orchestrator.resolve_liger_applier(model_type)
    assert "swiglu" in applier.defaults, "the toolkit no longer offers swiglu for its shared expert"
    assert applier.hands_upstream("swiglu") is False
    assert applier.hands_upstream("rms_norm") is True, "upstream still owns the norms"
    for needs_ep_wrappers in (False, True):
        assert orchestrator.liger_routed_expert_overrides(needs_ep_wrappers, _moe_config(model_type)) == {}


@pytest.mark.parametrize("model_type", sorted(_TINY_FAMILIES))
def test_a_built_model_has_no_liger_experts(model_type):
    """The whole path at ``ep_size: 1`` with grouped GEMM off: resolve, apply, build, inspect.

    Upstream-only families come out with ``swiglu`` off and their experts stock; the delegating ones
    keep ``swiglu`` on for the toolkit's fused shared expert while upstream is handed it off.
    """
    toolkit_glu = model_type in _TOOLKIT_GLU_FAMILIES
    payload = json.dumps([model_type, _TINY_FAMILIES[model_type], False, toolkit_glu])
    failures = probe_findings(f"PAYLOAD = {payload!r}\n" + _MODEL_PROBE, "FAILURES:")
    assert not failures, f"{model_type}:\n" + "\n".join(failures)


def test_no_registered_moe_family_installs_liger_experts():
    """Every family either registry resolves, applied at load with Halo's defaults and no EP wrapper.

    Swept off the registries rather than a list, so a MoE family upstream adds later is covered.
    """
    failures = probe_findings(_REGISTRY_SWEEP, "FAILURES:")
    assert not failures, "LigerExperts reached a family's modeling module:\n" + "\n".join(failures)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
