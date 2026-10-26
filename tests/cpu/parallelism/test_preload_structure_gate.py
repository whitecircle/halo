"""The CP and selective-TP refusals a module tree answers run on a meta-device shell before any weight is read, and
reach the verdicts the CP wrap and the TP sharding reach on the loaded model.

Every tiny MoE family goes through ``load_distributed_model`` with everything before the gate real: the config is
the family's own, the attention resolves on a Blackwell or a Hopper host, and the tokenizer load right after the
gate ends the run. A family the gate admits reaches the tokenizer; one it refuses never does.

Run: ``python tests/cpu/parallelism/test_preload_structure_gate.py`` (or ``pytest -m cpu``).
"""

from __future__ import annotations

import contextlib
import copy
import functools
import os
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from transformers import PreTrainedModel

from halo.distributed.context_parallel.config import CPConfigError
from halo.distributed.context_parallel.model_rules import reject_unsupported_ulysses_model
from halo.distributed.tensor_parallel.parallelize_attention import reject_unshardable_tp_model
from halo.model_loading.parallel import loader as model_loading
from halo.model_loading.parallel.load_rules import reject_unsupported_structure
from halo.model_loading.parallel.model_source import ModelSource
from halo.models.attention import selection
from tests.common.parallelism import make_parallelism_config
from tests.common.tiny_models import TINY_MOE_FAMILIES, TINY_MOE_VLM_FAMILIES, tiny_family_model

ROSTER = {**TINY_MOE_FAMILIES, **TINY_MOE_VLM_FAMILIES}
# The arch each host's detection resolves, as (is_blackwell, is_hopper, detected backend).
HOSTS = {
    "blackwell": (True, False, "flash_attention_4"),
    "hopper": (False, True, "flash_attention_3"),
}
_ACCELERATE_VARS = ("ACCELERATE_MIXED_PRECISION", "ACCELERATE_USE_FSDP")


class _PastTheGate(Exception):
    """Raised by the tokenizer load, the first step after the gate."""


class _UnbuildableFromConfig:
    """A model class whose config-only build fails, as one reading a parameter value in ``__init__`` would."""

    @classmethod
    def from_config(cls, _config, **_kwargs):
        raise TypeError("simulated: __init__ reads a parameter value")


def _tokenizer_load(*_args, **_kwargs):
    raise _PastTheGate


def _weight_read(*_args, **_kwargs):
    raise AssertionError("the loader dispatched to a weight read")


@functools.cache
def _materialized(name: str):
    return tiny_family_model(ROSTER[name])


@contextlib.contextmanager
def _host(host: str):
    """``host``'s attention resolution, and a model build that takes its flash label: transformers refuses one
    whose kernel this CPU box cannot import, which the shell, built on meta, never calls."""
    blackwell, hopper, detected = HOSTS[host]
    with (
        patch.object(selection, "_detect_attention_impl", lambda: detected),
        patch.object(selection, "is_blackwell_gpu", lambda: blackwell),
        patch.object(selection, "is_hopper_gpu", lambda: hopper),
        patch.object(PreTrainedModel, "_check_and_adjust_attn_implementation", lambda _self, label, **_: label),
    ):
        yield


def _gate_outcome(name: str, tmp_path, host: str = "hopper", **pc_fields) -> str | None:
    """The refusal message ``load_distributed_model`` raises for family ``name`` under ``pc_fields``, or ``None``
    when the run gets past the gate; asserts it never reaches a weight read."""
    family = ROSTER[name]
    config = copy.deepcopy(_materialized(name).config)
    env = {k: v for k, v in os.environ.items() if k not in _ACCELERATE_VARS}
    with (
        patch.dict(os.environ, env, clear=True),
        _host(host),
        patch.object(
            model_loading, "resolve_model_source", lambda *_a, **_k: ModelSource("org/x", None, str(tmp_path))
        ),
        patch.object(model_loading, "AutoConfig", SimpleNamespace(from_pretrained=lambda *a, **k: config)),
        patch.object(model_loading, "AutoTokenizer", SimpleNamespace(from_pretrained=_tokenizer_load)),
        patch.object(model_loading, "_dispatch_model_loading", _weight_read),
        # The family patches install process-wide (Mistral4's position_ids plumbing, FA4's varlen seqlen) and
        # would outlive this test; on CPU they hand the label back unchanged, and the gate reads only the label.
        patch.object(model_loading, "apply_family_patches", lambda _config, attn_implementation: attn_implementation),
    ):
        try:
            model_loading.load_distributed_model(
                "org/x",
                parallelism_config=make_parallelism_config(world_size=2, gpus_per_node=2, **pc_fields),
                trust_remote_code=family.trust_remote_code,
                attn_implementation=family.cp_attn_implementation if pc_fields.get("cp_size", 1) > 1 else None,
                model_class=family.load_class,
            )
        except _PastTheGate:
            return None
        except (CPConfigError, ValueError) as refusal:
            return str(refusal)
    raise AssertionError("the load returned without reaching the tokenizer")


def _post_load_outcome(check, *args) -> str | None:
    try:
        check(*args)
    except (CPConfigError, ValueError) as refusal:
        return str(refusal)
    return None


@pytest.mark.parametrize("host", HOSTS)
@pytest.mark.parametrize("name", ROSTER)
def test_cp_refuses_a_family_before_its_load_exactly_where_the_roster_says_ulysses_cannot_run(name, host, tmp_path):
    """The tiny roster's ``ulysses_cp`` is what the CP GPU suites expect of a family; the gate must refuse exactly
    those families, with the message the CP wrap gives the loaded model, on either host's default attention."""
    refusal = _gate_outcome(name, tmp_path, host, cp_size=2)
    assert (refusal is None) == ROSTER[name].ulysses_cp, f"{name} on {host}: {refusal or 'admitted'}"
    if refusal is not None:
        model = _materialized(name)
        assert refusal == _post_load_outcome(reject_unsupported_ulysses_model, model, 2)


@pytest.mark.parametrize("name", ROSTER)
def test_tp_refuses_a_family_before_its_load_with_the_loaded_models_verdict(name, tmp_path):
    """Selective TP on the shell refuses what sharding the loaded model would refuse, before the whole-model load."""
    assert _gate_outcome(name, tmp_path, tp_size=2) == _post_load_outcome(
        reject_unshardable_tp_model, _materialized(name), 2
    )


def test_both_tp_verdicts_occur_on_the_roster(tmp_path):
    """Anti-vacuity: the roster holds families selective TP shards and families it refuses."""
    verdicts = {_post_load_outcome(reject_unshardable_tp_model, _materialized(name), 2) is None for name in ROSTER}
    assert verdicts == {True, False}


@pytest.mark.parametrize(
    "pc_fields", [{"ep_size": 2, "cp_size": 2}, {"ep_size": 2, "tp_size": 2}], ids=["ep_cp", "ep_tp"]
)
def test_the_gate_runs_before_the_expert_parallel_loads(pc_fields, tmp_path):
    """EP+CP and EP+TP reach the gate too, ahead of the lazy EP stream and DeepEP setup."""
    refusal = _gate_outcome("laguna", tmp_path, **pc_fields)
    assert refusal is not None and "'laguna'" in refusal
    assert _gate_outcome("qwen3_moe", tmp_path, **pc_fields) is None


@pytest.mark.parametrize(
    "pc_fields, mode",
    [({"cp_size": 2}, "Context Parallelism"), ({"tp_size": 2}, "Attention-only tensor parallelism")],
    ids=["cp", "tp"],
)
def test_an_unbuildable_shell_names_the_structure_check_and_no_lazy_load_setting(pc_fields, mode, tmp_path):
    """The gate builds its shell whatever ``ep_lazy_loading`` says, so a failed build names the check that needed
    it, and nothing along the exception chain offers a lazy-load setting that would not get past it."""
    with pytest.raises(RuntimeError) as raised:
        reject_unsupported_structure(
            make_parallelism_config(world_size=2, gpus_per_node=2, **pc_fields),
            ModelSource("org/x", None, str(tmp_path)),
            _UnbuildableFromConfig,
            copy.deepcopy(_materialized("qwen3_moe").config),
            "flash_attention_3",
            torch.bfloat16,
            trust_remote_code=False,
        )
    chain, error = [], raised.value
    while error is not None:
        chain.append(str(error))
        error = error.__cause__
    assert mode in chain[0] and "simulated" in chain[0], chain[0]
    assert not any("lazy" in message.lower() for message in chain), chain


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
