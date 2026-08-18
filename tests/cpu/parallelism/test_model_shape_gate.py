#!/usr/bin/env python
"""``ParallelismConfig.validate_against_model_config`` — expert sharding vs the model, at config time.

``ParallelismConfig`` is model-blind, so an ``ep_size`` that does not divide the expert count would
otherwise survive until ``EPConfig.finalize_expert_assignment`` — after the checkpoint download and
after every EP process group exists — and an ``expert_tp_size`` that does not divide the expert FFN
width until layer construction. Both are pure arithmetic over ``config.json``, so both belong
before any of that.

The same gate carries the ``fp32_non_ep_params`` + FSDP-managed-ep1-experts refusal. That triple puts
fp32 dense parameters next to bf16 expert ones in one ``fully_shard`` group, and torch aborts with a
bare "FSDP expects uniform original parameter dtype" **after the whole model has loaded**. It cannot
live at the wrap-time seam (``_ep_fsdp_ignored_modules``): that walks the LIVE module tree, and under
PP a hybrid stack can leave one stage with no MoE layer at all — the MoE stages would raise while the
dense stages walked on into the wrap, hanging the job instead of failing it. Here it is pure
arithmetic over ``config.json`` and therefore runs identically on every rank.

The gate must also stay quiet on shapes it cannot judge (a dense model, a config that spells its
expert count in none of the registered fields), because a false raise here blocks a legal run.

    python tests/cpu/parallelism/test_model_shape_gate.py
"""

import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch.nn as nn
from transformers import CONFIG_MAPPING

from src.distributed.parallelism_config import ParallelismConfig, _divisors_up_to
from src.models.moe_balancing import EXPERT_FFN_WIDTH_FIELDS, resolve_expert_ffn_shard_width
from src.trainers.mixins.base import DistributedTrainerMixin
from tests.common.parallelism import make_parallelism_config

_MOD = "src.distributed.parallelism_config"
# Enough to make every gate below see a MoE model; the field is one of ROUTER_EXPERT_COUNT_FIELDS.
_MOE = {"num_experts": 8}


def _config(world: int = 8, local: int | None = None, **kwargs) -> ParallelismConfig:
    """``local`` defaults to ``world`` (single node); pass it to build a multi-node shape."""
    with patch(f"{_MOD}.validate_nvlink_domain_against_fabric"):
        return make_parallelism_config(world_size=world, gpus_per_node=local if local is not None else world, **kwargs)


class _Cfg:
    """A stand-in HF config; ``text_config`` mirrors the composite (VLM wrapper) layout."""

    def __init__(self, **fields):
        text = fields.pop("text_config", None)
        for name, value in fields.items():
            setattr(self, name, value)
        if text is not None:
            self.text_config = text


def test_expert_count_not_divisible_by_ep_size_is_rejected():
    """512 experts over ep_size=3: DeepEP's uniform expert-to-rank division cannot express it."""
    config = _config(8, ep_size=2)
    config.ep_size = 3  # past __post_init__: the gate must not depend on the topology validators
    with pytest.raises(ValueError, match="does not divide this model's 512 routed experts"):
        config.validate_against_model_config(_Cfg(num_experts=512))


def test_divisible_expert_count_passes():
    """Anti-over-rejection: the shape the 397B model actually wants must not raise."""
    _config(8, ep_size=8).validate_against_model_config(_Cfg(num_experts=512))


def test_expert_count_is_read_through_the_composite_text_config():
    """The 397B checkpoint is a VLM wrapper — every expert field lives on ``text_config``.

    Reading the top level alone returns nothing, which would make the gate silently inert on exactly
    the model it was written for. A REAL ``Qwen3_5MoeConfig`` is used because the descent goes through
    HF's ``get_text_config()``, which a hand-rolled stand-in would not reproduce.
    """
    composite = CONFIG_MAPPING["qwen3_5_moe"]()
    composite.get_text_config().num_experts = 512
    assert not hasattr(composite, "num_experts"), "the fixture must reproduce the composite layout"
    config = _config(8, ep_size=2)
    config.ep_size = 3
    with pytest.raises(ValueError, match="512 routed experts"):
        config.validate_against_model_config(composite)


def test_rejection_names_the_ep_sizes_that_divide_the_model():
    """A rejection that does not say what to use instead costs another launch to act on.

    It offers the model's own divisors and says the topology gates narrow them further, rather than
    claiming they are all legal — a single-domain job accepts only 2 or the domain width.
    """
    config = _config(8, ep_size=2)
    config.ep_size = 3
    with pytest.raises(ValueError, match=r"dividing 512 that fit this job: 2, 4, 8"):
        config.validate_against_model_config(_Cfg(num_experts=512))
    with pytest.raises(ValueError, match="topology gates narrow this further"):
        config.validate_against_model_config(_Cfg(num_experts=512))


def test_expert_tp_size_must_divide_the_expert_ffn_width():
    """Expert-TP shards ``moe_intermediate_size``; a non-dividing size would make ragged shards."""
    config = _config(8, expert_tp_size=2)
    config.expert_tp_size = 3
    with pytest.raises(ValueError, match="expert FFN width"):
        config.validate_against_model_config(_Cfg(num_experts=512, moe_intermediate_size=1024))


def test_dense_and_unknown_shapes_are_left_alone():
    """No expert fields → nothing to judge. Raising here would block every dense EP-less run."""
    _config(8, ep_size=8).validate_against_model_config(_Cfg(hidden_size=4096))
    _config(8, ep_size=8).validate_against_model_config(_Cfg(text_config=_Cfg(hidden_size=4096)))


def test_gate_is_inert_without_expert_parallelism():
    """ep_size == 1 shards no experts, so an odd expert count is not this gate's business."""
    _config(8).validate_against_model_config(_Cfg(num_experts=7))


def test_divisors_helper_bounds_by_the_job_width_and_skips_one():
    """Suggestions must fit the ranks available, and 1 is not an answer to "pick an ep_size"."""
    assert _divisors_up_to(512, 8) == [2, 4, 8]
    assert _divisors_up_to(512, 1024) == [2, 4, 8, 16, 32, 64, 128, 256, 512]
    assert _divisors_up_to(60, 8) == [2, 3, 4, 5, 6]


def test_every_width_spelling_resolves_to_the_declared_per_expert_width():
    """Each registered spelling names the per-expert dimension directly, and an unknown one is not
    guessed at — the gate must stay silent rather than judge a width it did not read."""
    for field in EXPERT_FFN_WIDTH_FIELDS:
        assert resolve_expert_ffn_shard_width(_Cfg(**{field: 1024})) == 1024
    assert resolve_expert_ffn_shard_width(_Cfg(ffn_hidden_size=1024)) is None


def test_the_moe_width_wins_over_the_dense_one_on_a_hybrid_config():
    """A hybrid dense+MoE config carries both spellings and only the MoE one is what ETP shards, so
    reading ``intermediate_size`` there would gate on the dense FFN and let a ragged expert split
    through. Registry ORDER is what enforces that."""
    assert resolve_expert_ffn_shard_width(_Cfg(intermediate_size=4096, moe_intermediate_size=512)) == 512


# fp32_non_ep_params + FSDP-managed ep1 experts — the mixed-dtype shard group


def test_the_shipped_defaults_are_what_make_the_dtype_triple_reachable():
    """Anti-drift: the illegal triple is two toolkit defaults plus one opt-in knob, so a user hits it
    by setting ``fp32_non_ep_params`` alone. If either default flipped, the gate below would be
    guarding a shape nobody can reach. How the defaults compose into ``experts_fsdp_managed`` across
    every EP/ETP shape is pinned in ``test_experts_fsdp_managed.py``."""
    config = _config(8, ep_size=1)
    assert config.fsdp_shard_ep1_experts is True
    assert config.fp32_non_ep_params is False


def test_the_dtype_triple_is_refused_before_anything_loads():
    config = _config(8, ep_size=1, fp32_non_ep_params=True)
    with pytest.raises(ValueError, match="uniform original parameter dtype") as excinfo:
        config.validate_against_model_config(_Cfg(**_MOE))
    message = str(excinfo.value)
    assert "set fsdp_shard_ep1_experts: false" in message, "the single-stage remedy must be the actionable one"
    assert "drop fp32_non_ep_params" in message


@pytest.mark.parametrize(
    ("pc_kwargs", "world", "local"),
    [({"pp_size": 2}, 16, 8), ({"tp_size": 2}, 8, 8), ({"cp_size": 2}, 8, 8)],
    ids=["pp", "tp", "cp"],
)
def test_the_remedy_drops_the_knob_on_every_axis_that_refuses_the_alternative(pc_kwargs, world, local):
    """``fsdp_shard_ep1_experts: false`` is itself refused under PP, TP and CP, so naming it on those
    axes would send the user around a loop between two raises."""
    config = _config(world, local=local, ep_size=1, fp32_non_ep_params=True, **pc_kwargs)
    with pytest.raises(ValueError, match="uniform original parameter dtype") as excinfo:
        config.validate_against_model_config(_Cfg(**_MOE))
    message = str(excinfo.value)
    assert "Remedy: drop fp32_non_ep_params" in message
    assert "set fsdp_shard_ep1_experts: false" not in message


def test_distributed_experts_keep_the_upcast():
    """``ep_size > 1``: the experts are FSDP-ignored and sync themselves, so nothing shares a shard
    group with the upcast dense parameters. Refusing here would ban the validated EP fp32 shape."""
    _config(8, ep_size=8, fp32_non_ep_params=True).validate_against_model_config(_Cfg(**_MOE))


def test_opting_the_experts_out_of_fsdp_is_the_documented_escape_hatch():
    """The remedy the message prints must actually work — otherwise the raise is a dead end."""
    config = _config(8, ep_size=1, fp32_non_ep_params=True, fsdp_shard_ep1_experts=False)
    config.validate_against_model_config(_Cfg(**_MOE))


def test_a_dense_model_may_still_upcast_its_parameters():
    """No experts ⇒ no mixed-dtype shard group. A gate keyed on the flags alone would ban
    ``fp32_non_ep_params`` for every dense run that left the ep1 default alone."""
    _config(8, ep_size=1, fp32_non_ep_params=True).validate_against_model_config(_Cfg())


def test_the_dtype_gate_is_inert_without_the_knob():
    _config(8, ep_size=1).validate_against_model_config(_Cfg(**_MOE))


# The wrap-time counterpart: ``_ep_fsdp_ignored_modules`` must stay silent


def _stub(*, experts_fsdp_managed: bool, fp32_non_ep_params: bool, ep_modules, incompatible=()):
    """Only what the wrap-time seam reads: the two config flags and the two module finders."""
    return SimpleNamespace(
        parallelism_config=SimpleNamespace(
            experts_fsdp_managed=experts_fsdp_managed, fp32_non_ep_params=fp32_non_ep_params
        ),
        _find_ep_modules=lambda: list(ep_modules),
        _find_fsdp_incompatible_modules=lambda: list(incompatible),
    )


def test_the_wrap_time_seam_does_not_re_refuse_the_triple():
    """The counterpart of the config-time raise: this rank-dependent walk must stay silent, because
    under PP it does not run on every rank and a raise from it would hang the job rather than end
    it. A guard re-added here is what this pins against."""
    stub = _stub(experts_fsdp_managed=True, fp32_non_ep_params=True, ep_modules=[nn.Linear(4, 4)])

    assert DistributedTrainerMixin._ep_fsdp_ignored_modules(stub) == ([], [], [])


def test_fsdp_managed_ep1_experts_hand_the_experts_to_fsdp():
    """The accepted half of the same branch: the EP modules are dropped from the ignored set (that IS
    how FSDP comes to own their reduce-scatter), so empty lists here are the behaviour."""
    expert = nn.Linear(4, 4)
    stub = _stub(experts_fsdp_managed=True, fp32_non_ep_params=False, ep_modules=[expert])

    ep_modules, dtype_incompatible, merged = DistributedTrainerMixin._ep_fsdp_ignored_modules(stub)

    assert ep_modules == []
    assert dtype_incompatible == []
    assert expert not in merged


def test_experts_outside_fsdp_stay_ignored():
    """``ep_group_size > 1`` (real EP, and pure ETP): the EP layer syncs its own expert grads, so the
    experts must remain in FSDP's ignored set."""
    expert = nn.Linear(4, 4)
    stub = _stub(experts_fsdp_managed=False, fp32_non_ep_params=True, ep_modules=[expert])

    ep_modules, _dtype_incompatible, merged = DistributedTrainerMixin._ep_fsdp_ignored_modules(stub)

    assert ep_modules == [expert]
    assert merged == [expert]


def test_a_module_that_is_both_ep_and_dtype_incompatible_is_ignored_once():
    """``merged`` feeds FSDP's ``ignored_params``; a duplicated module there is a silent hazard."""
    expert = nn.Linear(4, 4)
    stub = _stub(experts_fsdp_managed=False, fp32_non_ep_params=False, ep_modules=[expert], incompatible=[expert])

    _ep_modules, dtype_incompatible, merged = DistributedTrainerMixin._ep_fsdp_ignored_modules(stub)

    assert dtype_incompatible == [expert]
    assert merged == [expert]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
