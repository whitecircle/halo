#!/usr/bin/env python
"""``reject_unsupported_model``: expert sharding vs the model, pure arithmetic over ``config.json`` before the download
and process groups. Stays quiet on shapes it cannot judge, because a false raise blocks a legal run."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch.nn as nn
from transformers import CONFIG_MAPPING

from halo.distributed.parallelism_config import ParallelismConfig
from halo.model_loading.parallel.load_rules import _divisors_up_to, reject_unsupported_model
from halo.models.attention.gpt_oss_sinks import SinksPolicy
from halo.models.moe.config_fields import EXPERT_FFN_WIDTH_FIELDS, resolve_expert_ffn_width
from halo.trainers.distributed_trainer import DistributedTrainerMixin
from tests.common.parallelism import make_parallelism_config

_MOD = "halo.distributed.parallelism_config"
# The field is an expert-count spelling.
_MOE = {"num_experts": 8}


def _config(world: int = 8, local: int | None = None, **kwargs) -> ParallelismConfig:
    """``local`` defaults to ``world`` (single node)."""
    with patch(f"{_MOD}.validate_nvlink_domain_against_fabric"):
        return make_parallelism_config(world_size=world, gpus_per_node=local if local is not None else world, **kwargs)


class _Cfg:
    """``text_config`` mirrors the composite (VLM wrapper) layout."""

    def __init__(self, **fields):
        text = fields.pop("text_config", None)
        for name, value in fields.items():
            setattr(self, name, value)
        if text is not None:
            self.text_config = text


def _reject_unsupported_model(config: ParallelismConfig, model_config) -> None:
    reject_unsupported_model(config, model_config, sinks_policy=SinksPolicy.NEUTRALIZED, wraps_experts=False)


def test_expert_count_not_divisible_by_ep_size_is_rejected():
    """DeepEP's uniform expert-to-rank division cannot express 512 experts over 3."""
    config = _config(8, ep_size=2)
    config.ep_size = 3  # Past __post_init__: the gate must not depend on the topology validators.
    with pytest.raises(ValueError, match="does not divide this model's 512 routed experts"):
        _reject_unsupported_model(config, _Cfg(num_experts=512))


@pytest.mark.parametrize("flag", ["ep_fp32_router", "ep_fp32_experts"])
@pytest.mark.parametrize(
    ("model_type", "wraps_experts"),
    [("qwen3", False), ("gpt_oss", False), ("mixtral", False)],
    ids=["dense", "ep_layers_off", "family_without_an_ep_layer"],
)
def test_an_fp32_ep_master_needs_an_ep_layer(flag, model_type, wraps_experts):
    """Only the EP layers read them, so a run building none would train at the run dtype whatever they say."""
    config = _config(8, use_grouped_gemm=False, **{flag: True})
    with pytest.raises(ValueError, match="this run builds none"):
        reject_unsupported_model(
            config, CONFIG_MAPPING[model_type](), sinks_policy=SinksPolicy.NEUTRALIZED, wraps_experts=wraps_experts
        )


@pytest.mark.parametrize("flag", ["ep_fp32_router", "ep_fp32_experts"])
def test_an_fp32_ep_master_passes_where_the_ep_layers_are_built(flag):
    config = _config(8, **{flag: True})
    reject_unsupported_model(
        config, CONFIG_MAPPING["gpt_oss"](), sinks_policy=SinksPolicy.NEUTRALIZED, wraps_experts=True
    )


def test_train_sinks_on_a_model_without_sinks_is_refused_before_the_load():
    with pytest.raises(ValueError, match="model_type='qwen3'.*Remove train_sinks"):
        reject_unsupported_model(
            _config(8), CONFIG_MAPPING["qwen3"](), sinks_policy=SinksPolicy.TRAINABLE, wraps_experts=False
        )
    reject_unsupported_model(
        _config(8), CONFIG_MAPPING["gpt_oss"](), sinks_policy=SinksPolicy.TRAINABLE, wraps_experts=True
    )


def test_divisible_expert_count_passes():
    _reject_unsupported_model(_config(8, ep_size=8), _Cfg(num_experts=512))


def test_expert_count_is_read_through_the_composite_text_config():
    """The descent goes through HF's ``get_text_config()``, which a hand-rolled stand-in would not reproduce."""
    composite = CONFIG_MAPPING["qwen3_5_moe"]()
    composite.get_text_config().num_experts = 512
    assert not hasattr(composite, "num_experts"), "the fixture must reproduce the composite layout"
    config = _config(8, ep_size=2)
    config.ep_size = 3
    with pytest.raises(ValueError, match="512 routed experts"):
        _reject_unsupported_model(config, composite)


def test_rejection_names_the_ep_sizes_that_divide_the_model():
    """The topology gates narrow the divisors further, so it does not claim they are all legal."""
    config = _config(8, ep_size=2)
    config.ep_size = 3
    with pytest.raises(ValueError, match=r"dividing 512 that fit this job: 2, 4, 8"):
        _reject_unsupported_model(config, _Cfg(num_experts=512))
    with pytest.raises(ValueError, match="topology gates narrow this further"):
        _reject_unsupported_model(config, _Cfg(num_experts=512))


def test_expert_tp_size_must_divide_the_expert_ffn_width():
    """A non-dividing size would make ragged shards."""
    config = _config(8, expert_tp_size=2)
    config.expert_tp_size = 3
    with pytest.raises(ValueError, match="expert FFN width"):
        _reject_unsupported_model(config, _Cfg(num_experts=512, moe_intermediate_size=1024))


def test_dense_and_unknown_shapes_are_left_alone():
    _reject_unsupported_model(_config(8, ep_size=8), _Cfg(hidden_size=4096))
    _reject_unsupported_model(_config(8, ep_size=8), _Cfg(text_config=_Cfg(hidden_size=4096)))


def test_gate_is_inert_without_expert_parallelism():
    _reject_unsupported_model(_config(8), _Cfg(num_experts=7))


def test_divisors_helper_bounds_by_the_job_width_and_skips_one():
    """1 is not an answer to "pick an ep_size"."""
    assert _divisors_up_to(512, 8) == [2, 4, 8]
    assert _divisors_up_to(512, 1024) == [2, 4, 8, 16, 32, 64, 128, 256, 512]
    assert _divisors_up_to(60, 8) == [2, 3, 4, 5, 6]


def test_every_width_spelling_resolves_to_the_declared_per_expert_width():
    """An unknown spelling is not guessed at."""
    for field in EXPERT_FFN_WIDTH_FIELDS:
        assert resolve_expert_ffn_width(_Cfg(**{field: 1024})) == 1024
    assert resolve_expert_ffn_width(_Cfg(ffn_hidden_size=1024)) is None


def test_the_moe_width_wins_over_the_dense_one_on_a_hybrid_config():
    """Registry order enforces this; reading ``intermediate_size`` would let a ragged expert split through."""
    assert resolve_expert_ffn_width(_Cfg(intermediate_size=4096, moe_intermediate_size=512)) == 512


# fp32 dense params beside bf16 experts in one ``fully_shard`` group abort after the whole load, and at wrap time under
# PP some stages would hang.


def test_the_shipped_defaults_are_what_make_the_dtype_triple_reachable():
    """Anti-drift: two toolkit defaults plus one opt-in knob; ``test_experts_fsdp_managed.py`` pins how the defaults
    compose."""
    config = _config(8, ep_size=1)
    assert config.fsdp_shard_ep1_experts is True
    assert config.fp32_non_ep_params is False


def test_the_dtype_triple_is_refused_before_anything_loads():
    config = _config(8, ep_size=1, fp32_non_ep_params=True)
    with pytest.raises(ValueError, match="uniform original parameter dtype") as excinfo:
        _reject_unsupported_model(config, _Cfg(**_MOE))
    message = str(excinfo.value)
    assert "set fsdp_shard_ep1_experts: false" in message, "the single-stage remedy must be the actionable one"
    assert "drop fp32_non_ep_params" in message


@pytest.mark.parametrize(
    ("pc_kwargs", "world", "local"),
    [({"pp_size": 2}, 16, 8), ({"tp_size": 2}, 8, 8), ({"cp_size": 2}, 8, 8)],
    ids=["pp", "tp", "cp"],
)
def test_the_remedy_drops_the_knob_on_every_axis_that_refuses_the_alternative(pc_kwargs, world, local):
    """``fsdp_shard_ep1_experts: false`` is itself refused under PP, TP and CP, so naming it there loops between two
    raises."""
    config = _config(world, local=local, ep_size=1, fp32_non_ep_params=True, **pc_kwargs)
    with pytest.raises(ValueError, match="uniform original parameter dtype") as excinfo:
        _reject_unsupported_model(config, _Cfg(**_MOE))
    message = str(excinfo.value)
    assert "Remedy: drop fp32_non_ep_params" in message
    assert "set fsdp_shard_ep1_experts: false" not in message


def test_distributed_experts_keep_the_upcast():
    """The experts are FSDP-ignored and sync themselves; refusing would ban the validated EP fp32 shape."""
    _reject_unsupported_model(_config(8, ep_size=8, fp32_non_ep_params=True), _Cfg(**_MOE))


def test_opting_the_experts_out_of_fsdp_is_the_documented_escape_hatch():
    config = _config(8, ep_size=1, fp32_non_ep_params=True, fsdp_shard_ep1_experts=False)
    _reject_unsupported_model(config, _Cfg(**_MOE))


def test_a_dense_model_may_still_upcast_its_parameters():
    """A gate keyed on the flags alone would ban ``fp32_non_ep_params`` for every dense run."""
    _reject_unsupported_model(_config(8, ep_size=1, fp32_non_ep_params=True), _Cfg())


def test_the_dtype_gate_is_inert_without_the_knob():
    _reject_unsupported_model(_config(8, ep_size=1), _Cfg(**_MOE))


def _stub(*, experts_fsdp_managed: bool, fp32_non_ep_params: bool, ep_modules, dtype_params=()):
    return SimpleNamespace(
        # What the frozen-trunk half of the exclusions reads: no head here, so no trunk leaves the wrap.
        model=nn.Identity(),
        parallelism_config=SimpleNamespace(
            experts_fsdp_managed=experts_fsdp_managed, fp32_non_ep_params=fp32_non_ep_params
        ),
        _ep_layers=lambda: list(ep_modules),
        _dtype_excluded_params=lambda: list(dtype_params),
    )


def test_the_wrap_time_seam_does_not_re_refuse_the_triple():
    """This walk does not run on every rank under PP, so a raise from it would hang the job."""
    stub = _stub(experts_fsdp_managed=True, fp32_non_ep_params=True, ep_modules=[nn.Linear(4, 4)])

    assert DistributedTrainerMixin._fsdp_excluded_params(stub) == []


def test_fsdp_managed_ep1_experts_hand_the_experts_to_fsdp():
    """Dropping the EP modules from the exclusions is how FSDP comes to own their reduce-scatter."""
    expert = nn.Linear(4, 4)
    stub = _stub(experts_fsdp_managed=True, fp32_non_ep_params=False, ep_modules=[expert])

    assert DistributedTrainerMixin._fsdp_excluded_params(stub) == []


def test_experts_outside_fsdp_stay_excluded():
    """The EP layer syncs its own expert grads."""
    expert = nn.Linear(4, 4)
    stub = _stub(experts_fsdp_managed=False, fp32_non_ep_params=True, ep_modules=[expert])

    excluded = DistributedTrainerMixin._fsdp_excluded_params(stub)

    assert [id(p) for p in excluded] == [id(p) for p in expert.parameters()]


def test_a_parameter_that_is_both_ep_and_dtype_excluded_is_excluded_once():
    """The list feeds FSDP's ``ignored_params``; a duplicate there is a silent hazard."""
    expert = nn.Linear(4, 4)
    stub = _stub(
        experts_fsdp_managed=False, fp32_non_ep_params=False, ep_modules=[expert], dtype_params=[expert.weight]
    )

    excluded = DistributedTrainerMixin._fsdp_excluded_params(stub)

    assert [id(p) for p in excluded] == [id(p) for p in expert.parameters()]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
