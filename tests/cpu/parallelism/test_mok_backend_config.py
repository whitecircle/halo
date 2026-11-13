"""``moe_backend: mok`` is refused at config time wherever its kernels cannot run the configuration, and at layer build
for a block whose experts they would compute wrong; the padded token count covers the declared budget."""

from types import SimpleNamespace

import pytest
import torch.nn as nn
from transformers.models.gpt_oss.configuration_gpt_oss import GptOssConfig
from transformers.models.gpt_oss.modeling_gpt_oss import GptOssMLP
from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeSparseMoeBlock

from halo.configs.parallelism_arguments import ParallelismArguments, parallelism_config_from_args
from halo.distributed.expert_parallel import mok
from halo.distributed.expert_parallel.families.gpt_oss import EPGptOssMoELayer
from halo.distributed.expert_parallel.families.qwen3 import EPQwen3MoELayer
from halo.distributed.expert_parallel.moe_layer import layer as layer_module
from halo.kernels.lowp.spec import LowpSpec
from halo.models.moe.expert_lora_spec import ExpertLoraSpec
from halo.trainers.distributed_trainer import DistributedTrainerMixin
from tests.common.ep_stubs import stubbed_rank_ep_layer
from tests.common.parallelism import create_config

BUDGET = {"ep_rows_per_device": 2, "ep_declared_max_length": 4096}


def _mok(**kw):
    return create_config(moe_backend="mok", **{"world_size": 4, "gpus_per_node": 4, "ep_size": 4, **BUDGET, **kw})


@pytest.mark.parametrize("ep_size", [1, 4])
def test_pure_node_local_ep_with_a_budget_is_accepted(ep_size):
    assert _mok(ep_size=ep_size).moe_backend == "mok"


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"world_size": 2, "gpus_per_node": 2, "ep_size": 2}, "expert_parallel_size=2 is not one of"),
        ({"world_size": 16, "gpus_per_node": 8, "ep_size": 16}, "spans NVLink domains"),
        ({"world_size": 8, "gpus_per_node": 8, "ep_size": 4, "expert_tp_size": 2}, "expert_tensor_parallel_size=2"),
        ({"ep_size": 1, "cp_size": 4}, "context_parallel_size=4"),
        ({"expert_lora": ExpertLoraSpec(r=8, alpha=16)}, "expert LoRA"),
        ({"ep_fp32_experts": True}, "fp32_experts"),
        ({"lowp": LowpSpec(precision="fp8")}, "lowp_precision='fp8'"),
        ({"ep_rows_per_device": 0}, "no declared token budget"),
        ({"ep_declared_max_length": 0}, "no declared token budget"),
    ],
)
def test_a_configuration_mok_cannot_run_is_refused(overrides, message):
    with pytest.raises(ValueError, match="moe_backend='mok' cannot run") as raised:
        _mok(**overrides)
    assert message in str(raised.value)


def test_an_unknown_backend_is_refused():
    with pytest.raises(ValueError, match="moe_backend must be one of"):
        create_config(moe_backend="tutel")


@pytest.mark.parametrize(("tokens", "capacity"), [(0, 512), (100, 512), (1000, 1024), (1100, 1536), (8192, 8192)])
def test_the_token_capacity_is_the_budget_padded_to_mok_shapes(tokens, capacity):
    assert mok.mok_token_capacity(tokens) == capacity


@pytest.mark.parametrize(("backend", "rows"), [("mok", 8), ("deepep", 1)])
def test_an_evaluating_run_sizes_mok_buffers_for_its_eval_batch(backend, rows):
    """HF's default eval batch is 8: a buffer sized for one training row would raise at the first evaluation."""
    training = SimpleNamespace(
        per_device_train_batch_size=1, per_device_eval_batch_size=8, max_length=1024, eval_strategy="steps"
    )
    config = parallelism_config_from_args(
        ParallelismArguments(moe_backend=backend), trainer_cls=DistributedTrainerMixin, training_config=training
    )
    assert (config.ep_rows_per_device, config.ep_declared_max_length) == (rows, 1024)


def _qwen3_block(intermediate: int) -> nn.Module:
    return Qwen3MoeSparseMoeBlock(Qwen3MoeConfig(hidden_size=256, moe_intermediate_size=intermediate, num_experts=4))


def _gpt_oss_block() -> nn.Module:
    return GptOssMLP(GptOssConfig(hidden_size=256, intermediate_size=256, num_local_experts=4, num_experts_per_tok=2))


def _build(layer_cls, block, monkeypatch, *, available: bool = True):
    monkeypatch.setattr(layer_module, "mok_available", lambda: available)
    return stubbed_rank_ep_layer(layer_cls, block, 1, 1, 0, grouped=True, moe_backend="mok")


def test_a_supported_block_builds(monkeypatch):
    assert _build(EPQwen3MoELayer, _qwen3_block(256), monkeypatch).ep_config.moe_backend == "mok"


@pytest.mark.parametrize(
    ("layer_cls", "block", "message"),
    [
        (EPGptOssMoELayer, _gpt_oss_block, "EPGptOssMoELayer does not declare SUPPORTS_MOK"),
        (EPQwen3MoELayer, lambda: _qwen3_block(192), "intermediate 192 not multiples of 256"),
    ],
    ids=["biased-clamped-experts", "misaligned-intermediate"],
)
def test_experts_mok_would_compute_wrong_are_refused_at_build(monkeypatch, layer_cls, block, message):
    with pytest.raises(ValueError, match=message):
        _build(layer_cls, block(), monkeypatch)


def test_an_absent_package_is_refused_at_build(monkeypatch):
    with pytest.raises(RuntimeError, match="needs the mok package"):
        _build(EPQwen3MoELayer, _qwen3_block(256), monkeypatch, available=False)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
