"""Structural validator unit tests; native runtime equivalence lives in test_tp_lora_native."""

from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Replicate, Shard, distribute_tensor

from src.distributed.tensor_parallel import lora
from tests.common.distributed import fake_process_group_mesh


class _DenseModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = {"model_type": "custom"}
        self.embed_tokens = nn.Embedding(8, 8)
        self.q_proj = nn.Linear(8, 8, bias=False)
        self.o_proj = nn.Linear(8, 8, bias=False)
        self.lm_head = nn.Linear(8, 8, bias=False)

    def get_input_embeddings(self):
        return self.embed_tokens

    def get_output_embeddings(self):
        return self.lm_head


class _Style:
    """Controlled API/closure fixture, never executed as a native TP computation."""

    def validate_param(self, *args, **kwargs):
        del args, kwargs

    def shard_param(self, *args, **kwargs):
        del args, kwargs

    def install_forward(self, module, mesh):
        original_forward = module.forward

        def tp_forward(*args, **kwargs):
            self.validate_param(module, "weight", mesh)
            return original_forward(*args, **kwargs)

        module.forward = tp_forward


def _config(**kwargs):
    return LoraConfig(r=4, lora_alpha=8, lora_dropout=0.0, target_modules=["q_proj", "o_proj"], **kwargs)


def _set_native_metadata(model, mesh):
    model._device_mesh = mesh
    model.tp_plan = {"q_proj": "colwise", "o_proj": "rowwise", "lm_head": "colwise_gather_output"}
    model.config = SimpleNamespace(distributed_config=SimpleNamespace(tp_size=mesh.size(), fsdp_size=1, pp_size=1))


def _place(weight, mesh, placement):
    return nn.Parameter(
        distribute_tensor(weight.detach(), mesh, [placement], src_data_rank=None),
        requires_grad=weight.requires_grad,
    )


@pytest.fixture
def native_api(monkeypatch):
    styles = {name: _Style() for name in ("colwise", "rowwise")}
    api = SimpleNamespace(
        ALL_PARALLEL_STYLES=styles,
        _get_parameter_tp_plan=lambda name, plan, *, is_weight=False: plan.get(name),
    )
    monkeypatch.setattr(lora, "hf_tp", api)
    monkeypatch.setattr(lora.peft_import_utils, "is_transformers_dtensor_tp", True, raising=False)
    monkeypatch.setattr(lora.peft_lora_model, "add_lora_tp_hooks_dtensor", lambda: None, raising=False)
    return api


@pytest.fixture
def mesh():
    with fake_process_group_mesh(0, 2):
        yield init_device_mesh("cpu", (2,), mesh_dim_names=("tp",))


@pytest.fixture
def preflight_model(mesh, native_api):
    del native_api
    model = _DenseModel()
    _set_native_metadata(model, mesh)
    model.q_proj.weight = _place(model.q_proj.weight, mesh, Shard(0))
    model.o_proj.weight = _place(model.o_proj.weight, mesh, Shard(1))
    return model


@pytest.fixture
def postflight_model(mesh, native_api):
    # Build PEFT on the unsharded model, then construct the validator's expected structural state.
    # This does not make claims about native initialization or its forward/backward collectives.
    model = get_peft_model(_DenseModel(), _config())
    base_model = model.get_base_model()
    _set_native_metadata(base_model, mesh)
    for name, style in (("q_proj", "colwise"), ("o_proj", "rowwise")):
        layer = getattr(base_model, name)
        base = layer.get_base_layer()
        dim = 0 if style == "colwise" else 1
        base.weight = _place(base.weight, mesh, Shard(dim))
        base._hf_tp_plan = style
        base._hf_device_mesh = mesh
        factor = layer.lora_B["default"] if style == "colwise" else layer.lora_A["default"]
        factor.weight = _place(factor.weight, mesh, Shard(dim))
        native_api.ALL_PARALLEL_STYLES[style].install_forward(factor, mesh)
    return model


def test_preflight_accepts_native_targets_without_mutation(preflight_model):
    config = _config()
    before_config = copy.deepcopy(vars(config))
    before_params = {name: id(param) for name, param in preflight_model.named_parameters()}
    lora.validate_native_tp_lora_config(preflight_model, config, tp_size=2)
    assert vars(config) == before_config
    assert {name: id(param) for name, param in preflight_model.named_parameters()} == before_params


def test_preflight_resolves_all_linear_on_a_copy(preflight_model):
    config = _config()
    config.target_modules = "all-linear"
    config.exclude_modules = {"lm_head"}
    lora.validate_native_tp_lora_config(preflight_model, config)
    assert config.target_modules == "all-linear"
    assert config.exclude_modules == {"lm_head"}


@pytest.mark.parametrize("initialization", [False, "gaussian", "orthogonal", "pissa"])
def test_preflight_refuses_unvalidated_initializers(preflight_model, initialization):
    with pytest.raises(ValueError, match="init_lora_weights=True"):
        lora.validate_native_tp_lora_config(preflight_model, _config(init_lora_weights=initialization))


@pytest.mark.parametrize(
    "field,value",
    [
        ("lora_dropout", 0.1),
        ("bias", "all"),
        ("lora_bias", True),
        ("use_dora", True),
        ("fan_in_fan_out", True),
        ("modules_to_save", ["lm_head"]),
        ("target_parameters", ["q_proj.weight"]),
        ("rank_pattern", {"q_proj": 8}),
        ("alpha_pattern", {"q_proj": 16}),
        ("layer_replication", [(0, 1)]),
        ("trainable_token_indices", [1]),
        ("_custom_modules", {nn.Linear: nn.Linear}),
        ("future_variant", True),
    ],
)
def test_preflight_refuses_unsupported_options(preflight_model, field, value):
    config = _config()
    setattr(config, field, value)
    with pytest.raises(ValueError, match=field):
        lora.validate_native_tp_lora_config(preflight_model, config)


@pytest.mark.parametrize("style", [None, "colwise_gather_output", "packed_colwise", "rowwise_rep"])
def test_preflight_refuses_unplanned_or_unsupported_tp_styles(preflight_model, style):
    preflight_model.tp_plan["q_proj"] = style
    with pytest.raises(ValueError, match="unsupported TP style"):
        lora.validate_native_tp_lora_config(preflight_model, _config())


def test_preflight_refuses_plain_sharded_base(preflight_model):
    preflight_model.q_proj.weight = nn.Parameter(preflight_model.q_proj.weight.to_local().clone())
    with pytest.raises(ValueError, match="global base Linear geometry"):
        lora.validate_native_tp_lora_config(preflight_model, _config())


def test_preflight_refuses_wrong_base_placement(preflight_model, mesh):
    preflight_model.q_proj.weight = _place(nn.Parameter(torch.zeros(8, 8)), mesh, Shard(1))
    with pytest.raises(ValueError, match="incorrect TP placement"):
        lora.validate_native_tp_lora_config(preflight_model, _config())


def test_preflight_refuses_no_matching_targets(preflight_model):
    config = _config()
    config.target_modules = {"does_not_exist"}
    with pytest.raises(ValueError, match="matched no supported"):
        lora.validate_native_tp_lora_config(preflight_model, config)


def test_preflight_refuses_an_output_head_even_with_literal_colwise_plan(preflight_model, mesh):
    config = _config()
    config.target_modules = {"lm_head"}
    preflight_model.tp_plan["lm_head"] = "colwise"
    preflight_model.lm_head.weight = _place(preflight_model.lm_head.weight, mesh, Shard(0))
    with pytest.raises(ValueError, match="input/output embedding module"):
        lora.validate_native_tp_lora_config(preflight_model, config)


def test_preflight_refuses_a_renamed_input_embedding_by_identity(preflight_model):
    preflight_model.embed_tokens = preflight_model.q_proj
    with pytest.raises(ValueError, match="input/output embedding module"):
        lora.validate_native_tp_lora_config(preflight_model, _config())


@pytest.mark.parametrize("postflight", [False, True])
def test_requested_tp_size_must_match_native_mesh(preflight_model, postflight_model, postflight):
    with pytest.raises(ValueError, match="mesh size 2.*Halo tp_size=4"):
        if postflight:
            lora.validate_native_tp_lora_model(postflight_model, tp_size=4)
        else:
            lora.validate_native_tp_lora_config(preflight_model, _config(), tp_size=4)


def test_native_distributed_config_must_match_mesh(preflight_model):
    preflight_model.config.distributed_config.tp_size = 4
    with pytest.raises(ValueError, match="distributed_config.tp_size"):
        lora.validate_native_tp_lora_config(preflight_model, _config())


@pytest.mark.parametrize("axis", ["fsdp_size", "pp_size"])
def test_combined_native_axes_remain_refused(preflight_model, axis):
    setattr(preflight_model.config.distributed_config, axis, 2)
    with pytest.raises(ValueError, match="combined with TP"):
        lora.validate_native_tp_lora_config(preflight_model, _config())


def test_postflight_accepts_native_structure_without_mutation(postflight_model):
    before = {name: id(param) for name, param in postflight_model.named_parameters()}
    layer = postflight_model.get_base_model().q_proj
    forward = layer.lora_B["default"].forward
    lora.validate_native_tp_lora_model(postflight_model, tp_size=2)
    assert {name: id(param) for name, param in postflight_model.named_parameters()} == before
    assert layer.lora_B["default"].forward is forward


@pytest.mark.parametrize("target,field", [("q_proj", "out_features"), ("o_proj", "in_features")])
def test_postflight_refuses_local_fan_in_or_out_metadata(postflight_model, target, field):
    layer = getattr(postflight_model.get_base_model(), target)
    setattr(layer, field, 4)
    with pytest.raises(ValueError, match="local-shard layer metadata.*wrong fan-in"):
        lora.validate_native_tp_lora_model(postflight_model)


def test_postflight_refuses_missing_native_forward(postflight_model):
    factor = postflight_model.get_base_model().q_proj.lora_B["default"]
    factor.forward = nn.Linear.forward.__get__(factor, nn.Linear)
    with pytest.raises(ValueError, match="missing the native Transformers factor forward"):
        lora.validate_native_tp_lora_model(postflight_model)


def test_postflight_refuses_wrong_sharded_factor_placement(postflight_model, mesh):
    factor = postflight_model.get_base_model().q_proj.lora_B["default"]
    factor.weight = _place(nn.Parameter(torch.zeros(8, 4)), mesh, Replicate())
    with pytest.raises(ValueError, match="incorrect TP placement"):
        lora.validate_native_tp_lora_model(postflight_model)


def test_postflight_refuses_a_dtensor_replicated_half(postflight_model, mesh):
    factor = postflight_model.get_base_model().q_proj.lora_A["default"]
    factor.weight = _place(factor.weight, mesh, Replicate())
    with pytest.raises(ValueError, match="plain replicated factor"):
        lora.validate_native_tp_lora_model(postflight_model)


def test_postflight_refuses_wrong_global_factor_shape(postflight_model):
    factor = postflight_model.get_base_model().q_proj.lora_A["default"]
    factor.weight = nn.Parameter(torch.zeros(4, 4))
    with pytest.raises(ValueError, match="bias-free Linear of shape"):
        lora.validate_native_tp_lora_model(postflight_model)


def test_postflight_refuses_unfrozen_base_parameter(postflight_model):
    postflight_model.get_base_model().lm_head.weight.requires_grad_(True)
    with pytest.raises(ValueError, match="freeze every base parameter.*lm_head"):
        lora.validate_native_tp_lora_model(postflight_model)


def test_postflight_refuses_frozen_adapter(postflight_model):
    postflight_model.get_base_model().q_proj.lora_A["default"].weight.requires_grad_(False)
    with pytest.raises(ValueError, match="unexpectedly frozen"):
        lora.validate_native_tp_lora_model(postflight_model)


def test_postflight_refuses_scaling_drift(postflight_model):
    postflight_model.get_base_model().q_proj.scaling["default"] *= 2
    with pytest.raises(ValueError, match="inconsistent adapter metadata"):
        lora.validate_native_tp_lora_model(postflight_model)


def test_postflight_refuses_multiple_adapters(postflight_model):
    postflight_model.peft_config["second"] = copy.deepcopy(postflight_model.peft_config["default"])
    with pytest.raises(ValueError, match="exactly one adapter"):
        lora.validate_native_tp_lora_model(postflight_model)


def test_postflight_refuses_a_nondefault_adapter_name(postflight_model):
    postflight_model.peft_config["named"] = postflight_model.peft_config.pop("default")
    with pytest.raises(ValueError, match="adapter name 'default'.*checkpoint save and resume"):
        lora.validate_native_tp_lora_model(postflight_model)


def test_postflight_refuses_a_head_adapter_even_with_literal_colwise_plan(postflight_model):
    base = postflight_model.get_base_model()
    base.lm_head = base.q_proj
    base.tp_plan["lm_head"] = "colwise"
    with pytest.raises(ValueError, match="input/output embedding module"):
        lora.validate_native_tp_lora_model(postflight_model)


@pytest.mark.parametrize("feature", ["peft_flag", "peft_hook", "hf_module", "hf_install"])
def test_missing_native_dependency_features_fail_loud(preflight_model, native_api, monkeypatch, feature):
    if feature == "peft_flag":
        monkeypatch.setattr(lora.peft_import_utils, "is_transformers_dtensor_tp", False)
    elif feature == "peft_hook":
        monkeypatch.setattr(lora.peft_lora_model, "add_lora_tp_hooks_dtensor", None)
    elif feature == "hf_module":
        monkeypatch.setattr(lora, "hf_tp", None)
    else:
        monkeypatch.setattr(native_api.ALL_PARALLEL_STYLES["rowwise"], "install_forward", None)
    with pytest.raises(RuntimeError, match="installed runtime lacks that API"):
        lora.validate_native_tp_lora_config(preflight_model, _config())


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
