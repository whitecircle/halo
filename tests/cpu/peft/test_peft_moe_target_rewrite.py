#!/usr/bin/env python
"""On a transformers v5 fused-expert MoE, PEFT's wrap rewrites a regex or ``all-linear`` target: it adds the fused
experts and the router as ``target_parameters`` and widens a regex to bare leaf names. The run's config build hands
PEFT the module paths the target matches as written instead, so the wrap adapts what the pattern names, as before the
rewrite existed, and refuses a target PEFT would still rewrite."""

import copy
import re

import pytest
import torch
from peft import LoraConfig, get_peft_model
from peft.tuners.tuners_utils import BaseTunerLayer
from transformers import PreTrainedModel, Qwen3Config, Qwen3ForCausalLM

from halo.configs.script_arguments import ModelArguments
from halo.distributed.context_parallel.config import CPConfig
from halo.distributed.context_parallel.model_rules import SUPPORTED_ATTN_IMPLEMENTATIONS
from halo.distributed.context_parallel.ulysses_model import UlyssesModel
from halo.distributed.expert_parallel.parallelize import apply_ep_layers
from halo.model_loading.parallel.peft_setup import build_peft_config
from tests.common.models import TINY_QWEN3_CONFIG
from tests.common.parallelism import single_process_ep_config
from tests.common.tiny_models import TINY_MOE_FAMILIES, tiny_family_model

# The roster's families whose targets PEFT rewrites for the transformers v5 layout.
_REWRITTEN_FAMILIES = ("qwen3_moe", "qwen3_next", "glm4_moe_lite", "cohere2_moe")
_EXPERTS_REGEX = r".*\.(q_proj|v_proj|gate_proj|up_proj|down_proj)"


def _moe(family: str, *, layout: str) -> torch.nn.Module:
    torch.manual_seed(0)
    model = tiny_family_model(TINY_MOE_FAMILIES[family])
    if layout == "ep_layers":
        apply_ep_layers(model, single_process_ep_config(experts_fsdp_managed=False))
    elif layout == "ulysses":
        model.config._attn_implementation = SUPPORTED_ATTN_IMPLEMENTATIONS[0]
        model = UlyssesModel(model, CPConfig(1, stage_world_size=1, stage_base_rank=0, nvlink_domain_size=1))
    return model


def _target(kind: str, model: torch.nn.Module) -> str:
    if kind == "all_linear":
        return "all-linear"
    if kind == "experts_regex":
        return _EXPERTS_REGEX
    # One layer's attention projection whose leaf name another layer shares: a rewrite to bare leaf names takes it there.
    linears = [
        name
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear) and ".layers." in name and ".mlp." not in name
    ]
    leaves = [name.rsplit(".", 1)[-1] for name in linears]
    name, leaf = next((name, leaf) for name, leaf in zip(linears, leaves, strict=True) if leaves.count(leaf) > 1)
    return rf".*layers\.{re.search(r'layers\.(\d+)\.', name).group(1)}\..*\.{leaf}"


def _matched_as_written(model: torch.nn.Module, target: str) -> set[str]:
    """The modules the target names, matched as PEFT matched it before its rewrite: a regex in full against each module
    name, ``all-linear`` by the leaf names of every ``nn.Linear`` but, on a ``PreTrainedModel``, the output head."""
    names = [name for name, _ in model.named_modules() if name]
    if target != "all-linear":
        return {name for name in names if re.fullmatch(target, name)}
    head = model.get_output_embeddings() if isinstance(model, PreTrainedModel) else None
    leaves = {
        name.rsplit(".", 1)[-1]
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear) and module is not head
    }
    return {name for name in names if name.rsplit(".", 1)[-1] in leaves}


def _adapted(peft_model) -> tuple[set[str], set[str]]:
    """The modules the wrap built a tuner layer on, and the parameters it targets."""
    modules = {
        name.removeprefix("base_model.model.")
        for name, module in peft_model.named_modules()
        if isinstance(module, BaseTunerLayer)
    }
    return modules, set(peft_model.peft_config["default"].target_parameters or ())


def _run_config(target, **fields) -> ModelArguments:
    return ModelArguments(
        use_peft=True, lora_r=2, lora_alpha=4, lora_dropout=0.0, lora_target_modules=copy.copy(target), **fields
    )


@pytest.mark.parametrize("kind", ["all_linear", "one_layer_regex", "experts_regex"])
@pytest.mark.parametrize(
    ("family", "layout"),
    [
        (family, layout)
        for family in _REWRITTEN_FAMILIES
        for layout in ("hf_layout", "ep_layers", "ulysses")
        if layout != "ulysses" or TINY_MOE_FAMILIES[family].ulysses_cp
    ],
)
def test_the_wrap_adapts_exactly_what_the_target_matches_as_written(family, layout, kind):
    model = _moe(family, layout=layout)
    target = _target(kind, model)
    expected = _matched_as_written(model, target)
    assert expected, "premise: the target names modules on this model"
    try:
        peft_alone = _adapted(get_peft_model(copy.deepcopy(model), LoraConfig(r=2, target_modules=target)))
    except (AttributeError, ValueError) as error:
        peft_alone = error
    assert peft_alone != (expected, set()), "premise: PEFT's own rewrite changes what the target adapts"

    config = build_peft_config(model, _run_config(target))
    assert not config.exclude_modules, "premise: nothing the target matches is unadaptable here"

    modules, parameters = _adapted(get_peft_model(model, config))
    assert modules == expected, (sorted(modules - expected)[:3], sorted(expected - modules)[:3])
    assert not parameters


@pytest.mark.parametrize("layout", ["hf_layout", "ep_layers"])
def test_a_target_peft_keeps_as_written_is_handed_over_unchanged(layout):
    config = build_peft_config(_moe("qwen3_moe", layout=layout), _run_config(["q_proj", "v_proj"]))
    assert config.target_modules == {"q_proj", "v_proj"}


def test_a_dense_model_keeps_its_sentinel():
    model = Qwen3ForCausalLM(Qwen3Config(**TINY_QWEN3_CONFIG))
    assert build_peft_config(model, _run_config("all-linear")).target_modules == "all-linear"


@pytest.mark.parametrize("layout", ["hf_layout", "ep_layers"])
def test_a_target_peft_still_turns_into_parameters_is_refused_with_a_remedy_that_builds(layout):
    """The router matches as a module, which PEFT adapts only as its weight parameter."""
    with pytest.raises(
        ValueError,
        match=r"would still adapt 8 target_parameters \(e\.g\. \[.*mlp\.gate\.weight(?s:.*)lora_modules_to_save",
    ):
        build_peft_config(_moe("qwen3_moe", layout=layout), _run_config(["q_proj", "gate"]))

    model = _moe("qwen3_moe", layout=layout)
    config = build_peft_config(model, _run_config(["q_proj"], lora_modules_to_save=["gate"]))
    peft_model = get_peft_model(model, config)
    assert not _adapted(peft_model)[1]
    assert any("mlp.gate.modules_to_save" in name for name, p in peft_model.named_parameters() if p.requires_grad)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
