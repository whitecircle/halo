#!/usr/bin/env python
"""The lazy loaders' key plan and the resume's :class:`ModuleLayoutView` read every tiny family's checkpoint alike."""

import functools
import os
import tempfile
from dataclasses import dataclass

import pytest
import torch
from safetensors.torch import load_file, save_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForSequenceClassification

from halo.checkpoint.hub_layout import ModuleLayoutView
from halo.checkpoint.reading import resolve_checkpoint_weights
from halo.model_loading.lazy_safetensors.conversion import Concat
from halo.model_loading.lazy_safetensors.conversion_selection import family_key_mapping
from halo.model_loading.lazy_safetensors.materialization import materialize_weight_plan
from halo.model_loading.lazy_safetensors.meta_shell import instantiate_on_meta
from halo.model_loading.model_preparation import apply_family_patches
from halo.model_loading.parallel.expert_fuser import ExpertFuser
from halo.model_loading.parallel.lazy_plan import ExpertKeyLayout, WeightPlanner
from halo.models.moe.config_fields import require_expert_count
from halo.models.param_roles import persistent_buffers
from tests.common.models import TINY_QWEN3_MOE_CONFIG
from tests.common.tiny_models import (
    TINY_DENSE_FAMILIES,
    TINY_MOE_FAMILIES,
    TINY_MOE_VLM_FAMILIES,
    build_tiny_family_checkpoint,
)


def _backbone_key(key: str) -> str | None:
    """No head, no ``model.`` level."""
    return None if key.startswith("lm_head.") else key.removeprefix("model.")


def _mistral3_legacy_key(key: str) -> str:
    """The pre-5.0 Mistral3 hub layout."""
    for current, legacy in (
        ("model.language_model.", "language_model.model."),
        ("model.vision_tower.", "vision_tower."),
        ("model.multi_modal_projector.", "multi_modal_projector."),
        ("lm_head.", "language_model.lm_head."),
    ):
        if key.startswith(current):
            return legacy + key.removeprefix(current)
    return key


# Case → (tiny family, shell class where not the family's own, a key respelling).
CASES = {
    **{name: (family, None, None) for name, family in {**TINY_DENSE_FAMILIES, **TINY_MOE_FAMILIES}.items()},
    **{name: (family, None, None) for name, family in TINY_MOE_VLM_FAMILIES.items()},
    **{
        f"{name}[text-only]": (TINY_MOE_VLM_FAMILIES[name], AutoModelForCausalLM, None)
        for name in ("gemma4", "qwen3_5_moe")
    },
    **{
        f"{name}[backbone-keys]": (family, None, _backbone_key)
        for name, family in (
            ("qwen3", TINY_DENSE_FAMILIES["qwen3"]),
            ("qwen3_moe", TINY_MOE_FAMILIES["qwen3_moe"]),
            ("gpt_oss", TINY_MOE_FAMILIES["gpt_oss"]),
        )
    },
    **{
        f"{name}[seq_cls]": (family, AutoModelForSequenceClassification, None)
        for name, family in (("qwen3", TINY_DENSE_FAMILIES["qwen3"]), ("qwen3_moe", TINY_MOE_FAMILIES["qwen3_moe"]))
    },
    "mistral3[legacy-layout]": (TINY_MOE_VLM_FAMILIES["mistral3"], None, _mistral3_legacy_key),
}
# A backbone checkpoint's per-expert keys never reach the lazy fuser's target (spelled without ``model.``), so coverage
# refuses; the resume plans them.
DIVERGENT_CASE = "qwen3_moe[backbone-keys]"
DIVERGENT_KEYS = frozenset(
    f"layers.{layer}.mlp.experts.{expert}.{projection}.weight"
    for layer in range(TINY_QWEN3_MOE_CONFIG["num_hidden_layers"])
    for expert in range(TINY_QWEN3_MOE_CONFIG["num_experts"])
    for projection in ("gate_proj", "up_proj", "down_proj")
)


@dataclass(frozen=True)
class Engines:
    """Each engine's keys per live name, the plans behind them, and the names whose tensors differ."""

    live: tuple[str, ...]
    lazy: dict
    view: dict
    plans: list
    tasks: list
    tensor_mismatches: list


def _save_checkpoint(case: str, directory: str) -> None:
    family, _, respell = CASES[case]
    build_tiny_family_checkpoint(family, directory)
    if respell is None:
        return
    tensors = {}
    for name in os.listdir(directory):
        if ".safetensors" in name:
            if name.endswith(".safetensors"):
                tensors.update(load_file(os.path.join(directory, name)))
            os.remove(os.path.join(directory, name))
    respelled = {respell(key): tensor for key, tensor in tensors.items() if respell(key) is not None}
    save_file(respelled, os.path.join(directory, "model.safetensors"), metadata={"format": "pt"})


@functools.cache
def engines(case: str) -> Engines:
    family, shell_class, _ = CASES[case]
    with tempfile.TemporaryDirectory() as directory:
        _save_checkpoint(case, directory)
        config = AutoConfig.from_pretrained(directory, trust_remote_code=family.trust_remote_code)
        # A family patch may move a tensor out of the module state.
        apply_family_patches(config, "sdpa")
        shell = instantiate_on_meta(
            directory,
            shell_class or family.load_class,
            config,
            trust_remote_code=family.trust_remote_code,
            config_only=True,
        )
        weight_map = resolve_checkpoint_weights(directory).weight_map
        tensors = {}
        for shard in set(weight_map.values()):
            tensors.update(load_file(os.path.join(directory, shard)))

    model_keys = set(shell.state_dict())
    disk_to_model, fanout = family_key_mapping(shell, list(weight_map))
    plans = WeightPlanner(None).build(weight_map, disk_to_model, model_keys, fanout=fanout)
    tasks = []
    if ExpertKeyLayout.detect(weight_map) == ExpertKeyLayout.PER_EXPERT:
        tasks = ExpertFuser(0, require_expert_count(shell.config)).detect_tasks(weight_map, disk_to_model, model_keys)
    lazy = {plan.model_key: tuple(sorted(plan.disk_keys)) for plan in plans}
    lazy |= {task.model_key: tuple(sorted(task.disk_keys)) for task in tasks}

    reads, view, view_tensors, last = [], {}, {}, ()
    for name, tensor in ModuleLayoutView(shell, list(weight_map)).items(lambda key: reads.append(key) or tensors[key]):
        # A converted group reads all sources before its first target.
        last = tuple(sorted(reads)) or last
        reads.clear()
        view[name], view_tensors[name] = last, tensor

    mismatches = [
        plan.model_key
        for plan in plans
        if plan.model_key in view_tensors
        and not torch.equal(
            materialize_weight_plan(plan, lambda key, _plan: tensors[key]), view_tensors[plan.model_key]
        )
    ]
    live = tuple(name for name, _ in (*shell.named_parameters(), *persistent_buffers(shell)))
    return Engines(live, lazy, view, plans, tasks, mismatches)


@pytest.mark.parametrize("case", sorted(CASES))
def test_both_engines_read_every_live_name_alike(case):
    planned = engines(case)
    differing = {name for name in planned.live if planned.lazy.get(name) != planned.view.get(name)}
    lazy_read = {key for keys in planned.lazy.values() for key in keys}
    view_read = {key for keys in planned.view.values() for key in keys}
    if case == DIVERGENT_CASE:
        assert differing and all(name.endswith(("experts.gate_up_proj", "experts.down_proj")) for name in differing)
        assert not differing & set(planned.lazy)
        assert lazy_read <= view_read and view_read - lazy_read == DIVERGENT_KEYS
    else:
        assert not differing, sorted(differing)[:5]
        assert lazy_read == view_read
    assert not planned.tensor_mismatches, planned.tensor_mismatches[:5]


@pytest.mark.parametrize("case", ["deepseek_v4", "glm5_next", "inkling_mm_model", "step3p7"])
def test_vendor_namespace_families_plan_converted_keys(case):
    """Anti-vacuity: the live tree does not carry this spelling."""
    assert any(plan.disk_key != plan.model_key for plan in engines(case).plans)


@pytest.mark.parametrize("case", ["glm5_next", "step3p7"])
def test_fan_in_families_plan_a_concat(case):
    assert any(isinstance(op, Concat) for plan in engines(case).plans for op in plan.ops)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
