#!/usr/bin/env python
"""The gpt-oss code-contests LoRA recipes end in a servable checkpoint. They train a mixed adapter no tool folds later,
so each folds at save: the flag reaches the run through the recipe's own parse, and the tiny family through the merged
gather reloads in stock transformers with both halves and the full-trained modules."""

import pytest
import torch
from peft import get_peft_model
from transformers import AutoModelForCausalLM

from halo.checkpoint.adapters import reject_expert_lora_adapter
from halo.checkpoint.parallel.adapters import save_peft_adapters
from halo.checkpoint.parallel.ep import save_ep_gathered_checkpoint
from halo.configs.parallelism_arguments import parallelism_config_from_args
from halo.distributed.expert_parallel.parallelize import apply_ep_layers
from halo.model_loading.parallel import peft_setup
from halo.model_loading.parallel.peft_setup import configure_adapter_training, split_expert_lora_targets
from halo.trainers.grpo.environmental.trainer import DistributedAsyncEnvironmentalGRPOTrainer
from halo.trainers.parallelism.adapter_rules import reject_unmergeable_expert_lora_save
from tests.common.adapters import randomize_adapters
from tests.common.checkpoint_io import save_context
from tests.common.numerics import fro_rel_err
from tests.common.parallelism import make_parallelism_config, single_process_ep_config
from tests.common.repo import REPO_ROOT
from tests.common.tiny_models import TINY_MOE_FAMILIES, build_tiny_family_checkpoint, tiny_family_model
from tests.cpu.config.test_examples_parse import (
    ENV_GRPO_SCRIPT,
    _bf16_capable,  # noqa: F401 — autouse fixture, imported so the recipes validate as a training host does
    parser_for,
)
from tests.cpu.config.test_examples_roster import launch_topology, training_arguments

_ROOT = REPO_ROOT / "examples" / "grpo" / "environmental" / "gptoss"
_RECIPES = sorted(_ROOT.rglob("*-lora-*.yaml"))
_ID = {"ids": lambda recipe: recipe.relative_to(_ROOT).as_posix()}


def _parse_recipe(recipe, monkeypatch):
    """The recipe's model arguments and ``ParallelismConfig`` on its declared topology, the expert peel run against the
    tiny family's config so the hub checkpoint need not be cached."""
    tiny_config = tiny_family_model(TINY_MOE_FAMILIES["gpt_oss"]).config
    monkeypatch.setattr(peft_setup, "fetch_checkpoint_config", lambda *args, **kwargs: tiny_config)
    parsed = parser_for(ENV_GRPO_SCRIPT).parse_yaml_file(str(recipe))
    dist_args = next(obj for obj in parsed if type(obj).__name__ == "ParallelismArguments")
    model_args = next(obj for obj in parsed if type(obj).__name__ == "ModelArguments")
    spec = split_expert_lora_targets(model_args)
    world_size, gpus_per_node = launch_topology(recipe)
    for name, value in (
        ("get_global_world_size", world_size),
        ("get_local_world_size", gpus_per_node),
        ("get_global_rank", 0),
    ):
        monkeypatch.setattr(f"halo.distributed.parallelism_config.{name}", lambda _value=value: _value)
    config = parallelism_config_from_args(
        dist_args,
        trainer_cls=DistributedAsyncEnvironmentalGRPOTrainer,
        training_config=training_arguments(parsed),
        expert_lora=spec,
    )
    return parsed, model_args, config


def test_the_recipes_are_found():
    assert len(_RECIPES) == 3


@pytest.mark.parametrize("recipe", _RECIPES, **_ID)
def test_the_flag_is_admitted_on_the_recipes_own_shape(recipe, monkeypatch):
    _, _, config = _parse_recipe(recipe, monkeypatch)
    assert config.merge_expert_lora_on_save
    assert config.expert_lora is not None and config.expert_lora.projections == {"gate", "up", "down"}


@pytest.mark.parametrize("recipe", _RECIPES, **_ID)
def test_the_merged_save_reloads_in_stock_transformers(recipe, monkeypatch, tmp_path):
    parsed, model_args, config = _parse_recipe(recipe, monkeypatch)
    script_args = next(obj for obj in parsed if hasattr(obj, "unfreeze_layers_patterns"))
    family = TINY_MOE_FAMILIES["gpt_oss"]
    base_dir = str(tmp_path / "base")
    build_tiny_family_checkpoint(family, base_dir)
    load = {"dtype": torch.float32, "attn_implementation": "eager"}
    model = AutoModelForCausalLM.from_pretrained(base_dir, **load).eval()
    ids = torch.arange(3, 27).unsqueeze(0)
    with torch.no_grad():
        base_logits = model(input_ids=ids).logits.clone()

    apply_ep_layers(model, single_process_ep_config(expert_lora=config.expert_lora))
    peft_model = get_peft_model(model, configure_adapter_training(script_args, model, model_args, "CAUSAL_LM")).eval()
    reject_unmergeable_expert_lora_save(
        peft_model, merge_expert_lora_on_save=config.merge_expert_lora_on_save, accelerate_manages_fsdp=False
    )
    randomize_adapters(peft_model, std=0.02)
    with torch.no_grad():
        for name, param in peft_model.named_parameters():
            if "modules_to_save" in name:
                param.add_(torch.randn_like(param) * 0.1)
        live_logits = peft_model(input_ids=ids).logits.clone()

    one_process = make_parallelism_config(
        world_size=1, gpus_per_node=1, merge_expert_lora_on_save=True, expert_lora=config.expert_lora
    )
    context = save_context(peft_model, one_process, has_ep_layers=True, has_expert_lora=True, training_checkpoint=True)
    merged_dir = str(tmp_path / "merged")
    save_ep_gathered_checkpoint(context, merged_dir)
    merged = AutoModelForCausalLM.from_pretrained(merged_dir, **load).eval()
    with torch.no_grad():
        merged_logits = merged(input_ids=ids).logits
    assert fro_rel_err(live_logits, base_logits) > 0.05, "the adapters moved nothing, so the fold proves nothing"
    assert fro_rel_err(merged_logits, live_logits) < 1e-3, "the merged save lost part of the trained model"

    # What the flag is for: this same adapter saved unmerged has no tool that folds it.
    adapter_dir = str(tmp_path / "adapter")
    save_peft_adapters(context, adapter_dir)
    with pytest.raises(ValueError, match="merge_expert_lora_on_save=True from the run's first save"):
        reject_expert_lora_adapter(adapter_dir)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
