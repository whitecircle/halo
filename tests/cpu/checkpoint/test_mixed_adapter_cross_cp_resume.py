#!/usr/bin/env python
"""A mixed attention + native expert adapter resumes at another ``cp_size``: under CP its saver keeps the wrapper's
``model.`` level in the expert keys and normalizes the attention ones, and the restore matches both by plain name."""

import pytest
import torch
from peft import LoraConfig, get_peft_model

from halo.checkpoint.parallel.adapters import restore_adapters, save_peft_adapters
from halo.distributed.context_parallel.config import CPConfig
from halo.distributed.context_parallel.model_rules import SUPPORTED_ATTN_IMPLEMENTATIONS
from halo.distributed.context_parallel.ulysses_model import UlyssesModel
from halo.distributed.expert_parallel.moe_layer.expert_lora import EXPERT_LORA_RANK_MULTIPLE
from halo.distributed.expert_parallel.parallelize import apply_ep_layers
from halo.models.introspection.unwrapping import plain_adapter_key
from halo.models.moe.expert_lora_spec import ExpertLoraSpec
from tests.common.checkpoint_io import save_context
from tests.common.parallelism import make_parallelism_config, single_process_ep_config
from tests.common.tiny_models import TINY_MOE_FAMILIES, tiny_family_model

SPEC = ExpertLoraSpec(r=EXPERT_LORA_RANK_MULTIPLE, alpha=2 * EXPERT_LORA_RANK_MULTIPLE)


def _mixed_model(cp_size: int, seed: int):
    """A PeftModel over the tiny Qwen3-MoE with native expert LoRA, inside the Ulysses wrapper when ``cp_size > 1``,
    every adapter drawn from ``seed``."""
    torch.manual_seed(seed)
    model = tiny_family_model(TINY_MOE_FAMILIES["qwen3_moe"]).to(torch.bfloat16)
    model = apply_ep_layers(model, single_process_ep_config(expert_lora=SPEC))
    if cp_size > 1:
        model.config._attn_implementation = SUPPORTED_ATTN_IMPLEMENTATIONS[0]
        model = UlyssesModel(model, CPConfig(1, stage_world_size=1, stage_base_rank=0, nvlink_domain_size=1))
    peft_model = get_peft_model(model, LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj"], task_type="CAUSAL_LM"))
    with torch.no_grad():
        for name, param in peft_model.named_parameters():
            if "lora_B" in name:
                param.normal_(std=0.3)
    return peft_model


def _adapters(peft_model) -> dict[str, torch.Tensor]:
    """Every adapter tensor by its non-CP name, so two trees at different ``cp_size`` compare key for key."""
    return {
        plain_adapter_key(name): param.detach().clone()
        for name, param in peft_model.named_parameters()
        if "lora_" in name
    }


@pytest.mark.parametrize(("save_cp", "resume_cp"), [(2, 1), (1, 2), (2, 2), (1, 1)])
def test_a_mixed_adapter_resumes_at_another_cp_size(tmp_path, save_cp, resume_cp):
    trained = _mixed_model(save_cp, seed=0)
    run = make_parallelism_config(world_size=save_cp, gpus_per_node=save_cp, cp_size=save_cp, expert_lora=SPEC)
    save_peft_adapters(
        save_context(trained, run, has_ep_layers=True, has_expert_lora=True, training_checkpoint=True), str(tmp_path)
    )
    resumed = _mixed_model(resume_cp, seed=1)
    want, before = _adapters(trained), _adapters(resumed)
    assert want.keys() == before.keys(), sorted(want.keys() ^ before.keys())[:4]
    experts = {key for key in want if key.endswith(("_lora_A", "_lora_B"))}
    assert experts and experts != want.keys(), "premise: the adapter holds both halves"
    assert all(not torch.equal(want[key], before[key]) for key in want), "premise: the resumed run starts elsewhere"

    assert restore_adapters(str(tmp_path), resumed, is_cp_mode=resume_cp > 1) is not None

    got = _adapters(resumed)
    wrong = sorted(key for key in want if not torch.equal(got[key], want[key]))
    assert not wrong, f"{len(wrong)} adapter tensor(s) did not resume, e.g. {wrong[:4]}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
