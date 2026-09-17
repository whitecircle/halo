#!/usr/bin/env python
"""Every parameter is used inside the forward of the FSDP2 unit that owns it — across the roster.

FSDP2 unshards a unit's parameters in that unit's forward pre-hook. ``apply_fsdp2_per_layer`` picks
the units, so a parameter whose module runs *before* its unit's forward opens is still a sharded
DTensor at the call: ``aten.embedding`` (or ``aten.mm``) then raises "got mixed torch.Tensor and
DTensor" at the first training step. The multimodal wrappers are the shape that exposes it: their
``<Family>Model.forward`` embeds the ids itself and only then calls ``language_model``, so the unit
holding ``embed_tokens`` has to be that model or an ancestor — bare and under a ``PeftModel``, whose
attribute forwarding moves what ``model.model`` names.

The check is structural and family-agnostic: record the ``fully_shard`` calls, derive each
parameter's owning unit, run one real CPU forward, and require the owning unit to be on the forward
stack whenever a module with parameters of its own is called.

Run: pytest tests/cpu/parallelism/test_fsdp2_unit_active_at_param_use.py
"""

# The device-aware kernel-dispatch shim must land before the modeling modules below bind transformers'
# hub-kernel fallback at import, or a CPU forward of a conv / linear-attention family reaches the
# CUDA-only causal_conv1d kernel.
import src.models.patches.kernel_dispatch  # noqa: F401  # isort: skip

from unittest.mock import patch

import pytest
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from transformers import (
    Cohere2MoeConfig,
    Cohere2MoeForCausalLM,
    DeepseekV4Config,
    DeepseekV4ForCausalLM,
    Gemma4Config,
    Gemma4ForConditionalGeneration,
    Glm4MoeLiteConfig,
    Glm4MoeLiteForCausalLM,
    Glm5NextConfig,
    Glm5NextForConditionalGeneration,
    GptOssConfig,
    GptOssForCausalLM,
    InklingConfig,
    InklingForConditionalGeneration,
    InklingVisionConfig,
    LagunaConfig,
    LagunaForCausalLM,
    Lfm2MoeConfig,
    Lfm2MoeForCausalLM,
    Lfm2VlConfig,
    Lfm2VlForConditionalGeneration,
    Mistral3Config,
    Mistral3ForConditionalGeneration,
    Mistral4Config,
    Mistral4ForCausalLM,
    Qwen3_5Config,
    Qwen3_5ForConditionalGeneration,
    Qwen3_5MoeConfig,
    Qwen3_5MoeForConditionalGeneration,
    Qwen3Config,
    Qwen3ForCausalLM,
    Qwen3MoeConfig,
    Qwen3MoeForCausalLM,
    Qwen3VLConfig,
    Qwen3VLForConditionalGeneration,
    Step3p7Config,
    Step3p7ForConditionalGeneration,
    ZayaConfig,
    ZayaForCausalLM,
)

import src.distributed.fsdp as fsdp
from src.distributed.fsdp import IdentityParamSet
from tests.common.models import (
    TINY_COHERE2_MOE_CONFIG,
    TINY_DSV4_CONFIG,
    TINY_GEMMA4_MOE_CONFIG,
    TINY_GLM4_MOE_LITE_CONFIG,
    TINY_GLM5_CONFIG,
    TINY_GLM5_VISION_CONFIG,
    TINY_GPTOSS_CONFIG,
    TINY_INKLING_CONFIG,
    TINY_LAGUNA_CONFIG,
    TINY_LFM2_MOE_CONFIG,
    TINY_QWEN3_CONFIG,
    TINY_QWEN3_MOE_CONFIG,
    TINY_QWEN35_CONFIG,
    TINY_QWEN35_MOE_CONFIG,
    TINY_STEP3P7_CONFIG,
    TINY_STEP3P7_VISION_CONFIG,
    TINY_ZAYA_CONFIG,
)

_QWEN35_VISION = {"depth": 1, "hidden_size": 16, "intermediate_size": 16, "num_heads": 2, "out_hidden_size": 32}
_MISTRAL4_TEXT = {
    "vocab_size": 128,
    "hidden_size": 64,
    "intermediate_size": 128,
    "moe_intermediate_size": 64,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 4,
    "kv_lora_rank": 32,
    "q_lora_rank": None,
    "qk_nope_head_dim": 16,
    "qk_rope_head_dim": 8,
    "v_head_dim": 16,
    "num_local_experts": 4,
    "num_experts_per_tok": 2,
    "max_position_embeddings": 512,
    "tie_word_embeddings": False,
}
_PIXTRAL_VISION = {
    "hidden_size": 32,
    "intermediate_size": 64,
    "num_hidden_layers": 1,
    "num_attention_heads": 2,
    "head_dim": 16,
    "patch_size": 4,
    "image_size": 16,
}
# The upstream scale planner rejects most tiny shapes; this one builds a valid pixel-shuffle stack.
_INKLING_VISION = {"temporal_patch_size": 2, "patch_size": 4, "n_layers": 4, "hidden_size": 32}
_LFM2_TEXT = {
    "model_type": "lfm2",
    "vocab_size": 128,
    "hidden_size": 64,
    "intermediate_size": 128,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "max_position_embeddings": 128,
    "layer_types": ["conv", "full_attention"],
    "tie_word_embeddings": False,
    "tie_embedding": False,
}
_SIGLIP2_VISION = {
    "model_type": "siglip2_vision_model",
    "hidden_size": 32,
    "intermediate_size": 64,
    "num_hidden_layers": 1,
    "num_attention_heads": 2,
    "num_patches": 16,
    "patch_size": 4,
}


def _untied(config: dict) -> dict:
    """The tie hides the defect (a tied weight is reserved for the root unit), so every case runs untied."""
    return {**config, "tie_word_embeddings": False}


def _qwen35_vision(text: dict) -> dict:
    return {**_QWEN35_VISION, "out_hidden_size": text["hidden_size"]}


def _inkling_composite() -> InklingForConditionalGeneration:
    vision = InklingVisionConfig(**_INKLING_VISION, text_hidden_size=TINY_INKLING_CONFIG["hidden_size"])
    config = InklingConfig(
        text_config=_untied(TINY_INKLING_CONFIG),
        vision_config=vision.to_dict(),
        image_token_id=5,
        tie_word_embeddings=False,
    )
    config._attn_implementation = "eager"
    return InklingForConditionalGeneration(config)


def _qwen3_vl_composite() -> Qwen3VLForConditionalGeneration:
    text = {**_untied(TINY_QWEN3_CONFIG), "rope_scaling": {"rope_type": "default", "mrope_section": [2, 1, 1]}}
    vision = {**_qwen35_vision(text), "deepstack_visual_indexes": []}
    return Qwen3VLForConditionalGeneration(
        Qwen3VLConfig(text_config=text, vision_config=vision, tie_word_embeddings=False)
    )


_FAMILIES = {
    "qwen3": lambda: Qwen3ForCausalLM(Qwen3Config(**_untied(TINY_QWEN3_CONFIG))),
    "qwen3_moe": lambda: Qwen3MoeForCausalLM(Qwen3MoeConfig(**_untied(TINY_QWEN3_MOE_CONFIG))),
    "gpt_oss": lambda: GptOssForCausalLM(GptOssConfig(**_untied(TINY_GPTOSS_CONFIG))),
    "glm4_moe_lite": lambda: Glm4MoeLiteForCausalLM(Glm4MoeLiteConfig(**_untied(TINY_GLM4_MOE_LITE_CONFIG))),
    "laguna": lambda: LagunaForCausalLM(LagunaConfig(**_untied(TINY_LAGUNA_CONFIG))),
    "lfm2_moe": lambda: Lfm2MoeForCausalLM(Lfm2MoeConfig(**_untied(TINY_LFM2_MOE_CONFIG))),
    "cohere2_moe": lambda: Cohere2MoeForCausalLM(Cohere2MoeConfig(**_untied(TINY_COHERE2_MOE_CONFIG))),
    "deepseek_v4": lambda: DeepseekV4ForCausalLM(DeepseekV4Config(**_untied(TINY_DSV4_CONFIG))),
    "zaya": lambda: ZayaForCausalLM(ZayaConfig(**_untied(TINY_ZAYA_CONFIG))),
    "mistral4": lambda: Mistral4ForCausalLM(Mistral4Config(**_MISTRAL4_TEXT)),
    "mistral3_over_mistral4_composite": lambda: Mistral3ForConditionalGeneration(
        Mistral3Config(
            text_config=dict(_MISTRAL4_TEXT), vision_config=dict(_PIXTRAL_VISION), tie_word_embeddings=False
        )
    ),
    "inkling_composite": _inkling_composite,
    "lfm2_vl_composite": lambda: Lfm2VlForConditionalGeneration(
        Lfm2VlConfig(text_config=dict(_LFM2_TEXT), vision_config=dict(_SIGLIP2_VISION), tie_word_embeddings=False)
    ),
    "qwen3_vl_composite": _qwen3_vl_composite,
    "qwen3_5_composite": lambda: Qwen3_5ForConditionalGeneration(
        Qwen3_5Config(
            text_config=_untied(TINY_QWEN35_CONFIG),
            vision_config=_qwen35_vision(TINY_QWEN35_CONFIG),
            tie_word_embeddings=False,
        )
    ),
    "qwen3_5_moe_composite": lambda: Qwen3_5MoeForConditionalGeneration(
        Qwen3_5MoeConfig(
            text_config=_untied(TINY_QWEN35_MOE_CONFIG),
            vision_config=_qwen35_vision(TINY_QWEN35_MOE_CONFIG),
            tie_word_embeddings=False,
        )
    ),
    "glm5_next_composite": lambda: Glm5NextForConditionalGeneration(
        Glm5NextConfig(
            text_config=_untied(TINY_GLM5_CONFIG),
            vision_config=dict(TINY_GLM5_VISION_CONFIG),
            tie_word_embeddings=False,
        )
    ),
    "step3p7_composite": lambda: Step3p7ForConditionalGeneration(
        Step3p7Config(
            text_config=_untied(TINY_STEP3P7_CONFIG),
            vision_config=dict(TINY_STEP3P7_VISION_CONFIG),
            tie_word_embeddings=False,
        )
    ),
    "gemma4_composite": lambda: Gemma4ForConditionalGeneration(
        Gemma4Config(
            text_config=_untied(TINY_GEMMA4_MOE_CONFIG),
            vision_config=None,
            audio_config=None,
            tie_word_embeddings=False,
        )
    ),
}


def _owning_units(model: nn.Module) -> dict[int, nn.Module]:
    """``id(parameter) -> the fully_shard unit that manages it``, from the recorded wrap calls.

    A unit manages the parameters under its module that no earlier unit took and that the call did
    not pass as ignored — the rule ``fully_shard`` itself applies.
    """
    calls: list[tuple[nn.Module, IdentityParamSet]] = []

    def record(module, **kwargs):
        calls.append((module, kwargs.get("ignored_params") or IdentityParamSet()))

    with patch.object(fsdp, "fully_shard", side_effect=record):
        fsdp.apply_fsdp2_per_layer(
            model, dp_mesh=None, mp_policy=None, reshard_after_forward=False, ignored_params=IdentityParamSet()
        )

    owner: dict[int, nn.Module] = {}
    for module, ignored in calls:
        for param in module.parameters():
            if id(param) not in owner and param not in ignored:
                owner[id(param)] = module
    return owner


def _params_used_outside_their_unit(model: nn.Module, owner: dict[int, nn.Module]) -> list[str]:
    """Run one forward and name every module called while the unit owning its parameters is not open."""
    units = {id(unit): unit for unit in owner.values()}
    open_units: list[int] = []
    violations: list[str] = []
    names = {id(module): name for name, module in model.named_modules()}
    handles = []

    for unit in units.values():
        handles.append(unit.register_forward_pre_hook(lambda mod, _args: open_units.append(id(mod))))
        handles.append(
            unit.register_forward_hook(lambda mod, _args, _out: open_units.remove(id(mod)), always_call=True)
        )

    def check(module, _args):
        for param in module.parameters(recurse=False):
            unit = owner.get(id(param))
            if unit is not None and id(unit) not in open_units:
                violations.append(f"{names[id(module)]} ({type(module).__name__}) before {type(unit).__name__}")
                return

    for module in model.modules():
        if next(module.parameters(recurse=False), None) is not None:
            # Registered after the unit hooks, so a unit's own direct parameters see the unit open.
            handles.append(module.register_forward_pre_hook(check))

    try:
        with torch.no_grad():
            model(input_ids=torch.tensor([[6, 7, 8, 9, 10, 11]]), attention_mask=torch.ones(1, 6, dtype=torch.long))
    finally:
        for handle in handles:
            handle.remove()
    return violations


def _attention_lora(model: nn.Module) -> nn.Module:
    """Attention LoRA on the projections this family's text decoder names, read off the module tree."""
    targets = sorted(
        {
            name.rsplit(".", 1)[-1]
            for name, module in model.named_modules()
            if type(module) is nn.Linear and name.endswith("_proj") and ".layers." in name and "attn" in name
        }
    )
    assert targets, "no attention projection found — the case would adapt nothing"
    return get_peft_model(model, LoraConfig(r=2, lora_alpha=4, target_modules=targets, task_type="CAUSAL_LM"))


@pytest.mark.parametrize("peft", [False, True], ids=["bare", "peft"])
@pytest.mark.parametrize("family", sorted(_FAMILIES))
def test_every_parameter_is_used_inside_its_own_fsdp_unit(family, peft):
    torch.manual_seed(0)
    model = _FAMILIES[family]().eval()
    if peft:
        model = _attention_lora(model)

    owner = _owning_units(model)
    embed_weight = model.get_input_embeddings().weight
    assert id(embed_weight) in owner, "the wrap left the input embedding in no unit"

    assert _params_used_outside_their_unit(model, owner) == []


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
