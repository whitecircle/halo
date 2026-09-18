#!/usr/bin/env python
"""Every parameter is used inside the forward of the FSDP2 unit that owns it — across the roster.

FSDP2 unshards a unit's parameters in that unit's forward pre-hook. ``apply_fsdp2_per_layer`` picks
the units, so a parameter whose module runs *before* its unit's forward opens is still a sharded
DTensor at the call: ``aten.embedding`` (or ``aten.mm``) then raises "got mixed torch.Tensor and
DTensor" at the first training step. The multimodal wrappers are the shape that exposes it: their
``<Family>Model.forward`` embeds the ids itself and only then calls ``language_model``, so the unit
holding ``embed_tokens`` has to be that model or an ancestor — bare and under a ``PeftModel``, whose
attribute forwarding moves what ``model.model`` names. A tied embedding is reserved for the root unit
and hides the defect, except where PEFT's ``modules_to_save`` copies untie it again, or where a family
carries a second embedding table (Gemma 4's per-layer inputs) that is never tied.

The check is structural and family-agnostic: record the ``fully_shard`` calls, derive each
parameter's owning unit, run one real CPU forward, and require the owning unit to be on the forward
stack whenever a module with parameters of its own is called. The adapters go through the production
seam (:func:`build_peft_config`), so the towers PEFT cannot adapt are excluded exactly as in a run.

Run: pytest tests/cpu/parallelism/test_fsdp2_unit_active_at_param_use.py
"""

# The device-aware kernel-dispatch shim must land before the modeling modules below bind transformers'
# hub-kernel fallback at import, or a CPU forward of a conv / linear-attention family reaches the
# CUDA-only causal_conv1d kernel.
import src.models.patches.kernel_dispatch  # noqa: F401  # isort: skip

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn
from peft import get_peft_model
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
from trl import ModelConfig

import src.distributed.fsdp as fsdp
from src.distributed.fsdp import IdentityParamSet
from src.distributed.loading.peft_setup import build_peft_config
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
# Gemma 4's towers spell the attention projections as `Gemma4ClippableLinear` containers — the shape
# stock LoRA cannot adapt — so this case covers the exclusion seam and the unit resolution together.
_GEMMA4_VISION = {
    "hidden_size": 32,
    "intermediate_size": 64,
    "num_hidden_layers": 1,
    "num_attention_heads": 2,
    "num_key_value_heads": 2,
    "head_dim": 16,
    "patch_size": 4,
    "position_embedding_size": 16,
    "pooling_kernel_size": 1,
}
_GEMMA4_AUDIO = {
    "hidden_size": 32,
    "num_hidden_layers": 1,
    "num_attention_heads": 2,
    "subsampling_conv_channels": [8, 8],
    "conv_kernel_size": 3,
    "attention_chunk_size": 4,
    "attention_context_left": 3,
}
_MODULES_TO_SAVE = ["embed_tokens", "lm_head"]


def _tie(config: dict, tied: bool) -> dict:
    return {**config, "tie_word_embeddings": tied}


def _qwen35_vision(text: dict) -> dict:
    return {**_QWEN35_VISION, "out_hidden_size": text["hidden_size"]}


def _inkling_composite(tied: bool) -> InklingForConditionalGeneration:
    vision = InklingVisionConfig(**_INKLING_VISION, text_hidden_size=TINY_INKLING_CONFIG["hidden_size"])
    config = InklingConfig(
        text_config=_tie(TINY_INKLING_CONFIG, tied),
        vision_config=vision.to_dict(),
        image_token_id=5,
        tie_word_embeddings=tied,
    )
    config._attn_implementation = "eager"
    return InklingForConditionalGeneration(config)


def _qwen3_vl_composite(tied: bool) -> Qwen3VLForConditionalGeneration:
    text = {**_tie(TINY_QWEN3_CONFIG, tied), "rope_scaling": {"rope_type": "default", "mrope_section": [2, 1, 1]}}
    vision = {**_qwen35_vision(text), "deepstack_visual_indexes": []}
    return Qwen3VLForConditionalGeneration(
        Qwen3VLConfig(text_config=text, vision_config=vision, tie_word_embeddings=tied)
    )


class _Composite(nn.Module):
    """A composite whose text stack is a whole ``*ForCausalLM`` — the transformers-4 wrapper layout.

    It embeds the ids itself and enters the decoder-layer backbone one level deeper than the roster's
    composites, so the unit owning ``embed_tokens`` has to be found by ancestry, not by a
    ``language_model`` spelling.
    """

    def __init__(self, language_model: Qwen3ForCausalLM):
        super().__init__()
        self.language_model = language_model
        hidden = language_model.config.hidden_size
        self.projector = nn.Linear(hidden, hidden, bias=False)

    def forward(self, input_ids, attention_mask=None):
        inputs_embeds = self.projector(self.language_model.get_input_embeddings()(input_ids))
        return self.language_model.model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)


class _TwoLevelWrapper(nn.Module):
    def __init__(self, tied: bool):
        super().__init__()
        language_model = Qwen3ForCausalLM(Qwen3Config(**_tie(TINY_QWEN3_CONFIG, tied)))
        self.config = language_model.config
        self.model = _Composite(language_model)
        self.lm_head = language_model.lm_head

    def get_input_embeddings(self):
        return self.model.language_model.get_input_embeddings()

    def get_output_embeddings(self):
        return self.lm_head

    def forward(self, input_ids, attention_mask=None, labels=None, **_ignored):
        hidden = self.model(input_ids, attention_mask=attention_mask).last_hidden_state
        logits = self.lm_head(hidden)
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(logits[:, :-1].flatten(0, 1).float(), labels[:, 1:].flatten())
        return SimpleNamespace(logits=logits, loss=loss)

    def prepare_inputs_for_generation(self, *args, **kwargs):
        # PeftModelForCausalLM latches this at wrap time; this test never generates.
        raise NotImplementedError


_FAMILIES = {
    "qwen3": lambda tied: Qwen3ForCausalLM(Qwen3Config(**_tie(TINY_QWEN3_CONFIG, tied))),
    "qwen3_moe": lambda tied: Qwen3MoeForCausalLM(Qwen3MoeConfig(**_tie(TINY_QWEN3_MOE_CONFIG, tied))),
    "gpt_oss": lambda tied: GptOssForCausalLM(GptOssConfig(**_tie(TINY_GPTOSS_CONFIG, tied))),
    "glm4_moe_lite": lambda tied: Glm4MoeLiteForCausalLM(Glm4MoeLiteConfig(**_tie(TINY_GLM4_MOE_LITE_CONFIG, tied))),
    "laguna": lambda tied: LagunaForCausalLM(LagunaConfig(**_tie(TINY_LAGUNA_CONFIG, tied))),
    "lfm2_moe": lambda tied: Lfm2MoeForCausalLM(Lfm2MoeConfig(**_tie(TINY_LFM2_MOE_CONFIG, tied))),
    "cohere2_moe": lambda tied: Cohere2MoeForCausalLM(Cohere2MoeConfig(**_tie(TINY_COHERE2_MOE_CONFIG, tied))),
    "deepseek_v4": lambda tied: DeepseekV4ForCausalLM(DeepseekV4Config(**_tie(TINY_DSV4_CONFIG, tied))),
    "zaya": lambda tied: ZayaForCausalLM(ZayaConfig(**_tie(TINY_ZAYA_CONFIG, tied))),
    "mistral4": lambda tied: Mistral4ForCausalLM(Mistral4Config(**_tie(_MISTRAL4_TEXT, tied))),
    "mistral3_over_mistral4_composite": lambda tied: Mistral3ForConditionalGeneration(
        Mistral3Config(
            text_config=_tie(_MISTRAL4_TEXT, tied), vision_config=dict(_PIXTRAL_VISION), tie_word_embeddings=tied
        )
    ),
    "inkling_composite": _inkling_composite,
    "lfm2_vl_composite": lambda tied: Lfm2VlForConditionalGeneration(
        Lfm2VlConfig(text_config=_tie(_LFM2_TEXT, tied), vision_config=dict(_SIGLIP2_VISION), tie_word_embeddings=tied)
    ),
    "qwen3_vl_composite": _qwen3_vl_composite,
    "qwen3_5_composite": lambda tied: Qwen3_5ForConditionalGeneration(
        Qwen3_5Config(
            text_config=_tie(TINY_QWEN35_CONFIG, tied),
            vision_config=_qwen35_vision(TINY_QWEN35_CONFIG),
            tie_word_embeddings=tied,
        )
    ),
    "qwen3_5_moe_composite": lambda tied: Qwen3_5MoeForConditionalGeneration(
        Qwen3_5MoeConfig(
            text_config=_tie(TINY_QWEN35_MOE_CONFIG, tied),
            vision_config=_qwen35_vision(TINY_QWEN35_MOE_CONFIG),
            tie_word_embeddings=tied,
        )
    ),
    "glm5_next_composite": lambda tied: Glm5NextForConditionalGeneration(
        Glm5NextConfig(
            text_config=_tie(TINY_GLM5_CONFIG, tied),
            vision_config=dict(TINY_GLM5_VISION_CONFIG),
            tie_word_embeddings=tied,
        )
    ),
    "step3p7_composite": lambda tied: Step3p7ForConditionalGeneration(
        Step3p7Config(
            text_config=_tie(TINY_STEP3P7_CONFIG, tied),
            vision_config=dict(TINY_STEP3P7_VISION_CONFIG),
            tie_word_embeddings=tied,
        )
    ),
    "gemma4_composite": lambda tied: Gemma4ForConditionalGeneration(
        Gemma4Config(
            text_config=_tie(TINY_GEMMA4_MOE_CONFIG, tied),
            vision_config=dict(_GEMMA4_VISION),
            audio_config=dict(_GEMMA4_AUDIO),
            tie_word_embeddings=tied,
        )
    ),
    "two_level_composite": _TwoLevelWrapper,
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


def _attention_lora(model: nn.Module, modules_to_save: list[str] | None) -> nn.Module:
    """Attention LoRA on the projection names the family's decoder layers spell, through the run's seam.

    A name is taken when an ``nn.Linear`` in a decoder layer's attention carries it — what a recipe
    lists — and is then matched over the whole tree, so a tower spelling the same name as a container
    PEFT cannot adapt (Gemma 4) reaches :func:`build_peft_config`'s exclusion as in a run.
    """
    targets = sorted(
        {
            name.rsplit(".", 1)[-1]
            for name, module in model.named_modules()
            if type(module) is nn.Linear and name.endswith("_proj") and ".layers." in name and "attn" in name
        }
    )
    # Over one name, TRL hands PEFT a bare string, which PEFT reads as a regex rather than a name.
    assert len(targets) > 1, "fewer than two attention projections found — the case would not adapt by name"
    model_config = ModelConfig(
        use_peft=True,
        lora_r=2,
        lora_alpha=4,
        lora_dropout=0.0,
        lora_target_modules=targets,
        lora_modules_to_save=modules_to_save,
    )
    return get_peft_model(model, build_peft_config(model, model_config))


@pytest.mark.parametrize("tied", [False, True], ids=["untied", "tied"])
@pytest.mark.parametrize("mode", ["bare", "peft", "peft_modules_to_save"])
@pytest.mark.parametrize("family", sorted(_FAMILIES))
def test_every_parameter_is_used_inside_its_own_fsdp_unit(family, mode, tied):
    torch.manual_seed(0)
    model = _FAMILIES[family](tied).eval()
    if tied and model.get_input_embeddings().weight is not model.get_output_embeddings().weight:
        pytest.skip("the family does not tie its embeddings")
    if mode != "bare":
        model = _attention_lora(model, _MODULES_TO_SAVE if mode == "peft_modules_to_save" else None)

    owner = _owning_units(model)
    embed_weight = model.get_input_embeddings().weight
    assert id(embed_weight) in owner, "the wrap left the input embedding in no unit"

    assert _params_used_outside_their_unit(model, owner) == []


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
