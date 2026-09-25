#!/usr/bin/env python
"""The head-path transform contract (``src/models/head_transform.py``) across the supported roster.

The chunked GRPO log-prob sweep and the last pipeline stage compute logits from the backbone's last
hidden state instead of through the model's forward. A family whose forward transforms that path —
Inkling's μP hidden division and vocabulary cut, Cohere's ``logit_scale``, Granite's
``logits_scaling`` division, MiniCPM3's pre-head division, Gemma's softcap — is scored against the
wrong distribution unless the path applies the same transform, and nothing raises. This file pins:

* every roster family's head path verifies against its own forward — the MoE half is derived from
  the EP layer registry, so a new native MoE family is swept without editing this file; the dense
  families (``_DENSE_ROSTER``) and the remote-code ones (``_REMOTE_ROSTER``) are listed here;
* the chunked sweep reproduces the forward log-probs and every parameter gradient of each tiny model
  in ``_TINY_MODELS``, the transform families included, and those families' transforms are
  load-bearing here; a new family's tiny model is added there;
* a declaration removed makes its family refused, and an undeclared transform is refused.

    pytest -m cpu tests/cpu/models/test_head_transform.py
"""

import pytest
import torch
import transformers
from transformers import AutoConfig
from transformers.dynamic_module_utils import get_class_from_dynamic_module
from transformers.models.auto import modeling_auto
from trl.trainer.utils import selective_log_softmax

from src.distributed.expert_parallel.expert_weights import ep_layer_classes
from src.models import head_transform
from src.models.head_transform import (
    IDENTITY_HEAD_TRANSFORM,
    GraniteHeadTransform,
    HeadTransform,
    resolve_head_transform,
    resolve_head_transform_spec,
    verify_head_transform,
)
from src.models.patches.remote_code_compat import apply_remote_code_compat_shims
from src.trainers.grpo.mixins.chunked_logprobs import ChunkedLogprobsCore
from tests.common.models import (
    BAILING_MOE_LING_MINI,
    BAILING_MOE_RING_MINI,
    TINY_COHERE2_MOE_CONFIG,
    TINY_DSV4_CONFIG,
    TINY_GEMMA4_MOE_CONFIG,
    TINY_GLM4_MOE_LITE_CONFIG,
    TINY_GLM5_CONFIG,
    TINY_GLM5_VISION_CONFIG,
    TINY_GPTOSS_CONFIG,
    TINY_GRANITE_CONFIG,
    TINY_INKLING_CONFIG,
    TINY_LAGUNA_CONFIG,
    TINY_LFM2_MOE_CONFIG,
    TINY_MINICPM3_CONFIG,
    TINY_QWEN3_CONFIG,
    TINY_QWEN3_MOE_CONFIG,
    TINY_QWEN35_CONFIG,
    TINY_QWEN35_MOE_CONFIG,
    TINY_STEP3P7_CONFIG,
    TINY_STEP3P7_VISION_CONFIG,
    TINY_ZAYA_CONFIG,
)

apply_remote_code_compat_shims()

# The dense half of the roster; the MoE half is every model_type the EP layer registry claims.
_DENSE_ROSTER = ("qwen3", "qwen3_5_text", "qwen3_vl")

# Families whose transformers defaults do not build a model; their tiny config stands in.
_DEFAULT_CONFIG_OVERRIDES = {"lfm2_moe": TINY_LFM2_MOE_CONFIG}

# Values the shipped checkpoints carry and the defaults leave at identity, so the sweep verifies the
# transform the run applies: Gemma 4 caps at 30, Inkling cuts its padded vocabulary.
_UNPADDED_VOCAB = TINY_INKLING_CONFIG["vocab_size"] - 12
_GEMMA4_TEXT = {**TINY_GEMMA4_MOE_CONFIG, "final_logit_softcapping": 30.0}
# The tiny models' random-init logits stay far below 30, where the cap is the identity; this one bites.
_GEMMA4_SATURATED_TEXT = {**TINY_GEMMA4_MOE_CONFIG, "final_logit_softcapping": 0.02}
_INKLING_TEXT = {**TINY_INKLING_CONFIG, "unpadded_vocab_size": _UNPADDED_VOCAB}

# Remote-code roster families, verified from their cached hub configs.
_REMOTE_ROSTER = (BAILING_MOE_LING_MINI, BAILING_MOE_RING_MINI)

_BATCH, _SEQ, _COMPLETION = 2, 12, 6


def _causal_lm_class_names(model_type: str) -> set[str]:
    names = set()
    for mapping in (
        modeling_auto.MODEL_FOR_CAUSAL_LM_MAPPING_NAMES,
        modeling_auto.MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES,
    ):
        found = mapping.get(model_type, ())
        names.update([found] if isinstance(found, str) else found)
    return names


def _roster_classes() -> list[tuple[str, str]]:
    model_types = {model_type for cls in ep_layer_classes() for model_type in cls.HF_MODEL_TYPES}
    return sorted(
        (model_type, name)
        for model_type in model_types | set(_DENSE_ROSTER)
        for name in _causal_lm_class_names(model_type)
    )


def _model(class_name: str, config_class_name: str, **config) -> torch.nn.Module:
    """A seeded fp32 tiny model; classes are resolved by name at build time, after ``src`` installed
    its device-aware kernel dispatch, so the linear-attention families run their CPU paths."""
    torch.manual_seed(0)
    model_class = getattr(transformers, class_name)
    return model_class(getattr(transformers, config_class_name)(**config)).float().eval()


# (id, builder) for every roster family with a shared tiny config, plus the dense transform families.
_TINY_MODELS = {
    "qwen3": lambda: _model("Qwen3ForCausalLM", "Qwen3Config", **TINY_QWEN3_CONFIG),
    "qwen3_moe": lambda: _model("Qwen3MoeForCausalLM", "Qwen3MoeConfig", **TINY_QWEN3_MOE_CONFIG),
    "qwen3_5": lambda: _model("Qwen3_5ForCausalLM", "Qwen3_5TextConfig", **TINY_QWEN35_CONFIG),
    "qwen3_5_moe": lambda: _model("Qwen3_5MoeForCausalLM", "Qwen3_5MoeTextConfig", **TINY_QWEN35_MOE_CONFIG),
    "gpt_oss": lambda: _model("GptOssForCausalLM", "GptOssConfig", **TINY_GPTOSS_CONFIG),
    "glm4_moe_lite": lambda: _model("Glm4MoeLiteForCausalLM", "Glm4MoeLiteConfig", **TINY_GLM4_MOE_LITE_CONFIG),
    "laguna": lambda: _model("LagunaForCausalLM", "LagunaConfig", **TINY_LAGUNA_CONFIG),
    "lfm2_moe": lambda: _model("Lfm2MoeForCausalLM", "Lfm2MoeConfig", **TINY_LFM2_MOE_CONFIG),
    "zaya": lambda: _model("ZayaForCausalLM", "ZayaConfig", **TINY_ZAYA_CONFIG),
    "deepseek_v4": lambda: _model("DeepseekV4ForCausalLM", "DeepseekV4Config", **TINY_DSV4_CONFIG),
    "glm5_next": lambda: _model(
        "Glm5NextForConditionalGeneration",
        "Glm5NextConfig",
        text_config=dict(TINY_GLM5_CONFIG),
        vision_config=dict(TINY_GLM5_VISION_CONFIG),
    ),
    "step3p7": lambda: _model(
        "Step3p7ForConditionalGeneration",
        "Step3p7Config",
        text_config=dict(TINY_STEP3P7_CONFIG),
        vision_config=dict(TINY_STEP3P7_VISION_CONFIG),
    ),
    "inkling": lambda: _model("InklingForCausalLM", "InklingTextConfig", **_INKLING_TEXT),
    "cohere2_moe": lambda: _model("Cohere2MoeForCausalLM", "Cohere2MoeConfig", **TINY_COHERE2_MOE_CONFIG),
    "gemma4_text": lambda: _model("Gemma4ForCausalLM", "Gemma4TextConfig", **_GEMMA4_SATURATED_TEXT),
    "gemma4": lambda: _model("Gemma4ForConditionalGeneration", "Gemma4Config", text_config=_GEMMA4_SATURATED_TEXT),
    "granite": lambda: _model("GraniteForCausalLM", "GraniteConfig", **TINY_GRANITE_CONFIG),
    "minicpm3": lambda: _model("MiniCPM3ForCausalLM", "MiniCPM3Config", **TINY_MINICPM3_CONFIG),
}

# The families above whose forward transforms the head path; each must be load-bearing here.
_TRANSFORM_FAMILIES = ("inkling", "cohere2_moe", "gemma4_text", "gemma4", "granite", "minicpm3")


class _ForwardParityHarness(ChunkedLogprobsCore):
    """The chunked sweep over a real model, fed the hidden state TRL's and the offline trainer's
    ``_get_last_hidden_state`` take: ``base_model``, the final position dropped, the completion kept."""

    temperature = 1.0

    def _get_last_hidden_state(self, model, input_ids, attention_mask, logits_to_keep):
        hidden = model.base_model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        return hidden.last_hidden_state[:, :-1, :][:, -logits_to_keep:, :]


class _UndeclaredSpec:
    """What every family resolves to with its declaration removed, softcap included. Not a
    ``HeadTransformSpec`` subclass, so it joins no registry."""

    @classmethod
    def transform(cls, config) -> HeadTransform:
        return IDENTITY_HEAD_TRANSFORM


def _batch(model) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(input_ids, attention_mask, upstream)``; ids below the family's returned vocabulary."""
    text = model.config.get_text_config()
    vocab = getattr(text, "unpadded_vocab_size", None) or text.vocab_size
    generator = torch.Generator().manual_seed(1)
    input_ids = torch.randint(0, vocab, (_BATCH, _SEQ), generator=generator)
    upstream = torch.randn(_BATCH, _COMPLETION, generator=generator)
    return input_ids, torch.ones_like(input_ids), upstream


def _forward_logps(model, input_ids, attention_mask) -> torch.Tensor:
    logits = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits
    return selective_log_softmax(logits[:, :-1][:, -_COMPLETION:].float(), input_ids[:, -_COMPLETION:])


def _grads(model) -> dict[str, torch.Tensor]:
    grads = {name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None}
    model.zero_grad(set_to_none=True)
    return grads


@pytest.mark.parametrize(("model_type", "class_name"), _roster_classes())
def test_every_roster_head_path_verifies_against_its_forward(model_type, class_name):
    """A roster family the verification cannot reproduce would be refused on both hidden-state paths."""
    model_class = getattr(transformers, class_name)
    config = model_class.config_class(**_DEFAULT_CONFIG_OVERRIDES.get(model_type, {}))
    verify_head_transform(model_class, config)


@pytest.mark.parametrize("repo", _REMOTE_ROSTER)
def test_remote_code_roster_head_paths_verify(repo):
    try:
        config = AutoConfig.from_pretrained(repo, local_files_only=True, trust_remote_code=True)
    except OSError:
        pytest.skip(f"{repo} is not in the local HF cache")
    model_class = get_class_from_dynamic_module(config.auto_map["AutoModelForCausalLM"], repo, local_files_only=True)
    assert verify_head_transform(model_class, config) == IDENTITY_HEAD_TRANSFORM


_INKLING_TRANSFORM = HeadTransform(hidden_scale=1 / 24, vocab_size=_UNPADDED_VOCAB)
_GRANITE_TRANSFORM = HeadTransform(logit_scale=1 / 8)


@pytest.mark.parametrize(
    ("class_name", "config", "expected"),
    [
        pytest.param("InklingForCausalLM", _INKLING_TEXT, _INKLING_TRANSFORM, id="inkling"),
        pytest.param(
            "InklingForConditionalGeneration", {"text_config": _INKLING_TEXT}, _INKLING_TRANSFORM, id="inkling_mm"
        ),
        pytest.param("CohereForCausalLM", {}, HeadTransform(logit_scale=0.0625), id="cohere"),
        pytest.param("Cohere2ForCausalLM", {}, HeadTransform(logit_scale=0.0625), id="cohere2"),
        pytest.param("Cohere2MoeForCausalLM", {}, HeadTransform(logit_scale=0.0625), id="cohere2_moe"),
        pytest.param("GraniteForCausalLM", {"logits_scaling": 8.0}, _GRANITE_TRANSFORM, id="granite"),
        pytest.param("GraniteSWAForCausalLM", {"logits_scaling": 8.0}, _GRANITE_TRANSFORM, id="granite_swa"),
        pytest.param("GraniteMoeForCausalLM", {"logits_scaling": 8.0}, _GRANITE_TRANSFORM, id="granitemoe"),
        pytest.param("GraniteMoeSWAForCausalLM", {"logits_scaling": 8.0}, _GRANITE_TRANSFORM, id="granitemoe_swa"),
        pytest.param(
            "GraniteMoeSharedForCausalLM", {"logits_scaling": 8.0}, _GRANITE_TRANSFORM, id="granitemoeshared"
        ),
        pytest.param(
            "GraniteMoeHybridForCausalLM", {"logits_scaling": 8.0}, _GRANITE_TRANSFORM, id="granitemoehybrid"
        ),
        pytest.param(
            "FalconH1ForCausalLM", {"lm_head_multiplier": 0.5}, HeadTransform(logit_scale=0.5), id="falcon_h1"
        ),
        pytest.param("MiniCPM3ForCausalLM", TINY_MINICPM3_CONFIG, HeadTransform(hidden_scale=24 / 64), id="minicpm3"),
        pytest.param("Gemma2ForCausalLM", {}, HeadTransform(softcap=30.0), id="gemma2"),
        pytest.param(
            "Gemma4ForConditionalGeneration", {"text_config": _GEMMA4_TEXT}, HeadTransform(softcap=30.0), id="gemma4"
        ),
    ],
)
def test_declared_transforms_are_what_the_forwards_apply(class_name, config, expected):
    """Each declaration verifies, carries the values the forward applies, and is refused once removed."""
    model_class = getattr(transformers, class_name)
    config = model_class.config_class(**config)
    verified = verify_head_transform(model_class, config)
    for field in ("hidden_scale", "logit_scale", "softcap", "vocab_size"):
        want = getattr(expected, field)
        assert getattr(verified, field) == (want if want is None else pytest.approx(want)), (field, verified)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(head_transform, "resolve_head_transform_spec", lambda _cls: _UndeclaredSpec)
        with pytest.raises(ValueError, match=f"{class_name}'s forward transforms its head path"):
            verify_head_transform(model_class, config)


def test_an_undeclared_transform_is_refused():
    """HyperCLOVAX spells Granite's ``logits_scaling`` and MULTIPLIES by it: a config-field list would
    apply it backwards, and nothing declares it, so the verification must refuse it."""
    model_class = transformers.HyperCLOVAXForCausalLM
    config = model_class.config_class(logits_scaling=0.5)
    with pytest.raises(ValueError, match="HyperCLOVAXForCausalLM's forward transforms its head path") as refused:
        verify_head_transform(model_class, config)
    message = str(refused.value)
    assert "use_chunked_grpo_logprobs" in message and "pipeline stage" in message
    assert "HeadTransformSpec" in message

    # Unit scaling is the identity the undeclared base reproduces.
    assert verify_head_transform(model_class, model_class.config_class()) == IDENTITY_HEAD_TRANSFORM


def test_verification_leaves_the_global_generator_untouched():
    """The last pipeline stage verifies once more than its peers; a probe that consumed the global
    CPU generator would shift that stage's seed stream against every other rank's."""
    model_class = transformers.GraniteForCausalLM
    config = model_class.config_class(**TINY_GRANITE_CONFIG)
    before = torch.get_rng_state()
    verify_head_transform(model_class, config)
    assert torch.equal(torch.get_rng_state(), before)


def test_resolution_follows_the_class_hierarchy():
    """A toolkit subclass (or FSDP2's in-place ``FSDP<Name>`` swap) inherits its family's declaration."""
    subclass = type("ToolkitGraniteForCausalLM", (transformers.GraniteForCausalLM,), {})
    assert resolve_head_transform_spec(subclass) is GraniteHeadTransform
    model = subclass(transformers.GraniteConfig(**TINY_GRANITE_CONFIG))
    assert resolve_head_transform(model) == HeadTransform(logit_scale=1 / TINY_GRANITE_CONFIG["logits_scaling"])


def test_a_classification_head_has_no_head_transform():
    """No output embedding, nothing a causal-LM forward applies around it: the stage projects directly."""
    model = transformers.Qwen3ForSequenceClassification(transformers.Qwen3Config(**TINY_QWEN3_CONFIG, num_labels=3))
    assert resolve_head_transform(model) == IDENTITY_HEAD_TRANSFORM


def test_a_model_that_is_not_a_transformers_model_is_refused():
    with pytest.raises(TypeError, match="not a transformers model"):
        resolve_head_transform(torch.nn.Linear(4, 4))


@pytest.mark.parametrize("family", sorted(_TINY_MODELS))
def test_chunked_sweep_reproduces_the_family_forward(family):
    """Log-probs, entropy-carrying sweep and every parameter gradient against the family's own forward.

    The upstream gradient varies per token so a per-row scale error cannot hide in a sum. For the
    transform families, the untransformed sweep is checked to be off by whole nats first, so a sweep
    that dropped the transform could not pass by coincidence.
    """
    model = _TINY_MODELS[family]()
    input_ids, attention_mask, upstream = _batch(model)

    reference = _forward_logps(model, input_ids, attention_mask)
    (reference * upstream).sum().backward()
    reference_grads = _grads(model)

    harness = _ForwardParityHarness()
    logps, entropy = harness._chunked_logps_impl(model, input_ids, attention_mask, _COMPLETION, _BATCH, True)
    (logps * upstream).sum().backward()
    grads = _grads(model)

    torch.testing.assert_close(logps, reference, atol=1e-5, rtol=1e-5)
    assert torch.isfinite(entropy).all()
    assert grads.keys() == reference_grads.keys()
    for name, grad in grads.items():
        scale = reference_grads[name].abs().max().item()
        torch.testing.assert_close(grad, reference_grads[name], atol=1e-5 * max(scale, 1.0), rtol=1e-4, msg=name)

    if family in _TRANSFORM_FAMILIES:
        assert harness._head_transform(model) != IDENTITY_HEAD_TRANSFORM
        with torch.no_grad():
            hidden = harness._get_last_hidden_state(model, input_ids, attention_mask, _COMPLETION)
            head = model.get_output_embeddings()
            untransformed = selective_log_softmax(head(hidden).float(), input_ids[:, -_COMPLETION:])
        assert (untransformed - reference).abs().max() > 0.05, f"{family}'s transform is not load-bearing here"
    else:
        assert harness._head_transform(model) == IDENTITY_HEAD_TRANSFORM


def test_the_sweep_refuses_a_family_whose_declaration_is_removed(monkeypatch):
    """With Inkling's declaration gone the chunked sweep must raise, not score the wrong distribution."""
    model = _TINY_MODELS["inkling"]()
    input_ids, attention_mask, _ = _batch(model)
    monkeypatch.setattr(head_transform, "resolve_head_transform_spec", lambda _cls: _UndeclaredSpec)
    with pytest.raises(ValueError, match="InklingForCausalLM's forward transforms its head path"):
        _ForwardParityHarness()._chunked_logps_impl(model, input_ids, attention_mask, _COMPLETION, _BATCH, False)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
