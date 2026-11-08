"""The head-path contract: what a causal LM's forward applies between its backbone and its logits. A path computing
logits from the last hidden state applies the :class:`HeadTransform`, or it scores a different distribution."""

from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import BaseModelOutputWithPast

from halo.models.introspection.config_access import get_config_field, text_config
from halo.models.introspection.model_classes import transformers_model_class
from halo.models.introspection.module_tree import hidden_state_backbone, module_path
from halo.models.introspection.unwrapping import base_transformers_model
from halo.subclass_registry import build_class_claim_map


@dataclass(frozen=True)
class HeadTransform:
    """``softcap(logit_scale · head(hidden_scale · hidden))`` over the first ``vocab_size`` columns; ``None`` skips."""

    hidden_scale: float | None = None
    logit_scale: float | None = None
    softcap: float | None = None
    vocab_size: int | None = None

    def scale_hidden(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden if self.hidden_scale is None else hidden * self.hidden_scale

    def kept_rows(self, weight: torch.Tensor, bias: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.vocab_size is None:
            return weight, bias
        return weight[: self.vocab_size], None if bias is None else bias[: self.vocab_size]

    def project(
        self,
        head: nn.Module,
        hidden: torch.Tensor,
        *,
        weight: torch.Tensor | None = None,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Logits of ``hidden``; ``weight`` and ``bias`` stand in for ``head``'s own parameters, for a caller holding
        gathered values while the module's are sharded."""
        scaled = self.scale_hidden(hidden)
        logits = head(scaled) if weight is None else F.linear(scaled, weight, bias)
        if self.vocab_size is not None:
            logits = logits[..., : self.vocab_size]
        if self.logit_scale is not None:
            logits = logits * self.logit_scale
        if self.softcap is not None:
            logits = torch.tanh(logits / self.softcap) * self.softcap
        return logits


IDENTITY_HEAD_TRANSFORM = HeadTransform()


class HeadTransformSpec:
    """Subclasses claim ``HF_MODULE_NAMES``. The base applies ``final_logit_softcapping``, the one head transform
    transformers spells alike across families."""

    HF_MODULE_NAMES: tuple[str, ...] = ()

    @classmethod
    def transform(cls, config) -> HeadTransform:
        return HeadTransform(softcap=get_config_field(config, "final_logit_softcapping"))


class InklingHeadTransform(HeadTransformSpec):
    """The μP output multiplier divides the hidden state; the padded vocabulary columns are cut off."""

    HF_MODULE_NAMES = ("InklingForCausalLM", "InklingForConditionalGeneration")

    @classmethod
    def transform(cls, config) -> HeadTransform:
        text = text_config(config)
        return HeadTransform(hidden_scale=1.0 / text.logits_mup_width_multiplier, vocab_size=text.unpadded_vocab_size)


class CohereHeadTransform(HeadTransformSpec):
    """``logit_scale`` multiplies the logits."""

    HF_MODULE_NAMES = ("CohereForCausalLM", "Cohere2ForCausalLM", "Cohere2MoeForCausalLM")

    @classmethod
    def transform(cls, config) -> HeadTransform:
        return HeadTransform(logit_scale=text_config(config).logit_scale)


class GraniteHeadTransform(HeadTransformSpec):
    """``logits_scaling`` divides the logits."""

    HF_MODULE_NAMES = (
        "GraniteForCausalLM",
        "GraniteSWAForCausalLM",
        "GraniteMoeForCausalLM",
        "GraniteMoeSWAForCausalLM",
        "GraniteMoeSharedForCausalLM",
        "GraniteMoeHybridForCausalLM",
    )

    @classmethod
    def transform(cls, config) -> HeadTransform:
        return HeadTransform(logit_scale=1.0 / text_config(config).logits_scaling)


class FalconH1HeadTransform(HeadTransformSpec):
    """``lm_head_multiplier`` multiplies the logits."""

    HF_MODULE_NAMES = ("FalconH1ForCausalLM",)

    @classmethod
    def transform(cls, config) -> HeadTransform:
        return HeadTransform(logit_scale=text_config(config).lm_head_multiplier)


class MiniCPM3HeadTransform(HeadTransformSpec):
    """``logits_scaling`` divides the hidden state before the output embedding."""

    HF_MODULE_NAMES = ("MiniCPM3ForCausalLM",)

    @classmethod
    def transform(cls, config) -> HeadTransform:
        return HeadTransform(hidden_scale=1.0 / text_config(config).logits_scaling)


def build_head_transform_spec_map() -> dict[str, type[HeadTransformSpec]]:
    return build_class_claim_map(HeadTransformSpec, "HF causal-LM head class name")


_HEAD_TRANSFORM_SPEC_MAP = build_head_transform_spec_map()


def resolve_head_transform_spec(model_class: type) -> type[HeadTransformSpec]:
    """Nearest claimed ancestor, so a toolkit subclass or a framework class swap inherits its family's declaration."""
    return next(
        (
            _HEAD_TRANSFORM_SPEC_MAP[cls.__name__]
            for cls in model_class.__mro__
            if cls.__name__ in _HEAD_TRANSFORM_SPEC_MAP
        ),
        HeadTransformSpec,
    )


class _ProbeBackboneOutput(BaseModelOutputWithPast):
    """``None`` for every field but the hidden state that a family's forward reads off it."""

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        return None


def _probe_logits(model_class: type, config, declared: HeadTransform) -> tuple[torch.Tensor, torch.Tensor] | None:
    """``(forward logits, declared logits)`` from a meta-device shell with a stand-in head; ``None`` without a head."""
    with torch.device("meta"):
        shell = model_class._from_config(copy.deepcopy(config), dtype=torch.float32, attn_implementation="eager")
    head = shell.get_output_embeddings()
    if head is None:
        return None
    if not isinstance(head, nn.Linear) or type(head).forward is not nn.Linear.forward:
        raise ValueError(
            f"{model_class.__name__}'s output embedding is a {type(head).__name__}, not a plain "
            f"nn.Linear; the hidden-state head paths compute hidden @ weight.T + bias."
        )
    backbone = hidden_state_backbone(shell)
    if backbone is None or backbone is shell:
        raise ValueError(f"{model_class.__name__} exposes no backbone distinct from its head.")
    # Log-spaced magnitudes, so a softcap of any practical width saturates on some positions and stays linear on
    # others, and a missing scale or truncation moves every position.
    hidden = torch.diag(torch.logspace(-2.0, 3.0, 8)).unsqueeze(0)
    standin = nn.Linear(hidden.shape[-1], head.out_features, bias=head.bias is not None)
    generator = torch.Generator().manual_seed(0)
    with torch.no_grad():
        standin.weight.copy_(torch.randn(standin.weight.shape, generator=generator))
        if standin.bias is not None:
            standin.bias.copy_(torch.randn(standin.bias.shape, generator=generator))
    shell.set_submodule(module_path(shell, head), standin)

    def emit_probe_hidden(*args, **kwargs) -> _ProbeBackboneOutput:
        return _ProbeBackboneOutput(last_hidden_state=hidden)

    backbone.forward = emit_probe_hidden
    shell.eval()
    with torch.no_grad():
        logits = shell(input_ids=torch.zeros(hidden.shape[:2], dtype=torch.long), use_cache=False).logits
        if logits.is_meta:
            raise ValueError(
                f"{model_class.__name__}'s forward reads model state on its head path beyond the output "
                f"embedding (its logits came back on the meta device)."
            )
        return logits, declared.project(standin, hidden)


def verify_head_transform(model_class: type, config) -> HeadTransform:
    """Reads the class and the config, never a weight, so every rank and pipeline stage reaches the same verdict."""
    try:
        declared = resolve_head_transform_spec(model_class).transform(config)
        # Only the last pipeline stage re-verifies when built; a probe advancing the global generator would leave that
        # stage's seed stream shifted against every other rank's.
        with torch.random.fork_rng(devices=[]):
            probe = _probe_logits(model_class, config, declared)
    except Exception as e:
        raise ValueError(
            f"Could not verify the head path of {model_class.__name__} ({type(e).__name__}: {e}). The "
            f"hidden-state head paths (use_chunked_grpo_logprobs, the last pipeline stage) need that "
            f"verdict; run without them."
        ) from e
    if probe is None:
        return IDENTITY_HEAD_TRANSFORM
    logits, expected = probe
    # The same fp32 function in possibly different op forms (``x / s`` against ``x * (1 / s)``) agrees to rounding; a
    # wrong transform is off by orders. Per position, as the probe rows span five decades and a plane-wide floor would
    # leave the small-magnitude rows unchecked.
    row_scale = expected.abs().amax(dim=-1, keepdim=True)
    matches = logits.shape == expected.shape and bool(
        ((logits.float() - expected).abs() <= 1e-5 * (row_scale + expected.abs())).all()
    )
    if not matches:
        deviation = (
            f"max |Δ| {(logits.float() - expected).abs().max().item():.3g}"
            if logits.shape == expected.shape
            else f"logits {tuple(logits.shape)} vs {tuple(expected.shape)}"
        )
        raise ValueError(
            f"{model_class.__name__}'s forward transforms its head path beyond what is declared for it "
            f"({declared}; {deviation} on the head-path probe). The hidden-state head paths "
            f"(use_chunked_grpo_logprobs, the last pipeline stage) would score a different "
            f"distribution than the model's own forward. Declare the family's transform on a "
            f"HeadTransformSpec, or run without those paths."
        )
    return declared


def resolve_head_transform(model: nn.Module) -> HeadTransform:
    """Under any wrapper; identity for a model with no output embedding."""
    base = base_transformers_model(model)
    model_class = transformers_model_class(base)
    if model_class is None:
        raise TypeError(f"{type(base).__name__} is not a transformers model; its head path has no forward to verify.")
    if base.get_output_embeddings() is None:
        return IDENTITY_HEAD_TRANSFORM
    return verify_head_transform(model_class, base.config)
