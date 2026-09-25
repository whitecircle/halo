"""The head-path contract: what a causal LM's forward applies between its backbone and its logits.

The chunked GRPO log-prob sweep and the last pipeline stage compute logits from the backbone's last
hidden state rather than through the model's forward. A family whose forward does more than project
through the output embedding — Gemma's softcap, Cohere's ``logit_scale``, Granite's ``logits_scaling``
division, Inkling's μP hidden division and vocabulary truncation — must have that transform applied
on both paths, or they score a different distribution than the model samples from and nothing
raises. :class:`HeadTransform` is that transform, and both paths apply the one
:func:`resolve_head_transform` returns.

A family declares its transform on a :class:`HeadTransformSpec` subclass. The declaration is never
trusted: :func:`verify_head_transform` runs the family's own forward on a meta-device shell whose
backbone emits a fixed hidden state and whose output embedding is a small real stand-in, and refuses
a family whose logits the declaration does not reproduce. The verdict reads the class, the config and
whether the model has an output embedding at all, never a weight, so every rank of a sharded model,
and every pipeline stage before its head is dropped, reaches the same one.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
import torch.nn as nn
from transformers.modeling_outputs import BaseModelOutputWithPast

from src.distributed.module_registry import build_hf_module_name_map
from src.models.loading.config_levels import get_config_field, text_config
from src.models.structure import base_transformers_model, transformers_model_class

# The probe's hidden state is diagonal with log-spaced magnitudes, so its logits span 1e-2..1e3 times
# the stand-in weights: a softcap of any practical width saturates on some positions and stays linear
# on others, and a missing scale or truncation moves every position.
_PROBE_POSITIONS = 8
_PROBE_MAGNITUDES = (-2.0, 3.0)
_PROBE_SEED = 0
# The forward and the declaration compute the same fp32 function in possibly different op forms
# (``x / s`` against ``x * (1 / s)``), so they agree to rounding; a wrong transform is off by orders.
_PROBE_RTOL = 1e-5


@dataclass(frozen=True)
class HeadTransform:
    """``softcap(logit_scale · head(hidden_scale · hidden))`` over the first ``vocab_size`` columns.

    Each ``None`` field is skipped, so the default is the bare output embedding. The order is the
    one every declaring family's forward uses; the numeric verification holds each declaration to it.
    """

    hidden_scale: float | None = None
    logit_scale: float | None = None
    softcap: float | None = None
    vocab_size: int | None = None

    def scale_hidden(self, hidden: torch.Tensor) -> torch.Tensor:
        """``hidden`` as the family feeds it to the output embedding."""
        return hidden if self.hidden_scale is None else hidden * self.hidden_scale

    def kept_rows(self, weight: torch.Tensor, bias: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor | None]:
        """The output-embedding rows whose logits the family returns (all of them unless truncated)."""
        if self.vocab_size is None:
            return weight, bias
        return weight[: self.vocab_size], None if bias is None else bias[: self.vocab_size]

    def project(self, head: nn.Module, hidden: torch.Tensor) -> torch.Tensor:
        """The logits the family's forward returns for ``hidden``, computed through ``head``."""
        logits = head(self.scale_hidden(hidden))
        if self.vocab_size is not None:
            logits = logits[..., : self.vocab_size]
        if self.logit_scale is not None:
            logits = logits * self.logit_scale
        if self.softcap is not None:
            logits = torch.tanh(logits / self.softcap) * self.softcap
        return logits


IDENTITY_HEAD_TRANSFORM = HeadTransform()


class HeadTransformSpec:
    """A family's declared head transform, keyed by its causal-LM class names.

    Subclasses claim ``HF_MODULE_NAMES`` and override :meth:`transform`; a family with no spec gets the
    base, which applies ``final_logit_softcapping`` wherever the config sets it (the one head
    transform transformers spells alike across families). Resolution walks the model class's MRO, so a
    toolkit subclass or a framework class swap inherits its family's declaration.
    """

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


HEAD_TRANSFORM_SPEC_MAP = build_hf_module_name_map(HeadTransformSpec, "causal-LM head")


def resolve_head_transform_spec(model_class: type) -> type[HeadTransformSpec]:
    """The spec claiming ``model_class`` or its nearest claimed ancestor, else the base."""
    return next(
        (
            HEAD_TRANSFORM_SPEC_MAP[cls.__name__]
            for cls in model_class.__mro__
            if cls.__name__ in HEAD_TRANSFORM_SPEC_MAP
        ),
        HeadTransformSpec,
    )


class _ProbeBackboneOutput(BaseModelOutputWithPast):
    """The probe backbone's output: its hidden state, and ``None`` for every other field a family's
    forward reads off it (router logits, image features, rope deltas)."""

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        return None


def _probe_inputs(vocab: int, with_bias: bool) -> tuple[torch.Tensor, nn.Linear]:
    """The probe's ``(hidden, stand-in output embedding)``, deterministic."""
    generator = torch.Generator().manual_seed(_PROBE_SEED)
    hidden = torch.diag(torch.logspace(*_PROBE_MAGNITUDES, _PROBE_POSITIONS)).unsqueeze(0)
    head = nn.Linear(_PROBE_POSITIONS, vocab, bias=with_bias)
    with torch.no_grad():
        head.weight.copy_(torch.randn(head.weight.shape, generator=generator))
        if with_bias:
            head.bias.copy_(torch.randn(head.bias.shape, generator=generator))
    return hidden, head


def _probe_logits(model_class: type, config, declared: HeadTransform) -> tuple[torch.Tensor, torch.Tensor] | None:
    """``(forward logits, declared logits)`` from one probe, or ``None`` for a model with no output
    embedding (a classification or reward head, which the hidden-state paths project directly).

    The shell is built on the meta device from the class and config alone, so no weight is allocated;
    only the stand-in head and the probe hidden state are real, and the family's forward runs every
    op it applies around them.
    """
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
    backbone = shell.base_model
    if backbone is shell:
        raise ValueError(f"{model_class.__name__} exposes no backbone distinct from its head (base_model is itself).")
    hidden, standin = _probe_inputs(head.out_features, head.bias is not None)
    shell.set_submodule(next(name for name, module in shell.named_modules() if module is head), standin)

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
    """The declared head transform of ``model_class`` under ``config``, verified against its forward.

    Raises:
        ValueError: the family's forward returns logits its declaration does not reproduce, or the
            probe cannot run it. A path computing logits from the hidden state would then score a
            different distribution than the model's own forward, silently.
    """
    try:
        declared = resolve_head_transform_spec(model_class).transform(config)
        # Only the last pipeline stage re-verifies when it is built; a probe that advanced the global
        # generator would leave that stage's seed stream shifted against every other rank's.
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
    # Tolerance per position: the probe rows span five decades, so a plane-wide floor would leave the
    # small-magnitude rows unchecked.
    row_scale = expected.abs().amax(dim=-1, keepdim=True)
    matches = logits.shape == expected.shape and bool(
        ((logits.float() - expected).abs() <= _PROBE_RTOL * (row_scale + expected.abs())).all()
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
            f"HeadTransformSpec in src/models/head_transform.py, or run without those paths."
        )
    return declared


def resolve_head_transform(model: nn.Module) -> HeadTransform:
    """The verified head transform of ``model``, under any framework, toolkit or PEFT wrapper.

    Identity without verification for a model with no output embedding (a classification or reward
    head, which no causal-LM forward transforms).

    Raises:
        TypeError: ``model`` is not a transformers model, so there is no family forward to verify.
        ValueError: see :func:`verify_head_transform`.
    """
    base = base_transformers_model(model)
    model_class = transformers_model_class(base)
    if model_class is None:
        raise TypeError(f"{type(base).__name__} is not a transformers model; its head path has no forward to verify.")
    if base.get_output_embeddings() is None:
        return IDENTITY_HEAD_TRANSFORM
    return verify_head_transform(model_class, base.config)
