"""A model's last hidden state, read past its own forward by a path that applies the output embedding itself: the
redirection that keeps the wrapper's forward hooks firing, the EP capacity scope and the verified head transform."""

import inspect

import torch
from trl.models.utils import _ForwardRedirection

from halo.distributed.expert_parallel.deepep.capacity import bump_forward_generation
from halo.models.head_transform import HeadTransform, resolve_head_transform
from halo.models.introspection.module_tree import hidden_state_backbone
from halo.models.introspection.unwrapping import base_transformers_model


def _takes_pixel_values(module: torch.nn.Module) -> bool:
    return "pixel_values" in inspect.signature(module.forward).parameters


def reject_backbone_without_image_inputs(model: torch.nn.Module) -> None:
    """Refuse a model whose forward takes images its :func:`hidden_state_backbone` does not: that forward builds the
    image features into the text model's inputs itself, so a path entering the backbone would drop every image. Reads
    signatures only, so every rank reaches the same verdict."""
    base = base_transformers_model(model)
    backbone = hidden_state_backbone(base)
    if _takes_pixel_values(base) and not _takes_pixel_values(backbone):
        raise ValueError(
            f"{type(base).__name__} builds its image features into the inputs of its text model "
            f"{type(backbone).__name__}, which takes no pixel_values, and the hidden-state path enters that text "
            f"model directly, so it would drop every image. Set use_chunked_distillation_loss: false, or load the "
            f"text model alone for a text-only run (text_only_model: true)."
        )


class HiddenStateForwardMixin:
    """Read a model's last hidden state in place of its logits, inside the wrapper's forward."""

    @property
    def _chunked_forward_redirection(self) -> _ForwardRedirection:
        """Reuse TRL's ``_forward_redirection`` if built (only when ``use_liger_kernel``), else make one. A call
        through it runs inside the wrapper's forward, so FSDP2 has gathered the root parameters, ``lm_head``
        included, and gathers them again before the backward reaches what the call returned."""
        fr = getattr(self, "_forward_redirection", None)
        if fr is None:
            fr = _ForwardRedirection()
            self._forward_redirection = fr
        return fr

    def _backbone_hidden_state(self, unwrapped_model, input_ids, attention_mask, logits_to_keep):
        """Open its own EP capacity scope: it enters the backbone directly, past the pre-hook that scopes the shared
        DeepEP arena, and each dense row carries its own length."""
        bump_forward_generation()
        return self._get_last_hidden_state(unwrapped_model, input_ids, attention_mask, logits_to_keep)

    def _get_last_hidden_state(self, unwrapped_model, input_ids, attention_mask, logits_to_keep):
        """Forward the backbone, drop the final position (the causal next-token shift), keep the last
        ``logits_to_keep``."""
        backbone = hidden_state_backbone(unwrapped_model)
        hidden = backbone(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).last_hidden_state
        return hidden[:, :-1, :][:, -logits_to_keep:, :]  # (B, logits_to_keep, H)

    def _backbone_last_hidden_state(self, backbone, **backbone_inputs) -> torch.Tensor:
        """``[B, T, H]`` at every position of ``backbone`` (a model's :func:`hidden_state_backbone`), its inputs (a
        composite's image inputs too) passed as they are, in its own EP capacity scope as
        :meth:`_backbone_hidden_state` opens one."""
        bump_forward_generation()
        return backbone(**{**backbone_inputs, "use_cache": False}).last_hidden_state

    def _head_transform(self, unwrapped_model) -> HeadTransform:
        """Resolved once per model, since the verification builds a meta-device shell of the model's class."""
        base = base_transformers_model(unwrapped_model)
        transforms = getattr(self, "_head_transforms", None)
        if transforms is None:
            transforms = self._head_transforms = {}
        if id(base) not in transforms:
            transforms[id(base)] = resolve_head_transform(base)
        return transforms[id(base)]
