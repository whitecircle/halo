"""The one fused-linear-cross-entropy ``forward`` every toolkit Liger applier installs.

Reproduces the standard ``*ForCausalLM.forward`` body — backbone → ``logits_to_keep`` slice → ``lm_head``
→ loss — with Liger's fused head+CE on the training path, and refuses the two shapes it cannot reproduce
(a router aux loss, an ``lm_head`` bias). Anything else a family applies between head and loss must be
declared on its :class:`~src.kernels.liger.builder.LigerFamilySpec`, or it must not declare ``causal_lm``.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from liger_kernel.transformers.model.loss_utils import LigerForCausalLMLoss, unpack_cross_entropy_result
from liger_kernel.transformers.model.output_classes import LigerMoeCausalLMOutputWithPast

from src.models.loading.config_levels import get_config_field


def build_lce_forward(logit_scale_attr: str | None = None) -> Callable:
    """A fused-loss ``forward`` for one family.

    ``logit_scale_attr`` names a scalar the family multiplies the logits by before the loss (Cohere's
    ``logit_scale``). The fused path applies it to the hidden states instead, which is the same
    product — ``(s·h) @ Wᵀ == s·(h @ Wᵀ)`` — reassociated so the scale survives the fusion.
    """

    def lce_forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values=None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        skip_logits: bool | None = None,
        **kwargs,
    ) -> LigerMoeCausalLMOutputWithPast:
        shift_labels = kwargs.pop("shift_labels", None)
        # Taken from kwargs, NOT declared as a parameter. `honors_output_router_logits_config`
        # decides whether a family's aux loss can reach the objective by looking for this parameter
        # on `type(model).forward`, and Liger patches that forward before balancing resolves —
        # declaring it here would tell the resolver that a family whose own head has no such
        # parameter honors the flag, moving GLM-4.7-Flash off `bias_update` and onto an `aux_loss`
        # its config carries no coefficient for, i.e. no router balancing at all.
        output_router_logits = kwargs.pop("output_router_logits", None)

        # Ahead of the backbone: a loss-only eval forced onto the fused path with nothing to score
        # is a caller error, and discovering it after the forward wastes the forward.
        if skip_logits and labels is None and shift_labels is None:
            raise ValueError("skip_logits is True, but labels and shift_labels are None")

        if output_router_logits is None:
            output_router_logits = get_config_field(self.config, "output_router_logits", False)
        if output_router_logits:
            raise ValueError(
                f"Liger fused_linear_cross_entropy on {type(self).__name__} requires "
                f"output_router_logits=False: the fused loss replaces the lm_head projection the "
                f"family's own head runs before assembling a router aux loss, so nothing would add "
                f"that term to the objective. Use moe_balancing: bias_update (or none), or set "
                f"fused_linear_cross_entropy: false in liger_kernel_config."
            )

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        kept_hidden_states = hidden_states[:, slice_indices, :]

        if skip_logits is None:
            skip_logits = self.training and (labels is not None or shift_labels is not None)

        logit_scale = getattr(self, logit_scale_attr) if logit_scale_attr is not None else None

        logits = None
        loss = None
        token_accuracy = None
        if skip_logits:
            if getattr(self.lm_head, "bias", None) is not None:
                raise ValueError(
                    f"Liger fused_linear_cross_entropy computes logits from lm_head.weight alone, but "
                    f"{type(self).__name__} carries an lm_head bias — the fused loss would silently "
                    f"differ from the unfused one. Set fused_linear_cross_entropy: false for this "
                    f"checkpoint."
                )
            # LigerForCausalLMLoss, not the bare loss module: it owns the shift, the
            # `num_items_in_batch` normalisation that matches transformers' ForCausalLMLoss (a mean
            # reduction would normalise per-rank → loss ×world_size), and the token accuracy TRL asks
            # for whenever use_liger_kernel is on.
            result = LigerForCausalLMLoss(
                hidden_states=kept_hidden_states if logit_scale is None else kept_hidden_states * logit_scale,
                lm_head_weight=self.lm_head.weight,
                labels=labels,
                shift_labels=shift_labels,
                # Read off the head, not the config: a composite wrapper's `config.hidden_size` is
                # the wrapper's, and the fused loss reshapes against the projection's own width.
                hidden_size=self.lm_head.weight.shape[-1],
                **kwargs,
            )
            loss, _, token_accuracy, _ = unpack_cross_entropy_result(result)
        else:
            logits = self.lm_head(kept_hidden_states)
            if logit_scale is not None:
                logits = logits * logit_scale
            if labels is not None or shift_labels is not None:
                loss = self.loss_function(
                    logits=logits,
                    labels=labels,
                    shift_labels=shift_labels,
                    vocab_size=logits.shape[-1],
                    **kwargs,
                )

        # Liger's own output class, not the plain transformers one: TRL asks for token_accuracy
        # whenever use_liger_kernel is on (the toolkit default) and reads it back off the output —
        # returning a class with no such field costs the metric and emits a per-step warning naming
        # the wrong project.
        return LigerMoeCausalLMOutputWithPast(
            loss=loss,
            aux_loss=None,
            token_accuracy=token_accuracy,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            router_logits=getattr(outputs, "router_logits", None),
        )

    return lce_forward
