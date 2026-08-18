"""High-level CP model wrapper: state-dict remapping and CP-aware loss.

:class:`UlyssesCPModelWrapper` patches attention modules for Ulysses CP, computes
the boundary-aware causal LM loss itself (no token dropped at chunk boundaries), and
remaps state-dict keys to pre-patching paths so checkpoints stay HF-compatible.
:func:`patch_model_for_cp` is the idempotent entry point used by :mod:`.loading`.
"""

from __future__ import annotations

import logging

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from src.data.spans import LABEL_IGNORE_INDEX
from src.distributed.context_parallel.base_layer import UlyssesAttentionBase
from src.distributed.context_parallel.config import (
    CPConfig,
    cp_boundary_shift,
    split_sequence_for_cp,
)
from src.distributed.context_parallel.key_mapping import strip_cp_attention_prefix
from src.distributed.context_parallel.patching import patch_attention_for_ulysses
from src.distributed.expert_parallel.base_layer import EPMoELayerBase
from src.models.loading.config_levels import get_config_field

logger = logging.getLogger(__name__)


def _reject_left_padding(attention_mask) -> None:
    """Raise if any row is left-padded.

    The Ulysses path drops the mask (dense flash attention with ``causal=`` only, no varlen), so it
    tolerates only padding a causal mask already ignores: trailing pads with ignored labels. With
    leading pads every real token attends them and the loss differs from the same batch without CP.

    Checked on every forward rather than cached, because padding varies per batch:
    ``DataCollatorForSMPO`` left-pads prompts and right-pads completions, so a batch with
    equal-length prompts carries only trailing pads while the next ragged one carries leading pads.
    The check costs one device→host sync per step.
    """
    if attention_mask is None:
        return
    # A left-padded row starts with 0 and contains a 1 later; an all-zero row carries no tokens.
    starts_with_pad, has_tokens = torch.stack([(attention_mask[:, 0] == 0).any(), attention_mask.any()]).cpu().tolist()
    if starts_with_pad and has_tokens:
        raise ValueError(
            "Context Parallelism received a LEFT-padded batch (row starts with attention_mask == 0). "
            "The CP attention path ignores the mask and runs dense causal attention, so every real "
            "token would attend the leading pads and the loss would differ from the same batch "
            "without CP. Use right padding (tokenizer.padding_side='right'); SMPO's collator "
            "left-pads prompts, so under CP run it with per_device_train_batch_size=1, where no "
            "padding is emitted."
        )


def stale_dense_mlp_keys(keys, ep_mlp_paths=frozenset()):
    """Stale duplicate dense-MLP keys to drop from a CP-wrapped model's state dict.

    Everything is derived from the state dict and the model itself:

    - a layer counts as sparse when its ``.mlp`` scope carries ``experts.`` / ``router.`` sub-keys;
      ``ep_mlp_paths`` (module paths of live :class:`EPMoELayerBase` instances) exempts EP-wrapped
      layers, whose dense-named params are the grouped expert weights;
    - dense-MLP names come from the model's genuinely dense ``.mlp`` layers (hybrid families), and a
      sparse key is stale only on an exact relative-name match, so a substring collision cannot drop
      a live param.
    """
    suffixes_by_prefix: dict[str, set[str]] = {}
    for key in keys:
        prefix, sep, suffix = key.partition(".mlp.")
        if sep:
            suffixes_by_prefix.setdefault(prefix, set()).add(suffix)

    def is_sparse(suffixes: set[str]) -> bool:
        return any(s.startswith(("experts.", "router.")) for s in suffixes)

    dense_param_names: set[str] = set()
    for prefix, suffixes in suffixes_by_prefix.items():
        if not is_sparse(suffixes) and f"{prefix}.mlp" not in ep_mlp_paths:
            dense_param_names |= suffixes

    stale: set[str] = set()
    for prefix, suffixes in suffixes_by_prefix.items():
        if not is_sparse(suffixes) or f"{prefix}.mlp" in ep_mlp_paths:
            continue
        stale.update(f"{prefix}.mlp.{suffix}" for suffix in suffixes & dense_param_names)
    return stale


class UlyssesCPModelWrapper(nn.Module):
    """Wrapper that turns a HuggingFace model into a Ulysses CP model.

    On construction, supported attention layers are swapped for Ulysses wrappers. On
    forward, inputs are split along the sequence axis and loss is computed manually so
    boundary tokens between chunks contribute correctly. State-dict / parameter iteration
    strip the ``original_attention.`` prefix so checkpoints match the unwrapped layout.
    """

    # Toolkit unwrap protocol (src.models.structure.unwrap_model): peel the wrapper without importing this class.
    _toolkit_inner_model_attr = "model"

    def __init__(
        self,
        model: nn.Module,
        cp_config: CPConfig,
    ):
        super().__init__()
        self.model = model
        self.cp_config = cp_config
        self.cp_group = cp_config.process_group
        self.cp_size = cp_config.cp_size
        self.cp_rank = cp_config.cp_rank

        num_patched = patch_attention_for_ulysses(
            self.model,
            self.cp_group,
            self.cp_size,
            validate=True,
        )
        # Collected once at patch time: forward publishes the full-sequence position_ids onto these,
        # and re-walking the module tree every step would cost more than the publish itself.
        self._attention_layers = [m for m in self.model.modules() if isinstance(m, UlyssesAttentionBase)]

        logger.info(
            f"UlyssesCPModelWrapper: cp_size={self.cp_size}, cp_rank={self.cp_rank}, patched_layers={num_patched}"
        )

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        position_ids=None,
        **kwargs,
    ):
        """Forward with Ulysses CP and boundary-aware causal LM loss.

        Each rank holds ``seq_len / cp_size`` tokens but sees the full sequence in attention after
        the all-to-all. Causal LM shifts labels, so a plain split would lose each rank's prediction
        of the next chunk's first token; that token is passed through as the boundary label and
        predicted from the current chunk's last logit (the final rank has none).
        """
        if any(kwargs.get(key) is not None for key in ("pixel_values", "pixel_attention_mask")):
            # Multimodal CP unsupported: pixel features don't slice by token chunk, mrope is 3D.
            raise ValueError(
                "Context Parallelism supports text-only inputs: got multimodal features "
                "(pixel_values / pixel_attention_mask). Train VLMs without CP."
            )
        if input_ids is None:
            raise ValueError(
                "Context Parallelism needs input_ids: it splits the batch along the token axis and "
                "re-pairs each chunk with its labels, which an inputs_embeds-only call cannot "
                "supply. Pass input_ids, or train without CP."
            )
        batch_size, seq_len = input_ids.shape

        if seq_len % self.cp_size != 0:
            raise ValueError(f"Sequence length {seq_len} must be divisible by cp_size {self.cp_size}")

        _reject_left_padding(attention_mask)

        chunk_size = seq_len // self.cp_size
        start = self.cp_rank * chunk_size
        end = start + chunk_size
        is_last_rank = self.cp_rank == self.cp_size - 1

        local_input_ids = split_sequence_for_cp(input_ids, self.cp_config)
        local_attention_mask = (
            split_sequence_for_cp(attention_mask, self.cp_config) if attention_mask is not None else None
        )

        if labels is not None:
            local_labels = split_sequence_for_cp(labels, self.cp_config)
            boundary_label = labels[:, end : end + 1].contiguous() if not is_last_rank else None
        else:
            local_labels = None
            boundary_label = None

        if position_ids is None:
            position_ids = (
                torch.arange(seq_len, device=input_ids.device, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)
            )
        local_position_ids = split_sequence_for_cp(position_ids, self.cp_config)

        # The legacy attention path RoPEs after the all-to-all, where Q/K span the whole sequence, so
        # its hooks (Mistral4's llama-4 scale) need the full positions this wrapper holds before the
        # split. Not cleared afterwards: gradient-checkpoint recompute re-runs attention during the
        # backward, and the CP trainers run one forward per backward.
        for layer in self._attention_layers:
            layer.global_position_ids = position_ids

        # labels=None: the boundary-aware loss below replaces the model's.
        outputs = self.model(
            input_ids=local_input_ids,
            attention_mask=local_attention_mask,
            labels=None,
            position_ids=local_position_ids,
            **kwargs,
        )

        if local_labels is not None:
            loss = self._compute_cp_loss(
                outputs.logits,
                local_labels,
                boundary_label,
                is_last_rank,
            )

            # A per-chunk mean: FSDP's CP-rank average already recovers the global mean, so no
            # cp_size factor here (unlike the CE sum term).
            aux_loss = getattr(outputs, "aux_loss", None)
            if aux_loss is not None:
                aux_loss = aux_loss.to(loss.device)
                if not torch.is_grad_enabled() and self.cp_group is not None:
                    # Eval: the DP-scoped metric gather keeps one CP sibling's copy, so the
                    # chunk-local aux must be averaged over the group here rather than through the
                    # training-only grad average.
                    aux_loss = aux_loss.detach().clone()
                    dist.all_reduce(aux_loss, op=dist.ReduceOp.AVG, group=self.cp_group)
                loss = loss + self._router_aux_loss_coef() * aux_loss

            # Trainer checks `"loss" in outputs` — needs dict-style assignment.
            outputs["loss"] = loss

        return outputs

    def _router_aux_loss_coef(self) -> float:
        """The family's MoE router aux-loss weight. Raises when the config declares none.

        The model emitted an ``aux_loss`` term, so it must also say how to weigh it; a stand-in
        default would train a different objective than the same config without CP.
        """
        config = self.model.config
        coef = get_config_field(config, "router_aux_loss_coef")
        if coef is None:
            raise ValueError(
                f"The model returned an `aux_loss` but its config ({type(config).__name__}) declares "
                f"no `router_aux_loss_coef`, so Context Parallelism cannot weigh the router balancing "
                f"term. Set router_aux_loss_coef on the model config (model_init_kwargs), or "
                f"disable the aux loss (moe_balancing) for this family."
            )
        return coef

    def _compute_cp_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        boundary_label: torch.Tensor | None,
        is_last_rank: bool,
    ) -> torch.Tensor:
        """Causal LM loss with boundary handling and global (sum/global_tokens) normalization.

        For non-final ranks the last logit predicts the next chunk's first label
        (``boundary_label``). Sum-normalized (not local mean) so every token weighs equally
        regardless of CP rank; the ``× cp_size`` factor cancels FSDP's grad average over CP ranks.
        """
        vocab_size = logits.size(-1)

        shift_logits, shift_labels = cp_boundary_shift(logits, labels, boundary_label, is_last_rank)

        # fp32 CE like HF's ForCausalLMLoss / Liger FLCE: bf16 softmax+CE rounds differently.
        shift_logits = shift_logits.reshape(-1, vocab_size).float()
        shift_labels = shift_labels.reshape(-1)

        local_loss_sum = F.cross_entropy(shift_logits, shift_labels, ignore_index=LABEL_IGNORE_INDEX, reduction="sum")

        global_tokens = (shift_labels != LABEL_IGNORE_INDEX).sum()
        if self.cp_group is not None:
            dist.all_reduce(global_tokens, op=dist.ReduceOp.SUM, group=self.cp_group)

        # Eval (no grad): return the rank-uniform group mean. HF's DP-scoped metric gather keeps only
        # cp_rank 0's copy, so with loss tokens unevenly spread across chunks (any completion-masked
        # eval set) the rank-varying training form below would bias eval_loss by chunk 0's share. The
        # training path cannot use this all_reduce: it is not autograd-aware, and each rank must
        # backward its own chunk-partial sum.
        if not torch.is_grad_enabled() and self.cp_group is not None:
            group_loss_sum = local_loss_sum.detach().clone()
            dist.all_reduce(group_loss_sum, op=dist.ReduceOp.SUM, group=self.cp_group)
            return group_loss_sum / global_tokens.clamp(min=1).float()

        # clamp, not `if global_tokens > 0`: reading a 0-dim CUDA tensor syncs the step. An
        # all-ignored batch has local_loss_sum == 0, so the clamped denominator still yields 0.
        return local_loss_sum * self.cp_size / global_tokens.clamp(min=1).float()

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None, every_n_layers: int = 1):
        """Enable gradient checkpointing on the wrapped model."""
        self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs, every_n_layers=every_n_layers)

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing on the wrapped model."""
        self.model.gradient_checkpointing_disable()

    @property
    def dtype(self):
        """The first parameter's dtype, unlike the delegated ``PreTrainedModel.dtype``, which skips
        non-floating parameters. The two differ under QLoRA, whose ``Params4bit`` weights are uint8;
        ``config`` and ``device`` carry no such difference and stay delegated."""
        return next(self.model.parameters()).dtype

    def generate(self, *args, **kwargs):
        """Raise instead of letting ``__getattr__`` delegate generation to the wrapped model.

        Delegated, the patched layers would all-to-all over the CP group with no cache path, and the
        legacy-path hooks would re-read the ``global_position_ids`` of the previous forward. PEFT
        delegates through its own ``__getattr__``, so a caller-side isinstance guard cannot cover it.
        """
        raise NotImplementedError(
            "Context Parallelism does not support generate(): each rank holds one sequence chunk and "
            "the Ulysses attention has no KV-cache path. Generate from a saved checkpoint without CP."
        )

    def __getattr__(self, name):
        """Resolve on the wrapper first (``nn.Module`` params/buffers/submodules, incl. ``model``),
        then delegate to the wrapped model.

        Delegation is gated on ``model`` being registered: before ``__init__`` runs (``cls.__new__``
        during deepcopy/pickle, which then probes ``__setstate__``) the fallback would look up
        ``self.model``, re-enter here and recurse instead of raising the expected ``AttributeError``.
        """
        try:
            return super().__getattr__(name)
        except AttributeError:
            if name == "model" or "model" not in self.__dict__.get("_modules", {}):
                raise
            return getattr(self.model, name)

    def parameters(self, recurse=True):
        """The inner model's parameters, not ``nn.Module``'s walk of this wrapper.

        That walk yields each attention weight twice, under the wrapper's own name and under
        ``original_attention.``. :meth:`named_parameters` deduplicates, and this getter must agree
        with it, or an optimizer built from it steps the same tensor twice.
        """
        return self.model.parameters(recurse=recurse)

    def named_parameters(self, prefix="", recurse=True, **kwargs):
        seen = set()
        for name, param in self.model.named_parameters(prefix=prefix, recurse=recurse, **kwargs):
            clean_name = strip_cp_attention_prefix(name)
            if clean_name not in seen:
                seen.add(clean_name)
                yield clean_name, param

    def state_dict(self, *args, destination=None, prefix="", keep_vars=False):
        """Return the state dict with the ``original_attention.`` prefix stripped for HF compatibility.

        Params present both directly on the wrapper and via ``original_attention`` (GptOss sinks) are
        deduplicated, the direct one winning. Stale duplicate dense-MLP keys are dropped on sparse
        (MoE) layers only.

        A nested call — an outer module's ``state_dict`` recursion (a PeftModel root) passing
        ``destination``/``prefix`` — gets plain ``nn.Module`` behavior instead: raw module-tree keys
        under the wrapper's own ``model.`` level, which is the spelling ``named_parameters()`` and
        ``load_state_dict()`` resolve from that root, since recursion bypasses those overrides.
        Cleaning there instead (or forwarding the shared ``destination`` to the inner model, which
        writes raw keys at the wrapper's prefix and collapses its ``model.`` level) would respell the
        root's state dict away from its own load path, and PEFT adapter save/resume reads through
        this seam.
        """
        if args:  # legacy positional (destination, prefix, keep_vars), as nn.Module accepts
            if destination is None:
                destination = args[0]
            if len(args) > 1 and prefix == "":
                prefix = args[1]
            if len(args) > 2 and keep_vars is False:
                keep_vars = args[2]

        if destination is not None:
            return super().state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)

        state_dict = self.model.state_dict(keep_vars=keep_vars)

        cleaned = {}
        for key, value in state_dict.items():
            clean_key = strip_cp_attention_prefix(key)
            if clean_key not in cleaned:
                cleaned[clean_key] = value

        ep_mlp_paths = {name for name, module in self.model.named_modules() if isinstance(module, EPMoELayerBase)}
        for key in stale_dense_mlp_keys(cleaned.keys(), ep_mlp_paths):
            del cleaned[key]

        if prefix:
            cleaned = {f"{prefix}{key}": value for key, value in cleaned.items()}
        return cleaned

    def load_state_dict(self, *args, **kwargs):
        return self.model.load_state_dict(*args, **kwargs)


def patch_model_for_cp(model: nn.Module, cp_config: CPConfig) -> nn.Module:
    """Wrap a model for Ulysses CP. Idempotent for the same ``cp_config`` object.

    A re-wrap cannot retarget the patched layers' process groups, so any other config raises,
    including one with the same ``cp_size`` but different groups.
    """
    if cp_config.cp_size == 1:
        logger.info("CP size is 1, no wrapping needed")
        return model

    if isinstance(model, UlyssesCPModelWrapper):
        if model.cp_config is not cp_config:
            raise ValueError(
                f"Model already CP-wrapped with cp_size={model.cp_size}; cannot re-wrap with a "
                f"different CPConfig (cp_size={cp_config.cp_size}) — the patched layers' process "
                f"groups cannot be retargeted."
            )
        return model

    logger.info(f"Wrapping model for Ulysses CP: cp_size={cp_config.cp_size}")
    return UlyssesCPModelWrapper(model, cp_config)
