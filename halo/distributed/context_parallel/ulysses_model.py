"""The Ulysses CP model: attention layers run Ulysses on one sequence chunk per rank, scored with the boundary-aware
causal LM loss; parameter names and state dict keep the hub layout."""

from __future__ import annotations

import contextlib
import inspect
import logging
from collections.abc import Callable
from functools import cached_property

import torch
import torch.distributed as dist
import torch.nn as nn
from accelerate.utils import is_peft_model

from halo.distributed.context_parallel.config import CPConfig, CPConfigError
from halo.distributed.context_parallel.families import ULYSSES_ATTENTION_MAP
from halo.distributed.context_parallel.model_rules import reject_unsupported_ulysses_model
from halo.distributed.context_parallel.sequence_chunks import cp_local_targets, split_sequence_for_cp
from halo.distributed.expert_parallel.deepep.capacity import bump_forward_generation
from halo.distributed.expert_parallel.moe_layer.layer import named_ep_layers
from halo.kernels.logprobs import LABEL_IGNORE_INDEX, chunked_token_sum, cross_entropy_sum, logit_chunk_rows
from halo.log import warn_once
from halo.model_loading.liger.cross_entropy import FusedHeadLossUnavailable, fused_head_loss
from halo.model_loading.liger.dispatch import LIGER_CP_FUSED_LOSS_ATTR
from halo.models.forward_capture import capturing_forward_outputs
from halo.models.introspection.config_access import config_sources, get_config_field
from halo.models.introspection.unwrapping import (
    base_transformers_model,
    strip_cp_attention_level,
    unwrap_framework_wrappers,
)
from halo.models.moe.config_fields import require_router_aux_loss_coef
from halo.runtime.ranks import is_global_main_process
from halo.subclass_registry import swap_registered_modules

logger = logging.getLogger(__name__)


def _plain_named_items(items):
    """``(name, value)`` with the ``original_attention`` level dropped, the first of each resulting name kept (a param
    reachable both ways, GptOss sinks, appears once)."""
    seen = set()
    for name, value in items:
        name = strip_cp_attention_level(name)
        if name not in seen:
            seen.add(name)
            yield name, value


class UlyssesModel(nn.Module):
    """A transformers model under Ulysses CP. ``forward`` splits the inputs along the sequence axis and computes the
    loss itself, so the boundary token between chunks is scored. Attention ignores the mask: trailing pads only."""

    # Toolkit unwrap protocol (halo.models.introspection.unwrapping.unwrap_toolkit_wrappers).
    _toolkit_inner_model_attr = "model"

    def __init__(self, model: nn.Module, cp_config: CPConfig):
        super().__init__()
        self.model = model
        self.cp_config = cp_config
        reject_unsupported_ulysses_model(model, cp_config.cp_size)
        num_patched = apply_ulysses_attention(model, cp_config.process_group, cp_config.cp_size)
        logger.info(
            f"UlyssesModel: cp_size={cp_config.cp_size}, cp_rank={cp_config.cp_rank}, patched_layers={num_patched}"
        )

    def forward(self, input_ids=None, attention_mask=None, labels=None, position_ids=None, **kwargs):
        """The wrapped model's outputs on this rank's chunk, with ``loss`` against the full ``labels``. A training
        forward under the fused head loss keeps one position's logits, scores the backbone's hidden states through the
        fused kernel, and returns ``logits=None`` with ``token_accuracy``."""
        local_input_ids, local_attention_mask, local_position_ids = self._split_inputs(
            input_ids, attention_mask, position_ids, kwargs
        )
        head_loss = self._fused_head_loss if labels is not None and torch.is_grad_enabled() else None
        if labels is None and torch.is_grad_enabled() and self._fused_loss_requested and is_global_main_process():
            warn_once(
                logger,
                self._warned,
                "unlabelled_fused_loss",
                "Liger fused_linear_cross_entropy resolves on under Context Parallelism, but this training forward "
                "passed no labels, so its loss is computed from the full logits: only a labelled forward (SFT) "
                "runs the fused loss.",
            )
        if head_loss is not None:
            kwargs["logits_to_keep"] = 1
        # labels=None: the boundary-aware loss below replaces the model's.
        # The final hidden states the verified head path projects, captured off the backbone for the fused loss.
        capture = (
            capturing_forward_outputs(
                base_transformers_model(self.model).base_model, lambda out: out.last_hidden_state
            )
            if head_loss is not None
            else contextlib.nullcontext([])
        )
        with capture as captured:
            outputs = self.model(
                input_ids=local_input_ids,
                attention_mask=self._backbone_attention_mask(local_attention_mask, kwargs),
                labels=None,
                position_ids=local_position_ids,
                **kwargs,
            )
        if labels is None:
            return outputs

        if head_loss is None:
            loss = self._compute_cp_loss(outputs.logits, labels)
        else:
            loss, outputs["token_accuracy"] = self._compute_fused_cp_loss(head_loss, captured, labels)
            outputs["logits"] = None
        # A per-chunk mean: FSDP's CP-rank average already recovers the global mean, so no cp_size factor here
        # (unlike the CE sum term).
        aux_loss = getattr(outputs, "aux_loss", None)
        if aux_loss is not None:
            aux_loss = aux_loss.to(loss.device)
            if not torch.is_grad_enabled() and self.cp_config.process_group is not None:
                # Eval: rank-uniform like the CE term (see _cp_mean).
                aux_loss = aux_loss.detach().clone()
                dist.all_reduce(aux_loss, op=dist.ReduceOp.AVG, group=self.cp_config.process_group)
            loss = loss + require_router_aux_loss_coef(self.model.config) * aux_loss
        # Dict-style: the Trainer checks ``"loss" in outputs``.
        outputs["loss"] = loss
        return outputs

    def forward_hidden_states(self, input_ids=None, attention_mask=None, position_ids=None, **kwargs) -> torch.Tensor:
        """This rank's final backbone hidden states, without vocabulary logits: the caller scores the chunk and
        pairs it with the boundary labels."""
        if "labels" in kwargs:
            raise ValueError("forward_hidden_states does not accept labels; score local hidden states separately.")
        if kwargs.get("use_cache") not in (None, False):
            raise ValueError("forward_hidden_states requires use_cache=False under Context Parallelism.")
        kwargs.pop("use_cache", None)
        local_input_ids, local_attention_mask, local_position_ids = self._split_inputs(
            input_ids, attention_mask, position_ids, kwargs
        )
        base = base_transformers_model(self.model)
        backbone = base.base_model
        if backbone is base:
            raise ValueError("Context Parallelism cannot score hidden states: the model exposes no separate backbone.")
        bump_forward_generation()
        return backbone(
            input_ids=local_input_ids,
            attention_mask=self._backbone_attention_mask(local_attention_mask, kwargs),
            position_ids=local_position_ids,
            use_cache=False,
            **kwargs,
        ).last_hidden_state

    def generate(self, *args, **kwargs):
        """Refuse generation, which ``__getattr__`` (and PEFT's own) would delegate to the wrapped model."""
        raise NotImplementedError(
            "Context Parallelism does not support generate(): each rank holds one sequence chunk and "
            "the Ulysses attention has no KV-cache path. Generate from a saved checkpoint without CP."
        )

    @property
    def dtype(self):
        """The first parameter's dtype: ``PreTrainedModel.dtype`` skips non-floating parameters, which differs
        under QLoRA's ``Params4bit`` (uint8) weights."""
        return next(self.model.parameters()).dtype

    def __getattr__(self, name):
        """This module's own attribute, else the wrapped model's once ``model`` is registered: before ``__init__``
        (``cls.__new__`` in deepcopy/pickle) looking up ``self.model`` would recurse instead of raising."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            if name == "model" or "model" not in self.__dict__.get("_modules", {}):
                raise
            return getattr(self.model, name)

    def named_parameters(self, prefix="", recurse=True, **kwargs):
        """The wrapped model's parameters under their hub names. ``nn.Module.parameters`` iterates this, so an
        optimizer built from either sees each tensor once."""
        yield from _plain_named_items(self.model.named_parameters(prefix=prefix, recurse=recurse, **kwargs))

    def state_dict(self, *, destination=None, prefix="", keep_vars=False):
        """The wrapped model's state dict under hub keys, without stale dense-MLP duplicates of sparse layers. A nested
        call (``destination`` passed, a PeftModel root) gets plain ``nn.Module`` keys under ``model.``, the spelling
        PEFT adapter save and resume read through."""
        if destination is not None:
            return super().state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
        cleaned = dict(_plain_named_items(self.model.state_dict(keep_vars=keep_vars).items()))
        ep_mlp_paths = {name for name, _ in named_ep_layers(self.model)}
        for key in stale_dense_mlp_keys(cleaned.keys(), ep_mlp_paths):
            del cleaned[key]
        return {f"{prefix}{key}": value for key, value in cleaned.items()} if prefix else cleaned

    def load_state_dict(self, *args, **kwargs):
        """Refuse a load, which :meth:`state_dict`'s hub keys cannot reach: the wrapped model's keys keep the attention
        level, so a non-strict load would leave every attention projection stale."""
        raise NotImplementedError(
            "Context Parallelism does not support load_state_dict(): whole weights load when the model is constructed, "
            "and adapters through restore_adapters."
        )

    def _split_inputs(self, input_ids, attention_mask, position_ids, kwargs):
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
        local_input_ids = split_sequence_for_cp(input_ids, self.cp_config)
        local_attention_mask = (
            split_sequence_for_cp(attention_mask, self.cp_config) if attention_mask is not None else None
        )
        if position_ids is None:
            position_ids = (
                torch.arange(seq_len, device=input_ids.device, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)
            )
        local_position_ids = split_sequence_for_cp(position_ids, self.cp_config)
        return local_input_ids, local_attention_mask, local_position_ids

    def _backbone_attention_mask(self, attention_mask: torch.Tensor | None, kwargs: dict) -> torch.Tensor | None:
        """The chunk's ``attention_mask`` where the wrapped forward reads it outside attention, else ``None``:
        transformers' flash mask construction would read it back to the host (``.all()``). The router aux loss keeps
        padding out of its token statistics, so a forward computing it keeps the mask."""
        router_logits = kwargs.get("output_router_logits")
        if router_logits is None:
            config = getattr(base_transformers_model(self.model), "config", None)
            router_logits = config is not None and any(
                getattr(source, "output_router_logits", False) for source in config_sources(config)
            )
        return attention_mask if router_logits else None

    @cached_property
    def _warned(self) -> set[str]:
        return set()

    @cached_property
    def _fused_loss_requested(self) -> bool:
        """Whether the run resolves ``fused_linear_cross_entropy`` on (the Liger stamp)."""
        config = getattr(base_transformers_model(self.model), "config", None)
        return config is not None and bool(get_config_field(config, LIGER_CP_FUSED_LOSS_ATTR, False))

    @cached_property
    def _fused_head_loss(self) -> Callable[[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]] | None:
        """Liger's fused head loss where the run and head allow it; ``None`` keeps the fp32 logits loss. Resolved at
        the first training forward, after any adapter has wrapped the head."""
        if not self._fused_loss_requested:
            return None
        base = base_transformers_model(self.model)
        try:
            if "logits_to_keep" not in inspect.signature(base.forward).parameters:
                raise FusedHeadLossUnavailable(
                    f"{type(base).__name__}.forward takes no logits_to_keep, so its logits plane would be built anyway"
                )
            return fused_head_loss(base)
        except FusedHeadLossUnavailable as reason:
            if is_global_main_process():
                logger.warning(
                    f"Liger fused_linear_cross_entropy is on, but the CP loss cannot run it: {reason}. Each rank "
                    f"scores its chunk from fp32 logits instead."
                )
            return None

    def _compute_fused_cp_loss(
        self, head_loss: Callable, captured: list[torch.Tensor], labels: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """:meth:`_compute_cp_loss` scored from the chunk's final hidden states through ``head_loss``, and the
        chunk's next-token accuracy over its supervised positions."""
        if len(captured) != 1:
            raise RuntimeError(
                f"The CP fused head loss expects the backbone to run once per forward; it ran {len(captured)} times."
            )
        (hidden,) = captured
        # Every chunk position, the final rank's last included: its target is the ignore index.
        targets = cp_local_targets(labels, self.cp_config.cp_rank, self.cp_config.cp_size).reshape(-1)
        local_loss_sum, predicted = head_loss(hidden.reshape(-1, hidden.size(-1)), targets)
        supervised = targets != LABEL_IGNORE_INDEX
        accuracy = ((predicted == targets) & supervised).sum() / supervised.sum().clamp(min=1)
        return self._cp_mean(local_loss_sum, targets), accuracy

    def _compute_cp_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Causal LM loss of this rank's chunk against the full ``labels``. The fp32 cross-entropy runs over
        checkpointed chunks of the logits, so no fp32 copy of the chunk's logits plane is built."""
        vocab_size = logits.size(-1)
        targets = cp_local_targets(labels, self.cp_config.cp_rank, self.cp_config.cp_size).reshape(-1)
        local_loss_sum = chunked_token_sum(
            cross_entropy_sum, logits.reshape(-1, vocab_size), targets, logit_chunk_rows(vocab_size)
        )
        return self._cp_mean(local_loss_sum, targets)

    def _cp_mean(self, local_loss_sum: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """This rank's loss term from its chunk's summed loss over ``targets``: the sum over the group's token count
        weighs every token equally, and the ``cp_size`` factor cancels FSDP's gradient average over the CP ranks."""
        cp_size, group = self.cp_config.cp_size, self.cp_config.process_group
        global_tokens = (targets != LABEL_IGNORE_INDEX).sum()
        if group is not None:
            dist.all_reduce(global_tokens, op=dist.ReduceOp.SUM, group=group)

        # No grad (the eval loop): the rank-uniform group mean, since HF's DP-scoped metric gather keeps only
        # cp_rank 0's copy, which loss tokens spread unevenly across chunks would bias. Training cannot take this
        # all_reduce: it is not autograd-aware, and each rank backwards its own chunk-partial sum.
        if not torch.is_grad_enabled() and group is not None:
            group_loss_sum = local_loss_sum.detach().clone()
            dist.all_reduce(group_loss_sum, op=dist.ReduceOp.SUM, group=group)
            return group_loss_sum / global_tokens.clamp(min=1).float()

        # clamp, not `if global_tokens > 0`: reading a 0-dim CUDA tensor syncs the step. An
        # all-ignored batch has local_loss_sum == 0, so the clamped denominator still yields 0.
        return local_loss_sum * cp_size / global_tokens.clamp(min=1).float()


def apply_ulysses_attention(model: nn.Module, cp_group: dist.ProcessGroup, cp_size: int) -> int:
    """Swap every supported attention layer for its Ulysses wrapper; returns how many. Raises when nothing was swapped,
    which would attend over each rank's chunk alone."""

    def build(path: str, attention: nn.Module, wrapper_cls: type) -> nn.Module:
        wrapper = wrapper_cls(attention, cp_group, cp_size)
        logger.debug(f"Patched {path} ({type(attention).__name__}) -> {wrapper_cls.__name__}: {wrapper.extra_repr()}")
        return wrapper

    patched = len(swap_registered_modules(model, ULYSSES_ATTENTION_MAP, build, descend_into_match=True))

    if patched == 0:
        raise CPConfigError(
            f"No attention layers were patched for Ulysses CP — the run would silently attend over "
            f"each rank's local sequence chunk only. The model has none of the supported attention "
            f"classes: {tuple(ULYSSES_ATTENTION_MAP)}."
        )
    logger.info(f"✓ Patched {patched} attention layers for Ulysses CP")

    return patched


def stale_dense_mlp_keys(keys, ep_mlp_paths=frozenset()):
    """The keys of a sparse layer's ``.mlp`` (one with ``experts.``/``router.`` sub-keys) that exactly match a
    dense ``.mlp`` key of the same model: stale duplicates. ``ep_mlp_paths`` exempts the EP layers, whose
    dense-named parameters are the grouped expert weights."""
    suffixes_by_prefix: dict[str, set[str]] = {}
    for key in keys:
        prefix, sep, suffix = key.partition(".mlp.")
        if sep and f"{prefix}.mlp" not in ep_mlp_paths:
            suffixes_by_prefix.setdefault(prefix, set()).add(suffix)
    sparse = {
        prefix
        for prefix, suffixes in suffixes_by_prefix.items()
        if any(suffix.startswith(("experts.", "router.")) for suffix in suffixes)
    }
    dense_names = set().union(*(suffixes for prefix, suffixes in suffixes_by_prefix.items() if prefix not in sparse))
    return {f"{prefix}.mlp.{suffix}" for prefix in sparse for suffix in suffixes_by_prefix[prefix] & dense_names}


def find_ulysses_model(model: nn.Module) -> UlyssesModel | None:
    model = unwrap_framework_wrappers(model)
    if is_peft_model(model):
        model = model.get_base_model()
    return model if isinstance(model, UlyssesModel) else None
