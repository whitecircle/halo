"""The Ulysses CP model: a transformers model whose attention layers run Ulysses, fed one sequence chunk per rank
and scored with the boundary-aware causal LM loss; its parameter names and state dict keep the hub layout."""

from __future__ import annotations

import logging

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from accelerate.utils import is_peft_model

from src.distributed.context_parallel.config import CPConfig, CPConfigError
from src.distributed.context_parallel.families import ULYSSES_ATTENTION_MAP
from src.distributed.context_parallel.model_rules import reject_unsupported_ulysses_model
from src.distributed.context_parallel.sequence_chunks import cp_shift_against_full_labels, split_sequence_for_cp
from src.distributed.expert_parallel.deepep.capacity import bump_forward_generation
from src.distributed.expert_parallel.moe_layer.layer import named_ep_layers
from src.kernels.logprobs import LABEL_IGNORE_INDEX
from src.models.config_access import get_config_field
from src.models.unwrapping import base_transformers_model, strip_cp_attention_level, unwrap_framework_wrappers
from src.subclass_registry import swap_registered_modules

logger = logging.getLogger(__name__)


class UlyssesModel(nn.Module):
    """A transformers model under Ulysses CP over ``cp_config``'s process group.

    Construction swaps every supported attention layer for its Ulysses wrapper. ``forward`` splits the
    inputs along the sequence axis and computes the loss itself, so the boundary token between two chunks
    is scored. ``named_parameters`` and ``state_dict`` drop the ``original_attention`` level, so
    checkpoints keep the hub layout.
    """

    # Toolkit unwrap protocol (src.models.unwrapping.unwrap_toolkit_wrappers).
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
        """The wrapped model's outputs on this rank's chunk, with ``loss`` against the full ``labels`` when given."""
        local_input_ids, local_attention_mask, local_position_ids = self._split_inputs(
            input_ids, attention_mask, position_ids, kwargs
        )
        # labels=None: the boundary-aware loss below replaces the model's.
        outputs = self.model(
            input_ids=local_input_ids,
            attention_mask=local_attention_mask,
            labels=None,
            position_ids=local_position_ids,
            **kwargs,
        )
        if labels is None:
            return outputs

        loss = self._compute_cp_loss(outputs.logits, labels)
        # A per-chunk mean: FSDP's CP-rank average already recovers the global mean, so no cp_size factor here
        # (unlike the CE sum term).
        aux_loss = getattr(outputs, "aux_loss", None)
        if aux_loss is not None:
            aux_loss = aux_loss.to(loss.device)
            if not torch.is_grad_enabled() and self.cp_config.process_group is not None:
                # Eval: rank-uniform like the CE term (see _compute_cp_loss).
                aux_loss = aux_loss.detach().clone()
                dist.all_reduce(aux_loss, op=dist.ReduceOp.AVG, group=self.cp_config.process_group)
            # No stand-in default: it would train a different objective than the same config without CP.
            config = self.model.config
            coef = get_config_field(config, "router_aux_loss_coef")
            if coef is None:
                raise ValueError(
                    f"The model returned an `aux_loss` but its config ({type(config).__name__}) declares "
                    f"no `router_aux_loss_coef`, so Context Parallelism cannot weigh the router balancing "
                    f"term. Set router_aux_loss_coef on the model config (model_init_kwargs), or "
                    f"disable the aux loss (moe_balancing) for this family."
                )
            loss = loss + coef * aux_loss
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
            attention_mask=local_attention_mask,
            position_ids=local_position_ids,
            use_cache=False,
            **kwargs,
        ).last_hidden_state

    def generate(self, *args, **kwargs):
        """Refuse generation, which ``__getattr__`` (and PEFT's own) would delegate to the wrapped model, whose
        patched layers all-to-all over the CP group with no cache path."""
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
        """The wrapped model's state dict under hub keys, without the stale dense-MLP duplicates of sparse layers.

        A nested call (an outer module's recursion, a PeftModel root, passing ``destination``) gets plain
        ``nn.Module`` keys under this module's ``model.`` level instead: the spelling that root's own
        ``named_parameters`` and ``load_state_dict`` resolve, which PEFT adapter save and resume read through.
        """
        if destination is not None:
            return super().state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
        cleaned = dict(_plain_named_items(self.model.state_dict(keep_vars=keep_vars).items()))
        ep_mlp_paths = {name for name, _ in named_ep_layers(self.model)}
        for key in stale_dense_mlp_keys(cleaned.keys(), ep_mlp_paths):
            del cleaned[key]
        return {f"{prefix}{key}": value for key, value in cleaned.items()} if prefix else cleaned

    def load_state_dict(self, *args, **kwargs):
        return self.model.load_state_dict(*args, **kwargs)

    def _split_inputs(self, input_ids, attention_mask, position_ids, kwargs):
        """This rank's chunk of ``input_ids``, ``attention_mask`` and ``position_ids``, after the input refusals."""
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
        _reject_left_padding(attention_mask)

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

    def _compute_cp_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Causal LM loss of this rank's chunk against the full ``labels``: a non-final rank's last logit predicts
        the next chunk's first label, and the sum over the group's token count weighs every token equally; the
        ``cp_size`` factor cancels FSDP's gradient average over the CP ranks."""
        vocab_size = logits.size(-1)
        cp_size, group = self.cp_config.cp_size, self.cp_config.process_group
        shift_logits, shift_labels = cp_shift_against_full_labels(logits, labels, self.cp_config.cp_rank, cp_size)

        # fp32 CE like HF's ForCausalLMLoss / Liger FLCE: bf16 softmax+CE rounds differently.
        shift_logits = shift_logits.reshape(-1, vocab_size).float()
        shift_labels = shift_labels.reshape(-1)

        local_loss_sum = F.cross_entropy(shift_logits, shift_labels, ignore_index=LABEL_IGNORE_INDEX, reduction="sum")

        global_tokens = (shift_labels != LABEL_IGNORE_INDEX).sum()
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
    """Swap every supported attention layer of ``model`` for its Ulysses wrapper over ``cp_group``; returns how
    many. Raises :class:`CPConfigError` when nothing was swapped, which would attend over each rank's chunk
    alone."""

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


def _reject_left_padding(attention_mask) -> None:
    """Raise on a left-padded row: the Ulysses path drops the mask and attends causally, so it tolerates only
    trailing pads, while leading ones would be attended by every real token.

    Checked on every forward (one device-to-host sync): under a collator that left-pads prompts, a batch whose
    prompts happen to be equal-length certifies nothing about the next one.
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
            "without CP. Use right padding (tokenizer.padding_side='right')."
        )


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


def _plain_named_items(items):
    """``(name, value)`` pairs with the ``original_attention`` level dropped from each name, the first of each
    resulting name kept (a param reachable both directly and through that level, GptOss sinks, appears once)."""
    seen = set()
    for name, value in items:
        name = strip_cp_attention_level(name)
        if name not in seen:
            seen.add(name)
            yield name, value


def find_ulysses_model(model: nn.Module) -> UlyssesModel | None:
    """The :class:`UlyssesModel` under framework and PEFT wrappers, or ``None``."""
    model = unwrap_framework_wrappers(model)
    if is_peft_model(model):
        model = model.get_base_model()
    return model if isinstance(model, UlyssesModel) else None
