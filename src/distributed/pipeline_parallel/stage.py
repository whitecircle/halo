"""The pipeline stage module: one contiguous slice of a causal LM with bare-tensor I/O.

``torch.distributed.pipelining`` requires every stage's forward to take and return plain tensors (a
``ModelOutput`` is consumed as ``tuple(dict)`` → its keys → ``AttributeError``), and it hands
non-first stages only the received activation positionally plus the step's kwargs.

A stage is built by mutating the real model in place rather than reimplementing a decoder forward:
the backbone's layer list is sliced, the final norm is neutralized on non-last stages, and non-first
stages are driven through ``inputs_embeds`` so HF's own ``if inputs_embeds is None: embed_tokens(...)``
guard skips the embedding. Each family's rotary embeddings, mask construction, ``layer_types``,
sliding-window and MoE handling are therefore inherited, not duplicated.
"""

from __future__ import annotations

import functools
import logging
from collections.abc import Callable

import torch
import torch.nn as nn
from torch.distributed.tensor import DTensor
from torch.utils.checkpoint import checkpoint
from transformers import PretrainedConfig

from src.checkpoint.config_export import hf_architecture_name
from src.distributed.expert_parallel.base_layer import find_ep_layers
from src.distributed.expert_parallel.dispatcher import register_forward_generation_hook
from src.distributed.pipeline_parallel.split import (
    decoder_layers_attr,
    layer_types_shift_invariant,
    resolve_layer_partition,
    resolve_module_attr,
    resolve_pp_spec,
    validate_model_supports_pp,
)
from src.models.loading.config_levels import text_config
from src.models.structure import backbone_with_layers, decoder_layers

logger = logging.getLogger(__name__)


# Set by the stage-aware loader on a model whose decoder-layer list is already this rank's slice, so
# :func:`build_pipeline_stage` re-uses that partition instead of re-deriving it from the short list.
PP_STAGE_PARTITION_ATTR = "_pp_stage_partition"


def _join(*parts: str) -> str:
    """Dotted-join the non-empty parts (the backbone prefix is empty when the model IS the backbone)."""
    return ".".join(part for part in parts if part)


def split_layer_path(name: str, layer_root: str) -> tuple[int, str] | None:
    """``(layer index, remainder)`` of a ``<layer_root><index>[.<rest>]`` path, else ``None``."""
    if not name.startswith(layer_root):
        return None
    index, _, rest = name[len(layer_root) :].partition(".")
    return (int(index), rest) if index.isdigit() else None


def rebase_layer_index(name: str, src_root: str, dst_root: str, delta: int) -> str | None:
    """Move a layer path from ``src_root`` to ``dst_root``, shifting its index by ``delta``.

    Both directions of the stage naming map
    (:meth:`PipelineStageModule.global_parameter_name` / :meth:`~PipelineStageModule.local_parameter_name`)
    and the stage-aware loader's key filter go through this, so a slice and its inverse cannot drift
    apart. Returns ``None`` when ``name`` is not a layer path under ``src_root``.
    """
    parts = split_layer_path(name, src_root)
    if parts is None:
        return None
    index, rest = parts
    return f"{dst_root}{index + delta}.{rest}" if rest else f"{dst_root}{index + delta}"


class PipelineStageModule(nn.Module):
    """A contiguous slice of a causal LM, exposing the tensor-in/tensor-out contract PP needs.

    Forward takes the pipeline activation positionally — ``input_ids`` on stage 0, the previous
    stage's ``hidden_states`` elsewhere — and per-example side inputs as kwargs, which the schedule
    microbatches and delivers to every stage.

    Naming contract: the module reports its own local names everywhere and
    :meth:`global_parameter_name` is applied explicitly by whoever serializes, which keeps stage 1
    from writing stage 0's layer names. The accessors must not be overridden to return global names:
    HF derives the weight-decay set from ``named_children`` (local) and ``load_state_dict``/DCP
    resolve keys against the real module tree.
    """

    # Source-model metadata :func:`build_pipeline_stage` carries over, declared here because the
    # checkpoint writers read it off the stage: the config the reassembled checkpoint ships (with
    # ``architectures`` restamped to the live class), the generation config for a generative model,
    # and the remote-code class whose modeling module the save copies. ``config`` is left
    # un-defaulted: ``save_model_config`` skips a model that carries none, and a class-level ``None``
    # would turn that skip into an attribute error.
    config: PretrainedConfig
    generation_config = None
    _remote_code_model_class: type | None = None

    def __init__(
        self,
        backbone: nn.Module,
        head: nn.Module | None,
        is_first: bool,
        is_last: bool,
        *,
        backbone_prefix: str,
        head_attr: str,
        layer_attr: str,
        layer_offset: int,
    ):
        super().__init__()
        # Named ``model`` so ``backbone_with_layers`` finds the layer list through its ``.model`` probe.
        self.model = backbone
        self.head = head
        self.is_first = is_first
        self.is_last = is_last
        # ``(head, hidden_states, labels) -> summed loss``, installed by :class:`PipelineRuntime` on
        # the last stage when the objective decomposes over token chunks (the causal-LM CE). Forward
        # then returns the scalar loss instead of logits, so no [mb, S, V] plane is built.
        # ``None`` = return logits and let the schedule's ``loss_fn`` consume them.
        self.fused_loss_fn: Callable[[nn.Module, torch.Tensor, torch.Tensor], torch.Tensor] | None = None
        # Slicing re-bases layer indices to 0, so without these a stage claims stage 0's names on save.
        self._backbone_prefix = backbone_prefix
        self._head_attr = head_attr
        self._layer_attr = layer_attr
        self._layer_offset = layer_offset
        # The EP capacity generation must ride the module the schedule calls. EP patching registers
        # it on the model it patched — under PP the un-sliced CausalLM, whose forward no pipeline
        # runs — so without this the generation never advances and every microbatch reuses the first
        # forward's DeepEP capacity, refusing the first longer batch.
        register_forward_generation_hook(self)

    def can_generate(self) -> bool:
        """Whether the reassembled checkpoint is generative. Mirrors PreTrainedModel.

        ``save_model_config`` writes ``generation_config.json`` only for a model that reports True,
        so without this a PP save ships ``config.json`` alone and the served model falls back to
        default sampling and EOS — the custom ``eos_token_id`` sets (harmony/GPT-OSS) live nowhere
        else. Reward and classification stages report False, as their unsplit models do.
        """
        return self.generation_config is not None

    def add_model_tags(self, tags) -> None:
        """Record trainer tags (TRL calls this unguarded on every model). Mirrors PreTrainedModel."""
        tags = [tags] if isinstance(tags, str) else list(tags)
        if not hasattr(self, "model_tags"):
            self.model_tags = []
        for tag in tags:
            if tag not in self.model_tags:
                self.model_tags.append(tag)

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None, every_n_layers: int = 1) -> None:
        """Enable non-reentrant activation checkpointing on this stage's decoder layers.

        Mirrors ``PreTrainedModel.gradient_checkpointing_enable`` with two deviations:
        ``use_reentrant=True`` is rejected (it runs the layer under ``no_grad``, so FSDP2 skips its
        pre-backward hook registration and re-runs its forward inside the schedule-driven backward),
        and there is no ``enable_input_require_grads`` (a non-first stage has no input embedding, and
        its recv buffer already requires grad). ``every_n_layers`` counts from this stage's first
        local layer, so every stage checkpoints the same fraction of its own layers.
        """
        kwargs = dict(gradient_checkpointing_kwargs or {})
        if kwargs.get("use_reentrant", False):
            raise ValueError(
                "use_reentrant=True gradient checkpointing is not supported under pipeline "
                "parallelism: the reentrant checkpoint runs its forward under no_grad, so FSDP2 "
                "never registers pre-backward hooks and recompute re-enters its full forward from "
                "inside the schedule's backward. Use the default use_reentrant=False."
            )
        kwargs.setdefault("use_reentrant", False)
        if not getattr(self.model, "supports_gradient_checkpointing", False):
            raise ValueError(f"{type(self.model).__name__} does not support gradient checkpointing.")
        self.model._set_gradient_checkpointing(
            enable=True,
            gradient_checkpointing_func=functools.partial(checkpoint, **kwargs),
            every_n_layers=every_n_layers,
        )

    def gradient_checkpointing_disable(self) -> None:
        """Disable activation checkpointing (mirrors ``PreTrainedModel``)."""
        if getattr(self.model, "supports_gradient_checkpointing", False):
            self.model._set_gradient_checkpointing(enable=False)

    @property
    def is_gradient_checkpointing(self) -> bool:
        """Whether any module on this stage currently checkpoints (mirrors ``PreTrainedModel``)."""
        return any(getattr(m, "gradient_checkpointing", False) for m in self.model.modules())

    @property
    def layer_range(self) -> tuple[int, int]:
        """This stage's ``[lo, hi)`` slice of the unsplit model's decoder layers."""
        return self._layer_offset, self._layer_offset + len(getattr(self.model, self._layer_attr))

    @property
    def local_layer_root(self) -> str:
        """Dotted prefix of this stage's decoder-layer paths in its own naming."""
        return f"model.{self._layer_attr}."

    @property
    def global_layer_root(self) -> str:
        """Dotted prefix of the decoder-layer paths in the unsplit model's naming."""
        return f"{_join(self._backbone_prefix, self._layer_attr)}."

    def global_parameter_name(self, local_name: str) -> str:
        """Map a stage-local parameter / buffer / module path back to its FQN in the unsplit model."""
        rebased = rebase_layer_index(local_name, self.local_layer_root, self.global_layer_root, self._layer_offset)
        if rebased is not None:
            return rebased
        if local_name == "model" or local_name.startswith("model."):
            return _join(self._backbone_prefix, local_name[len("model.") :])
        if local_name == "head" or local_name.startswith("head."):
            return _join(self._head_attr, local_name[len("head.") :])
        return local_name

    def local_parameter_name(self, global_name: str) -> str:
        """Inverse of :meth:`global_parameter_name`: an unsplit-model FQN → this stage's local path.

        Only meaningful for names this stage holds; a layer outside its range comes back with a
        negative or out-of-range index, which resolves to no live module.
        """
        rebased = rebase_layer_index(global_name, self.global_layer_root, self.local_layer_root, -self._layer_offset)
        if rebased is not None:
            return rebased
        # Head first: with an empty backbone prefix (the model IS the backbone) every global name is
        # bare, so the backbone rule below would claim ``lm_head.weight`` as ``model.lm_head.weight``.
        if global_name == self._head_attr or global_name.startswith(f"{self._head_attr}."):
            return _join("head", global_name[len(self._head_attr) + 1 :])
        if not self._backbone_prefix:
            return _join("model", global_name)
        if global_name == self._backbone_prefix or global_name.startswith(f"{self._backbone_prefix}."):
            return _join("model", global_name[len(self._backbone_prefix) + 1 :])
        return global_name

    def ep_moe_layers(self) -> list[tuple[str, nn.Module]]:
        """This stage's EP MoE layers, by local module path: the modules whose expert weights are
        exported by their family's gather rather than by ``state_dict``."""
        return find_ep_layers(self)

    def checkpoint_name_map(self) -> dict[str, str]:
        """Global (unsplit-model) name → local ``state_dict`` key, for the tensors this stage serializes.

        The PP save and the PP resume both walk this filter+map, so a key cannot be written under one
        policy and read back under another. EP MoE entries are excluded: they hold this rank's
        internal shard layout (local expert ranges, de-interleaved splits) and are re-emitted in HF
        layout by the family gather.

        Raises:
            RuntimeError: a local name whose global name does not map back to it. The map must be
                injective — two stage tensors sharing one global name would collapse into a single
                checkpoint entry on save, and both take that entry's weights on resume.
        """
        ep_prefixes = tuple(f"{name}." for name, _ in self.ep_moe_layers())
        name_map: dict[str, str] = {}
        for local in self.state_dict():
            if local.startswith(ep_prefixes):
                continue
            global_name = self.global_parameter_name(local)
            # local_parameter_name is the declared inverse, so a collision fails the round trip.
            if self.local_parameter_name(global_name) != local:
                raise RuntimeError(
                    f"Pipeline stage tensor '{local}' maps to global name '{global_name}', which maps "
                    f"back to '{self.local_parameter_name(global_name)}' — the stage naming map is not "
                    f"invertible here, so this tensor would share a checkpoint entry with another."
                )
            name_map[global_name] = local
        return name_map

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """This stage's activation, or on the last stage its logits or its summed loss.

        ``labels`` reaches every stage (the schedule microbatches kwargs for all of them) and is used
        only by a last stage running :attr:`fused_loss_fn`; everywhere else it is inert.
        """
        if self.is_first:
            outputs = self.model(
                input_ids=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=False,
            )
        else:
            outputs = self.model(
                inputs_embeds=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=False,
            )

        hidden = outputs.last_hidden_state
        if isinstance(hidden, DTensor):
            # torch's pipelining is DTensor-unaware: its boundary buffer is a plain torch.empty.
            raise RuntimeError(
                "A pipeline stage produced a DTensor boundary activation. torch's pipelining P2P "
                "path handles plain local tensors only; a TP/SequenceParallel plan that emits "
                "DTensor activations (use_local_output=False) cannot cross a stage boundary."
            )
        if not self.is_last:
            return hidden
        if self.fused_loss_fn is None:
            return self.head(hidden)
        if labels is None:
            raise RuntimeError(
                "The last pipeline stage computes its loss inside forward (fused head) but received "
                "no `labels` kwarg, so it has no targets to score. The runtime supplies them on "
                "every call; a caller driving the schedule directly must do the same."
            )
        return self.fused_loss_fn(self.head, hidden, labels)


def module_path(model: nn.Module, module: nn.Module) -> str:
    """Dotted path of ``module`` inside ``model``, by identity match rather than name assumptions.

    Raises:
        ValueError: ``module`` is not part of ``model``'s tree. Fatal for every caller: the path is
            what names a stage's parameters globally and what the loader's droppable-prefix filter
            matches on, so a miss would otherwise surface as a bare ``StopIteration`` or an
            unfiltered key set.
    """
    path = next((name for name, mod in model.named_modules() if mod is module), None)
    if path is None:
        raise ValueError(
            f"{type(module).__name__} is not a module of {type(model).__name__}; pipeline "
            f"parallelism locates it by identity and cannot name it."
        )
    return path


def input_embedding_module(model: nn.Module) -> nn.Module:
    """``model``'s input embedding, via the HF-canonical accessor.

    The attribute name varies by family (``embed_tokens``, ``word_embeddings``, ``wte``), so a name
    guess that misses would leave the table in every stage's state_dict and the merged index would
    resolve the embedding to the last stage's untrained copy.

    Raises:
        ValueError: the model exposes no input embedding — the stage build drops it and the
            stage-aware loader names it as droppable state, and neither may guess.
    """
    getter = getattr(model, "get_input_embeddings", None)
    embedding = getter() if callable(getter) else None
    if embedding is None:
        raise ValueError(
            f"{type(model).__name__} exposes no input embedding via get_input_embeddings(); a "
            f"non-first pipeline stage must drop it and cannot locate it."
        )
    return embedding


def _drop_input_embeddings(model: nn.Module) -> None:
    """Drop the input-embedding module from a non-first stage, located by identity."""
    parent_path, _, attr = module_path(model, input_embedding_module(model)).rpartition(".")
    setattr(model.get_submodule(parent_path) if parent_path else model, attr, None)


def resolve_layer_root(model: nn.Module) -> str:
    """Dotted prefix of ``model``'s decoder-layer paths in its own (unsplit) naming.

    ``model.layers.``, ``model.language_model.layers.``, ``transformer.h.``, … — the key space the
    checkpoint and the meta shell share, and the one
    :class:`~src.distributed.pipeline_parallel.lazy_loader.PPWeightPlanner` re-bases within. The
    stage module's :attr:`PipelineStageModule.global_layer_root` composes the same string from the
    fields it stores.
    """
    backbone = backbone_with_layers(model)
    return f"{_join(module_path(model, backbone), decoder_layers_attr(backbone))}."


def slice_backbone_to_stage(model: nn.Module, lo: int, hi: int) -> None:
    """Replace the backbone's decoder-layer list with this stage's ``[lo, hi)`` slice, in place.

    The single slicing seam: the stage-aware loader applies it to a meta shell before any weight is
    materialized and :func:`build_pipeline_stage` applies it to a whole-model load, so neither can
    re-base the layer indices differently from the other.
    """
    backbone = backbone_with_layers(model)
    layer_attr = decoder_layers_attr(backbone)
    layers = getattr(backbone, layer_attr)
    setattr(backbone, layer_attr, nn.ModuleList(list(layers)[lo:hi]))


def reject_layer_type_rebase(model: nn.Module, layer_offset: int, pp_rank: int) -> None:
    """Refuse a split that would re-index a heterogeneous ``config.layer_types``.

    Several families (GptOss, Qwen3, Qwen3.5, Qwen3-Next, LFM2, Laguna) index their mask as
    ``causal_mask_mapping[config.layer_types[i]]`` with ``i`` the enumerate position in the layer
    list. Slicing re-bases that position, so a stage whose offset is not a whole number of pattern
    periods swaps sliding-window and full causal masks with no error. Stage 0 and a uniform
    ``layer_types`` are always safe, as is a family whose spec declares ``LAYER_TYPES_REBASE_SAFE``
    (its stages select per-layer inputs by the layer's own attributes, never by list position).
    """
    backbone = backbone_with_layers(model)
    if backbone is not None and resolve_pp_spec(backbone).LAYER_TYPES_REBASE_SAFE:
        return
    # A composite/VLM wrapper carries the decoder's layer_types on its text sub-config, read through
    # the same accessor ``layer_types_period`` uses: this gate rejects the offsets that period does
    # not snap the partition to, so a config the two read differently would leave a manual
    # ``pp_split`` neither snapped nor gated.
    layer_types = getattr(text_config(getattr(model, "config", None)), "layer_types", None)
    if layer_types_shift_invariant(layer_types, layer_offset):
        return  # no list, no shift, or a whole number of pattern periods: re-basing is a no-op
    raise ValueError(
        f"{type(model).__name__} selects its attention mask by position in the decoder-layer list "
        f"(config.layer_types), and pipeline stage {pp_rank} starts at layer {layer_offset}, which is "
        f"not a whole number of periods of that pattern {tuple(layer_types[:4])}... . Slicing the "
        f"list would shift every layer's type and silently apply the wrong attention mask. Choose a "
        f"pipeline_parallel_size whose stage boundaries land on the pattern period."
    )


def build_pipeline_stage(
    model: nn.Module,
    pp_rank: int,
    pp_size: int,
    moe_balancing: str = "auto",
    pp_split: list[int] | None = None,
) -> PipelineStageModule:
    """Convert ``model`` into this rank's pipeline stage, in place.

    Mutates ``model``: the backbone's layer list is sliced to this stage's range and the modules the
    stage does not hold are dropped so they occupy no memory. A model that arrives already sliced
    (the stage-aware loader stamps its partition on ``PP_STAGE_PARTITION_ATTR``) keeps that split,
    since the partition cannot be re-derived from a list that is already one stage long. Use the
    returned module afterwards, not the argument.

    Args:
        model: a loaded causal LM (``*ForCausalLM``) or VLM wrapper.
        pp_rank: this rank's stage index.
        pp_size: total stages.
        moe_balancing: the run's ``moe_balancing`` knob, checked against the aux-loss path a stage
            severs (see :func:`~src.distributed.pipeline_parallel.split.validate_model_supports_pp`).

    Raises:
        ValueError: the model cannot be split (see :mod:`.split`), or has fewer layers than stages.
    """
    validate_model_supports_pp(model, moe_balancing)
    backbone = backbone_with_layers(model)
    spec = resolve_pp_spec(backbone)

    # Where the backbone sits in the unsplit model, so parameters can be named globally again.
    backbone_prefix = module_path(model, backbone)

    partition = getattr(model, PP_STAGE_PARTITION_ATTR, None)
    pre_sliced = partition is not None
    if not pre_sliced:
        partition = resolve_layer_partition(model, pp_size, pp_split)
    lo, hi = partition[pp_rank]
    num_layers = partition[-1][1]
    is_first, is_last = pp_rank == 0, pp_rank == pp_size - 1
    if pp_rank == 0:
        logger.info(
            "Pipeline layer partition (%s): %s",
            "pp_split" if pp_split is not None else "head-weighted",
            partition,
        )

    # Every rank validates every stage's offset: a rank-local raise leaves the others at the P2P
    # handshake.
    for stage_rank, (stage_lo, _) in enumerate(partition):
        reject_layer_type_rebase(model, stage_lo, stage_rank)
    # Same whole-partition scope for the family's own boundary constraints (indexer chains,
    # input_ids-consuming layers): every rank checks every boundary, so all fail together.
    spec.validate_partition(model, partition)

    # Slicing re-bases the list index, which is why reject_layer_type_rebase must run first.
    layer_attr = decoder_layers_attr(backbone)
    if pre_sliced:
        held = len(decoder_layers(backbone))
        if held != hi - lo:
            raise RuntimeError(
                f"Pipeline stage {pp_rank} was loaded stage-aware with layers [{lo}, {hi}) but holds "
                f"{held} decoder layers. The loader's partition and the trainer's disagree, so every "
                f"saved tensor would carry the wrong global layer index."
            )
    else:
        slice_backbone_to_stage(model, lo, hi)

    if not is_last:
        # Neutralize the post-layers norm: it belongs to the end of the network, not to every cut.
        norm_attr = resolve_module_attr(backbone, spec.FINAL_NORM_ATTRS, "final norm")
        setattr(backbone, norm_attr, nn.Identity())
        # Same for the spec's declared tail modules (hyper-connection collapse): applied after the
        # last layer only, so a mid-chain stage must pass the un-collapsed stream through.
        for tail_attr in spec.TAIL_MODULE_ATTRS:
            setattr(backbone, resolve_module_attr(backbone, (tail_attr,), "tail module"), nn.Identity())

    if not is_first:
        # Never used (non-first stages run on inputs_embeds); dropped so it cannot reach this stage's state_dict.
        _drop_input_embeddings(model)

    head_attr = resolve_module_attr(model, spec.HEAD_ATTRS, "task head")
    head = getattr(model, head_attr) if is_last else None
    if not is_last:
        setattr(model, head_attr, None)

    # Family adaptation runs last, over the sliced and neutralized backbone: for the hyper-connection
    # families it rebinds the mid-chain forward that accepts the widened boundary stream.
    spec.adapt_stage_backbone(backbone, is_first=is_first, is_last=is_last)

    stage = PipelineStageModule(
        backbone,
        head,
        is_first,
        is_last,
        backbone_prefix=backbone_prefix,
        head_attr=head_attr,
        layer_attr=layer_attr,
        layer_offset=lo,
    )
    # HF Trainer and the checkpoint writers read ``model.config``; it is what a reassembled ckpt ships.
    stage.config = model.config
    # ``architectures`` still carries the hub's class (e.g. ``*ForCausalLM`` on a model built as
    # ``*ForSequenceClassification`` — PP supports reward and classification), and save_model_config
    # cannot re-derive it from this nn.Module carrier, so the live class is stamped here. An
    # ``architectures``-keyed consumer (vLLM, TGI, Auto* resolution) would otherwise serve the wrong
    # head.
    architecture = hf_architecture_name(model)
    if architecture is not None:
        stage.config.architectures = [architecture]
    # The writers hand this module to save_model_config and PretrainedConfig does not carry the
    # generation config; without the pair a PP checkpoint serves with default sampling/EOS.
    if model.can_generate():
        stage.generation_config = model.generation_config
    # Same carry for a remote-code model's defining class: the stage module's MRO has no
    # ``transformers_modules`` entry, so without it the save ships config + auto_map but no
    # modeling_*.py, which cannot be loaded back (resume re-from_pretrains the saved dir).
    stage._remote_code_model_class = next(
        (cls for cls in type(model).__mro__ if cls.__module__.startswith("transformers_modules")), None
    )

    logger.info(
        "Pipeline stage %d/%d: decoder layers [%d, %d) of %d%s%s",
        pp_rank,
        pp_size,
        lo,
        hi,
        num_layers,
        " +embedding" if is_first else "",
        f" +norm+{head_attr}" if is_last else "",
    )
    return stage
