"""A pipeline stage: the real model mutated in place, its layer list sliced and post-layers modules neutralized off the
last stage. Non-first stages run on ``inputs_embeds`` so HF's own ``embed_tokens`` guard skips the embedding."""

from __future__ import annotations

import functools
import inspect
import logging
from dataclasses import dataclass, replace
from types import MethodType

import torch
import torch.nn as nn
from torch.distributed.tensor import DTensor
from torch.utils.checkpoint import checkpoint
from transformers import PretrainedConfig

from halo.checkpoint.config_export import hf_architecture_name
from halo.checkpoint.hub_layout import CheckpointNamespace
from halo.checkpoint.weights_source import LOADED_WEIGHTS_FROM_ATTR
from halo.distributed.expert_parallel.deepep.capacity import register_forward_generation_hook
from halo.distributed.expert_parallel.moe_layer.layer import named_ep_layers
from halo.distributed.pipeline_parallel.families import require_pp_attr, resolve_pp_spec
from halo.distributed.pipeline_parallel.model_rules import reject_unsupported_pp_model
from halo.distributed.pipeline_parallel.partition import resolve_layer_partition, validate_partition
from halo.distributed.pipeline_parallel.stage_losses import FusedHeadLossFn
from halo.models.attention.selection import effective_attn_implementation, is_flash_attn_implementation
from halo.models.head_transform import IDENTITY_HEAD_TRANSFORM, HeadTransform, resolve_head_transform
from halo.models.model_classes import REMOTE_CODE_MODEL_CLASS_ATTR, remote_code_model_class
from halo.models.module_tree import (
    backbone_with_layers,
    decoder_layers,
    decoder_layers_attr,
    embedding_module,
    module_path,
    split_layer_path,
)
from halo.models.segment_markers import packed_row_segment_kwargs, segment_markers_for
from halo.models.upstream_guard import verify_upstream_parameters

logger = logging.getLogger(__name__)

# Set by the stage-aware loader on a model whose decoder-layer list is already this rank's slice, so
# build_pipeline_stage re-uses that partition instead of re-deriving it from the short list.
PP_STAGE_PARTITION_ATTR = "_pp_stage_partition"


def _join(*parts: str) -> str:
    return ".".join(part for part in parts if part)


def rebase_layer_index(name: str, src_root: str, dst_root: str, delta: int) -> str | None:
    """Move a layer path from ``src_root`` to ``dst_root``, shifting its index by ``delta``; ``None`` off a layer path.
    Both naming directions and the loader's key filter use this, so a slice and its inverse cannot drift."""
    parts = split_layer_path(name, src_root)
    if parts is None:
        return None
    index, rest = parts
    return f"{dst_root}{index + delta}.{rest}" if rest else f"{dst_root}{index + delta}"


@dataclass(frozen=True)
class StageLayout:
    """Where a model's pipeline-relevant modules sit, as dotted paths in its unsplit naming. ``post_layer_paths``
    (final norm, tail modules) are neutralized off the last stage; ``embedding_path`` is dropped off stage 0."""

    backbone_prefix: str
    layer_attr: str
    head_attr: str
    post_layer_paths: tuple[str, ...]
    embedding_path: str

    @classmethod
    def of(cls, model: nn.Module) -> StageLayout:
        backbone = backbone_with_layers(model)
        spec = resolve_pp_spec(backbone)
        prefix = module_path(model, backbone)
        norm = require_pp_attr(backbone, spec.final_norm_attrs, "final norm")
        tails = [require_pp_attr(backbone, (attr,), "tail module") for attr in spec.tail_module_attrs]
        # A non-first stage must drop the embedding: a name guess that misses would leave it in every
        # stage's state_dict, and the merged index would resolve it to the last stage's untrained copy.
        embedding = embedding_module(model, "input")
        if embedding is None:
            raise ValueError(
                f"{type(model).__name__} exposes no input embedding via get_input_embeddings(); a "
                f"non-first pipeline stage must drop it and cannot locate it."
            )
        return cls(
            backbone_prefix=prefix,
            layer_attr=decoder_layers_attr(backbone),
            head_attr=require_pp_attr(model, spec.head_attrs, "task head"),
            post_layer_paths=tuple(_join(prefix, attr) for attr in (norm, *tails)),
            embedding_path=module_path(model, embedding),
        )

    @property
    def layer_root(self) -> str:
        """Dotted prefix of the decoder-layer paths (``model.layers.``, ``transformer.h.``, …)."""
        return f"{_join(self.backbone_prefix, self.layer_attr)}."

    def dropped_prefixes(self, is_first: bool, is_last: bool) -> tuple[str, ...]:
        dropped_inputs = () if is_first else (self.embedding_path,)
        return dropped_inputs if is_last else (*dropped_inputs, *self.post_layer_paths, self.head_attr)


def reject_reentrant_checkpointing(gradient_checkpointing_kwargs: dict | None) -> None:
    if (gradient_checkpointing_kwargs or {}).get("use_reentrant", False):
        raise ValueError(
            "use_reentrant=True gradient checkpointing is not supported under pipeline parallelism: "
            "the reentrant checkpoint runs its forward under no_grad, so FSDP2 never registers its "
            "pre-backward hooks and the recompute re-enters the full forward from inside the "
            "schedule's backward. Drop the kwarg; the default use_reentrant=False is supported."
        )


class PipelineStageModule(nn.Module):
    """A contiguous slice of a model with a tensor-in/tensor-out forward: the pipeline activation positionally
    (``input_ids`` on stage 0, hidden states elsewhere), side inputs as kwargs. It reports stage-local names and
    serializers apply :meth:`global_parameter_name`, since HF derives weight decay from ``named_children`` and
    ``load_state_dict``/DCP resolve keys against the real tree. ``source`` is read before the cut: writers take its
    config, generation config, remote-code class and namespace off the stage, which transformers cannot resolve."""

    # Undeclared on a source without one: save_model_config skips a model that carries no config.
    config: PretrainedConfig
    generation_config = None

    def __init__(
        self,
        source: nn.Module,
        layout: StageLayout,
        *,
        is_first: bool,
        is_last: bool,
        layer_offset: int,
        head_transform: HeadTransform = IDENTITY_HEAD_TRANSFORM,
    ):
        super().__init__()
        # Named ``model`` so ``backbone_with_layers`` finds the layer list through its ``.model`` probe.
        self.model = source.get_submodule(layout.backbone_prefix)
        self.head = getattr(source, layout.head_attr) if is_last else None
        self.layout = layout
        self.is_first = is_first
        self.is_last = is_last
        self._layer_offset = layer_offset
        # The backbone kwarg this stage hands the microbatch's token ids under; stage 0 reads them positionally.
        self.token_ids_kwarg = None if is_first else resolve_pp_spec(self.model).token_ids_kwarg
        # What the family's forward applies around its head; the stage replaces that forward.
        self.head_transform = head_transform
        # Set by the runner on a last stage running FusedHeadLoss: forward then returns the summed
        # loss instead of logits, so no [mb, S, V] plane is built.
        self.fused_loss_fn: FusedHeadLossFn | None = None

        self.checkpoint_namespace = CheckpointNamespace.of(source)
        setattr(self, REMOTE_CODE_MODEL_CLASS_ATTR, remote_code_model_class(source))
        if source.can_generate():
            self.generation_config = source.generation_config
        if hasattr(source, "config"):
            self.config = source.config
            architecture = hf_architecture_name(source)
            if architecture is not None:
                self.config.architectures = [architecture]
        # Where load_distributed_model read the weights: the resume's construction-identity verdict.
        if hasattr(source, LOADED_WEIGHTS_FROM_ATTR):
            setattr(self, LOADED_WEIGHTS_FROM_ATTR, getattr(source, LOADED_WEIGHTS_FROM_ATTR))
        # The EP capacity generation must ride the module the schedule calls: EP patching registered
        # it on the unsplit model, whose forward no pipeline runs.
        register_forward_generation_hook(self)

    def can_generate(self) -> bool:
        """``save_model_config`` writes ``generation_config.json`` only for a model that says so."""
        return self.generation_config is not None

    def add_model_tags(self, tags) -> None:
        """Record trainer tags (TRL calls this unguarded on every model). Mirrors PreTrainedModel."""
        tags = [tags] if isinstance(tags, str) else list(tags)
        if not hasattr(self, "model_tags"):
            self.model_tags = []
        for tag in tags:
            if tag not in self.model_tags:
                self.model_tags.append(tag)

    def gradient_checkpointing_enable(
        self, gradient_checkpointing_kwargs=None, every_n_layers: int = 1, offload: bool = False
    ) -> None:
        """Enable non-reentrant activation checkpointing on this stage's layers. Mirrors ``PreTrainedModel`` minus
        ``enable_input_require_grads`` (a non-first stage has no input embedding, and its recv buffer requires grad)
        and ``offload`` (host-staged activations are unvalidated against the schedule's interleaving)."""
        if offload:
            raise ValueError(
                "Activation offload (gradient_checkpointing_kwargs offload=True) is not supported under "
                "pipeline parallelism. Drop offload, or train without pipeline_parallel_size."
            )
        kwargs = dict(gradient_checkpointing_kwargs or {})
        reject_reentrant_checkpointing(kwargs)
        kwargs.setdefault("use_reentrant", False)
        if not getattr(self.model, "supports_gradient_checkpointing", False):
            raise ValueError(f"{type(self.model).__name__} does not support gradient checkpointing.")
        self.model._set_gradient_checkpointing(
            enable=True,
            gradient_checkpointing_func=functools.partial(checkpoint, **kwargs),
            every_n_layers=every_n_layers,
        )

    def gradient_checkpointing_disable(self) -> None:
        if getattr(self.model, "supports_gradient_checkpointing", False):
            self.model._set_gradient_checkpointing(enable=False)

    @property
    def is_gradient_checkpointing(self) -> bool:
        return any(getattr(m, "gradient_checkpointing", False) for m in self.model.modules())

    @property
    def layer_range(self) -> tuple[int, int]:
        return self._layer_offset, self._layer_offset + len(getattr(self.model, self.layout.layer_attr))

    def global_parameter_name(self, local_name: str) -> str:
        """Map a stage-local parameter / buffer / module path back to its FQN in the unsplit model."""
        layout = self.layout
        rebased = rebase_layer_index(local_name, f"model.{layout.layer_attr}.", layout.layer_root, self._layer_offset)
        if rebased is not None:
            return rebased
        if local_name == "model" or local_name.startswith("model."):
            return _join(layout.backbone_prefix, local_name[len("model.") :])
        if local_name == "head" or local_name.startswith("head."):
            return _join(layout.head_attr, local_name[len("head.") :])
        return local_name

    def local_parameter_name(self, global_name: str) -> str:
        """Inverse of :meth:`global_parameter_name`. Only meaningful for names this stage holds; a layer outside its
        range gets a negative or out-of-range index."""
        layout = self.layout
        rebased = rebase_layer_index(
            global_name, layout.layer_root, f"model.{layout.layer_attr}.", -self._layer_offset
        )
        if rebased is not None:
            return rebased
        # Head first: with an empty backbone prefix every global name is bare, so the backbone rule
        # below would claim ``lm_head.weight`` as ``model.lm_head.weight``.
        if global_name == layout.head_attr or global_name.startswith(f"{layout.head_attr}."):
            return _join("head", global_name[len(layout.head_attr) + 1 :])
        if not layout.backbone_prefix:
            return _join("model", global_name)
        if global_name == layout.backbone_prefix or global_name.startswith(f"{layout.backbone_prefix}."):
            return _join("model", global_name[len(layout.backbone_prefix) + 1 :])
        return global_name

    def checkpoint_name_map(self) -> dict[str, str]:
        """Global name → local ``state_dict`` key for the tensors this stage serializes, which the PP save and resume
        both walk. EP MoE entries are excluded: they hold this rank's shard layout and are re-emitted by the family
        gather. Raises on a non-invertible name, which would collapse two tensors into one entry."""
        ep_prefixes = tuple(f"{name}." for name, _ in named_ep_layers(self))
        name_map: dict[str, str] = {}
        for local in self.state_dict():
            if local.startswith(ep_prefixes):
                continue
            global_name = self.global_parameter_name(local)
            if self.local_parameter_name(global_name) != local:
                raise RuntimeError(
                    f"Pipeline stage tensor '{local}' maps to global name '{global_name}', which maps "
                    f"back to '{self.local_parameter_name(global_name)}' — the stage naming map is not "
                    f"invertible here, so this tensor would share a checkpoint entry with another."
                )
            name_map[global_name] = local
        return name_map

    def _segment_kwargs(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None, position_ids: torch.Tensor | None
    ) -> dict[str, torch.Tensor | int]:
        """A packed microbatch's segment markers, built once from one host read of its row. The schedule microbatches
        only batch-dim tensors, so the collator's flattened-row markers cannot ride in, and each attention layer would
        otherwise derive the varlen set from ``position_ids`` (a host sync each) again in the recompute."""
        config = getattr(self, "config", None)
        if attention_mask is not None or position_ids is None or hidden_states.shape[0] != 1:
            return {}
        markers = segment_markers_for(config)
        if is_flash_attn_implementation(effective_attn_implementation(config)):
            markers = replace(markers, cu_seq_lens=True)
        if not (markers.seq_idx or markers.cu_seq_lens):
            return {}
        return packed_row_segment_kwargs(position_ids, markers)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        token_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """This stage's activation, or on the last stage its logits or summed loss. ``labels`` reaches every stage (the
        schedule microbatches kwargs for all) and only a fused last stage reads it."""
        if (token_ids is None) != (self.token_ids_kwarg is None):
            expected = f"as {self.token_ids_kwarg!r}" if self.token_ids_kwarg is not None else "never"
            raise RuntimeError(
                f"The pipeline stage holding layers {self.layer_range} of {type(self.model).__name__} takes the "
                f"microbatch's token ids {expected}, but {'none' if token_ids is None else 'some'} arrived: the "
                f"runner and the stage disagree on which stages re-read them."
            )
        activation = {"input_ids" if self.is_first else "inputs_embeds": hidden_states}
        if token_ids is not None:
            activation[self.token_ids_kwarg] = token_ids
        hidden = self.model(
            **activation,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
            **self._segment_kwargs(hidden_states, attention_mask, position_ids),
        ).last_hidden_state
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
            return self.head_transform.project(self.head, hidden)
        if labels is None:
            raise RuntimeError(
                "The last pipeline stage computes its loss inside forward (fused head) but received "
                "no `labels` kwarg. The runner supplies them on every call; a caller driving the "
                "schedule directly must do the same."
            )
        return self.fused_loss_fn(self.head, self.head_transform, hidden, labels)


def slice_backbone_to_stage(model: nn.Module, lo: int, hi: int) -> None:
    """Replace the backbone's decoder-layer list with ``[lo, hi)``, in place. The one slicing seam of the stage-aware
    loader and :func:`build_pipeline_stage`, so neither re-bases differently."""
    backbone = backbone_with_layers(model)
    layer_attr = decoder_layers_attr(backbone)
    setattr(backbone, layer_attr, nn.ModuleList(list(getattr(backbone, layer_attr))[lo:hi]))


def _replace_submodule(model: nn.Module, path: str, replacement: nn.Module | None) -> None:
    parent, _, attr = path.rpartition(".")
    setattr(model.get_submodule(parent), attr, replacement)


def build_pipeline_stage(
    model: nn.Module,
    pp_rank: int,
    pp_size: int,
    moe_balancing: str = "auto",
    pp_split: list[int] | None = None,
) -> PipelineStageModule:
    """Convert ``model`` into this rank's pipeline stage, in place; use the returned module afterwards. A model the
    stage-aware loader already sliced keeps that split. ``moe_balancing`` is checked against the aux-loss path."""
    partition = getattr(model, PP_STAGE_PARTITION_ATTR, None)
    pre_sliced = partition is not None
    reject_unsupported_pp_model(model, moe_balancing, pre_sliced=pre_sliced)
    if not pre_sliced:
        partition = resolve_layer_partition(model, pp_size, pp_split)
    lo, hi = partition[pp_rank]
    is_first, is_last = pp_rank == 0, pp_rank == pp_size - 1
    if pp_rank == 0:
        logger.info(
            "Pipeline layer partition (%s): %s",
            "pp_split" if pp_split is not None else "head-weighted",
            partition,
        )
    validate_partition(model, partition)

    backbone = backbone_with_layers(model)
    if pre_sliced and len(decoder_layers(backbone)) != hi - lo:
        raise RuntimeError(
            f"Pipeline stage {pp_rank} was loaded stage-aware with layers [{lo}, {hi}) but holds "
            f"{len(decoder_layers(backbone))} decoder layers. The loader's partition and the trainer's "
            f"disagree, so every saved tensor would carry the wrong global layer index."
        )
    layout = StageLayout.of(model)
    stage = PipelineStageModule(
        model,
        layout,
        is_first=is_first,
        is_last=is_last,
        layer_offset=lo,
        head_transform=resolve_head_transform(model) if is_last else IDENTITY_HEAD_TRANSFORM,
    )

    if not pre_sliced:
        slice_backbone_to_stage(model, lo, hi)
    if not is_last:
        for path in layout.post_layer_paths:
            _replace_submodule(model, path, nn.Identity())
        setattr(model, layout.head_attr, None)
    if not is_first:
        _replace_submodule(model, layout.embedding_path, None)
        stream_forward = resolve_pp_spec(backbone).stream_forward
        if stream_forward is not None:
            verify_upstream_parameters(
                type(backbone).forward, tuple(inspect.signature(stream_forward).parameters), stream_forward.__name__
            )
            # Per instance, not on the class: stage 0 and any unsplit model in the process keep the
            # real forward, while nn.Module.__call__ and every FSDP2/GC hook resolve this one.
            backbone.forward = MethodType(stream_forward, backbone)

    logger.info(
        "Pipeline stage %d/%d: decoder layers [%d, %d) of %d%s%s",
        pp_rank,
        pp_size,
        lo,
        hi,
        partition[-1][1],
        " +embedding" if is_first else "",
        f" +norm+{layout.head_attr}" if is_last else "",
    )
    return stage
