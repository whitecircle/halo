"""HF-native tensor parallelism (``tp_plan="auto"``): what the plan shards, the tied pair, the post-load finish."""

from __future__ import annotations

import logging
from contextlib import contextmanager

import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor
from transformers.distributed.expert_parallel import ExpertParallelLayer
from transformers.distributed.parallel_utils import ALL_PARALLEL_STYLES, _get_parameter_plan
from transformers.distributed.tensor_parallel import TensorParallelLayer

from halo.distributed.tensor_parallel.local_linears import install_planned_local_forwards
from halo.distributed.tensor_parallel.plain_params import hand_sliced_suffixes, register_per_head_norm_params
from halo.models.introspection.config_access import text_config, ties_word_embeddings
from halo.models.introspection.model_classes import mapped_model_class
from halo.models.introspection.module_tree import find_module_path
from halo.runtime.ranks import is_global_main_process

logger = logging.getLogger(__name__)

# transformers' replicated-norm gradient handling: the plan style a config asks for, and the class whose forward
# wrapper installs the per-accumulation all-reduce for it.
_HF_REPLICATED_GRAD_STYLE = "replicated_with_grad_allreduce"
_HF_REPLICATED_GRAD_CLASS = "ReplicatedWithGradAllReduce"
# The backbone attribute transformers plans as a vocab-parallel embedding by default, and the style that keeps a
# parameter whole on every rank.
_HF_IMPLICIT_EMBEDDING = "embed_tokens"
_HF_KEEP_WHOLE_STYLE = "replicate"
# The no-op ``shard_param`` of each base class the registered TP and EP styles derive from.
_NO_OP_SHARD_PARAMS = (TensorParallelLayer.shard_param, ExpertParallelLayer.shard_param)


def plan_shards_param(key: str, tp_plan: dict[str, str]) -> bool:
    """Whether ``tp_plan`` puts ``key`` under a style whose class overrides its base's no-op ``shard_param``. Both
    lookup and decision are transformers' own, which a restated table would drift from."""
    style_name = _get_parameter_plan(key, tp_plan)
    if style_name is None:
        return False
    style = ALL_PARALLEL_STYLES.get(style_name)
    if style is None:
        # transformers rejects such a plan at apply time (``_validate_parallel_plan_styles``); answering
        # "replicated" here would report a load that cannot run as one that shards nothing.
        raise ValueError(
            f"TP plan style {style_name!r} (for parameter {key!r}) is not registered in transformers' "
            f"ALL_PARALLEL_STYLES ({sorted(ALL_PARALLEL_STYLES.keys())}); nothing can be derived for it."
        )
    return type(style).shard_param not in _NO_OP_SHARD_PARAMS


def _tied_module_names(concrete) -> tuple[set[str], set[str]]:
    """``(head modules, embedding modules)`` the class's ``_tied_weights_keys`` names; empty without one."""
    raw = getattr(concrete, "_tied_weights_keys", None) or {}
    pairs = list(raw.items()) if isinstance(raw, dict) else [(key, key) for key in raw]
    heads = {target.rsplit(".", 1)[0] for target, _source in pairs}
    embeddings = {source.rsplit(".", 1)[0] for _target, source in pairs}
    return heads, embeddings


def _plan_keys_naming(plan: dict, modules: set[str]) -> set[str]:
    """Keys of ``plan`` naming one of ``modules``, model-absolute (``_tp_plan``) or backbone-relative (config)."""
    return {
        key
        for key in plan
        for module in modules
        if module == key or module.endswith("." + key) or key.endswith("." + module)
    }


@contextmanager
def toolkit_tp_plan(model_class, model_config):
    """While inside, the plan keeps an untied input embedding whole and carries a tied embedding/head pair on both
    ends or neither. transformers vocab-shards a planned backbone's ``embed_tokens`` unless the plan names it, and
    DTensor's embedding backward returns a replicated gradient beside that shard, which the optimizer and the norm clip
    read as the shard's own; a tied pair's head use brings it back sharded. transformers ties after sharding, so a
    lone tied entry hands the unplanned end a weight of the other kind, failing on mixed operands."""
    concrete = mapped_model_class(model_config, model_class) or model_class
    class_plan = getattr(concrete, "_tp_plan", None)
    # The embedding entry lives on the backbone's plan, whose config is the text one: a composite
    # wrapper's own base_model_tp_plan is None (Qwen3.5), so the outer config reports no entry.
    plan_config = text_config(model_config)
    base_plan = getattr(plan_config, "base_model_tp_plan", None)
    heads, embeddings = _tied_module_names(concrete)
    tied = ties_word_embeddings(model_config) and bool(heads) and bool(embeddings)
    head_keys = _plan_keys_naming(class_plan or {}, heads)
    embedding_keys = _plan_keys_naming(class_plan or {}, embeddings) | _plan_keys_naming(base_plan or {}, embeddings)
    # An entry, or transformers' own ``embedding_rowwise`` for a planned backbone's unnamed ``embed_tokens``.
    embedding_planned = bool(embedding_keys) or (
        bool(base_plan) and _HF_IMPLICIT_EMBEDDING in {module.rsplit(".", 1)[-1] for module in embeddings}
    )
    one_ended = tied and bool(head_keys) != embedding_planned
    drop = (head_keys or embedding_keys) if one_ended else set()
    new_base_plan = base_plan
    if base_plan:
        new_base_plan = {key: value for key, value in base_plan.items() if key not in drop}
        # A tied pair planned on both ends or neither keeps the plan it has. Naming ``embed_tokens`` under the no-op
        # style is the only way to keep it whole.
        if (not tied or one_ended) and not _plan_keys_naming(new_base_plan, {_HF_IMPLICIT_EMBEDDING}):
            new_base_plan[_HF_IMPLICIT_EMBEDDING] = _HF_KEEP_WHOLE_STYLE
    if not drop and new_base_plan == base_plan:
        yield
        return

    if class_plan is not None and drop:
        concrete._tp_plan = {key: value for key, value in class_plan.items() if key not in drop}
    if new_base_plan is not None:
        plan_config.base_model_tp_plan = new_base_plan
    try:
        yield
    finally:
        if class_plan is not None:
            concrete._tp_plan = class_plan
        if base_plan is not None:
            plan_config.base_model_tp_plan = base_plan


def validate_tied_pair_consistent(model, applied_plan: dict) -> None:
    """Raise when a tied embedding/head pair loaded as two parameters (a refused tie: each end would train on
    half the gradient), or with a sharding the plan's transforms on the head disagree with."""
    in_emb = model.get_input_embeddings()
    out_emb = model.get_output_embeddings()
    if in_emb is None or out_emb is None:
        return
    if out_emb.weight is not in_emb.weight:
        raise RuntimeError(
            "tie_word_embeddings=True but the TP load produced two independent parameters for the "
            "tied embedding/head — each would train on half the tied gradient. transformers ties "
            "AFTER sharding, so this is a refused tie: the checkpoint carries both keys with "
            "different values. Re-export it with tie_word_embeddings=false."
        )
    # By module rather than parameter name: the tie makes both ends one object, which
    # ``named_parameters`` emits only under the embedding's name.
    head_name = find_module_path(model, out_emb)
    if head_name is None:
        return
    head_sharded = plan_shards_param(f"{head_name}.weight", applied_plan)
    if head_sharded != isinstance(in_emb.weight, DTensor):
        raise RuntimeError(
            f"The tied embedding/head pair and the applied TP plan disagree: {head_name!r} is "
            f"{'covered' if head_sharded else 'not covered'} by a sharding style while the tied "
            f"weight loaded {'sharded' if not head_sharded else 'replicated'}. The forward would die "
            f"on mixed plain/DTensor operands. Train this architecture without tensor parallelism."
        )


def reject_plan_sharded_plain_params(model: nn.Module, tp_plan: dict[str, str]) -> None:
    """Reject a load whose plan shards a parameter that materialized as a plain tensor: a rank-local slice the
    gradient sync would average with other ranks' different slices. Hand-sliced parameters are exempt."""
    hand_sliced = hand_sliced_suffixes(model)
    offenders = [
        name
        for name, param in model.named_parameters()
        if not isinstance(param, DTensor) and not name.endswith(hand_sliced) and plan_shards_param(name, tp_plan)
    ]
    if offenders:
        raise ValueError(
            f"The applied TP plan shards {len(offenders)} parameter(s) that materialized as plain "
            f"tensors (e.g. {offenders[:3]}): each rank holds a bare slice that the gradient sync "
            f"would average against other ranks' DIFFERENT slices. This is a transformers "
            f"tensor-parallel materialization regression — pin a transformers version whose "
            f"tp_plan='auto' places DTensors, or drop tensor_parallel_size for this model."
        )


def _installs_hf_replicated_grad_hooks(module: nn.Module) -> bool:
    forward = vars(module).get("forward")
    return getattr(forward, "__qualname__", "").startswith(f"{_HF_REPLICATED_GRAD_CLASS}.")


def _has_post_accumulate_hooks(module: nn.Module) -> bool:
    return any(getattr(param, "_post_accumulate_grad_hooks", None) for param in module.parameters(recurse=False))


def retarget_hf_replicated_grad_hooks(model: nn.Module) -> None:
    """Move transformers' per-backward norm all-reduce onto the trainer's step-time SUM, since it reduces
    ``param.grad`` in place on every accumulation and earlier micro-batches compound by ``tp_size``. transformers
    wraps the module's forward to register a post-accumulate-grad hook on each parameter at its first grad-enabled
    forward; dropping the wrapper restores the class forward before any hook exists. Raises when the plan asks for
    that style and no wrapper was found, or when a forward already registered the hooks."""
    plan = getattr(model, "_tp_plan", None) or {}
    plan_wants_it = any(style == _HF_REPLICATED_GRAD_STYLE for style in plan.values())

    modules = [module for module in model.modules() if _installs_hf_replicated_grad_hooks(module)]
    for module in modules:
        if _has_post_accumulate_hooks(module):
            raise RuntimeError(
                f"A {type(module).__name__} under {_HF_REPLICATED_GRAD_STYLE!r} already ran a grad-enabled "
                "forward, which registered transformers' per-accumulation all-reduce on its parameters: retarget "
                "the norm gradients before the first forward."
            )
        del module.forward

    if plan_wants_it and not modules:
        raise RuntimeError(
            f"The TP plan requests {_HF_REPLICATED_GRAD_STYLE!r} but no {_HF_REPLICATED_GRAD_CLASS} forward "
            "wrapper was found to retarget — transformers has changed how it installs that reduction. Without the "
            "retarget the per-head attention norms are re-reduced on every backward, which compounds under "
            "gradient accumulation."
        )

    register_per_head_norm_params(model, modules)
    if modules:
        logger.info(f"Retargeted {len(modules)} per-backward norm all-reduces to the step-time TP SUM")


def is_input_embedding_tp_sharded(model: nn.Module) -> bool:
    """Whether the input embedding is an HF-native TP shard, which ``resize_token_embeddings`` cannot re-shard."""
    return isinstance(model.get_input_embeddings().weight, DTensor)


def finish_hf_tp_load(model: nn.Module, device_mesh: DeviceMesh, tp_size: int) -> None:
    """Check a ``tp_plan="auto"`` load, stamp ``device_mesh``, run planned linears on local shards and move norm
    all-reduces to the step-time SUM. A plan sharding no backbone parameter (Qwen3-VL ships no ``base_model_tp_plan``,
    and transformers only warns) is refused by name."""
    applied_plan = getattr(model, "_tp_plan", None) or {}
    head = model.get_output_embeddings()
    head_ids = {id(param) for param in head.parameters()} if head is not None else set()
    if tp_size > 1 and not any(
        plan_shards_param(name, applied_plan) for name, param in model.named_parameters() if id(param) not in head_ids
    ):
        raise ValueError(
            f"tensor_parallel_size={tp_size} sharded ZERO parameters: this architecture ships no "
            f"base_model_tp_plan, so TP would run {tp_size} full replicas at 1/{tp_size} throughput. "
            f"Drop tensor_parallel_size for this model."
        )
    if tp_size > 1:
        reject_plan_sharded_plain_params(model, applied_plan)
    if ties_word_embeddings(model.config):
        validate_tied_pair_consistent(model, applied_plan)
        if tp_size > 1 and not is_input_embedding_tp_sharded(model) and is_global_main_process():
            logger.warning(
                "This architecture plans only one end of its tied embedding/head pair, so the pair "
                "stays REPLICATED: budget a full vocab-sized weight, gradient and optimizer state on "
                "every TP rank."
            )
    if tp_size > 1:
        # After the tie check, which diagnoses a planned head whose tied weight loaded replicated; the install
        # would only refuse that weight as a plain tensor.
        installed = install_planned_local_forwards(model, applied_plan)
        if is_global_main_process():
            logger.info(f"{installed} TP linears compute on their local shards")
    model._device_mesh = device_mesh
    retarget_hf_replicated_grad_hooks(model)
