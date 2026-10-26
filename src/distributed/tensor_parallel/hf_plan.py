"""HF-native tensor parallelism (``tp_plan="auto"``): what the plan shards, the tied embedding/head pair kept on one
side of it, and the load finish: the post-load checks and transformers' per-backward norm all-reduce moved onto the
trainer's step-time SUM."""

from __future__ import annotations

import logging
from contextlib import contextmanager

import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor
from transformers.distributed.tensor_parallel import ALL_PARALLEL_STYLES, TensorParallelLayer, _get_parameter_plan

from src.distributed.tensor_parallel.plain_params import hand_sliced_suffixes, register_per_head_norm_params
from src.model_loading.model_preparation import concrete_model_class
from src.models.config_access import text_config, ties_word_embeddings
from src.models.module_tree import find_module_path
from src.runtime.ranks import is_global_main_process

logger = logging.getLogger(__name__)

# transformers' replicated-norm gradient handling: the plan style a config asks for, and the class whose
# backward hook installs the per-backward all-reduce for it.
_HF_REPLICATED_GRAD_STYLE = "replicated_with_grad_allreduce"
_HF_REPLICATED_GRAD_CLASS = "ReplicatedWithGradAllReduce"


def plan_shards_param(key: str, tp_plan: dict[str, str]) -> bool:
    """Whether ``tp_plan`` puts ``key`` under a style that shards parameters: one whose class overrides the base
    no-op ``shard_param`` (the activation-transform styles, ``mla_kv_a_proj`` or ``ep_router``, leave theirs
    whole). Both the lookup and the decision are transformers' own, which a restated table would drift from."""
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
    return type(style).shard_param is not TensorParallelLayer.shard_param


def _tied_module_names(concrete) -> tuple[set[str], set[str]]:
    """``(head modules, embedding modules)`` the class's ``_tied_weights_keys`` names; empty without one."""
    raw = getattr(concrete, "_tied_weights_keys", None) or {}
    pairs = list(raw.items()) if isinstance(raw, dict) else [(key, key) for key in raw]
    heads = {target.rsplit(".", 1)[0] for target, _source in pairs}
    embeddings = {source.rsplit(".", 1)[0] for _target, source in pairs}
    return heads, embeddings


def _plan_keys_naming(plan: dict, modules: set[str]) -> set[str]:
    """Keys of ``plan`` naming one of ``modules``, model-absolute (a class ``_tp_plan``) or backbone-relative (a
    config ``base_model_tp_plan``)."""
    return {
        key
        for key in plan
        for module in modules
        if module == key or module.endswith("." + key) or key.endswith("." + module)
    }


@contextmanager
def consistent_tied_tp_plan(model_class, model_config):
    """While inside, drop the plan entry of a tied embedding/head pair that only one end carries.

    transformers shards module by module and ties afterwards, so a lone entry (a multimodal class planning no
    ``lm_head``, a backbone shipping no ``base_model_tp_plan``) hands the unplanned end a weight of the other
    kind, and the first forward fails on mixed plain/DTensor operands. Without the entry the pair stays
    replicated.
    """
    concrete = concrete_model_class(model_class, model_config)
    heads, embeddings = _tied_module_names(concrete)
    if not ties_word_embeddings(model_config) or not heads or not embeddings:
        yield
        return
    class_plan = getattr(concrete, "_tp_plan", None)
    # The embedding entry lives on the backbone's plan, whose config is the text one: a composite
    # wrapper's own base_model_tp_plan is None (Qwen3.5), so the outer config reports no entry.
    plan_config = text_config(model_config)
    base_plan = getattr(plan_config, "base_model_tp_plan", None)
    head_keys = _plan_keys_naming(class_plan or {}, heads)
    embedding_keys = _plan_keys_naming(class_plan or {}, embeddings) | _plan_keys_naming(base_plan or {}, embeddings)
    if bool(head_keys) == bool(embedding_keys):
        yield
        return

    drop = head_keys or embedding_keys
    original_class_plan, original_base_plan = class_plan, base_plan
    if class_plan is not None:
        concrete._tp_plan = {key: value for key, value in class_plan.items() if key not in drop}
    if base_plan is not None:
        plan_config.base_model_tp_plan = {key: value for key, value in base_plan.items() if key not in drop}
    try:
        yield
    finally:
        if original_class_plan is not None:
            concrete._tp_plan = original_class_plan
        if original_base_plan is not None:
            plan_config.base_model_tp_plan = original_base_plan


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


def retarget_hf_replicated_grad_hooks(model: nn.Module) -> None:
    """Move transformers' per-backward norm all-reduce onto the trainer's step-time SUM.

    ``ReplicatedWithGradAllReduce``'s backward hook all-reduces ``param.grad`` in place on every backward, so
    under gradient accumulation earlier micro-batches compound by ``tp_size`` per micro-step. Raises when the
    plan asks for that style and no hook was found.
    """
    plan = getattr(model, "_tp_plan", None) or {}
    plan_wants_it = any(style == _HF_REPLICATED_GRAD_STYLE for style in plan.values())

    modules: list[nn.Module] = []
    for module in model.modules():
        hooks = getattr(module, "_backward_hooks", None)
        if not hooks:
            continue
        stale = [
            handle
            for handle, fn in hooks.items()
            if getattr(fn, "__qualname__", "").startswith(f"{_HF_REPLICATED_GRAD_CLASS}.")
        ]
        if not stale:
            continue
        for handle in stale:
            del hooks[handle]
        modules.append(module)

    if plan_wants_it and not modules:
        raise RuntimeError(
            f"The TP plan requests {_HF_REPLICATED_GRAD_STYLE!r} but no "
            f"{_HF_REPLICATED_GRAD_CLASS} backward hook was found to retarget — transformers has "
            "changed how it installs that reduction. Without the retarget the per-head attention "
            "norms are re-reduced on every backward, which compounds under gradient accumulation."
        )

    register_per_head_norm_params(model, modules)
    if modules:
        logger.info(f"Retargeted {len(modules)} per-backward norm all-reduces to the step-time TP SUM")


def is_input_embedding_tp_sharded(model: nn.Module) -> bool:
    """Whether ``model``'s input embedding is an HF-native TP shard, which ``resize_token_embeddings`` cannot
    re-shard."""
    return isinstance(model.get_input_embeddings().weight, DTensor)


def finish_hf_tp_load(model: nn.Module, device_mesh: DeviceMesh, tp_size: int) -> None:
    """Check a ``tp_plan="auto"`` load against the plan it applied, stamp ``device_mesh`` on ``model`` and move
    the per-backward norm all-reduces onto the trainer's step-time SUM.

    ``tp_plan="auto"`` plans at most the head for an architecture shipping no ``base_model_tp_plan`` (Qwen3-VL)
    and transformers only warns, so a plan sharding no backbone parameter is refused here by name.
    """
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
    model._device_mesh = device_mesh
    retarget_hf_replicated_grad_hooks(model)
