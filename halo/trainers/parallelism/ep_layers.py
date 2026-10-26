"""The model's EP layers and the per-run facts read off them: parameter-id sets and EP gradient checkpointing.

EP layers and their parameters are identified by module membership, never by name, so shared experts
and dense MLPs are not misclassified.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

from halo.distributed.expert_parallel.moe_layer.layer import EPMoELayerBase, named_ep_layers
from halo.models.gradient_checkpointing import enable_scoped_gradient_checkpointing
from halo.models.moe.config_fields import has_routed_experts
from halo.runtime.ranks import is_global_main_process
from halo.trainers.parallelism.trainer_rules import reject_ep_gradient_checkpointing

logger = logging.getLogger(__name__)


def uses_reentrant_checkpointing(parallelism_config, model_config) -> bool:
    """Whether gradient checkpointing has to run reentrant for this run.

    CP's sequence all-to-alls do not survive non-reentrant recompute, and EP's DeepEP barriers desync
    ranks under its lazy recompute. A MoE routes in its recompute too: a near-tie pick on the recomputed
    hidden state can flip and change the routing tensors' shapes, which non-reentrant checkpointing
    rejects. Pipeline parallelism is the exception: reentrant runs the original forward in no_grad, so
    FSDP2 registers no pre-backward hooks there.
    """
    if parallelism_config.is_pp_mode:
        return False
    return parallelism_config.has_ep_group or parallelism_config.is_cp_mode or has_routed_experts(model_config)


def resolve_use_reentrant(parallelism_config, model_config, configured: bool | None) -> bool | None:
    """``True`` where :func:`uses_reentrant_checkpointing` requires it, warning when that overrides a
    configured ``False``; else ``configured``."""
    if not uses_reentrant_checkpointing(parallelism_config, model_config):
        return configured
    if configured is False and is_global_main_process():
        logger.warning(
            "Expert/Context Parallelism and MoE routing use use_reentrant=True for gradient "
            "checkpointing; overriding the configured use_reentrant=False. CP's sequence "
            "all-to-alls do not survive non-reentrant recompute, and a recomputed router may "
            "route differently, which non-reentrant checkpointing rejects; non-reentrant is "
            "validated only under pipeline parallelism, which requires it."
        )
    return True


def require_ep_config(ep_config):
    """``ep_config`` itself, or a raise on a path that only runs with EP layers present: without it every
    process group read off it would come back ``None``, turning each collective into a silent no-op."""
    if ep_config is None:
        raise RuntimeError(
            "EP state is required on this path but no EPConfig was captured: _configure_distributed_modes "
            "must capture it from a model carrying EP layers first."
        )
    return ep_config


class EPLayersMixin:
    """The model's EP layers, their parameter-id sets and EP gradient checkpointing."""

    # ``id(param)`` sets by name. FSDP2 replaces the Parameter objects it manages, so a set taken before
    # the wrap names dead objects: the wrap clears them all.
    _param_id_caches: dict[str, set[int]]

    def _ep_layers(self) -> list[EPMoELayerBase]:
        """Every EP-patched layer of the model, in ``named_modules`` order.

        Memoized per model object, since it is read every logging step and save. Module objects survive
        the FSDP2 wrap (``fully_shard`` swaps their class in place), so unlike the id sets it needs no
        clearing.
        """
        model, layers = getattr(self, "_ep_layers_of", (None, None))
        if model is not self.model:
            layers = [layer for _name, layer in named_ep_layers(self.model)]
            self._ep_layers_of = (self.model, layers)
        return layers

    def _memoized_param_ids(self, key: str, build: Callable[[], set[int]]) -> set[int]:
        """``build()``, memoized in ``_param_id_caches`` under ``key`` until the wrap clears it."""
        if key not in self._param_id_caches:
            self._param_id_caches[key] = build()
        return self._param_id_caches[key]

    def _get_ep_param_ids(self) -> set[int]:
        """Ids of every EP layer's parameters, the replicated router and shared expert included."""
        return self._memoized_param_ids(
            "ep", lambda: {id(param) for layer in self._ep_layers() for param in layer.parameters()}
        )

    def _get_sharded_expert_param_ids(self) -> set[int]:
        """Ids of the EP/ETP-distributed expert weights only (each layer's ``expert_named_params()``), whose
        shard norms the grad-norm expert bucket sums; a replicated parameter would be over-counted there.

        ep1 FSDP-managed experts are left out: they are DTensors normed over their FSDP shard group, and
        the expert bucket's legs reduce nothing at ep1, so only the local shard would count.
        """
        return self._memoized_param_ids(
            "sharded_experts",
            lambda: {
                id(param)
                for layer in self._ep_layers()
                if not layer.ep_config.experts_fsdp_managed
                for _name, param in layer.expert_named_params()
                if param is not None
            },
        )

    def _install_ep_gradient_checkpointing(self) -> None:
        """EP-safe gradient checkpointing, enabled here rather than by HF: EP's recompute must replay the
        checkpoint frame's dispatch/combine results rather than issue a second DeepEP cycle."""
        ep_layers = self._ep_layers()
        if not ep_layers:
            return
        config = self.parallelism_config
        # Combine and cross-replica grad-sync are different-membership collectives, hence deferred averaging.
        if config.num_ep_groups > 1 and not config.is_expert_tp_mode:
            # Outside the logging gate, so a missing EP config raises on every rank rather than rank 0 alone.
            deferred_dp = require_ep_config(self._ep_config).is_deferred_dp
            if deferred_dp and is_global_main_process():
                scope = "node-local across domains" if config.ep_scope == "node" else "cross-node"
                logger.info(
                    f"Multi-group EP ({scope}, {config.num_ep_groups} EP groups, "
                    f"ep_group_size={config.ep_group_size}): cross-replica DP average deferred to a "
                    "post-backward sweep (EP-group FSDP + DeepEP combine only during backward)."
                )
        if not self.args.gradient_checkpointing:
            return
        reject_ep_gradient_checkpointing(ep_layers)
        gc_kwargs = self.args.gradient_checkpointing_kwargs or {}
        # The config the constructor resolved use_reentrant from, read off the same top-level model.
        model_config = getattr(self._top_level_model(), "config", None)
        use_reentrant = resolve_use_reentrant(config, model_config, gc_kwargs.get("use_reentrant"))
        # Explicit where nothing forces reentrancy too: the PP stage's checkpoint takes the value as given.
        gc_kwargs = {**gc_kwargs, "use_reentrant": bool(use_reentrant)}
        enable_scoped_gradient_checkpointing(self.model, gradient_checkpointing_kwargs=gc_kwargs)
        # TRL's disable_gradient_checkpointing context manager re-enables GC from these kwargs on exit.
        self.args.gradient_checkpointing_kwargs = gc_kwargs
        self.args.gradient_checkpointing = False
