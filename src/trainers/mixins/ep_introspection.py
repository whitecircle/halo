"""Expert-Parallel module/parameter introspection and EP-safe gradient checkpointing.

Identifies EP-patched modules and their parameters by module membership (not name matching, so
shared experts and dense MLPs are not misclassified), and sets up EP gradient checkpointing with
multi-group topology guards.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from accelerate.logging import get_logger

from src.distributed.expert_parallel.base_layer import EPMoELayerBase
from src.distributed.expert_parallel.patching import enable_ep_gradient_checkpointing
from src.distributed.runtime import is_global_main_process

logger = get_logger(__name__, log_level="info")


def named_ep_layers(model: nn.Module) -> dict[str, EPMoELayerBase]:
    """FQN → EP MoE layer for every EP-wrapped module in ``model``.

    Matches by ``isinstance`` rather than an ``ep_config`` probe: a PEFT ``modules_to_save`` wrapper
    forwards ``__getattr__`` to the EP layer it holds, so a probe would match the wrapper too, and
    the wrapper's own type declares neither the family gather nor its export renames.
    """
    return {name: module for name, module in model.named_modules() if isinstance(module, EPMoELayerBase)}


class EpIntrospectionMixin:
    """EP module/parameter discovery + EP gradient-checkpointing setup. Mixed into the trainer."""

    def _find_ep_modules(self) -> list[EPMoELayerBase]:
        """Find all EP-patched modules, capturing the EP config from the first one."""
        ep_modules = list(named_ep_layers(self.model).values())
        if ep_modules and self._ep_config is None:
            self._ep_config = ep_modules[0].ep_config
        return ep_modules

    @property
    def _has_ep_layers(self) -> bool:
        """Whether the model actually has EP-patched layers.

        Asked of the model directly rather than read off ``_ep_config``, which is populated only as
        a side effect of ``_find_ep_modules`` and so may be unset under CP-only or TP-only setup.
        This property selects the checkpoint saver, the resume path and the EP grad-clip branch.
        """
        return bool(named_ep_layers(self.model))

    def _get_ep_param_ids(self) -> set:
        """Parameter ids belonging to EP modules, by module membership (not name matching).

        Memoized; called every step (grad clipping, the deferred cross-node grad sweep, the TP
        pre-step hook). FSDP2 replaces managed Parameter objects rather than swapping ``.data`` in
        place, so the ids are only stable once wrapping is done. Do not call this before wrapping.
        """
        cached = getattr(self, "_ep_param_ids_cache", None)
        if cached is not None:
            return cached
        ep_param_ids = set()
        for module in named_ep_layers(self.model).values():
            for param in module.parameters():
                ep_param_ids.add(id(param))
        self._ep_param_ids_cache = ep_param_ids
        return ep_param_ids

    def _invalidate_param_id_caches(self) -> None:
        """Drop every memoized ``id(param)`` set. Called once wrapping is complete.

        The ids are only meaningful against the Parameter objects that survive wrapping. Anything
        cached earlier (the fp32 upcast populates :meth:`_get_ep_param_ids` before FSDP runs) names
        replaced objects and would classify every EP param as non-EP, and every hand-sliced sink as
        replicated.
        """
        self._ep_param_ids_cache = None
        self._sharded_expert_param_ids_cache = None
        self._tp_sharded_plain_ids_cache = None
        self._tp_per_head_norm_ids_cache = None

    def _get_sharded_expert_param_ids(self) -> set:
        """Ids of the EP/ETP-distributed expert weights only (each layer's ``expert_named_params()``).

        Strict subset of :meth:`_get_ep_param_ids` excluding the replicated router/shared expert, so
        the grad-norm expert-bucket all-reduce sums distinct shard norms (a replicated param would be
        over-counted by group size). Memoized.
        """
        cached = getattr(self, "_sharded_expert_param_ids_cache", None)
        if cached is not None:
            return cached
        ids = set()
        for module in named_ep_layers(self.model).values():
            # ep1 FSDP DTensors belong in the standard bucket; the replica division would shrink their norm².
            if module.ep_config.experts_fsdp_managed:
                continue
            for _name, param in module.expert_named_params():
                if param is not None:
                    ids.add(id(param))
        self._sharded_expert_param_ids_cache = ids
        return ids

    def _upcast_non_ep_params_to_fp32(self):
        """Upcast non-EP (dense) params to FP32 master weights (BF16 compute via autocast).

        Iterates the full top-level model so ``lm_head`` (non-EP on untied-embedding models) is
        upcast too; EP params keep their own precision control.
        """
        model = self._top_level_model()
        ep_param_ids = self._get_ep_param_ids()

        upcast_count = 0
        skipped_ep_count = 0

        for _name, param in model.named_parameters():
            if id(param) in ep_param_ids:
                skipped_ep_count += 1
                continue

            # A QLoRA base is bnb Params4bit over uint8-packed storage; .float() would reinterpret nibbles as values.
            if param.dtype != torch.float32 and param.is_floating_point():
                param.data = param.data.float()
                upcast_count += 1

        if is_global_main_process():
            logger.info(
                f"✓ FP32 master weights for {upcast_count} non-EP params "
                f"(skipped {skipped_ep_count} EP params, compute stays BF16)"
            )

    def _setup_ep_gradient_checkpointing(self):
        """Setup EP-safe gradient checkpointing.

        Handled here, not by HF Trainer, because EP's recompute must replay the checkpoint frame's
        dispatch/combine results rather than issue a second DeepEP cycle.
        """
        if not self._has_ep_layers:
            return

        config = self.parallelism_config
        gc_enabled = self.args.gradient_checkpointing

        # Rejected at config time too (shared predicate); re-checked for hand-built configs.
        if config.is_racy_single_domain_multigroup_ep:
            raise RuntimeError(config.racy_ep_topology_message)

        # Combine and cross-replica grad-sync are different-membership collectives, hence deferred averaging.
        if config.num_ep_groups > 1 and not config.is_expert_tp_mode and is_global_main_process():
            scope = "node-local across domains" if config.is_node_local_ep else "cross-node"
            if self._ep_config.is_deferred_dp:
                logger.info(
                    f"Multi-group EP ({scope}, {config.num_ep_groups} EP groups, "
                    f"ep_group_size={config.ep_group_size}): cross-replica DP average deferred to a "
                    "post-backward sweep (EP-group FSDP + DeepEP combine only during backward)."
                )

        if not gc_enabled:
            return

        # Class-declared contract; a class that does not set the flag is treated as supported.
        unsupported = sorted(
            {
                type(module).__name__
                for module in named_ep_layers(self.model).values()
                if not module._supports_gradient_checkpointing
            }
        )
        if unsupported:
            raise ValueError(
                f"gradient_checkpointing=True is not supported with Expert Parallelism for "
                f"{', '.join(unsupported)}: the family declares _supports_gradient_checkpointing="
                f"False (see the EP layer class docstring for the mechanism — e.g. Zaya's EDA/CCA "
                f"cross-layer state makes checkpoint recompute recurse polynomially). Disable "
                f"gradient_checkpointing for this model."
            )

        gc_kwargs = getattr(self.args, "gradient_checkpointing_kwargs", None) or {}
        # Outside PP, non-reentrant recompute fires lazily and desyncs ranks past the DeepEP barrier;
        # under PP reentrant runs the original forward in no_grad, so FSDP2 registers no pre-backward hooks.
        use_reentrant = not config.is_pp_mode
        if gc_kwargs.get("use_reentrant") not in (None, use_reentrant) and is_global_main_process():
            logger.warning(
                "Expert Parallelism %s use_reentrant=%s for gradient checkpointing; overriding the "
                "configured use_reentrant=%s.",
                "under pipeline parallelism requires" if config.is_pp_mode else "requires",
                use_reentrant,
                gc_kwargs["use_reentrant"],
            )
        gc_kwargs = {**gc_kwargs, "use_reentrant": use_reentrant}
        enable_ep_gradient_checkpointing(self.model, gradient_checkpointing_kwargs=gc_kwargs)

        # TRL's disable_gradient_checkpointing context manager re-enables GC from these kwargs on exit.
        self.args.gradient_checkpointing_kwargs = gc_kwargs

        self.args.gradient_checkpointing = False
