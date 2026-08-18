"""Router-balancing state for EP MoE layers.

:class:`EPRouterBalancingMixin` holds the DeepSeek-V3 sign-updated bias — in the family's own exported
slot where one exists, a transient side-buffer where it does not — and the per-expert load counts
:class:`~src.callbacks.router_bias_balancing.RouterBiasBalancingCallback` steers on.

A bias the trained model routes with must land in a slot the family's checkpoint carries, or a served
copy routes on the pretrained one. ``moe_balancing: bias_update`` rejects a family that can only hold
the transient buffer; ``bias_update_transient`` is the explicit opt-in.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.tensor import DTensor

from src.distributed.expert_parallel.gc_scope import counts_toward_expert_load
from src.distributed.runtime import current_device, is_global_main_process
from src.kernels.histogram import accumulate_bincount
from src.models.moe_balancing import NATIVE_BALANCING_BIAS_ADOPTED_ATTR, resolve_balancing_slot

logger = logging.getLogger(__name__)


class EPRouterBalancingMixin:
    """Bias-update balancing state and the selection-only biases the forward applies.

    Mixed into :class:`EPMoELayerBase`. A family declares where its balancing bias lives and whether
    a transient one would be read at all.
    """

    # True when the EP wrapper selects routing itself (bias before top-k); False = aux-loss balancing.
    _supports_bias_balancing: bool = False

    # Dotted path to a checkpoint-persistent bias the family's own selection already applies. Adopted as
    # the bias-update state so it exports with checkpoints; ``_balancing_bias`` then returns None (a
    # second application would double-bias selection).
    _NATIVE_BALANCING_BIAS_ATTR: str | None = None

    # Config field that tells serving engines to load and apply the native slot, flipped to True when a
    # materialized slot would otherwise stay dormant at inference (LFM-2 ``use_expert_bias``). None for
    # families whose slot is unconditional in the architecture.
    _NATIVE_BALANCING_CONFIG_FLAG: str | None = None

    # False when selection happens entirely inside the hub gate (Bailing), where a transient
    # side-buffer could never shift routing: ``_create_balancing_bias`` raises instead of falling
    # back to a buffer the callback would sign-update while selection ignores it.
    _supports_transient_balancing_bias: bool = True

    # True when the EP wrapper bypasses the HF router module, severing the aux-loss path;
    # ``moe_balancing: auto`` then resolves the family to ``bias_update``.
    _ep_severs_aux_loss: bool = False

    def _native_slot_absence_is_legal(self) -> bool:
        """Whether this instance may carry no :attr:`_NATIVE_BALANCING_BIAS_ATTR` tensor.

        Default False: the declaring family's selection reads the slot unconditionally, so an absent
        one is an upstream rename and :meth:`_selection_scores` raises rather than routing unbiased.
        A family whose architecture does gate the slot off overrides this off the live tree, not
        class-wide, which would disarm the read for the revisions that do carry it.
        """
        return False

    def enable_bias_balancing(self) -> bool:
        """Create bias-update balancing state, or no-op (returning False) for aux-loss families."""
        if not self._supports_bias_balancing:
            return False
        self._create_balancing_bias()
        return True

    def can_adopt_native_balancing(self) -> bool:
        """Whether bias-update balancing on this layer would land in checkpoint-exported state.

        True when the declared native slot exists on this instance, or the layer can materialize it
        (:meth:`_can_materialize_native_balancing_slot`). False means only the transient side-buffer
        is possible, which ``moe_balancing: bias_update`` rejects and ``bias_update_transient`` opts
        into.
        """
        if not self._supports_bias_balancing or not self._NATIVE_BALANCING_BIAS_ATTR:
            return False
        return self._native_balancing_target() is not None or self._can_materialize_native_balancing_slot()

    def _can_materialize_native_balancing_slot(self) -> bool:
        """Whether :meth:`_materialize_native_balancing_slot` can create the slot on this instance.

        Default: whether the family overrides the hook at all. Most native slots are unconditional in
        the architecture, so a declared-but-absent slot means an upstream rename; only a config-gated
        one (LFM-2 ``use_expert_bias``) implements the hook.
        """
        return (
            type(self)._materialize_native_balancing_slot
            is not EPRouterBalancingMixin._materialize_native_balancing_slot
        )

    def _materialize_native_balancing_slot(self) -> bool:
        """Create the declared native slot on an instance that shipped without it; True on success.

        The base returns False: an absent slot is an upstream rename unless the family declares
        otherwise."""
        return False

    def _native_balancing_target(self) -> tuple[nn.Module, str] | None:
        """Resolve ``_NATIVE_BALANCING_BIAS_ATTR`` to ``(owner_module, attr_name)``, or None.

        None when the family declares no native slot or this instance lacks it (e.g. LFM-2 with
        ``use_expert_bias: false`` never registers the buffer) — the side-buffer path then applies.
        """
        return resolve_balancing_slot(self, self._NATIVE_BALANCING_BIAS_ATTR)

    @staticmethod
    def _set_native_balancing_tensor(owner: nn.Module, name: str, value: torch.Tensor) -> None:
        """Re-register through the owner so its forward reads the new tensor, preserving Parameter-ness
        (``nn.Module`` raises on assigning a plain tensor onto a registered ``nn.Parameter`` slot)."""
        current = getattr(owner, name)
        if isinstance(current, nn.Parameter) and not isinstance(value, nn.Parameter):
            value = nn.Parameter(value, requires_grad=current.requires_grad)
        setattr(owner, name, value)

    def _create_balancing_bias(self) -> None:
        """Attach the bias-update balancing state to this layer (idempotent).

        With a usable native slot (``_NATIVE_BALANCING_BIAS_ATTR``) the model's own
        checkpoint-persistent tensor becomes the state, so the final bias exports with every gathered
        save. An ``nn.Parameter`` slot is re-registered as a persistent buffer under the same
        state-dict key, which also keeps it out of gradient training and out of FSDP2's sharding
        under ``fsdp_shard_ep1_experts``, either of which would break the callback's in-place updates.

        Otherwise a transient fp32 side-buffer is added to selection scores before top-k (gate
        weights stay unbiased). It is a plain attribute: FSDP2 ignores it, and it never exports
        (resume uses the ``router_balancing_biases.pt`` sidecar). ``expert_load_counter`` feeds
        :class:`RouterBiasBalancingCallback` in both modes.
        """
        if getattr(self, "balancing_biases", None) is not None:
            return
        target = self._native_balancing_target()
        if target is None and self._NATIVE_BALANCING_BIAS_ATTR and self._materialize_native_balancing_slot():
            target = self._native_balancing_target()
        if target is None and self._NATIVE_BALANCING_BIAS_ATTR and self._supports_transient_balancing_bias:
            # Declared but absent and not materializable, typically an upstream rename. The
            # fallback stops exporting the trained bias, which strict ``bias_update`` rejects at the
            # strategy layer.
            logger.warning(
                f"{type(self).__name__}: declares '{self._NATIVE_BALANCING_BIAS_ATTR}' as its native "
                f"balancing slot, but this instance has no such tensor — falling back to the "
                f"transient side-buffer, so the trained bias will NOT export with checkpoints. If "
                f"the family did not disable its bias, the upstream slot was renamed: update "
                f"_NATIVE_BALANCING_BIAS_ATTR."
            )
        if target is not None:
            owner, name = target
            native = getattr(owner, name)
            if native.is_meta:
                raise RuntimeError(
                    f"{type(self).__name__}: native balancing tensor "
                    f"'{self._NATIVE_BALANCING_BIAS_ATTR}' is still on meta — the loader never "
                    f"materialized it, so bias updates would be lost. Fix the load path; do not "
                    f"silently fall back to a zero bias."
                )
            if isinstance(native, DTensor):
                if self._supports_transient_balancing_bias:
                    logger.warning(
                        f"{type(self).__name__}: native balancing tensor "
                        f"'{self._NATIVE_BALANCING_BIAS_ATTR}' is already FSDP-sharded (DTensor); "
                        f"falling back to the transient side-buffer — the trained bias will NOT export "
                        f"with checkpoints. Enable balancing before FSDP sharding to adopt the native slot."
                    )
            else:
                if isinstance(native, nn.Parameter):
                    del owner._parameters[name]
                    owner.register_buffer(name, native.data.float())
                elif native.dtype != torch.float32:
                    # 1e-3 sign-steps are sub-eps in bf16 at score scale; selection is fp32 anyway.
                    self._set_native_balancing_tensor(owner, name, native.float())
                setattr(self, NATIVE_BALANCING_BIAS_ADOPTED_ATTR, True)
                self.expert_load_counter: torch.Tensor | None = None
                if is_global_main_process():
                    logger.info(
                        f"{type(self).__name__}: bias-update balancing via native "
                        f"'{self._NATIVE_BALANCING_BIAS_ATTR}' (exports with every checkpoint)"
                    )
                return
        if not self._supports_transient_balancing_bias:
            raise RuntimeError(
                f"{type(self).__name__}: routing happens entirely inside the hub gate, so a transient "
                f"side-buffer could never shift expert selection — bias-update balancing requires the "
                f"native '{self._NATIVE_BALANCING_BIAS_ATTR}' slot, which is unavailable here (renamed "
                f"upstream, or already FSDP-sharded because balancing was enabled after fully_shard). "
                f"Fix the slot or the enable ordering; a sign-updated buffer selection never reads is "
                f"not a fallback."
            )
        device = next((p.device for p in self.parameters()), None)
        if device is None or device.type == "meta":
            device = current_device()
        self.balancing_biases = torch.zeros(self.num_experts, dtype=torch.float32, device=device)
        self.expert_load_counter = None
        if is_global_main_process():
            logger.info(f"{type(self).__name__}: bias-update balancing enabled ({self.num_experts} experts)")

    @property
    def balancing_biases(self) -> torch.Tensor:
        """Bias-update state: the adopted native tensor, or the transient side-buffer.

        Raises ``AttributeError`` until :meth:`enable_bias_balancing` runs, so the attribute appears
        exactly when balancing is enabled (``iter_balancing_routers``, the sidecar persist and the
        load recording all key on its presence). Native mode returns a live view over the owner
        module, so in-place updates and the sidecar restore's ``copy_`` reach whatever tensor the gate
        currently reads.
        """
        if self.__dict__.get(NATIVE_BALANCING_BIAS_ADOPTED_ATTR, False):
            target = self._native_balancing_target()
            if target is None:  # the slot was externally deregistered — absent, not a crash
                raise AttributeError("balancing_biases")
            return getattr(*target)
        if "balancing_biases" in self.__dict__:
            return self.__dict__["balancing_biases"]
        raise AttributeError("balancing_biases")

    @balancing_biases.setter
    def balancing_biases(self, value: torch.Tensor) -> None:
        if self.__dict__.get(NATIVE_BALANCING_BIAS_ADOPTED_ATTR, False):
            target = self._native_balancing_target()
            if target is None:
                raise AttributeError("balancing_biases")
            self._set_native_balancing_tensor(*target, value)
        else:
            self.__dict__["balancing_biases"] = value

    def _balancing_bias(self, ref: torch.Tensor):
        """Return the transient side-buffer on ``ref``'s device, or None when there is nothing to add.

        None both when balancing is disabled and in native mode: the adopted tensor already sits
        inside the family's own selection arithmetic, so adding it here would double-bias selection.

        The side-buffer is a plain attribute, so ``module.to()`` doesn't move it; relocate once if the
        module migrated devices.
        """
        if self.__dict__.get(NATIVE_BALANCING_BIAS_ADOPTED_ATTR, False):
            return None
        # getattr, not a __dict__ probe: hand-built stubs may carry the buffer as a class attribute.
        bias = getattr(self, "balancing_biases", None)
        if bias is None:
            return None
        if bias.device != ref.device:
            bias = bias.to(ref.device)
            self.balancing_biases = bias
        return bias

    def _record_expert_load(self, indices: torch.Tensor) -> None:
        """Accumulate per-expert selection counts for the balancing callback (training only).

        Exactly one pass per microbatch counts, in every checkpoint mode
        (:func:`counts_toward_expert_load`)."""
        if getattr(self, "balancing_biases", None) is None or not self.training or not counts_toward_expert_load():
            return
        with torch.no_grad():
            self.expert_load_counter = accumulate_bincount(self.expert_load_counter, indices, self.num_experts)

    def _biased_topk(self, logits: torch.Tensor) -> torch.Tensor:
        """Top-k on the bias-adjusted softmax probs: selection only, no gating.

        Balancing changes only which experts are picked; the weights stay whatever the family's own
        router computes. A family whose gating is not an unbiased renormalized softmax (Qwen3 MoE,
        where ``norm_topk_prob`` is configurable and defaults off) pairs this with its own
        ``_gate_weights_at``, so a zero bias reproduces the unbalanced route exactly.
        """
        bias = self._balancing_bias(logits)
        if bias is None:
            # Native-adoption mode (GptOss ``router.bias``): the bias already sits inside the
            # logits, so select on the raw logits as the served router does. Not on softmax probs:
            # monotonicity holds in exact arithmetic, but a large logit gap underflows the losing
            # probs to exact 0.0 and top-k then breaks the tie differently than the logit order.
            _, indices = torch.topk(logits.detach(), self.top_k, dim=-1)
        else:
            probs = F.softmax(logits.float(), dim=-1)
            _, indices = torch.topk(probs.detach() + bias, self.top_k, dim=-1)
        return self._maybe_replace_selection(indices)

    def _selection_scores(self, scores: torch.Tensor) -> torch.Tensor:
        """``scores`` shifted by every selection-only bias, in the order a served copy applies them.

        First the family's own declared correction slot (:attr:`_NATIVE_BALANCING_BIAS_ATTR`), the
        tensor its pretrained router adds before top-k, absent only where the architecture gates it
        off (LFM-2 ``use_expert_bias: false``); then the transient balancing side-buffer, which is
        None in native-adoption mode because the slot above is the balancing state there. Returns
        ``scores`` itself when neither applies, so a caller can tell an unbiased selection by identity.

        Gate weights stay derived from the unbiased ``scores``: this seam decides which experts run,
        and a bias reaching the weights would rescale every routed token. :meth:`_biased_topk` is the
        sibling for families selecting on raw logits (GptOss, Qwen3), where the slot is already
        inside them.
        """
        target = self._native_balancing_target()
        if target is None and self._NATIVE_BALANCING_BIAS_ATTR and not self._native_slot_absence_is_legal():
            raise AttributeError(
                f"{type(self).__name__} declares _NATIVE_BALANCING_BIAS_ATTR="
                f"'{self._NATIVE_BALANCING_BIAS_ATTR}' as the bias its own selection applies, but this "
                f"instance carries no such tensor — the family renamed it upstream. Routing on would "
                f"silently drop the pretrained correction bias. Fix the declaration, or override "
                f"_native_slot_absence_is_legal() to answer True for the configuration that omits it."
            )
        if target is not None:
            scores = scores + getattr(*target)
        bias = self._balancing_bias(scores)
        if bias is not None:
            scores = scores + bias
        return scores
