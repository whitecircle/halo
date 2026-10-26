"""The balancing bias on EP MoE layers: the DeepSeek-V3 sign-updated bias, adopted into the family's own
exported slot where one exists or kept in a transient side-buffer, and the expert-load counts
:class:`~src.callbacks.balancing.BalancingBiasCallback` steers it by.

A bias outside the checkpoint leaves a served copy routing on the pretrained one, so ``moe_balancing:
bias_update`` rejects a family that only holds the side-buffer; ``bias_update_transient`` opts into it.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

import torch
import torch.nn as nn
from torch.distributed.tensor import DTensor

from src.kernels.histogram import accumulate_bincount
from src.models.gradient_checkpointing import counts_toward_expert_load
from src.models.moe_balancing import NATIVE_BALANCING_BIAS_ADOPTED_ATTR, resolve_balancing_slot
from src.runtime.ranks import is_global_main_process

logger = logging.getLogger(__name__)


class EPBalancingBiasMixin:
    """Bias-update balancing state, mixed into :class:`EPMoELayerBase`. A family declares where its bias lives
    and whether a transient one would be read at all; :class:`EPRoutingMixin` applies it."""

    num_experts: int
    _resident_device: Callable[[], torch.device]

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
        """Whether this instance may lack its declared native slot. Default False: an absent slot is an
        upstream rename; a family whose architecture gates the slot off answers off the live tree."""
        return False

    def enable_bias_balancing(self) -> bool:
        """Create bias-update balancing state, or no-op (returning False) for aux-loss families."""
        if not self._supports_bias_balancing:
            return False
        self._create_balancing_bias()
        return True

    def can_adopt_native_balancing(self) -> bool:
        """Whether bias-update balancing here lands in checkpoint-exported state: the declared native slot
        exists or can be materialized. False leaves only the side-buffer, which strict ``bias_update`` rejects.
        """
        if not self._supports_bias_balancing or not self._NATIVE_BALANCING_BIAS_ATTR:
            return False
        return self._native_balancing_target() is not None or self._can_materialize_native_balancing_slot()

    def _can_materialize_native_balancing_slot(self) -> bool:
        """Whether the family implements :meth:`_materialize_native_balancing_slot` (a config-gated slot, LFM-2
        ``use_expert_bias``)."""
        return (
            type(self)._materialize_native_balancing_slot
            is not EPBalancingBiasMixin._materialize_native_balancing_slot
        )

    def _materialize_native_balancing_slot(self) -> bool:
        """Create the declared native slot on an instance shipped without it; True on success, which the
        base never claims."""
        return False

    def _native_balancing_target(self) -> tuple[nn.Module, str] | None:
        """``(owner_module, attr_name)`` of the declared native slot, None where it is undeclared or this
        instance lacks it."""
        return resolve_balancing_slot(self, self._NATIVE_BALANCING_BIAS_ATTR)

    @staticmethod
    def _set_native_balancing_tensor(owner: nn.Module, name: str, value: torch.Tensor) -> None:
        """Set the slot through its owner, keeping a ``nn.Parameter`` slot a Parameter (``nn.Module`` refuses a
        plain tensor there)."""
        current = getattr(owner, name)
        if isinstance(current, nn.Parameter) and not isinstance(value, nn.Parameter):
            value = nn.Parameter(value, requires_grad=current.requires_grad)
        setattr(owner, name, value)

    def _create_balancing_bias(self) -> None:
        """Attach the bias-update state once: the native slot re-registered as an fp32 persistent buffer (out of
        grad and FSDP2 sharding), else a plain fp32 side-buffer that never exports (the sidecar restores it)."""
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
        self.balancing_biases = torch.zeros(self.num_experts, dtype=torch.float32, device=self._resident_device())
        self.expert_load_counter = None
        if is_global_main_process():
            logger.info(f"{type(self).__name__}: bias-update balancing enabled ({self.num_experts} experts)")

    @property
    def balancing_biases(self) -> torch.Tensor:
        """The adopted native tensor or the side-buffer; ``AttributeError`` until :meth:`enable_bias_balancing`,
        since every consumer keys on its presence. Native mode reads the owner live, so in-place updates and
        the sidecar restore reach the tensor the gate reads.
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
        """The side-buffer on ``ref``'s device (moved once: ``module.to()`` skips a plain attribute), None when
        disabled or in native mode, where the gate already applies the slot."""
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
        """Accumulate per-expert selection counts while training, once per microbatch in every checkpoint
        mode (:func:`counts_toward_expert_load`)."""
        if getattr(self, "balancing_biases", None) is None or not self.training or not counts_toward_expert_load():
            return
        with torch.no_grad():
            self.expert_load_counter = accumulate_bincount(self.expert_load_counter, indices, self.num_experts)
