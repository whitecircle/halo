"""The balancing bias on EP MoE layers: the DeepSeek-V3 sign-updated bias, adopted into the family's exported slot or
kept in a transient side-buffer. A bias outside the checkpoint leaves a served copy routing on the pretrained one, so
``moe_balancing: bias_update`` rejects a side-buffer-only family; ``bias_update_transient`` opts in."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import ClassVar

import torch
import torch.nn as nn
from torch.distributed.tensor import DTensor

from halo.kernels.histogram import accumulate_bincount
from halo.models.gradient_checkpointing import counts_toward_expert_load
from halo.models.moe.balancing import (
    BALANCING_BIASES_ATTR,
    NATIVE_BALANCING_BIAS_ADOPTED_ATTR,
    resolve_balancing_slot,
    resolve_expert_load_weights,
)
from halo.models.moe.family_specs import MoEFamilySpec
from halo.runtime.ranks import is_global_main_process

logger = logging.getLogger(__name__)


class EPBalancingBiasMixin:
    """Bias-update balancing state. The family's spec declares where its bias lives, the class whether a transient one
    would be read at all."""

    FAMILY: ClassVar[MoEFamilySpec]
    num_experts: int
    _resident_device: Callable[[], torch.device]

    # True when the EP layer selects routing itself (bias before top-k); False = aux-loss balancing.
    SUPPORTS_BIAS_BALANCING: bool = False

    # False when selection happens entirely inside the hub gate (Bailing), where a transient
    # side-buffer could never shift routing: ``_ensure_balancing_bias`` raises instead of falling
    # back to a buffer the callback would sign-update while selection ignores it.
    _supports_transient_balancing_bias: bool = True

    # True when the EP layer bypasses the HF router module, severing the aux-loss path;
    # ``moe_balancing: auto`` then resolves the family to ``bias_update``.
    SEVERS_AUX_LOSS: bool = False

    @property
    def _native_balancing_attr(self) -> str | None:
        """The live path of ``FAMILY.balancing_slot``, a checkpoint-persistent bias the family's selection already
        applies, so ``_balancing_bias`` returns None once adopted (a second application would double-bias)."""
        slot = self.FAMILY.balancing_slot
        return slot.attrs[0] if slot else None

    def _native_slot_absence_is_legal(self) -> bool:
        """Whether this instance may lack its declared native slot. Default False: an absent slot is an
        upstream rename; a family whose architecture gates the slot off answers off the live tree."""
        return False

    def enable_bias_balancing(self) -> bool:
        if not self.SUPPORTS_BIAS_BALANCING:
            return False
        self._ensure_balancing_bias()
        return True

    def can_adopt_native_balancing(self) -> bool:
        """Whether bias-update balancing here lands in checkpoint-exported state; False leaves only the side-buffer,
        which strict ``bias_update`` rejects."""
        if not self.SUPPORTS_BIAS_BALANCING or not self._native_balancing_attr:
            return False
        return self._native_balancing_target() is not None or self._can_materialize_native_balancing_slot()

    def _can_materialize_native_balancing_slot(self) -> bool:
        """Whether the family implements :meth:`_materialize_native_balancing_slot` (LFM-2 ``use_expert_bias``)."""
        return (
            type(self)._materialize_native_balancing_slot
            is not EPBalancingBiasMixin._materialize_native_balancing_slot
        )

    def _materialize_native_balancing_slot(self) -> bool:
        """Create the declared native slot on an instance shipped without it; True on success."""
        return False

    def _native_balancing_target(self) -> tuple[nn.Module, str] | None:
        """``(owner_module, attr_name)`` of the declared slot, None where undeclared or this instance lacks it."""
        return resolve_balancing_slot(self, self._native_balancing_attr)

    @staticmethod
    def _set_native_balancing_tensor(owner: nn.Module, name: str, value: torch.Tensor) -> None:
        """Set the slot through its owner, keeping a Parameter slot a Parameter (``nn.Module`` refuses a tensor)."""
        current = getattr(owner, name)
        if isinstance(current, nn.Parameter) and not isinstance(value, nn.Parameter):
            value = nn.Parameter(value, requires_grad=current.requires_grad)
        setattr(owner, name, value)

    def _ensure_balancing_bias(self) -> None:
        """Attach the bias-update state once: the native slot re-registered as an fp32 persistent buffer (out of
        grad and FSDP2 sharding), else a plain fp32 side-buffer that never exports (the sidecar restores it)."""
        if getattr(self, BALANCING_BIASES_ATTR, None) is not None:
            return
        target = self._native_balancing_target()
        if target is None and self._native_balancing_attr and self._materialize_native_balancing_slot():
            target = self._native_balancing_target()
        if target is None and self._native_balancing_attr and self._supports_transient_balancing_bias:
            # Declared but absent and not materializable, typically an upstream rename. Strict ``bias_update`` rejects
            # this fallback at the strategy layer.
            logger.warning(
                f"{type(self).__name__}: declares '{self._native_balancing_attr}' as its native "
                f"balancing slot, but this instance has no such tensor — falling back to the "
                f"transient side-buffer, so the trained bias will NOT export with checkpoints. If "
                f"the family did not disable its bias, the upstream slot was renamed: update "
                f"its spec's balancing_slot."
            )
        if target is not None:
            owner, name = target
            native = getattr(owner, name)
            if native.is_meta:
                raise RuntimeError(
                    f"{type(self).__name__}: native balancing tensor "
                    f"'{self._native_balancing_attr}' is still on meta — the loader never "
                    f"materialized it, so bias updates would be lost. Fix the load path; do not "
                    f"silently fall back to a zero bias."
                )
            if isinstance(native, DTensor):
                if self._supports_transient_balancing_bias:
                    logger.warning(
                        f"{type(self).__name__}: native balancing tensor "
                        f"'{self._native_balancing_attr}' is already FSDP-sharded (DTensor); "
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
                        f"'{self._native_balancing_attr}' (exports with every checkpoint)"
                    )
                return
        if not self._supports_transient_balancing_bias:
            raise RuntimeError(
                f"{type(self).__name__}: routing happens entirely inside the hub gate, so a transient "
                f"side-buffer could never shift expert selection — bias-update balancing requires the "
                f"native '{self._native_balancing_attr}' slot, which is unavailable here (renamed "
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
        """The adopted native tensor or the side-buffer; ``AttributeError`` until :meth:`enable_bias_balancing`, since
        every consumer keys on its presence. Native mode reads the owner live, the tensor the gate reads."""
        if self.__dict__.get(NATIVE_BALANCING_BIAS_ADOPTED_ATTR, False):
            target = self._native_balancing_target()
            if target is None:  # the slot was externally deregistered: absent, not a crash
                raise AttributeError(BALANCING_BIASES_ATTR)
            return getattr(*target)
        if BALANCING_BIASES_ATTR in self.__dict__:
            return self.__dict__[BALANCING_BIASES_ATTR]
        raise AttributeError(BALANCING_BIASES_ATTR)

    @balancing_biases.setter
    def balancing_biases(self, value: torch.Tensor) -> None:
        if self.__dict__.get(NATIVE_BALANCING_BIAS_ADOPTED_ATTR, False):
            target = self._native_balancing_target()
            if target is None:
                raise AttributeError(BALANCING_BIASES_ATTR)
            self._set_native_balancing_tensor(*target, value)
        else:
            self.__dict__[BALANCING_BIASES_ATTR] = value

    def _balancing_bias(self, ref: torch.Tensor):
        """The side-buffer on ``ref``'s device (moved once: ``module.to()`` skips a plain attribute), None when
        disabled or in native mode, where the gate already applies the slot."""
        if self.__dict__.get(NATIVE_BALANCING_BIAS_ADOPTED_ATTR, False):
            return None
        # getattr, not a __dict__ probe: hand-built stubs may carry the buffer as a class attribute.
        bias = getattr(self, BALANCING_BIASES_ATTR, None)
        if bias is None:
            return None
        if bias.device != ref.device:
            bias = bias.to(ref.device)
            self.balancing_biases = bias
        return bias

    def _record_expert_load(self, indices: torch.Tensor) -> None:
        """Accumulate per-expert selection counts while training, once per microbatch in every checkpoint mode."""
        if getattr(self, BALANCING_BIASES_ATTR, None) is None or not self.training or not counts_toward_expert_load():
            return
        with torch.no_grad():
            self.expert_load_counter = accumulate_bincount(
                self.expert_load_counter, indices, self.num_experts, weights=resolve_expert_load_weights(indices)
            )
