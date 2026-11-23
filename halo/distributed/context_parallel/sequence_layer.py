"""Base contract for Ulysses wrappers of sequence-state layers."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch.nn as nn


class UlyssesSequenceLayerBase(nn.Module, ABC):
    """A sequence-state layer that exchanges its full row across CP ranks before stateful operations."""

    HF_MODULE_NAMES: tuple[str, ...] = ()

    @staticmethod
    @abstractmethod
    def reject_unsupported_geometry(module: nn.Module, cp_size: int) -> None:
        """Refuse a layer geometry that cannot exchange full rows over this CP group."""
