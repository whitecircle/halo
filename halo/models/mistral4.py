"""Toolkit patch for the upstream Mistral4 modeling."""

from __future__ import annotations

import logging

from transformers.models.mistral4 import modeling_mistral4 as m4

from halo.models.attention.packed_position_ids import install_packed_position_ids_injection
from halo.runtime.ranks import is_global_main_process

logger = logging.getLogger(__name__)


def patch_mistral4_flash_packed_position_ids() -> None:
    """Mistral4's attention forward takes ``position_ids`` for the attention scale and passes only ``**kwargs`` on."""
    installed = install_packed_position_ids_injection(m4, m4.Mistral4Attention, lambda attention: (attention,))
    if installed and is_global_main_process():
        logger.info("Patched Mistral4 flash attention to receive position_ids (packed-document isolation)")
