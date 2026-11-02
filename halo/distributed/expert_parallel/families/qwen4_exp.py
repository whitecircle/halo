"""EP layer for Qwen3.8-Flash-Next (``qwen4_exp``): Qwen3-Next's router contract over Qwen3.5-MoE's block."""

from __future__ import annotations

from halo.distributed.expert_parallel.families.qwen3_next import EPQwen3NextMoELayer
from halo.models.moe.family_specs import QWEN4_EXP


class EPQwen4ExpMoELayer(EPQwen3NextMoELayer):
    """EP layer for ``Qwen4ExpTextSparseMoeBlock``.

    Fused ``gate_up_proj``/``down_proj`` experts and a sigmoid-gated shared expert, as Qwen3.5-MoE's;
    ``Qwen4ExpTextTopKRouter`` renormalizes the top-k weights only under ``norm_topk_prob``, as Qwen3-Next's.
    """

    HF_MODULE_NAMES = ("Qwen4ExpTextSparseMoeBlock",)
    FAMILY = QWEN4_EXP
