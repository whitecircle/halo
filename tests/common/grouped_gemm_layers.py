"""Real EP layers whose expert widths sit past torch's grouped-GEMM tile switch, and dispatched receives whose rows per
expert fall on either side of it: the cases of the checks that the EP warm-up launches every grouped-GEMM kernel the expert compute can (kernel
classes on CPU, ``tests/cpu/parallelism/test_ep_activation_warmup.py``; lazy loads on GPU,
``tests/gpu/parallelism/ep/test_ep_sort_sync_free.py``).
"""

from __future__ import annotations

import torch
from torch import nn
from transformers.models.gpt_oss.configuration_gpt_oss import GptOssConfig
from transformers.models.gpt_oss.modeling_gpt_oss import GptOssMLP
from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeTextConfig
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeSparseMoeBlock
from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeSparseMoeBlock

from halo.distributed.expert_parallel.deepep.capacity import padded_wire_hidden
from halo.distributed.expert_parallel.families.gpt_oss import EPGptOssMoELayer
from halo.distributed.expert_parallel.families.qwen3 import EPQwen3MoELayer
from halo.distributed.expert_parallel.families.qwen3_5 import EPQwen3_5MoELayer
from halo.distributed.expert_parallel.moe_layer.layer import EPMoELayerBase
from halo.models.moe.expert_lora_spec import ExpertLoraSpec
from tests.common.ep_reference import dispatched_receive_counts
from tests.common.ep_stubs import stubbed_rank_ep_layer
from tests.common.kernel_launches import GROUPED_MM_SMALL_TILE_DIM
from tests.common.models import TINY_GPTOSS_CONFIG, TINY_QWEN3_MOE_CONFIG, TINY_QWEN35_MOE_CONFIG

# Expert widths past the small tile's, so every projection outside expert TP can take either tile.
GEMM_WIDTH = 256
# (EP layer, HF block, its config, the config's fields) of the expert layouts the grouped GEMM is handed: GptOss's
# de-interleaved pair with biases, Qwen3's separate halves, Qwen3.5's fused pair.
GEMM_FAMILIES = {
    "gpt_oss": (
        EPGptOssMoELayer,
        GptOssMLP,
        GptOssConfig,
        {**TINY_GPTOSS_CONFIG, "hidden_size": GEMM_WIDTH, "intermediate_size": GEMM_WIDTH},
    ),
    "qwen3_moe": (
        EPQwen3MoELayer,
        Qwen3MoeSparseMoeBlock,
        Qwen3MoeConfig,
        {**TINY_QWEN3_MOE_CONFIG, "hidden_size": GEMM_WIDTH, "moe_intermediate_size": GEMM_WIDTH},
    ),
    "qwen3_5_moe": (
        EPQwen3_5MoELayer,
        Qwen3_5MoeSparseMoeBlock,
        Qwen3_5MoeTextConfig,
        {**TINY_QWEN35_MOE_CONFIG, "hidden_size": GEMM_WIDTH, "moe_intermediate_size": GEMM_WIDTH},
    ),
}
# (world size, ep_size) of each rank layout; expert TP is the quotient.
GEMM_LAYOUTS = {"ep1": (1, 1), "ep2": (2, 2), "etp2": (2, 1), "ep4_etp2": (8, 4)}
# EPConfig fields of each variant; mxfp8 is the fake-quant compute set on the built layer.
GEMM_VARIANTS = {
    "defaults": {},
    "fp32_experts": {"fp32_experts": True},
    "expert_lora": {"expert_lora": ExpertLoraSpec(r=8, alpha=16.0, dropout=0.1)},
    "mxfp8": {},
}
# Local slots per local expert of the swept receives: one to past the tile switch.
SWEPT_ROWS_PER_EXPERT = (1, 64, GROUPED_MM_SMALL_TILE_DIM, GROUPED_MM_SMALL_TILE_DIM + 1, 192)


def gemm_layer(family: str, layout: str, variant: str, device: torch.device | str = "cpu") -> EPMoELayerBase:
    """Rank 0's bf16 EP layer of ``family`` in ``layout`` and ``variant`` on ``device``, storing the grouped layout."""
    layer_cls, block_cls, config_cls, fields = GEMM_FAMILIES[family]
    torch.manual_seed(0)
    block = block_cls(config_cls(**fields))
    for parameter in block.parameters():
        nn.init.normal_(parameter, std=0.02)
    world_size, ep_size = GEMM_LAYOUTS[layout]
    layer = stubbed_rank_ep_layer(
        layer_cls, block.to(torch.bfloat16), world_size, ep_size, 0, grouped=True, **GEMM_VARIANTS[variant]
    ).to(device)
    if variant == "mxfp8":
        layer.set_lowp_compute("mxfp8", weight_cacheable=True)
    return layer


def routed_receive(layer: EPMoELayerBase, rows_per_expert: int):
    """``(tokens, local expert ids, gate weights, receive counts)`` as ``layer``'s expert compute receives them, with
    ``rows_per_expert`` local slots per local expert: each token routed to ``top_k`` distinct local experts, round
    robin; at ``ep_size > 1`` the tokens a view of a receive buffer at DeepEP's padded wire width."""
    device = next(layer.parameters()).device
    local, top_k, hidden = layer.experts_per_rank, layer.top_k, layer.hidden_dim
    rows = -(-rows_per_expert * local // top_k)
    experts = (torch.arange(rows, device=device).unsqueeze(1) * top_k + torch.arange(top_k, device=device)) % local
    width = padded_wire_hidden(hidden) if layer.ep_size > 1 else hidden
    tokens = torch.zeros((rows, width), device=device, dtype=torch.bfloat16, requires_grad=True)[:, :hidden]
    weights = torch.full((rows, top_k), 1.0 / top_k, device=device, requires_grad=True)
    return tokens, experts, weights, dispatched_receive_counts(layer, experts)
