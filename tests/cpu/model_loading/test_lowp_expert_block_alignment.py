"""``apply_lowp_compute`` refuses, at load, an EP layer whose grouped-GEMM expert weights contract over an axis
the run's block-scaled format cannot split into blocks.

The fake-quant blocks both operands of every expert projection along the contraction axis: the hidden size for
gate/up, this rank's expert-TP shard of the intermediate for down. Expert TP divides that shard, so a format and
an ``expert_tp_size`` the config accepts can still leave it misaligned, and the first MoE forward would raise
deep inside the quantizer. Only what the conversion quantizes is checked: the expert-LoRA adapters (bf16 under
lowp) and the experts of a kept block load whatever their widths. Every layer here is a real family EP layer
(GptOss, whose ETP layers take grouped GEMM, and Qwen3-MoE, for the family-agnostic check) for one rank of a
world-2 job, pure ETP unless a test says otherwise; only DeepEP is stubbed.

    python tests/cpu/model_loading/test_lowp_expert_block_alignment.py
"""

from __future__ import annotations

import copy
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn
from transformers.models.gpt_oss.configuration_gpt_oss import GptOssConfig
from transformers.models.gpt_oss.modeling_gpt_oss import GptOssMLP
from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeSparseMoeBlock

from halo.distributed.expert_parallel import config as ep_config_module
from halo.distributed.expert_parallel.families.gpt_oss import EPGptOssMoELayer
from halo.distributed.expert_parallel.families.qwen3 import EPQwen3MoELayer
from halo.distributed.expert_parallel.moe_layer import layer as layer_module
from halo.kernels.lowp.spec import LowpSpec
from halo.model_loading.parallel.lowp_compute import apply_lowp_compute
from halo.models.moe.expert_lora_spec import ExpertLoraSpec
from tests.common.ep_reference import dispatched_receive_counts
from tests.common.parallelism import stubbed_rank_ep_config

E, H, K = 4, 32, 2  # experts, hidden (a multiple of every block size), top_k
WORLD, EP_SIZE = 2, 1  # pure ETP, two-way
# Expert intermediates whose two-way shard is 48 (16 | 48, 32 does not) and 64 (both divide).
MISALIGNED_M, ALIGNED_M = 96, 128
MISALIGNED_H = 48  # a hidden size, which no expert-TP size shards, that only the 16-block format divides
LAYER_PATH = "layers.0.mlp"
# Expert LoRA trains under dispatch EP alone; its lora_B [E, r, N] contracts over r, which 32-blocks cannot split.
LORA_EP_SIZE, LORA = WORLD, ExpertLoraSpec(r=16, alpha=32)


class _NoDispatcher:
    def __init__(self, *args, **kwargs):
        pass


def _gpt_oss(hidden: int, intermediate: int) -> tuple[type, nn.Module]:
    config = GptOssConfig(
        hidden_size=hidden,
        intermediate_size=intermediate,
        num_local_experts=E,
        num_experts_per_tok=K,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        vocab_size=32,
    )
    return EPGptOssMoELayer, GptOssMLP(config)


def _qwen3_moe(hidden: int, intermediate: int) -> tuple[type, nn.Module]:
    config = Qwen3MoeConfig(
        hidden_size=hidden,
        intermediate_size=4 * hidden,
        moe_intermediate_size=intermediate,
        num_experts=E,
        num_experts_per_tok=K,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        vocab_size=32,
    )
    return EPQwen3MoELayer, Qwen3MoeSparseMoeBlock(config)


FAMILIES = {"gpt_oss": _gpt_oss, "qwen3_moe": _qwen3_moe}


class _Model(nn.Module):
    """One decoder block per EP layer, its MLP, so each layer sits where a loaded model puts it."""

    def __init__(self, *layers: nn.Module):
        super().__init__()
        self.layers = nn.ModuleList(nn.Module() for _ in layers)
        for block, layer in zip(self.layers, layers, strict=True):
            block.mlp = layer


def _layer(
    family: str,
    intermediate: int,
    *,
    grouped: bool,
    hidden: int = H,
    ep_size: int = EP_SIZE,
    expert_lora: ExpertLoraSpec | None = None,
) -> nn.Module:
    """Rank 1's layer: under pure ETP its down projection contracts over ``intermediate / 2``."""
    torch.manual_seed(0)
    wrapper, block = FAMILIES[family](hidden, intermediate)
    for parameter in block.parameters():
        nn.init.normal_(parameter, std=0.2)
    config = stubbed_rank_ep_config(WORLD, ep_size, rank=1, num_experts=E, expert_lora=expert_lora)
    with (
        patch.object(layer_module, "DeepEPDispatcher", _NoDispatcher),
        patch.object(ep_config_module, "supports_grouped_mm", return_value=grouped),
    ):
        layer = wrapper(copy.deepcopy(block), config)
    assert layer.expert_tp_size == WORLD // ep_size and layer._grouped_mm_enabled() is grouped
    return layer


def _run_experts(layer: nn.Module) -> torch.Tensor:
    """The grouped expert compute on every local expert, as a dispatched forward reaches it."""
    generator = torch.Generator().manual_seed(1)
    tokens = torch.randn(8, H, generator=generator)
    experts = torch.rand(8, layer.experts_per_rank, generator=generator).argsort(dim=-1)[:, :K]
    weights = torch.rand(8, K, generator=generator)
    return layer._compute_experts(
        tokens, experts, weights, torch.float32, recv_counts=dispatched_receive_counts(layer, experts)
    )


@pytest.mark.parametrize("family", list(FAMILIES))
@pytest.mark.parametrize("precision", ["fp8", "mxfp4"])
def test_a_misaligned_expert_shard_is_refused_at_load(family, precision):
    """A 48-wide down-projection shard under a 32-block format: refused at load, naming the layer, the weight,
    the axis, the block, the expert-TP remedy, the precision whose block divides it and the opt-out."""
    model = _Model(_layer(family, MISALIGNED_M, grouped=True))
    with pytest.raises(ValueError) as raised:
        apply_lowp_compute(model, LowpSpec(precision=precision), expert_weight_cacheable=False)
    message = str(raised.value)
    for fragment in (
        LAYER_PATH,
        "down_proj",
        "over 48",
        "blocks of 32",
        "expert_tensor_parallel_size",
        "['fp4']",
        "lowp_apply_moe_experts",
    ):
        assert fragment in message, f"{fragment!r} missing from: {message}"


@pytest.mark.parametrize("family", list(FAMILIES))
@pytest.mark.parametrize("ep_size", [1, 2], ids=["etp2", "ep2"])
def test_a_misaligned_whole_axis_is_refused_without_the_expert_tp_remedy(family, ep_size):
    """A 48-wide hidden size under a 32-block format, on the ETP layout and the unsharded one: the gate projection
    is refused, and the error does not offer a smaller ``expert_tensor_parallel_size``, which leaves that axis
    whole."""
    model = _Model(_layer(family, ALIGNED_M, grouped=True, hidden=MISALIGNED_H, ep_size=ep_size))
    with pytest.raises(ValueError) as raised:
        apply_lowp_compute(model, LowpSpec(precision="fp8"), expert_weight_cacheable=False)
    message = str(raised.value)
    for fragment in (LAYER_PATH, "gate", "over 48", "['fp4']", "lowp_apply_moe_experts"):
        assert fragment in message, f"{fragment!r} missing from: {message}"
    assert "expert_tensor_parallel_size" not in message, message


@pytest.mark.parametrize("family", list(FAMILIES))
def test_the_refusal_guards_a_forward_that_fails(family):
    """The premise: past the load, the same layer's first grouped forward fails inside the quantizer."""
    layer = _layer(family, MISALIGNED_M, grouped=True)
    assert layer.set_lowp_compute("mxfp8", weight_cacheable=False)
    with pytest.raises(ValueError, match="not divisible by block_size 32"):
        _run_experts(layer)


@pytest.mark.parametrize("family", list(FAMILIES))
@pytest.mark.parametrize(
    ("intermediate", "precision"), [(MISALIGNED_M, "fp4"), (ALIGNED_M, "fp8"), (ALIGNED_M, "mxfp4")]
)
def test_an_aligned_expert_shard_loads_and_runs(family, intermediate, precision):
    """Every contraction axis a multiple of the block: the load converts the layer and its grouped forward
    runs in low precision, off the bf16 forward of the same weights."""
    model = _Model(_layer(family, intermediate, grouped=True))
    summary = apply_lowp_compute(model, LowpSpec(precision=precision), expert_weight_cacheable=False)
    assert summary["moe_layers"] == 1
    lowp_out = _run_experts(model.layers[0].mlp)
    assert torch.isfinite(lowp_out).all()
    assert not torch.equal(lowp_out, _run_experts(_layer(family, intermediate, grouped=True)))


@pytest.mark.parametrize("family", list(FAMILIES))
def test_a_loop_path_layer_is_not_checked(family, caplog):
    """The per-expert loop never quantizes, so a misaligned shard there loads with the loop-path warning."""
    model = _Model(_layer(family, MISALIGNED_M, grouped=False))
    model.dense = nn.Module()
    model.dense.mlp = nn.Module()
    model.dense.mlp.down_proj = nn.Linear(H, H)  # one dense linear, so the run converts something
    apply_lowp_compute(model, LowpSpec(precision="fp8"), expert_weight_cacheable=False)
    assert "per-expert LOOP path" in caplog.text


@pytest.mark.parametrize("family", list(FAMILIES))
def test_a_kept_block_is_not_checked(family):
    """``lowp_keep_last_blocks`` leaves the last block's experts bf16, so a misaligned shard there loads and runs
    the bf16 forward; with no block kept, the same model is refused on that block."""

    def model() -> _Model:
        return _Model(_layer(family, ALIGNED_M, grouped=True), _layer(family, MISALIGNED_M, grouped=True))

    kept = model()
    summary = apply_lowp_compute(kept, LowpSpec(precision="fp8", keep_last_blocks=1), expert_weight_cacheable=False)
    assert summary["kept_blocks"] == [1] and summary["moe_layers"] == 1
    assert torch.equal(_run_experts(kept.layers[1].mlp), _run_experts(_layer(family, MISALIGNED_M, grouped=True)))

    with pytest.raises(ValueError, match=r"layers\.1\.mlp expert weight down_proj"):
        apply_lowp_compute(model(), LowpSpec(precision="fp8"), expert_weight_cacheable=False)


@pytest.mark.parametrize("family", list(FAMILIES))
def test_expert_lora_adapters_are_exempt_and_their_bases_are_not(family):
    """Under dispatch EP with rank-16 expert LoRA: an aligned layer loads at fp8 and its grouped forward runs, the
    adapters in bf16; a misaligned hidden size is refused on the adapted base weight, not on an adapter."""
    model = _Model(_layer(family, ALIGNED_M, grouped=True, ep_size=LORA_EP_SIZE, expert_lora=LORA))
    layer = model.layers[0].mlp
    assert layer.has_expert_lora, "premise: the spec adapts this family's expert weights"
    assert apply_lowp_compute(model, LowpSpec(precision="fp8"), expert_weight_cacheable=False)["moe_layers"] == 1
    assert torch.isfinite(_run_experts(layer)).all()

    misaligned = _layer(family, ALIGNED_M, grouped=True, hidden=MISALIGNED_H, ep_size=LORA_EP_SIZE, expert_lora=LORA)
    with pytest.raises(ValueError) as raised:
        apply_lowp_compute(_Model(misaligned), LowpSpec(precision="fp8"), expert_weight_cacheable=False)
    message = str(raised.value)
    bases = {name for name, _ in misaligned.expert_named_params()} - {
        name for name, _ in misaligned.expert_lora_named_params()
    }
    assert any(f"expert weight {name} (" in message for name in bases) and "over 48" in message, message


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
