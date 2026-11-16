"""``apply_lowp_compute`` refuses at load an EP layer whose expert weights contract over an axis the format cannot
block. Otherwise the first MoE forward raises inside the quantizer."""

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
from halo.kernels.lowp import deepgemm
from halo.kernels.lowp.spec import LowpSpec
from halo.model_loading.parallel.lowp_compute import apply_lowp_compute
from halo.models.moe.expert_lora_spec import ExpertLoraSpec
from tests.common.ep_reference import dispatched_receive_counts
from tests.common.parallelism import stubbed_rank_ep_config

E, H, K = 4, 32, 2  # experts, hidden (a multiple of every block size), top_k
WORLD, EP_SIZE = 2, 1  # Pure ETP, two-way.
# Two-way shards of 48 (16 | 48, 32 does not) and 64 (both divide).
MISALIGNED_M, ALIGNED_M = 96, 128
MISALIGNED_H = 48  # A hidden size, which no expert-TP size shards, that only the 16-block format divides.
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
    """Under pure ETP its down projection contracts over ``intermediate / 2``."""
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
    assert layer.expert_tp_size == WORLD // ep_size and layer._use_grouped_mm is grouped
    return layer


def _run_experts(layer: nn.Module) -> torch.Tensor:
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
    """A smaller ``expert_tensor_parallel_size`` leaves that axis whole, so the error does not offer it."""
    model = _Model(_layer(family, ALIGNED_M, grouped=True, hidden=MISALIGNED_H, ep_size=ep_size))
    with pytest.raises(ValueError) as raised:
        apply_lowp_compute(model, LowpSpec(precision="fp8"), expert_weight_cacheable=False)
    message = str(raised.value)
    for fragment in (LAYER_PATH, "gate", "over 48", "['fp4']", "lowp_apply_moe_experts"):
        assert fragment in message, f"{fragment!r} missing from: {message}"
    assert "expert_tensor_parallel_size" not in message, message


@pytest.mark.parametrize("family", list(FAMILIES))
def test_native_without_deep_gemm_is_refused_at_load(family, monkeypatch):
    """The Hopper image ships no ``deep_gemm``; a simulated run in its place would train other numerics silently."""
    monkeypatch.setattr(deepgemm, "_deep_gemm", lambda: None)
    model = _Model(_layer(family, ALIGNED_M, grouped=True))
    with pytest.raises(RuntimeError, match="needs the deep_gemm package"):
        apply_lowp_compute(
            model, LowpSpec(precision="fp8", backend="native", apply_dense_mlp=False), expert_weight_cacheable=False
        )
    assert not model.layers[0].mlp._lowp_native


@pytest.mark.parametrize("family", list(FAMILIES))
@pytest.mark.parametrize("backward", ["bf16", "fp8"])
def test_native_reaches_the_layer_with_its_backward(family, backward, monkeypatch):
    monkeypatch.setattr(deepgemm, "_deep_gemm", lambda: (object(), object()))
    model = _Model(_layer(family, ALIGNED_M, grouped=True))
    lowp = LowpSpec(precision="fp8", backend="native", backward_precision=backward, apply_dense_mlp=False)
    assert apply_lowp_compute(model, lowp, expert_weight_cacheable=False)["moe_layers"] == 1
    layer = model.layers[0].mlp
    assert (layer._lowp_quant_format, layer._lowp_native, layer._lowp_fp8_backward) == (
        "mxfp8",
        True,
        backward == "fp8",
    )


@pytest.mark.parametrize("family", list(FAMILIES))
def test_the_refusal_guards_a_forward_that_fails(family):
    """The premise: past the load, the first grouped forward fails inside the quantizer."""
    layer = _layer(family, MISALIGNED_M, grouped=True)
    assert layer.set_lowp_compute(LowpSpec(precision="fp8"), weight_cacheable=False)
    with pytest.raises(ValueError, match="not divisible by block_size 32"):
        _run_experts(layer)


@pytest.mark.parametrize("family", list(FAMILIES))
@pytest.mark.parametrize(
    ("intermediate", "precision"), [(MISALIGNED_M, "fp4"), (ALIGNED_M, "fp8"), (ALIGNED_M, "mxfp4")]
)
def test_an_aligned_expert_shard_loads_and_runs(family, intermediate, precision):
    model = _Model(_layer(family, intermediate, grouped=True))
    summary = apply_lowp_compute(model, LowpSpec(precision=precision), expert_weight_cacheable=False)
    assert summary["moe_layers"] == 1
    lowp_out = _run_experts(model.layers[0].mlp)
    assert torch.isfinite(lowp_out).all()
    assert not torch.equal(lowp_out, _run_experts(_layer(family, intermediate, grouped=True)))


@pytest.mark.parametrize("family", list(FAMILIES))
def test_a_loop_path_layer_is_not_checked(family, caplog):
    """The per-expert loop never quantizes."""
    model = _Model(_layer(family, MISALIGNED_M, grouped=False))
    model.dense = nn.Module()
    model.dense.mlp = nn.Module()
    model.dense.mlp.down_proj = nn.Linear(H, H)  # One dense linear, so the run converts something.
    apply_lowp_compute(model, LowpSpec(precision="fp8"), expert_weight_cacheable=False)
    assert "per-expert LOOP path" in caplog.text


@pytest.mark.parametrize("family", list(FAMILIES))
def test_a_kept_block_is_not_checked(family):
    """``lowp_keep_last_blocks`` leaves the last block's experts bf16."""

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
    """Adapters stay bf16; a misaligned hidden size is refused on the adapted base weight."""
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
