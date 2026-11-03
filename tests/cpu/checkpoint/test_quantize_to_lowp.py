"""halo/checkpoint/tools/lowp_export.py: bf16/fp32 to block-scaled mxfp8/mxfp4/nvfp4 for quantization-aware training.
The dequantized weight must reproduce the fake-quant forward exactly."""

import json
import os
import re
import sys
import tempfile
from unittest import mock

import pytest
import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

from halo.checkpoint.reading import checkpoint_shard_files
from halo.checkpoint.tools import lowp_export
from halo.kernels.lowp.quantization import BlockScaledTensor, dequantize, fake_quant
from halo.kernels.lowp.spec import MLP_PROJECTIONS
from halo.models.moe.expert_layout import HF_FUSED_EXPERT_KEYS
from halo.models.moe.family_specs import per_expert_layouts
from tests.common.checkpoint_io import weight_files
from tests.common.repo import load_script_module

_FMT_TOL = {"mxfp8": 0.08, "nvfp4": 0.25, "mxfp4": 0.35}


def _make_checkpoint(d: str) -> torch.Tensor:
    torch.manual_seed(0)
    gate = torch.randn(4096, 2048, dtype=torch.bfloat16) * 2048**-0.5
    ckpt = {
        "model.layers.0.mlp.gate_proj.weight": gate.clone(),
        "model.layers.0.mlp.down_proj.weight": torch.randn(2048, 4096, dtype=torch.bfloat16) * 4096**-0.5,
        "model.layers.0.self_attn.q_proj.weight": torch.randn(2048, 2048, dtype=torch.bfloat16),
        "model.layers.0.input_layernorm.weight": torch.randn(2048, dtype=torch.bfloat16),
        "model.embed_tokens.weight": torch.randn(1000, 2048, dtype=torch.bfloat16),
        "model.layers.0.mlp.gate_proj.bias": torch.randn(4096, dtype=torch.bfloat16),
    }
    save_file(ckpt, os.path.join(d, "model.safetensors"), metadata={"format": "pt"})
    with open(os.path.join(d, "config.json"), "w") as f:
        json.dump({"model_type": "test"}, f)
    return gate


def _check(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)
    print(f"  {msg} — OK")


def test_quantize_to_lowp():
    for fmt in ("mxfp8", "nvfp4", "mxfp4"):
        print(f"test_quantize_to_lowp[{fmt}]")
        with tempfile.TemporaryDirectory() as root:
            src, out = os.path.join(root, "src"), os.path.join(root, "out")
            os.makedirs(src)
            gate = _make_checkpoint(src)
            lowp_export.quantize_checkpoint(src, out, fmt, verify=True)
            saved = load_file(os.path.join(out, "model.safetensors"))

            g = "model.layers.0.mlp.gate_proj"
            _check(
                all(f"{g}.{s}" in saved for s in ("weight_packed", "weight_scale", "weight_shape"))
                and f"{g}.weight" not in saved,
                "targeted MLP weight quantized (packed/scale/shape)",
            )
            _check("model.layers.0.mlp.down_proj.weight_packed" in saved, "second MLP weight quantized")
            _check(
                all(
                    k in saved
                    for k in (
                        "model.layers.0.self_attn.q_proj.weight",
                        "model.layers.0.input_layernorm.weight",
                        "model.embed_tokens.weight",
                        f"{g}.bias",
                    )
                ),
                "attention / norm / embed / bias copied unchanged",
            )

            block = 32 if fmt in ("mxfp8", "mxfp4") else 16
            # Without the per-tensor global scale, nvfp4's reconstruction is off by up to E4M3_MAX*E2M1_MAX, so only
            # that format may ship one.
            global_scale = saved.get(f"{g}.weight_global_scale")
            _check(
                (global_scale is not None) == (fmt == "nvfp4"),
                f"{fmt}: weight_global_scale present={global_scale is not None}, expected={fmt == 'nvfp4'}",
            )
            q = BlockScaledTensor(
                saved[f"{g}.weight_packed"],
                saved[f"{g}.weight_scale"],
                block,
                -1,
                packed=(fmt in ("nvfp4", "mxfp4")),
                pow2_scale=(fmt in ("mxfp8", "mxfp4")),
                # The export stores compressed-tensors' reciprocal of the in-memory multiplier.
                global_scale=None if global_scale is None else global_scale.reciprocal(),
            )
            w_deq = dequantize(q)
            round_trip = ((w_deq.float() - gate.float()).norm() / gate.float().norm()).item()
            _check(round_trip <= _FMT_TOL[fmt], f"round-trip relerr {round_trip:.4f} within format error")

            x = torch.randn(128, 2048, dtype=torch.bfloat16)
            qat = F.linear(x, fake_quant(gate, fmt, axis=-1))
            infer = F.linear(x, w_deq)
            consistency = ((infer.float() - qat.float()).norm() / qat.float().norm().clamp_min(1e-9)).item()
            _check(consistency < 1e-6, f"QAT→inference consistency exact (relerr {consistency:.1e})")

            with open(os.path.join(out, "quantization_config.json")) as f:
                mani = json.load(f)
            _check(
                mani["format"] == fmt and mani["block_size"] == block and len(mani["quantized_weights"]) == 2,
                "manifest records scheme + quantized weights",
            )


def _compressed_tensors_nvfp4_dequant(saved: dict, name: str) -> torch.Tensor:
    """Even element in the low nibble; the ``scale / global_scale`` its ``_dequantize`` applies and vLLM's nvfp4
    schemes invert at load."""
    packed = saved[f"{name}.weight_packed"]
    codes = torch.stack([packed & 0x0F, packed >> 4], dim=-1).flatten(-2).long()
    levels = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
    values = levels[codes & 0x7] * (1 - 2 * (codes >> 3))
    block_scale = saved[f"{name}.weight_scale"].float().repeat_interleave(16, dim=-1)
    return (values * (block_scale / saved[f"{name}.weight_global_scale"])).to(torch.bfloat16)


def test_the_nvfp4_export_reads_back_under_the_compressed_tensors_convention(tmp_path):
    """The in-memory multiplier stored as-is reads back rescaled by its own square."""
    src, out = tmp_path / "src", tmp_path / "out"
    src.mkdir()
    gate = _make_checkpoint(str(src))
    lowp_export.quantize_checkpoint(str(src), str(out), "nvfp4")
    saved = load_file(str(out / "model.safetensors"))
    g = "model.layers.0.mlp.gate_proj"

    ratio = saved[f"{g}.weight_global_scale"].item() * gate.float().abs().max().item() / (448.0 * 6.0)
    assert 0.5 < ratio <= 1.0, f"weight_global_scale is not compressed-tensors' amax-normalizer (ratio {ratio})"
    torch.testing.assert_close(_compressed_tensors_nvfp4_dequant(saved, g), fake_quant(gate, "nvfp4"), rtol=0, atol=0)


def _write_mlp_checkpoint(directory, gate: torch.Tensor) -> None:
    directory.mkdir()
    down = torch.randn(64, 32, dtype=torch.bfloat16)
    tensors = {"model.layers.0.mlp.gate_proj.weight": gate, "model.layers.0.mlp.down_proj.weight": down}
    save_file(tensors, str(directory / "model.safetensors"), metadata={"format": "pt"})
    (directory / "config.json").write_text(json.dumps({"model_type": "test"}))


def test_an_all_zero_nvfp4_weight_stores_a_finite_global_scale(tmp_path):
    """The reciprocal of the quantizer's floor overflows to +inf, which vLLM's max over a fused layer's partition
    scales would spread."""
    src, out = tmp_path / "src", tmp_path / "out"
    _write_mlp_checkpoint(src, torch.zeros(32, 64, dtype=torch.bfloat16))
    lowp_export.quantize_checkpoint(str(src), str(out), "nvfp4", verify=True)
    saved = load_file(str(out / "model.safetensors"))
    zero, live = "model.layers.0.mlp.gate_proj", "model.layers.0.mlp.down_proj"

    assert torch.equal(saved[f"{zero}.weight_global_scale"], torch.ones(1))
    assert torch.equal(_compressed_tensors_nvfp4_dequant(saved, zero), torch.zeros(32, 64, dtype=torch.bfloat16))
    assert saved[f"{live}.weight_global_scale"].shape == (1,)
    assert torch.isfinite(saved[f"{live}.weight_global_scale"]).all()


def test_a_nonzero_weight_below_the_global_scale_range_is_refused(tmp_path):
    """1.0 would rescale it, so the export refuses rather than write a corrupt weight."""
    src, out = tmp_path / "src", tmp_path / "out"
    _write_mlp_checkpoint(src, torch.full((32, 64), 1e-37, dtype=torch.bfloat16))
    with pytest.raises(ValueError, match="below the range an nvfp4 global scale can store"):
        lowp_export.quantize_checkpoint(str(src), str(out), "nvfp4")


def test_should_quantize_scoping():
    """Attention, norms, embeddings, router and biases stay high precision."""
    inc, exc = re.compile(lowp_export.DEFAULT_INCLUDE), re.compile(lowp_export.DEFAULT_EXCLUDE)
    w2d = torch.zeros(16, 16)

    def q(name, t=w2d):
        return lowp_export._should_quantize(name, t, inc, exc)

    assert q("model.layers.0.mlp.gate_proj.weight")
    assert q("model.layers.0.mlp.down_proj.weight")
    assert q("model.layers.0.mlp.experts.0.gate_proj.weight")
    assert q("model.layers.0.mlp.experts.0.w1.weight")
    assert q("model.layers.0.mlp.experts.0.w2.weight")
    assert q("model.layers.0.mlp.routed_experts.3.gate_proj.weight")
    assert not lowp_export._should_quantize(
        "model.layers.0.mlp.routed_experts.3.gate_proj.weight", w2d, inc, exc, apply_moe_experts=False
    )
    # LFM-2's dense ``feed_forward.w1`` is a projection the trainer never converts: it trains in bf16 and exports in
    # bf16.
    assert not q("model.layers.0.feed_forward.w1.weight")
    assert not q("model.layers.0.feed_forward.w2.weight")
    # Fused MoE experts are 3-D nn.Parameters with no trailing .weight, so they match on shape + suffix.
    w3d = torch.zeros(8, 16, 16)
    assert lowp_export._should_quantize("model.layers.0.mlp.experts.gate_up_proj", w3d, inc, exc)
    assert lowp_export._should_quantize("model.layers.0.mlp.experts.down_proj", w3d, inc, exc)
    assert not q("model.layers.0.mlp.experts.gate_up_proj")

    assert not q("model.layers.0.mlp.router.weight"), "router must stay high precision"
    assert not q("model.layers.0.mlp.gate.weight"), "MoE gate must stay high precision"
    assert not q("model.embed_tokens.weight")
    assert not q("model.layers.0.input_layernorm.weight")
    assert not q("lm_head.weight")
    assert not q("model.layers.0.mlp.gate_proj.bias")

    assert not q("model.layers.0.self_attn.q_proj.weight")

    # No contraction axis to block along.
    assert not lowp_export._should_quantize("model.layers.0.mlp.gate_proj.weight", torch.zeros(16), inc, exc)
    assert not q("model.layers.0.mlp.gate_up_proj_packed")


def test_non_block_divisible_axis_left_high_precision():
    with tempfile.TemporaryDirectory() as root:
        src, out = os.path.join(root, "src"), os.path.join(root, "out")
        os.makedirs(src)
        odd = torch.randn(64, 100, dtype=torch.bfloat16)
        save_file(
            {
                "model.layers.0.mlp.gate_proj.weight": odd,
                "model.layers.0.mlp.down_proj.weight": torch.randn(64, 64, dtype=torch.bfloat16),
            },
            os.path.join(src, "model.safetensors"),
            metadata={"format": "pt"},
        )
        with open(os.path.join(src, "config.json"), "w") as f:
            json.dump({"model_type": "test"}, f)

        lowp_export.quantize_checkpoint(src, out, "mxfp8")
        saved = load_file(os.path.join(out, "model.safetensors"))
        assert "model.layers.0.mlp.gate_proj.weight" in saved
        assert "model.layers.0.mlp.gate_proj.weight_packed" not in saved
        assert torch.equal(saved["model.layers.0.mlp.gate_proj.weight"], odd)
        with open(os.path.join(out, "quantization_config.json")) as f:
            mani = json.load(f)
        assert mani["quantized_weights"] == ["model.layers.0.mlp.down_proj.weight"]


@pytest.mark.parametrize(
    ("shapes", "scope"),
    [
        ({"model.layers.0.mlp.gate_proj.weight": (64, 64)}, {"include": "nomatch"}),
        ({"model.layers.0.mlp.gate_proj.weight": (64, 64)}, {"apply_dense_mlp": False}),
        ({"model.layers.0.mlp.gate_proj.weight": (64, 100)}, {}),
    ],
    ids=["include-matches-nothing", "dense-mlp-off", "no-block-divisible-axis"],
)
def test_an_export_that_scales_nothing_is_refused_before_any_write(tmp_path, shapes, scope):
    """The source's weights under a ``block_scaled`` config, which both engines refuse, would pass for an export."""
    src = tmp_path / "src"
    src.mkdir()
    tensors = {name: torch.randn(*shape, dtype=torch.bfloat16) for name, shape in shapes.items()}
    save_file(tensors, src / "model.safetensors", metadata={"format": "pt"})
    (src / "config.json").write_text(json.dumps({"model_type": "test"}))
    out = tmp_path / "out"

    with pytest.raises(ValueError, match="is block-scaled under this scope"):
        lowp_export.quantize_checkpoint(str(src), str(out), "mxfp8", **scope)
    assert not out.exists()


def test_multi_shard_checkpoint_quantized():
    with tempfile.TemporaryDirectory() as root:
        src, out = os.path.join(root, "src"), os.path.join(root, "out")
        os.makedirs(src)
        save_file(
            {"model.layers.0.mlp.gate_proj.weight": torch.randn(64, 64, dtype=torch.bfloat16)},
            os.path.join(src, "model-00001-of-00002.safetensors"),
            metadata={"format": "pt"},
        )
        save_file(
            {"model.layers.1.mlp.down_proj.weight": torch.randn(64, 64, dtype=torch.bfloat16)},
            os.path.join(src, "model-00002-of-00002.safetensors"),
            metadata={"format": "pt"},
        )
        with open(os.path.join(src, "config.json"), "w") as f:
            json.dump({"model_type": "test"}, f)

        lowp_export.quantize_checkpoint(src, out, "mxfp8")
        merged = {}
        for fn in ("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"):
            merged.update(load_file(os.path.join(out, fn)))
        assert "model.layers.0.mlp.gate_proj.weight_packed" in merged
        assert "model.layers.1.mlp.down_proj.weight_packed" in merged
        with open(os.path.join(out, "quantization_config.json")) as f:
            mani = json.load(f)
        assert len(mani["quantized_weights"]) == 2


def test_shard_enumeration_raises_when_empty():
    with tempfile.TemporaryDirectory() as d, pytest.raises(FileNotFoundError, match="safetensors"):
        checkpoint_shard_files(d)


def _quantize_one_expert(model_type: str, shape: tuple[int, int, int]) -> tuple[dict, dict]:
    with tempfile.TemporaryDirectory() as root:
        src, out = os.path.join(root, "src"), os.path.join(root, "out")
        os.makedirs(src)
        experts = torch.randn(*shape, dtype=torch.bfloat16) * shape[1] ** -0.5
        save_file(
            {"model.layers.0.mlp.experts.gate_up_proj": experts},
            os.path.join(src, "model.safetensors"),
            metadata={"format": "pt"},
        )
        with open(os.path.join(src, "config.json"), "w") as f:
            json.dump({"model_type": model_type}, f)
        lowp_export.quantize_checkpoint(src, out, "mxfp8", verify=True)
        saved = load_file(os.path.join(out, "model.safetensors"))
        with open(os.path.join(out, "quantization_config.json")) as f:
            mani = json.load(f)
        return saved, mani


def test_fused_glu_experts_quantized_on_last_axis():
    """Hardcoding axis 1 blocks along N=100 and silently skips these experts, or corrupts them when N is divisible."""
    base = "model.layers.0.mlp.experts.gate_up_proj"
    saved, mani = _quantize_one_expert("qwen3_5_moe", (4, 100, 64))
    assert f"{base}.weight_packed" in saved, "fused-GLU expert must quantize along the last (contraction) axis"
    assert base not in saved
    assert mani["weight_axes"][base] == -1, "manifest must record the last-axis contraction for fused-GLU"
    assert list(saved[f"{base}.weight_scale"].shape) == [4, 100, 2]


def test_gptoss_experts_quantized_on_axis1():
    """The shape is divisible by the block only on axis 1, so it quantizes iff axis 1 is chosen."""
    base = "model.layers.0.mlp.experts.gate_up_proj"
    saved, mani = _quantize_one_expert("gpt_oss", (4, 64, 100))
    assert f"{base}.weight_packed" in saved, "gpt-oss expert must quantize along axis 1 (contraction)"
    assert base not in saved
    assert mani["weight_axes"][base] == 1, "manifest must record axis-1 contraction for gpt-oss"
    assert list(saved[f"{base}.weight_scale"].shape) == [4, 2, 100]


def _write_config(tmp_path, cfg: dict) -> str:
    d = tmp_path / "ckpt"
    d.mkdir()
    with open(d / "config.json", "w") as f:
        json.dump(cfg, f)
    return str(d)


@pytest.mark.parametrize(
    "cfg,expected_axis",
    [
        ({"model_type": "gpt_oss"}, 1),
        ({"model_type": "gpt_oss_vlm", "text_config": {"model_type": "gpt_oss"}}, 1),
        ({"model_type": "qwen3_5_moe"}, -1),
        ({"model_type": "glm4_moe_lite"}, -1),
        ({"model_type": "not_a_registered_family"}, None),
    ],
)
def test_fused_expert_axis_comes_from_the_layer_class(tmp_path, cfg, expected_axis):
    """A wrong axis block-scales the output dim and silently corrupts every dequantized expert, so an unregistered
    family resolves to ``None``."""
    assert lowp_export._fused_expert_axis(_write_config(tmp_path, cfg)) == expected_axis


def test_fused_expert_axis_missing_config(tmp_path):
    assert lowp_export._fused_expert_axis(str(tmp_path)) is None


def test_unknown_family_fused_expert_raises():
    with pytest.raises(ValueError, match="matches no MoE family spec"):
        _quantize_one_expert("not_a_registered_family", (4, 64, 64))


def test_an_expert_bank_under_an_undeclared_fused_name_is_refused_before_any_write(tmp_path):
    """Only ``moe.down_proj`` is a class-declared fused key, so the export would quantize it alone and fail to
    reproduce the QAT forward."""
    src, out = tmp_path / "src", tmp_path / "out"
    src.mkdir()
    prefix = "model.language_model.layers.1.moe"
    save_file(
        {
            f"{prefix}.gate_proj": torch.randn(4, 64, 64, dtype=torch.bfloat16),
            f"{prefix}.up_proj": torch.randn(4, 64, 64, dtype=torch.bfloat16),
            f"{prefix}.down_proj": torch.randn(4, 64, 64, dtype=torch.bfloat16),
            "model.language_model.layers.0.mlp.gate_proj.weight": torch.randn(64, 64, dtype=torch.bfloat16),
        },
        os.path.join(src, "model.safetensors"),
        metadata={"format": "pt"},
    )
    with open(src / "config.json", "w") as f:
        json.dump({"model_type": "step3p7"}, f)

    with pytest.raises(ValueError, match="its name is not a fused expert key"):
        lowp_export.quantize_checkpoint(str(src), str(out), "mxfp8")
    assert not out.exists(), "a refused conversion must not create its output directory"

    lowp_export.quantize_checkpoint(str(src), str(out), "mxfp8", apply_moe_experts=False)
    saved = set(load_file(os.path.join(out, "model.safetensors")))
    assert {f"{prefix}.{name}" for name in ("gate_proj", "up_proj", "down_proj")} <= saved, (
        "with experts out of scope every bank is copied through unchanged"
    )
    assert "model.language_model.layers.0.mlp.gate_proj.weight_packed" in saved


def test_a_moe_checkpoint_whose_experts_match_nothing_is_refused(tmp_path):
    """Inkling's hub spelling matches no roster name, so the export would copy every expert in bf16 and still claim
    QAT parity."""
    src, out = tmp_path / "src", tmp_path / "out"
    src.mkdir()
    tensors = {
        "model.layers.0.mlp.gate_proj.weight": torch.randn(64, 64, dtype=torch.bfloat16),
        "model.layers.0.mlp.experts.w13_weight": torch.randn(4, 128, 64, dtype=torch.bfloat16),
        "model.layers.0.mlp.experts.w2_weight": torch.randn(4, 64, 64, dtype=torch.bfloat16),
    }
    save_file(tensors, os.path.join(src, "model.safetensors"), metadata={"format": "pt"})
    with open(src / "config.json", "w") as f:
        json.dump({"model_type": "qwen3_moe"}, f)

    with pytest.raises(ValueError, match="no expert weight is in the low-precision scope"):
        lowp_export.quantize_checkpoint(str(src), str(out), "mxfp8")
    assert not out.exists()

    lowp_export.quantize_checkpoint(str(src), str(out), "mxfp8", apply_moe_experts=False)
    written = load_file(os.path.join(out, "model.safetensors"))
    assert "model.layers.0.mlp.gate_proj.weight_packed" in written
    assert torch.equal(
        written["model.layers.0.mlp.experts.w2_weight"], tensors["model.layers.0.mlp.experts.w2_weight"]
    )


def test_vision_tower_projections_are_never_quantized():
    """``apply_lowp_compute`` converts the text backbone only; a quantized tower computes in a format training never
    saw."""
    inc, exc = re.compile(lowp_export.DEFAULT_INCLUDE), re.compile(lowp_export.DEFAULT_EXCLUDE)
    w2d = torch.zeros(16, 16)
    for name in (
        "vision_tower.vision_model.encoder.layers.0.mlp.fc1.weight",
        "model.vision_tower.encoder.layers.3.mlp.fc2.weight",
        "visual.blocks.0.mlp.gate_proj.weight",
        "model.visual.blocks.7.mlp.down_proj.weight",
        "multi_modal_projector.linear_1.weight",
        "model.multi_modal_projector.mlp.up_proj.weight",
    ):
        assert not lowp_export._should_quantize(name, w2d, inc, exc), f"{name} was quantized but trains in bf16"


def test_the_quantizable_name_set_is_derived_from_the_two_rosters():
    """A restated pattern matches names no lowp path converts, and a new per-expert layout has to be remembered
    twice."""
    declared = set(MLP_PROJECTIONS) | set(HF_FUSED_EXPERT_KEYS) | {k for keys in per_expert_layouts() for k in keys}
    assert set(lowp_export._QUANTIZABLE_WEIGHT_NAMES) == declared
    assert {"gate_proj", "up_proj", "down_proj", "gate_up_proj", "w1", "w2", "w3"} <= declared
    for never_converted in ("fc1", "fc2", "mlp", "feed_forward", "experts"):
        assert never_converted not in lowp_export._QUANTIZABLE_WEIGHT_NAMES


def test_fused_expert_names_name_no_per_expert_projection():
    """A per-expert projection among them invites quantizing the wrong tensor on a guessed axis."""
    assert {"gate_up_proj", "down_proj"} <= set(HF_FUSED_EXPERT_KEYS)
    for never_fused in ("w1", "w2", "w3", "fc1", "fc2"):
        assert never_fused not in HF_FUSED_EXPERT_KEYS


@pytest.mark.parametrize("poison", [float("nan")])
@pytest.mark.parametrize("fmt", ["mxfp8", "mxfp4", "nvfp4"])
def test_verify_refuses_a_non_finite_weight(fmt, poison):
    """The relative error is NaN and ``max(x, nan)`` returns ``x``, so the running maximum stays 0.0 and the tool
    prints OK over a poisoned export."""
    with tempfile.TemporaryDirectory() as root:
        src, out = os.path.join(root, "src"), os.path.join(root, "out")
        os.makedirs(src)
        weight = torch.randn(64, 128, dtype=torch.bfloat16) * 0.02
        weight[3, 7] = poison
        save_file({"model.layers.0.mlp.down_proj.weight": weight}, os.path.join(src, "model.safetensors"))
        with pytest.raises(ValueError, match="non-finite"):
            lowp_export.quantize_checkpoint(src, out, fmt, verify=True)


def test_verify_fails_past_the_format_tolerance(monkeypatch, tmp_path):
    """A relerr past the format's own error is structural, usually the contraction axis."""
    monkeypatch.setattr(lowp_export, "_VERIFY_RELERR_TOL", {"mxfp8": 0.0})
    src = tmp_path / "src"
    src.mkdir()
    weight = torch.randn(64, 128, dtype=torch.bfloat16) * 0.02
    save_file({"model.layers.0.mlp.down_proj.weight": weight}, os.path.join(src, "model.safetensors"))
    (src / "config.json").write_text('{"model_type": "qwen3"}')

    with pytest.raises(ValueError, match="down_proj.weight.*past the mxfp8 tolerance"):
        lowp_export.quantize_checkpoint(str(src), str(tmp_path / "out"), "mxfp8", verify=True)
    written = {path.name for path in (tmp_path / "out").iterdir()}
    assert not written & {"config.json", "quantization_config.json"}, f"a refused export must not load: {written}"


def _plant_previous_run(out_dir: str, *, single: bool, index_shards: tuple[str, ...] = ()) -> None:
    os.makedirs(out_dir, exist_ok=True)
    if single:
        save_file({"stale.weight": torch.zeros(2, 2)}, os.path.join(out_dir, "model.safetensors"))
    for shard in index_shards:
        save_file({"stale.weight": torch.zeros(2, 2)}, os.path.join(out_dir, shard))
    if index_shards:
        with open(os.path.join(out_dir, "model.safetensors.index.json"), "w") as f:
            json.dump({"metadata": {}, "weight_map": {f"stale.{i}": s for i, s in enumerate(index_shards)}}, f)


def _weight_files(out_dir: str) -> set[str]:
    return set(weight_files(out_dir))


def test_single_file_output_sweeps_a_previous_runs_index():
    """Every index-first reader would follow the stale index to shards this run does not have."""
    with tempfile.TemporaryDirectory() as root:
        src, out = os.path.join(root, "src"), os.path.join(root, "out")
        os.makedirs(src)
        _make_checkpoint(src)
        _plant_previous_run(out, single=False, index_shards=("model-00001-of-00002.safetensors",))

        lowp_export.quantize_checkpoint(src, out, "mxfp8")

        assert _weight_files(out) == {"model.safetensors"}, f"stale files survived: {sorted(_weight_files(out))}"
        assert not os.path.exists(os.path.join(out, "model.safetensors.index.json"))
        saved = load_file(os.path.join(out, "model.safetensors"))
        assert "stale.weight" not in saved and "model.layers.0.mlp.gate_proj.weight_packed" in saved
        assert os.path.isfile(os.path.join(out, "quantization_config.json"))
        assert os.path.isfile(os.path.join(out, "config.json"))


def test_sharded_output_keeps_its_own_index_and_drops_the_stale_single_file():
    """``from_pretrained`` prefers a leftover ``model.safetensors`` over the index, serving the previous run's
    weights."""
    with tempfile.TemporaryDirectory() as root:
        src, out = os.path.join(root, "src"), os.path.join(root, "out")
        os.makedirs(src)
        for i, name in enumerate(("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors")):
            save_file(
                {f"model.layers.{i}.mlp.gate_proj.weight": torch.randn(64, 64, dtype=torch.bfloat16)},
                os.path.join(src, name),
                metadata={"format": "pt"},
            )
        with open(os.path.join(src, "config.json"), "w") as f:
            json.dump({"model_type": "test"}, f)
        _plant_previous_run(out, single=True)

        lowp_export.quantize_checkpoint(src, out, "mxfp8")

        assert _weight_files(out) == {"model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"}
        with open(os.path.join(out, "model.safetensors.index.json")) as f:
            written = json.load(f)["weight_map"]
        assert set(written.values()) == _weight_files(out), (
            "the surviving index does not describe the surviving shards"
        )


def test_keep_window_refuses_two_interleaved_block_numberings():
    """A second stack shares the ``layers.N`` index space, so the window holds back whichever block it hits; nothing
    at export time can resolve a prefix."""
    with tempfile.TemporaryDirectory() as root:
        src, out = os.path.join(root, "src"), os.path.join(root, "out")
        os.makedirs(src)
        save_file(
            {
                "model.layers.0.mlp.gate_proj.weight": torch.randn(64, 64, dtype=torch.bfloat16),
                "model.layers.1.mlp.gate_proj.weight": torch.randn(64, 64, dtype=torch.bfloat16),
                "model.mtp.layers.0.mlp.gate_proj.weight": torch.randn(64, 64, dtype=torch.bfloat16),
            },
            os.path.join(src, "model.safetensors"),
            metadata={"format": "pt"},
        )
        with open(os.path.join(src, "config.json"), "w") as f:
            json.dump({"model_type": "test"}, f)

        with pytest.raises(ValueError, match="span 2"):
            lowp_export.quantize_checkpoint(src, out, "mxfp8", keep_last_blocks=1)

        # The guard must not become a blanket refusal of multi-stack checkpoints.
        lowp_export.quantize_checkpoint(src, out, "mxfp8")
        assert "model.mtp.layers.0.mlp.gate_proj.weight_packed" in load_file(os.path.join(out, "model.safetensors"))


def test_an_unreadable_config_is_refused_before_any_write(tmp_path):
    """Read as no model_type, it skips the MoE expert gate and fails at the manifest stamp with a half-written
    output directory."""
    src, out = tmp_path / "src", tmp_path / "out"
    src.mkdir()
    save_file(
        {
            "model.layers.0.mlp.gate_proj.weight": torch.randn(64, 64, dtype=torch.bfloat16),
            "model.layers.0.mlp.experts.w13_weight": torch.randn(4, 128, 64, dtype=torch.bfloat16),
        },
        os.path.join(src, "model.safetensors"),
        metadata={"format": "pt"},
    )
    (src / "config.json").write_text('{"model_type": "qwen3_moe"')

    with pytest.raises(json.JSONDecodeError):
        lowp_export.quantize_checkpoint(str(src), str(out), "mxfp8")
    assert not out.exists(), "an unreadable config must be refused before the output directory is created"


def test_a_dense_weight_blocks_on_the_axis_the_qat_forward_quantizes(tmp_path):
    """Blocked on another axis the dequantized weight reproduces no QAT forward, so the axis takes no flag."""
    src, out = tmp_path / "src", tmp_path / "out"
    src.mkdir()
    gate = _make_checkpoint(str(src))
    tool = load_script_module("scripts/after_training/quantize_to_lowp.py", "quantize_to_lowp_flags")
    argv = ["quantize_to_lowp.py", "--input_dir", str(src), "--output_dir", str(out), "--format", "mxfp8"]
    with mock.patch.object(sys, "argv", [*argv, "--contraction_axis", "0"]), pytest.raises(SystemExit):
        tool.main()
    assert not out.exists()

    with mock.patch.object(sys, "argv", argv):
        tool.main()
    with open(out / "quantization_config.json") as f:
        manifest = json.load(f)
    assert manifest["contraction_axis"] == -1
    assert manifest["weight_axes"]["model.layers.0.mlp.gate_proj.weight"] == -1
    saved = load_file(str(out / "model.safetensors"))
    expected = fake_quant(gate.float(), "mxfp8", axis=-1)
    q = BlockScaledTensor(
        saved["model.layers.0.mlp.gate_proj.weight_packed"],
        saved["model.layers.0.mlp.gate_proj.weight_scale"],
        32,
        -1,
        packed=False,
        pow2_scale=True,
    )
    assert torch.equal(dequantize(q).float(), expected.float())


def test_quantize_refuses_to_write_into_its_own_input_dir():
    """The output's stale-file sweep deletes the shards this run did not write."""
    with tempfile.TemporaryDirectory() as src:
        save_file({"model.layers.0.mlp.down_proj.weight": torch.randn(64, 64)}, os.path.join(src, "model.safetensors"))
        with pytest.raises(ValueError, match="not in-place"):
            lowp_export.quantize_checkpoint(src, src, "mxfp8")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
