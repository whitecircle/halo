"""CPU test for scripts/after_training/quantize_to_lowp.py — the bf16/fp32 → block-scaled mxfp8/nvfp4
checkpoint conversion that pairs with quantization-aware training.

Verifies: (1) only the targeted MLP/expert weights are quantized (attention/norm/embed/bias copied
unchanged); (2) the output layout is the compressed-tensors block-scaled form (weight_packed /
weight_scale / weight_shape) + a quantization_config.json manifest; (3) the round-trip dequant matches the
original within the inherent format error; (4) QAT→inference consistency — the dequantized weight
reproduces the fake-quant (training) forward exactly. Pure-CPU (the quant math is dependency-free torch).

Run: python tests/cpu/checkpoint/test_quantize_to_lowp.py
"""

import json
import os
import re
import tempfile

import pytest
import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

from src.distributed.expert_parallel.expert_weights import hf_fused_expert_keys, per_expert_layouts
from src.kernels.lowp.mixed_precision import MLP_PROJECTIONS
from src.kernels.lowp.quantization import BlockScaledTensor, dequantize, fake_quant
from tests.common.checkpoint_io import weight_files
from tests.common.utils import load_script_module

_quantize_to_lowp = load_script_module("scripts/after_training/quantize_to_lowp.py")
_DEFAULT_EXCLUDE = _quantize_to_lowp._DEFAULT_EXCLUDE
_DEFAULT_INCLUDE = _quantize_to_lowp._DEFAULT_INCLUDE
_FUSED_EXPERT_SUFFIXES = _quantize_to_lowp._FUSED_EXPERT_SUFFIXES
_QUANTIZABLE_WEIGHT_NAMES = _quantize_to_lowp._QUANTIZABLE_WEIGHT_NAMES
_fused_expert_axis = _quantize_to_lowp._fused_expert_axis
_should_quantize = _quantize_to_lowp._should_quantize
checkpoint_shard_files = _quantize_to_lowp.checkpoint_shard_files
quantize_checkpoint = _quantize_to_lowp.quantize_checkpoint

_FMT_TOL = {"mxfp8": 0.08, "nvfp4": 0.25, "mxfp4": 0.35}  # inherent block-scaled format error


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
            quantize_checkpoint(src, out, fmt, verify=True)
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
            # nvfp4 is two-level: without the per-tensor global scale the reconstruction is off by
            # up to E4M3_MAX*E2M1_MAX, so only that format may ship one.
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
                global_scale=global_scale,
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


def test_should_quantize_scoping():
    """The --include/--exclude defaults scope quantization to dense-MLP / expert FFN
    weights only — attention, norms, embeddings, router and biases stay high precision."""
    inc, exc = re.compile(_DEFAULT_INCLUDE), re.compile(_DEFAULT_EXCLUDE)
    w2d = torch.zeros(16, 16)

    def q(name, t=w2d):
        return _should_quantize(name, t, inc, exc)

    assert q("model.layers.0.mlp.gate_proj.weight")
    assert q("model.layers.0.mlp.down_proj.weight")
    assert q("model.layers.0.mlp.experts.0.gate_proj.weight")
    # The other per-expert un-fused layout the roster declares (LFM-2 / DeepSeek-V4).
    assert q("model.layers.0.mlp.experts.0.w1.weight")
    assert q("model.layers.0.mlp.experts.0.w2.weight")
    # The roster's other expert-container spelling (GLM-4's ``routed_experts``) is an expert too.
    assert q("model.layers.0.mlp.routed_experts.3.gate_proj.weight")
    assert not _should_quantize(
        "model.layers.0.mlp.routed_experts.3.gate_proj.weight", w2d, inc, exc, apply_moe_experts=False
    )
    # A per-expert roster spelling OUTSIDE an expert segment is a dense projection the trainer never
    # converts (LFM-2's dense ``feed_forward.w1``): it trains in bf16 and must export in bf16.
    assert not q("model.layers.0.feed_forward.w1.weight")
    assert not q("model.layers.0.feed_forward.w2.weight")
    # Fused MoE experts are 3-D nn.Parameters with no trailing .weight: matched on shape + suffix.
    w3d = torch.zeros(8, 16, 16)
    assert _should_quantize("model.layers.0.mlp.experts.gate_up_proj", w3d, inc, exc)
    assert _should_quantize("model.layers.0.mlp.experts.down_proj", w3d, inc, exc)
    # ...but a 2-D tensor with a fused-expert-like name is not (no .weight, not 3-D).
    assert not q("model.layers.0.mlp.experts.gate_up_proj")

    # Excluded by --exclude even though they match --include's broad terms.
    assert not q("model.layers.0.mlp.router.weight"), "router must stay high precision"
    assert not q("model.layers.0.mlp.gate.weight"), "MoE gate must stay high precision"
    assert not q("model.embed_tokens.weight")
    assert not q("model.layers.0.input_layernorm.weight")
    assert not q("lm_head.weight")
    assert not q("model.layers.0.mlp.gate_proj.bias")

    # Not matched by --include at all (attention).
    assert not q("model.layers.0.self_attn.q_proj.weight")

    # 1-D tensors are never quantized (no contraction axis to block along).
    assert not _should_quantize("model.layers.0.mlp.gate_proj.weight", torch.zeros(16), inc, exc)
    assert not q("model.layers.0.mlp.gate_up_proj_packed")


def test_non_block_divisible_axis_left_high_precision():
    """A matched weight whose contraction axis is not divisible by the format block
    size is copied through unquantized (skip path) rather than crashing."""
    with tempfile.TemporaryDirectory() as root:
        src, out = os.path.join(root, "src"), os.path.join(root, "out")
        os.makedirs(src)
        # in_features = 100, not divisible by mxfp8 block 32.
        odd = torch.randn(64, 100, dtype=torch.bfloat16)
        save_file(
            {"model.layers.0.mlp.gate_proj.weight": odd},
            os.path.join(src, "model.safetensors"),
            metadata={"format": "pt"},
        )
        with open(os.path.join(src, "config.json"), "w") as f:
            json.dump({"model_type": "test"}, f)

        quantize_checkpoint(src, out, "mxfp8")
        saved = load_file(os.path.join(out, "model.safetensors"))
        assert "model.layers.0.mlp.gate_proj.weight" in saved
        assert "model.layers.0.mlp.gate_proj.weight_packed" not in saved
        assert torch.equal(saved["model.layers.0.mlp.gate_proj.weight"], odd)
        with open(os.path.join(out, "quantization_config.json")) as f:
            mani = json.load(f)
        assert mani["quantized_weights"] == []


def test_multi_shard_checkpoint_quantized():
    """Each .safetensors shard of a multi-file checkpoint is processed independently."""
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

        quantize_checkpoint(src, out, "mxfp8")
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
    """Quantize a single 3-D fused expert under the given model_type; return (saved tensors, manifest)."""
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
        quantize_checkpoint(src, out, "mxfp8", verify=True)
        saved = load_file(os.path.join(out, "model.safetensors"))
        with open(os.path.join(out, "quantization_config.json")) as f:
            mani = json.load(f)
        return saved, mani


def test_fused_glu_experts_quantized_on_last_axis():
    """Non-gpt-oss MoE families (Qwen3.5/3.6, GLM4, LFM2, Mistral4, Gemma4) store 3-D experts in
    nn.Linear convention ``[E, N, K]`` — contraction is the LAST axis. The shape ``[E, N=100, K=64]``
    is divisible by the mxfp8 block (32) ONLY on the last axis, so it quantizes iff the last axis is
    chosen. Regression guard: hardcoding axis 1 for every 3-D tensor blocks along N=100 (not
    divisible) and silently SKIPS these experts (or, when N is divisible, corrupts them by scaling
    the output dim)."""
    base = "model.layers.0.mlp.experts.gate_up_proj"
    saved, mani = _quantize_one_expert("qwen3_5_moe", (4, 100, 64))
    assert f"{base}.weight_packed" in saved, "fused-GLU expert must quantize along the last (contraction) axis"
    assert base not in saved
    assert mani["weight_axes"][base] == -1, "manifest must record the last-axis contraction for fused-GLU"
    # weight_scale blocks the last axis: [E, N, K/block] = [4, 100, 2].
    assert list(saved[f"{base}.weight_scale"].shape) == [4, 100, 2]


def test_gptoss_experts_quantized_on_axis1():
    """gpt-oss stores 3-D experts in matmul convention ``[E, K, N]`` — contraction is axis 1. The shape
    ``[E, K=64, N=100]`` is divisible by the block only on axis 1, so it quantizes iff axis 1 is chosen
    (detected from ``model_type == 'gpt_oss'``)."""
    base = "model.layers.0.mlp.experts.gate_up_proj"
    saved, mani = _quantize_one_expert("gpt_oss", (4, 64, 100))
    assert f"{base}.weight_packed" in saved, "gpt-oss expert must quantize along axis 1 (contraction)"
    assert base not in saved
    assert mani["weight_axes"][base] == 1, "manifest must record axis-1 contraction for gpt-oss"
    # weight_scale blocks axis 1: [E, K/block, N] = [4, 2, 100].
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
        ({"model_type": "gpt_oss_vlm", "text_config": {"model_type": "gpt_oss"}}, 1),  # VLM wrapper nests the LM
        ({"model_type": "qwen3_5_moe"}, -1),
        ({"model_type": "glm4_moe_lite"}, -1),
        ({"model_type": "not_a_registered_family"}, None),  # unknown -> no guess
    ],
)
def test_fused_expert_axis_comes_from_the_layer_class(tmp_path, cfg, expected_axis):
    """The fused-expert contraction axis is resolved through the EP layer class that writes the
    layout, not a model_type table in the script. A wrong axis block-scales the OUTPUT dim and
    silently corrupts every dequantized expert, so an unregistered family must resolve to ``None``
    (the caller raises) rather than defaulting."""
    assert _fused_expert_axis(_write_config(tmp_path, cfg)) == expected_axis


def test_fused_expert_axis_missing_config(tmp_path):
    assert _fused_expert_axis(str(tmp_path)) is None


def test_unknown_family_fused_expert_raises():
    """A 3-D expert whose family declares no axis must fail loud, never quantize on a guessed axis."""
    with pytest.raises(ValueError, match="matches no EP layer class"):
        _quantize_one_expert("not_a_registered_family", (4, 64, 64))


def test_an_expert_bank_under_an_undeclared_fused_name_is_refused_before_any_write(tmp_path):
    """Step-3.7's hub export stores its experts per layer as ``moe.gate_proj``/``moe.up_proj``
    ``[E, M, H]`` beside ``moe.down_proj`` ``[E, H, M]``. Only the last spelling is a class-declared
    fused key, so without the refusal the export quantizes ``down_proj`` alone and copies the other
    two through in bf16 — an artifact that does not reproduce the QAT forward. The scope knob that
    excludes experts, and an explicit ``--exclude``, keep the copy-through legitimate."""
    src, out = tmp_path / "src", tmp_path / "out"
    src.mkdir()
    prefix = "model.language_model.layers.1.moe"
    save_file(
        {
            f"{prefix}.gate_proj": torch.randn(4, 64, 64, dtype=torch.bfloat16),
            f"{prefix}.up_proj": torch.randn(4, 64, 64, dtype=torch.bfloat16),
            f"{prefix}.down_proj": torch.randn(4, 64, 64, dtype=torch.bfloat16),
        },
        os.path.join(src, "model.safetensors"),
        metadata={"format": "pt"},
    )
    with open(src / "config.json", "w") as f:
        json.dump({"model_type": "step3p7"}, f)

    with pytest.raises(ValueError, match="no EP layer class declares its name as a fused expert key"):
        quantize_checkpoint(str(src), str(out), "mxfp8")
    assert not out.exists(), "a refused conversion must not create its output directory"

    quantize_checkpoint(str(src), str(out), "mxfp8", apply_moe_experts=False)
    assert set(load_file(os.path.join(out, "model.safetensors"))) == {
        f"{prefix}.{name}" for name in ("gate_proj", "up_proj", "down_proj")
    }, "with experts out of scope every bank is copied through unchanged"


def test_a_moe_checkpoint_whose_experts_match_nothing_is_refused(tmp_path):
    """Inkling's hub spelling (``experts.w13_weight`` / ``w2_weight``) matches no roster name, so the
    export would quantize the dense MLPs, copy every expert through in bf16 and still write a
    ``quantization_config`` claiming QAT parity. A registered MoE family with zero expert matches is
    refused before any write; declaring the experts out of scope keeps the copy-through legitimate."""
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
        quantize_checkpoint(str(src), str(out), "mxfp8")
    assert not out.exists()

    quantize_checkpoint(str(src), str(out), "mxfp8", apply_moe_experts=False)
    written = load_file(os.path.join(out, "model.safetensors"))
    assert "model.layers.0.mlp.gate_proj.weight_packed" in written
    assert torch.equal(
        written["model.layers.0.mlp.experts.w2_weight"], tensors["model.layers.0.mlp.experts.w2_weight"]
    )


def test_vision_tower_projections_are_never_quantized():
    """A VLM's vision tower and projector run in bf16 during QAT, so the export must leave them there.

    ``apply_mixed_precision_compute`` converts ``MLP_PROJECTIONS`` inside the TEXT backbone only
    (``_text_backbone_prefix``): a SigLIP-style tower's ``fc1``/``fc2`` is not in that roster at all,
    and a Qwen-VL-style tower names its projections like the text MLP but sits outside the backbone.
    Quantizing either on export contradicts this file's "reproduces the QAT forward exactly" claim —
    the served vision encoder would compute in a format training never saw.
    """
    inc, exc = re.compile(_DEFAULT_INCLUDE), re.compile(_DEFAULT_EXCLUDE)
    w2d = torch.zeros(16, 16)
    for name in (
        "vision_tower.vision_model.encoder.layers.0.mlp.fc1.weight",
        "model.vision_tower.encoder.layers.3.mlp.fc2.weight",
        "visual.blocks.0.mlp.gate_proj.weight",
        "model.visual.blocks.7.mlp.down_proj.weight",
        "multi_modal_projector.linear_1.weight",
        "model.multi_modal_projector.mlp.up_proj.weight",
    ):
        assert not _should_quantize(name, w2d, inc, exc), f"{name} was quantized but trains in bf16"


def test_the_quantizable_name_set_is_derived_from_the_two_rosters():
    """The include set must BE the trainer's dense roster plus the EP classes' expert layouts.

    A pattern restated here drifts in both directions: it matches names no lowp path converts
    (``fc1``/``fc2``, a bare ``mlp``/``feed_forward``), and a new family's per-expert layout has to
    be remembered a second time to be covered at all.
    """
    declared = set(MLP_PROJECTIONS) | set(hf_fused_expert_keys()) | {k for keys in per_expert_layouts() for k in keys}
    assert set(_QUANTIZABLE_WEIGHT_NAMES) == declared
    assert {"gate_proj", "up_proj", "down_proj", "gate_up_proj", "w1", "w2", "w3"} <= declared
    for never_converted in ("fc1", "fc2", "mlp", "feed_forward", "experts"):
        assert never_converted not in _QUANTIZABLE_WEIGHT_NAMES


def test_fused_expert_suffixes_are_the_class_declared_union():
    """The fused-expert names must come from the EP layer classes' ``_HF_FUSED_EXPERT_KEYS``, not a
    hand-kept list: a hardcoded list drifts from the roster (both by missing a new family's fused
    tensor and by naming projections — w1/w2/w3, fc1/fc2 — that no family stores fused, which
    invites quantizing the wrong tensor on a guessed axis)."""
    assert set(_FUSED_EXPERT_SUFFIXES) == set(hf_fused_expert_keys())
    # Anchored against the declarations so a same-source comparison alone cannot pass vacuously.
    assert {"gate_up_proj", "down_proj"} <= set(_FUSED_EXPERT_SUFFIXES)
    for never_fused in ("w1", "w2", "w3", "fc1", "fc2"):
        assert never_fused not in _FUSED_EXPERT_SUFFIXES


@pytest.mark.parametrize("poison", [float("nan")])
@pytest.mark.parametrize("fmt", ["mxfp8", "mxfp4", "nvfp4"])
def test_verify_refuses_a_non_finite_weight(fmt, poison):
    """``--verify`` must not report a non-finite weight as a clean round-trip.

    The relative error of a poisoned weight is NaN, and ``max(x, nan)`` returns ``x``, so the running
    maximum stayed 0.0 and the tool printed "OK (inherent format error)" over an export that dequantizes
    to inf/NaN. Parametrized over the formats because each spreads the poison differently — NVFP4's
    per-tensor global scale carries it to every block, MX keeps it in one.
    """
    with tempfile.TemporaryDirectory() as root:
        src, out = os.path.join(root, "src"), os.path.join(root, "out")
        os.makedirs(src)
        weight = torch.randn(64, 128, dtype=torch.bfloat16) * 0.02
        weight[3, 7] = poison
        save_file({"model.layers.0.mlp.down_proj.weight": weight}, os.path.join(src, "model.safetensors"))
        with pytest.raises(ValueError, match="non-finite"):
            quantize_checkpoint(src, out, fmt, verify=True)


def _plant_previous_run(out_dir: str, *, single: bool, index_shards: tuple[str, ...] = ()) -> None:
    """Leave a previous quantization run's weight files in the output directory."""
    os.makedirs(out_dir, exist_ok=True)
    if single:
        save_file({"stale.weight": torch.zeros(2, 2)}, os.path.join(out_dir, "model.safetensors"))
    for shard in index_shards:
        save_file({"stale.weight": torch.zeros(2, 2)}, os.path.join(out_dir, shard))
    if index_shards:
        with open(os.path.join(out_dir, "model.safetensors.index.json"), "w") as f:
            json.dump({"metadata": {}, "weight_map": {f"stale.{i}": s for i, s in enumerate(index_shards)}}, f)


def _weight_files(out_dir: str) -> set[str]:
    return set(weight_files(out_dir))  # index excluded: these assertions pin exactly the shard set


def test_single_file_output_sweeps_a_previous_runs_index():
    """A one-file quantization writes no index — so a previous sharded run's index must not be left
    behind claiming to describe the directory. Every index-first reader in the toolkit would follow
    it to shards this run does not have, and ``from_pretrained`` would see two contradictory
    descriptions of the same checkpoint."""
    with tempfile.TemporaryDirectory() as root:
        src, out = os.path.join(root, "src"), os.path.join(root, "out")
        os.makedirs(src)
        _make_checkpoint(src)
        _plant_previous_run(out, single=False, index_shards=("model-00001-of-00002.safetensors",))

        quantize_checkpoint(src, out, "mxfp8")

        assert _weight_files(out) == {"model.safetensors"}, f"stale files survived: {sorted(_weight_files(out))}"
        assert not os.path.exists(os.path.join(out, "model.safetensors.index.json"))
        saved = load_file(os.path.join(out, "model.safetensors"))
        assert "stale.weight" not in saved and "model.layers.0.mlp.gate_proj.weight_packed" in saved
        assert os.path.isfile(os.path.join(out, "quantization_config.json"))
        assert os.path.isfile(os.path.join(out, "config.json"))


def test_sharded_output_keeps_its_own_index_and_drops_the_stale_single_file():
    """The mirror: when this run DOES write an index, the index is the keep-set's business and the
    leftover single ``model.safetensors`` is what has to go — ``from_pretrained`` prefers it over the
    index, so leaving it serves the previous run's weights."""
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

        quantize_checkpoint(src, out, "mxfp8")

        assert _weight_files(out) == {"model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"}
        with open(os.path.join(out, "model.safetensors.index.json")) as f:
            written = json.load(f)["weight_map"]
        assert set(written.values()) == _weight_files(out), (
            "the surviving index does not describe the surviving shards"
        )


def test_keep_window_refuses_two_interleaved_block_numberings():
    """A keep-first/last window is drawn over ONE ``layers.N`` numbering. A checkpoint carrying a
    second stack whose keys pass the include roster — an MTP/draft head, a vision tower spelled
    outside the exclude fence — shares that index space, so block ``N`` names two different blocks
    and the window holds back whichever one it happens to hit. Nothing at export time can resolve a
    text-backbone prefix (no live module), so this must refuse rather than guess."""
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
            quantize_checkpoint(src, out, "mxfp8", keep_last_blocks=1)

        # Without a keep window there is no numbering to be ambiguous about, so the same checkpoint
        # exports: the guard must not become a blanket refusal of multi-stack checkpoints.
        quantize_checkpoint(src, out, "mxfp8")
        assert "model.mtp.layers.0.mlp.gate_proj.weight_packed" in load_file(os.path.join(out, "model.safetensors"))


def test_quantize_refuses_to_write_into_its_own_input_dir():
    """save_sharded_state_dict-style rewrites delete the shards they do not own, so an in-place
    conversion destroys the source checkpoint."""
    with tempfile.TemporaryDirectory() as src:
        save_file({"model.layers.0.mlp.down_proj.weight": torch.randn(64, 64)}, os.path.join(src, "model.safetensors"))
        with pytest.raises(ValueError, match="not in-place"):
            quantize_checkpoint(src, src, "mxfp8")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
