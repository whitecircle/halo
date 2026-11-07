#!/usr/bin/env python
"""A native MXFP4 gpt-oss checkpoint (``openai/gpt-oss-*``) loads lazily, its experts decoded to bf16 while reading:
- the decode is bit-identical to transformers' ``convert_moe_packed_tensors`` for every code and scale,
- the lazy load equals ``from_pretrained(..., quantization_config=Mxfp4Config(dequantize=True))`` at ep1, ep2 and with
  fp32 experts, and consumes every ``_blocks``/``_scales`` key,
- no MXFP4 quantizer is built, and the loaded model's config and its save no longer claim MXFP4, so a stock
  ``from_pretrained`` reads the save back as bf16,
- any other checkpoint-native quantization, and a checkpoint missing a ``_scales`` key, are still refused.
Tiny random checkpoints on CPU, EP patching stubbed."""

import json
import shutil

import pytest
import torch
from safetensors.torch import load_file, save_file
from transformers import AutoConfig, GptOssConfig, GptOssForCausalLM, Mxfp4Config
from transformers.integrations.mxfp4 import convert_moe_packed_tensors
from transformers.quantizers import quantizer_mxfp4

import halo.model_loading.parallel.ep as ep_lazy_loader
import halo.model_loading.parallel.lazy_plan as lazy_plan
import halo.model_loading.parallel.pp as pp_loader
from halo.distributed.expert_parallel.config import EPConfig
from halo.distributed.pipeline_parallel.stage import PP_STAGE_PARTITION_ATTR
from halo.model_loading.lazy_safetensors.conversion import MXFP4_PROJECTIONS, Mxfp4Decode
from halo.model_loading.parallel.ep import load_ep_model_lazy
from halo.model_loading.parallel.lazy_plan import LAZY_QUANTIZED, lazy_load_blocker
from halo.models.moe.family_specs import expert_weight_roots
from tests.common.models import TINY_GPTOSS_CONFIG
from tests.common.native_mxfp4 import MXFP4_QUANTIZATION_CONFIG, write_native_mxfp4

RUN_DTYPE = torch.bfloat16


def _same_bits(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.shape == b.shape and a.dtype == b.dtype and torch.equal(a.view(torch.int16), b.view(torch.int16))


def _decode(blocks: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    return Mxfp4Decode(("x_scales",)).decode(blocks, scales)


# ---------------------------------------------------------------------------------------------------------- decode


def test_decode_matches_transformers_on_random_blocks():
    g = torch.Generator().manual_seed(0)
    blocks = torch.randint(0, 256, (3, 40, 6, 16), dtype=torch.uint8, generator=g)
    scales = torch.randint(100, 150, (3, 40, 6), dtype=torch.uint8, generator=g)
    assert _same_bits(_decode(blocks, scales), convert_moe_packed_tensors(blocks, scales))


def test_decode_matches_transformers_on_every_code_and_scale():
    """All 256 bytes (every nibble pair) under all 256 e8m0 codes, both extremes included: code 0 gives subnormal
    values in bf16 and code 255 overflows to inf, the same way in both."""
    blocks = torch.arange(256, dtype=torch.uint8).reshape(16, 16).repeat(256, 1, 1)[None]  # [1, 256, 16, 16]
    scales = torch.arange(256, dtype=torch.uint8)[None, :, None].expand(1, 256, 16).contiguous()
    reference = convert_moe_packed_tensors(blocks, scales)
    assert _same_bits(_decode(blocks, scales), reference)
    assert reference.isinf().any() and (reference.abs() < 2.0**-126).logical_and(reference != 0).any()


def test_decode_test_inputs_catch_a_wrong_bias_and_swapped_nibbles():
    """Negative controls: a decoder off by one in the scale bias, or reading the high nibble first, fails the check
    above on its inputs."""
    blocks = torch.arange(256, dtype=torch.uint8).reshape(16, 16).repeat(254, 1, 1)[None]
    scales = torch.arange(1, 255, dtype=torch.uint8)[None, :, None].expand(1, 254, 16).contiguous()
    reference = convert_moe_packed_tensors(blocks, scales)
    swapped = ((blocks & 0x0F) << 4) | (blocks >> 4)
    assert not _same_bits(_decode(swapped, scales), reference)
    for off_by_one in (scales - 1, scales + 1):
        assert not _same_bits(_decode(blocks, off_by_one), reference)


def test_decode_refuses_mismatched_shapes():
    with pytest.raises(ValueError, match="uint8 blocks"):
        _decode(torch.zeros(2, 4, 3, 16, dtype=torch.uint8), torch.zeros(2, 4, 2, dtype=torch.uint8))


# ------------------------------------------------------------------------------------------------------ checkpoint


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory) -> str:
    torch.manual_seed(0)
    path = tmp_path_factory.mktemp("gpt_oss_native_mxfp4")
    GptOssForCausalLM(GptOssConfig(**TINY_GPTOSS_CONFIG)).to(RUN_DTYPE).save_pretrained(path)
    write_native_mxfp4(path)
    return str(path)


@pytest.fixture(scope="module")
def reference(checkpoint) -> dict[str, torch.Tensor]:
    """What transformers loads from the same checkpoint when asked to dequantize."""
    model = GptOssForCausalLM.from_pretrained(
        checkpoint, dtype=RUN_DTYPE, quantization_config=Mxfp4Config(dequantize=True)
    )
    return {key: value.detach().clone() for key, value in model.state_dict().items()}


@pytest.fixture(autouse=True)
def _stub_ep_patching(monkeypatch):
    """EP patching needs DeepEP and process groups; the load itself does not."""
    monkeypatch.setattr(ep_lazy_loader, "apply_ep_layers", lambda model, *a, **k: model)
    monkeypatch.setattr(pp_loader, "apply_ep_layers", lambda model, *a, **k: model)


def _ep_config(ep_size: int, fp32_experts: bool = False) -> EPConfig:
    return EPConfig(
        ep_size=ep_size,
        stage_world_size=ep_size,
        nvlink_domain_size=ep_size,
        experts_fsdp_managed=ep_size == 1,
        fp32_experts=fp32_experts,
    )


def _lazy(path: str, ep_size: int = 1, *, fp32_experts: bool = False, config=None):
    return load_ep_model_lazy(
        path,
        _ep_config(ep_size, fp32_experts),
        config if config is not None else AutoConfig.from_pretrained(path),
        dtype=RUN_DTYPE,
        trust_remote_code=False,
        model_class=GptOssForCausalLM,
    )


def _is_expert_key(key: str) -> bool:
    """An expert-bank tensor, sliced per rank: the decoded projections and their biases."""
    return key.rpartition(".")[2] in expert_weight_roots()


_ALL = slice(None)


def _mismatches(lazy_sd: dict, reference: dict, expert_slice: slice = _ALL) -> list[str]:
    bad = []
    for key, lazy in lazy_sd.items():
        expected = reference[key][expert_slice] if _is_expert_key(key) else reference[key]
        if lazy.is_floating_point() and lazy.dtype != expected.dtype:
            # The lazy loader casts parameters to the run dtype and keeps fp32 masters; compare in the wider one.
            lazy, expected = lazy.float(), expected.float()
        if lazy.shape != expected.shape or not torch.equal(lazy, expected):
            bad.append(key)
    return bad


def test_only_native_mxfp4_gpt_oss_passes_the_lazy_gate(checkpoint, tmp_path):
    assert lazy_load_blocker(checkpoint) is None
    for model_type, quantization in [
        ("gpt_oss", {"quant_method": "fp8", "weight_block_size": [128, 128]}),
        ("gpt_oss", {"quant_method": "bitsandbytes"}),
        ("qwen3_moe", MXFP4_QUANTIZATION_CONFIG),
    ]:
        other = tmp_path / f"{model_type}_{quantization['quant_method']}"
        other.mkdir()
        (other / "config.json").write_text(json.dumps({"model_type": model_type, "quantization_config": quantization}))
        assert lazy_load_blocker(str(other)) == LAZY_QUANTIZED, (model_type, quantization)


def test_ep1_lazy_load_equals_dequantized_from_pretrained(checkpoint, reference):
    model = _lazy(checkpoint)
    lazy_sd = model.state_dict()
    assert set(lazy_sd) == set(reference)
    assert not _mismatches(lazy_sd, reference)
    assert {lazy_sd[k].dtype for k in lazy_sd if k.rpartition(".")[2] in MXFP4_PROJECTIONS} == {RUN_DTYPE}
    assert not any(t.is_meta for t in lazy_sd.values())


def test_ep1_load_consumes_every_blocks_and_scales_key(checkpoint, caplog):
    with caplog.at_level("WARNING", logger=lazy_plan.logger.name):
        _lazy(checkpoint)
    unexpected = [r.getMessage() for r in caplog.records if "align to no model tensor" in r.getMessage()]
    assert not unexpected, unexpected


def test_ep2_rank0_decodes_its_expert_half(checkpoint, reference):
    model = _lazy(checkpoint, ep_size=2)
    lazy_sd = model.state_dict()
    half = TINY_GPTOSS_CONFIG["num_local_experts"] // 2
    experts = [k for k in lazy_sd if _is_expert_key(k)]
    assert experts and all(lazy_sd[k].shape[0] == half for k in experts)
    assert not _mismatches(lazy_sd, reference, expert_slice=slice(0, half))


def test_fp32_experts_hold_the_exact_decoded_values(checkpoint, reference):
    """At ep2, as FSDP-managed ep1 experts ignore fp32_experts. Every decoded value is exact in bf16, so the fp32
    master equals the bf16 reference."""
    lazy_sd = _lazy(checkpoint, ep_size=2, fp32_experts=True).state_dict()
    decoded = [k for k in lazy_sd if k.rpartition(".")[2] in MXFP4_PROJECTIONS]
    assert decoded and {lazy_sd[k].dtype for k in decoded} == {torch.float32}
    half = TINY_GPTOSS_CONFIG["num_local_experts"] // 2
    assert not _mismatches(lazy_sd, reference, expert_slice=slice(0, half))


@pytest.mark.parametrize("ep_size", [1, 2], ids=["pp2", "pp2_x_ep2"])
@pytest.mark.parametrize("pp_rank", range(2))
def test_pipeline_stage_decodes_the_layers_it_claims(checkpoint, reference, pp_rank, ep_size):
    """PP takes the same prologue and plans, filtered to the stage: each stage holds its own layers, decoded, and
    under pp2 x ep2 the first expert half of them."""
    stage = pp_loader._load_stage_lazy(
        checkpoint,
        pp_rank,
        2,
        config=AutoConfig.from_pretrained(checkpoint),
        ep_config=_ep_config(ep_size) if ep_size > 1 else None,
        dtype=RUN_DTYPE,
        trust_remote_code=False,
        model_class=GptOssForCausalLM,
    )
    lo, hi = getattr(stage, PP_STAGE_PARTITION_ATTR)[pp_rank]
    experts = slice(0, TINY_GPTOSS_CONFIG["num_local_experts"] // ep_size)
    state = stage.state_dict()
    layer_keys = [k for k in state if k.startswith("model.layers.")]
    assert len(stage.model.layers) == hi - lo
    assert any(k.rpartition(".")[2] in MXFP4_PROJECTIONS for k in layer_keys)
    bad = []
    for key in layer_keys:
        local, rest = key.removeprefix("model.layers.").split(".", 1)
        expected = reference[f"model.layers.{lo + int(local)}.{rest}"]
        expected = expected[experts] if _is_expert_key(key) else expected
        if not torch.equal(state[key], expected):
            bad.append(key)
    assert not bad, bad


def test_no_mxfp4_quantizer_is_built(checkpoint, monkeypatch):
    """The shell is built from a config without ``quantization_config``, so transformers never constructs the
    quantizer whose kernels check could otherwise swap in packed uint8 experts."""

    def refuse(*_, **__):
        raise AssertionError("the lazy load built an MXFP4 quantizer")

    monkeypatch.setattr(quantizer_mxfp4.Mxfp4HfQuantizer, "__init__", refuse)
    model = _lazy(checkpoint)
    assert getattr(model, "hf_quantizer", None) is None


def test_loaded_config_and_its_save_no_longer_claim_mxfp4(checkpoint, reference, tmp_path):
    run_config = AutoConfig.from_pretrained(checkpoint)
    assert run_config.quantization_config is not None
    model = _lazy(checkpoint, config=run_config)
    # The run's config too, as later readers (tree attention's quantization guard) take it from there.
    assert getattr(run_config, "quantization_config", None) is None
    assert getattr(model.config, "quantization_config", None) is None
    model.save_pretrained(tmp_path / "saved")
    saved = json.loads((tmp_path / "saved" / "config.json").read_text())
    assert "quantization_config" not in saved
    # A stock load of the save is a plain bf16 checkpoint and reads back the decoded weights.
    reloaded = GptOssForCausalLM.from_pretrained(tmp_path / "saved", dtype=RUN_DTYPE)
    assert getattr(reloaded, "hf_quantizer", None) is None
    assert not _mismatches(reloaded.state_dict(), reference)


def test_a_missing_scales_key_is_refused_at_plan_time(checkpoint, tmp_path):
    broken = tmp_path / "missing_scales"
    shutil.copytree(checkpoint, broken)
    (weights_file,) = broken.glob("*.safetensors")
    state = load_file(weights_file)
    dropped = next(k for k in sorted(state) if k.endswith("down_proj_scales"))
    del state[dropped]
    save_file(state, weights_file, metadata={"format": "pt"})
    with pytest.raises(ValueError, match="truncated tensor"):
        _lazy(str(broken))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
