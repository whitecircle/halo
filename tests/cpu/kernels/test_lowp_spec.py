"""The low-precision vocabulary stays one closed set, and ``LowpSpec.validate`` refuses what cannot apply."""

from typing import get_args

import pytest

from halo.checkpoint.tools.lowp_export import _VERIFY_RELERR_TOL
from halo.hardware import GPU_PEAK_FLOPS
from halo.kernels.lowp.deepgemm import DEEPGEMM_FORMATS
from halo.kernels.lowp.quantization import _FUSED_FAKE_QUANT, _POW2_SCALE, FORMAT_BLOCK_SIZE, QUANTIZERS
from halo.kernels.lowp.spec import LOWP_RECIPES, LowpPrecision, LowpSpec, QuantFormat


def test_every_enabled_mode_has_a_recipe():
    """A mode without a recipe falls back to the parameter dtype's peak and no quantization format."""
    assert set(LOWP_RECIPES) == set(get_args(LowpPrecision)) - {"bf16"}


def test_recipes_cover_every_quant_format_once():
    formats = [recipe.quant_format for recipe in LOWP_RECIPES.values()]
    assert sorted(formats) == sorted(get_args(QuantFormat)), "each format is trained by exactly one mode"


def test_the_kernel_tables_are_keyed_on_the_quant_formats():
    assert set(FORMAT_BLOCK_SIZE) == set(get_args(QuantFormat))
    assert set(QUANTIZERS) == set(get_args(QuantFormat))
    assert set(_POW2_SCALE) == set(get_args(QuantFormat))
    assert set(_VERIFY_RELERR_TOL) == set(get_args(QuantFormat))


def test_the_fused_fake_quant_table_covers_the_pow2_scale_formats():
    """A power-of-two block scale is what lets one kernel pass reproduce eager bit for bit, so each such format has
    a fused round trip and no other format does."""
    assert set(_FUSED_FAKE_QUANT) == {fmt for fmt, pow2 in _POW2_SCALE.items() if pow2}
    assert set(DEEPGEMM_FORMATS) <= set(get_args(QuantFormat))


def test_every_peak_flops_key_is_a_peak_table_precision():
    """A key no GPU entry spells leaves MFU unscored for every native run."""
    spelled = {precision for peak in GPU_PEAK_FLOPS.values() for precision in peak.flops}
    assert {recipe.peak_flops_key for recipe in LOWP_RECIPES.values()} <= spelled


@pytest.mark.parametrize(
    ("spec", "message"),
    [
        (LowpSpec(precision="int4"), "lowp_precision must be one of 'bf16', 'fp8', 'fp4', 'mxfp4'"),
        (LowpSpec(keep_last_blocks=-1), "lowp_keep_first_blocks / lowp_keep_last_blocks must be >= 0"),
        (LowpSpec(precision="fp8", apply_dense_mlp=False, apply_moe_experts=False), "would apply to nothing"),
    ],
)
def test_validate_refuses(spec, message):
    with pytest.raises(ValueError, match=message):
        spec.validate()


def test_validate_accepts_an_off_spec_with_shape_fields_set():
    """A warning at the config, not a refusal."""
    LowpSpec(apply_dense_mlp=False, apply_moe_experts=False).validate()


def test_non_default_shape_fields_name_the_yaml_keys():
    assert LowpSpec(precision="fp8").non_default_shape_fields() == []
    assert LowpSpec(apply_moe_experts=False, keep_first_blocks=2).non_default_shape_fields() == [
        "lowp_apply_moe_experts",
        "lowp_keep_first_blocks",
    ]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
