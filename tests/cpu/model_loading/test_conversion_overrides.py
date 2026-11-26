#!/usr/bin/env python
"""A family's corrections to transformers' conversion mapping: registered at import on every path that reads the
mapping, refused the moment transformers' entries stop being what they were written against, and a native model refused
once a remote modeling file replaces them for the process."""

import subprocess
import sys

import pytest
import torch
from transformers import (
    AutoModelForImageTextToText,
    NemotronH_Omni_Reasoning_V3,
    NemotronHForCausalLM,
)
from transformers.conversion_mapping import get_checkpoint_conversion_mapping, register_checkpoint_conversion_mapping
from transformers.core_model_loading import WeightRenaming

from halo.model_loading.checkpoint_coverage import from_pretrained_verified
from halo.model_loading.checkpoint_keys import build_key_plan, resolve_model_conversions
from halo.model_loading.lazy_safetensors.conversion_selection import family_key_mapping
from halo.models.upstream import conversion_overrides
from halo.models.upstream.conversion_overrides import ConversionOverride, register_conversion_overrides
from tests.common.conversion_mapping import restored_conversion_mapping
from tests.common.tiny_models import TINY_MOE_VLM_FAMILIES, tiny_family_model

_COMPOSITE_CLASS = "NemotronH_Omni_Reasoning_V3"
_TEST_KEY = "halo_test_family"
_TEST_UPSTREAM = (
    (("alpha.",), ("beta.",)),
    (("gamma.",), ("delta.",)),
)
_NEMOTRON_FAMILY = TINY_MOE_VLM_FAMILIES["nemotron_h_omni"]


def _appending(entries: list) -> list:
    return [*entries, WeightRenaming(source_patterns="epsilon.", target_patterns="zeta.")]


def _as_the_remote_file_registers() -> None:
    """The release's modeling file, from its ``__init__``, for the process."""
    register_checkpoint_conversion_mapping(
        _COMPOSITE_CLASS,
        [WeightRenaming(r"^mlp1\.0\.", r"vision_projector\.mlp1\.norm\.")],
        overwrite=True,
    )


def _remote_composite_class() -> type:
    """The release's class as a remote load builds it: transformers' name, outside transformers' package."""
    return type(_COMPOSITE_CLASS, (NemotronH_Omni_Reasoning_V3,), {"__module__": "transformers_modules.release"})


@pytest.fixture
def restored_mapping(monkeypatch):
    monkeypatch.setattr(conversion_overrides, "_natively_read_keys", set())
    with restored_conversion_mapping():
        register_checkpoint_conversion_mapping(
            _TEST_KEY,
            [
                WeightRenaming(source_patterns="alpha.", target_patterns="beta."),
                WeightRenaming(source_patterns="gamma.", target_patterns="delta."),
            ],
            overwrite=True,
        )
        yield


def test_an_override_is_what_the_key_plan_then_reads(restored_mapping):
    register_conversion_overrides([ConversionOverride(_TEST_KEY, _TEST_UPSTREAM, _appending)])
    plan = build_key_plan(["epsilon.x"], ["zeta.x"], get_checkpoint_conversion_mapping(_TEST_KEY), None)
    assert plan.renamed_by_key == {"epsilon.x": "zeta.x"}


def test_the_shipped_override_is_what_the_text_class_plans_with():
    """A composite checkpoint's keys reach the text class, through the entries upstream ships and the dropped level."""
    plan = build_key_plan(
        ["language_model.backbone.norm_f.weight"],
        ["model.norm_f.weight"],
        get_checkpoint_conversion_mapping("nemotron_h"),
        "model",
    )
    assert plan.renamed_by_key == {"language_model.backbone.norm_f.weight": "model.norm_f.weight"}


def test_without_the_override_the_plan_misses_the_tower():
    upstream = get_checkpoint_conversion_mapping("nemotron_h")[1:]
    plan = build_key_plan(["language_model.backbone.norm_f.weight"], ["model.norm_f.weight"], upstream, "model")
    assert "model.norm_f.weight" not in plan.renamed_by_key.values()


def test_the_lazy_plan_reads_the_override():
    """A lazy load plans through the same mapping, so a text class meets a composite checkpoint's keys there too."""
    text_config = tiny_family_model(_NEMOTRON_FAMILY).config.text_config
    with torch.device("meta"):
        model = NemotronHForCausalLM(text_config)
    keys = [
        "language_model.backbone.norm_f.weight",
        "language_model.backbone.embeddings.weight",
        "language_model.lm_head.weight",
    ]
    renames, _ = family_key_mapping(model, keys)
    assert set(renames.values()) == {"model.norm_f.weight", "model.embeddings.weight", "lm_head.weight"}, renames


def test_a_key_transformers_ships_nothing_for_takes_new_entries(restored_mapping):
    rename = WeightRenaming(source_patterns=r"vision_projector\.", target_patterns="")
    register_conversion_overrides([ConversionOverride("nemotron_h_omni", (), lambda entries: [*entries, rename])])
    assert [entry.source_patterns for entry in get_checkpoint_conversion_mapping("nemotron_h_omni")] == [
        [r"vision_projector\."]
    ]


@pytest.mark.parametrize(
    ("key", "upstream"),
    [
        (_TEST_KEY, _TEST_UPSTREAM[:-1]),
        (_TEST_KEY, ()),
        ("nemotron_h_omni", _TEST_UPSTREAM),
    ],
    ids=["entries_moved", "transformers_began_shipping_entries", "transformers_dropped_its_entries"],
)
def test_entries_that_moved_refuse_the_override_and_register_nothing(restored_mapping, key, upstream):
    before = get_checkpoint_conversion_mapping(key)
    with pytest.raises(RuntimeError, match="re-derive"):
        register_conversion_overrides([ConversionOverride(key, upstream, _appending)])
    after = get_checkpoint_conversion_mapping(key)
    assert repr(after) == repr(before), "a refused override must leave transformers' entries in place"


@pytest.mark.parametrize(
    "module",
    [
        "halo.model_loading.checkpoint_keys",
        # The load funnel, which scripts reach through ``model_preparation`` with no key plan in their import graph.
        "halo.model_loading.checkpoint_coverage",
    ],
)
def test_every_path_reading_the_mapping_imports_the_overrides(module):
    """Fresh interpreter, as this one has already imported the leaf."""
    probe = f"import sys, {module}; sys.exit(0 if 'halo.models.upstream.conversion_overrides' in sys.modules else 3)"
    assert subprocess.run([sys.executable, "-c", probe], capture_output=True, check=False).returncode == 0


def test_the_lazy_read_refuses_a_native_composite_after_a_remote_file(restored_mapping):
    torch.manual_seed(0)
    model = tiny_family_model(_NEMOTRON_FAMILY)
    assert resolve_model_conversions(model, add_legacy=False), "an intact mapping reads as before"
    _as_the_remote_file_registers()
    with pytest.raises(RuntimeError, match=f"{_COMPOSITE_CLASS}.*one way per process"):
        resolve_model_conversions(model, add_legacy=False)


def test_the_eager_load_names_the_mixed_process_not_the_checkpoint(restored_mapping, tmp_path):
    """The coverage gate would otherwise blame a truncated checkpoint for the keys the replaced entries misplace."""
    torch.manual_seed(0)
    tiny_family_model(_NEMOTRON_FAMILY).save_pretrained(tmp_path)
    _as_the_remote_file_registers()
    with pytest.raises(RuntimeError, match="one way per process"):
        from_pretrained_verified(AutoModelForImageTextToText, str(tmp_path), dtype=torch.float32)


def test_a_remote_load_after_a_native_one_is_refused_where_it_lands(restored_mapping, tmp_path):
    """A teacher loaded remote beside a native policy, which would only fail at the policy's first save otherwise."""
    torch.manual_seed(0)
    tiny_family_model(_NEMOTRON_FAMILY).save_pretrained(tmp_path)
    from_pretrained_verified(AutoModelForImageTextToText, str(tmp_path), dtype=torch.float32)

    def remote_load(path, **kwargs):
        _as_the_remote_file_registers()
        return _remote_composite_class().from_pretrained(path, **kwargs)

    class _RemoteAuto:
        from_pretrained = staticmethod(remote_load)

    with pytest.raises(RuntimeError, match="one way per process"):
        from_pretrained_verified(_RemoteAuto, str(tmp_path), dtype=torch.float32)


def test_remote_loads_alone_keep_their_own_entries(restored_mapping):
    """The remote class reads what its file registered; only a native reader is refused."""
    _as_the_remote_file_registers()
    torch.manual_seed(0)
    native = tiny_family_model(_NEMOTRON_FAMILY)
    native.__class__ = _remote_composite_class()
    assert native.is_custom_code()
    conversions = resolve_model_conversions(native, add_legacy=False)
    assert [entry.target_patterns for entry in conversions][0] == ["vision_projector.mlp1.norm."]


def test_a_text_class_is_untouched_by_the_composites_remote_entries(restored_mapping):
    """The remote file replaces the composite's key alone, so a text-only load beside it still reads its own."""
    _as_the_remote_file_registers()
    torch.manual_seed(0)
    composite = tiny_family_model(_NEMOTRON_FAMILY)
    with torch.device("meta"):
        model = NemotronHForCausalLM(composite.config.text_config)
    assert resolve_model_conversions(model, add_legacy=False)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
