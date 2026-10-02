#!/usr/bin/env python
"""A frozen reference or teacher is built with the attention implementation its policy is, picked by one seam.

``apply_family_attention_patches`` keys the family patches on the resolved backend (``sdpa`` for Gemma 4, whose
wide heads need the SDPA pin) and returns the implementation to build with (its variant, ``sdpa_flex_sliding``).
The policy loader and the frozen reference/teacher loader both build and finalize the model with that return
value. A frozen model built on plain ``sdpa`` would score every logprob of the objective's other half through a
different attention than the policy it anchors, and a loader holding its own copy of the backend-to-variant step
can drift into exactly that.

    python tests/cpu/models/test_frozen_model_attention_variant.py
"""

import ast
import os
import pathlib
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from transformers import Gemma4TextConfig

from src.distributed.loading import frozen_models, model_loading
from src.models.loading import model_preparation
from src.models.patches.flex_sliding_attention import FLEX_SLIDING
from tests.common.parallelism import make_parallelism_config

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
VARIANT_RESOLVER = "resolve_flex_sliding_attn_implementation"
SEAM_MODULE = "src/models/loading/model_preparation.py"


def _wide_gemma4_config() -> Gemma4TextConfig:
    return Gemma4TextConfig(
        vocab_size=256,
        hidden_size=128,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=256,
        global_head_dim=512,
        num_global_key_value_heads=1,
        sliding_window=32,
        layer_types=["sliding_attention", "full_attention"],
        enable_moe_block=False,
        hidden_size_per_layer_input=0,
        num_kv_shared_layers=0,
    )


@pytest.mark.parametrize(("opt_out", "built"), [(False, FLEX_SLIDING), (True, "sdpa")])
def test_the_seam_pins_the_backend_and_returns_the_build_implementation(opt_out, built):
    """A wide-head model resolved to ``sdpa`` on a CUDA host gets the SDPA pin, keyed on the backend, and is built
    with the flex-sliding variant; ``HALO_FLEX_SLIDING=0`` keeps the pin and builds plain ``sdpa``."""
    pins = []
    env = {"HALO_FLEX_SLIDING": "0"} if opt_out else {}
    with (
        patch.dict(os.environ, env),
        patch.object(torch.cuda, "is_available", lambda: True),
        patch.object(model_preparation, "patch_sdpa_for_wide_heads", lambda: pins.append(1)),
    ):
        assert model_preparation.apply_family_attention_patches(_wide_gemma4_config(), "sdpa") == built
    assert pins == [1], "the wide-head SDPA pin must apply whatever the model is built with"


class _Finalized(Exception):
    """Stops the policy loader once it has finalized the model."""


def _load_policy(config, seam, seen) -> None:
    def dispatch(name, parallelism_config, model_class, common_kwargs):
        seen["build"] = common_kwargs["attn_implementation"]
        return MagicMock()

    def finalize(model, model_config, **kwargs):
        seen["finalize"] = kwargs["attn_implementation"]
        raise _Finalized

    with (
        patch.object(model_loading, "configure_float32_matmul_precision"),
        patch.object(model_loading, "apply_remote_code_compat_shims"),
        patch.object(model_loading, "_ensure_model_downloaded"),
        patch.object(model_loading, "AutoConfig", SimpleNamespace(from_pretrained=lambda *a, **k: config)),
        patch.object(model_loading, "resolve_attn_implementation", lambda *a, **k: "sdpa"),
        patch.object(model_loading, "apply_family_attention_patches", seam),
        patch.object(model_loading, "AutoTokenizer", SimpleNamespace(from_pretrained=lambda *a, **k: MagicMock())),
        patch.object(model_loading, "_dispatch_model_loading", dispatch),
        patch.object(model_loading, "finalize_run_model", finalize),
        pytest.raises(_Finalized),
    ):
        model_loading.load_distributed_model(
            "org/policy", parallelism_config=make_parallelism_config(world_size=1, gpus_per_node=1)
        )


def _load_frozen(config, seam, seen) -> None:
    with (
        patch.object(frozen_models, "apply_remote_code_compat_shims"),
        patch.object(frozen_models, "AutoConfig", SimpleNamespace(from_pretrained=lambda *a, **k: config)),
        patch.object(frozen_models, "resolve_attn_implementation", lambda *a, **k: "sdpa"),
        patch.object(frozen_models, "apply_family_attention_patches", seam),
        patch.object(frozen_models, "auto_load_model", lambda *a, **k: seen.update(build=k["attn_implementation"])),
        patch.object(frozen_models, "cast_parameters_to_run_dtype"),
        patch.object(frozen_models, "finalize_loaded_model"),
        patch.object(
            frozen_models, "finalize_run_model", lambda *a, **k: seen.update(finalize=k["attn_implementation"])
        ),
        patch.object(frozen_models, "warm_attention_kernels"),
    ):
        frozen_models.load_frozen_auxiliary_model("org/reference", dtype=torch.bfloat16)


@pytest.mark.parametrize("load", [_load_policy, _load_frozen], ids=["policy", "frozen"])
def test_each_loader_builds_with_what_the_seam_returns(load):
    """Each loader hands the seam the resolved backend, then builds and finalizes with its return value."""
    seen = {}

    def seam(model_config, attn_implementation):
        seen["patched"] = attn_implementation
        return f"{attn_implementation}_variant"

    load(_wide_gemma4_config(), seam, seen)
    assert seen == {"patched": "sdpa", "build": "sdpa_variant", "finalize": "sdpa_variant"}


def test_no_loader_resolves_the_variant_itself():
    """Only the seam calls the variant resolver: a loader binding it again holds a second copy of the
    backend-to-variant step, free to drift from the seam."""
    binders = set()
    for path in (*REPO_ROOT.glob("src/**/*.py"), *REPO_ROOT.glob("scripts/**/*.py")):
        for node in ast.walk(ast.parse(path.read_text(), filename=str(path))):
            names = [alias.name for alias in node.names] if isinstance(node, ast.ImportFrom) else []
            names += [getattr(node, "id", None), getattr(node, "attr", None)]
            if VARIANT_RESOLVER in names:
                binders.add(path.relative_to(REPO_ROOT).as_posix())
    assert binders == {SEAM_MODULE}, sorted(binders)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
