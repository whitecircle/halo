#!/usr/bin/env python
"""A frozen reference or teacher is built with the attention implementation its policy is, picked by one seam.
On plain ``sdpa`` it would score the objective's other half through a different attention than its policy."""

import ast
import os
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from transformers import Gemma4TextConfig

from halo.model_loading import model_preparation
from halo.model_loading.parallel import frozen_models, whole_model
from halo.model_loading.parallel import loader as model_loading
from halo.model_loading.parallel.model_source import ModelSource
from halo.models.attention.sdpa_flex_sliding import SDPA_FLEX_SLIDING
from tests.common.models import TINY_GEMMA4_WIDE_HEAD_CONFIG
from tests.common.parallelism import make_parallelism_config
from tests.common.repo import REPO_ROOT

VARIANT_RESOLVER = "resolve_flex_sliding_attn_implementation"
SEAM_MODULE = "halo/model_loading/model_preparation.py"
AGREED_COMMIT = "0123456789abcdef0123456789abcdef01234567"


def _wide_gemma4_config() -> Gemma4TextConfig:
    return Gemma4TextConfig(**TINY_GEMMA4_WIDE_HEAD_CONFIG, layer_types=["sliding_attention", "full_attention"])


@pytest.mark.parametrize(("opt_out", "built"), [(False, SDPA_FLEX_SLIDING), (True, "sdpa")])
def test_the_seam_pins_the_backend_and_returns_the_build_implementation(opt_out, built):
    """``HALO_FLEX_SLIDING=0`` keeps the pin and builds plain ``sdpa``."""
    pins = []
    env = {"HALO_FLEX_SLIDING": "0"} if opt_out else {}
    with (
        patch.dict(os.environ, env),
        patch.object(torch.cuda, "is_available", lambda: True),
        patch.object(model_preparation, "patch_sdpa_for_wide_heads", lambda: pins.append(1)),
    ):
        assert model_preparation.apply_family_patches(_wide_gemma4_config(), "sdpa") == built
    assert pins == [1], "the wide-head SDPA pin must apply whatever the model is built with"


def _recording_seam(seen):
    """Returns a variant of the backend, or None for none, as the seam does."""

    def seam(model_config, attn_implementation):
        seen["patched"] = attn_implementation
        return None if attn_implementation is None else f"{attn_implementation}_variant"

    return seam


class _Finalized(Exception):
    """Stops the policy loader once it has finalized the model."""


class _Built(Exception):
    """Stops the trainer load at its build call."""


def _load_policy(config, seam, seen) -> None:
    def dispatch(*args, **kwargs):
        seen["build"] = kwargs["attn_implementation"]
        return MagicMock()

    def finalize(model, model_config, **kwargs):
        seen["finalize"] = kwargs["attn_implementation"]
        raise _Finalized

    with (
        patch.object(model_loading, "configure_float32_matmul_precision"),
        patch.object(model_loading, "apply_remote_code_compat_shims"),
        patch.object(
            model_loading, "resolve_model_source", lambda path, revision, **kwargs: ModelSource(path, revision, path)
        ),
        patch.object(model_loading, "AutoConfig", SimpleNamespace(from_pretrained=lambda *a, **k: config)),
        patch.object(model_loading, "resolve_attn_implementation", lambda *a, **k: "sdpa"),
        patch.object(model_loading, "apply_family_patches", seam),
        patch.object(model_loading, "AutoTokenizer", SimpleNamespace(from_pretrained=lambda *a, **k: MagicMock())),
        patch.object(model_loading, "_dispatch_model_loading", dispatch),
        patch.object(model_loading, "apply_run_model_policies", finalize),
        pytest.raises(_Finalized),
    ):
        model_loading.load_distributed_model(
            "org/policy",
            parallelism_config=make_parallelism_config(world_size=1, gpus_per_node=1),
            trust_remote_code=True,
        )


def _load_frozen(config, seam, seen) -> None:
    with (
        patch.object(frozen_models, "resolve_build_config", lambda *a, **k: config),
        patch.object(frozen_models, "resolve_attn_implementation", lambda *a, **k: "sdpa"),
        patch.object(frozen_models, "apply_family_patches", seam),
        patch.object(frozen_models, "auto_load_model", lambda *a, **k: seen.update(build=k["attn_implementation"])),
        patch.object(frozen_models, "cast_parameters_to_run_dtype"),
        patch.object(frozen_models, "finalize_loaded_model"),
        patch.object(
            frozen_models, "apply_run_model_policies", lambda *a, **k: seen.update(finalize=k["attn_implementation"])
        ),
        patch.object(frozen_models, "warm_attention_kernels"),
    ):
        frozen_models.load_frozen_auxiliary_model(
            "org/reference", dtype=torch.bfloat16, trust_remote_code=False, download_tag=None
        )


@pytest.mark.parametrize("load", [_load_policy, _load_frozen], ids=["policy", "frozen"])
def test_each_loader_builds_with_what_the_seam_returns(load):
    seen = {}
    load(_wide_gemma4_config(), _recording_seam(seen), seen)
    assert seen == {"patched": "sdpa", "build": "sdpa_variant", "finalize": "sdpa_variant"}


def _load_trainer_path(model_init_kwargs: dict, seen: dict) -> None:
    def build(*args, **kwargs):
        seen["build"] = kwargs.get("attn_implementation")
        raise _Built

    args = SimpleNamespace(model_init_kwargs=model_init_kwargs, gradient_checkpointing=False)
    config = _wide_gemma4_config()
    with (
        patch.object(
            whole_model, "resolve_model_source", lambda path, revision, **k: ModelSource(path, AGREED_COMMIT, path)
        ),
        patch.object(
            whole_model,
            "resolve_build_config",
            lambda *a, **k: seen.update(config_revision=k.get("revision")) or config,
        ),
        patch.object(whole_model, "apply_family_patches", _recording_seam(seen)),
        patch.object(whole_model, "auto_load_model", build),
        pytest.raises(_Built),
    ):
        whole_model.resolve_trainer_model("org/policy", args, parallelism_config=None)


def test_the_trainer_path_load_builds_with_what_the_seam_returns():
    """SMPO, offline GRPO and a teacher-distillation student take this path; the caller's kwargs keep their pin."""
    seen = {}
    model_init_kwargs = {"attn_implementation": "sdpa"}
    _load_trainer_path(model_init_kwargs, seen)
    assert seen == {"config_revision": AGREED_COMMIT, "patched": "sdpa", "build": "sdpa_variant"}
    assert model_init_kwargs["attn_implementation"] == "sdpa"


def test_an_unpinned_trainer_path_load_still_takes_the_family_patches():
    """The seam still runs for the patches no backend keys (Zaya's load recording and checkpointing refusal)."""
    seen = {}
    _load_trainer_path({}, seen)
    assert seen == {"config_revision": AGREED_COMMIT, "patched": None, "build": None}


def test_no_loader_resolves_the_variant_itself():
    """A loader binding the resolver again holds a second copy of the step, free to drift from the seam."""
    binders = set()
    for path in (*REPO_ROOT.glob("halo/**/*.py"), *REPO_ROOT.glob("scripts/**/*.py")):
        for node in ast.walk(ast.parse(path.read_text(), filename=str(path))):
            names = [alias.name for alias in node.names] if isinstance(node, ast.ImportFrom) else []
            names += [getattr(node, "id", None), getattr(node, "attr", None)]
            if VARIANT_RESOLVER in names:
                binders.add(path.relative_to(REPO_ROOT).as_posix())
    assert binders == {SEAM_MODULE}, sorted(binders)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
