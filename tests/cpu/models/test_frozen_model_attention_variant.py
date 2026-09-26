#!/usr/bin/env python
"""A frozen reference or teacher is built with the same attention variant as its policy.

The policy loader resolves the backend (``sdpa`` for Gemma 4) and builds the model with that backend's
variant (``sdpa_flex_sliding``), while the family patches key on the backend itself. A frozen model
resolved to plain ``sdpa`` would score every logprob of the objective's other half through a different
attention than the policy it anchors.

    python tests/cpu/models/test_frozen_model_attention_variant.py
"""

from unittest.mock import MagicMock, patch

import pytest
import torch

from src.distributed.loading import frozen_models


def test_the_frozen_model_is_built_with_the_policys_attention_variant():
    seen = {}
    with (
        patch.object(frozen_models, "apply_remote_code_compat_shims"),
        patch.object(frozen_models, "AutoConfig", MagicMock()),
        patch.object(frozen_models, "resolve_attn_implementation", lambda *a, **k: "sdpa"),
        patch.object(
            frozen_models, "resolve_flex_sliding_attn_implementation", lambda config, attn: f"{attn}_variant"
        ),
        patch.object(frozen_models, "apply_family_attention_patches", lambda config, attn: seen.update(patches=attn)),
        patch.object(frozen_models, "auto_load_model", lambda *a, **k: seen.update(build=k["attn_implementation"])),
        patch.object(frozen_models, "cast_parameters_to_run_dtype"),
        patch.object(frozen_models, "finalize_loaded_model"),
        patch.object(
            frozen_models, "finalize_run_model", lambda *a, **k: seen.update(finalize=k["attn_implementation"])
        ),
        patch.object(frozen_models, "warm_attention_kernels"),
    ):
        frozen_models.load_frozen_auxiliary_model("org/reference", dtype=torch.bfloat16)
    assert seen == {"patches": "sdpa", "build": "sdpa_variant", "finalize": "sdpa_variant"}


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
