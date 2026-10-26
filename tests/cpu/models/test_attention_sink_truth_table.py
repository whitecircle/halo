#!/usr/bin/env python
"""The attention backend and sink-reset verdict the policy loader resolves for GptOss, per sink-flag combination.

``load_distributed_model`` folds ``reset_sinks``, ``train_sinks``, ``init_from_scratch`` and context
parallelism into the one verdict the attention resolver approves a sink-dropping backend on. Each row pins, for
every requested backend, what the model is built with and the verdict stamped on its config, or which gate
refused it.

    python tests/cpu/models/test_attention_sink_truth_table.py
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from accelerate import PartialState
from transformers import AutoModelForCausalLM, GptOssConfig
from transformers.models.gpt_oss.modeling_gpt_oss import GptOssPreTrainedModel

from src.distributed.loading import model_loading
from src.models.attention.gpt_oss_sinks import RESOLVED_SINKS_RESET_ATTR
from tests.common.parallelism import make_parallelism_config

PartialState()  # the loader logs through accelerate's adapter, which needs the state

REQUESTS = ("sdpa", "flash_attention_2", "flex_attention", "eager")

# The refusals a row can end in, each keyed by the message fragment of the gate that raises it.
CONTRADICTION = "refused: train_sinks contradicts reset_sinks"
CP_LIVE_SINKS = "refused: CP with live sinks"
DROPS_SINKS = "refused: backend drops live sinks"
REFUSALS = {
    "contradicts reset_sinks": CONTRADICTION,
    "requires reset_sinks: true": CP_LIVE_SINKS,
    "silently drops the learnable attention sinks": DROPS_SINKS,
}
NEUTRALIZED_FLASH = ("flash_attention_2", True)
LIVE_FLEX = ("flex_attention", False)
LIVE_EAGER = ("eager", False)

# (reset_sinks, train_sinks, init_from_scratch, context_parallel) -> per entry of REQUESTS, the backend the model is
# built with and the reset verdict on its config, or the refusal. A from-scratch build has no pretrained sinks to
# neutralize, so it resolves as live; CP must refuse live sinks and runs flex requests on flash.
EXPECTED = {
    (True, True, True, True): (CONTRADICTION,) * 4,
    (True, True, True, False): (CONTRADICTION,) * 4,
    (True, True, False, True): (CONTRADICTION,) * 4,
    (True, True, False, False): (CONTRADICTION,) * 4,
    (True, False, True, True): (DROPS_SINKS, DROPS_SINKS, DROPS_SINKS, LIVE_EAGER),
    (True, False, True, False): (DROPS_SINKS, DROPS_SINKS, LIVE_FLEX, LIVE_EAGER),
    (True, False, False, True): (("sdpa", True), NEUTRALIZED_FLASH, NEUTRALIZED_FLASH, ("eager", True)),
    (True, False, False, False): (("sdpa", True), NEUTRALIZED_FLASH, ("flex_attention", True), ("eager", True)),
    (False, True, True, True): (CP_LIVE_SINKS,) * 4,
    (False, True, True, False): (DROPS_SINKS, DROPS_SINKS, LIVE_FLEX, LIVE_EAGER),
    (False, True, False, True): (CP_LIVE_SINKS,) * 4,
    (False, True, False, False): (DROPS_SINKS, DROPS_SINKS, LIVE_FLEX, LIVE_EAGER),
    (False, False, True, True): (CP_LIVE_SINKS,) * 4,
    (False, False, True, False): (DROPS_SINKS, DROPS_SINKS, LIVE_FLEX, LIVE_EAGER),
    (False, False, False, True): (CP_LIVE_SINKS,) * 4,
    (False, False, False, False): (DROPS_SINKS, DROPS_SINKS, LIVE_FLEX, LIVE_EAGER),
}


class _Built(Exception):
    """Stops the loader at the build seam, past every attention decision."""


@pytest.fixture(autouse=True)
def _restore_gpt_oss_class_flags():
    """The resolver flips GptOss's class-level sdpa refusal and flash fallback list; undo both per case."""
    flagged = {
        cls: cls.__dict__["_supports_sdpa"]
        for cls in AutoModelForCausalLM._model_mapping[GptOssConfig].__mro__
        if "_supports_sdpa" in cls.__dict__
    }
    compatible = GptOssPreTrainedModel._compatible_flash_implementations
    yield
    for cls, value in flagged.items():
        cls._supports_sdpa = value
    GptOssPreTrainedModel._compatible_flash_implementations = compatible


def _resolve(requested, reset_sinks, train_sinks, init_from_scratch, context_parallel):
    config = GptOssConfig(
        num_hidden_layers=1,
        hidden_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        intermediate_size=128,
        num_local_experts=2,
        num_experts_per_tok=1,
        vocab_size=128,
    )
    built = {}

    def seam(model_config, attn_implementation):
        built["backend"] = attn_implementation
        raise _Built

    pc = make_parallelism_config(world_size=2, gpus_per_node=2, cp_size=2 if context_parallel else 1)
    with (
        patch.object(model_loading, "configure_float32_matmul_precision"),
        patch.object(model_loading, "apply_remote_code_compat_shims"),
        patch.object(model_loading, "resolve_model_source", lambda path, revision, **kwargs: revision),
        patch.object(model_loading, "AutoConfig", SimpleNamespace(from_pretrained=lambda *a, **k: config)),
        patch.object(model_loading, "AutoTokenizer", SimpleNamespace(from_pretrained=lambda *a, **k: MagicMock())),
        patch.object(model_loading, "is_blackwell_gpu", lambda: False),
        patch.object(model_loading, "patch_flex_attention_compile"),
        patch.object(model_loading, "apply_family_patches", seam),
    ):
        try:
            model_loading.load_distributed_model(
                "org/gpt-oss",
                parallelism_config=pc,
                dtype=torch.bfloat16,
                attn_implementation=requested,
                reset_sinks=reset_sinks,
                train_sinks=train_sinks,
                init_from_scratch=init_from_scratch,
            )
        except _Built:
            return built["backend"], getattr(config, RESOLVED_SINKS_RESET_ATTR, None)
        except ValueError as err:
            return next((label for fragment, label in REFUSALS.items() if fragment in str(err)), str(err))
    raise AssertionError("the loader passed the build seam without reaching it")


@pytest.mark.parametrize("flags", list(EXPECTED), ids=lambda flags: "-".join(map(str, flags)))
@pytest.mark.parametrize("requested", REQUESTS)
def test_sink_flags_resolve_as_tabled(flags, requested):
    assert _resolve(requested, *flags) == EXPECTED[flags][REQUESTS.index(requested)]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
