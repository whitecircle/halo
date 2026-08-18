"""Stub harness for the shared frozen-auxiliary-model loader.

Every unparallelized frozen model the toolkit scores a policy against (the DPO/KTO reference, the
SDPG KL anchor, both distillation teacher paths) loads through
:func:`~src.distributed.loading.frozen_models.load_frozen_auxiliary_model`, so its module namespace is
where a test intercepts the hub fetches. That also makes these stubs a consolidation check: a call
site that re-implemented the load would fetch through its own names, and :func:`captured_load` would
find no recorded fetch.
"""

import contextlib
import types
from unittest.mock import patch

import src.distributed.loading.frozen_models as frozen_loader
import src.models.patches.gpt_oss_sinks as sink_patches

STUB_CONFIG = "AUXILIARY_CONFIG"
STUB_RESOLVED_ATTN = "flex_attention"


@contextlib.contextmanager
def stub_frozen_loader():
    """Patch every hub call, fetch-coordination scope and sink patch the loader reaches; yields the captures."""
    resolver_calls: list[dict] = []

    # No default for sinks_reset: the real resolver defaults it to True, so a stub that did the same
    # would let a caller drop the argument entirely and still satisfy the reset_sinks=True cases.
    def fake_resolve(model_config, attn_implementation, dtype, *, sinks_reset):
        resolver_calls.append(
            {
                "model_config": model_config,
                "attn_implementation": attn_implementation,
                "dtype": dtype,
                "sinks_reset": sinks_reset,
            }
        )
        return STUB_RESOLVED_ATTN

    with (
        patch.object(frozen_loader, "AutoConfig") as auto_config,
        patch.object(frozen_loader, "resolve_attn_implementation", fake_resolve),
        patch.object(frozen_loader, "auto_load_model") as auto_load,
        patch.object(frozen_loader, "from_pretrained_verified") as vlm_load,
        patch.object(frozen_loader, "fs_aware_main_first") as fetch_scope,
        # Patched where they are defined rather than where the loader imports from: the loader reaches
        # them through ``apply_sinks_policy``, so intercepting them here keeps the assertion
        # end-to-end (loader, shared policy, the branch that policy picks).
        patch.object(sink_patches, "_reset_gpt_oss_sinks") as reset_sinks,
        patch.object(sink_patches, "_set_gpt_oss_sinks_trainable") as freeze_sinks,
    ):
        auto_config.from_pretrained.return_value = STUB_CONFIG
        fetch_scope.return_value = contextlib.nullcontext()
        # The real live branch returns the layer count, which apply_sinks_policy compares (> 0) to
        # stamp the model live-sinks; a bare MagicMock return would TypeError on that comparison.
        freeze_sinks.return_value = 1
        yield types.SimpleNamespace(
            resolver_calls=resolver_calls,
            auto_config=auto_config,
            auto_load=auto_load,
            vlm_load=vlm_load,
            fetch_scope=fetch_scope,
            reset_sinks=reset_sinks,
            freeze_sinks=freeze_sinks,
        )


def captured_load(caps, *, is_vlm: bool = False):
    """The single weight fetch the stubs recorded, with the config fetch and resolver call beside it.

    ``is_vlm`` selects which loader the caller was expected to reach: the pinned
    ``AutoModelForImageTextToText`` fetch, or the config-resolved ``auto_load_model``.
    """
    loader_mock = caps.vlm_load if is_vlm else caps.auto_load
    assert loader_mock.call_count == 1, f"expected exactly one frozen weight fetch, got {loader_mock.call_count}"
    assert len(caps.resolver_calls) == 1, caps.resolver_calls
    return types.SimpleNamespace(
        model=loader_mock.return_value,
        load=loader_mock.call_args.kwargs,
        load_positional=loader_mock.call_args.args,
        resolver=caps.resolver_calls[0],
        config=caps.auto_config.from_pretrained.call_args.kwargs,
        fetch_scope=caps.fetch_scope,
        reset_sinks=caps.reset_sinks,
        freeze_sinks=caps.freeze_sinks,
    )
