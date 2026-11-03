"""Stub harness for the shared frozen-auxiliary-model loader; a call site re-implementing the load fetches elsewhere
and :func:`captured_load` finds nothing."""

import contextlib
import types
from unittest.mock import patch

import halo.model_loading.parallel.frozen_models as frozen_loader
import halo.models.attention.gpt_oss_sinks as sink_patches
from halo.model_loading.parallel.model_source import ModelSource

STUB_CONFIG = "AUXILIARY_CONFIG"
STUB_RESOLVED_ATTN = "flex_attention"


@contextlib.contextmanager
def stub_frozen_loader():
    """Patch every hub call, fetch scope and sink patch the loader reaches; yields the captures."""
    resolver_calls: list[dict] = []

    def fake_resolve(model_config, attn_implementation, dtype, *, sinks_policy):
        resolver_calls.append(
            {
                "model_config": model_config,
                "attn_implementation": attn_implementation,
                "dtype": dtype,
                "sinks_policy": sinks_policy,
            }
        )
        return STUB_RESOLVED_ATTN

    with (
        patch.object(frozen_loader, "resolve_build_config") as build_config,
        patch.object(frozen_loader, "resolve_attn_implementation", fake_resolve),
        patch.object(frozen_loader, "auto_load_model") as auto_load,
        patch.object(frozen_loader, "from_pretrained_verified") as vlm_load,
        patch.object(frozen_loader, "resolve_model_source") as fetch_scope,
        # Where they are defined: the loader reaches them through ``apply_sinks_policy``, keeping the assertion
        # end-to-end.
        patch.object(sink_patches, "_reset_gpt_oss_sinks") as reset_sinks,
        patch.object(sink_patches, "_set_gpt_oss_sinks_trainable") as freeze_sinks,
    ):
        build_config.return_value = STUB_CONFIG
        # The source agreement hands back the source every rank reads at its revision; unchanged here.
        fetch_scope.side_effect = lambda model_name_or_path, revision, *, tag: ModelSource(
            model_name_or_path, revision, model_name_or_path
        )
        # apply_sinks_policy compares the layer count (> 0), and a bare MagicMock would TypeError on it.
        freeze_sinks.return_value = 1
        yield types.SimpleNamespace(
            resolver_calls=resolver_calls,
            build_config=build_config,
            auto_load=auto_load,
            vlm_load=vlm_load,
            fetch_scope=fetch_scope,
            reset_sinks=reset_sinks,
            freeze_sinks=freeze_sinks,
        )


def captured_load(caps, *, is_vlm: bool = False):
    """The single weight fetch recorded, with its config fetch and resolver call; ``is_vlm`` picks the expected
    loader."""
    loader_mock = caps.vlm_load if is_vlm else caps.auto_load
    assert loader_mock.call_count == 1, f"expected exactly one frozen weight fetch, got {loader_mock.call_count}"
    assert len(caps.resolver_calls) == 1, caps.resolver_calls
    return types.SimpleNamespace(
        model=loader_mock.return_value,
        load=loader_mock.call_args.kwargs,
        load_positional=loader_mock.call_args.args,
        resolver=caps.resolver_calls[0],
        config=caps.build_config.call_args.kwargs,
        fetch_scope=caps.fetch_scope,
        reset_sinks=caps.reset_sinks,
        freeze_sinks=caps.freeze_sinks,
    )
