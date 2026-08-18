#!/usr/bin/env python
"""The rollout-server config-parity fixtures must cover the roster and match what the toolkit exports.

``docker/vllm/parity/generate.py`` renders one ``config.json`` per served family — from the tiny
roster config, or from the release config at a pinned revision — through the export rewrites, and
``Dockerfile.vllm`` asserts the server's transformers parses them at build. Three ways that gate can
quietly stop meaning anything, and one test each.

A family the weight sync ADMITS but no fixture covers is never checked against the server at all,
though every sync into it starts with the server parsing its config; the roster is therefore derived
from the EP registry, not listed. A fixture that drifts from what the writer produces today (a tiny
config edited, a family's legacy keys changed, a rewrite altered) certifies a config the toolkit does
not write, so the roster fixtures are re-rendered here and compared as parsed payloads — not bytes,
which would fail on key order or indentation alone. And a fixture pinned to a release is one this
offline suite cannot re-render at all, so the pin is confined to the single thing a tiny config
cannot express: an export that carries its SOURCE checkpoint's schema.

That one pinned fixture is checked structurally instead — rendering it downloads its release config;
``generate.py`` is what refreshes it.

Run: ``pytest -m cpu tests/cpu/checkpoint/test_vllm_parity_fixtures.py``
(regenerate with ``python docker/vllm/parity/generate.py`` in the training image)
"""

from __future__ import annotations

import importlib.util
import json
import os

import pytest

from tests.common.utils import REPO_ROOT

_PARITY_ROOT = os.path.join(REPO_ROOT, "docker", "vllm", "parity")


def _generate_module():
    spec = importlib.util.spec_from_file_location("vllm_parity_generate", os.path.join(_PARITY_ROOT, "generate.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _refuse_download(*args, **kwargs):
    raise AssertionError(
        f"{args[0] if args else '<repo>'}: this suite runs offline, so no fixture may reach the Hub "
        f"— read a pin off the fixture type rather than asking it for its source snapshot"
    )


@pytest.fixture(scope="module")
def generate():
    module = _generate_module()
    # Every roster render is in-memory; only a pinned fixture's `source` downloads, and nothing here
    # asks for one. Refusing the call keeps that true on a machine whose Hub cache is already warm.
    module.snapshot_download = _refuse_download
    return module


def _on_disk(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def _roster_fixtures(generate):
    """The fixtures built in memory — the ones this offline suite can re-render."""
    return [fixture for fixture in generate.FIXTURES if not isinstance(fixture, generate.HubFixture)]


def _pinned_fixtures(generate):
    """The fixtures built from a release config at a pinned revision."""
    return [fixture for fixture in generate.FIXTURES if isinstance(fixture, generate.HubFixture)]


def test_every_weight_sync_family_has_a_fixture(generate):
    """A family the sync admits with no fixture is one the gate never parsed: the first thing a
    sync needs is a server that loaded that family's checkpoint."""
    assert generate.weight_sync_families(), "premise: the EP registry admits at least one family"
    assert generate.uncovered_families() == [], (
        f"{generate.uncovered_families()}: the weight sync admits these families but no parity "
        f"fixture covers them — add one, or refuse the family with _supports_weight_sync = False"
    )


def test_every_roster_fixture_matches_a_fresh_render(generate):
    roster = {fixture.name for fixture in _roster_fixtures(generate)}
    assert roster, "premise: at least one fixture is built from the tiny roster"
    for role, fixtures, rewrite in (
        ("fixtures", generate.FIXTURES, True),
        ("unparseable", generate.UNPARSEABLE, False),
    ):
        for fixture in fixtures:
            path = generate.fixture_path(role, fixture)
            assert os.path.isfile(path), f"{role}/{fixture.name} is missing — run docker/vllm/parity/generate.py"
            if fixture.name not in roster:
                continue
            assert _on_disk(path) == generate.render(fixture, rewrite=rewrite), (
                f"{role}/{fixture.name}/config.json drifted from the roster — regenerate it"
            )


def test_only_a_source_schema_carry_is_pinned_to_a_release(generate):
    """A pinned fixture is one the drift check above cannot re-render, so nothing may be pinned that
    a tiny roster config could express — including a family transformers ships no class for, which
    builds through the vendor config module the fixture already carries."""
    for fixture in _pinned_fixtures(generate):
        assert generate.exports_source_config_schema(fixture.model_type), (
            f"{fixture.name} is pinned to a release revision but its export carries no source "
            f"schema — build it from a tiny roster config so this suite can re-render it"
        )


def test_every_hub_fixture_declares_the_family_it_ships(generate):
    """The offline half of the drift check: a checked-in hub payload must still be the family and
    revision the fixture declares, so the roster coverage above is not satisfied by a stale file."""
    hub = _pinned_fixtures(generate)
    assert hub, "premise: at least one fixture comes from a pinned release config"
    for fixture in hub:
        payload = _on_disk(generate.fixture_path("fixtures", fixture))
        assert payload.get("model_type") == fixture.model_type, (
            f"fixtures/{fixture.name}/config.json declares model_type {payload.get('model_type')!r}, "
            f"not the {fixture.model_type!r} the roster covers with it"
        )
        assert payload.get("architectures"), f"fixtures/{fixture.name} declares no architectures to resolve"
        assert len(fixture.revision) == 40, f"{fixture.name}: pin a full commit sha, not {fixture.revision!r}"


def test_the_gemma4_control_is_the_folded_form_its_rewrite_removes(generate):
    """The build gate is load-bearing only while the unrewritten Gemma 4 form still carries
    ``per_layer_config`` and the rewritten one carries the flat keys instead."""
    (folded,) = (fixture for fixture in generate.UNPARSEABLE if fixture.model_type == "gemma4")
    unrewritten = _on_disk(generate.fixture_path("unparseable", folded))["text_config"]
    rewritten = _on_disk(generate.fixture_path("fixtures", folded))["text_config"]
    assert "per_layer_config" in unrewritten and "per_layer_config" not in rewritten
    assert {"global_head_dim", "num_global_key_value_heads"} <= set(rewritten)


def test_the_step3p7_control_is_the_native_schema_its_carry_replaces(generate):
    """Same, for the source-schema carry: unrewritten, the export is transformers' own spellings —
    which the pinned server has no class for — and the carried one is the release's."""
    (native,) = (fixture for fixture in generate.UNPARSEABLE if fixture.model_type == "step3p7")
    unrewritten = _on_disk(generate.fixture_path("unparseable", native))["text_config"]
    carried = _on_disk(generate.fixture_path("fixtures", native))["text_config"]
    assert {"n_routed_experts", "num_experts_per_tok", "per_layer_config"} <= set(unrewritten)
    assert not {"n_routed_experts", "num_experts_per_tok", "per_layer_config"} & set(carried)
    assert {"moe_num_experts", "moe_top_k", "moe_layers_enum", "attention_other_setting"} <= set(carried)


def test_a_remote_code_fixture_ships_the_config_module_its_auto_map_names(generate):
    """A family the server parses through ``auto_map`` needs the module beside the config; one it
    parses natively must not be silently treated as if it had shipped one."""
    named = 0
    for fixture in generate.FIXTURES:
        directory = os.path.dirname(generate.fixture_path("fixtures", fixture))
        module = generate.config_module(_on_disk(os.path.join(directory, "config.json")))
        if module is None:
            continue
        named += 1
        assert os.path.isfile(os.path.join(directory, module)), (
            f"fixtures/{fixture.name} names {module} in its auto_map but does not ship it — the gate "
            f"cannot parse the config at all"
        )
    assert named, "premise: at least one fixture is parsed through remote code"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
