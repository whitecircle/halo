#!/usr/bin/env python
"""Rollout-server config-parity fixtures (``docker/vllm/parity/generate.py``) cover the roster and match what the
toolkit exports, compared as parsed payloads. Only source-schema carries are pinned."""

from __future__ import annotations

import importlib.util
import json
import os
import shutil

import pytest

from halo.checkpoint.config_export import drop_unshipped_auto_map
from tests.common.repo import REPO_ROOT

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
    # Refusing the call keeps every render in-memory on a machine with a warm Hub cache.
    module.snapshot_download = _refuse_download
    return module


def _on_disk(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def _roster_fixtures(generate):
    return [fixture for fixture in generate.FIXTURES if not isinstance(fixture, generate.HubFixture)]


def _pinned_fixtures(generate):
    return [fixture for fixture in generate.FIXTURES if isinstance(fixture, generate.HubFixture)]


def test_every_weight_sync_family_has_a_fixture(generate):
    """The first thing a sync needs is a server that loaded that family's checkpoint."""
    assert generate.weight_sync_families(), "premise: the EP registry admits at least one family"
    assert generate.uncovered_families() == [], (
        f"{generate.uncovered_families()}: the weight sync admits these families but no parity "
        f"fixture covers them — add one, or refuse the family (a weight_sync_refusal on its spec, or the vLLM "
        f"client's UNSERVABLE_MODEL_TYPES)"
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
    """A pinned fixture is one the drift check cannot re-render, so nothing a tiny config could express may be
    pinned."""
    for fixture in _pinned_fixtures(generate):
        assert generate.find_moe_family_spec(fixture.model_type).exports_source_config_schema, (
            f"{fixture.name} is pinned to a release revision but its export carries no source "
            f"schema — build it from a tiny roster config so this suite can re-render it"
        )


def test_every_hub_fixture_declares_the_family_it_ships(generate):
    """So the roster coverage is not satisfied by a stale file."""
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


def test_the_step3p7_control_is_the_native_schema_its_carry_replaces(generate):
    """Unrewritten, the export is spellings the pinned vLLM refuses to read; the gate means something only while the
    control differs."""
    (native,) = (fixture for fixture in generate.UNPARSEABLE if fixture.model_type == "step3p7")
    unrewritten = _on_disk(generate.fixture_path("unparseable", native))["text_config"]
    carried = _on_disk(generate.fixture_path("fixtures", native))["text_config"]
    assert {"n_routed_experts", "num_experts_per_tok", "per_layer_config"} <= set(unrewritten)
    assert not {"n_routed_experts", "num_experts_per_tok", "per_layer_config"} & set(carried)
    assert {"moe_num_experts", "moe_top_k", "moe_layers_enum", "attention_other_setting"} <= set(carried)


def test_a_control_names_no_remote_code_it_does_not_ship(generate, tmp_path):
    """A control still naming a module is refused for the missing file alone, and the gate then proves nothing."""
    assert generate.UNPARSEABLE, "premise: at least one family's export is rewritten, so it has a control"
    for fixture in generate.UNPARSEABLE:
        path = generate.fixture_path("unparseable", fixture)
        staged = tmp_path / fixture.name
        shutil.copytree(os.path.dirname(path), staged)
        drop_unshipped_auto_map(str(staged))
        assert _on_disk(staged / "config.json") == _on_disk(path), (
            f"unparseable/{fixture.name} names remote code it does not ship, which the export without the "
            f"rewrite never does — regenerate it"
        )


def test_a_remote_code_fixture_ships_the_config_module_its_auto_map_names(generate):
    """A family parsed through ``auto_map`` needs the module beside the config."""
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
