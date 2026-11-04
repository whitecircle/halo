#!/usr/bin/env python
"""A checkpoint no engine can take an update for is refused off its config alone, before the online and environmental
scripts read a dataset, load the model or wait on a server, and the refusal says what to run instead."""

import contextlib
import re
import sys
import types
from pathlib import Path
from unittest import mock

import pytest

from halo.models.moe.family_specs import MOE_FAMILY_SPECS, NEMOTRON_H
from halo.trainers.grpo.rollout.weight_push import reject_unsyncable_config, reject_unsyncable_model
from halo.weight_sync.registry import resolve_weight_sync_client, rollout_backends
from tests.common.repo import REPO_ROOT, load_script_module
from tests.common.tiny_models import TINY_MOE_FAMILIES, TINY_MOE_VLM_FAMILIES, tiny_family_model

_ONLINE = ("scripts/training/grpo/online.py", "max_completion_length: 64\n")
_ENVIRONMENTAL = ("scripts/training/grpo/environmental.py", "environment_type: react_math\n")
_ROUTES = {"nemotron_h": TINY_MOE_FAMILIES["nemotron_h"], "nemotron_h_omni": TINY_MOE_VLM_FAMILIES["nemotron_h_omni"]}
_REFUSED = sorted(spec.model_types[0] for spec in MOE_FAMILY_SPECS if spec.weight_sync_refusal is not None)
_DOC_LINK = re.compile(r"(agent-docs/[\w./-]+\.md)#([\w-]+)")


def _config(model_type: str) -> types.SimpleNamespace:
    return types.SimpleNamespace(model_type=model_type)


def _run_with_everything_past_the_gate_failing(tmp_path, script: tuple[str, str], config) -> None:
    path, script_body = script
    module = load_script_module(path, "halo_test_sync_refusal_" + path.rpartition("/")[2].removesuffix(".py"))
    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text(
        f"model_name_or_path: stub/checkpoint\ndataset:\n- dummy/dataset\noutput_dir: {tmp_path / 'out'}\n"
        f"bf16: false\nuse_cpu: true\n{script_body}"
    )
    runtime = types.SimpleNamespace(parallelism_config=types.SimpleNamespace(), model_source="stub/checkpoint")
    past_the_gate = AssertionError("the run went past the gate")
    with contextlib.ExitStack() as stack:
        stack.enter_context(mock.patch.object(module, "init_training_script", return_value=runtime))
        stack.enter_context(mock.patch.object(module, "probe_checkpoint", return_value=(config, False)))
        stack.enter_context(mock.patch.object(module, "load_script_datasets", side_effect=past_the_gate))
        stack.enter_context(mock.patch.object(module, "load_model_and_tokenizer", side_effect=past_the_gate))
        stack.enter_context(mock.patch("halo.entrypoints.script.install_log_tee"))
        stack.enter_context(mock.patch.object(sys, "argv", ["prog", str(yaml_path)]))
        module.main()


@pytest.mark.parametrize("script", [_ONLINE, _ENVIRONMENTAL], ids=["online", "environmental"])
@pytest.mark.parametrize("route", sorted(_ROUTES))
def test_the_scripts_refuse_the_family_before_any_data_or_load(tmp_path, script, route):
    config = tiny_family_model(_ROUTES[route]).config
    with pytest.raises(ValueError, match="does not support weight sync"):
        _run_with_everything_past_the_gate_failing(tmp_path, script, config)


@pytest.mark.parametrize("script", [_ONLINE, _ENVIRONMENTAL], ids=["online", "environmental"])
@pytest.mark.parametrize("config", [None, _config("gpt_oss"), _config("qwen3")], ids=["unread", "servable", "dense"])
def test_a_config_the_gate_has_no_verdict_on_reaches_the_datasets(tmp_path, script, config):
    with pytest.raises(AssertionError, match="went past the gate"):
        _run_with_everything_past_the_gate_failing(tmp_path, script, config)


@pytest.mark.parametrize("backend", rollout_backends())
def test_an_engine_fact_is_settled_off_the_config_too(backend):
    client_cls = resolve_weight_sync_client(backend)
    for model_type in client_cls.UNSERVABLE_MODEL_TYPES:
        with pytest.raises(ValueError, match=f"cannot serve model_type .*{model_type}"):
            reject_unsyncable_config(_config(model_type), backend)


@pytest.mark.parametrize("backend", rollout_backends())
@pytest.mark.parametrize("model_type", _REFUSED)
def test_every_refused_family_is_told_what_to_run_instead(model_type, backend):
    with pytest.raises(ValueError, match="does not support weight sync") as refusal:
        reject_unsyncable_config(_config(model_type), backend)
    assert "Train it offline" in str(refusal.value)
    assert (REPO_ROOT / "agent-docs/training-methods/grpo/offline.md").is_file()


@pytest.mark.parametrize("backend", rollout_backends())
@pytest.mark.parametrize("route", sorted(_ROUTES))
def test_the_construction_gate_says_the_same_on_a_built_model(route, backend):
    with pytest.raises(ValueError, match="Train it offline") as refusal:
        reject_unsyncable_model(tiny_family_model(_ROUTES[route]), backend)
    assert NEMOTRON_H.weight_sync_refusal in str(refusal.value)


def test_nemotrons_refusal_links_a_section_that_names_the_way_forward():
    path, anchor = _DOC_LINK.search(NEMOTRON_H.weight_sync_refusal).groups()
    page = Path(REPO_ROOT, path).read_text(encoding="utf-8")
    sections = re.split(r"^## ", page, flags=re.MULTILINE)
    section = next(body for body in sections if body.lower().replace(" ", "-").startswith(anchor))
    assert "offline-grpo-nemotron-3.5-super-vl-gsm8k-ep8.yaml" in section
    assert (REPO_ROOT / "examples/grpo/offline/nemotron_h/offline-grpo-nemotron-3.5-super-vl-gsm8k-ep8.yaml").is_file()
    for lifted_by in ("test_weight_push_exact.py", "_SYNC_EXACTNESS_SWEEP_FAMILIES", "docker/vllm/parity/fixtures"):
        assert lifted_by in section


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
