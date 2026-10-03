#!/usr/bin/env python
"""Resume provenance makes checkpoint master coverage strict across every supported Path-B loader."""

import ast
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from accelerate import PartialState
from transformers import Qwen3MoeConfig
from trl import ModelConfig

import src.distributed.loading.model_loading as loading
import src.distributed.loading.vlm_setup as vlm_setup
from src.distributed.parallelism_config import ParallelismConfig
from src.training.environment import prepare_distributed_resume
from src.training.script_runner import ScriptRuntime, load_script_model
from tests.common.parallelism import make_parallelism_config
from tests.common.source_sweep import call_name, functions_in

PartialState()


class _Dispatched(Exception):
    """Stop a public-loader plumbing probe before weights or distributed groups are constructed."""


@pytest.mark.parametrize(
    "axes,expected",
    [
        pytest.param({}, True, id="dp"),
        pytest.param({"cp": True}, True, id="cp_only"),
        pytest.param({"ep": True}, True, id="ep"),
        pytest.param({"ep": True, "cp": True}, True, id="ep_cp"),
        pytest.param({"tp": True}, True, id="tp"),
        pytest.param({"ep": True, "tp": True}, True, id="ep_tp"),
        pytest.param({"etp": True}, True, id="pure_etp"),
        pytest.param({"ep": True, "etp": True}, True, id="ep_etp"),
        pytest.param({"pp": True}, False, id="pp"),
        pytest.param({"ep": True, "pp": True}, False, id="pp_ep"),
    ],
)
@pytest.mark.parametrize("preserve", (False, True))
def test_public_loader_forwards_strict_master_coverage_to_supported_loaders(monkeypatch, axes, expected, preserve):
    pc = Mock(spec=ParallelismConfig)
    pc.ep_size = 8 if axes.get("ep") else 1
    pc.is_cp_mode = axes.get("cp", False)
    pc.is_tp_mode = axes.get("tp", False)
    pc.is_expert_tp_mode = axes.get("etp", False)
    pc.is_pp_mode = axes.get("pp", False)
    config = Qwen3MoeConfig()
    monkeypatch.setattr(loading, "_validate_launch_method_for_parallelism", lambda config: None)
    monkeypatch.setattr(loading, "_validate_gmm_launch_method", lambda *args: None)
    monkeypatch.setattr(loading, "_validate_fp32_non_ep_params", lambda *args: None)
    monkeypatch.setattr(loading, "_ensure_model_downloaded", lambda *args, **kwargs: None)
    monkeypatch.setattr(loading.AutoConfig, "from_pretrained", lambda *args, **kwargs: config)
    monkeypatch.setattr(loading.AutoTokenizer, "from_pretrained", lambda *args, **kwargs: SimpleNamespace())
    seen = {}

    def dispatch(source, parallelism, model_class, kwargs):
        seen.update(kwargs)
        raise _Dispatched

    monkeypatch.setattr(loading, "_dispatch_model_loading", dispatch)
    with pytest.raises(_Dispatched):
        loading.load_distributed_model(
            "org/base", pc, dtype=torch.bfloat16, attn_implementation="eager", preserve_checkpoint_precision=preserve
        )
    assert seen.get("preserve_checkpoint_precision", False) is (preserve and expected)
    assert seen["dtype"] is torch.bfloat16


@pytest.mark.parametrize("resume", (False, "explicit", True, "adapter"), ids=("fresh", "explicit", "auto", "adapter"))
def test_script_model_uses_resolved_checkpoint_identity_not_resume_flag(tmp_path, monkeypatch, resume):
    output = tmp_path / "run"
    checkpoint_dir = output / "checkpoint-3"
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "trainer_state.json").write_text('{"global_step": 3}')
    weights = "adapter_model.safetensors" if resume == "adapter" else "model.safetensors"
    (checkpoint_dir / weights).touch()
    training = SimpleNamespace(
        output_dir=str(output),
        resume_from_checkpoint=str(checkpoint_dir) if resume == "explicit" else bool(resume),
        model_init_kwargs=None,
        bf16=True,
        fp16=False,
        use_liger_kernel=False,
        liger_kernel_config=None,
    )
    model_config = ModelConfig(model_name_or_path="org/base")
    pc = make_parallelism_config(world_size=8, gpus_per_node=8, ep_size=8)
    checkpoint, source = prepare_distributed_resume(training, model_config, pc)
    runtime = ScriptRuntime(pc, "ep", 0, checkpoint, source)
    loader = Mock(return_value=(SimpleNamespace(), SimpleNamespace()))
    monkeypatch.setattr(vlm_setup, "load_distributed_model", loader)
    load_script_model(
        runtime,
        training,
        model_config,
        SimpleNamespace(reset_sinks=True, train_sinks=False, text_only_model=False),
    )
    expected = resume in ("explicit", True)
    assert loader.call_args.kwargs["preserve_checkpoint_precision"] is expected
    assert runtime.policy_from_checkpoint is expected
    assert loader.call_args.kwargs["model_name_or_path"] == (str(checkpoint_dir) if expected else "org/base")
    assert model_config.model_name_or_path == "org/base"


def test_modality_aware_training_calls_forward_the_runtime_checkpoint_identity():
    calls = [
        (path, node)
        for path, function in functions_in(("scripts/training",))
        for node in ast.walk(function)
        if isinstance(node, ast.Call) and call_name(node) == "load_model_for_training"
    ]
    assert len(calls) >= 6, "the modality-aware policy-construction sweep must remain non-vacuous"
    for path, call in calls:
        value = {keyword.arg: keyword.value for keyword in call.keywords}.get("preserve_checkpoint_precision")
        assert isinstance(value, ast.Attribute) and value.attr == "policy_from_checkpoint", path
        assert isinstance(value.value, ast.Name) and value.value.id == "runtime", path


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
