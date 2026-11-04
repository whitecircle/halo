"""Bind hybrid GPU fixture requests to real runtime APIs before allocating CUDA."""

import ast
import inspect
from types import MappingProxyType
from unittest.mock import patch

import pytest
import torch
from transformers import DataCollatorForSeq2Seq
from trl import SFTConfig, SFTTrainer

from halo.distributed.context_parallel.families import Qwen3_5UlyssesGatedDelta
from halo.distributed.parallelism_config import ParallelismConfig
from halo.model_loading.finalization import finalize_loaded_model
from halo.model_loading.parallel.loader import load_distributed_model
from tests.common import resume_export
from tests.common.checkpoint_io import save_schedule
from tests.common.repo import REPO_ROOT
from tests.common.resume_export import ResumeExportRoute, run_resume_export, save_base_checkpoint

FIXTURES = (
    "tests/gpu/parallelism/cp/test_hybrid_cp_sft_equivalence.py",
    "tests/gpu/trainers/sft/test_sft_hybrid_cp.py",
    "tests/common/resume_export.py",
)
CALLABLES = MappingProxyType(
    {
        function.__name__: function
        for function in (
            load_distributed_model,
            SFTConfig,
            ParallelismConfig,
            DataCollatorForSeq2Seq,
            ResumeExportRoute,
            run_resume_export,
            save_base_checkpoint,
            finalize_loaded_model,
            Qwen3_5UlyssesGatedDelta,
        )
    }
)


def _arguments(call):
    assert not any(isinstance(value, ast.Starred) for value in call.args), "Bind an explicit fixture request"
    return [None] * len(call.args), {item.arg: None for item in call.keywords if item.arg is not None}


@pytest.mark.parametrize("path", FIXTURES)
def test_hybrid_fixture_calls_bind_real_runtime_signatures(path):
    tree = ast.parse((REPO_ROOT / path).read_text())
    bound = []
    for call in ast.walk(tree):
        if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name):
            continue
        if call.func.id in CALLABLES:
            args, kwargs = _arguments(call)
            inspect.signature(CALLABLES[call.func.id]).bind(*args, **kwargs)
            bound.append(call.func.id)
        elif call.func.id == "DistributedSFTTrainer":
            args, kwargs = _arguments(call)
            for field in ("parallelism_config", "bf16_optimizer"):
                kwargs.pop(field, None)
            inspect.signature(SFTTrainer).bind(*args, **kwargs)
            bound.append(call.func.id)
    assert "load_distributed_model" in bound or "ResumeExportRoute" in bound
    assert len(bound) >= 3


def test_loader_call_contract_rejects_the_missing_trust_argument():
    signature = inspect.signature(load_distributed_model)
    with pytest.raises(TypeError, match="trust_remote_code"):
        signature.bind("tiny-local", object(), dtype=torch.bfloat16)
    signature.bind("tiny-local", object(), dtype=torch.bfloat16, trust_remote_code=False)


def test_hybrid_local_loader_call_explicitly_disables_remote_code():
    tree = ast.parse((REPO_ROOT / FIXTURES[0]).read_text())
    calls = [
        call
        for call in ast.walk(tree)
        if isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id == "load_distributed_model"
    ]
    assert len(calls) == 1
    trust = next(keyword.value for keyword in calls[0].keywords if keyword.arg == "trust_remote_code")
    assert isinstance(trust, ast.Constant) and trust.value is False


@pytest.mark.parametrize("resume", (False, True))
def test_local_resume_route_forwards_trust_and_precision_to_the_real_loader_contract(resume):
    expected = inspect.signature(load_distributed_model)
    route = ResumeExportRoute(lambda: None, object, torch.bfloat16, "flash_attention_2", trust_remote_code=False)

    def loaded(*args, **kwargs):
        expected.bind(*args, **kwargs)
        assert kwargs["trust_remote_code"] is False
        assert kwargs["preserve_checkpoint_precision"] is resume
        return "loaded-model", None

    with patch.object(resume_export, "load_distributed_model", loaded):
        assert resume_export.load_route_model(route, "tiny-local", object(), resume=resume) == "loaded-model"


def test_existing_resume_route_trust_default_is_unchanged():
    assert ResumeExportRoute(lambda: None, object, torch.bfloat16, "flash_attention_2").trust_remote_code is True


@pytest.mark.parametrize("save_every", (None, 2))
def test_lifecycle_save_schedule_matches_sft_config_signature(save_every):
    inspect.signature(SFTConfig).bind(output_dir="tiny-local", **save_schedule(save_every))


def test_lifecycle_trainer_callback_accepts_the_shared_resume_driver_request():
    fixture = ast.parse((REPO_ROOT / FIXTURES[1]).read_text())
    builder = next(
        node for node in ast.walk(fixture) if isinstance(node, ast.FunctionDef) and node.name == "build_trainer"
    )
    params = [inspect.Parameter(value.arg, inspect.Parameter.POSITIONAL_OR_KEYWORD) for value in builder.args.args]
    params.extend(inspect.Parameter(value.arg, inspect.Parameter.KEYWORD_ONLY) for value in builder.args.kwonlyargs)
    signature = inspect.Signature(params)
    driver = ast.parse((REPO_ROOT / FIXTURES[2]).read_text())
    requests = [
        node
        for node in ast.walk(driver)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "build_trainer"
    ]
    assert len(requests) >= 2
    for request in requests:
        args, kwargs = _arguments(request)
        signature.bind(*args, **kwargs)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
