#!/usr/bin/env python
"""Resume provenance makes checkpoint master coverage strict across every supported Path-B loader."""

import ast
import inspect
import re
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from accelerate import PartialState
from transformers import Qwen3Config, Qwen3MoeConfig

import halo.entrypoints.model_setup as model_setup
import halo.model_loading.parallel.loader as loading
from halo.configs.script_arguments import ModelArguments
from halo.entrypoints.model_setup import load_model_and_tokenizer
from halo.entrypoints.run_setup import prepare_distributed_resume
from halo.entrypoints.script import ScriptContext
from halo.model_loading.parallel.model_source import ModelSource
from halo.trainers.sft import DistributedSFTTrainer
from tests.common.parallelism import make_parallelism_config
from tests.common.source_sweep import call_name, functions_in

PartialState()


class _Dispatched(Exception):
    """Stop the public loader at the leaf the dispatcher picked, before weights or groups are built."""

    def __init__(self, loader: str, from_pretrained_kwargs: dict, preserve_checkpoint_precision: bool):
        super().__init__(loader)
        self.loader = loader
        self.from_pretrained_kwargs = from_pretrained_kwargs
        self.preserve_checkpoint_precision = preserve_checkpoint_precision


# Each axis set the public loader serves: its parallel sizes, the checkpoint family it is asked to
# build (a dense and a MoE TP load take different leaves) and the leaf loader the dispatch must reach.
# PP rows sit on two NVLink domains, so each stage owns a whole one.
DISPATCH_CASES = {
    "dp": ({}, Qwen3MoeConfig, "_load_undistributed_model"),
    "cp_only": ({"cp_size": 2}, Qwen3MoeConfig, "load_cp_model"),
    "ep": ({"ep_size": 8}, Qwen3MoeConfig, "load_ep_model"),
    "ep_cp": ({"ep_size": 8, "cp_size": 2}, Qwen3MoeConfig, "load_ep_cp_model"),
    "dense_tp": ({"tp_size": 2}, Qwen3Config, "load_tp_model"),
    "moe_tp": ({"tp_size": 2}, Qwen3MoeConfig, "load_tp_moe_model"),
    "ep_tp": ({"ep_size": 8, "tp_size": 2}, Qwen3MoeConfig, "load_ep_tp_model"),
    "pure_etp": ({"expert_tp_size": 2}, Qwen3MoeConfig, "load_ep_model"),
    "ep_etp": ({"ep_size": 4, "expert_tp_size": 2}, Qwen3MoeConfig, "load_ep_model"),
    "pp": ({"pp_size": 2}, Qwen3MoeConfig, "load_pp_stage_model"),
    "pp_ep": ({"pp_size": 2, "ep_size": 8}, Qwen3MoeConfig, "load_pp_stage_model"),
}


def _leaf_loaders() -> set[str]:
    """The per-mode loaders ``_dispatch_model_loading`` calls, read off its source."""
    tree = ast.parse(inspect.getsource(loading._dispatch_model_loading))
    names = {call_name(node) for node in ast.walk(tree) if isinstance(node, ast.Call)}
    return {name for name in names if name and re.fullmatch(r"_?load_\w+_model", name)}


def _stopping_loader(name: str):
    def load(source, parallelism, model_class, from_pretrained_kwargs, *, preserve_checkpoint_precision, **kwargs):
        raise _Dispatched(name, dict(from_pretrained_kwargs), preserve_checkpoint_precision)

    return load


@pytest.mark.parametrize("case", sorted(DISPATCH_CASES))
@pytest.mark.parametrize("preserve", (False, True), ids=("fresh", "strict_resume"))
def test_the_real_dispatch_hands_each_axis_set_its_loader_and_the_strict_master_flag(monkeypatch, case, preserve):
    axes, config_class, expected_loader = DISPATCH_CASES[case]
    pc = make_parallelism_config(world_size=16 if "pp_size" in axes else 8, gpus_per_node=8, **axes)
    config = config_class()
    monkeypatch.setattr(
        loading, "resolve_model_source", lambda path, revision, **kwargs: ModelSource(path, revision, path)
    )
    monkeypatch.setattr(loading.AutoConfig, "from_pretrained", lambda *args, **kwargs: config)
    monkeypatch.setattr(loading.AutoTokenizer, "from_pretrained", lambda *args, **kwargs: SimpleNamespace())
    # The module-tree gate has its own test; it would refuse the eager label standing in for attention under CP.
    monkeypatch.setattr(loading, "reject_unsupported_structure", lambda *args, **kwargs: None)
    for name in _leaf_loaders():
        monkeypatch.setattr(loading, name, _stopping_loader(name))
    with pytest.raises(_Dispatched) as dispatched:
        loading.load_distributed_model(
            "org/base",
            pc,
            dtype=torch.bfloat16,
            attn_implementation="eager",
            preserve_checkpoint_precision=preserve,
            trust_remote_code=True,
        )
    assert dispatched.value.loader == expected_loader
    assert dispatched.value.preserve_checkpoint_precision is preserve
    assert dispatched.value.from_pretrained_kwargs["dtype"] is torch.bfloat16


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
    model_config = ModelArguments(model_name_or_path="org/base")
    pc = make_parallelism_config(world_size=8, gpus_per_node=8, ep_size=8)
    checkpoint, source = prepare_distributed_resume(training, model_config, pc, overwrite_output_dir=False)
    runtime = ScriptContext(pc, 0, checkpoint, source, DistributedSFTTrainer)

    def load(**kwargs):
        # The name from_pretrained gives the config of the directory it read.
        return SimpleNamespace(config=SimpleNamespace(_name_or_path=kwargs["model_name_or_path"])), SimpleNamespace()

    loader = Mock(side_effect=load)
    monkeypatch.setattr(model_setup, "load_distributed_model", loader)
    model, _ = load_model_and_tokenizer(
        runtime,
        training,
        model_config,
        SimpleNamespace(reset_sinks=True, train_sinks=False, text_only_model=False, init_from_scratch=False),
    )
    expected = resume in ("explicit", True)
    assert loader.call_args.kwargs["preserve_checkpoint_precision"] is expected
    assert runtime.policy_from_checkpoint is expected
    assert loader.call_args.kwargs["model_name_or_path"] == (str(checkpoint_dir) if expected else "org/base")
    assert model_config.model_name_or_path == "org/base"
    # A resumed run's model card names this as its base_model (tests/cpu/grpo/test_resume_reference_source.py).
    assert model.config._name_or_path == "org/base"


def test_modality_aware_training_calls_forward_the_runtime_checkpoint_identity():
    """The checkpoint identity reaches the loader only through the script's own context."""
    calls = [
        (path, node)
        for path, function in functions_in(("scripts/training",))
        for node in ast.walk(function)
        if isinstance(node, ast.Call) and call_name(node) == "load_policy_model"
    ]
    assert len(calls) >= 6, "the modality-aware policy-construction sweep must remain non-vacuous"
    for path, call in calls:
        assert isinstance(call.args[0], ast.Name) and call.args[0].id == "context", path


def test_the_dispatch_matrix_reaches_every_leaf_loader():
    assert {loader for _, _, loader in DISPATCH_CASES.values()} == _leaf_loaders()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
