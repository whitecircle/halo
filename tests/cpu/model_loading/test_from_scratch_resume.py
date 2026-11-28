#!/usr/bin/env python
"""A resumed from-scratch run restores its checkpoint and never draws again (issue #146). Under EP, CP and TP the policy
loads from the resume checkpoint and the trainer skips its weight reload, so a draw there would train on fresh random
weights. A GptOss run's sinks keep the from-scratch handling they started with."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from accelerate import PartialState
from transformers import GptOssConfig, GptOssForCausalLM, Qwen3Config, Qwen3ForCausalLM

import halo.entrypoints.model_setup as model_setup
from halo.configs.script_arguments import ModelArguments
from halo.entrypoints.model_setup import load_model_and_tokenizer
from halo.entrypoints.run_setup import prepare_distributed_resume
from halo.entrypoints.script import ScriptContext
from halo.model_loading.parallel import loader
from halo.model_loading.parallel.model_source import ModelSource
from halo.trainers.sft import DistributedSFTTrainer
from tests.common.models import TINY_QWEN3_CONFIG
from tests.common.parallelism import make_parallelism_config

PartialState()

SEED = 1234


def _load_kwargs(tmp_path, monkeypatch, *, resume: bool, **layout) -> dict:
    """The keywords ``load_model_and_tokenizer`` hands the loader for a from-scratch run, fresh or resumed at step 3."""
    base_dir, run_dir = tmp_path / "base", tmp_path / "run"
    Qwen3Config(**TINY_QWEN3_CONFIG).save_pretrained(base_dir)
    if resume:
        torch.manual_seed(0)
        Qwen3ForCausalLM(Qwen3Config(**TINY_QWEN3_CONFIG)).save_pretrained(run_dir / "checkpoint-3")
        (run_dir / "checkpoint-3" / "trainer_state.json").write_text('{"global_step": 3}')
    training = SimpleNamespace(
        output_dir=str(run_dir),
        resume_from_checkpoint=resume,
        seed=SEED,
        model_init_kwargs=None,
        bf16=False,
        fp16=False,
        use_liger_kernel=False,
        liger_kernel_config=None,
    )
    model_config = ModelArguments(model_name_or_path=str(base_dir))
    parallelism_config = make_parallelism_config(world_size=8, gpus_per_node=8, **layout)
    checkpoint, source = prepare_distributed_resume(
        training, model_config, parallelism_config, overwrite_output_dir=False
    )
    context = ScriptContext(parallelism_config, 0, checkpoint, source, DistributedSFTTrainer)
    seen = {}

    def recording_loader(**kwargs):
        seen.update(kwargs)
        return Qwen3ForCausalLM(Qwen3Config(**TINY_QWEN3_CONFIG)), None

    monkeypatch.setattr(model_setup, "load_distributed_model", recording_loader)
    load_model_and_tokenizer(
        context,
        training,
        model_config,
        SimpleNamespace(reset_sinks=True, train_sinks=False, text_only_model=False),
        init_from_scratch=True,
    )
    return {**seen, "checkpoint": checkpoint, "base": str(base_dir)}


@pytest.mark.parametrize("layout", [{"ep_size": 8}, {"cp_size": 2}, {"tp_size": 2}], ids=["ep8", "cp2", "tp2"])
def test_a_resume_that_loads_from_its_checkpoint_reads_it_and_does_not_draw(tmp_path, monkeypatch, layout):
    seen = _load_kwargs(tmp_path, monkeypatch, resume=True, **layout)
    assert seen["model_name_or_path"] == seen["checkpoint"], (
        "premise: this layout loads the policy from the checkpoint"
    )
    assert seen["init_from_scratch"] is False, "a redraw here would train on fresh random weights"
    assert seen["init_seed"] is None
    assert seen["resumed_from_scratch"] is True, "the run's GptOss sinks must keep their from-scratch handling"


def test_a_fresh_run_draws_keyed_on_its_seed(tmp_path, monkeypatch):
    seen = _load_kwargs(tmp_path, monkeypatch, resume=False, ep_size=8)
    assert (seen["init_from_scratch"], seen["init_seed"], seen["resumed_from_scratch"]) == (True, SEED, False)


def test_a_data_parallel_resume_draws_and_leaves_the_restore_to_the_trainer(tmp_path, monkeypatch):
    """The trainer restores this resume's checkpoint over the built model, so the build is the run's own draw."""
    seen = _load_kwargs(tmp_path, monkeypatch, resume=True, use_grouped_gemm=False)
    assert seen["model_name_or_path"] == seen["base"], "premise: a data-parallel resume builds from the base"
    assert (seen["init_from_scratch"], seen["init_seed"], seen["resumed_from_scratch"]) == (True, SEED, False)


def _tiny_gpt_oss_config() -> GptOssConfig:
    return GptOssConfig(
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


def _sinks_after_load(*, init_from_scratch: bool, resumed_from_scratch: bool) -> tuple[str, list[torch.Tensor]]:
    """The backend and the sinks a GptOss load under the default ``reset_sinks: true`` ends with, the checkpoint's
    sinks set to known values."""
    config = _tiny_gpt_oss_config()
    torch.manual_seed(0)
    model = GptOssForCausalLM(config)
    with torch.no_grad():
        for layer in model.model.layers:
            layer.self_attn.sinks.fill_(0.25)
    built = {}

    def dispatch(source, pc, model_class, model_config, dtype, *, attn_implementation, **kwargs):
        built["backend"] = attn_implementation
        return model

    pc = make_parallelism_config(world_size=2, gpus_per_node=2, use_grouped_gemm=False)
    with (
        patch.object(loader, "configure_float32_matmul_precision"),
        patch.object(
            loader, "resolve_model_source", lambda path, revision, **kwargs: ModelSource(path, revision, path)
        ),
        patch.object(loader, "resolve_build_config", lambda *a, **k: config),
        patch.object(loader, "AutoTokenizer", SimpleNamespace(from_pretrained=lambda *a, **k: MagicMock())),
        patch.object(loader, "patch_flex_attention_compile"),
        patch.object(loader, "_dispatch_model_loading", dispatch),
        patch.object(loader, "warm_attention_kernels"),
    ):
        loaded, _ = loader.load_distributed_model(
            "org/gpt-oss",
            parallelism_config=pc,
            dtype=torch.bfloat16,
            attn_implementation="eager",
            init_from_scratch=init_from_scratch,
            init_seed=SEED if init_from_scratch else None,
            resumed_from_scratch=resumed_from_scratch,
            trust_remote_code=False,
        )
    return built["backend"], [layer.self_attn.sinks.detach().clone() for layer in loaded.model.layers]


def test_a_resumed_from_scratch_gpt_oss_keeps_its_trained_sinks():
    backend, sinks = _sinks_after_load(init_from_scratch=False, resumed_from_scratch=True)
    scratch_backend, scratch_sinks = _sinks_after_load(init_from_scratch=True, resumed_from_scratch=False)
    assert backend == scratch_backend, "the resume must run the backend the from-scratch run trained under"
    assert all(torch.equal(s, torch.full_like(s, 0.25)) for s in sinks), "reset_sinks neutralized the trained sinks"
    assert all(torch.equal(a, b) for a, b in zip(sinks, scratch_sinks, strict=True))


def test_a_resumed_pretrained_gpt_oss_still_neutralizes_its_sinks():
    """The control: the same load without the flag resets the sinks, so the test above can see a reset."""
    _, sinks = _sinks_after_load(init_from_scratch=False, resumed_from_scratch=False)
    assert not any(torch.equal(s, torch.full_like(s, 0.25)) for s in sinks)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
