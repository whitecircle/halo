#!/usr/bin/env python
"""A resumed on-policy GRPO run keeps the run's base as its KL reference.

A resume whose policy is built from the checkpoint (every EP/CP/TP resume, dense included under the
default grouped GEMM) reads its weights from that directory. Were the reference read from the same
place, it would be the step-N weights: the KL restarts near zero and the anchor moves again on every
resume. The entry scripts load the reference from the configured model and hand it to TRL. The policy
config keeps the base as its ``_name_or_path`` (the ``base_model`` TRL's model card names), while its
weights, and the stamps the resume and the export read, come from the checkpoint.

    python tests/cpu/grpo/test_resume_reference_source.py
"""

from types import SimpleNamespace

import pytest
import torch
from accelerate import PartialState
from transformers import Qwen3Config, Qwen3ForCausalLM
from trl import ModelConfig

import src.distributed.loading.vlm_setup as vlm_setup
from src.checkpoint.config_export import LOADED_WEIGHTS_FROM_ATTR, checkpoint_source_ref
from src.distributed.checkpoint.loader import weights_read_from
from src.distributed.loading.frozen_models import load_reference_model_for_on_policy_grpo
from src.training.environment import prepare_distributed_resume
from src.training.script_runner import ScriptRuntime, load_script_model
from tests.common.models import TINY_QWEN3_CONFIG
from tests.common.offline_grpo import make_offline_tokenizer
from tests.common.parallelism import make_parallelism_config
from tests.common.source_sweep import called_names, functions_in

PartialState()

# The two on-policy GRPO entry scripts, whose loaded policy names the KL reference's source.
ON_POLICY_GRPO_SCRIPTS = ("scripts/training/online_grpo/rlvr.py", "scripts/training/environmental_grpo.py")


def _save_tiny_qwen3(directory, seed: int) -> dict[str, torch.Tensor]:
    torch.manual_seed(seed)
    model = Qwen3ForCausalLM(Qwen3Config(**TINY_QWEN3_CONFIG))
    model.save_pretrained(directory)
    return {name: tensor.clone() for name, tensor in model.state_dict().items()}


def _from_pretrained_loader(*, model_name_or_path, **_):
    """What every leaf of ``load_distributed_model`` leaves on the policy's identity: ``from_pretrained``
    names the config after the directory it read, and the loader stamps where the weights came from."""
    model = Qwen3ForCausalLM.from_pretrained(model_name_or_path, dtype=torch.float32)
    setattr(model, LOADED_WEIGHTS_FROM_ATTR, model_name_or_path)
    return model, None


def _same_weights(model: torch.nn.Module, expected: dict[str, torch.Tensor]) -> bool:
    return all(torch.equal(tensor, expected[name]) for name, tensor in model.state_dict().items())


def _resumed_policy(tmp_path, monkeypatch):
    """A step-3 EP resume of a tiny base: ``(policy, model_config, parallelism_config, checkpoint, base,
    trained)``, the policy built from the checkpoint as ``load_script_model`` builds it."""
    base_dir, checkpoint_dir = tmp_path / "base", tmp_path / "run" / "checkpoint-3"
    base = _save_tiny_qwen3(base_dir, seed=0)
    trained = _save_tiny_qwen3(checkpoint_dir, seed=1)
    (checkpoint_dir / "trainer_state.json").write_text('{"global_step": 3}')
    training = SimpleNamespace(
        output_dir=str(tmp_path / "run"),
        resume_from_checkpoint=True,
        model_init_kwargs=None,
        bf16=False,
        fp16=False,
        use_liger_kernel=False,
        liger_kernel_config=None,
    )
    model_config = ModelConfig(model_name_or_path=str(base_dir))
    parallelism_config = make_parallelism_config(world_size=8, gpus_per_node=8, ep_size=8)
    checkpoint, source = prepare_distributed_resume(training, model_config, parallelism_config)
    runtime = ScriptRuntime(parallelism_config, "ep8", 0, checkpoint, source)
    assert runtime.policy_from_checkpoint, "premise: an EP resume builds the policy from its checkpoint"
    monkeypatch.setattr(vlm_setup, "load_distributed_model", _from_pretrained_loader)

    policy, _ = load_script_model(
        runtime, training, model_config, SimpleNamespace(reset_sinks=True, train_sinks=False, text_only_model=False)
    )
    return policy, model_config, parallelism_config, checkpoint, base, trained


def test_a_resumed_policy_names_the_base_and_holds_the_checkpoint_weights(tmp_path, monkeypatch):
    policy, model_config, _, checkpoint, _, trained = _resumed_policy(tmp_path, monkeypatch)

    assert policy.config._name_or_path == model_config.model_name_or_path, (
        "a resumed run's model card would name the checkpoint directory instead of the base"
    )
    assert _same_weights(policy, trained), "the policy itself must still resume from the checkpoint"
    assert weights_read_from(policy) == checkpoint, "the resume's skip-the-reload decision reads this"
    assert checkpoint_source_ref(policy) == checkpoint, "the export's config schema source reads this"


def test_a_resumed_runs_toolkit_reference_is_the_base(tmp_path, monkeypatch):
    """The reference the scripts load and the trainer hands TRL reads the configured model, whatever the
    policy resumed from."""
    policy, model_config, parallelism_config, _, base, trained = _resumed_policy(tmp_path, monkeypatch)
    token_args = SimpleNamespace(
        eos_token=None,
        bos_token=None,
        pad_token=None,
        chat_template=None,
        force_chat_template=False,
        added_special_tokens=None,
        tokenizer_backend="hf",
    )
    reference = load_reference_model_for_on_policy_grpo(
        token_args,
        model_config,
        SimpleNamespace(beta=0.04, bf16=False, fp16=False),
        parallelism_config,
        make_offline_tokenizer(),
        peft_config=None,
        reset_sinks=True,
        attn_default="sdpa",
    )
    assert _same_weights(reference, base), "the KL reference must be the run's base"
    assert not _same_weights(reference, trained), "the KL reference loaded the resume checkpoint"
    assert _same_weights(policy, trained)


def test_the_on_policy_grpo_scripts_load_their_policy_through_the_corrected_seam():
    """The base name lives in ``load_script_model``; a script loading its policy elsewhere loses it."""
    loading = {
        path
        for path, function in functions_in(ON_POLICY_GRPO_SCRIPTS)
        if load_script_model.__name__ in called_names(function)
    }
    assert loading == set(ON_POLICY_GRPO_SCRIPTS)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
