#!/usr/bin/env python
"""The on-policy GRPO trainers hold the KL reference the toolkit loaded, never one TRL builds itself.

TRL's ``GRPOTrainer`` builds a reference at ``beta != 0`` on a policy no PEFT adapter wraps, by
``create_model_from_path`` on the policy config's name: fp32, the hub's default revision, the default
attention, the non-persistent buffers as ``from_pretrained`` left them. The entry scripts load it with
the frozen loader instead (:func:`load_reference_model_for_on_policy_grpo`: the run dtype, the policy's
revision pin, attention request and sinks policy, the buffers repaired), and the trainers hand it to
TRL's own build call, so TRL's placement and ``prepare_model`` still run on it. A run that holds no
reference loads none; a reference TRL asks for with none passed, and a passed one it never reads, are
refused.

    python tests/cpu/grpo/test_on_policy_kl_reference.py
"""

import ast
import inspect
import logging
import textwrap
import types

import pytest
import torch
import trl.generation.vllm_generation as trl_vllm_generation
import trl.trainer.grpo_trainer as trl_grpo_trainer
from accelerate import PartialState
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import Qwen3Config, Qwen3ForCausalLM
from trl import GRPOConfig, GRPOTrainer, ModelConfig

import src.distributed.loading.frozen_models as frozen_models
from src.distributed.loading.frozen_models import (
    PREFERENCE_REFERENCE_ALTERNATIVES,
    load_reference_model_for_on_policy_grpo,
)
from src.distributed.parallelism_config import ParallelismConfig
from src.trainers.grpo import online
from src.trainers.grpo.environmental import DistributedAsyncEnvironmentalGRPOTrainer
from src.trainers.grpo.mixins.on_policy_init import OnPolicyGRPOInitMixin
from src.trainers.grpo.online import DistributedGRPOTrainer
from tests.common.frozen_loader import captured_load, stub_frozen_loader
from tests.common.models import TINY_QWEN3_CONFIG
from tests.common.offline_grpo import make_offline_tokenizer
from tests.common.parallelism import make_parallelism_config
from tests.common.source_sweep import call_name, functions_in

PartialState()  # the trainers' accelerate loggers require an initialized state

BASE = "org/base"
ON_POLICY_GRPO_SCRIPTS = ("scripts/training/online_grpo/rlvr.py", "scripts/training/environmental_grpo.py")
BETA = 0.04
# TRL's own build call, which every constructor must leave bound once it returns or raises.
_TRL_BUILD = trl_grpo_trainer.create_model_from_path
# The client class TRL's server-mode generation builds, which the online trainer swaps for the duration of its ctor.
_TRL_VLLM_CLIENT = trl_vllm_generation.VLLMClient


def _token_setup_args():
    """The script arguments the tokenizer setup seam reads, all at their defaults."""
    return types.SimpleNamespace(
        eos_token=None,
        bos_token=None,
        pad_token=None,
        chat_template=None,
        force_chat_template=False,
        added_special_tokens=None,
        tokenizer_backend="hf",
    )


def _load_reference(source: str, *, beta: float = BETA, peft_config=None, revision=None, reset_sinks=True):
    """The scripts' reference load for a bf16 run, as their ``main`` calls it."""
    return load_reference_model_for_on_policy_grpo(
        _token_setup_args(),
        ModelConfig(model_name_or_path=source, model_revision=revision, trust_remote_code=False),
        types.SimpleNamespace(beta=beta, bf16=True, fp16=False),
        ParallelismConfig(),
        make_offline_tokenizer(),
        peft_config=peft_config,
        reset_sinks=reset_sinks,
        attn_default="sdpa",
    )


class _OnPolicyHost(OnPolicyGRPOInitMixin, GRPOTrainer):
    """TRL's own constructor under the trainers' hand-off, without the rest of the toolkit's spine."""

    def __init__(self, *, ref_model=None, **kwargs):
        self._supplied_ref_model = ref_model
        self.parallelism_config = ParallelismConfig()
        with self._supplying_kl_reference():
            super().__init__(**kwargs)


class _StubWeightSyncClient:
    """The vendored vLLM client TRL's server-mode generation builds, with no server behind it."""

    def __init__(self, **_):
        pass

    def init_communicator(self, device):
        pass


def _grpo_args(tmp_path, *, beta: float = BETA, **overrides) -> GRPOConfig:
    return GRPOConfig(
        output_dir=str(tmp_path / "out"),
        use_cpu=True,
        bf16=False,
        beta=beta,
        per_device_train_batch_size=2,
        num_generations=2,
        max_completion_length=4,
        report_to="none",
        **overrides,
    )


def _trainer_kwargs(tmp_path, *, beta: float = BETA, peft_config=None, **arg_overrides) -> dict:
    """What a GRPO trainer is built from here, the model aside."""
    return {
        "reward_funcs": lambda completions, **_: [0.0] * len(completions),
        "args": _grpo_args(tmp_path, beta=beta, **arg_overrides),
        "train_dataset": Dataset.from_dict({"prompt": ["question answer"] * 2}),
        "processing_class": make_offline_tokenizer(),
        "peft_config": peft_config,
    }


def _host(tmp_path, model, *, beta: float = BETA, ref_model=None, peft_config=None) -> _OnPolicyHost:
    return _OnPolicyHost(
        model=model, ref_model=ref_model, **_trainer_kwargs(tmp_path, beta=beta, peft_config=peft_config)
    )


@pytest.fixture
def base_dir(tmp_path):
    """A tiny fp32 Qwen3 checkpoint on disk, the run's configured model."""
    torch.manual_seed(0)
    Qwen3ForCausalLM(Qwen3Config(**TINY_QWEN3_CONFIG)).save_pretrained(tmp_path / "base")
    return str(tmp_path / "base")


def test_trl_holds_the_toolkit_reference_in_the_run_dtype_with_its_buffers_repaired(tmp_path, base_dir, monkeypatch):
    """TRL's own build would hold an fp32 copy read back by ``from_pretrained`` alone; the run's reference is
    the bf16 one the frozen loader built and repaired, placed and frozen by the hand-off."""
    finalized = []
    repair = frozen_models.finalize_loaded_model

    def recording_repair(model):
        finalized.append(model)
        return repair(model)

    monkeypatch.setattr(frozen_models, "finalize_loaded_model", recording_repair)
    reference = _load_reference(base_dir)
    trainer = _host(tmp_path, Qwen3ForCausalLM.from_pretrained(base_dir), ref_model=reference)

    held = trainer.accelerator.unwrap_model(trainer.ref_model)
    assert held is reference, "TRL holds a reference of its own, not the one the script loaded"
    assert {p.dtype for p in held.parameters()} == {torch.bfloat16}
    assert any(model is held for model in finalized), "the held reference skipped the buffer repair"
    expected = Qwen3ForCausalLM(Qwen3Config(**TINY_QWEN3_CONFIG)).model.rotary_emb.inv_freq
    torch.testing.assert_close(held.model.rotary_emb.inv_freq, expected.to(held.model.rotary_emb.inv_freq.dtype))
    assert not any(p.requires_grad for p in held.parameters()), "the reference must be frozen"
    assert not held.training


def test_the_scripts_reference_takes_the_policys_pin_dtype_attention_and_sinks():
    """Every setting that changes a log-prob reaches the reference load from the policy's own config."""
    with stub_frozen_loader() as caps:
        reference = _load_reference(BASE, revision="abc123", reset_sinks=False)
    captured = captured_load(caps)
    assert reference is captured.model
    assert captured.load_positional[0] == BASE
    assert captured.config["revision"] == captured.load["revision"] == "abc123"
    assert captured.load["dtype"] is torch.bfloat16
    assert captured.resolver["attn_implementation"] == "sdpa"
    assert captured.resolver["sinks_reset"] is False
    captured.freeze_sinks.assert_called_once()


@pytest.mark.parametrize(
    ("beta", "peft_config"),
    [(0.0, None), (BETA, LoraConfig(target_modules=["q_proj"]))],
    ids=["beta-zero", "peft"],
)
def test_a_run_that_holds_no_reference_loads_none(beta, peft_config):
    """``beta: 0`` takes no KL, and a PEFT policy is its own reference with the adapter disabled."""
    with stub_frozen_loader() as caps:
        assert _load_reference(BASE, beta=beta, peft_config=peft_config) is None
    caps.auto_load.assert_not_called()
    caps.vlm_load.assert_not_called()


def test_a_reference_trl_asks_for_with_none_passed_is_refused(tmp_path, base_dir):
    with pytest.raises(ValueError, match="load_reference_model_for_on_policy_grpo"):
        _host(tmp_path, Qwen3ForCausalLM.from_pretrained(base_dir))
    assert trl_grpo_trainer.create_model_from_path is _TRL_BUILD, "the refusal left TRL's build call patched"


@pytest.mark.parametrize(
    ("beta", "peft_config"),
    [(0.0, None), (BETA, LoraConfig(target_modules=["q_proj"], task_type="CAUSAL_LM"))],
    ids=["beta-zero", "peft"],
)
def test_a_passed_reference_the_run_never_reads_is_refused(tmp_path, base_dir, beta, peft_config):
    unread = Qwen3ForCausalLM.from_pretrained(base_dir)
    with pytest.raises(ValueError, match="never be read"):
        _host(
            tmp_path, Qwen3ForCausalLM.from_pretrained(base_dir), beta=beta, ref_model=unread, peft_config=peft_config
        )
    assert trl_grpo_trainer.create_model_from_path is _TRL_BUILD


def test_a_policy_passed_by_name_is_still_built_by_trl(tmp_path, base_dir):
    """TRL builds a named policy through the same call; only the reference build is handed over."""
    reference = _load_reference(base_dir)
    trainer = _host(tmp_path, base_dir, ref_model=reference)
    assert isinstance(trainer.model, Qwen3ForCausalLM) and trainer.model is not reference
    assert trainer.accelerator.unwrap_model(trainer.ref_model) is reference


@pytest.mark.parametrize("trainer_cls", [DistributedGRPOTrainer, DistributedAsyncEnvironmentalGRPOTrainer])
def test_the_replica_warning_names_the_on_policy_way_around_the_reference(trainer_cls, monkeypatch, caplog):
    """Under EP the reference is a dense replica per rank; the warning offers what these trainers honor,
    not DPO's precompute."""
    monkeypatch.setattr(frozen_models, "_REPORTED_REFERENCES", set())
    host = types.SimpleNamespace(
        model=None,
        ref_model=object(),
        parallelism_config=make_parallelism_config(ep_size=8),
        _reference_alternatives=trainer_cls._reference_alternatives,
    )
    with caplog.at_level(logging.WARNING, logger=frozen_models.__name__):
        trainer_cls._validate_held_reference_model(host)
    assert "dense replica" in caplog.text and "beta: 0" in caplog.text, caplog.text
    assert trainer_cls._reference_alternatives != PREFERENCE_REFERENCE_ALTERNATIVES


def test_the_constructor_takes_ref_model_off_the_kwargs_trl_receives():
    """TRL's constructor has no ``ref_model``; the opening half keeps it for the hand-off."""

    class _Spine(OnPolicyGRPOInitMixin):
        def _init_distributed_config(self, kwargs, **_):
            return kwargs

    reference = object()
    host = _Spine()
    _, kwargs = host._begin_on_policy_init((), {"args": None, "ref_model": reference})
    assert "ref_model" not in kwargs
    assert host._supplied_ref_model is reference


@pytest.mark.parametrize("trainer_cls", [DistributedGRPOTrainer, DistributedAsyncEnvironmentalGRPOTrainer])
def test_both_trainers_run_trls_constructor_under_the_hand_off(trainer_cls):
    init = ast.parse(textwrap.dedent(inspect.getsource(trainer_cls.__init__))).body[0]
    wrapped = [
        node
        for node in ast.walk(init)
        if isinstance(node, ast.With)
        and any(
            isinstance(item.context_expr, ast.Call) and call_name(item.context_expr) == "_supplying_kl_reference"
            for item in node.items
        )
    ]
    assert any(
        isinstance(call, ast.Call) and ast.unparse(call.func) == "super().__init__"
        for node in wrapped
        for call in ast.walk(node)
    ), f"{trainer_cls.__name__}.__init__ no longer builds TRL's trainer under _supplying_kl_reference"


@pytest.mark.parametrize("script", ON_POLICY_GRPO_SCRIPTS)
def test_both_scripts_load_the_reference_as_their_policy_and_pass_it(script):
    """The reference load reads the policy's own attention request and sinks policy, and its result is the
    ``ref_model`` the trainer gets; anything else leaves TRL's build to refuse or loads a second model nobody
    reads."""
    (main,) = (function for _, function in functions_in([script], top_level=True) if function.name == "main")
    calls = [node for node in ast.walk(main) if isinstance(node, ast.Call)]
    (load,) = (call for call in calls if call_name(call) == load_reference_model_for_on_policy_grpo.__name__)
    keywords = {keyword.arg: ast.unparse(keyword.value) for keyword in load.keywords}
    (policy_load,) = (call for call in calls if call_name(call) == "load_script_model")
    policy_attn = next(ast.unparse(k.value) for k in policy_load.keywords if k.arg == "attn_implementation")
    assert keywords == {
        "peft_config": "peft_config",
        "reset_sinks": "dist_args.reset_sinks",
        "attn_default": policy_attn,
    }
    assert ast.unparse(load.args[1]) == "model_config", "the reference must load from the configured model"
    passed = [k for call in calls for k in call.keywords if k.arg == "ref_model"]
    assert [ast.unparse(k.value) for k in passed] == ["ref_model"]


def test_the_online_trainer_hands_trl_the_reference_under_its_vendored_client_patch(tmp_path, base_dir, monkeypatch):
    """The real online constructor at ``beta != 0``: TRL builds its server-mode generation through the vendored
    client and holds the supplied reference, and both patches are unwound once construction succeeds."""
    monkeypatch.setattr(online, "VLLMWeightSyncClient", _StubWeightSyncClient)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    reference = _load_reference(base_dir)
    trainer = DistributedGRPOTrainer(
        model=Qwen3ForCausalLM.from_pretrained(base_dir),
        ref_model=reference,
        parallelism_config=ParallelismConfig(),
        **_trainer_kwargs(tmp_path, use_vllm=True, vllm_mode="server"),
    )
    assert isinstance(trainer.vllm_generation.vllm_client, _StubWeightSyncClient)
    assert trainer.accelerator.unwrap_model(trainer.ref_model) is reference
    assert trl_grpo_trainer.create_model_from_path is _TRL_BUILD, "the hand-off left TRL's build call patched"
    assert trl_vllm_generation.VLLMClient is _TRL_VLLM_CLIENT, "the vendored client patch was left installed"


def _trl_constructor_reached(*_, **__):
    raise AssertionError("TRL's constructor ran: on online GRPO it has opened the rollout client by now")


@pytest.mark.parametrize("trainer_cls", [DistributedGRPOTrainer, DistributedAsyncEnvironmentalGRPOTrainer])
@pytest.mark.parametrize("unread", ["beta-zero", "peft-config", "peft-model"])
def test_a_reference_the_run_never_reads_is_refused_before_trls_constructor(
    tmp_path, base_dir, monkeypatch, trainer_cls, unread
):
    """Whether TRL holds a reference is known from the ctor arguments, so the refusal needs no TRL constructor,
    and on online GRPO that constructor connects to the rollout server. The model goes positionally, the slot
    TRL's signature gives it."""
    monkeypatch.setattr(GRPOTrainer, "__init__", _trl_constructor_reached)
    lora = LoraConfig(target_modules=["q_proj"], task_type="CAUSAL_LM")
    model = Qwen3ForCausalLM.from_pretrained(base_dir)
    if unread == "peft-model":
        model = get_peft_model(model, lora)
    kwargs = _trainer_kwargs(
        tmp_path,
        beta=0.0 if unread == "beta-zero" else BETA,
        peft_config=lora if unread == "peft-config" else None,
        use_vllm=True,
        vllm_mode="server",
    )
    with pytest.raises(ValueError, match="never be read"):
        trainer_cls(model, ref_model=object(), parallelism_config=ParallelismConfig(), **kwargs)


@pytest.mark.parametrize("script", ON_POLICY_GRPO_SCRIPTS)
def test_both_scripts_run_the_server_preflights_before_the_reference_load(script):
    """A misconfigured server fails at the cheap probes, not after a second model load."""
    (main,) = (function for _, function in functions_in([script], top_level=True) if function.name == "main")
    first_line = {}
    for node in ast.walk(main):
        if isinstance(node, ast.Call) and (name := call_name(node)) is not None:
            first_line[name] = min(node.lineno, first_line.get(name, node.lineno))
    load = first_line[load_reference_model_for_on_policy_grpo.__name__]
    for preflight in ("verify_context_window_synced", "verify_sampler_logprob_reference_synced"):
        assert first_line[preflight] < load, f"{script} loads the reference before {preflight}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
