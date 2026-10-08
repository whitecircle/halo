#!/usr/bin/env python
"""The one EP/TP rule on a dense frozen reference, reached through each caller's own way around it.

A reference is never parallelized, so beside a policy sharded by EP, ETP or TP every rank holds a whole
dense replica, experts included. Its log-probs still match the policy's up to kernel numerics (the
model's own MoE forward is what the EP layers are tested to reproduce), so this is a cost, reported once
by ``warn_unparallelized_reference``: the DPO/KTO script loader, the DPO/KTO trainer gate
(``_validate_reference_model``), the on-policy GRPO script loader and trainer gate, and
self-distillation (its constructor). Only PP refuses a live reference, since no pipeline stage holds
the whole model.

Run: pytest tests/cpu/parallelism/test_reference_model_gate.py
"""

import logging
import sys
import types
from unittest import mock

import pytest
from accelerate import PartialState
from trl import ModelConfig, SFTTrainer

import src.distributed.loading.frozen_models as frozen_models
from src.distributed.loading.frozen_models import (
    ON_POLICY_GRPO_REFERENCE_ALTERNATIVES,
    PREFERENCE_REFERENCE_ALTERNATIVES,
    load_reference_model_for_on_policy_grpo,
    load_reference_model_for_preference,
)
from src.trainers.distillation.self_distillation import DistributedSelfDistillationTrainer
from src.trainers.grpo.online import DistributedGRPOTrainer
from src.trainers.mixins.validation import ParallelismValidationMixin
from src.trainers.sft import DistributedSFTTrainer
from tests.common.parallelism import make_parallelism_config
from tests.common.utils import load_script_module

PartialState()  # the gate warns through accelerate's logger, which requires an initialized state


@pytest.fixture(autouse=True)
def _fresh_reports(monkeypatch):
    """Each test re-arms the once-per-message report."""
    monkeypatch.setattr(frozen_models, "_REPORTED_REFERENCES", set())


SHARDED = {
    "ep": {"ep_size": 8},
    "etp": {"expert_tp_size": 2},
    "tp": {"tp_size": 2},
}
PREFIX = "A frozen, unparallelized reference beside a policy sharded by EP/ETP/TP"


def _config(**axes):
    """An 8-rank world; a pipeline split takes two 4-GPU nodes, so each stage fills an NVLink domain."""
    return make_parallelism_config(world_size=8, gpus_per_node=4 if "pp_size" in axes else 8, **axes)


def _sizes(config) -> str:
    return (
        f"expert_parallel_size={config.ep_size}, expert_tensor_parallel_size={config.expert_tp_size}, "
        f"tensor_parallel_size={config.tp_size}"
    )


def _preference_gate(config, ref_model):
    host = types.SimpleNamespace(
        parallelism_config=config, _reference_alternatives=ParallelismValidationMixin._reference_alternatives
    )
    ParallelismValidationMixin._validate_reference_model(host, ref_model)


def _self_distillation(config, reference_kl_coef):
    """The real constructor, the HF/TRL machinery stubbed out; returns the references it set up."""
    setup_calls = []

    def init_config(self, kwargs, **_):
        self.parallelism_config = config
        return kwargs

    with (
        mock.patch.object(DistributedSFTTrainer, "_init_distributed_config", init_config),
        mock.patch.object(SFTTrainer, "__init__", lambda self, *a, **k: setattr(self, "data_collator", None)),
        mock.patch.object(DistributedSelfDistillationTrainer, "_setup_distributed_modes"),
        mock.patch.object(DistributedSelfDistillationTrainer, "_resolve_stop_token_ids"),
        mock.patch.object(
            DistributedSelfDistillationTrainer, "_setup_reference_model", lambda self: setup_calls.append(True)
        ),
    ):
        DistributedSelfDistillationTrainer(
            reference_model=object(),
            reference_kl_coef=reference_kl_coef,
            reference_kl_loss="unnormalized_kl",
            confidence_weight_opd=True,
            opd_exclude_eos=True,
            sdpg_beta_base=0.0,
        )
    return setup_calls


def _warnings(caplog) -> list[str]:
    return [record.getMessage() for record in caplog.records if record.levelno == logging.WARNING]


@pytest.mark.parametrize("axes", SHARDED.values(), ids=list(SHARDED))
def test_the_preference_gate_reports_the_replica_and_its_way_around(axes, caplog):
    config = _config(**axes)
    with caplog.at_level(logging.WARNING):
        _preference_gate(config, object())
    (message,) = _warnings(caplog)
    assert message.startswith(PREFIX) and _sizes(config) in message
    assert message.endswith(PREFERENCE_REFERENCE_ALTERNATIVES.under(config))


@pytest.mark.parametrize("axes", SHARDED.values(), ids=list(SHARDED))
def test_self_distillation_sets_the_anchor_up_and_reports_it(axes, caplog):
    config = _config(**axes)
    with caplog.at_level(logging.WARNING):
        assert _self_distillation(config, 0.1) == [True]
    (message,) = _warnings(caplog)
    assert message.startswith(PREFIX) and _sizes(config) in message
    assert message.endswith("reference_kl_coef: 0 loads no reference.")


def test_nothing_is_reported_where_no_replica_is_sharded_away(caplog):
    """A reference beside an unsharded policy costs what the policy does; no reference, or an
    unweighted anchor (never set up), is none."""
    with caplog.at_level(logging.WARNING):
        _preference_gate(_config(), object())
        _preference_gate(_config(ep_size=8), None)
        assert _self_distillation(_config(), 0.1) == [True]
        assert _self_distillation(_config(ep_size=8), 0.0) == []
    assert _warnings(caplog) == []


class _LoadedReference:
    """What the stubbed frozen-reference load returns: identity is the assertion."""


def _preference_reference(config, *, precompute=False):
    """The DPO/KTO full-finetune loader with the frozen load stubbed; returns ``(result, loaded)``."""
    loaded = _LoadedReference()
    with mock.patch.object(frozen_models, "load_frozen_reference_model", return_value=loaded):
        result = load_reference_model_for_preference(
            types.SimpleNamespace(),
            ModelConfig(model_name_or_path="org/policy"),
            types.SimpleNamespace(precompute_ref_log_probs=precompute),
            config,
            tokenizer=None,
            is_vlm=False,
            method="DPO",
        )
    return result, loaded


def _on_policy_grpo_reference(config):
    """The on-policy GRPO script loader for a full fine-tune at ``beta != 0``, the frozen load stubbed;
    returns ``(result, loaded)``."""
    loaded = _LoadedReference()
    with mock.patch.object(frozen_models, "load_frozen_reference_model", return_value=loaded):
        result = load_reference_model_for_on_policy_grpo(
            types.SimpleNamespace(),
            ModelConfig(model_name_or_path="org/policy"),
            types.SimpleNamespace(beta=0.1),
            config,
            None,
            peft_config=None,
            reset_sinks=True,
            attn_default=None,
        )
    return result, loaded


@pytest.mark.parametrize("axes", [SHARDED["ep"], SHARDED["tp"]], ids=["ep", "tp"])
def test_the_preference_loader_reports_and_loads_a_dense_reference_under_ep_and_tp(axes, caplog):
    """The replica is correct, so it loads; precompute (exact and cheaper) and PEFT are named."""
    config = _config(**axes)
    with caplog.at_level(logging.WARNING):
        result, loaded = _preference_reference(config)
    assert result is loaded
    (message,) = _warnings(caplog)
    assert message.startswith(PREFIX) and message.endswith(PREFERENCE_REFERENCE_ALTERNATIVES.under(config))


def test_the_preference_loader_refuses_a_live_reference_under_pp():
    """No pipeline stage holds the whole model, so no stage can score a live reference."""
    with pytest.raises(ValueError, match="under PP needs precompute_ref_log_probs: true"):
        _preference_reference(_config(pp_size=2))


@pytest.mark.parametrize("axes", [SHARDED["ep"], SHARDED["tp"], {"pp_size": 2}], ids=["ep", "tp", "pp"])
def test_precomputed_reference_log_probs_load_no_reference(axes, caplog):
    with caplog.at_level(logging.WARNING):
        result, _ = _preference_reference(_config(**axes), precompute=True)
    assert result is None
    assert _warnings(caplog) == []


@pytest.mark.parametrize(
    "report",
    [lambda config: _preference_gate(config, object()), _on_policy_grpo_reference],
    ids=["preference", "on-policy-grpo"],
)
@pytest.mark.parametrize("axes", SHARDED.values(), ids=list(SHARDED))
def test_the_named_way_around_offers_peft_only_where_adapters_train(report, axes, caplog):
    """TP refuses every LoRA shape, so under it the warning names only the PEFT-free way around the replica."""
    config = _config(**axes)
    with caplog.at_level(logging.WARNING):
        report(config)
    (message,) = _warnings(caplog)
    assert ("use_peft" in message) is not config.is_tp_mode, message


def test_the_script_loader_and_the_trainer_gate_report_one_reference_once(caplog):
    """The DPO script loads the reference and hands it to the trainer, whose gate sees it again."""
    config = _config(ep_size=8)
    with caplog.at_level(logging.WARNING):
        result, _ = _preference_reference(config)
        _preference_gate(config, result)
    assert len(_warnings(caplog)) == 1


def test_the_on_policy_grpo_loader_and_trainer_gate_report_one_reference_once(caplog):
    """The GRPO scripts load the reference and the trainer's gate sees it again, naming the GRPO way around it."""
    config = _config(ep_size=8)
    with caplog.at_level(logging.WARNING):
        result, loaded = _on_policy_grpo_reference(config)
        host = types.SimpleNamespace(
            model=None,
            ref_model=result,
            parallelism_config=config,
            _reference_alternatives=DistributedGRPOTrainer._reference_alternatives,
        )
        DistributedGRPOTrainer._validate_held_reference_model(host)
    assert result is loaded
    (message,) = _warnings(caplog)
    assert message.startswith(PREFIX) and message.endswith(ON_POLICY_GRPO_REFERENCE_ALTERNATIVES.under(config))


class _TrainerReached(Exception):
    """Raised by the stubbed trainer: the script got as far as constructing it."""


def test_the_dpo_script_hands_the_loaded_reference_to_the_trainer(tmp_path):
    """Under EP the real loader loads the reference, and the trainer receives that very object."""
    module = load_script_module("scripts/training/preference/dpo.py", "halo_test_dpo_reference_handoff")
    runtime = types.SimpleNamespace(
        parallelism_config=_config(ep_size=8),
        model_source="stub/policy",
        mode_suffix="",
        local_rank=0,
        resume_checkpoint=None,
        policy_from_checkpoint=False,
    )
    tokenizer = types.SimpleNamespace(padding_side="right")
    received = {}

    def trainer(**kwargs):
        received.update(kwargs)
        raise _TrainerReached

    loaded = _LoadedReference()
    config = tmp_path / "config.yaml"
    config.write_text(
        f"model_name_or_path: stub/policy\ndataset:\n- dummy/dataset\noutput_dir: {tmp_path / 'out'}\n"
        f"bf16: false\nuse_cpu: true\nmax_length: 512\n"
    )
    stubs = {
        "init_training_script": mock.Mock(return_value=runtime),
        "load_script_datasets": mock.Mock(return_value=(None, False)),
        "alias_images_column": lambda ds, *a, **k: ds,
        "resolve_vlm_run": mock.Mock(return_value=False),
        "load_model_for_training": mock.Mock(return_value=(object(), tokenizer, tokenizer, False)),
        "setup_peft_model": mock.Mock(return_value=None),
        "apply_max_length": lambda cfg, args, model, tok: tok,
        "install_resolved_tokenizer": lambda processing_class, tok: processing_class,
        "log_model_info": mock.Mock(),
        "enforce_text_path_padding_side": mock.Mock(),
        "prepare_script_preference_data": mock.Mock(return_value=(None, None, None)),
        "apply_distributed_trainer_config": mock.Mock(),
        "build_training_callbacks": mock.Mock(return_value=[]),
        "barrier": mock.Mock(),
        "DistributedDPOTrainer": trainer,
    }
    with (
        mock.patch("src.training.parser.install_log_tee"),
        mock.patch.object(sys, "argv", ["prog", str(config)]),
        mock.patch.object(frozen_models, "load_frozen_reference_model", return_value=loaded),
        mock.patch.multiple(module, **stubs),
        pytest.raises(_TrainerReached),
    ):
        module.main()
    assert received["ref_model"] is loaded


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
