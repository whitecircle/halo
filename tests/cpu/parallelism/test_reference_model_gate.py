#!/usr/bin/env python
"""The EP/TP rule on a dense frozen reference: never parallelized, so it costs a whole replica, reported once. Only
PP refuses a live reference, since no stage holds the whole model."""

import logging
import sys
import types
from unittest import mock

import pytest
from transformers import Qwen3Config, Qwen3ForCausalLM
from trl import SFTTrainer

import halo.model_loading.parallel.frozen_models as frozen_models
import halo.trainers.distillation.self_distillation as self_distillation
from halo.configs.script_arguments import ModelArguments
from halo.entrypoints import model_setup
from halo.entrypoints.model_setup import PolicyLoad
from halo.model_loading.parallel.frozen_models import (
    ON_POLICY_GRPO_REFERENCE_ALTERNATIVES,
    PREFERENCE_REFERENCE_ALTERNATIVES,
    load_reference_model_for_on_policy_grpo,
    load_reference_model_for_preference,
)
from halo.trainers.distillation.self_distillation import DistributedSelfDistillationTrainer
from halo.trainers.distributed_trainer import DistributedTrainerMixin
from halo.trainers.grpo.online import DistributedOnlineGRPOTrainer
from halo.trainers.sft import DistributedSFTTrainer
from tests.common.parallelism import make_parallelism_config
from tests.common.repo import load_script_module


@pytest.fixture(autouse=True)
def _fresh_reports(monkeypatch):
    """Re-arms the once-per-message report."""
    monkeypatch.setattr(frozen_models, "_reported_references", set())


SHARDED = {
    "ep": {"ep_size": 8},
    "etp": {"expert_tp_size": 2},
    "tp": {"tp_size": 2},
}
PREFIX = "A frozen, unparallelized reference beside a policy sharded by EP/ETP/TP"


def _config(**axes):
    """A pipeline split takes two 4-GPU nodes, so each stage fills an NVLink domain."""
    return make_parallelism_config(world_size=8, gpus_per_node=4 if "pp_size" in axes else 8, **axes)


def _sizes(config) -> str:
    return (
        f"expert_parallel_size={config.ep_size}, expert_tensor_parallel_size={config.expert_tp_size}, "
        f"tensor_parallel_size={config.tp_size}"
    )


def _on_policy_grpo_gate(config, ref_model):
    host = types.SimpleNamespace(
        model=None,
        ref_model=ref_model,
        parallelism_config=config,
        REFERENCE_ALTERNATIVES=DistributedOnlineGRPOTrainer.REFERENCE_ALTERNATIVES,
    )
    DistributedOnlineGRPOTrainer._validate_held_reference_model(host)


def _preference_gate(config, ref_model):
    host = types.SimpleNamespace(
        parallelism_config=config, REFERENCE_ALTERNATIVES=DistributedTrainerMixin.REFERENCE_ALTERNATIVES
    )
    DistributedTrainerMixin._validate_reference_model(host, ref_model)


def _self_distillation(config, reference_kl_coef):
    """The HF/TRL machinery stubbed out; returns the references it set up."""
    setup_calls = []

    def init_config(self, kwargs, **_):
        self.parallelism_config = config
        return kwargs

    with (
        mock.patch.object(DistributedSFTTrainer, "_init_distributed_config", init_config),
        mock.patch.object(
            SFTTrainer,
            "__init__",
            lambda self, *a, **k: vars(self).update(data_collator=None, model=None, processing_class=None),
        ),
        mock.patch.object(DistributedSelfDistillationTrainer, "_configure_distributed_modes"),
        mock.patch.object(DistributedSelfDistillationTrainer, "_resolve_stop_token_ids"),
        mock.patch.object(
            self_distillation, "place_frozen_scorer", lambda *a: setup_calls.append(True) or (None, None)
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
    with caplog.at_level(logging.WARNING):
        _preference_gate(_config(), object())
        _preference_gate(_config(ep_size=8), None)
        assert _self_distillation(_config(), 0.1) == [True]
        assert _self_distillation(_config(ep_size=8), 0.0) == []
    assert _warnings(caplog) == []


class _LoadedReference:
    """Identity is the assertion."""


def _preference_reference(config, *, precompute=False):
    loaded = _LoadedReference()
    with mock.patch.object(frozen_models, "load_frozen_reference_model", return_value=loaded):
        result = load_reference_model_for_preference(
            types.SimpleNamespace(),
            ModelArguments(model_name_or_path="org/policy"),
            types.SimpleNamespace(precompute_ref_log_probs=precompute),
            config,
            tokenizer=None,
            is_vlm=False,
            method="DPO",
            reset_sinks=True,
            attn_default=None,
        )
    return result, loaded


def _on_policy_grpo_reference(config):
    loaded = _LoadedReference()
    with (
        mock.patch.object(frozen_models, "load_frozen_reference_model", return_value=loaded),
        mock.patch.object(frozen_models, "estimate_reference_bytes", return_value=1),
    ):
        result = load_reference_model_for_on_policy_grpo(
            types.SimpleNamespace(),
            ModelArguments(model_name_or_path="org/policy"),
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
    """Precompute and PEFT are named."""
    config = _config(**axes)
    with caplog.at_level(logging.WARNING):
        result, loaded = _preference_reference(config)
    assert result is loaded
    (message,) = _warnings(caplog)
    assert message.startswith(PREFIX) and message.endswith(PREFERENCE_REFERENCE_ALTERNATIVES.under(config))


def test_the_preference_loader_refuses_a_live_reference_under_pp():
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
    [lambda config: _preference_gate(config, object()), lambda config: _on_policy_grpo_gate(config, object())],
    ids=["preference", "on-policy-grpo"],
)
@pytest.mark.parametrize("axes", SHARDED.values(), ids=list(SHARDED))
def test_the_named_way_around_offers_peft_only_where_adapters_train(report, axes, caplog):
    """TP refuses every LoRA shape."""
    config = _config(**axes)
    with caplog.at_level(logging.WARNING):
        report(config)
    (message,) = _warnings(caplog)
    assert ("use_peft" in message) is not config.is_tp_mode, message


def test_the_script_loader_and_the_trainer_gate_report_one_reference_once(caplog):
    config = _config(ep_size=8)
    with caplog.at_level(logging.WARNING):
        result, _ = _preference_reference(config)
        _preference_gate(config, result)
    assert len(_warnings(caplog)) == 1


def test_the_on_policy_grpo_loader_is_silent_and_the_trainer_gate_reports_the_reference_once(caplog):
    config = _config(ep_size=8)
    with caplog.at_level(logging.WARNING):
        result, loaded = _on_policy_grpo_reference(config)
        assert _warnings(caplog) == []
        _on_policy_grpo_gate(config, result)
    assert result is loaded
    (message,) = _warnings(caplog)
    assert message.startswith(PREFIX) and message.endswith(ON_POLICY_GRPO_REFERENCE_ALTERNATIVES.under(config))


class _TrainerReached(Exception):
    """The script got as far as constructing the trainer."""


def test_the_dpo_script_hands_the_loaded_reference_to_the_trainer(tmp_path):
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
        "resolve_preference_vlm_run": mock.Mock(return_value=False),
        "apply_text_path_padding_side": mock.Mock(),
        "prepare_script_preference_data": mock.Mock(return_value=(None, None, None)),
        "configure_balancing_and_build_callbacks": mock.Mock(return_value=[]),
        "barrier": mock.Mock(),
        "DistributedDPOTrainer": trainer,
    }
    # Shared with KTO; the policy half is stubbed where it runs and the reference half is real down to the frozen load.
    shared_stubs = {
        "load_policy_model": mock.Mock(return_value=PolicyLoad(object(), tokenizer, tokenizer, False)),
        "configure_adapter_training": mock.Mock(return_value=None),
        "apply_max_length": lambda cfg, args, model, tok: tok,
        "install_resolved_tokenizer": lambda processing_class, tok: processing_class,
        "log_model_info": mock.Mock(),
    }
    with (
        mock.patch("halo.entrypoints.script.install_log_tee"),
        mock.patch.object(sys, "argv", ["prog", str(config)]),
        mock.patch.object(frozen_models, "load_frozen_reference_model", return_value=loaded),
        mock.patch.multiple(module, **stubs),
        mock.patch.multiple(model_setup, **shared_stubs),
        pytest.raises(_TrainerReached),
    ):
        module.main()
    assert received["ref_model"] is loaded


# Sizes of the unsharded reference of a model that fits no GPU, and one that fits any.
HUGE_REFERENCE_BYTES = 800 * 10**9
GPU_FREE_BYTES = 270 * 10**9
HOST_AVAILABLE_BYTES = 2000 * 10**9


def _sized_on_policy_grpo_reference(config, reference_bytes, *, gpu_free=GPU_FREE_BYTES, host=HOST_AVAILABLE_BYTES):
    loaded = _LoadedReference()
    with (
        mock.patch.object(frozen_models, "load_frozen_reference_model", return_value=loaded) as load,
        mock.patch.object(frozen_models, "estimate_reference_bytes", return_value=reference_bytes),
        mock.patch.object(frozen_models.torch.cuda, "is_available", return_value=gpu_free is not None),
        mock.patch.object(frozen_models, "current_device", return_value=0),
        mock.patch.object(frozen_models.torch.cuda, "mem_get_info", return_value=(gpu_free, gpu_free)),
        mock.patch.object(frozen_models, "available_host_ram_bytes", return_value=host),
        mock.patch.object(frozen_models, "get_local_world_size", return_value=8),
    ):
        try:
            result = load_reference_model_for_on_policy_grpo(
                types.SimpleNamespace(),
                ModelArguments(model_name_or_path="org/policy"),
                types.SimpleNamespace(beta=0.1),
                config,
                None,
                peft_config=None,
                reset_sinks=True,
                attn_default=None,
            )
        finally:
            loaded_calls = load.call_count
    return result, loaded_calls


@pytest.mark.parametrize("axes", [{}, SHARDED["ep"], SHARDED["etp"], SHARDED["tp"]], ids=["fsdp2", "ep", "etp", "tp"])
def test_a_reference_larger_than_the_gpu_share_is_refused_before_it_loads(axes):
    """A plain FSDP2 run is covered too: the reference is never sharded, whatever shards the policy."""
    config = _config(**axes)
    with pytest.raises(ValueError) as excinfo:
        _sized_on_policy_grpo_reference(config, HUGE_REFERENCE_BYTES)
    message = str(excinfo.value)
    assert "800 GB on every GPU" in message
    assert "beta: 0" in message and "Offline GRPO" in message
    assert ("use_peft" in message) is not config.is_tp_mode


def test_a_refused_reference_never_reaches_the_loader():
    loaded = _LoadedReference()
    with mock.patch.object(frozen_models, "load_frozen_reference_model", return_value=loaded) as load:
        with pytest.raises(ValueError, match="800 GB on every GPU"):
            with (
                mock.patch.object(frozen_models, "estimate_reference_bytes", return_value=HUGE_REFERENCE_BYTES),
                mock.patch.object(frozen_models.torch.cuda, "is_available", return_value=True),
                mock.patch.object(frozen_models, "current_device", return_value=0),
                mock.patch.object(frozen_models.torch.cuda, "mem_get_info", return_value=(GPU_FREE_BYTES,) * 2),
            ):
                load_reference_model_for_on_policy_grpo(
                    types.SimpleNamespace(),
                    ModelArguments(model_name_or_path="org/policy"),
                    types.SimpleNamespace(beta=0.1),
                    _config(),
                    None,
                    peft_config=None,
                    reset_sinks=True,
                    attn_default=None,
                )
    load.assert_not_called()


def test_a_reference_the_gpus_hold_but_the_node_host_cannot_stage_is_refused():
    """Every rank of a node stages its own copy on the host before the move."""
    with pytest.raises(ValueError, match=r"8 ranks of a node each stage ~100 GB"):
        _sized_on_policy_grpo_reference(_config(), 100 * 10**9, host=HOST_AVAILABLE_BYTES // 3)


def test_a_reference_that_fits_loads_unchanged():
    result, load_calls = _sized_on_policy_grpo_reference(_config(ep_size=8), 8 * 10**9)
    assert isinstance(result, _LoadedReference) and load_calls == 1


@pytest.mark.parametrize(
    ("kwargs", "fits"),
    [
        ({"reference_bytes": 100, "free_device_bytes": 200, "available_host_bytes": 10**6}, True),
        ({"reference_bytes": 101, "free_device_bytes": 200, "available_host_bytes": 10**6}, False),
        ({"reference_bytes": 100, "free_device_bytes": None, "available_host_bytes": 10**6}, True),
        ({"reference_bytes": 100, "free_device_bytes": None, "available_host_bytes": 900}, False),
        ({"reference_bytes": 100, "free_device_bytes": None, "available_host_bytes": None}, True),
    ],
    ids=["device-at-limit", "device-over", "device-unreadable", "host-over", "nothing-readable"],
)
def test_the_misfit_boundaries(kwargs, fits):
    misfit = frozen_models.resolve_reference_misfit(**kwargs, ranks_per_node=8)
    assert (misfit is None) is fits


def test_the_estimate_is_the_run_dtype_parameter_bytes_of_a_config_only_shell(tmp_path):
    config = Qwen3Config(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
    )
    config.save_pretrained(tmp_path)
    expected = sum(p.numel() for p in Qwen3ForCausalLM(config).parameters())
    training_config = types.SimpleNamespace(bf16=True, fp16=False)
    model_config = ModelArguments(model_name_or_path=str(tmp_path), trust_remote_code=False)
    assert frozen_models.estimate_reference_bytes(model_config, training_config) == 2 * expected
    assert frozen_models.estimate_reference_bytes(model_config, types.SimpleNamespace(bf16=False, fp16=False)) == (
        4 * expected
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
