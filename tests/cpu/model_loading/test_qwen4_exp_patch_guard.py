#!/usr/bin/env python
"""A Qwen3.8-Flash-Next model built without the family patches is refused at the run policies: it would train its n-gram
table as a parameter and run transformers' per-query QSA loop and [T, T] masks."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
from torch import nn
from transformers import AutoModelForCausalLM

from halo.model_loading.model_preparation import apply_family_patches, apply_run_model_policies
from halo.models.attention.gpt_oss_sinks import SinksPolicy
from halo.models.families.qwen4_exp.forwards import verify_qwen4_exp_patched
from halo.models.families.qwen4_exp.ngram_table import HostNGramTable
from halo.trainers.distributed_trainer import DistributedTrainerMixin
from tests.common.parallelism import make_parallelism_config
from tests.common.qwen4_exp import NGRAM_TABLE_PATH, tiny_qwen4_exp_text_config, transformers_forwards

REMEDY = "apply_family_patches"


def _unpatched_build():
    """The config as ``config.json`` gives it, with no stamp: the class builds the stock table."""
    return AutoModelForCausalLM.from_config(tiny_qwen4_exp_text_config())


def _patched_build():
    config = tiny_qwen4_exp_text_config()
    apply_family_patches(config, "sdpa")
    return AutoModelForCausalLM.from_config(config)


def test_the_premise_a_patched_build_holds_a_host_table_and_an_unpatched_one_a_parameter():
    assert isinstance(_patched_build().get_submodule(NGRAM_TABLE_PATH), HostNGramTable)
    assert _unpatched_build().get_parameter(f"{NGRAM_TABLE_PATH}.weight").requires_grad


def test_an_unstamped_build_is_refused_with_its_table_and_the_remedy():
    with pytest.raises(RuntimeError, match=REMEDY) as refused:
        verify_qwen4_exp_patched(_unpatched_build())
    assert "model.layers.1.ple.ple_embedding" in str(refused.value)


def test_a_patched_build_passes():
    verify_qwen4_exp_patched(_patched_build())


def test_a_model_built_before_the_forwards_were_installed_is_refused():
    model = _patched_build()
    with transformers_forwards(), pytest.raises(RuntimeError, match="forwards are transformers' own"):
        verify_qwen4_exp_patched(model)


def test_the_run_policies_refuse_an_unpatched_model():
    model = _unpatched_build()
    with pytest.raises(RuntimeError, match=REMEDY):
        apply_run_model_policies(model, model.config, sinks_policy=SinksPolicy.NEUTRALIZED, attn_implementation="sdpa")


def test_a_stamped_config_without_split_ngram_parts_is_refused():
    config = tiny_qwen4_exp_text_config()
    apply_family_patches(config, "sdpa")
    object.__setattr__(config, "split_ngram_parts", None)  # the strict config class refuses None on assignment
    with pytest.raises(ValueError, match="sets no split_ngram_parts"):
        AutoModelForCausalLM.from_config(config)


class _PastTheGuard(Exception):
    """Carries control out of the trainer's mode setup once the guard has passed."""


class _ModeSetupHost(DistributedTrainerMixin):
    """The real ``_configure_distributed_modes`` on one rank, stopped right after the Flash-Next guard."""

    _deferred_liger_kernel = False
    _accelerate_manages_fsdp = False
    label_smoother = None
    compute_loss_func = None

    def __init__(self, model):
        self.model = model
        self.parallelism_config = make_parallelism_config(world_size=1, gpus_per_node=1)
        self.args = SimpleNamespace(
            load_best_model_at_end=False,
            optim="adamw_torch",
            per_device_train_batch_size=1,
            eval_strategy="no",
            eval_on_start=False,
        )
        self._bf16_optimizer = False
        self._ep_config = None

    def _cast_peft_params_to_compute_dtype(self):
        raise _PastTheGuard


def _configure_modes(model):
    with patch("halo.trainers.distributed_trainer.agree_rank_uniform_env"):
        DistributedTrainerMixin._configure_distributed_modes(_ModeSetupHost(model))


def test_trainer_construction_refuses_a_model_handed_in_already_built():
    """A model passed to a trainer built skips the loaders' run policies, so trainer construction runs the guard."""
    with pytest.raises(RuntimeError, match=REMEDY):
        _configure_modes(_unpatched_build())


def test_trainer_construction_passes_a_patched_model():
    with pytest.raises(_PastTheGuard):
        _configure_modes(_patched_build())


def test_a_model_without_flash_next_modules_is_not_checked():
    verify_qwen4_exp_patched(nn.Linear(2, 2))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
