#!/usr/bin/env python
"""``verify_lowp_conversion``: at trainer setup a model holds low-precision modules exactly when ``lowp_precision`` is
on. A model handed to a trainer, or built from an id, never passed the loader's lowp step, so without the check it
trains unquantized under a lowp config, or fake-quantized under a bf16 one."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch.nn as nn

from halo.distributed.expert_parallel.moe_layer.layer import EPMoELayerBase
from halo.kernels.lowp.spec import LowpSpec
from halo.model_loading.parallel.lowp_compute import apply_lowp_compute
from halo.trainers.construction.trainer_rules import verify_lowp_conversion
from halo.trainers.distillation.teacher_distillation import DistributedTeacherDistillationTrainer
from halo.trainers.distributed_trainer import DistributedTrainerMixin
from halo.trainers.sft import DistributedSFTTrainer
from tests.common.parallelism import make_parallelism_config

WIDTH = 32  # A multiple of every block size: the conversion refuses any other contraction.
PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
PRECISIONS = ("fp8", "fp4", "mxfp4")
UNCONVERTED = r"lowp_precision='{}', but the model holds no low-precision module"
CONVERTED_UNDER_BF16 = r"The model holds {} low-precision module\(s\) \({}\), but lowp_precision='bf16'"


class _MLPBlock(nn.Module):
    def __init__(self):
        super().__init__()
        for name in PROJECTIONS:
            setattr(self, name, nn.Linear(WIDTH, WIDTH))


class _Stack(nn.Module):
    def __init__(self, n_layers: int = 2):
        super().__init__()
        self.layers = nn.ModuleList(_MLPBlock() for _ in range(n_layers))


class _EPStub(EPMoELayerBase):
    """A real subclass, since the check identifies EP layers by ``isinstance``; ``__init__`` needs a wrapped HF layer
    and DeepEP, so it is bypassed for the two attributes ``set_lowp_compute`` reads and writes."""

    def __init__(self):
        nn.Module.__init__(self)
        self._lowp_quant_format = None
        self._use_grouped_mm = True

    def forward(self, hidden_states):  # Abstract on the base; never called here.
        raise NotImplementedError


def _converted(precision: str, n_layers: int = 2) -> _Stack:
    model = _Stack(n_layers)
    apply_lowp_compute(model, LowpSpec(precision=precision), expert_weight_cacheable=True)
    return model


@pytest.mark.parametrize("precision", PRECISIONS)
def test_an_unconverted_model_under_lowp_is_refused(precision):
    with pytest.raises(ValueError, match=UNCONVERTED.format(precision)):
        verify_lowp_conversion(_Stack(), LowpSpec(precision=precision))


@pytest.mark.parametrize("precision", PRECISIONS)
def test_a_model_the_loader_converted_passes(precision):
    verify_lowp_conversion(_converted(precision), LowpSpec(precision=precision))


def test_a_converted_model_under_bf16_is_refused():
    with pytest.raises(ValueError, match=CONVERTED_UNDER_BF16.format(6, r"layers\.0\.gate_proj, .* and 3 more")):
        verify_lowp_conversion(_converted("mxfp4"), LowpSpec(precision="bf16"))


def test_a_bf16_model_under_bf16_passes():
    verify_lowp_conversion(_Stack(), LowpSpec(precision="bf16"))


def test_an_expert_layer_carrying_a_format_counts_in_both_directions():
    model = nn.ModuleDict({"moe": _EPStub()})
    verify_lowp_conversion(model, LowpSpec(precision="bf16"))  # EP, but no format: a bf16 model.
    with pytest.raises(ValueError, match=UNCONVERTED.format("fp4")):
        verify_lowp_conversion(model, LowpSpec(precision="fp4"))

    model["moe"].set_lowp_compute("nvfp4", weight_cacheable=True)
    verify_lowp_conversion(model, LowpSpec(precision="fp4"))
    with pytest.raises(ValueError, match=CONVERTED_UNDER_BF16.format(1, "moe")):
        verify_lowp_conversion(model, LowpSpec(precision="bf16"))


class _PastTheCheck(Exception):
    """Carries control out of the trainer's mode setup once the check has passed."""


class _ModeSetupHost(DistributedTrainerMixin):
    """The real ``_configure_distributed_modes`` on one rank, stopped at the PEFT cast, past the check."""

    _deferred_liger_kernel = False
    _accelerate_manages_fsdp = False
    label_smoother = None
    compute_loss_func = None

    def __init__(self, model, precision: str):
        self.model = model
        self.parallelism_config = make_parallelism_config(
            world_size=1, gpus_per_node=1, lowp=LowpSpec(precision=precision)
        )
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
        raise _PastTheCheck


def _configure_modes(model, precision: str):
    with patch("halo.trainers.distributed_trainer.agree_rank_uniform_env"):
        DistributedTrainerMixin._configure_distributed_modes(_ModeSetupHost(model, precision))


def test_trainer_setup_refuses_an_unconverted_model_under_lowp():
    """The SFT trainer loads a model id through ``resolve_trainer_model``, and any trainer takes a module as is: neither
    applies lowp, so the shared setup is where a mismatch is caught."""
    with pytest.raises(ValueError, match=UNCONVERTED.format("mxfp4")):
        _configure_modes(_Stack(), "mxfp4")


def test_trainer_setup_refuses_a_converted_model_under_bf16():
    with pytest.raises(ValueError, match=CONVERTED_UNDER_BF16.format(3, r"layers\.0\.gate_proj, .*down_proj")):
        _configure_modes(_converted("fp4", n_layers=1), "bf16")


@pytest.mark.parametrize(("precision", "converted"), [("fp4", True), ("bf16", False)])
def test_trainer_setup_passes_a_model_that_matches_its_config(precision, converted):
    model = _converted(precision) if converted else _Stack()
    with pytest.raises(_PastTheCheck):
        _configure_modes(model, precision)


@pytest.mark.parametrize("trainer_cls", [DistributedSFTTrainer, DistributedTeacherDistillationTrainer])
def test_both_lowp_trainers_run_the_shared_setup(trainer_cls):
    """The two trainers that accept lowp_precision inherit the setup that runs the check."""
    assert trainer_cls.SUPPORTS_LOWP_PRECISION
    assert trainer_cls._configure_distributed_modes is DistributedTrainerMixin._configure_distributed_modes


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
