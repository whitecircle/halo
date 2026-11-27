#!/usr/bin/env python
"""``reject_lowp_teacher``: ``lowp_precision`` quantizes the distillation student only, so the trainer refuses a teacher
that holds any module :func:`apply_lowp_compute` converted, a dense projection or an EP expert layer. It also refuses a
student passed as a path under ``lowp_precision``, which it would load unquantized."""

from unittest.mock import MagicMock

import pytest
import torch.nn as nn
from transformers import PreTrainedTokenizerBase

from halo.distributed.expert_parallel.moe_layer.layer import EPMoELayerBase
from halo.distributed.parallelism_config import ParallelismConfig
from halo.kernels.lowp.spec import LowpSpec
from halo.model_loading.parallel.lowp_compute import apply_lowp_compute, lowp_converted_modules
from halo.trainers.distillation.teacher_distillation import DistributedTeacherDistillationTrainer, reject_lowp_teacher

WIDTH = 32  # A multiple of every block size: the conversion refuses any other contraction.
PROJECTIONS = ("gate_proj", "up_proj", "down_proj")


class _MLPBlock(nn.Module):
    def __init__(self):
        super().__init__()
        for name in PROJECTIONS:
            setattr(self, name, nn.Linear(WIDTH, WIDTH))


class _Stack(nn.Module):
    def __init__(self, n_layers: int):
        super().__init__()
        self.layers = nn.ModuleList(_MLPBlock() for _ in range(n_layers))


class _EPStub(EPMoELayerBase):
    """A real subclass, since the guard identifies EP layers by ``isinstance``; ``__init__`` needs a wrapped HF layer
    and DeepEP, so it is bypassed for the two attributes ``set_lowp_compute`` reads and writes."""

    def __init__(self):
        nn.Module.__init__(self)
        self._lowp = None
        self._use_grouped_mm = True

    def forward(self, hidden_states):  # Abstract on the base; never called here.
        raise NotImplementedError


def test_a_bf16_teacher_passes():
    teacher = _Stack(2)
    assert lowp_converted_modules(teacher) == []
    reject_lowp_teacher(teacher)


@pytest.mark.parametrize("precision", ["fp8", "fp4", "mxfp4"])
def test_a_teacher_converted_by_apply_lowp_compute_is_refused(precision):
    teacher = _Stack(2)
    apply_lowp_compute(teacher, LowpSpec(precision=precision), expert_weight_cacheable=True)
    converted = lowp_converted_modules(teacher)
    assert converted == [f"layers.{i}.{proj}" for i in range(2) for proj in PROJECTIONS]
    with pytest.raises(ValueError, match=r"6 low-precision module\(s\) \(layers\.0\.gate_proj, .* and 3 more\)"):
        reject_lowp_teacher(teacher)


def test_one_converted_block_is_enough():
    """The guard refuses any converted module, not only a fully converted teacher."""
    teacher = _Stack(2)
    apply_lowp_compute(teacher, LowpSpec(precision="mxfp4", keep_last_blocks=1), expert_weight_cacheable=True)
    assert lowp_converted_modules(teacher) == [f"layers.0.{proj}" for proj in PROJECTIONS]
    with pytest.raises(ValueError, match=r"3 low-precision module\(s\) \(layers\.0\.gate_proj, "):
        reject_lowp_teacher(teacher)


def test_an_expert_layer_carrying_a_format_is_refused():
    teacher = nn.ModuleDict({"moe": _EPStub()})
    reject_lowp_teacher(teacher)  # The layer is EP but carries no format: a bf16 teacher.

    teacher["moe"].set_lowp_compute(LowpSpec(precision="fp4"), weight_cacheable=True)
    assert lowp_converted_modules(teacher) == ["moe"]
    with pytest.raises(ValueError, match=r"1 low-precision module\(s\) \(moe\)"):
        reject_lowp_teacher(teacher)


def test_a_loop_path_expert_layer_is_refused_too():
    """``set_lowp_compute`` returns False on the per-expert loop, which never quantizes, but the layer still carries
    the format a later switch to the grouped path would apply."""
    layer = _EPStub()
    layer._use_grouped_mm = False
    assert layer.set_lowp_compute(LowpSpec(precision="mxfp4"), weight_cacheable=True) is False
    with pytest.raises(ValueError, match=r"\(moe\)"):
        reject_lowp_teacher(nn.ModuleDict({"moe": layer}))


def test_the_trainer_constructor_refuses_a_converted_teacher():
    """The guard runs in the constructor, before the student is resolved, so a teacher built outside the script is
    refused too, ahead of any distributed setup."""
    teacher = _Stack(1)
    apply_lowp_compute(teacher, LowpSpec(precision="fp4"), expert_weight_cacheable=True)
    with pytest.raises(ValueError, match=r"The teacher holds 3 low-precision module\(s\)"):
        DistributedTeacherDistillationTrainer(
            student_model=nn.Linear(2, 2),
            teacher_model=teacher,
            teacher_tokenizer=MagicMock(spec=PreTrainedTokenizerBase),
            processing_class=MagicMock(spec=PreTrainedTokenizerBase),
        )


@pytest.mark.parametrize("precision", ["fp8", "fp4", "mxfp4"])
def test_a_student_path_under_lowp_is_refused_before_it_loads(precision):
    """The trainer loads a path through ``resolve_trainer_model``, which applies no lowp, so the student would train
    unquantized. The refusal comes before the load: the path does not exist."""
    with pytest.raises(ValueError, match=rf"student_model is a path .* lowp_precision='{precision}'"):
        DistributedTeacherDistillationTrainer(
            student_model="/nonexistent/student",
            teacher_model=_Stack(1),
            teacher_tokenizer=MagicMock(spec=PreTrainedTokenizerBase),
            processing_class=MagicMock(spec=PreTrainedTokenizerBase),
            parallelism_config=ParallelismConfig(lowp=LowpSpec(precision=precision)),
        )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
