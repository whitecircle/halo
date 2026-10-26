"""Hand-built pipeline stages and a stand-in torch schedule for the CPU pipeline tests."""

import torch
import torch.nn as nn

from src.distributed.pipeline_parallel.runner import PipelineRunner
from src.distributed.pipeline_parallel.stage import PipelineStageModule, StageLayout
from src.distributed.pipeline_parallel.stage_losses import TokenLoss

# The layout of a ``*ForCausalLM`` whose backbone sits under ``model`` and its head under ``lm_head``.
CAUSAL_LM_LAYOUT = StageLayout(
    backbone_prefix="model",
    layer_attr="layers",
    head_attr="lm_head",
    post_layer_paths=("model.norm",),
    input_paths=("model.embed_tokens",),
)


class _StageSource(nn.Module):
    """The unsplit model a hand-built stage is cut from: no config, so no checkpoint metadata."""

    def __init__(self, backbone: nn.Module, head: nn.Module | None):
        super().__init__()
        self.model = backbone
        self.lm_head = head

    def can_generate(self) -> bool:
        return False


def hand_built_stage(
    backbone: nn.Module, head: nn.Module | None, *, is_first: bool, is_last: bool, layer_offset: int
) -> PipelineStageModule:
    """A stage over ``backbone`` (and ``head`` on the last stage) in the :data:`CAUSAL_LM_LAYOUT`."""
    return PipelineStageModule(
        _StageSource(backbone, head), CAUSAL_LM_LAYOUT, is_first=is_first, is_last=is_last, layer_offset=layer_offset
    )


class FakeStage:
    """Stand-in for a ``PipelineStage`` around an FSDP2 module.

    ``perform_reduce_grad`` counts the reshard it would run, and is defined on the class: the runner
    suppresses it by shadowing the attribute in the instance dict, so a failed restore stays visible.
    """

    def __init__(self):
        self.has_backward = True
        self.cleared = 0
        self.reshards = 0

    def clear_runtime_states(self):
        self.cleared += 1

    def perform_reduce_grad(self, grad_scale_factor):
        del grad_scale_factor
        self.reshards += 1


class FakeSchedule:
    """Mirrors the single-stage schedules: ``step`` and ``eval`` both funnel through
    ``_step_microbatches``, which ends in reduce-grad, and it records what it received."""

    def __init__(self, n_microbatches: int = 2, has_backward: bool = True):
        self._stage = FakeStage()
        self._n_microbatches = n_microbatches
        self._has_backward = has_backward
        self._loss_fn = lambda output, target: output
        self._stage_forward_initialized = True
        self._stage_backward_initialized = True
        self.eval_calls = 0
        self.step_calls = 0
        self.captured = None
        self.cleared_before_microbatches = None

    def _split_inputs(self, args, kwargs):
        # torch splits args/kwargs with tensor_split too.
        n = self._n_microbatches
        arg_mbs = [tuple(torch.tensor_split(a, n)[i] for a in args) for i in range(n)]
        kwarg_mbs = [{k: torch.tensor_split(v, n)[i] for k, v in (kwargs or {}).items()} for i in range(n)]
        return arg_mbs, kwarg_mbs

    def _step_microbatches(self, arg_mbs, kwarg_mbs, target_mbs, losses, return_outputs=False):
        self.captured = (arg_mbs, kwarg_mbs, target_mbs, return_outputs)
        self.cleared_before_microbatches = self._stage.cleared
        losses.append(torch.tensor(1.0))
        self._stage.perform_reduce_grad(1)

    def step(self, *args, target=None, losses=None, return_outputs=False, **kwargs):
        self.step_calls += 1
        self._step_microbatches(None, None, None, losses)

    def eval(self, *args, target=None, losses=None, return_outputs=False, **kwargs):
        self.eval_calls += 1
        self._step_microbatches(None, None, None, losses if losses is not None else [])


def fake_runner(schedule: FakeSchedule | None = None) -> PipelineRunner:
    """A one-stage runner around ``schedule``; ``__init__`` needs dist + FSDP, the drive logic does not."""
    runner = object.__new__(PipelineRunner)
    runner.schedule = schedule or FakeSchedule()
    runner.stage = runner.schedule._stage
    runner.stage_module = None
    runner.is_first = True
    runner.is_last = True
    runner.loss = TokenLoss()
    runner.rows_per_example = 1
    runner.n_microbatches = runner.schedule._n_microbatches
    runner._num_items_in_batch = 1
    return runner
