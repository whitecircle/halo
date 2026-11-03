"""Hand-built pipeline stages, a stand-in torch schedule and the family-registration opt-in for the pipeline tests."""

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace

import torch
import torch.nn as nn

from halo.distributed.pipeline_parallel.families import GENERIC_PP, PP_SPEC_MAP
from halo.distributed.pipeline_parallel.runner import PipelineRunner
from halo.distributed.pipeline_parallel.stage import PipelineStageModule, StageLayout
from halo.distributed.pipeline_parallel.stage_losses import TokenLoss

# A ``*ForCausalLM`` with its backbone under ``model`` and head under ``lm_head``.
CAUSAL_LM_LAYOUT = StageLayout(
    backbone_prefix="model",
    layer_attr="layers",
    head_attr="lm_head",
    post_layer_paths=("model.norm",),
    embedding_path="model.embed_tokens",
)


@contextmanager
def registered_pp_backbones(*names: str) -> Iterator[None]:
    """Register ``names`` for PP inside the block: a claimed class keeps its spec with the refusal lifted, an unclaimed
    one takes default probes."""
    saved = dict(PP_SPEC_MAP)
    PP_SPEC_MAP.update(
        {name: replace(PP_SPEC_MAP.get(name, GENERIC_PP), refusal=None, validated=True) for name in names}
    )
    try:
        yield
    finally:
        PP_SPEC_MAP.clear()
        PP_SPEC_MAP.update(saved)


class _StageSource(nn.Module):
    """The unsplit model a stage is cut from: no config, so no checkpoint metadata."""

    def __init__(self, backbone: nn.Module, head: nn.Module | None):
        super().__init__()
        self.model = backbone
        self.lm_head = head

    def can_generate(self) -> bool:
        return False


def hand_built_stage(
    backbone: nn.Module, head: nn.Module | None, *, is_first: bool, is_last: bool, layer_offset: int
) -> PipelineStageModule:
    return PipelineStageModule(
        _StageSource(backbone, head), CAUSAL_LM_LAYOUT, is_first=is_first, is_last=is_last, layer_offset=layer_offset
    )


class FakeStage:
    """Stand-in for a ``PipelineStage``; ``perform_reduce_grad`` is on the class since the runner shadows it per
    instance, so a failed restore shows."""

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
    """Mirrors the single-stage schedules: ``step`` and ``eval`` funnel through ``_step_microbatches``, ending in
    reduce-grad and recording its input."""

    def __init__(self, n_microbatches: int = 2, has_backward: bool = True):
        self._stage = FakeStage()
        self._n_microbatches = n_microbatches
        self._has_backward = has_backward
        self._loss_fn = lambda output, target: output
        self._stage_forward_initialized = True
        self._stage_backward_initialized = True
        self.eval_calls = 0
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
    runner.token_ids_kwarg = None
    runner.is_first = True
    runner.is_last = True
    runner.loss = TokenLoss()
    runner.rows_per_example = 1
    runner.n_microbatches = runner.schedule._n_microbatches
    runner._num_items_in_batch = 1
    return runner
