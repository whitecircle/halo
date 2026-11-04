"""A poisoned optimizer step and the parameter state around one, for suites checking a non-finite step is skipped. NaN
on the last rank's final norm output leaves every forward collective finite and makes the backward NaN, so reductions
carry it to every rank."""

import torch
import torch.distributed as dist
from transformers import TrainerCallback

from halo.distributed.fsdp import reshard_fsdp2_modules
from halo.runtime.dtensor import to_local


def final_norm(model) -> torch.nn.Module:
    """The decoder's final norm, the one module named ``norm``."""
    (norm,) = [module for name, module in model.named_modules() if name.split(".")[-1] == "norm"]
    return norm


def local_parameter_state(model) -> dict[str, torch.Tensor]:
    """This rank's copy of every parameter shard, the groups resharded first."""
    reshard_fsdp2_modules(model)
    return {name: to_local(param.detach()).clone() for name, param in model.named_parameters()}


class PoisonSteps(TrainerCallback):
    def __init__(self, trainer, steps):
        self.steps = frozenset(steps)
        self.armed = False
        if self.steps and dist.get_rank() == dist.get_world_size() - 1:
            final_norm(trainer.model).register_forward_hook(self._poison)

    def _poison(self, module, inputs, output):
        return output * float("nan") if self.armed else None

    def on_step_begin(self, args, state, control, **kwargs):
        self.armed = state.global_step + 1 in self.steps

    def on_step_end(self, args, state, control, **kwargs):
        self.armed = False


class AroundStep(TrainerCallback):
    def __init__(self, trainer, step: int):
        self.trainer, self.step = trainer, step
        self.before = self.after = None

    def on_step_begin(self, args, state, control, **kwargs):
        if state.global_step + 1 == self.step:
            self.before = local_parameter_state(self.trainer.model)

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step == self.step:
            self.after = local_parameter_state(self.trainer.model)

    def moved(self) -> set[str]:
        """The parameters whose shard the step changed; every one when the step was never captured."""
        if self.before is None or self.after is None:
            return {"<step not captured>"}
        return {name for name, value in self.before.items() if not torch.equal(value, self.after[name])}
