"""The optimizer step a device-side flag skips, so the host never waits for the verdict that decides it. Its host
bookkeeping still moves, and :meth:`DeviceStepSkipMixin.settle_step` rolls that back once the flag is read."""

import torch
from torch import Tensor


def write_unless_skipped(dst: Tensor, value: Tensor, skip: Tensor | None) -> None:
    if skip is None:
        dst.copy_(value)
    else:
        torch.where(skip, dst, value, out=dst)


class DeviceStepSkipMixin:
    """The subclass's step calls :meth:`_take_step_skip` once, before any state, and hands the flag to every write."""

    _armed_skip: Tensor | None = None
    # Shallow copies, None where a parameter had no entry.
    _state_before_step: list[tuple[Tensor, dict | None]] | None = None

    def skip_next_step_if(self, flag: Tensor) -> None:
        """``flag`` is a 0-dim bool on the parameters' device."""
        self._armed_skip = flag

    def settle_step(self, skipped: bool) -> None:
        """Where ``skipped``, return every step count and state entry to its value before the step. Either way drop the
        recorded state, which would keep tensors a later ``load_state_dict`` replaces alive until the next step."""
        before_step, self._state_before_step = self._state_before_step, None
        if not skipped:
            return
        if before_step is None:
            raise RuntimeError(
                f"{type(self).__name__}.settle_step: the last step ran without a skip flag, so it recorded no state "
                f"to return to. Settle as skipped only a step skip_next_step_if armed."
            )
        for p, before in before_step:
            if before is None:
                self.state.pop(p, None)
            else:
                self.state[p] = before

    def _take_step_skip(self, param_groups: list[dict]) -> Tensor | None:
        """A flagged step first records the state of ``param_groups``' parameters for :meth:`settle_step`."""
        flag, self._armed_skip = self._armed_skip, None
        self._state_before_step = (
            None
            if flag is None
            else [
                (p, dict(self.state[p]) if p in self.state else None)
                for group in param_groups
                for p in group["params"]
            ]
        )
        return flag
