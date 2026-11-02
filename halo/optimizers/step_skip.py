"""The optimizer step a device-side flag skips, so the host never waits for the verdict that decides it.

Under the flag a step changes no parameter or moment; its host bookkeeping (step counts, lazily created state) still
moves, and :meth:`DeviceStepSkipMixin.settle_step` rolls it back once the host has read the flag.
"""

import torch
from torch import Tensor


def write_unless_skipped(dst: Tensor, value: Tensor, skip: Tensor | None) -> None:
    """``dst.copy_(value)``, except where the device flag ``skip`` is set, which keeps ``dst`` as it is."""
    if skip is None:
        dst.copy_(value)
    else:
        torch.where(skip, dst, value, out=dst)


class DeviceStepSkipMixin:
    """A ``torch.optim.Optimizer`` whose next step a device flag can skip.

    The subclass's step calls :meth:`_take_step_skip` once, before it touches any state, and hands the flag it
    returns to every write: a kernel's, :func:`write_unless_skipped`, or operands that leave their values as they were
    where it is set.
    """

    _armed_skip: Tensor | None = None
    # Every parameter's state before the last flagged step: a shallow copy, or None where it had no entry.
    _state_before_step: list[tuple[Tensor, dict | None]] | None = None

    def skip_next_step_if(self, flag: Tensor) -> None:
        """Make the next step change no parameter or moment where ``flag``, a 0-dim bool on the parameters' device, is
        set."""
        self._armed_skip = flag

    def settle_step(self, skipped: bool) -> None:
        """Settle the last step once the host has read its flag: where ``skipped``, return every step count and state
        entry to its value before the step. Either way drop the recorded state, which would otherwise keep tensors
        a later ``load_state_dict`` replaces alive until the next step.

        Raises ``RuntimeError`` when ``skipped`` and the last step ran without a flag, which recorded nothing to
        return to.
        """
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
        """The flag armed for this step, or None; a flagged step first records the state of ``param_groups``'
        parameters (the ones whose state this optimizer holds) for :meth:`settle_step`."""
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
