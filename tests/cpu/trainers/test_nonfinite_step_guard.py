#!/usr/bin/env python
"""CPU test: an optimizer step whose global gradient norm is NaN or inf is never applied.

``NonFiniteStepGuardMixin`` takes its verdict off the norm transformers computes right before ``optimizer.step()``
and discards the step's gradients and loss instead of stepping. Every case runs on both of its paths: a torch
optimizer, skipped on the host after a read of the norm, and AdamWBF16, skipped on the device under a flag whose norm
the host reads a step late (at the next step's norm, a logging or saving step, or the end of training). Pinned on a
real LR scheduler, on AdamWBF16's deferred clip scale, and end to end through ``Trainer.train()``: the guard depends
on that loop's order (norm, step, scheduler, log), so a transformers bump that reorders it fails here.

    python tests/cpu/trainers/test_nonfinite_step_guard.py
"""

import copy
import math
import warnings
import weakref
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from accelerate.optimizer import AcceleratedOptimizer
from datasets import Dataset
from transformers import Qwen3Config, Qwen3ForCausalLM, Trainer, TrainerCallback, TrainingArguments
from trl import SFTConfig

from halo.configs.parser import HaloArgumentParser
from halo.configs.script_arguments import CommonScriptArguments
from halo.optimizers.adamw_bf16 import AdamWBF16
from halo.trainers.parallelism.grad_clipping import GradientClippingMixin, is_clipping_enabled
from halo.trainers.sft import DistributedSFTTrainer
from halo.trainers.step_guard import NonFiniteStepGuardMixin
from tests.common.offline_grpo import OFFLINE_VOCAB, make_offline_tokenizer
from tests.common.parallelism import make_parallelism_config
from tests.common.step_guard import Guarded, guarded, guarded_step

# The guard's two paths: a torch optimizer skipped on the host, AdamWBF16 skipped on the device.
PATHS = pytest.mark.parametrize("optimizer_cls", [torch.optim.AdamW, AdamWBF16], ids=["host", "device"])


class _NoHostRead(torch.Tensor):
    """A norm that refuses to be read back to the host."""

    def item(self):
        raise AssertionError("the gradient norm was read back to the host")

    __float__ = __bool__ = tolist = item


def _weights(me: Guarded) -> list[torch.Tensor]:
    return [param.detach().clone() for param in me.model.parameters()]


def _moments(me: Guarded) -> list[dict]:
    return [copy.deepcopy(me.optimizer.state[param]) for param in me.model.parameters()]


def _moment_tensors(me: Guarded) -> list[torch.Tensor]:
    """The Adam moments alone, which a device skip leaves unwritten before its verdict is read."""
    return [
        me.optimizer.state[param][key].clone() for param in me.model.parameters() for key in ("exp_avg", "exp_avg_sq")
    ]


def _same(a: list[torch.Tensor], b: list[torch.Tensor]) -> bool:
    return len(a) == len(b) and all(torch.equal(x, y) for x, y in zip(a, b, strict=True))


def _same_moments(a: list[dict], b: list[dict]) -> bool:
    return all(
        x.keys() == y.keys() and all(torch.equal(torch.as_tensor(x[k]), torch.as_tensor(y[k])) for k in x)
        for x, y in zip(a, b, strict=True)
    )


@PATHS
@pytest.mark.parametrize("bad", [float("nan"), float("inf")], ids=["nan", "inf"])
def test_a_nonfinite_step_leaves_weights_and_moments_untouched(bad, optimizer_cls):
    me = guarded(optimizer_cls=optimizer_cls)
    guarded_step(me, 0.5)
    me._read_device_verdict()
    weights, moments, tensors = _weights(me), _moments(me), _moment_tensors(me)

    guarded_step(me, bad)

    assert _same(weights, _weights(me)), "a non-finite step reached the weights"
    assert _same(tensors, _moment_tensors(me)), "a non-finite step reached the optimizer moments"
    me._read_device_verdict()
    assert _same_moments(moments, _moments(me)), "the skipped step left the optimizer state changed"
    if optimizer_cls is torch.optim.AdamW:
        assert all(param.grad is None for param in me.model.parameters()), "the skipped step's gradients survived"
    assert me._nonfinite_steps_skipped == 1 and me._nonfinite_streak == 1

    guarded_step(me, 0.5)
    me._read_device_verdict()
    assert not _same(weights, _weights(me)), "the step after a skipped one was not applied"
    assert me._nonfinite_streak == 0


def test_the_device_verdict_is_read_a_step_late_without_a_host_read_before_the_step():
    """The device path neither reads the norm before its step nor waits for it: the next step's norm settles it,
    rolling the optimizer's step counts back, while that step's own verdict stays unread."""
    me = guarded(optimizer_cls=AdamWBF16)
    guarded_step(me, 0.5)
    weights = _weights(me)
    for param in me.model.parameters():
        param.grad = torch.full_like(param, float("nan"))
    me._get_grad_norm(me.model, grad_norm=torch.tensor(float("nan")).as_subclass(_NoHostRead))
    me.optimizer.step()
    me.state.global_step += 1

    assert me._nonfinite_steps_skipped == 0, "the verdict was read before the step"
    assert _same(weights, _weights(me)), "the flagged step reached the weights"
    guarded_step(me, 0.5)
    assert me._nonfinite_steps_skipped == 1, "the next step's norm did not settle the late verdict"
    assert me._unread_verdict is not None and me._unread_verdict[0] == 3, "the third step's own verdict is unread"
    assert all(me.optimizer.state[param]["step"] == 2 for param in me.model.parameters()), (
        "the skipped step still counts in the optimizer's step counts"
    )


def test_a_finite_late_read_lets_go_of_the_optimizer_state():
    """A finite verdict settles its step too: what the step recorded for a skip must not keep alive the optimizer
    state an in-process resume replaces after the last step's read, a whole extra copy of the rank's state."""
    me = guarded(optimizer_cls=AdamWBF16)
    guarded_step(me, 0.5)
    guarded_step(me, 0.5)
    me._read_device_verdict()
    old = weakref.ref(me.optimizer.state[next(me.model.parameters())]["exp_avg"])

    me.optimizer.state.clear()

    assert old() is None, "the settled step's record keeps the replaced optimizer state alive"


@pytest.mark.parametrize("flag", ["should_log", "should_evaluate", "should_save", None])
def test_a_step_that_logs_evaluates_or_saves_reads_the_verdict_first(flag):
    """A log reports the late verdict, and a save, or an evaluation that decides a ``save_strategy: best`` save,
    writes an optimizer state that leaves the skipped step out; a step doing none of them leaves it unread."""
    me = guarded(optimizer_cls=AdamWBF16)
    guarded_step(me, 0.5)
    guarded_step(me, float("nan"))
    me.control = SimpleNamespace(should_log=False, should_evaluate=False, should_save=False)
    if flag is not None:
        setattr(me.control, flag, True)
    me._globalstep_last_logged = me.state.global_step

    me._maybe_log_save_evaluate(me._tr_loss)

    read = flag is not None
    assert (me._unread_verdict is None) == read and me._nonfinite_steps_skipped == int(read)
    assert me.counted_at_save == [2 - read] * 2, "the pass ran before the verdict that rolls the count back"


def test_the_host_path_reads_the_norm_before_the_step():
    """Negative control for the test above: a torch optimizer's verdict is a host read of the norm."""
    me = guarded(optimizer_cls=torch.optim.AdamW)
    with pytest.raises(AssertionError, match="read back to the host"):
        me._get_grad_norm(me.model, grad_norm=torch.tensor(1.0).as_subclass(_NoHostRead))


def test_the_skip_drops_a_deferred_clip_scale():
    """AdamWBF16 multiplies the clip coefficient in at its next step: one deferred for discarded gradients
    must not scale the next step's."""
    me = guarded(optimizer_cls=AdamWBF16)
    guarded_step(me, 0.5)
    weights = _weights(me)
    me.optimizer.defer_grad_scale(torch.tensor(0.25))

    guarded_step(me, float("nan"))
    me._read_device_verdict()

    assert _same(weights, _weights(me)), "the step was not skipped"
    assert me.optimizer._grad_scale is None, "the discarded step's clip scale would reach the next step"


@PATHS
def test_consecutive_nonfinite_steps_raise_past_the_limit(optimizer_cls):
    """On the device path the raise comes at the late read, before any later step is applied."""
    me = guarded(limit=2, optimizer_cls=optimizer_cls)
    guarded_step(me, 0.5)
    weights = _weights(me)
    guarded_step(me, float("nan"))
    guarded_step(me, float("nan"))

    with pytest.raises(RuntimeError, match=r"steps 2-4: 3 consecutive steps .* weights are those after step 1"):
        guarded_step(me, float("nan"))
        me._read_device_verdict()
    assert _same(weights, _weights(me))


@PATHS
def test_a_finite_step_resets_the_streak(optimizer_cls):
    me = guarded(limit=1, optimizer_cls=optimizer_cls)
    for grad in (float("nan"), 0.5, float("nan"), 0.5, float("nan")):
        guarded_step(me, grad)
    me._read_device_verdict()

    assert me._nonfinite_steps_skipped == 3 and me._nonfinite_streak == 1


@PATHS
def test_limit_zero_raises_at_the_first_nonfinite_step(optimizer_cls):
    me = guarded(limit=0, optimizer_cls=optimizer_cls)

    with pytest.raises(RuntimeError, match="1 consecutive steps"):
        guarded_step(me, float("nan"))
        me._read_device_verdict()


@PATHS
def test_nonfinite_weights_raise_at_once_naming_the_parameter(optimizer_cls):
    """A skip cannot repair weights that are already NaN: the run must stop, whatever the streak."""
    me = guarded(limit=5, optimizer_cls=optimizer_cls)
    with torch.no_grad():
        me.model.bias[0] = float("nan")

    with pytest.raises(RuntimeError, match=r"1 parameter\(s\) already hold NaN/inf before the update: bias\."):
        guarded_step(me, float("nan"))
        me._read_device_verdict()


def _scheduler_warnings(caught) -> list[str]:
    return [str(w.message) for w in caught if "lr_scheduler" in str(w.message) or "optimizer.step" in str(w.message)]


@PATHS
def test_the_lr_scheduler_wraps_the_guarded_step_without_warning(optimizer_cls):
    """torch's scheduler re-wraps ``optimizer.step`` through ``__func__`` and warns on a step it cannot
    track; ``create_optimizer`` installs the guard before the scheduler exists."""
    me = guarded(optimizer_cls=optimizer_cls)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        scheduler = torch.optim.lr_scheduler.LambdaLR(me.optimizer, lambda step: 1.0 / (step + 1))
        guarded_step(me, float("nan"))
        scheduler.step()
        guarded_step(me, 0.5)
        scheduler.step()

    assert not _scheduler_warnings(caught)
    assert me._nonfinite_steps_skipped == 1 and scheduler.last_epoch == 2


@PATHS
def test_a_scheduler_built_before_the_guard_keeps_stepping_and_tracking(optimizer_cls):
    """A caller's ``optimizers=(optimizer, scheduler)`` wraps ``step`` before ``create_optimizer`` installs the guard:
    a skipped first step must still count as taken for the scheduler, and its wrapper's marker must survive."""
    me = guarded(optimizer_cls=optimizer_cls, install=False)
    scheduler = torch.optim.lr_scheduler.LambdaLR(me.optimizer, lambda step: 1.0 / (step + 1))
    me._install_nonfinite_step_skip()
    weights = _weights(me)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        guarded_step(me, float("nan"))
        scheduler.step()
        assert _same(weights, _weights(me)), "a non-finite step reached the weights"
        guarded_step(me, 0.5)
        scheduler.step()

    assert not _scheduler_warnings(caught)
    assert not _same(weights, _weights(me)), "the step after a skipped one was not applied"
    assert me._nonfinite_steps_skipped == 1 and scheduler.last_epoch == 2
    assert me.optimizer.param_groups[0]["lr"] == pytest.approx(0.1 / 3)


@PATHS
def test_a_loss_scaler_stepping_the_optimizer_owns_overflow_steps(optimizer_cls):
    """Under fp16 accelerate's optimizer wrapper steps through the scaler, which skips an overflow step
    itself and lowers its scale; the guard takes no verdict."""
    me = guarded(limit=0, optimizer_cls=optimizer_cls)
    wrapper = object.__new__(AcceleratedOptimizer)
    wrapper.optimizer, wrapper.scaler = me.optimizer, object()
    me.optimizer = wrapper

    me._get_grad_norm(me.model, grad_norm=torch.tensor(float("inf")))
    me._read_device_verdict()

    assert not me._skip_next_optimizer_step and me._device_skip is None and me._nonfinite_steps_skipped == 0


@PATHS
def test_a_scaler_that_never_steps_the_optimizer_does_not_disarm_the_guard(optimizer_cls):
    """The toolkit's no-wrap prepare keeps the raw optimizer, so an accelerator scaler never skips for it."""
    me = guarded(optimizer_cls=optimizer_cls, scaler=object())
    guarded_step(me, 0.5)
    weights = _weights(me)

    guarded_step(me, float("inf"))
    me._read_device_verdict()

    assert _same(weights, _weights(me)) and me._nonfinite_steps_skipped == 1


def test_a_negative_limit_is_refused_at_parse_time():
    with pytest.raises(ValueError, match="max_consecutive_nonfinite_steps must be an integer >= 0"):
        HaloArgumentParser((CommonScriptArguments,)).parse_dict({"max_consecutive_nonfinite_steps": -1})


@PATHS
def test_a_pending_skip_never_swallows_the_restore_step(optimizer_cls):
    """The zero-LR step that materializes optimizer state during a restore always steps, and leaves the verdict to
    the step it belongs to."""
    me = guarded(optimizer_cls=optimizer_cls)
    guarded_step(me, 0.5)
    me._read_device_verdict()
    for param in me.model.parameters():
        param.grad = torch.full_like(param, float("nan"))
    me._get_grad_norm(me.model, grad_norm=None)
    me._restoring_optimizer_state = True
    calls = int(me.optimizer.state[next(me.model.parameters())]["step"])

    me.optimizer.step()

    assert int(me.optimizer.state[next(me.model.parameters())]["step"]) == calls + 1, "the restore step was skipped"
    assert me._skip_next_optimizer_step or me._device_skip is not None, (
        "the verdict was consumed by a step it was not meant for"
    )


class _InfGrad(torch.autograd.Function):
    """Identity forward, infinite backward: a finite loss whose gradients overflow."""

    @staticmethod
    def forward(ctx, x):
        return x.clone()

    @staticmethod
    def backward(ctx, grad):
        return grad * math.inf


_POISONS = {"nan_loss": lambda loss: loss * math.nan, "inf_grad": _InfGrad.apply}


class _Tiny(torch.nn.Module):
    """A regression recording every forward's loss; ``poison`` rewrites forward ``poison_at``'s (NaN by default)."""

    def __init__(self, poison: str = "nan_loss"):
        super().__init__()
        self.linear = torch.nn.Linear(4, 1)
        self.rewrite = _POISONS[poison]
        self.poison_at = None
        self.losses = []

    def forward(self, x, y):
        loss = torch.nn.functional.mse_loss(self.linear(x).squeeze(-1), y)
        if len(self.losses) == self.poison_at:
            loss = self.rewrite(loss)
        self.losses.append(loss.item())
        return {"loss": loss}


class _Rows(torch.utils.data.Dataset):
    def __init__(self, n: int):
        generator = torch.Generator().manual_seed(0)
        self.x = torch.randn(n, 4, generator=generator)
        self.y = self.x.sum(-1)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return {"x": self.x[i], "y": self.y[i]}


class _GuardedTrainer(GradientClippingMixin, NonFiniteStepGuardMixin, Trainer):
    """The install ``DistributedTrainerMixin.create_optimizer`` performs, on a plain Trainer, behind the clipping
    mixin in the order the spine composes them."""

    def create_optimizer(self, model=None):
        super().create_optimizer(model)
        self._install_nonfinite_step_skip()
        return self.optimizer


class _PoisonOneStep(TrainerCallback):
    """Poisons the loss of optimizer step ``step``'s last micro-batch, leaving any earlier one finite, and records
    the weights around the step and after every step."""

    def __init__(self, model: _Tiny, step: int | None):
        self.model, self.step = model, step
        self.before = self.after = None
        self.after_step: dict[int, list[torch.Tensor]] = {}

    def _weights(self):
        return [param.detach().clone() for param in self.model.parameters()]

    def on_step_begin(self, args, state, control, **kwargs):
        if state.global_step + 1 == self.step:
            self.before = self._weights()
            self.model.poison_at = len(self.model.losses) + args.gradient_accumulation_steps - 1

    def on_step_end(self, args, state, control, **kwargs):
        self.after_step[state.global_step] = self._weights()
        if state.global_step == self.step:
            self.after = self._weights()


def _trainer(
    tmp_path,
    model,
    poison,
    *,
    total,
    max_grad_norm,
    logging_steps,
    logging_strategy="steps",
    accumulation=1,
    nan_inf_filter=False,
    optimizer_cls=torch.optim.AdamW,
    limit=3,
    save_steps=None,
):
    """``optimizer_cls`` AdamWBF16 takes the device path; torch's AdamW, transformers' own build, the host one."""
    args = TrainingArguments(
        output_dir=str(tmp_path),
        max_steps=total,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=accumulation,
        learning_rate=0.05,
        lr_scheduler_type="linear",
        max_grad_norm=max_grad_norm,
        logging_strategy=logging_strategy,
        logging_steps=logging_steps,
        logging_nan_inf_filter=nan_inf_filter,
        save_strategy="no" if save_steps is None else "steps",
        save_steps=save_steps or 500,
        report_to=[],
        use_cpu=True,
        disable_tqdm=True,
        dataloader_num_workers=0,
        seed=0,
    )
    optimizer = None if optimizer_cls is torch.optim.AdamW else optimizer_cls(model.named_parameters(), lr=0.05)
    trainer = _GuardedTrainer(
        model=model, args=args, train_dataset=_Rows(32), callbacks=[poison], optimizers=(optimizer, None)
    )
    trainer._max_consecutive_nonfinite_steps = limit
    return trainer


def _step_counts(optimizer) -> list[int]:
    return [int(state["step"]) for state in optimizer.state.values()]


@PATHS
@pytest.mark.parametrize("max_grad_norm", [1.0, 0.0], ids=["clipped", "unclipped"])
def test_trainer_skips_the_poisoned_step_and_trains_on(tmp_path, max_grad_norm, optimizer_cls):
    poison_step, total = 3, 6
    model = _Tiny()
    poison = _PoisonOneStep(model, poison_step)
    trainer = _trainer(
        tmp_path, model, poison, total=total, max_grad_norm=max_grad_norm, logging_steps=1, optimizer_cls=optimizer_cls
    )

    result = trainer.train()

    steps = {log["step"]: log for log in trainer.state.log_history if "loss" in log}
    assert sorted(steps) == list(range(1, total + 1))
    assert poison.before is not None and _same(poison.before, poison.after), "the poisoned step was applied"
    assert math.isnan(steps[poison_step]["loss"]), "the skipped step's NaN loss must stay visible in its log"
    assert all(math.isfinite(steps[s]["loss"]) for s in steps if s != poison_step), (
        "a later window logged NaN: the skipped step's loss was left in the accumulator"
    )
    assert [steps[s].get("nonfinite_steps_skipped", 0) for s in sorted(steps)] == [0, 0, 1, 1, 1, 1]
    rates = [steps[s]["learning_rate"] for s in sorted(steps)]
    assert all(a > b for a, b in zip(rates, rates[1:], strict=False)), "the LR schedule stalled on the skipped step"
    final = [param.detach() for param in model.parameters()]
    assert all(torch.isfinite(t).all() for t in final) and not _same(poison.after, final)
    assert result.training_loss == pytest.approx(
        sum(steps[s]["loss"] for s in steps if s != poison_step) / (total - 1)
    ), "the run's mean loss is over its applied steps: nothing of the skipped one, in neither sum nor count"
    assert _step_counts(trainer.optimizer) == [total - 1] * 2, "the optimizer counts the skipped step"


@PATHS
def test_an_unclipped_run_skips_a_poisoned_step_it_does_not_log(tmp_path, optimizer_cls):
    """At ``max_grad_norm <= 0`` transformers asks for the norm on every step, logged or not; a norm skipped on
    the steps that log nothing would let a poisoned one through unread."""
    total = 6
    model = _Tiny()
    poison = _PoisonOneStep(model, 3)
    trainer = _trainer(
        tmp_path, model, poison, total=total, max_grad_norm=0.0, logging_steps=4, optimizer_cls=optimizer_cls
    )

    result = trainer.train()

    assert poison.before is not None and _same(poison.before, poison.after), "the poisoned step was applied"
    assert trainer._nonfinite_steps_skipped == 1
    assert all(torch.isfinite(param).all() for param in model.parameters())
    assert math.isfinite(result.training_loss)


@PATHS
def test_a_skipped_last_step_is_reported_before_training_ends(tmp_path, optimizer_cls):
    """transformers adds the unlogged accumulator to the run's mean at the end of training, without a log; the
    device path reads the last step's verdict there, so it is counted and kept out of ``train_loss``."""
    total = 6
    model = _Tiny()
    poison = _PoisonOneStep(model, total)
    trainer = _trainer(
        tmp_path, model, poison, total=total, max_grad_norm=1.0, logging_steps=10, optimizer_cls=optimizer_cls
    )

    result = trainer.train()

    assert poison.before is not None and _same(poison.before, poison.after), "the poisoned step was applied"
    assert trainer._nonfinite_steps_skipped == 1, "the last step's verdict was never read"
    assert trainer._unread_verdict is None
    assert result.training_loss == pytest.approx(sum(model.losses[: total - 1]) / (total - 1))


@PATHS
@pytest.mark.parametrize("poison_step", [3, 6], ids=["mid_run", "last_step"])
def test_the_limit_raises_before_any_later_step_is_applied(tmp_path, poison_step, optimizer_cls):
    """At ``max_consecutive_nonfinite_steps=0`` the host path raises before the poisoned step; the device path at the
    late read, before the next step's update (or as training ends), the weights still those of the step before."""
    total = 6
    model = _Tiny()
    poison = _PoisonOneStep(model, poison_step)
    trainer = _trainer(
        tmp_path, model, poison, total=total, max_grad_norm=1.0, logging_steps=10, optimizer_cls=optimizer_cls, limit=0
    )

    with pytest.raises(RuntimeError, match=rf"steps {poison_step}-{poison_step}: 1 consecutive steps"):
        trainer.train()

    late = optimizer_cls is AdamWBF16
    assert trainer.state.global_step == poison_step - 1 + late
    assert _same(poison.after_step[poison_step - 1], [param.detach() for param in model.parameters()])


@PATHS
def test_a_save_after_a_skipped_step_resumes_as_the_run_went_on(tmp_path, optimizer_cls):
    """A save right after a skipped step reads its verdict first: the checkpoint's optimizer state counts the applied
    steps only, and a run resumed from it ends on the uninterrupted run's weights, bit for bit. Nothing else of the
    guard is checkpointed, so the resumed run starts with no streak and nothing unread."""
    poison_step, total = 3, 6
    model = _Tiny()
    poison = _PoisonOneStep(model, poison_step)
    trainer = _trainer(
        tmp_path,
        model,
        poison,
        total=total,
        max_grad_norm=1.0,
        logging_steps=10,
        optimizer_cls=optimizer_cls,
        save_steps=poison_step,
    )
    trainer.train()
    assert trainer._nonfinite_steps_skipped == 1 and poison.before is not None, "premise: the poisoned step skipped"
    checkpoint = Path(tmp_path) / f"checkpoint-{poison_step}"
    saved = torch.load(checkpoint / "optimizer.pt", weights_only=False)

    assert [int(state["step"]) for state in saved["state"].values()] == [poison_step - 1] * 2, (
        "the checkpoint counts the skipped step"
    )
    resumed_model = _Tiny()
    resumed = _trainer(
        tmp_path,
        resumed_model,
        _PoisonOneStep(resumed_model, None),
        total=total,
        max_grad_norm=1.0,
        logging_steps=10,
        optimizer_cls=optimizer_cls,
    )
    resumed.train(resume_from_checkpoint=str(checkpoint))
    assert resumed._nonfinite_steps_skipped == 0 and resumed._nonfinite_streak == 0
    assert _same([p.detach() for p in model.parameters()], [p.detach() for p in resumed_model.parameters()])


class _RaiseAfterStep(TrainerCallback):
    """Raises at the end of optimizer step ``step``, before anything reads its verdict, while armed."""

    def __init__(self, step: int):
        self.step, self.armed = step, True

    def on_step_end(self, args, state, control, **kwargs):
        if self.armed and state.global_step == self.step:
            raise RuntimeError("interrupted")


def test_a_resume_after_an_interrupted_run_settles_its_unread_verdict_first(tmp_path):
    """A run that raised right after a device-skipped step leaves its verdict unread; the next ``train()`` on the
    trainer settles it before loading the checkpoint, so the stale rollback never lands on the loaded state and the
    resumed run ends on the uninterrupted run's weights, bit for bit. The stale skip belongs to the interrupted run, so
    the resumed run's ``train_loss`` divides by its own applied steps alone."""
    poison_step, total = 3, 6
    torch.manual_seed(0)
    reference_model = _Tiny()
    reference = _trainer(
        tmp_path / "reference",
        reference_model,
        _PoisonOneStep(reference_model, poison_step),
        total=total,
        max_grad_norm=1.0,
        logging_steps=10,
        optimizer_cls=AdamWBF16,
    )
    reference.train()
    torch.manual_seed(0)
    model = _Tiny()
    trainer = _trainer(
        tmp_path / "interrupted",
        model,
        _PoisonOneStep(model, poison_step),
        total=total,
        max_grad_norm=1.0,
        logging_steps=10,
        optimizer_cls=AdamWBF16,
        save_steps=1,
    )
    interrupt = _RaiseAfterStep(poison_step)
    trainer.add_callback(interrupt)
    with pytest.raises(RuntimeError, match="interrupted"):
        trainer.train()
    assert trainer._unread_verdict is not None, "premise: the interrupted run left the skipped step's verdict unread"

    interrupt.armed = False
    resumed_from = 1
    result = trainer.train(resume_from_checkpoint=str(tmp_path / "interrupted" / f"checkpoint-{resumed_from}"))

    assert _same([p.detach() for p in reference_model.parameters()], [p.detach() for p in model.parameters()])
    assert _step_counts(trainer.optimizer) == _step_counts(reference.optimizer) == [total - 1] * 2
    resumed = dict(zip(range(resumed_from + 1, total + 1), model.losses[resumed_from - total :], strict=True))
    applied = [loss for step, loss in resumed.items() if step != poison_step]
    assert result.training_loss == pytest.approx(sum(applied) / len(applied)), (
        "the interrupted run's skipped step was counted against the resumed run's steps"
    )


@PATHS
@pytest.mark.parametrize("accumulation", [1, 2], ids=["ga1", "ga2"])
@pytest.mark.parametrize("poison", sorted(_POISONS))
@pytest.mark.parametrize(
    ("logging_strategy", "logging_steps", "nan_inf_filter"),
    [("steps", 1, False), ("steps", 1, True), ("steps", 3, False), ("epoch", 1, False), ("no", 1, False)],
    ids=["steps1", "steps1-filter", "steps3", "epoch", "no"],
)
def test_train_loss_and_every_window_average_the_applied_steps(
    tmp_path, poison, logging_strategy, logging_steps, nan_inf_filter, accumulation, optimizer_cls
):
    """The run's mean and every logged window average the applied steps' losses over the applied steps alone,
    whatever the window and whether the skipped step's loss was finite; a window with no applied step logs NaN,
    under ``logging_nan_inf_filter`` too. Under gradient accumulation only the step's last micro-batch is poisoned,
    and its finite earlier ones go with it."""
    poison_step, total = 3, 6
    torch.manual_seed(0)
    model = _Tiny(poison)
    tracker = _PoisonOneStep(model, poison_step)
    trainer = _trainer(
        tmp_path,
        model,
        tracker,
        total=total,
        max_grad_norm=1.0,
        logging_steps=logging_steps,
        logging_strategy=logging_strategy,
        accumulation=accumulation,
        nan_inf_filter=nan_inf_filter,
        optimizer_cls=optimizer_cls,
    )

    result = trainer.train()

    assert tracker.before is not None and _same(tracker.before, tracker.after), "the poisoned step was applied"
    assert trainer._nonfinite_steps_skipped == 1 and len(model.losses) == total * accumulation
    # transformers scales a micro-batch's loss by 1 / accumulation for a model that takes no loss kwargs.
    step_losses = [
        sum(model.losses[start : start + accumulation]) / accumulation
        for start in range(0, len(model.losses), accumulation)
    ]
    applied = {step: loss for step, loss in enumerate(step_losses, start=1) if step != poison_step}
    assert result.training_loss == pytest.approx(sum(applied.values()) / len(applied))
    first = 1
    for log in (log for log in trainer.state.log_history if "loss" in log):
        window = [applied[s] for s in range(first, log["step"] + 1) if s in applied]
        if window:
            assert log["loss"] == pytest.approx(sum(window) / len(window)), log
        else:
            assert math.isnan(log["loss"]), f"a window with no applied step must log NaN: {log}"
        first = log["step"] + 1


@PATHS
def test_a_run_none_of_whose_steps_was_applied_has_a_nan_train_loss(tmp_path, optimizer_cls):
    model = _Tiny()
    tracker = _PoisonOneStep(model, 1)
    trainer = _trainer(
        tmp_path, model, tracker, total=1, max_grad_norm=1.0, logging_steps=10, optimizer_cls=optimizer_cls
    )

    result = trainer.train()

    assert trainer._nonfinite_steps_skipped == 1
    assert math.isnan(result.training_loss), "a run with no applied step has no loss to report"


@PATHS
def test_a_second_run_divides_by_its_own_applied_steps(tmp_path, optimizer_cls):
    """``train()`` counts a run's skipped steps afresh, its logging window's included, as transformers restarts its
    loss accumulators: a first run ending on an unlogged window that holds a skip leaves nothing to the second."""
    total, logging_steps = 6, 4
    model = _Tiny()
    tracker = _PoisonOneStep(model, logging_steps + 1)
    trainer = _trainer(
        tmp_path,
        model,
        tracker,
        total=total,
        max_grad_norm=1.0,
        logging_steps=logging_steps,
        optimizer_cls=optimizer_cls,
    )
    trainer.train()
    assert trainer._nonfinite_steps_skipped == 1, "premise: the first run skips its poisoned step"
    tracker.step = None
    first_run = len(model.losses)

    result = trainer.train()

    second = model.losses[first_run:]
    assert trainer._nonfinite_steps_skipped == 1, "premise: the second run skips no step"
    (window,) = [log for log in trainer.state.log_history if "loss" in log]
    assert window["step"] == logging_steps
    assert window["loss"] == pytest.approx(sum(second[:logging_steps]) / logging_steps), (
        "the first run's unlogged skip was counted against the second run's first window"
    )
    assert result.training_loss == pytest.approx(sum(second) / total), (
        "the first run's skipped step was counted against the second run's steps"
    )


def test_the_norm_only_pass_clips_nothing():
    """transformers sends an infinite threshold for its unclipped norm; scaling every gradient by 1 is a wasted pass,
    and by ``nan_to_num(inf / inf)`` a NaN norm's coefficient."""
    assert not is_clipping_enabled(math.inf)
    assert not is_clipping_enabled(0.0) and not is_clipping_enabled(None)
    assert is_clipping_enabled(1.0)


class _PoisonFinalNorm(TrainerCallback):
    """Multiplies the decoder's final norm output by NaN on optimizer step ``step``, recording the weights around it."""

    def __init__(self, model, step: int):
        self.model, self.step, self.armed = model, step, False
        self.before = self.after = None
        model.model.norm.register_forward_hook(
            lambda module, inputs, output: output * math.nan if self.armed else None
        )

    def _weights(self):
        return [param.detach().clone() for param in self.model.parameters()]

    def on_step_begin(self, args, state, control, **kwargs):
        self.armed = state.global_step + 1 == self.step
        if self.armed:
            self.before = self._weights()

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step == self.step:
            self.after = self._weights()
        self.armed = False


def _sft_trainer(
    tmp_path, *, poison_step: int, optimizers_for=None, **trainer_kwargs
) -> tuple[DistributedSFTTrainer, _PoisonFinalNorm]:
    """The real spine on CPU at world 1, unclipped, logging every fourth of eight steps; ``optimizers_for(model)``
    builds the caller's ``optimizers`` pair."""
    torch.manual_seed(0)
    vocab = len(OFFLINE_VOCAB)
    model = Qwen3ForCausalLM(
        Qwen3Config(
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            vocab_size=vocab,
            pad_token_id=0,
            eos_token_id=1,
        )
    )
    rows = [[3 + (i + j) % (vocab - 3) for j in range(16)] for i in range(32)]
    args = SFTConfig(
        output_dir=str(tmp_path),
        use_cpu=True,
        bf16=False,
        max_length=16,
        per_device_train_batch_size=2,
        max_steps=8,
        max_grad_norm=0.0,
        logging_steps=4,
        logging_nan_inf_filter=False,
        gradient_checkpointing=False,
        use_liger_kernel=False,
        dataloader_num_workers=0,
        save_strategy="no",
        report_to="none",
        disable_tqdm=True,
    )
    poison = _PoisonFinalNorm(model, poison_step)
    if optimizers_for is not None:
        trainer_kwargs["optimizers"] = optimizers_for(model)
    trainer = DistributedSFTTrainer(
        model=model,
        args=args,
        train_dataset=Dataset.from_dict({"input_ids": rows}),
        processing_class=make_offline_tokenizer(),
        parallelism_config=make_parallelism_config(world_size=1, gpus_per_node=1),
        callbacks=[poison],
        **trainer_kwargs,
    )
    return trainer, poison


@pytest.mark.parametrize("bf16_optimizer", [False, True], ids=["host", "device"])
def test_the_distributed_trainer_skips_a_poisoned_step_it_does_not_log(tmp_path, bf16_optimizer):
    """Through ``DistributedTrainerMixin.create_optimizer`` and the spine's composed ``_get_grad_norm``."""
    trainer, poison = _sft_trainer(tmp_path, poison_step=3, bf16_optimizer=bf16_optimizer)

    trainer.train()

    optimizer = getattr(trainer.optimizer, "optimizer", trainer.optimizer)  # accelerate's wrapper
    assert isinstance(optimizer, AdamWBF16) == bf16_optimizer, "premise: the run's optimizer"
    assert poison.before is not None and _same(poison.before, poison.after), "the poisoned step was applied"
    logged = {log["step"]: log for log in trainer.state.log_history if "loss" in log}
    assert logged[4]["nonfinite_steps_skipped"] == 1
    assert math.isfinite(logged[4]["loss"]), "the skipped step's NaN reached the window holding it"
    assert math.isfinite(logged[8]["loss"]), "the window after the skipped one logged the skipped step's NaN"
    assert all(torch.isfinite(param).all() for param in trainer.model.parameters())
    assert set(_step_counts(trainer.optimizer)) == {7}, "the optimizer counts the skipped step"


@pytest.mark.parametrize("bf16_optimizer", [False, True], ids=["host", "device"])
def test_the_trainer_kwarg_sets_the_limit(tmp_path, bf16_optimizer):
    trainer, _ = _sft_trainer(
        tmp_path, poison_step=3, max_consecutive_nonfinite_steps=0, bf16_optimizer=bf16_optimizer
    )

    with pytest.raises(RuntimeError, match=r"steps 3-3: 1 consecutive steps .* max_consecutive_nonfinite_steps=0"):
        trainer.train()


@pytest.mark.parametrize(
    "optimizer_for",
    [
        lambda model: torch.optim.AdamW(model.parameters(), lr=1e-3),
        lambda model: AdamWBF16(model.named_parameters(), lr=1e-3),
    ],
    ids=["host", "device"],
)
def test_the_distributed_trainer_guards_a_callers_prebuilt_scheduler(tmp_path, optimizer_for):
    """``optimizers=(optimizer, scheduler)`` reaches ``create_optimizer`` with ``step`` already scheduler-wrapped."""
    built = {}

    def optimizers_for(model):
        optimizer = optimizer_for(model)
        built["scheduler"] = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0 / (step + 1))
        return optimizer, built["scheduler"]

    trainer, poison = _sft_trainer(tmp_path, poison_step=1, optimizers_for=optimizers_for)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        trainer.train()

    assert not _scheduler_warnings(caught)
    assert poison.before is not None and _same(poison.before, poison.after), "the poisoned step was applied"
    assert trainer._nonfinite_steps_skipped == 1 and built["scheduler"].last_epoch == 8
    logged = {log["step"]: log for log in trainer.state.log_history if "loss" in log}
    assert math.isfinite(logged[8]["loss"])
    assert all(torch.isfinite(param).all() for param in trainer.model.parameters())


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
