#!/usr/bin/env python
"""CPU test: an optimizer step whose global gradient norm is NaN or inf is never applied.

``NonFiniteStepGuardMixin`` reads the norm transformers computes right before ``optimizer.step()`` and
discards the step's gradients and loss instead of stepping. Pinned on a real torch optimizer and LR scheduler,
on AdamWBF16's deferred clip scale, and end to end through ``Trainer.train()``: the guard depends on
that loop's order (norm, step, scheduler, log), so a transformers bump that reorders it fails here.

    python tests/cpu/trainers/test_nonfinite_step_guard.py
"""

import copy
import math
import warnings
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


class _HFBase:
    """The transformers method the guard extends, as the trainer runs it."""

    _get_grad_norm = Trainer._get_grad_norm


class _Guarded(GradientClippingMixin, NonFiniteStepGuardMixin, _HFBase):
    """The guard behind the clipping mixin and ahead of the Trainer base, as in every distributed trainer."""


def _guarded(limit: int = 3, optimizer_cls=torch.optim.AdamW, scaler=None, install: bool = True) -> _Guarded:
    torch.manual_seed(0)
    me = object.__new__(_Guarded)
    me.model = torch.nn.Linear(4, 2)
    me._max_consecutive_nonfinite_steps = limit
    me.state = SimpleNamespace(global_step=0)
    me.accelerator = SimpleNamespace(
        distributed_type="NO", scaler=scaler, clip_grad_norm_=torch.nn.utils.clip_grad_norm_
    )
    me.optimizer = optimizer_cls(me.model.named_parameters(), lr=0.1)
    # transformers' loss accumulator, and its value before the step that a skip restores.
    me._tr_loss, me._tr_loss_before_step = torch.zeros(()), torch.zeros(())
    if install:
        me._install_nonfinite_step_skip()
    return me


def _step(me: _Guarded, grad: float) -> None:
    """One optimizer step on gradients of ``grad``, in the trainer's order: norm, then step."""
    for param in me.model.parameters():
        param.grad = torch.full_like(param, grad)
    me._get_grad_norm(me.model, grad_norm=None)
    me.optimizer.step()
    me.state.global_step += 1


def _weights(me: _Guarded) -> list[torch.Tensor]:
    return [param.detach().clone() for param in me.model.parameters()]


def _moments(me: _Guarded) -> list[dict]:
    return [copy.deepcopy(me.optimizer.state[param]) for param in me.model.parameters()]


def _same(a: list[torch.Tensor], b: list[torch.Tensor]) -> bool:
    return len(a) == len(b) and all(torch.equal(x, y) for x, y in zip(a, b, strict=True))


def _same_moments(a: list[dict], b: list[dict]) -> bool:
    return all(
        x.keys() == y.keys() and all(torch.equal(torch.as_tensor(x[k]), torch.as_tensor(y[k])) for k in x)
        for x, y in zip(a, b, strict=True)
    )


@pytest.mark.parametrize("bad", [float("nan"), float("inf")], ids=["nan", "inf"])
def test_a_nonfinite_step_leaves_weights_and_moments_untouched(bad):
    me = _guarded()
    _step(me, 0.5)
    weights, moments = _weights(me), _moments(me)

    _step(me, bad)

    assert _same(weights, _weights(me)), "a non-finite step reached the weights"
    assert _same_moments(moments, _moments(me)), "a non-finite step reached the optimizer moments"
    assert all(param.grad is None for param in me.model.parameters()), "the skipped step's gradients survived"
    assert me._nonfinite_steps_skipped == 1 and me._nonfinite_streak == 1

    _step(me, 0.5)
    assert not _same(weights, _weights(me)), "the step after a skipped one was not applied"
    assert me._nonfinite_streak == 0


def test_the_skip_drops_a_deferred_clip_scale():
    """AdamWBF16 multiplies the clip coefficient in at its next step: one deferred for discarded gradients
    must not scale the next step's."""
    me = _guarded(optimizer_cls=AdamWBF16)
    _step(me, 0.5)
    weights = _weights(me)
    me.optimizer.defer_grad_scale(torch.tensor(0.25))

    _step(me, float("nan"))

    assert _same(weights, _weights(me)), "the step was not skipped"
    assert me.optimizer._grad_scale is None, "the discarded step's clip scale would reach the next step"


def test_consecutive_nonfinite_steps_raise_past_the_limit():
    me = _guarded(limit=2)
    _step(me, 0.5)
    weights = _weights(me)
    _step(me, float("nan"))
    _step(me, float("nan"))

    with pytest.raises(RuntimeError, match=r"steps 2-4: 3 consecutive steps .* weights are those after step 1"):
        _step(me, float("nan"))
    assert _same(weights, _weights(me))


def test_a_finite_step_resets_the_streak():
    me = _guarded(limit=1)
    for grad in (float("nan"), 0.5, float("nan"), 0.5, float("nan")):
        _step(me, grad)

    assert me._nonfinite_steps_skipped == 3 and me._nonfinite_streak == 1


def test_limit_zero_raises_at_the_first_nonfinite_step():
    me = _guarded(limit=0)

    with pytest.raises(RuntimeError, match="1 consecutive steps"):
        _step(me, float("nan"))


def test_nonfinite_weights_raise_at_once_naming_the_parameter():
    """A skip cannot repair weights that are already NaN: the run must stop, whatever the streak."""
    me = _guarded(limit=5)
    with torch.no_grad():
        me.model.bias[0] = float("nan")

    with pytest.raises(RuntimeError, match=r"1 parameter\(s\) already hold NaN/inf before the update: bias\."):
        _step(me, float("nan"))


def _scheduler_warnings(caught) -> list[str]:
    return [str(w.message) for w in caught if "lr_scheduler" in str(w.message) or "optimizer.step" in str(w.message)]


def test_the_lr_scheduler_wraps_the_guarded_step_without_warning():
    """torch's scheduler re-wraps ``optimizer.step`` through ``__func__`` and warns on a step it cannot
    track; ``create_optimizer`` installs the guard before the scheduler exists."""
    me = _guarded()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        scheduler = torch.optim.lr_scheduler.LambdaLR(me.optimizer, lambda step: 1.0 / (step + 1))
        _step(me, float("nan"))
        scheduler.step()
        _step(me, 0.5)
        scheduler.step()

    assert not _scheduler_warnings(caught)
    assert me._nonfinite_steps_skipped == 1 and scheduler.last_epoch == 2


def test_a_scheduler_built_before_the_guard_keeps_stepping_and_tracking():
    """A caller's ``optimizers=(optimizer, scheduler)`` wraps ``step`` before ``create_optimizer`` installs the guard:
    a skipped first step must still count as taken for the scheduler, and its wrapper's marker must survive."""
    me = _guarded(install=False)
    scheduler = torch.optim.lr_scheduler.LambdaLR(me.optimizer, lambda step: 1.0 / (step + 1))
    me._install_nonfinite_step_skip()
    weights = _weights(me)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _step(me, float("nan"))
        scheduler.step()
        assert _same(weights, _weights(me)), "a non-finite step reached the weights"
        _step(me, 0.5)
        scheduler.step()

    assert not _scheduler_warnings(caught)
    assert not _same(weights, _weights(me)), "the step after a skipped one was not applied"
    assert me._nonfinite_steps_skipped == 1 and scheduler.last_epoch == 2
    assert me.optimizer.param_groups[0]["lr"] == pytest.approx(0.1 / 3)


def test_a_loss_scaler_stepping_the_optimizer_owns_overflow_steps():
    """Under fp16 accelerate's optimizer wrapper steps through the scaler, which skips an overflow step
    itself and lowers its scale; the guard stays out."""
    me = _guarded(limit=0)
    me.optimizer = object.__new__(AcceleratedOptimizer)
    me.optimizer.scaler = object()

    assert me._step_is_finite(torch.tensor(float("inf"))) and me._nonfinite_steps_skipped == 0


def test_a_scaler_that_never_steps_the_optimizer_does_not_disarm_the_guard():
    """The toolkit's no-wrap prepare keeps the raw optimizer, so an accelerator scaler never skips for it."""
    me = _guarded(scaler=object())
    _step(me, 0.5)
    weights = _weights(me)

    _step(me, float("inf"))

    assert _same(weights, _weights(me)) and me._nonfinite_steps_skipped == 1


def test_a_negative_limit_is_refused_at_parse_time():
    with pytest.raises(ValueError, match="max_consecutive_nonfinite_steps must be an integer >= 0"):
        HaloArgumentParser((CommonScriptArguments,)).parse_dict({"max_consecutive_nonfinite_steps": -1})


def test_a_pending_skip_never_swallows_the_restore_step():
    """The zero-LR step that materializes optimizer state during a restore always steps."""
    me = _guarded()
    _step(me, 0.5)
    for param in me.model.parameters():
        param.grad = torch.full_like(param, float("nan"))
    me._get_grad_norm(me.model, grad_norm=None)
    me._restoring_optimizer_state = True
    calls = int(me.optimizer.state[next(me.model.parameters())]["step"])

    me.optimizer.step()

    assert int(me.optimizer.state[next(me.model.parameters())]["step"]) == calls + 1, "the restore step was skipped"
    assert me._skip_next_optimizer_step, "the verdict was consumed by a step it was not meant for"


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
    the weights around the step."""

    def __init__(self, model: _Tiny, step: int):
        self.model, self.step = model, step
        self.before = self.after = None

    def _weights(self):
        return [param.detach().clone() for param in self.model.parameters()]

    def on_step_begin(self, args, state, control, **kwargs):
        if state.global_step + 1 == self.step:
            self.before = self._weights()
            self.model.poison_at = len(self.model.losses) + args.gradient_accumulation_steps - 1

    def on_step_end(self, args, state, control, **kwargs):
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
):
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
        save_strategy="no",
        report_to=[],
        use_cpu=True,
        disable_tqdm=True,
        dataloader_num_workers=0,
        seed=0,
    )
    return _GuardedTrainer(model=model, args=args, train_dataset=_Rows(32), callbacks=[poison])


@pytest.mark.parametrize("max_grad_norm", [1.0, 0.0], ids=["clipped", "unclipped"])
def test_trainer_skips_the_poisoned_step_and_trains_on(tmp_path, max_grad_norm):
    poison_step, total = 3, 6
    model = _Tiny()
    poison = _PoisonOneStep(model, poison_step)
    trainer = _trainer(tmp_path, model, poison, total=total, max_grad_norm=max_grad_norm, logging_steps=1)

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
    assert result.training_loss == pytest.approx(sum(steps[s]["loss"] for s in steps if s != poison_step) / total), (
        "the run's mean loss must count every finite window and nothing of the skipped one"
    )


def test_an_unclipped_run_skips_a_poisoned_step_it_does_not_log(tmp_path):
    """At ``max_grad_norm <= 0`` transformers asks for the norm on every step, logged or not; a norm skipped on
    the steps that log nothing would let a poisoned one through unread."""
    total = 6
    model = _Tiny()
    poison = _PoisonOneStep(model, 3)
    trainer = _trainer(tmp_path, model, poison, total=total, max_grad_norm=0.0, logging_steps=4)

    result = trainer.train()

    assert poison.before is not None and _same(poison.before, poison.after), "the poisoned step was applied"
    assert trainer._nonfinite_steps_skipped == 1
    assert all(torch.isfinite(param).all() for param in model.parameters())
    assert math.isfinite(result.training_loss)


def test_a_skipped_last_step_in_an_unlogged_window_leaves_train_loss_finite(tmp_path):
    """transformers adds the unlogged accumulator to the run's mean at the end of training, without a log."""
    total = 6
    model = _Tiny()
    poison = _PoisonOneStep(model, total)
    trainer = _trainer(tmp_path, model, poison, total=total, max_grad_norm=1.0, logging_steps=10)

    result = trainer.train()

    assert poison.before is not None and _same(poison.before, poison.after), "the poisoned step was applied"
    assert math.isfinite(result.training_loss), "the skipped step's NaN reached the run's mean loss"


@pytest.mark.parametrize("accumulation", [1, 2], ids=["ga1", "ga2"])
@pytest.mark.parametrize("poison", sorted(_POISONS))
@pytest.mark.parametrize(
    ("logging_strategy", "logging_steps", "nan_inf_filter"),
    [("steps", 1, False), ("steps", 1, True), ("steps", 3, False), ("epoch", 1, False), ("no", 1, False)],
    ids=["steps1", "steps1-filter", "steps3", "epoch", "no"],
)
def test_a_skipped_step_removes_only_its_own_loss_from_train_loss(
    tmp_path, poison, logging_strategy, logging_steps, nan_inf_filter, accumulation
):
    """The run's mean and every logged window sum the applied steps' losses over all of their steps, whatever the
    window and whether the skipped step's loss was finite; a window with no applied step logs NaN, under
    ``logging_nan_inf_filter`` too. Under gradient accumulation only the step's last micro-batch is poisoned, and
    its finite earlier ones go with it."""
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
    assert result.training_loss == pytest.approx(sum(applied.values()) / total)
    first = 1
    for log in (log for log in trainer.state.log_history if "loss" in log):
        window = [applied[s] for s in range(first, log["step"] + 1) if s in applied]
        if window:
            assert log["loss"] == pytest.approx(sum(window) / (log["step"] - first + 1)), log
        else:
            assert math.isnan(log["loss"]), f"a window with no applied step must log NaN: {log}"
        first = log["step"] + 1


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


def test_the_distributed_trainer_skips_a_poisoned_step_it_does_not_log(tmp_path):
    """Through ``DistributedTrainerMixin.create_optimizer`` and the spine's composed ``_get_grad_norm``."""
    trainer, poison = _sft_trainer(tmp_path, poison_step=3)

    trainer.train()

    assert poison.before is not None and _same(poison.before, poison.after), "the poisoned step was applied"
    logged = {log["step"]: log for log in trainer.state.log_history if "loss" in log}
    assert logged[4]["nonfinite_steps_skipped"] == 1
    assert math.isfinite(logged[4]["loss"]), "the skipped step's NaN reached the window holding it"
    assert math.isfinite(logged[8]["loss"]), "the window after the skipped one logged the skipped step's NaN"
    assert all(torch.isfinite(param).all() for param in trainer.model.parameters())


def test_the_trainer_kwarg_sets_the_limit(tmp_path):
    trainer, _ = _sft_trainer(tmp_path, poison_step=3, max_consecutive_nonfinite_steps=0)

    with pytest.raises(RuntimeError, match=r"steps 3-3: 1 consecutive steps .* max_consecutive_nonfinite_steps=0"):
        trainer.train()


def test_the_distributed_trainer_guards_a_callers_prebuilt_scheduler(tmp_path):
    """``optimizers=(optimizer, scheduler)`` reaches ``create_optimizer`` with ``step`` already scheduler-wrapped."""
    built = {}

    def optimizers_for(model):
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
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
