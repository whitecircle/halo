"""CPU tests: the SFT trainer weighs the router aux loss at its coefficient, once per optimizer step.

HF's ``num_items_in_batch`` path multiplies a micro-batch loss by the process count and sums the
accumulation window. The language-model term needs that — it is a token sum over the window's global
count — but the aux term the model folds into the same loss is a per-micro-batch mean, and would weigh
``processes x accumulation`` times its coefficient. Under a label smoother or a ``compute_loss_func``
HF withholds the labels and the forward adds no aux term at all. Driven through the trainer's own
``training_step`` and ``prediction_step``; the aux term's share of a loss is the difference from the
same weights at ``router_aux_loss_coef=0``.
"""

import pytest
import torch
from accelerate import Accelerator
from datasets import Dataset
from trl import SFTConfig

from src.kernels.logprobs import LABEL_IGNORE_INDEX
from src.trainers.sft import DistributedSFTTrainer
from tests.common.offline_grpo import make_offline_tokenizer
from tests.common.parallelism import make_parallelism_config
from tests.common.tiny_models import TINY_MOE_FAMILIES, tiny_family_model

COEF = 1.0  # large, so a mis-weighted term is far above float noise
SEQ_LEN, PROMPT_LEN = 24, 8
# The model's own loss; HF popping the labels for its label smoother; TRL's loss function (DFT).
LOSS_PATHS = {"model": {}, "label_smoothing": {"label_smoothing_factor": 0.1}, "dft": {"loss_type": "dft"}}


def _trainer(tmp_path, processes: int, monkeypatch, coef: float = COEF, **loss_args) -> DistributedSFTTrainer:
    torch.manual_seed(0)
    # output_router_logits as the loader sets it for moe_balancing=aux_loss (apply_balancing_strategy)
    overrides = {"router_aux_loss_coef": coef, "output_router_logits": True}
    model = tiny_family_model(TINY_MOE_FAMILIES["qwen3_moe"], overrides=overrides)
    args = SFTConfig(
        output_dir=str(tmp_path),
        use_cpu=True,
        bf16=False,
        max_length=SEQ_LEN,
        per_device_train_batch_size=1,
        gradient_checkpointing=False,
        use_liger_kernel=False,
        remove_unused_columns=False,
        dataloader_num_workers=0,
        eval_strategy="no",
        save_strategy="no",
        report_to="none",
        **loss_args,
    )
    trainer = DistributedSFTTrainer(
        model=model,
        args=args,
        train_dataset=Dataset.from_dict({"input_ids": [list(range(3, 3 + SEQ_LEN))] * 2}),
        processing_class=make_offline_tokenizer(),
        parallelism_config=make_parallelism_config(world_size=1, gpus_per_node=1),
        moe_balancing="aux_loss",
    )
    monkeypatch.setattr(Accelerator, "num_processes", property(lambda self: processes))
    return trainer


def _row(trainer):
    generator = torch.Generator().manual_seed(1)
    input_ids = torch.randint(0, trainer.model.config.vocab_size, (1, SEQ_LEN), generator=generator)
    labels = input_ids.clone()
    labels[:, :PROMPT_LEN] = LABEL_IGNORE_INDEX
    return {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids), "labels": labels}


def _router_grads(model) -> dict[str, torch.Tensor]:
    grads = {n: p.grad.clone() for n, p in model.named_parameters() if ".gate." in n and p.grad is not None}
    assert grads, "the routers must receive gradient"
    return grads


def _aux(trainer, row) -> tuple[float, dict[str, torch.Tensor]]:
    """The row's raw router aux loss and its router gradient, apart from any language-model term."""
    trainer.model.zero_grad(set_to_none=True)
    aux = trainer.model(**row).aux_loss
    aux.backward()
    assert float(aux) > 0
    return float(aux), _router_grads(trainer.model)


def _step(trainer, micro_batches):
    """One optimizer step's window through ``training_step``: the summed loss and the router gradients."""
    trainer.model.zero_grad(set_to_none=True)
    trainer.current_gradient_accumulation_steps = len(micro_batches)
    count = torch.tensor(sum(int((b["labels"][:, 1:] != LABEL_IGNORE_INDEX).sum()) for b in micro_batches))
    loss = sum(float(trainer.training_step(trainer.model, dict(b), count)) for b in micro_batches)
    return loss, _router_grads(trainer.model)


@pytest.mark.parametrize("processes", [1, 2])
@pytest.mark.parametrize("loss_path", LOSS_PATHS)
def test_a_step_adds_the_aux_loss_once_at_its_coefficient(tmp_path, loss_path, processes, monkeypatch):
    """A window of one row and a window of two copies of it are the same step: the aux term adds
    ``coef x aux`` to either loss and ``coef x grad(aux)`` to the routers' gradient."""
    trainer = _trainer(tmp_path, processes, monkeypatch, **LOSS_PATHS[loss_path])
    no_aux = _trainer(tmp_path, processes, monkeypatch, coef=0.0, **LOSS_PATHS[loss_path])
    assert (trainer.label_smoother is not None) == (loss_path == "label_smoothing")
    assert (trainer.compute_loss_func is not None) == (loss_path == "dft")
    row = _row(trainer)
    aux, aux_grads = _aux(trainer, row)

    for window in ([row], [row, row]):
        loss, grads = _step(trainer, window)
        without_aux, without_aux_grads = _step(no_aux, window)
        assert loss - without_aux == pytest.approx(COEF * aux, rel=1e-4), f"window of {len(window)}"
        for name, grad in aux_grads.items():
            torch.testing.assert_close(grads[name] - without_aux_grads[name], COEF * grad, rtol=1e-4, atol=1e-6)


@pytest.mark.parametrize("processes", [1, 2])
def test_an_eval_batch_adds_the_aux_loss_at_its_coefficient(tmp_path, processes, monkeypatch):
    """``eval_on_start`` evaluates before any accumulation window exists."""
    trainer = _trainer(tmp_path, processes, monkeypatch)
    no_aux = _trainer(tmp_path, processes, monkeypatch, coef=0.0)
    assert not hasattr(trainer, "current_gradient_accumulation_steps")
    row = _row(trainer)
    aux, _ = _aux(trainer, row)

    losses = []
    for t in (trainer, no_aux):
        t.model.eval()  # as the evaluation loop leaves it for prediction_step
        losses.append(float(t.prediction_step(t.model, dict(row), prediction_loss_only=True)[0]))
    loss, without_aux = losses

    assert loss - without_aux == pytest.approx(COEF * aux, rel=1e-4)


def test_without_the_rescale_the_aux_loss_weighs_processes_times_window(tmp_path, monkeypatch):
    """Anti-vacuity: HF's path alone counts the term once per micro-batch, times the process count."""
    monkeypatch.setattr(DistributedSFTTrainer, "_weigh_router_aux_loss", lambda self, model, loss, *_: loss)
    trainer = _trainer(tmp_path, 2, monkeypatch)
    no_aux = _trainer(tmp_path, 2, monkeypatch, coef=0.0)
    row = _row(trainer)
    aux, _ = _aux(trainer, row)

    loss, _ = _step(trainer, [row, row])
    without_aux, _ = _step(no_aux, [row, row])

    assert loss - without_aux == pytest.approx(2 * 2 * COEF * aux, rel=1e-4)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
