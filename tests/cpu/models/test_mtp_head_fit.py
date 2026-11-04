#!/usr/bin/env python
"""What a fit learns says what the head reads. On a random stream the token two ahead is unpredictable, so a held-out
top-1 near chance is the true objective's mark. A head scored one position short fits to the token whose embedding it
reads, the input-copy trap: its loss falls and it reaches top-1 1.0 on that wrong target with chance on the true one.
That the trap fits inside this step budget is what gives the first test its power."""

import pytest
import torch

from halo.models.mtp.head import freeze_trunk_for_head
from tests.common.mtp_heads import install_term, tiny_model_with_head

_STEPS = 80
_LEARNING_RATE = 3e-3
_LOW_TOKEN, _HIGH_TOKEN = 4, 90
_CHANCE_BAR = 0.1
_FIT_BAR = 0.9
_NAMES = ("qwen3_5_moe_text", "glm4_moe_lite", "nemotron_h")


def _tokens(seed: int, rows: int = 16, length: int = 24) -> torch.Tensor:
    return torch.randint(_LOW_TOKEN, _HIGH_TOKEN, (rows, length), generator=torch.Generator().manual_seed(seed))


def _fit(name: str, labels_of):
    """A ``head_only`` hard fit on fresh random rows; returns the model, the head and each step's head cross-entropy."""
    model, head = tiny_model_with_head(name, dtype=torch.float32)
    freeze_trunk_for_head(model, head)
    install_term(model, head, weight=1.0, loss_type="hard", scope="head_only")
    optimizer = torch.optim.AdamW(head.parameters(), lr=_LEARNING_RATE)
    start = {key: value.detach().clone() for key, value in head.named_parameters()}
    losses = []
    for step in range(_STEPS):
        input_ids = _tokens(step)
        output = model(input_ids=input_ids, labels=labels_of(input_ids), use_cache=False)
        optimizer.zero_grad()
        output.loss.backward()
        optimizer.step()
        losses.append(float(output.mtp_loss_split.head_sum / output.mtp_loss_split.head_targets))
    moved = [key for key, value in head.named_parameters() if not torch.equal(value, start[key])]
    assert moved, "the head took no step"
    return model, head, losses


def _held_out_top1(model, head) -> dict[str, float]:
    """On rows the fit never saw: top-1 against the token the head reads and against the true target."""
    input_ids = _tokens(10_000)
    seen = []
    handle = head.register_forward_hook(lambda _module, _args, output: seen.append(output.detach()))
    with torch.no_grad():
        model(input_ids=input_ids, labels=input_ids, use_cache=False)
    handle.remove()
    predicted = model.get_output_embeddings()(seen[0]).argmax(-1)
    return {
        "read_token": float((predicted[:, :-1] == input_ids[:, 1:]).float().mean()),
        "true_target": float((predicted[:, :-2] == input_ids[:, 2:]).float().mean()),
    }


def _one_position_short(input_ids: torch.Tensor) -> torch.Tensor:
    """Labels whose token two ahead is the next one, which is what an off-by-one target would score."""
    return torch.roll(input_ids, 1, dims=1)


@pytest.mark.parametrize("name", _NAMES)
def test_the_true_objective_stays_at_chance_on_tokens_it_cannot_predict(name):
    model, head, _losses = _fit(name, lambda input_ids: input_ids)
    top1 = _held_out_top1(model, head)
    assert top1["true_target"] < _CHANCE_BAR, f"the head reads its own target: {top1}"
    assert top1["read_token"] < _CHANCE_BAR, f"the head copies its input: {top1}"


@pytest.mark.parametrize("name", _NAMES)
def test_a_target_one_position_short_fits_to_the_token_the_head_reads(name):
    model, head, losses = _fit(name, _one_position_short)
    top1 = _held_out_top1(model, head)
    assert losses[-1] < 0.7 * losses[0], f"the trap's loss should fall: {losses[0]} -> {losses[-1]}"
    assert top1["read_token"] > _FIT_BAR, f"the trap did not fit inside the step budget: {top1}"
    assert top1["true_target"] < _CHANCE_BAR, f"the trap learned the true target too: {top1}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
