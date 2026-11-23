"""The reward trainer's PP batch transform treats the margin column symmetrically: a missing margin raises where
zero-filling would silently train the margin-free objective, and an unexpected one raises too."""

import pytest
import torch

from halo.trainers.reward.bradley_terry import DistributedRewardTrainer


def _batch(with_margin: bool) -> dict:
    input_ids = torch.tensor([[5, 6, 0], [7, 8, 9], [1, 2, 0], [3, 4, 5]])
    batch = {"input_ids": input_ids, "attention_mask": (input_ids != 0).long()}
    if with_margin:
        batch["margin"] = torch.tensor([0.25, 0.5])
    return batch


def _trainer(use_margins: bool) -> DistributedRewardTrainer:
    trainer = object.__new__(DistributedRewardTrainer)
    trainer._pp_use_margins = use_margins
    trainer._pp_pool_pad_id = 0
    return trainer


def test_reward_pp_transform_raises_on_a_missing_margin():
    with pytest.raises(ValueError, match="margin"):
        _trainer(True)._pp_reward_batch_transform(_batch(with_margin=False))


def test_reward_pp_transform_interleaves_a_present_margin():
    out = _trainer(True)._pp_reward_batch_transform(_batch(with_margin=True))
    # One margin per pair, repeated onto both interleaved rows of that pair.
    assert torch.equal(out["margin"], torch.tensor([0.25, 0.25, 0.5, 0.5]))


def test_reward_pp_transform_still_rejects_an_unexpected_margin():
    with pytest.raises(ValueError, match="margin"):
        _trainer(False)._pp_reward_batch_transform(_batch(with_margin=True))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
