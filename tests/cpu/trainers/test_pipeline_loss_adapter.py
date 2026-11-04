#!/usr/bin/env python
"""The trainer's pipeline-loss contract: inert defaults, and which objectives fuse into the last stage.
Fusing hands the last stage the causal-LM CE in token chunks, so a loss reading the assembled output must never fuse.
"""

import pytest

from halo.distributed.pipeline_parallel.stage_losses import FusedHeadLoss, TokenLoss, causal_lm_token_loss
from halo.trainers.parallelism.pipeline_loss_adapter import PipelineLossAdapter


def _sequence_loss(logits, target):
    return logits.sum()


def test_adapter_defaults_are_inert():
    adapter = PipelineLossAdapter(token_loss_fn=_sequence_loss)
    assert adapter.batch_transform is None and adapter.normalizer is None
    assert adapter.extra_target_keys == () and adapter.pad_spec is None


def test_the_causal_lm_contract_fuses_when_asked():
    assert PipelineLossAdapter(token_loss_fn=causal_lm_token_loss).stage_loss(fuse_head=True) == FusedHeadLoss()


@pytest.mark.parametrize(
    "adapter",
    [
        PipelineLossAdapter(token_loss_fn=_sequence_loss),
        PipelineLossAdapter(token_loss_fn=causal_lm_token_loss, predictions_fn=lambda outputs, inputs: outputs),
    ],
    ids=["own-loss", "reduced-predictions"],
)
def test_an_objective_reading_the_assembled_output_never_fuses(adapter):
    assert adapter.stage_loss(fuse_head=True) == TokenLoss(adapter.token_loss_fn)


def test_the_knob_off_keeps_the_logits_path():
    adapter = PipelineLossAdapter(token_loss_fn=causal_lm_token_loss)
    assert adapter.stage_loss(fuse_head=False) == TokenLoss(causal_lm_token_loss)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
