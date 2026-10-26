#!/usr/bin/env python
"""Expert-load recording counts each microbatch EXACTLY ONCE, in every checkpoint mode.

``_record_expert_load`` feeds the bias-update balancer's per-step counts, so the gate has to hold
across the whole matrix:

* no gradient checkpointing — the single forward counts;
* REENTRANT checkpointing (forced for every non-PP EP/CP run) — the original forward runs under
  ``no_grad``, so a bare grad gate would record NOTHING and the bias would never move; the recompute
  must not count it a second time;
* non-reentrant checkpointing (the PP mode) — both passes run with grad enabled, so only the scope's
  pass counter separates them;
* eval and a genuine ``no_grad`` pass outside any checkpoint — a frozen reference or teacher scoring
  through the same modules drives no optimizer step, so counting it would skew the balance.

The hub-router Zaya patch gates through the same predicate.

    python tests/cpu/parallelism/test_expert_load_recording.py
"""

import types
from unittest import mock

import pytest
import torch
import torch.utils.checkpoint as torch_checkpoint
from transformers import Trainer

from halo.distributed.expert_parallel.moe_layer.layer import EPMoELayerBase
from halo.models.attention.tree_plan import TREE_ATTENTION_KEY
from halo.models.gradient_checkpointing import scoped_checkpoint_func
from halo.models.moe.balancing import expert_load_token_weights
from halo.trainers.sft import DistributedSFTTrainer
from halo.trainers.world_metrics import WorldMetrics

_INDICES = torch.tensor([0, 1, 1, 3])
_EXPECTED = torch.tensor([1.0, 2.0, 0.0, 1.0])


def _stub(num_experts: int = 4):
    """EP-layer stand-in carrying only what ``_record_expert_load`` reads."""
    return types.SimpleNamespace(
        balancing_biases=torch.zeros(num_experts),
        training=True,
        num_experts=num_experts,
        expert_load_counter=None,
    )


def _checkpointed_step(stub, *, use_reentrant: bool) -> torch.Tensor:
    """Run one forward+backward through a checkpointed body that records a load, as training does.

    The body is wrapped with the toolkit's own :func:`scoped_checkpoint_func` — the same wrapper
    ``install_recompute_scopes`` installs on every module after ``gradient_checkpointing_enable``
    — so both passes share one scope, exactly as they do in a real run.
    """
    checkpoint_func = scoped_checkpoint_func(
        lambda function, *args: torch_checkpoint.checkpoint(function, *args, use_reentrant=use_reentrant)
    )

    def body(x):
        EPMoELayerBase._record_expert_load(stub, _INDICES)
        return x * 2

    x = torch.ones(2, requires_grad=True)
    out = checkpoint_func(body, x)
    out.sum().backward()
    return x.grad


def test_training_forward_counts():
    stub = _stub()
    EPMoELayerBase._record_expert_load(stub, _INDICES)
    assert torch.equal(stub.expert_load_counter, _EXPECTED)


def test_token_weights_count_a_shared_token_once_per_example():
    """A tree-packed row routes a shared prefix token once for the three examples that contain it;
    flat packing would have routed it three times. The balancer must see the flat load."""
    selections = torch.tensor([[0, 1], [1, 3], [2, 3]])  # three tokens, top-2
    stub = _stub()
    with expert_load_token_weights(torch.tensor([3, 1, 1])):
        EPMoELayerBase._record_expert_load(stub, selections)
    assert stub.expert_load_counter.tolist() == [3.0, 4.0, 1.0, 2.0]

    EPMoELayerBase._record_expert_load(stub, selections)
    assert stub.expert_load_counter.tolist() == [4.0, 6.0, 2.0, 4.0], "outside the block every token counts once"


def _sft_step_load(inputs: dict) -> list[float]:
    """Expert load recorded by a forward the SFT trainer's own ``compute_loss`` drives."""
    stub = _stub()

    def routed(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        EPMoELayerBase._record_expert_load(stub, torch.tensor([[0, 1], [1, 3], [2, 3]]))
        return (torch.zeros(()), {}) if return_outputs else torch.zeros(())

    trainer = object.__new__(DistributedSFTTrainer)
    trainer.__dict__.update(
        _warned_empty_labels=True,
        parallelism_config=types.SimpleNamespace(is_cp_mode=False, non_dp_replication_factor=1),
        model=types.SimpleNamespace(training=True),
        args=types.SimpleNamespace(use_liger_kernel=False),
        aux_loss_enabled=False,
        _loss_metrics={"train": WorldMetrics(), "eval": WorldMetrics()},
    )
    with mock.patch.object(Trainer, "compute_loss", routed):
        DistributedSFTTrainer.compute_loss(trainer, None, {"input_ids": torch.zeros(1, 3, dtype=torch.long), **inputs})
    return stub.expert_load_counter.tolist()


def test_the_sft_trainer_weights_a_tree_batch_by_its_leaf_count():
    assert _sft_step_load({TREE_ATTENTION_KEY: {"leaf_count": torch.tensor([3, 1, 1])}}) == [3.0, 4.0, 1.0, 2.0]
    assert _sft_step_load({}) == [1.0, 2.0, 1.0, 2.0]


def test_token_weights_for_another_forward_are_refused():
    """Weights that do not cover this forward's tokens would be scattered onto the wrong ones."""
    with expert_load_token_weights(torch.ones(5)), pytest.raises(ValueError, match="do not describe this forward"):
        EPMoELayerBase._record_expert_load(_stub(), torch.tensor([[0, 1], [1, 3], [2, 3]]))


@pytest.mark.parametrize("use_reentrant", [True, False])
def test_checkpointed_step_counts_exactly_once(use_reentrant):
    """Reentrant checkpointing runs its ORIGINAL forward under no_grad, so a
    ``torch.is_grad_enabled()`` gate records nothing at all — and every shipped balancing config
    sets ``gradient_checkpointing: true``, so the bias never moves. The recompute must still not
    double-count."""
    stub = _stub()
    grad = _checkpointed_step(stub, use_reentrant=use_reentrant)
    assert grad is not None, "premise: the checkpointed body actually ran its backward"
    assert stub.expert_load_counter is not None, "the original checkpoint pass must record the load"
    assert torch.equal(stub.expert_load_counter, _EXPECTED), (
        "expected one count per microbatch — the recompute pass must not count its tokens again"
    )


def test_no_grad_pass_outside_a_checkpoint_does_not_count():
    stub = _stub()
    with torch.no_grad():
        EPMoELayerBase._record_expert_load(stub, _INDICES)
    assert stub.expert_load_counter is None, "a frozen reference/teacher pass must not skew the balance"


def test_eval_mode_does_not_count():
    stub = _stub()
    stub.training = False
    EPMoELayerBase._record_expert_load(stub, _INDICES)
    assert stub.expert_load_counter is None


def test_no_grad_forward_through_a_checkpointed_block_does_not_count():
    """Offline GRPO's KL reference pass runs ``torch.no_grad()`` through a TRAIN-mode GC model —
    HF gates checkpointing on ``self.training``, not grad mode, so the checkpoint scope is still
    entered. The scope must carry the OUTER grad mode: counting here folds the reference forward's
    routing into the same counter the bias update consumes, balancing a policy+reference mixture."""
    stub = _stub()
    checkpoint_func = scoped_checkpoint_func(
        lambda function, *args: torch_checkpoint.checkpoint(function, *args, use_reentrant=False)
    )

    def body(x):
        EPMoELayerBase._record_expert_load(stub, _INDICES)
        return x * 2

    with torch.no_grad():
        checkpoint_func(body, torch.ones(2))
    assert stub.expert_load_counter is None, (
        "a no_grad reference pass through a checkpointed block must not record expert load"
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
