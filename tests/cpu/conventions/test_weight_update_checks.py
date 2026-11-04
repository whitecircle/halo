#!/usr/bin/env python
"""``tests.common.training_checks.weight_update_checks`` holds a full fine-tune's tensors to being stepped and
moved. At the rows' rate of ``1e-5`` a bf16 element moves only below ``2**-8``, so norm gains near 1 stay
bit-identical in a healthy run and only the optimizer's first moment shows they train."""

import json
import types

import pytest
import torch
import transformers
from torch.distributed.device_mesh import init_device_mesh

from halo.distributed.fsdp import IdentityParamSet, apply_fsdp2_per_layer, mixed_precision_policy
from halo.optimizers.adamw_bf16 import adamw_bf16_optimizer_from_args
from tests.common.gloo import run_gloo_ranks
from tests.common.tiny_models import TINY_DENSE_FAMILIES, tiny_family_model
from tests.common.training_checks import (
    movable_elements,
    movable_tensors,
    stepped_parameters,
    weight_update_checks,
)
from tests.common.weight_sync import local_parameters, moved_parameters

WORLD = 2
LEARNING_RATE = 1e-5
STEPS = 3
TOKENS = 16
VARIANTS = ("healthy", "no_decay_group_unstepped", "norm_gradients_dropped", "zero_learning_rate")
# gloo reduces in fp32; the rows' bf16 reduce changes no gradient's sign or presence.
POLICY = mixed_precision_policy(torch.bfloat16, fp32_reduce=True, cast_forward_inputs=True)
# Qwen3-0.6B's largest ``k_norm`` gain.
GAIN = 96.5


def _in_layer0(name: str, param: torch.nn.Parameter) -> bool:
    return param.requires_grad and ".layers.0." in name


def _train(rank: int, variant: str) -> dict:
    """Three steps of ``variant`` on this rank's FSDP2 shards; what the checks read and return."""
    torch.manual_seed(0)
    model = tiny_family_model(TINY_DENSE_FAMILIES["qwen3"]).to(torch.bfloat16)
    apply_fsdp2_per_layer(model, init_device_mesh("cpu", (WORLD,)), POLICY, False, IdentityParamSet())
    decay = transformers.Trainer.get_decay_parameter_names(None, model)
    rate = 0.0 if variant == "zero_learning_rate" else LEARNING_RATE
    args = types.SimpleNamespace(
        learning_rate=rate, adam_beta1=0.9, adam_beta2=0.999, adam_epsilon=1e-8, weight_decay=0.0
    )
    optimizer = adamw_bf16_optimizer_from_args(model, args, decay)
    before = local_parameters(model, keep=_in_layer0)
    norms = set(before) - set(decay)
    if variant == "no_decay_group_unstepped":
        del optimizer.param_groups[1]
    generator = torch.Generator().manual_seed(rank)
    for _ in range(STEPS):
        ids = torch.randint(0, model.config.vocab_size, (1, TOKENS), generator=generator)
        model(input_ids=ids, labels=ids).loss.backward()
        if variant == "norm_gradients_dropped":
            for group in optimizer.param_groups:
                for param, name in zip(group["params"], group["param_names"], strict=True):
                    if name in norms:
                        param.grad = None
        optimizer.step()
        optimizer.zero_grad()
    moved = set(moved_parameters(before, local_parameters(model, keep=_in_layer0)))
    return {
        "checks": weight_update_checks(model, optimizer, before, moved, LEARNING_RATE),
        "names": sorted(before),
        "norms": sorted(norms),
        "stepped": sorted(stepped_parameters(model, optimizer, before)),
        "movable": sorted(movable_tensors(before, LEARNING_RATE)),
        "moved": sorted(moved),
    }


def _worker(rank: int, out_dir: str) -> None:
    results = {variant: _train(rank, variant) for variant in VARIANTS}
    with open(f"{out_dir}/rank{rank}.json", "w") as f:
        json.dump(results, f)


@pytest.fixture(scope="module")
def runs(tmp_path_factory) -> list[dict]:
    out_dir = tmp_path_factory.mktemp("weight_update_checks")
    run_gloo_ranks(_worker, WORLD, str(out_dir))
    return [json.loads((out_dir / f"rank{rank}.json").read_text()) for rank in range(WORLD)]


def test_spacing_counts_follow_the_dtype():
    gains = torch.full((128,), GAIN)
    assert movable_elements(gains.to(torch.bfloat16), LEARNING_RATE) == 0
    # fp32 spaces 96.5 by 7.6e-6, which an update of 1e-5 crosses half of.
    assert movable_elements(gains, LEARNING_RATE) == gains.numel()
    # bf16 spaces values under 2**-8 finer than 2e-5, and zero finest of all; gains near 1 not at all.
    small = torch.tensor([0.0, 1e-3, 2**-9, 2**-8, 1e-2, 1.0], dtype=torch.bfloat16)
    assert movable_elements(small, LEARNING_RATE) == 3


def test_a_healthy_run_steps_every_tensor_and_moves_the_movable_ones(runs):
    for rank in runs:
        run = rank["healthy"]
        assert run["checks"] == {"weights_stepped": True, "movable_weights_moved": True}
        assert run["stepped"] == run["names"]
        # The bit-for-bit check alone cannot hold the norms: no element of theirs is within one update's reach.
        assert run["norms"] and not set(run["norms"]) & set(run["movable"])
        assert set(run["movable"]) == set(run["names"]) - set(run["norms"])


@pytest.mark.parametrize("variant", ["no_decay_group_unstepped", "norm_gradients_dropped"])
def test_norms_the_optimizer_never_steps_fail(runs, variant):
    """The norms left out of the optimizer, or reached by no gradient, fail ``weights_stepped`` by name, which the
    movable tensors' bit-for-bit check passes."""
    for rank in runs:
        run = rank[variant]
        assert run["checks"] == {"weights_stepped": False, "movable_weights_moved": True}
        assert sorted(set(run["names"]) - set(run["stepped"])) == run["norms"]


def test_a_zero_learning_rate_fails_the_movable_tensors(runs):
    """Gradients reach every tensor and the optimizer steps it, but nothing moves."""
    for rank in runs:
        run = rank["zero_learning_rate"]
        assert run["checks"] == {"weights_stepped": True, "movable_weights_moved": False}
        assert run["movable"] and not run["moved"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
