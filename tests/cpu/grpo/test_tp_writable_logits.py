"""TRL's full-logits GRPO path divides the sliced logits by the temperature in place. A TP-gathered head on
transformers' DTensor forward returns a view autograd forbids writing, so ``_writable_logits`` clones that head's
output; a head on the local-shard forward returns a fresh tensor and needs no clone.

    python tests/cpu/grpo/test_tp_writable_logits.py
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from torch.distributed.tensor import Shard, distribute_tensor
from transformers.distributed.tensor_parallel import ALL_PARALLEL_STYLES

from halo.distributed.tensor_parallel import local_linears
from halo.trainers.grpo.logprob_scoring import _tp_gathered_output_head
from tests.common.distributed import fake_process_group_mesh

TP = 2
STYLE = "colwise_gather_output"


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.lm_head = nn.Linear(8, 16)
        self._tp_plan = {"lm_head": STYLE}

    def get_output_embeddings(self):
        return self.lm_head


def _sharded_model(mesh) -> _Model:
    model = _Model()
    for pname in ("weight", "bias"):
        param = getattr(model.lm_head, pname)
        setattr(model.lm_head, pname, nn.Parameter(distribute_tensor(param, mesh, [Shard(0)], src_data_rank=None)))
    return model


def _divide_in_place(head: nn.Module) -> None:
    logits = head(torch.randn(2, 3, 8, requires_grad=True))
    logits[:, :-1].div_(0.7)
    logits.sum().backward()


def test_a_local_shard_head_is_not_cloned_and_its_logits_take_the_in_place_divide():
    with fake_process_group_mesh(rank=0, world_size=TP) as mesh:
        model = _sharded_model(mesh)
        local_linears.install_local_forward(model.lm_head, STYLE)
        assert _tp_gathered_output_head(model) is None
        _divide_in_place(model.lm_head)


def test_a_head_on_transformers_forward_is_cloned():
    """Premise and gate together: transformers' gathered output refuses the divide, so that head keeps the clone."""
    with fake_process_group_mesh(rank=0, world_size=TP) as mesh:
        model = _sharded_model(mesh)
        ALL_PARALLEL_STYLES[STYLE].install_forward(model.lm_head, mesh)
        assert _tp_gathered_output_head(model) is model.lm_head
        with pytest.raises(RuntimeError, match="modified inplace"):
            _divide_in_place(model.lm_head)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
