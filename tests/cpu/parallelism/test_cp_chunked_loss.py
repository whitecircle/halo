#!/usr/bin/env python
"""The CP logits loss runs its fp32 CE over checkpointed chunks (``halo.kernels.logprobs.chunked_token_sum``). A
whole fp32 copy plus the saved log-softmax is two planes, too much for a long sequence."""

from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F
from accelerate import PartialState
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_leaves

from halo.distributed.context_parallel.sequence_chunks import cp_shift_against_full_labels
from halo.distributed.context_parallel.ulysses_model import UlyssesModel
from halo.distributed.parallelism_config import ParallelismConfig
from halo.kernels import logprobs
from halo.kernels.logprobs import LABEL_IGNORE_INDEX
from tests.common.cp_model import unpatched_cp_wrapper
from tests.common.ep_reference import random_token_batch
from tests.common.equivalence.cp_equivalence import BATCH, CP_SIZE, SEED, SEQ, tier_controls, tiny_model
from tests.common.gloo import run_gloo_ranks

# Several chunks per rank's [BATCH * SEQ / CP_SIZE] rows.
CHUNK_ROWS = 24
VOCAB, ROWS = 96, 200
# The two forms sum the same fp32 terms in a different order.
RELATIVE_TOLERANCE = 1e-5


def _unchunked_cp_loss(self: UlyssesModel, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """One fp32 copy of the shifted logits and one cross-entropy over it."""
    shift_logits, shift_labels = cp_shift_against_full_labels(
        logits, labels, self.cp_config.cp_rank, self.cp_config.cp_size
    )
    shift_labels = shift_labels.reshape(-1)
    local_loss_sum = F.cross_entropy(
        shift_logits.reshape(-1, logits.size(-1)).float(),
        shift_labels,
        ignore_index=LABEL_IGNORE_INDEX,
        reduction="sum",
    )
    return self._cp_mean(local_loss_sum, shift_labels)


def _step(cp_config) -> dict:
    model = tiny_model("qwen3", "cpu", torch.device("cpu"))
    wrapper = UlyssesModel(model, cp_config).train()
    ids, labels = random_token_batch(model.get_input_embeddings().num_embeddings, BATCH, SEQ, "cpu", seed=SEED)
    labels[:, ::5] = LABEL_IGNORE_INDEX
    loss = wrapper(input_ids=ids, labels=labels).loss
    loss.backward()
    return {
        "loss": loss.detach(),
        "grads": {n: p.grad.detach().clone() for n, p in wrapper.named_parameters() if p.grad is not None},
    }


def _record_worker(rank: int, output: str) -> None:
    PartialState()
    cp_config = ParallelismConfig(cp_size=CP_SIZE).cp_config()
    vocab = tiny_model("qwen3", "cpu", torch.device("cpu")).get_input_embeddings().num_embeddings
    with tier_controls("cpu"), patch.object(logprobs, "LOGIT_CHUNK_ELEMENTS", CHUNK_ROWS * vocab):
        runs = {"chunked": _step(cp_config)}
        with patch.object(UlyssesModel, "_compute_cp_loss", _unchunked_cp_loss):
            runs["unchunked"] = _step(cp_config)
    torch.save(runs, Path(output, f"rank{rank}.pt"))


def _gap(value: torch.Tensor, reference: torch.Tensor) -> float:
    return ((value - reference).abs().max() / reference.abs().max().clamp(min=1e-12)).item()


def test_the_chunked_loss_matches_the_unchunked_one_on_two_gloo_ranks(tmp_path):
    run_gloo_ranks(_record_worker, CP_SIZE, str(tmp_path))
    for rank in range(CP_SIZE):
        runs = torch.load(tmp_path / f"rank{rank}.pt", weights_only=False)
        chunked, unchunked = runs["chunked"], runs["unchunked"]
        assert unchunked["loss"] > 0
        assert _gap(chunked["loss"], unchunked["loss"]) <= RELATIVE_TOLERANCE, f"rank {rank}"
        assert chunked["grads"].keys() == unchunked["grads"].keys()
        gaps = {n: _gap(g, unchunked["grads"][n]) for n, g in chunked["grads"].items()}
        assert max(gaps.values()) <= RELATIVE_TOLERANCE, f"rank {rank}: {max(gaps, key=gaps.get)}"


class _LargestFp32(TorchDispatchMode):
    """Backward included."""

    def __init__(self):
        super().__init__()
        self.numel = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        output = func(*args, **(kwargs or {}))
        for tensor in tree_leaves(output):
            if isinstance(tensor, torch.Tensor) and tensor.dtype == torch.float32:
                self.numel = max(self.numel, tensor.numel())
        return output


def _largest_fp32_of_a_loss_step(cp_rank: int) -> int:
    wrapper = unpatched_cp_wrapper(torch.nn.Identity(), cp_size=CP_SIZE, cp_rank=cp_rank)
    generator = torch.Generator().manual_seed(cp_rank)
    logits = torch.randn(1, ROWS, VOCAB, generator=generator).to(torch.bfloat16).requires_grad_(True)
    labels = torch.randint(0, VOCAB, (1, ROWS * CP_SIZE), generator=generator)
    with _LargestFp32() as largest:
        wrapper._compute_cp_loss(logits, labels).backward()
    return largest.numel


@pytest.mark.parametrize("cp_rank", range(CP_SIZE))
def test_the_loss_builds_no_fp32_plane_beyond_one_chunk(monkeypatch, cp_rank):
    monkeypatch.setattr(logprobs, "LOGIT_CHUNK_ELEMENTS", CHUNK_ROWS * VOCAB)
    assert _largest_fp32_of_a_loss_step(cp_rank) <= CHUNK_ROWS * VOCAB

    monkeypatch.setattr(UlyssesModel, "_compute_cp_loss", _unchunked_cp_loss)
    assert _largest_fp32_of_a_loss_step(cp_rank) >= (ROWS - 1) * VOCAB, "premise: the unchunked form builds the plane"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
