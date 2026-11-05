#!/usr/bin/env python
"""A loss forward armed with its round's row spans reads the attention mask without a host sync on CUDA.

The unarmed forward copies the mask to the host for its spans, which ``set_sync_debug_mode`` flags: the negative
control. A table built from another mask is tallied on the device and read at the window's end.
"""

import torch
from transformers import PretrainedConfig

from halo.models.head_transform import IDENTITY_HEAD_TRANSFORM
from halo.trainers.grpo.logprob_scoring import ChunkedGRPOLogprobsMixin, row_spans_on_device
from tests.common.harness import cuda_syncs_are_errors, gpu_test_main, record_check
from tests.common.rank_log import log

LTK = 4
HIDDEN = 8
VOCAB = 40


class _Scorer(ChunkedGRPOLogprobsMixin):
    """A position-free embedding backbone at batch size 1, so every row forwards densely, and a head sweep that is a
    plain reduction: only the span reads can synchronize."""

    temperature = 1.0

    def __init__(self, device: torch.device):
        self.embed = torch.nn.Embedding(VOCAB, HIDDEN).to(device)
        self.model = torch.nn.Module()
        self.model.config = PretrainedConfig()
        self.model.config._attn_implementation = "sdpa"

    def _head_transform(self, unwrapped_model):
        return IDENTITY_HEAD_TRANSFORM

    def _head_sweep(self, unwrapped_model):
        return lambda hidden, targets, compute_entropy=False: (hidden.sum(-1), None)

    def _get_last_hidden_state(self, model, input_ids, attention_mask, logits_to_keep):
        return self.embed(input_ids)[:, :-1, :][:, -logits_to_keep:, :]

    def score(self, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self._chunked_logps_impl(self.model, ids, mask, LTK, 1, False)[0]


def _batch(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """TRL's layout: prompts left-padded, completions right-padded, each at least one token."""
    lengths = torch.tensor([[3, 2], [5, 4], [1, 1], [8, 3], [2, 4], [6, 1]])
    prompt = torch.arange(8).expand(6, 8) >= (8 - lengths[:, :1])
    completion = torch.arange(LTK).expand(6, LTK) < lengths[:, 1:]
    mask = torch.cat([prompt, completion], dim=1).long().to(device)
    return torch.randint(1, VOCAB, mask.shape, device=device), mask


def check_an_armed_forward_adds_no_host_sync(scorer: _Scorer, ids: torch.Tensor, mask: torch.Tensor):
    scorer._armed_row_spans = row_spans_on_device(mask, LTK)[:, :3].cpu()
    scorer.score(ids, mask)
    torch.cuda.synchronize()
    with cuda_syncs_are_errors():
        scorer.score(ids, mask)
    assert scorer._row_span_mismatches is not None and int(scorer._row_span_mismatches) == 0


def check_an_unarmed_forward_trips_the_sync_check(scorer: _Scorer, ids: torch.Tensor, mask: torch.Tensor):
    """Negative control: without a table the forward copies the mask to the host."""
    scorer._armed_row_spans = None
    with cuda_syncs_are_errors():
        try:
            scorer.score(ids, mask)
        except RuntimeError as error:
            assert "synchronizing" in str(error), error
        else:
            raise AssertionError("the unarmed forward passed the sync check: the check is blind")


def check_a_table_of_another_mask_is_tallied_per_row(scorer: _Scorer, ids: torch.Tensor, mask: torch.Tensor):
    table = row_spans_on_device(mask, LTK)[:, :3].cpu()
    table[1, 1] += 1
    table[4, 0] -= 1
    scorer._armed_row_spans = table
    scorer._row_span_mismatches = None
    with cuda_syncs_are_errors():
        scorer.score(ids, mask)
    assert int(scorer._row_span_mismatches) == 2


@gpu_test_main(exact_world_size=1, prefix="test_row_spans_sync_free", partial_state=False)
def run(ctx) -> dict:
    torch.cuda.set_device(ctx.device)
    log(f"GPU: {torch.cuda.get_device_name(ctx.local_rank)}")
    scorer = _Scorer(ctx.device)
    ids, mask = _batch(ctx.device)
    checks: dict[str, bool] = {}
    for check in (
        check_an_armed_forward_adds_no_host_sync,
        check_an_unarmed_forward_trips_the_sync_check,
        check_a_table_of_another_mask_is_tallied_per_row,
    ):
        record_check(checks, check.__name__.removeprefix("check_"), lambda check=check: check(scorer, ids, mask))
    return {"checks": checks}


if __name__ == "__main__":
    run()
