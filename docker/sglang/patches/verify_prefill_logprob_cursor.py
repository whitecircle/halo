"""Build-time check that the prefill result loop serves each request its own input log-probs.

The loop reads every request's prompt log-probs out of one flat array at a running offset, and a
request it skips (retracted while its prefill ran, or finished in a mixed batch) still owns a block
of that array. This drives the installed ``process_batch_result_prefill`` with a retracted re-score
request ahead of a live one and asserts the live one reads its own block. It imports sglang, which
only this image has, so the image build is its only gate.

    python3 verify_prefill_logprob_cursor.py
"""

from array import array
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.managers.scheduler_components.batch_result_processor import SchedulerBatchResultProcessor
from sglang.srt.managers.scheduler_components.logprob_result_processor import SchedulerLogprobResultProcessor
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode
from sglang.srt.runtime_context import get_context
from sglang.srt.sampling.sampling_params import SamplingParams

_MODULE = "sglang.srt.managers.scheduler_components.batch_result_processor"

# (prompt, completion) ids; each request's rows run from its last prompt token to its end.
RETRACTED = ([101, 102], [103, 104, 105])
LIVE = ([201, 202, 203], [204, 205])
# The flat array holds one block per request in batch order, skipped or not.
RETRACTED_BLOCK = [-1.0, -2.0, -3.0, -4.0]
LIVE_BLOCK = [-31.0, -32.0, -33.0]


def _rescore_req(rid: str, prompt: list[int], completion: list[int]) -> Req:
    """Shaped like the trainer's re-score: prefill-only, input log-probs from the last prompt token."""
    req = Req(rid, "", array("q", prompt + completion), SamplingParams(max_new_tokens=0), return_logprob=True)
    req.logprob_start_len = len(prompt) - 1
    return req


def _processor() -> SchedulerBatchResultProcessor:
    model_config = SimpleNamespace(think_end_ids=None, vocab_size=1000)
    return SchedulerBatchResultProcessor(
        is_generation=True,
        disaggregation_mode=None,
        enable_overlap=False,
        enable_overlap_mlx=False,
        server_args=SimpleNamespace(return_hidden_states_mode=None, enable_return_hidden_states=False),
        model_config=model_config,
        token_to_kv_pool_allocator=Mock(),
        tree_cache=None,
        hisparse_coordinator=None,
        req_to_token_pool=None,
        decode_offload_manager=None,
        metrics_collector=None,
        metrics_reporter=Mock(),
        draft_worker=None,
        model_worker=Mock(),
        logprob_result_processor=SchedulerLogprobResultProcessor(model_config=model_config),
        output_streamer=Mock(),
        abort_request=Mock(),
    )


def main() -> None:
    retracted = _rescore_req("retracted", *RETRACTED)
    live = _rescore_req("live", *LIVE)
    retracted.is_retracted = True
    reqs = [retracted, live]
    batch = SimpleNamespace(
        reqs=reqs,
        decoding_reqs=[],
        return_logprob=True,
        return_hidden_states=False,
        return_hidden_states_mode=CaptureHiddenMode.NULL,
        spec_info=None,
        prefill_stats=None,
        dp_cooperation_info=None,
    )
    result = GenerationBatchResult(
        logits_output=LogitsProcessorOutput(
            next_token_logits=None,
            input_token_logprobs=torch.tensor(RETRACTED_BLOCK + LIVE_BLOCK),
            next_token_logprobs=torch.tensor([-0.5, -0.6]),
        ),
        next_token_ids=torch.tensor([7, 8]),
        extend_input_len_per_req=[len(req.origin_input_ids) for req in reqs],
        extend_logprob_start_len_per_req=[req.logprob_start_len for req in reqs],
    )
    with (
        get_context().override_server_args(),
        patch(f"{_MODULE}.release_kv_cache"),
    ):
        _processor().process_batch_result_prefill(batch, result)

    served = live.logprob.input_token_logprobs_val
    expected = [None, *LIVE_BLOCK[:-1]]
    if served != expected:
        raise SystemExit(
            f"a retracted request shifts the next request's input log-probs: served {served}, expected {expected}"
        )
    print(f"prefill log-prob cursor: OK (a retracted request's {len(RETRACTED_BLOCK)} rows are stepped over)")


if __name__ == "__main__":
    main()
