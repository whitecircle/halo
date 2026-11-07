"""SGLang's candidate log-probs per generated token, as the two arrays score centering trains on, parsed from its flat
base64 arrays."""

import base64
import math
from typing import Any, NamedTuple

import numpy as np

# An unused candidate slot: no token id, and a probability of zero.
NO_CANDIDATE_ID = -1
NO_CANDIDATE_LOGP = -math.inf
# SGLang clamps a log-prob of -inf to the float32 minimum; anything below this reads as probability zero.
_ZERO_PROBABILITY_BELOW = -1e3


class CandidateLogprobs(NamedTuple):
    """``ids`` int32 and ``logps`` float32, both ``[tokens, K]``: slot ``j`` of row ``i`` is a candidate at the ``i``-th
    generated token, unused slots :data:`NO_CANDIDATE_ID` and :data:`NO_CANDIDATE_LOGP`."""

    ids: np.ndarray
    logps: np.ndarray


def _decode_flat_array(meta_info: dict[str, Any], name: str, dtype: str) -> np.ndarray | None:
    """The base64 little-endian array SGLang's chat route sends by default; ``None`` when absent or of another width."""
    raw = meta_info.get(f"output_top_logprobs_{name}_flat_b64")
    if not isinstance(raw, str) or meta_info.get(f"output_top_logprobs_{name}_flat_b64_dtype", dtype) != dtype:
        return None
    return np.frombuffer(base64.b64decode(raw), dtype=np.dtype(dtype).newbyteorder("<"))


def parse_sglang_candidates(meta_info: dict[str, Any], tokens: int, k: int) -> CandidateLogprobs | None:
    """From a chat reply's ``meta_info`` under ``return_flat_raw_output_top_logprobs``: ``[rows, width]`` arrays covering
    the positions after ``output_top_logprobs_null_prefix`` unfilled leading ones. Each row holds the engine's top
    ``width`` of the full vocabulary, cut at ``k`` or padded to it. ``None`` when the fields are absent or do not cover
    ``tokens`` positions."""
    shape = meta_info.get("output_top_logprobs_shape")
    ids = _decode_flat_array(meta_info, "idx", "int32")
    logps = _decode_flat_array(meta_info, "val", "float32")
    if shape is None or ids is None or logps is None:
        return None
    rows, width = shape
    prefix = int(meta_info.get("output_top_logprobs_null_prefix") or 0)
    if prefix + rows != tokens or ids.size != rows * width or logps.size != rows * width:
        return None
    out = CandidateLogprobs(
        np.full((tokens, k), NO_CANDIDATE_ID, dtype=np.int32),
        np.full((tokens, k), NO_CANDIDATE_LOGP, dtype=np.float32),
    )
    kept = min(k, width)
    out.ids[prefix:, :kept] = ids.reshape(rows, width)[:, :kept]
    out.logps[prefix:, :kept] = logps.reshape(rows, width)[:, :kept]
    out.logps[out.logps < _ZERO_PROBABILITY_BELOW] = NO_CANDIDATE_LOGP
    return out
