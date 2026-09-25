"""Per-document isolation oracle for SMPO's padding-free row, shared by its CPU and GPU tests.

The flattened row keeps its documents apart when each one's mean completion log-prob equals the same
document run alone. Holds the tiny per-family models, the preference batch and that lone-document
reference.
"""

# The device-aware kernel-dispatch shim must land before the modeling modules below bind
# transformers' hub-kernel fallback factory at import: after them, a CPU forward reaches the
# CUDA-only kernels and a CUDA forward can capture the torch fallback instead of the fla kernel.
import src.models.patches.kernel_dispatch  # noqa: F401  # isort: skip

import torch
import torch.nn.functional as F
from transformers import (
    Lfm2MoeConfig,
    Lfm2MoeForCausalLM,
    Qwen3_5ForCausalLM,
    Qwen3_5TextConfig,
    Qwen3Config,
    Qwen3ForCausalLM,
)

from src.data.collators.smpo import DataCollatorForSMPO
from src.models.segment_markers import SegmentMarkers
from tests.common.models import TINY_LFM2_MOE_CONFIG, TINY_QWEN3_CONFIG, TINY_QWEN35_CONFIG

PAD_ID = 0
SEED = 1234
# At the default 0.02 init the mixers' activations are too small for a crossed boundary to move a
# tiny model's log-probs measurably; 0.1 puts the leak orders of magnitude above rounding.
INIT_RANGE = 0.1
# (prompt, chosen, rejected) token ids per pair; short prompts put completion tokens inside the conv
# window that straddles each document boundary.
PAIRS = (
    ([3, 17, 9], [22, 41, 5, 8], [30, 12]),
    ([11, 4], [7, 19, 33], [50, 2, 44, 21, 6]),
    ([25, 38, 14, 60], [13, 27], [9, 45, 31]),
)
FAMILIES = {
    "lfm2": (Lfm2MoeForCausalLM, Lfm2MoeConfig, TINY_LFM2_MOE_CONFIG),
    "qwen3_5": (Qwen3_5ForCausalLM, Qwen3_5TextConfig, TINY_QWEN35_CONFIG),
    "qwen3": (Qwen3ForCausalLM, Qwen3Config, TINY_QWEN3_CONFIG),
}
# Marker subsets under which a family's row must leak, proving the isolation is the markers' doing.
# GatedDeltaNet's conv-only subset leaves the delta rule crossing, so a kernel that ignored
# cu_seq_lens could not hide under the isolation tolerance. Dense Qwen3 reads no marker.
LEAK_CONTROLS = {
    "lfm2": {"unmarked": SegmentMarkers()},
    "qwen3_5": {"unmarked": SegmentMarkers(), "conv_only": SegmentMarkers(seq_idx=True)},
    "qwen3": {},
}


def tiny_model(family: str, attn_implementation: str, *, dtype=torch.float32, device="cpu", **overrides):
    """A seeded random-init model of ``family`` on ``attn_implementation``; ``overrides`` edit its config."""
    model_cls, config_cls, config = FAMILIES[family]
    torch.manual_seed(SEED)
    model = model_cls(config_cls(**{**config, "initializer_range": INIT_RANGE, **overrides}))
    model = model.to(device=device, dtype=dtype).eval()
    model.config._attn_implementation = attn_implementation
    return model


def preference_batch(device="cpu") -> dict[str, torch.Tensor]:
    """``PAIRS`` through the SMPO collator: left-padded prompts, right-padded completions."""
    batch = DataCollatorForSMPO(pad_token_id=PAD_ID).torch_call(
        [
            {"prompt_input_ids": prompt, "chosen_input_ids": chosen, "rejected_input_ids": rejected}
            for prompt, chosen, rejected in PAIRS
        ]
    )
    return {key: value.to(device) for key, value in batch.items()}


def lone_document_logps(model) -> torch.Tensor:
    """Each document's mean completion log-prob when run by itself — the isolation reference.

    Chosen documents then rejected ones, the order the padding-free row lays them out in.
    """
    device = next(model.parameters()).device
    means = []
    for side in (1, 2):
        for pair in PAIRS:
            prompt, completion = pair[0], pair[side]
            ids = torch.tensor([prompt + completion], device=device)
            with torch.no_grad():
                logits = model(input_ids=ids, use_cache=False).logits[0, :-1]
            token_logps = F.log_softmax(logits.float(), dim=-1).gather(-1, ids[0, 1:, None]).squeeze(-1)
            means.append(token_logps[len(prompt) - 1 :].mean())
    return torch.stack(means)
