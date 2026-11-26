"""Per-document isolation oracle for SMPO's padding-free row: each document's mean log-prob must equal the same
document run alone."""

import torch
import torch.nn.functional as F

from halo.data.collators.smpo import DataCollatorForSMPO
from halo.models.segment_markers import SegmentMarkers
from tests.common.tiny_models import (
    TINY_DENSE_FAMILIES,
    TINY_MOE_FAMILIES,
    tiny_family_model,
)

PAD_ID = 0
SEED = 1234
# At the default 0.02 init a crossed boundary moves log-probs too little to measure; 0.1 puts the leak orders above
# rounding.
INIT_RANGE = 0.1
# (prompt, chosen, rejected) ids; short prompts put completion tokens inside the conv window straddling each boundary.
PAIRS = (
    ([3, 17, 9], [22, 41, 5, 8], [30, 12]),
    ([11, 4], [7, 19, 33], [50, 2, 44, 21, 6]),
    ([25, 38, 14, 60], [13, 27], [9, 45, 31]),
)
# GatedDeltaNet runs through Qwen3-Next.
FAMILIES = {
    "lfm2": TINY_MOE_FAMILIES["lfm2_moe"],
    "qwen3_next": TINY_MOE_FAMILIES["qwen3_next"],
    "qwen3": TINY_DENSE_FAMILIES["qwen3"],
}
# Marker subsets under which a row must leak, proving the isolation is the markers' doing. GatedDeltaNet's conv-only
# subset leaves the delta rule crossing, so a kernel ignoring cu_seq_lens cannot hide. Dense Qwen3 reads no marker.
LEAK_CONTROLS = {
    "lfm2": {"unmarked": SegmentMarkers()},
    "qwen3_next": {"unmarked": SegmentMarkers(), "conv_only": SegmentMarkers(seq_idx=True)},
    "qwen3": {},
}


def tiny_model(family: str, attn_implementation: str, *, dtype=torch.float32, device="cpu"):
    torch.manual_seed(SEED)
    model = tiny_family_model(FAMILIES[family], overrides={"initializer_range": INIT_RANGE})
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
    """Each document's mean completion log-prob run alone, chosen then rejected, as the padding-free row lays them
    out."""
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
