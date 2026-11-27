"""Build-time check that vLLM's split top-p keeps a thinking-budget cut whole.

vLLM's Triton top-k/top-p sends the p-only rows (top-p < 1, top-k off) of a batch of at most 64 rows through a
split-row pipeline. When a turn's thinking budget runs out, Model Runner V2 writes ``FORCED_LOGIT`` (1e9) on the forced
token; the pipeline must keep that token, or the row comes back all ``-inf``, the engine samples token 0 and the
processed log-prob is NaN. An ordinary row must still be masked. The kernels run on the Triton interpreter, since the
build has no GPU:

    CUDA_VISIBLE_DEVICES= TRITON_INTERPRET=1 python3 verify_topp_budget_cut.py
"""

import torch
from vllm.v1.sample.ops import topk_topp_triton
from vllm_thinking_budget_completion_patch import FORCED_LOGIT

VOCAB_SIZE = 16384
TOP_P = 0.6
NUM_SM = 148  # B200: 32 slices per row


def split_top_p(row: torch.Tensor) -> torch.Tensor:
    logits = row.clone().unsqueeze(0)
    topk_topp_triton._apply_topp_split(logits, None, torch.full((1,), TOP_P), float("-inf"), NUM_SM)
    return logits[0]


if __name__ == "__main__":
    ordinary = torch.round(torch.randn(VOCAB_SIZE, generator=torch.Generator().manual_seed(0)) * 1.5)
    cut = ordinary.clone()
    cut[-1] = FORCED_LOGIT

    kept = split_top_p(cut)
    assert kept[-1] == FORCED_LOGIT, "split top-p masks the token a thinking-budget cut forces"

    masked = split_top_p(ordinary)
    survivors = int(torch.isfinite(masked).sum())
    assert 0 < survivors < VOCAB_SIZE, f"split top-p does not mask an ordinary row ({survivors}/{VOCAB_SIZE} kept)"
    assert torch.isfinite(masked[int(ordinary.argmax())]), "split top-p masks an ordinary row's top token"
    print(f"split top-p: OK (the forced token kept, an ordinary row masked to {survivors}/{VOCAB_SIZE})")
