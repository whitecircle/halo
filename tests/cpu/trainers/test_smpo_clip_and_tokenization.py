#!/usr/bin/env python
"""SMPO percentile clipping and prompt/completion boundary tokenization.

Two mechanisms with no other coverage:

* the percentile clip bound is a FLOOR, so it must not carry gradient — left differentiable, every
  clamped token routes its gradient back through the quantile into the single element that IS the
  quantile, handing that one token ``clamped_count`` times the gradient (the opposite of the
  outlier suppression the clip exists for), and disagreeing with the CP path where the all-gather
  detaches the bound anyway;
* when the tokenizer merges the last prompt token with the first completion character, the split
  point moves back one token — and the PROMPT must move with it, because the trainer concatenates
  ``prompt_input_ids ++ completion_input_ids`` verbatim.
"""

import os
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from src.trainers.preference.smpo import SmoothMarginPOTrainer, tokenize_preference_row
from tests.common.ports import free_port

CP_WORLD_SIZE = 2


class _Clipper:
    """Minimal stand-in exposing the real clip methods without building a Trainer."""

    def __init__(self, lower=None, upper=None, min_log_prob=None):
        self.lower_clip_percentile = lower
        self.upper_clip_percentile = upper
        self.min_log_prob = min_log_prob

    _cp_quantile = SmoothMarginPOTrainer._cp_quantile
    _clip_log_probs = SmoothMarginPOTrainer._clip_log_probs


def _rejected_logps(rows=1, seq=600, seed=0):
    """Log-probs of a well-SFT'd policy: mostly near 0 with a thin low tail (Exponential(3)).

    Drawn by inverse-CDF from a seeded uniform so the fixture is deterministic without relying on
    the global RNG.
    """
    generator = torch.Generator().manual_seed(seed)
    uniform = torch.rand(rows, seq, generator=generator).clamp(min=1e-9)
    return uniform.log() / 3.0


def test_percentile_clip_bound_carries_no_gradient():
    """No token may receive more than unit gradient through the clip.

    A correct floor gives each token d(sum)/d(logp) of exactly 1 (unclamped) or 0 (clamped). A
    differentiable bound instead concentrates every clamped token's gradient onto the quantile
    element, so its gradient equals the number of clamped tokens.
    """
    logps = _rejected_logps().requires_grad_(True)
    loss_mask = torch.ones_like(logps, dtype=torch.bool)
    is_chosen = torch.zeros(logps.size(0), dtype=torch.bool).unsqueeze(1)  # every row rejected

    clipper = _Clipper(lower=0.02)
    clipped = clipper._clip_log_probs(logps, loss_mask, is_chosen, cp_config=None)
    clipped.sum().backward()

    max_grad = logps.grad.abs().max().item()
    assert max_grad <= 1.0 + 1e-6, (
        f"a single token received {max_grad:.1f}x gradient through the percentile clip — the "
        f"quantile bound is differentiable, so every clamped token back-propagates into it"
    )


def test_padding_free_percentile_clip_bound_carries_no_gradient():
    """The padding-free row shares the contract, and now the same method — only the selector differs."""
    flat = _rejected_logps(rows=1, seq=800).reshape(-1).requires_grad_(True)
    loss_mask = torch.ones_like(flat, dtype=torch.bool)
    is_chosen = torch.zeros_like(flat, dtype=torch.bool)

    clipper = _Clipper(lower=0.02, upper=0.98)
    clipped = clipper._clip_log_probs(flat, loss_mask, is_chosen)
    clipped.sum().backward()

    assert flat.grad.abs().max().item() <= 1.0 + 1e-6


def test_clip_still_clamps_the_low_tail():
    """Detaching the bound must not disable the clip itself."""
    logps = _rejected_logps()
    loss_mask = torch.ones_like(logps, dtype=torch.bool)
    is_chosen = torch.zeros(logps.size(0), dtype=torch.bool).unsqueeze(1)
    clipper = _Clipper(lower=0.02)
    clipped = clipper._clip_log_probs(logps, loss_mask, is_chosen, cp_config=None)

    assert clipped.min().item() > logps.min().item(), "the low tail was not raised at all"
    bound = torch.quantile(logps.float(), 0.02)
    assert clipped.min().item() == pytest.approx(bound.item(), abs=1e-5)


def _legacy_padded_clip(clipper, per_token_logps, loss_mask, num_chosen, cp_config=None):
    """The pre-merge padded spelling: row-slice selectors, ``torch.where`` writes."""
    per_token_logps = per_token_logps.clone()
    if clipper.lower_clip_percentile is not None:
        rejected_logps = per_token_logps[num_chosen:][loss_mask[num_chosen:]]
        lower_bound = clipper._cp_quantile(rejected_logps, clipper.lower_clip_percentile, cp_config)
        if lower_bound is not None:
            per_token_logps[num_chosen:] = torch.where(
                loss_mask[num_chosen:] & (per_token_logps[num_chosen:] < lower_bound),
                lower_bound.to(per_token_logps.dtype),
                per_token_logps[num_chosen:],
            )
    if clipper.upper_clip_percentile is not None:
        chosen_logps = per_token_logps[:num_chosen][loss_mask[:num_chosen]]
        upper_bound = clipper._cp_quantile(chosen_logps, clipper.upper_clip_percentile, cp_config)
        if upper_bound is not None:
            per_token_logps[:num_chosen] = torch.where(
                loss_mask[:num_chosen] & (per_token_logps[:num_chosen] > upper_bound),
                upper_bound.to(per_token_logps.dtype),
                per_token_logps[:num_chosen],
            )
    if clipper.min_log_prob is not None:
        per_token_logps[num_chosen:] = torch.where(
            loss_mask[num_chosen:] & (per_token_logps[num_chosen:] < clipper.min_log_prob),
            clipper.min_log_prob,
            per_token_logps[num_chosen:],
        )
    return per_token_logps


def _legacy_flat_clip(clipper, per_token_logps, loss_mask, is_chosen):
    """The pre-merge padding-free spelling: boolean-mask selectors, ``clamp`` writes."""
    per_token_logps = per_token_logps.clone()
    rejected_valid = loss_mask & ~is_chosen
    chosen_valid = loss_mask & is_chosen
    if clipper.lower_clip_percentile is not None:
        lower_bound = clipper._cp_quantile(per_token_logps[rejected_valid], clipper.lower_clip_percentile, None)
        if lower_bound is not None:
            per_token_logps[rejected_valid] = per_token_logps[rejected_valid].clamp(
                min=lower_bound.to(per_token_logps.dtype)
            )
    if clipper.upper_clip_percentile is not None:
        upper_bound = clipper._cp_quantile(per_token_logps[chosen_valid], clipper.upper_clip_percentile, None)
        if upper_bound is not None:
            per_token_logps[chosen_valid] = per_token_logps[chosen_valid].clamp(
                max=upper_bound.to(per_token_logps.dtype)
            )
    if clipper.min_log_prob is not None:
        per_token_logps[rejected_valid] = per_token_logps[rejected_valid].clamp(min=clipper.min_log_prob)
    return per_token_logps


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_merged_clip_reproduces_both_pre_merge_spellings(dtype):
    """The one clip must equal BOTH spellings it replaced, on the same data, in the same dtype.

    The two differed only in selector (row split vs boolean mask) and write (``where`` vs ``clamp``);
    if the merge changed the rule — an order swap, a masked position touched, a dtype cast moved —
    one of these two comparisons breaks.
    """
    num_chosen, seq = 3, 400
    logps = _rejected_logps(rows=2 * num_chosen, seq=seq, seed=7).to(dtype)
    loss_mask = torch.ones_like(logps, dtype=torch.bool)
    loss_mask[:, -17:] = False  # padded tail: must stay untouched by every branch
    clipper = _Clipper(lower=0.05, upper=0.9, min_log_prob=-2.0)

    row_is_chosen = (torch.arange(2 * num_chosen) < num_chosen).unsqueeze(1)
    merged = clipper._clip_log_probs(logps, loss_mask, row_is_chosen, cp_config=None)

    legacy = _legacy_padded_clip(clipper, logps, loss_mask, num_chosen)
    assert torch.equal(merged, legacy), "the merged clip disagrees with the padded spelling it replaced"

    flat_logps = logps.reshape(-1)
    flat_mask = loss_mask.reshape(-1)
    flat_is_chosen = row_is_chosen.expand_as(logps).reshape(-1)
    merged_flat = clipper._clip_log_probs(flat_logps, flat_mask, flat_is_chosen)
    assert torch.equal(merged_flat, _legacy_flat_clip(clipper, flat_logps, flat_mask, flat_is_chosen)), (
        "the merged clip disagrees with the padding-free spelling it replaced"
    )
    assert torch.equal(merged_flat.view_as(logps), merged), (
        "padded and padding-free batches of the same data clip differently — the selector, not the "
        "rule, is supposed to be the only difference"
    )
    assert torch.equal(merged[~loss_mask], logps[~loss_mask]), "a masked (padding) position was clipped"


def _cp_clip_worker(rank: int, out_path: str, port: int) -> None:
    """Rank-local slice + a real gloo CP group must clip exactly like one process holding it all."""
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), RANK=str(rank), WORLD_SIZE=str(CP_WORLD_SIZE))
    dist.init_process_group("gloo", rank=rank, world_size=CP_WORLD_SIZE)
    try:
        num_chosen, seq = 2, 512
        logps = _rejected_logps(rows=2 * num_chosen, seq=seq, seed=11)
        loss_mask = torch.ones_like(logps, dtype=torch.bool)
        row_is_chosen = (torch.arange(2 * num_chosen) < num_chosen).unsqueeze(1)
        clipper = _Clipper(lower=0.05, upper=0.9, min_log_prob=-2.0)

        whole = clipper._clip_log_probs(logps, loss_mask, row_is_chosen, cp_config=None)

        chunk = seq // CP_WORLD_SIZE
        sl = slice(rank * chunk, (rank + 1) * chunk)
        local = clipper._clip_log_probs(
            logps[:, sl],
            loss_mask[:, sl],
            row_is_chosen,
            cp_config=SimpleNamespace(process_group=dist.group.WORLD),
        )
        ok = torch.equal(local, whole[:, sl])
        if rank == 0:
            with open(out_path, "w") as fh:
                fh.write("PASS" if ok else "FAIL: CP-sliced clip differs from the whole-sequence clip")
    finally:
        dist.destroy_process_group()


def test_cp_sliced_clip_matches_the_whole_sequence_clip(tmp_path):
    """The collective branch of ``_cp_quantile`` is what keeps the clip independent of ``cp_size``.

    Each rank clips only its own sequence chunk; the bound is the quantile of the GATHERED tokens, so
    the result must be the whole-sequence clip restricted to that chunk. A local quantile (or a bound
    taken before the gather) gives each rank a different floor and fails here.
    """
    out = str(tmp_path / "result.txt")
    mp.start_processes(_cp_clip_worker, args=(out, free_port()), nprocs=CP_WORLD_SIZE, join=True, start_method="spawn")
    with open(out) as fh:
        assert fh.read() == "PASS"


class _MergingTokenizer:
    """Char-level tokenizer where 'a'+'b' merges into a single token — the BPE boundary case."""

    bos_token_id = None
    eos_token_id = None

    def __call__(self, text, add_special_tokens=False):
        ids, i = [], 0
        while i < len(text):
            if text[i : i + 2] == "ab":
                ids.append(1000)  # the merged token
                i += 2
            else:
                ids.append(ord(text[i]))
                i += 1
        return {"input_ids": ids}


def _assembled(row):
    return row["prompt_input_ids"] + row["chosen_input_ids"]


def _assembled_rejected(row):
    return row["prompt_input_ids"] + row["rejected_input_ids"]


def test_boundary_merge_keeps_prompt_plus_completion_equal_to_the_true_render():
    """prompt_input_ids ++ completion_input_ids must equal tok(prompt + completion), exactly."""
    tokenizer = _MergingTokenizer()
    prompt, chosen, rejected = "xya", "bz", "bq"

    row = tokenize_preference_row(
        {"prompt": prompt, "chosen": chosen, "rejected": rejected},
        tokenizer,
        max_prompt_length=None,
        max_completion_length=None,
        truncation_mode="keep_start",
    )

    assert _assembled(row) == tokenizer(prompt + chosen)["input_ids"], (
        "chosen assembles to a different token sequence than tokenizing prompt+chosen directly — "
        "the merged boundary token is duplicated"
    )
    assert _assembled_rejected(row) == tokenizer(prompt + rejected)["input_ids"]


def test_boundary_merge_on_one_side_only_keeps_both_prompts_identical():
    """Chosen and rejected share one prompt field; a per-side split would condition them differently."""
    tokenizer = _MergingTokenizer()
    # 'b' merges with the prompt's trailing 'a'; 'q' does not.
    prompt, chosen, rejected = "xya", "bz", "qq"

    row = tokenize_preference_row(
        {"prompt": prompt, "chosen": chosen, "rejected": rejected},
        tokenizer,
        max_prompt_length=None,
        max_completion_length=None,
        truncation_mode="keep_start",
    )

    assert _assembled(row) == tokenizer(prompt + chosen)["input_ids"]
    assert _assembled_rejected(row) == tokenizer(prompt + rejected)["input_ids"]


def test_no_merge_leaves_the_split_untouched():
    """The common (non-merging) path must be unchanged."""
    tokenizer = _MergingTokenizer()
    prompt, chosen, rejected = "xyz", "qq", "rr"

    row = tokenize_preference_row(
        {"prompt": prompt, "chosen": chosen, "rejected": rejected},
        tokenizer,
        max_prompt_length=None,
        max_completion_length=None,
        truncation_mode="keep_start",
    )

    assert row["prompt_input_ids"] == tokenizer(prompt)["input_ids"]
    assert _assembled(row) == tokenizer(prompt + chosen)["input_ids"]
    assert _assembled_rejected(row) == tokenizer(prompt + rejected)["input_ids"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
