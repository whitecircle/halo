#!/usr/bin/env python
"""The chunked GRPO scorer with a LoRA adapter on ``lm_head``: the log-probs, entropies and every adapter gradient
match the full-logits scorer on the padded, dense-row and packed paths and with candidate ids, and with dropout on the
mask is the one PEFT's own head draws over the scored rows, shared by an armed completion term."""

import pytest
import torch
from peft import LoraConfig, get_peft_model

from halo.models.introspection.unwrapping import base_transformers_model
from halo.trainers.grpo.logprob_scoring import CompletionTerm
from tests.common.length_deal import ragged_round, scoring_host, single_process_accelerator, stacked, tiny_llama

_VOCAB = 64
_TOL = {"rtol": 1e-4, "atol": 1e-5}
# One prompt length, so the full-logits forward's positions match a trimmed dense row's; the completions are ragged.
_PROMPTS = [4, 4, 4]
_COMPLETIONS = [5, 2, 6]
_LAYOUTS = {"padded": (3, False), "dense": (1, False), "packed": (1, True)}


def _adapted(targets: list[str], *, dropout: float = 0.0, lora_bias: bool = False, use_dora: bool = False):
    config = LoraConfig(
        r=4, lora_alpha=8, target_modules=targets, lora_dropout=dropout, lora_bias=lora_bias, use_dora=use_dora
    )
    model = get_peft_model(tiny_llama(_VOCAB, seed=0), config).train()
    # lora_B starts at zero, which would make the adapted head the base one.
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "lora_B" in name:
                param.normal_(std=0.5, generator=torch.Generator().manual_seed(len(name)))
    return model


def _host(model, *, chunked: bool = True, batch_size: int = 1, packed: bool = False):
    host = scoring_host(model, None, chunks=1, accelerator=single_process_accelerator())
    host._use_chunked_grpo_logprobs = chunked
    host._packs_microbatches = packed
    host.args.per_device_train_batch_size = batch_size
    host.model_kwarg_keys = {"input_ids", "attention_mask", "logits_to_keep"}
    host._entropy_bonus_enabled = False
    host.pad_token_id = 0
    return host


def _round() -> tuple[torch.Tensor, torch.Tensor, int]:
    return stacked(ragged_round(_PROMPTS, _COMPLETIONS, vocab=_VOCAB, seed=3))


def _scored(host, ids, mask, keep):
    """``(logps, entropies, gradients)`` of one train-mode scoring pass and the backward of its masked sum."""
    host.model.zero_grad(set_to_none=True)
    logps, entropies, _ = host._get_per_token_logps_and_entropies(host.model, ids, mask, keep, compute_entropy=True)
    (logps * mask[:, -keep:]).sum().backward()
    grads = {name: p.grad.clone() for name, p in host.model.named_parameters() if p.grad is not None}
    return logps.detach(), entropies, grads


def _head_grads(grads: dict) -> dict:
    return {name: grad for name, grad in grads.items() if "lm_head.lora_" in name}


@pytest.mark.parametrize("lora_bias", [False, True], ids=["plain", "lora-bias"])
@pytest.mark.parametrize("layout", list(_LAYOUTS))
def test_the_chunked_scorer_matches_the_full_logits_one(layout, lora_bias):
    batch_size, packed = _LAYOUTS[layout]
    model = _adapted(["q_proj", "lm_head"], lora_bias=lora_bias)
    ids, mask, keep = _round()
    completion = mask[:, -keep:].bool()
    full_logps, full_entropies, full_grads = _scored(
        _host(model, chunked=False, batch_size=batch_size), ids, mask, keep
    )
    logps, entropies, grads = _scored(_host(model, batch_size=batch_size, packed=packed), ids, mask, keep)
    with model.disable_adapter(), torch.no_grad():
        base_logps, _, _ = _host(model, batch_size=batch_size, packed=packed)._get_per_token_logps_and_entropies(
            model, ids, mask, keep
        )
    assert (base_logps - full_logps)[completion].abs().max() > 1e-2, "the adapters must move the log-probs"
    torch.testing.assert_close(logps[completion], full_logps[completion], **_TOL)
    torch.testing.assert_close(entropies[completion], full_entropies[completion], **_TOL)
    head_grads = _head_grads(full_grads)
    assert len(head_grads) == (3 if lora_bias else 2) and all(g.abs().sum() > 0 for g in head_grads.values())
    assert grads.keys() == full_grads.keys()
    for name, grad in full_grads.items():
        torch.testing.assert_close(grads[name], grad, **_TOL, msg=name)


@pytest.mark.parametrize("layout", ["padded", "dense"])
def test_candidate_ids_score_under_the_adapted_head(layout):
    """Each position's sampled token and candidates share the adapted head's one ``log Z``."""
    batch_size, _ = _LAYOUTS[layout]
    model = _adapted(["lm_head"])
    ids, mask, keep = _round()
    completion = mask[:, -keep:].bool()
    candidates = torch.randint(0, _VOCAB, (*completion.shape, 3), generator=torch.Generator().manual_seed(4))
    candidates[:, :, -1] = -1
    weights = torch.rand(*completion.shape, 4, generator=torch.Generator().manual_seed(5))
    host = _host(model, batch_size=batch_size)
    with host._scoring_with_candidates(candidates):
        logps, _, _ = host._get_per_token_logps_and_entropies(model, ids, mask, keep)
    picked = logps[completion]
    (torch.where(picked.isfinite(), picked, 0.0) * weights[completion]).sum().backward()
    got = {name: p.grad.clone() for name, p in model.named_parameters() if "lm_head.lora_" in name}
    model.zero_grad(set_to_none=True)

    logits = model(input_ids=ids, attention_mask=mask).logits[:, -keep - 1 : -1] / host.temperature
    targets = torch.cat([ids[:, -keep:, None], candidates], -1)
    want = logits.log_softmax(-1).gather(-1, targets.clamp(min=0)).masked_fill(targets < 0, float("-inf"))
    torch.testing.assert_close(picked, want[completion], **_TOL)
    picked = want[completion]
    (torch.where(picked.isfinite(), picked, 0.0) * weights[completion]).sum().backward()
    for name, grad in got.items():
        torch.testing.assert_close(grad, dict(model.named_parameters())[name].grad, **_TOL, msg=name)


def _peft_head_logps(model, hidden: torch.Tensor, targets: torch.Tensor, seed: int) -> torch.Tensor:
    """PEFT's own adapted head on ``hidden``, its dropout seeded; ``targets`` one id per row or ``[rows, m]``."""
    torch.manual_seed(seed)
    log_probs = base_transformers_model(model).lm_head(hidden).log_softmax(-1)
    if targets.dim() == hidden.dim():
        return log_probs.gather(-1, targets.clamp(min=0)).masked_fill(targets < 0, float("-inf"))
    return log_probs.gather(-1, targets[..., None])[..., 0]


@pytest.mark.parametrize(
    ("layout", "candidates"),
    [("padded", False), ("dense", False), ("packed", False), ("dense", True)],
    ids=["padded", "dense", "packed", "dense-candidates"],
)
def test_dropout_draws_peft_s_mask_over_the_scored_rows(layout, candidates):
    """The scorer draws the adapter's dropout over the rows it scores, as PEFT's head does on them, and its backward
    reads that mask, so a redraw in the recompute differs. The packed rows are the dense rows concatenated alike."""
    batch_size, packed = _LAYOUTS[layout]
    model = _adapted(["lm_head"], dropout=0.5)
    ids, mask, keep = _round()
    completion = mask[:, -keep:].bool()
    targets = ids[:, -keep:]
    if candidates:
        extra = torch.randint(0, _VOCAB, (*completion.shape, 2), generator=torch.Generator().manual_seed(4))
        extra[:, :, -1] = -1
        targets = torch.cat([targets[..., None], extra], -1)
    host = _host(model, batch_size=batch_size, packed=packed)
    scored = {}
    for seed in (11, 12):
        model.zero_grad(set_to_none=True)
        torch.manual_seed(seed)
        if candidates:
            with host._scoring_with_candidates(extra):
                logps, _, _ = host._get_per_token_logps_and_entropies(model, ids, mask, keep)
        else:
            logps, _, _ = host._get_per_token_logps_and_entropies(model, ids, mask, keep)
        picked = logps[completion]
        torch.where(picked.isfinite(), picked, 0.0).sum().backward()
        grads = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
        scored[seed] = picked.detach(), grads
    moved = (scored[11][0] - scored[12][0]).nan_to_num(posinf=0.0, neginf=0.0).abs().max()
    assert moved > 1e-2, "the dropout must move the log-probs"

    model.zero_grad(set_to_none=True)
    with torch.no_grad():
        if layout == "padded":
            hidden = host._get_last_hidden_state(model, ids, mask, keep)
        else:
            hidden, widths = host._dense_last_hidden_state(model, ids, mask, keep)
            targets = torch.cat([row[:width] for row, width in zip(targets, widths, strict=True)])
    want = _peft_head_logps(model, hidden, targets, seed=11)
    want = want[completion] if layout == "padded" else want
    torch.where(want.isfinite(), want, 0.0).sum().backward()
    torch.testing.assert_close(scored[11][0], want.detach(), **_TOL)
    assert len(scored[11][1]) == 2
    for name, grad in scored[11][1].items():
        torch.testing.assert_close(grad, dict(model.named_parameters())[name].grad, **_TOL, msg=name)


def test_an_armed_term_reads_the_dropped_inputs_the_logprobs_read():
    """The term projects the rows the sweep scored after the adapter's dropout, so the GRPO and the term's logits
    come from one head and one mask."""
    model = _adapted(["lm_head"], dropout=0.5)
    ids, mask, keep = _round()
    completion = mask[:, -keep:].bool()
    host = _host(model, batch_size=3)
    term = CompletionTerm(completion.float(), lambda head, rows, positions: head.project(*rows).logsumexp(-1))
    torch.manual_seed(11)
    with host._scoring_with_completion_term(term):
        logps, _, values = host._chunked_logps_impl(model, ids, mask, keep, 3, False)

    with torch.no_grad():
        hidden = host._get_last_hidden_state(model, ids, mask, keep)
        torch.manual_seed(11)
        logits = base_transformers_model(model).lm_head(hidden)
    torch.testing.assert_close(values[completion], logits.logsumexp(-1)[completion], **_TOL)
    torch.testing.assert_close(
        logps[completion], (logits.log_softmax(-1).gather(-1, ids[:, -keep:, None]))[..., 0][completion], **_TOL
    )


def test_a_dora_head_is_refused_at_construction():
    model = _adapted(["q_proj", "lm_head"], use_dora=True)
    with pytest.raises(ValueError, match="variant"):
        _host(model)._reject_unreproducible_chunked_head()
    _host(model, chunked=False)._reject_unreproducible_chunked_head()
    _host(_adapted(["q_proj", "lm_head"]))._reject_unreproducible_chunked_head()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
