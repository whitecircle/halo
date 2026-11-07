#!/usr/bin/env python
"""Online GRPO's vLLM importance-sampling ratio and KL tail clamp built in the loss forward from its detached log-probs
(``loss_forward_importance_weights``): the same loss, gradient and metrics as TRL's old-policy pass for every ratio mode,
packed or not; the pass kept wherever a consumer needs the round's log-probs earlier."""

import math
from collections import defaultdict

import pytest
import torch

from halo.trainers.grpo.logprob_scoring import ProtectedTokenEntropyMixin
from halo.trainers.grpo.loss_forward_seam import LossForwardSeamMixin
from halo.trainers.grpo.microbatch_plan import agree_microbatch_geometry
from halo.trainers.grpo.objective.logratio import KL_CLAMP_FRAC_KEY, clamp_ref_logps
from halo.trainers.grpo.objective.loss_inputs import NET_TOKEN_MASS_KEY, record_token_mass
from halo.trainers.grpo.online_loss_forward import online_loss_forward_blocker
from halo.trainers.token_metrics import effective_loss_mask
from tests.common.length_deal import (
    recompute_logps,
    sampling_side_round,
    single_process_accelerator,
    tiny_llama,
    train_window,
)
from tests.common.online_rounds import online_host

_VOCAB, _ROWS, _CLOSE = 64, 8, 7
_CLIP_MIN, _CLIP_MAX = 0.5, 3.0
_MODES = ("token_truncate", "token_mask", "sequence_truncate", "sequence_mask")

_PASS_FACTS = {"runs_pass": True, "num_iterations": 1, "misaligned_accumulation": False, "balance_token_mass": False}
_BLOCKERS = [
    ("runs_pass", False, "nothing to fuse"),
    ("num_iterations", 2, "PPO ratio's denominator"),
    ("misaligned_accumulation", True, "straddles a generation round"),
    ("balance_token_mass", True, "balance_token_mass"),
]


def test_a_plain_aligned_one_step_round_is_fusable():
    assert online_loss_forward_blocker(**_PASS_FACTS) is None


@pytest.mark.parametrize(("fact", "value", "named"), _BLOCKERS)
def test_every_consumer_of_the_round_start_log_probs_blocks_the_fusion(fact, value, named):
    blocker = online_loss_forward_blocker(**{**_PASS_FACTS, fact: value})
    assert blocker is not None and named in blocker


def _host(*, mode: str = "token_truncate", loss_type: str = "dapo", beta: float = 0.0, packed: bool = False, **facts):
    host = online_host(
        tiny_llama(_VOCAB, 0),
        slots=4 if packed else _ROWS,
        budget=40,
        accelerator=single_process_accelerator(),
        packed=packed,
        loss_type=loss_type,
    )
    host.beta = beta
    host.vllm_importance_sampling_mode = mode
    host.vllm_importance_sampling_clip_min, host.vllm_importance_sampling_clip_max = _CLIP_MIN, _CLIP_MAX
    host.args.use_bias_correction_kl = False
    host._balance_token_mass = False
    host._requests_loss_forward_weights = facts.get("requested", False)
    return host


@pytest.mark.parametrize("requested", [True, False])
def test_the_knob_opts_in_and_off_leaves_trls_pass(requested):
    host = _host(requested=requested)
    host._configure_loss_forward_weights()
    assert host._skips_logprob_pass is requested
    with host._without_trl_old_pass():
        assert host.vllm_importance_sampling_correction is (not requested)
    assert host.vllm_importance_sampling_correction is True


def test_eval_rounds_keep_the_pass():
    host = _host(requested=True)
    host._configure_loss_forward_weights()
    host.model.eval()
    with host._without_trl_old_pass():
        assert host.vllm_importance_sampling_correction is True


@pytest.mark.parametrize(
    ("change", "named"),
    [
        ({"vllm_importance_sampling_correction": False}, "nothing to fuse"),
        ({"num_iterations": 2}, "PPO ratio's denominator"),
        ({"_balance_token_mass": True}, "balance_token_mass"),
    ],
)
def test_true_raises_where_a_consumer_needs_the_pass_and_false_never_does(change, named):
    host = _host(requested=True)
    for name, value in change.items():
        setattr(host, name, value)
    with pytest.raises(ValueError, match=rf"loss_forward_importance_weights: true, but .*{named}.* Set it to false"):
        host._configure_loss_forward_weights()
    host._requests_loss_forward_weights = False
    host._configure_loss_forward_weights()
    assert host._skips_logprob_pass is False


def test_a_misaligned_window_is_refused_under_true():
    host = _host(requested=True)
    host.args.gradient_accumulation_steps = 6
    host.args.steps_per_generation = 4
    with pytest.raises(ValueError, match="straddles a generation round"):
        host._configure_loss_forward_weights()


# ---- the loss: fused equals the pass -------------------------------------------------------------------------------


def _trl_ratio(old: torch.Tensor, sampling: torch.Tensor, mask: torch.Tensor, mode: str) -> torch.Tensor:
    """TRL's ratio block of ``_generate_and_score_completions``, transcribed."""
    diff = torch.nan_to_num((old - sampling) * mask, nan=0.0)
    if mode.startswith("sequence"):
        diff = diff.sum(dim=-1, keepdim=True)
    ratio = torch.exp(diff)
    if mode.endswith("truncate"):
        return torch.clamp(ratio, min=_CLIP_MIN, max=_CLIP_MAX)
    return ratio.masked_fill((ratio < _CLIP_MIN) | (ratio > _CLIP_MAX), 0.0)


def _round(mode: str, beta: float, *, seed: int = 11) -> tuple[dict, dict, torch.Tensor]:
    """``(pass batch, fused batch, recompute)`` of one round: what TRL's pass would build and the inputs the loss forward
    builds the same from."""
    gen = torch.Generator().manual_seed(seed)
    batch = sampling_side_round(vocab=_VOCAB, close=_CLOSE, rows=_ROWS)
    batch.pop("importance_sampling_ratio")
    recompute = recompute_logps(tiny_llama(_VOCAB, 0), batch)
    sampling = recompute - torch.randn(recompute.shape, generator=gen) * 0.9
    sampling[0, 0] = math.nan
    sampling[5] = math.nan
    mask = effective_loss_mask(batch)
    ratio = _trl_ratio(recompute, sampling, mask, mode)
    assert (ratio != 1).any() and ((ratio == 0) | (ratio == _CLIP_MIN) | (ratio == _CLIP_MAX)).any(), (
        "the clip must engage"
    )
    assert not torch.isnan(ratio).any()
    in_pass = batch | {
        "old_per_token_logps": recompute,
        "importance_sampling_ratio": ratio,
        "sampling_per_token_logps": sampling,
    }
    fused = batch | {"sampling_per_token_logps": sampling}
    if beta:
        ref = recompute + (torch.rand(recompute.shape, generator=gen) - 0.5) * 16
        clamped_ref, was_clamped = clamp_ref_logps(ref, recompute)
        assert was_clamped.any() and not was_clamped.all()
        in_pass["ref_per_token_logps"], fused["ref_per_token_logps"] = clamped_ref, ref
    return in_pass, fused, recompute


def _train(batch: dict, *, mode: str, loss_type: str, beta: float, fused: bool, packed: bool):
    host = _host(mode=mode, loss_type=loss_type, beta=beta, packed=packed)
    if fused:
        host._adopt_loss_forward_weights(None)
    plan = None
    if packed:
        lengths = (batch["prompt_mask"].sum(1) + batch["completion_mask"].sum(1)).clamp(min=2).tolist()
        plan = agree_microbatch_geometry(lengths, 40, slots=4, ceiling=None, round_rows=_ROWS)
    total, grads = train_window(host, batch, plan)
    return host, total, grads


def _close_to(got, want) -> bool:
    return got[0] == pytest.approx(want[0], rel=1e-4, abs=1e-6) and all(
        torch.allclose(g, w, rtol=1e-3, atol=1e-6) for g, w in zip(got[1], want[1], strict=True)
    )


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("beta", [0.0, 0.04])
@pytest.mark.parametrize("loss_type", ["grpo", "dapo", "cispo"])
@pytest.mark.parametrize("mode", _MODES)
def test_the_loss_forward_ratio_gives_the_pass_loss_and_gradient(mode, loss_type, beta, packed):
    in_pass, fused, _ = _round(mode, beta)
    _, *want = _train(in_pass, mode=mode, loss_type=loss_type, beta=beta, fused=False, packed=packed)
    host, *got = _train(fused, mode=mode, loss_type=loss_type, beta=beta, fused=True, packed=packed)
    assert abs(want[0]) > 1e-4, "a vanishing loss compares nothing"
    assert _close_to(got, want)
    assert host._loss_forward_seam is None


def test_a_ratio_that_carries_a_graph_changes_the_gradient(monkeypatch):
    """Negative control: a ratio built from the live log-probs adds ``A·logπ·∂w/∂θ`` the objective does not have."""
    in_pass, fused, _ = _round("token_truncate", 0.0)
    want = _train(in_pass, mode="token_truncate", loss_type="dapo", beta=0.0, fused=False, packed=False)[1:]

    def leaky(self, model, input_ids, attention_mask, logits_to_keep, *args, **kwargs):
        scored = ProtectedTokenEntropyMixin._get_per_token_logps_and_entropies(
            self, model, input_ids, attention_mask, logits_to_keep, *args, **kwargs
        )
        seam = self._loss_forward_seam
        if seam is not None and torch.is_grad_enabled() and kwargs.get("compute_entropy"):
            seam.fired += 1
            self._weigh_from_loss_forward(seam, scored[0])
        return scored

    monkeypatch.setattr(LossForwardSeamMixin, "_get_per_token_logps_and_entropies", leaky)
    got = _train(fused, mode="token_truncate", loss_type="dapo", beta=0.0, fused=True, packed=False)[1:]
    assert not _close_to(got, want)


def test_a_round_without_engine_log_probs_raises():
    _, fused, _ = _round("token_truncate", 0.0)
    del fused["sampling_per_token_logps"]
    with pytest.raises(RuntimeError, match="no sampling_per_token_logps"):
        _train(fused, mode="token_truncate", loss_type="dapo", beta=0.0, fused=True, packed=False)


# ---- the metrics the pass fed --------------------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["token_truncate", "sequence_mask"])
def test_the_round_end_flush_logs_what_trls_pass_logged(mode):
    in_pass, fused, recompute = _round(mode, 0.04)
    host, *_ = _train(fused, mode=mode, loss_type="dapo", beta=0.04, fused=True, packed=False)
    host._metrics = {"train": defaultdict(list)}
    host._flush_loss_forward_metrics()
    logged = {key: values[-1] for key, values in host._metrics["train"].items()}

    mask = effective_loss_mask(in_pass).bool()
    sampling = in_pass["sampling_per_token_logps"]
    delta = (recompute - sampling).abs()
    delta = delta[mask & ~delta.isnan()]
    ratio = in_pass["importance_sampling_ratio"]
    flat = ratio.flatten() if mode.startswith("sequence") else ratio[mask]
    if mode.startswith("sequence"):
        flat = ratio[mask.any(dim=1)].flatten()
    expected = {
        "sampling/sampling_logp_difference/mean": delta.mean().item(),
        "sampling/sampling_logp_difference/max": delta.max().item(),
        "sampling/importance_sampling_ratio/min": flat.min().item(),
        "sampling/importance_sampling_ratio/mean": flat.mean().item(),
        "sampling/importance_sampling_ratio/max": flat.max().item(),
    }
    reference = defaultdict(list)
    record_token_mass(in_pass["advantages"], mask.float(), ratio, lambda x: x, reference, False)
    expected[NET_TOKEN_MASS_KEY] = reference[NET_TOKEN_MASS_KEY][-1]
    clamped = clamp_ref_logps(fused["ref_per_token_logps"], recompute)[1]
    expected[KL_CLAMP_FRAC_KEY] = ((clamped & mask).sum() / mask.sum()).item()
    assert set(logged) == set(expected)
    for key, value in expected.items():
        assert logged[key] == pytest.approx(value, rel=1e-4, abs=1e-6), key


@pytest.mark.parametrize("fused", [True, False])
def test_the_round_build_logs_the_token_mass_only_where_the_pass_supplies_the_ratio(fused):
    in_pass, _, _ = _round("token_truncate", 0.0)
    host = _host()
    if fused:
        host._adopt_loss_forward_weights(None)
    host._apply_token_mass_balance(dict(in_pass))
    assert (NET_TOKEN_MASS_KEY in host._metrics["train"]) is (not fused)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
