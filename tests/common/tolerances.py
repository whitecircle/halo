"""Named tolerances shared by correctness tests.

Each value is keyed to what it compares rather than to a number, so two tests asserting
the same invariant ("EP loss matches the non-parallel reference") move together.
Import the named constant rather than inlining a literal: the name states which
invariant the bound guards, which is what a reviewer needs to judge whether it is
too loose to catch a regression or tight enough to flake.

A test whose comparison is different (a different model scale, a different
aggregation) declares its own module-level constant with the measured noise floor and
the bug signal it must stay under, rather than stretching a shared value to fit.

    from tests.common.tolerances import TOL
    assert abs(ep_loss - fsdp_loss) < TOL.parallel_vs_baseline_loss_abs
"""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class _Tolerances:
    # ── Cross-rank agreement ────────────────────────────────────────────────
    # Identical batch on every rank, so only a reduction order varies (the EP combine; TP's partial-sum
    # all-reduce hands every rank one sum); 1e-3 is headroom over that reorder, and a real mis-dispatch
    # or mis-shard moves one rank's loss well past it.
    ep_identical_batch_rank_spread_abs: float = 1e-3

    # ── Parallel mode vs single-GPU / FSDP reference ────────────────────────
    # Step-0 loss, before optimizer drift: routing/all-to-all is not bitwise-dense.
    parallel_vs_baseline_loss_abs: float = 0.05
    # After a few steps SR and reduce order diverge the trajectories; bounds the trend.
    parallel_vs_baseline_train_loss_abs: float = 0.15
    # bf16 reduction reorder flips near-tied top-k picks (gpt-oss top-4-of-32), which puts EP+TP and
    # EP+ETP past the generic bound above while still sitting several times under the shift a
    # rotated-expert control produces. The generic bound holds for modes the router never sees
    # reordered (ETP on Mistral4 matches to 1e-2); stretching it would weaken them.
    router_pick_flip_loss_abs: float = 0.1

    # ── Log-probs under CP ──────────────────────────────────────────────────
    # Per-token log-probs from Ulysses-gathered logits vs the full-sequence reference, at bf16.
    logprob_atol: float = 0.05

    # ── Weights / optimizer ─────────────────────────────────────────────────
    # save→load round-trip at bf16 ULP scale.
    weight_atol: float = 1e-6
    # Loss across a resume boundary (same data, same step).
    resume_loss_abs: float = 0.05

    # ── Gradients through a sharded axis ────────────────────────────────────
    # Two independent bug classes a collapsed relative-L2 bound cannot separate. Scale: a missing
    # cross-rank reduction multiplies the norm by the axis size (>=2x) with direction intact, so the
    # ceiling sits between bf16 reduction noise and that factor. Direction: routing/permutation/sign
    # corruption reorients the gradient at unchanged norm, which no norm ratio can see.
    grad_norm_ratio_max: float = 1.25
    grad_direction_cosine_min: float = 0.90

    # ── EP gradients vs a replicated reference, tiny random-init MoE ─────────
    # bf16 grads on a ~128-token model carry real rounding noise, the paths accumulate in different
    # orders and fp32 EP routing flips occasional bf16 near-ties, so direction is checked loosely while
    # the norm ratio stays tight enough that a missing or doubled /world_size divide (2.0 / 0.5) fails.
    ep_grad_cosine_min: float = 0.9
    ep_grad_norm_ratio_band: tuple[float, float] = (0.67, 1.5)

    # ── Exact-objective pins ────────────────────────────────────────────────
    # Independent reimplementation vs the logged loss. The residual is dtype rather than objective:
    # an fp32 reference of a preference objective over bf16 sequence log-prob sums lands 4e-3 relative
    # away (KTO on Qwen3-0.6B), while degenerate objectives miss by >0.5. Relative, applied against
    # max(1, |expected|).
    exact_objective_rel: float = 2e-2

    # ── Generic finite-difference / numerical kernels ───────────────────────
    kernel_atol: float = 1e-2
    kernel_rtol: float = 1e-2

    # ── Muon orthogonalization ──────────────────────────────────────────────
    # Singular values of the Newton-Schulz output. In exact arithmetic the five composed Polar Express
    # quintics map an input singular value of at least 1.22e-3 of the Frobenius norm into
    # [0.846, 1.124]; inside each path's domain the bf16 output moves those extremes by under 3e-3,
    # within the band's margin. Steps 1-3 only act on inputs below ~1e-2, so a dropped one shows only
    # there. Without the safety factor, fp16 rounding overshoots the band on either path (to 1.2-1.4
    # on the Gram path), which the rectangular cases catch.
    muon_orthogonal_sv_min: float = 0.84
    muon_orthogonal_sv_max: float = 1.13
    # The smallest input singular value, relative to the Frobenius norm, each path holds the band from:
    # a round margin over 1.22e-3 on the square (standard iteration) path, and a measured one on the
    # rectangular (Gram iteration) path, which loses the band below ~4e-3.
    muon_band_domain_square: float = 1.4e-3
    muon_band_domain_rectangular: float = 4e-3

    def muon_polar_cosine_min(self) -> float:
        """Smallest cosine between an orthogonalized update and its source's polar factor ``U V^T``.

        Derived from the singular-value band (the Kantorovich bound over spectra inside it), so the
        two checks cannot drift apart. An update orthogonalized from another matrix's momentum sits
        near 0.
        """
        low, high = self.muon_orthogonal_sv_min, self.muon_orthogonal_sv_max
        return 2 * math.sqrt(low * high) / (low + high)

    def control_min_loss_shift(self, bound: float | None = None) -> float:
        """Minimum loss shift a negative control must produce for a match to be meaningful.

        Derived rather than declared: a control that perturbs the mechanism under test has to move
        the compared quantity by more than the tolerance guarding it, or the "it matches" verdict
        carries no information. 2x leaves room for a partial perturbation (rotating only the experts
        one rank owns, say) without letting noise satisfy it.

        ``bound`` is the tolerance in force at the call site, so a test on a wider bound cannot keep
        a control floor derived from a narrower one, which would let a control pass without
        underwriting the match.
        """
        return 2 * (self.parallel_vs_baseline_loss_abs if bound is None else bound)


TOL = _Tolerances()
