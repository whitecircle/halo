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

from dataclasses import dataclass


@dataclass(frozen=True)
class _Tolerances:
    # ── Cross-rank agreement ────────────────────────────────────────────────
    # DP-averaged loss: all ranks reduce the same scalar, so only numerical noise.
    rank_loss_consistency_abs: float = 0.01
    # Identical broadcast batch, so only the combine reduction order varies; 1e-3 is headroom over
    # that reorder. The 10x-looser DP band above would absorb a real mis-dispatch.
    ep_identical_batch_rank_spread_abs: float = 1e-3
    # DTensor full-tensor norm; a spread above this means the norm came off a shard.
    tp_grad_norm_spread_abs: float = 1e-3

    # ── Parallel mode vs single-GPU / FSDP reference ────────────────────────
    # Step-0 loss, before optimizer drift: routing/all-to-all is not bitwise-dense.
    parallel_vs_baseline_loss_abs: float = 0.05
    # After a few steps SR and reduce order diverge the trajectories; bounds the trend.
    parallel_vs_baseline_train_loss_abs: float = 0.15
    # Per-rank EP loss vs mean: ranks see distinct micro-batches, so this differs by data.
    ep_rank_loss_abs: float = 0.5
    # bf16 reduction reorder flips near-tied top-k picks (gpt-oss top-4-of-32), which puts EP+TP and
    # EP+ETP past the generic bound above while still sitting several times under the shift a
    # rotated-expert control produces. The generic bound holds for modes the router never sees
    # reordered (ETP on Mistral4 matches to 1e-2); stretching it would weaken them.
    router_pick_flip_loss_abs: float = 0.1

    # ── Log-probs (preference / CP aggregation) ─────────────────────────────
    # CP-aggregated sequence log-probs vs the full-sequence reference.
    logprob_atol: float = 0.05
    logprob_rtol: float = 0.02

    # ── Weights / optimizer ─────────────────────────────────────────────────
    # save→load round-trip at bf16 ULP scale.
    weight_atol: float = 1e-6
    # Loss across a resume boundary (same data, same step).
    resume_loss_abs: float = 0.05
    grad_norm_rel: float = 0.10
    # Norm achieved after clipping vs the max_norm asked for. The clip coefficient is applied in fp32
    # over bf16 grads, so the re-measured norm lands a rounding step off the target, while a wrong
    # coefficient or a missed plain-vs-DTensor scale misses by a factor.
    clip_achieved_norm_rel: float = 1e-4

    # ── Re-derived quantities (recompute / independent recount) ─────────────
    # Same step, activation checkpointing on vs off. The recompute replays the forward with a
    # different reduction order only, so the losses agree to bf16 accumulation noise; the bug this
    # bounds (a corrupted checkpoint frame, such as a replayed DeepEP dispatch) moves them further.
    gc_recompute_loss_rel: float = 1e-3
    # A trainer-computed loss normalizer vs the same quantity recounted from the collective. Both
    # are fp32 divisions of the same integers, so only float rounding separates them; the failure
    # mode is a normalizer off by a whole factor (per-rank instead of stage-global, or /dp twice).
    recomputed_normalizer_abs: float = 1e-4

    # ── Gradients through a sharded axis ────────────────────────────────────
    # Two independent bug classes a collapsed relative-L2 bound cannot separate. Scale: a missing
    # cross-rank reduction multiplies the norm by the axis size (>=2x) with direction intact, so the
    # ceiling sits between bf16 reduction noise and that factor. Direction: routing/permutation/sign
    # corruption reorients the gradient at unchanged norm, which no norm ratio can see.
    grad_norm_ratio_max: float = 1.25
    grad_direction_cosine_min: float = 0.90

    # ── Exact-objective pins ────────────────────────────────────────────────
    # Independent reimplementation vs the logged loss. The residual is dtype rather than objective:
    # an fp32 reference of a preference objective over bf16 sequence log-prob sums lands 4e-3 relative
    # away (KTO on Qwen3-0.6B), while degenerate objectives miss by >0.5. Relative, applied against
    # max(1, |expected|).
    exact_objective_rel: float = 2e-2

    # ── Generic finite-difference / numerical kernels ───────────────────────
    kernel_atol: float = 1e-2
    kernel_rtol: float = 1e-2

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
