"""Named tolerances shared by correctness tests.

Each value is keyed to *what it compares*, not to a number, so two tests asserting
the same invariant ("EP loss matches the non-parallel reference") move together.
Import the named constant rather than re-inlining a literal — the name states which
invariant the bound guards, which is what a reviewer needs in order to judge whether
it is too loose to catch a regression or tight enough to flake.

A test whose comparison is genuinely different (a different model scale, a different
aggregation) declares its own module-level constant with the measured noise floor and
the bug signal it must stay under; it does not stretch a shared value to fit.

    from tests.common.tolerances import TOL
    assert abs(ep_loss - fsdp_loss) < TOL.parallel_vs_baseline_loss_abs
"""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class _Tolerances:
    # ── Cross-rank agreement ────────────────────────────────────────────────
    # Identical batch on every rank, and no forward reduction depends on arrival order: an all-reduce
    # hands every rank one sum, DeepEP's combine sums a token's partials in top-k slot order, the grouped
    # path's un-permute is a fixed-order gather-reduce, and the per-expert loop's (use_grouped_gemm:
    # false) index_add_ adds each row once per local expert, in a fixed expert order. DeepEP's receive
    # order does vary without full_determinism, but it reorders only the expert weight-gradient sums.
    # So every rank's loss is the same float (measured 0.0 on every row reading this); compare through
    # :meth:`identical_batch_ranks_agree`. A mis-dispatch or mis-shard moves one rank's loss off it.
    ep_identical_batch_rank_spread_abs: float = 0.0
    # Identical batch through an all-reduce with no EP un-permute in the path (pure TP, pure ETP): the
    # all-reduce hands every rank one sum, and a loss or grad norm read off a shard misses by orders of
    # magnitude more.
    all_reduced_rank_spread_abs: float = 1e-4
    # The per-step losses every rank of a pipeline run logs (``torch.allclose``, its default rtol on top):
    # the last stage broadcasts each loss down its chain and HF logs the world mean, so every rank holds
    # the same floats. A chain fed another batch, or a wrong broadcast source, moves a rank's curve by far more.
    pp_rank_identical_loss_abs: float = 1e-6

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
    # A resume that restores the saved state and replays the uninterrupted run exactly: nothing is reset
    # between the runs, the bf16 optimizer keys its rounding by step and parameter name, and the run is
    # deterministic (``full_determinism``, or pinned DeepEP dispatch where only the dispatch reorders
    # sums). Every compared loss and final tensor measured bit-exact; 1e-4 is kernel headroom, under the
    # 1.2e-4 first-step miss of adapters restarted from init.
    replayed_resume_loss_abs: float = 1e-4
    # The same comparison on the final trainable tensors, as a relative L2; a restart misses by ~1.
    replayed_resume_weight_rtol: float = 1e-4

    # Loss across a resume boundary (same data, same step).
    resume_loss_abs: float = 0.05
    # Fixed-batch forward loss before a save and after the resume, on a forward with no
    # nondeterministic reduction: FSDP, TP, CP, and grouped-GEMM EP, whose expert rows are computed
    # independently and combined in top-k slot order whatever order DeepEP receives them in. bf16
    # round-trip noise; an unrestored or mis-gathered weight set moves it by more than 1.
    resume_fixed_batch_loss_abs: float = 1e-2
    # Norm achieved after clipping vs the max_norm asked for. The coefficient multiply and the
    # re-measured norm differ from the target only by float rounding, while a wrong coefficient or a
    # missed plain-vs-DTensor scale misses by a factor.
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
    # Two independent bug classes a collapsed relative-L2 bound cannot separate. SCALE: a
    # missing cross-rank reduction multiplies the norm by the axis size, direction intact
    # (measured 2.14 at axis 2 vs a 1.10 noise ceiling). DIRECTION: routing/permutation/sign
    # corruption reorients at unchanged norm (noise floor 0.96, 24-layer bf16 MoE router).
    grad_norm_ratio_max: float = 1.25
    grad_direction_cosine_min: float = 0.90

    # ── CP gradients vs the full-sequence reference ─────────────────────────
    # The CP-rank-averaged gradient against the single-rank full-sequence one. Ulysses re-partitions
    # attention exactly, so only bf16 all-to-all and accumulation order separate them: each parameter's
    # direction holds to 0.99 and the total norm to 10%. A wrong sequence split or RoPE offset reorients
    # the attention grads, and a CP reduction that sums instead of averaging scales the norm by cp_size.
    # In a routed MoE model that noise also flips a few near-tied top-k picks, which moves a token's whole
    # contribution to its router and experts and perturbs every parameter downstream (gpt-oss on the CP
    # roster: flipped routers at 0.984, the worst downstream parameter 0.9897). There the router/expert group
    # holds 0.99 as one vector, each router or expert parameter grad_direction_cosine_min, and every other
    # parameter cp_routed_grad_cosine_min (first_step.parameter_gradient_checks).
    cp_grad_cosine_min: float = 0.99
    cp_routed_grad_cosine_min: float = 0.98
    cp_grad_norm_rtol: float = 0.10

    # ── EP gradients vs a replicated reference, tiny random-init MoE ─────────
    # bf16 grads on a ~128-token model carry real rounding noise, the paths accumulate in different
    # orders and fp32 EP routing flips occasional bf16 near-ties, so direction is checked loosely while
    # the norm ratio stays tight enough that a missing or doubled /world_size divide (2.0 / 0.5) fails.
    ep_grad_cosine_min: float = 0.9
    ep_grad_norm_ratio_band: tuple[float, float] = (0.67, 1.5)
    # A reference gradient this small against the largest in its decoder layer has no reliable direction, so the
    # TP and PP family suites hold it to its absolute error under that floor instead of its cosine and ratio:
    # GatedDeltaNet's A_log on a hard-decaying head is a sum of ~1e-9 terms that cancel 5-29x, so every FLA kernel
    # config lands a different rounding pattern on it. On the PP suite's batch, layer 5's A_log / dt_bias sit at
    # 1.5e-7..2.5e-7 of it and its in_proj_a at 2.3e-6.
    # On the TP suite's, Qwen3.5's A_log / dt_bias on layers 1, 5 and 6, whose heads all decay hard,
    # sit at 3e-9..7e-7 of it, where on GPU the bf16 unsharded model itself scores cos 0.17-0.85 against fp32 and two
    # fp32 runs 0.11-0.36 against each other; on layers 0, 2 and 4 they sit at 1.6e-5..8e-4, where a dropped or doubled
    # TP reduce moves the norm ratio by 2; every other parameter sits at 1.5e-6 or more.
    grad_negligible_layer_fraction: float = 1e-6
    # A near-cancelled pair outside the EP bars above whose distance from an fp32 copy of the reference stays within this
    # multiple of the parameter's measured bf16 noise (the bf16 reference's largest distance from the fp32 one over the
    # run's steps) is rounding, not a sync fault. The 1-8 element hyper-connection vectors of DeepSeek-V4 and GLM-5 Next
    # cancel to 3e-5..1e-2 of their layer's largest gradient: there the bf16 reference misses fp32 by up to 3.3x their
    # norm, and an EP policy's own rounding lands up to 1.6x (DeepSeek-V4) and 3.6x (GLM-5 Next, moving between launches)
    # that noise from fp32. Routers and expert banks are never excused and every parameter must land inside the bars at
    # its best resolved step, so what the factor admits is a fault confined to a parameter's other steps on a pair under
    # the fraction below. A vector whose own rounding sits near a bar at its best resolved step takes a doubling inside
    # them there (GLM-5 Next's layer-3 attn_hc.scale, at 0.72 of its bf16 reference) and is held by its kind's others.
    bf16_noise_excuse_factor: float = 8.0
    # Near-cancelled for that excuse: a bf16 reference under this fraction of the largest bf16 reference gradient in its
    # decoder layer. The pairs the excuse takes on the GPU rows (the hc vectors, Qwen3.5's GatedDeltaNet A_log and
    # dt_bias) sit at 2e-5..1.8e-3. Routers and expert banks need not sit above it, being never excused; Gemma 4's
    # router, a sibling module outside its EP layers, does not count as one, and its pairs from 4e-5 to 3.8e-3 stay
    # excusable: it syncs with its decoder layer's FSDP2 group, whose fault reaches that layer's largest gradients. So do
    # other small kinds at the steps they fall under it (on the CPU rows GPT-OSS's sinks and q/k biases, Gemma 4's norms,
    # DeepSeek-V4's compressor kv_norm and attn_hc.base, Qwen3-MoE's k_norm), where the factor excuses a one-step fault
    # of up to about 3x their norm.
    bf16_noise_excusable_layer_fraction: float = 4e-3
    # How far a parallel run's top-k pick may sit below the reference router's own k-th logit, on a tiny GptOss
    # whose router logits spread ~0.3: bf16 reduction order flips only near-ties (measured worst 4.4e-3 at etp2,
    # a few bf16 ULPs at these magnitudes), while a wrong selection falls short by a top-k gap (median 0.06-0.17
    # per block; the third-best expert picked for the second misses by up to 0.66). On the reentrant suite's rows, for
    # each family whose references route on the policy's picks (GptOss and the Qwen3, Qwen3.5, Qwen3-Next and Qwen4-Exp
    # MoE), a bf16 copy routed on an fp32 copy's picks falls at most 1e-2 short, while on every layer some token's
    # (k+1)-th best expert falls 0.085 or more short.
    router_near_tie_logit_abs: float = 3e-2
    # How far a router's own top-k pick may sit below its k-th logit: a router selecting on fp32 softmax probabilities
    # rounds logits an fp32 step or two apart (~3e-8 at these magnitudes) to one probability and breaks the tie by
    # index, while a selection on anything but those logits misses by a top-k gap.
    router_own_pick_logit_abs: float = 1e-6

    # ── Routed path vs the stock expert loop on the run's own picks ─────────
    # A layout's EP layers against the stock MoE blocks of a reference pinned on the top-k picks they dispatched
    # (tests.common.routed_first_step), on the tiny random-init Qwen3-MoE of the SMPO layouts suite in bf16. Its CPU
    # twin (EP layers at ep1, and pure expert TP 2 over gloo, three seeds): each layer's output within 1.2e-2 relative
    # L2, every router and expert slice cos >= 0.9998 and norm ratio within 0.6%; the stock bf16 model itself sits
    # 1.4e-2, cos 0.9997 and 1% from an fp32 copy, so two bf16 implementations stay within ~3e-2, 0.999 and 2%. A halved
    # combine moves the outputs to 0.50 and every ratio to 0.5; a dropped expert-TP reduce moves the outputs to 0.72 or
    # more and some expert slice's cosine to 0.94 or less, its partners routing on diverged hidden states. Its picks
    # are held to router_near_tie_logit_abs: the router logits spread like that bar's GptOss (|logit| median 0.21,
    # max 1.3), and over six seeds a pick sits at most 7.8e-3 below the bf16 stock router's top-k and 1.2e-2 below an
    # fp32 copy's; a halved combine or a dropped expert-TP reduce moves later layers' picks 2.7e-2..6.2e-2 below.
    routed_output_rel_l2: float = 0.1
    routed_grad_cosine_min: float = 0.99
    routed_grad_norm_ratio_band: tuple[float, float] = (0.9, 1.1)

    # ── Exact-objective pins ────────────────────────────────────────────────
    # A test's independent reimplementation of its trainer's objective (offline GRPO, distillation,
    # reward) vs the loss the trainer computes or logs. Only precision and reduction order separate
    # them, so the bound stays far under a wrong objective's gap; the reward and distillation pins
    # require each of their controls to miss by more than it. Relative: against max(1, |expected|)
    # in the reward and distillation pins, against |expected| in offline GRPO's.
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

    def identical_batch_ranks_agree(self, spread: float) -> bool:
        """Whether an identical-batch loss spread across ranks is within
        :attr:`ep_identical_batch_rank_spread_abs` (inclusive: the bound is an exact 0.0)."""
        return spread <= self.ep_identical_batch_rank_spread_abs

    def muon_polar_cosine_min(self) -> float:
        """Smallest cosine between an orthogonalized update and its source's polar factor ``U V^T``.

        Derived from the singular-value band (the Kantorovich bound over spectra inside it), so the
        two checks cannot drift apart. An update orthogonalized from another matrix's momentum sits
        near 0.
        """
        low, high = self.muon_orthogonal_sv_min, self.muon_orthogonal_sv_max
        return 2 * math.sqrt(low * high) / (low + high)

    def control_min_loss_shift(self, bound: float | None = None) -> float:
        """Minimum loss shift a negative control must produce to prove a match is non-vacuous.

        Derived rather than declared: a control that perturbs the mechanism under test has to
        move the compared quantity by more than the tolerance guarding it, or the "it matches"
        verdict carries no information. 2x leaves room for the perturbation being partial (e.g.
        rotating only the experts one rank owns) without letting noise satisfy it.

        ``bound`` is the tolerance actually in force at the call site, so a test on a wider bound
        cannot keep a control floor derived from a narrower one — which would let a control pass
        while proving nothing about the match it is supposed to underwrite.
        """
        return 2 * (self.parallel_vs_baseline_loss_abs if bound is None else bound)


TOL = _Tolerances()
