"""Named tolerances shared by correctness tests, keyed to what they compare so tests of one invariant move together. A
genuinely different comparison declares its own constant with its measured noise floor; do not stretch a shared
value."""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class _Tolerances:
    # Identical batch, and no forward reduction depends on arrival order (all-reduce, top-k slot combine, fixed-order
    # un-permute and index_add_), so every rank's loss is the same float (measured 0.0). Compare through
    # :meth:`identical_batch_ranks_agree`. DeepEP's receive order reorders only expert weight-gradient sums.
    ep_identical_batch_rank_spread_abs: float = 0.0
    # Pure TP or ETP: the all-reduce hands every rank one sum; a loss or norm read off a shard misses by orders more.
    all_reduced_rank_spread_abs: float = 1e-4
    # A pipeline run's per-step losses: the last stage broadcasts each one down its chain, so every rank holds the same
    # floats.
    pp_rank_identical_loss_abs: float = 1e-6

    # Step-0 loss, before optimizer drift: routing/all-to-all is not bitwise-dense.
    parallel_vs_baseline_loss_abs: float = 0.05
    # After a few steps SR and reduce order diverge the trajectories; bounds the trend.
    parallel_vs_baseline_train_loss_abs: float = 0.15
    # bf16 reduction reorder flips near-tied top-k picks (gpt-oss top-4-of-32), putting EP+TP and EP+ETP past the bound
    # above but several times under a rotated-expert control's shift. Other modes hold the generic bound (ETP on
    # Mistral4: 1e-2).
    router_pick_flip_loss_abs: float = 0.1

    # Per-token log-probs from Ulysses-gathered logits vs the full-sequence reference, at bf16.
    logprob_atol: float = 0.05

    # save→load round-trip at bf16 ULP scale.
    weight_atol: float = 1e-6
    # A resume replaying the uninterrupted run exactly. Measured bit-exact; 1e-4 is kernel headroom, under the 1.2e-4
    # first-step miss of adapters restarted from init.
    replayed_resume_loss_abs: float = 1e-4
    # The same comparison on the final trainable tensors, as a relative L2; a restart misses by ~1.
    replayed_resume_weight_rtol: float = 1e-4

    # Loss across a resume boundary (same data, same step).
    resume_loss_abs: float = 0.05
    # Fixed-batch loss before a save and after the resume, on a forward with no nondeterministic reduction (FSDP, TP,
    # CP, grouped-GEMM EP). bf16 round-trip noise; an unrestored or mis-gathered weight set moves it by more than 1.
    resume_fixed_batch_loss_abs: float = 1e-2
    # Norm after clipping vs max_norm: float rounding only. A wrong coefficient or missed plain-vs-DTensor scale misses
    # by a factor.
    clip_achieved_norm_rel: float = 1e-4

    # GC on vs off: the recompute changes only reduction order, so losses agree to bf16 noise; a corrupted frame moves
    # them further.
    gc_recompute_loss_rel: float = 1e-3
    # A trainer's loss normalizer vs a recount: fp32 divisions of the same integers. The failure is a whole factor
    # (per-rank, or /dp twice).
    recomputed_normalizer_abs: float = 1e-4

    # SCALE: a missing cross-rank reduction multiplies the norm by the axis size (measured 2.14 at axis 2 vs a 1.10
    # noise ceiling). DIRECTION: routing, permutation or sign corruption reorients at unchanged norm (noise floor 0.96,
    # 24-layer bf16 MoE router).
    grad_norm_ratio_max: float = 1.25
    grad_direction_cosine_min: float = 0.90

    # CP-rank-averaged gradient vs the full-sequence one. Only bf16 all-to-all and accumulation order separate them:
    # each direction holds 0.99, the total norm 10%. A wrong split reorients attention grads; a CP sum scales the norm
    # by cp_size. In a routed MoE that noise flips near-tied picks and perturbs downstream (gpt-oss: flipped routers
    # 0.984, worst other parameter 0.9897), so there the router/expert group holds 0.99 as one vector and the rest
    # cp_routed_grad_cosine_min.
    cp_grad_cosine_min: float = 0.99
    cp_routed_grad_cosine_min: float = 0.98
    cp_grad_norm_rtol: float = 0.10

    # bf16 noise on a ~128-token model and near-tie flips loosen direction; the norm ratio stays tight enough that a
    # missing or doubled /world_size divide (2.0 / 0.5) fails.
    ep_grad_cosine_min: float = 0.9
    ep_grad_norm_ratio_band: tuple[float, float] = (0.67, 1.5)
    # Under this fraction of its layer's largest, a gradient has no reliable direction, so TP and PP suites hold it to
    # absolute error: GatedDeltaNet's A_log is a sum of ~1e-9 terms that cancel 5-29x. Qwen3.5's A_log / dt_bias on
    # hard-decaying heads sit at 3e-9..7e-7, where the bf16 model scores cos 0.17-0.85 against fp32. Elsewhere they sit
    # at 1.6e-5..8e-4, where a dropped or doubled TP reduce moves the ratio by 2, and every other parameter at 1.5e-6
    # or more.
    grad_negligible_layer_fraction: float = 1e-6
    # Nemotron-H's tiny two-stage split leaves two Mamba layers on a stage, whose A_log and dt_bias sit near 4e-9 of
    # their layer's largest, under the fraction above on every layer, so no layer would score the kind. They hold cos
    # 0.99999 in bf16 and 1.0 in fp32 at this floor.
    nemotron_mamba_grad_negligible_layer_fraction: float = 1e-10
    # A near-cancelled pair outside the EP bars, within this multiple of its measured bf16 noise from fp32, is
    # rounding. The 1-8 element hyper-connection vectors of DeepSeek-V4 and GLM-5 Next cancel to 3e-5..1e-2 of their
    # layer's largest: bf16 misses fp32 by up to 3.3x their norm, and an EP policy lands up to 1.6x (DeepSeek-V4) and
    # 3.6x (GLM-5 Next) that noise. Routers and expert banks are never excused, and every parameter must land inside
    # the bars at its best resolved step.
    bf16_noise_excuse_factor: float = 8.0
    # Near-cancelled means a bf16 reference under this fraction of its layer's largest. Excused GPU pairs (hc vectors,
    # GatedDeltaNet A_log, dt_bias) sit at 2e-5..1.8e-3. Gemma 4's router sits outside its EP layers, so its pairs
    # (4e-5..3.8e-3) stay excusable: it syncs with its layer's FSDP2 group, whose fault reaches that layer's largest
    # gradients.
    bf16_noise_excusable_layer_fraction: float = 4e-3
    # How far a parallel run's pick may sit below the reference router's k-th logit, on tiny GptOss with logits spread
    # ~0.3. bf16 reduction order flips only near-ties (worst 4.4e-3 at etp2); a wrong selection falls short by a top-k
    # gap (median 0.06-0.17). On the reentrant suite a bf16 copy on an fp32 copy's picks falls at most 1e-2 short, the
    # (k+1)-th best 0.085+.
    router_near_tie_logit_abs: float = 3e-2
    # A router's own pick: selecting on fp32 softmax probabilities rounds logits ~3e-8 apart to one probability and
    # breaks the tie by index, while a selection on anything but those logits misses by a top-k gap.
    router_own_pick_logit_abs: float = 1e-6

    # EP layers vs stock MoE blocks pinned on the picks they dispatched (tests.common.routed_first_step), tiny bf16
    # Qwen3-MoE. CPU twin: outputs within 1.2e-2 rel L2, router and expert cos >= 0.9998, norm ratio within 0.6%. Two
    # bf16 implementations stay within ~3e-2, 0.999 and 2%. A halved combine moves outputs to 0.50 and every ratio to
    # 0.5; a dropped expert-TP reduce moves outputs to 0.72+ and some cosine to 0.94 or less. Picks hold
    # router_near_tie_logit_abs (faults: 2.7e-2..6.2e-2).
    routed_output_rel_l2: float = 0.1
    routed_grad_cosine_min: float = 0.99
    routed_grad_norm_ratio_band: tuple[float, float] = (0.9, 1.1)

    # A test's reimplementation of its trainer's objective vs the trainer's loss: only precision and reduction order
    # separate them. Relative to max(1, |expected|) in the reward and distillation pins, |expected| in offline GRPO's.
    exact_objective_rel: float = 2e-2

    kernel_atol: float = 1e-2
    kernel_rtol: float = 1e-2

    # Singular values of the Newton-Schulz output. Five composed Polar Express quintics map an input SV of at least
    # 1.22e-3 of the Frobenius norm into [0.846, 1.124], and bf16 moves those extremes by under 3e-3. Steps 1-3 only
    # act below ~1e-2. Without the safety factor, fp16 rounding overshoots the band on either path (to 1.2-1.4 on the
    # Gram path).
    muon_orthogonal_sv_min: float = 0.84
    muon_orthogonal_sv_max: float = 1.13
    # Smallest input SV, relative to the Frobenius norm, each path holds the band from: a margin over 1.22e-3 on the
    # square path, and measured on the rectangular (Gram) path, which loses the band below ~4e-3.
    muon_band_domain_square: float = 1.4e-3
    muon_band_domain_rectangular: float = 4e-3

    def identical_batch_ranks_agree(self, spread: float) -> bool:
        """Whether an identical-batch loss spread is within the bound, inclusive since the bound is exactly 0.0."""
        return spread <= self.ep_identical_batch_rank_spread_abs

    def muon_polar_cosine_min(self) -> float:
        """Smallest cosine between an orthogonalized update and its source's polar factor: the Kantorovich bound over
        the SV band."""
        low, high = self.muon_orthogonal_sv_min, self.muon_orthogonal_sv_max
        return 2 * math.sqrt(low * high) / (low + high)

    def control_min_loss_shift(self, bound: float | None = None) -> float:
        """Minimum loss shift a negative control must produce to prove a match is non-vacuous. ``bound`` is the
        tolerance in force at the call site. 2x leaves room for a partial perturbation without letting noise satisfy
        it."""
        return 2 * (self.parallel_vs_baseline_loss_abs if bound is None else bound)


TOL = _Tolerances()
