"""Shared building blocks for the PP × (EP / ep1-GMM / dense-FSDP) gradient-equivalence tests:

* a tiny GptOss MoE with **decisive routing** (router weights scaled ×16 so bf16-vs-fp32 hidden
  states cannot flip near-tied top-k picks — the compared math is unchanged);
* an **fp32 raw-HF reference** on the same bf16-rounded weights anchoring the absolute checks
  (per-replica loss, router grad norm-ratio + cosine, clip norm) under flip-tolerant bands;
* a **bf16 ep1-wrapper twin** (identical routing decisions, grouped-mm kernels and expert-output
  rounding) carrying the tight per-expert gradient gate;
* an **fp32 ep1-wrapper reference** for the modes where a bf16 twin is not a valid gate — ETP
  reorders the bf16 summation of the expert output and flips near-tied top-k picks, so the twin
  measures routing divergence rather than the sync under test;
* check blocks that emit the same check/metric names in every consuming test.

Every helper is topology-parametric (dp size, per-device batch, microbatch count); the consuming
test owns the :class:`~src.distributed.parallelism_config.ParallelismConfig` and the
topology-specific assertions.
"""

import math
from dataclasses import dataclass
from unittest import mock

import torch
import torch.distributed as dist
import torch.nn.functional as F
from datasets import Dataset
from torch.distributed.tensor import DTensor
from transformers import (
    AutoTokenizer,
    GptOssConfig,
    GptOssForCausalLM,
    Qwen3Config,
    Qwen3ForCausalLM,
    default_data_collator,
)
from trl import SFTConfig

from src.distributed.expert_parallel import base_layer
from src.distributed.expert_parallel.base_layer import EPMoELayerBase
from src.distributed.expert_parallel.config import EPConfig
from src.distributed.expert_parallel.patching import patch_moe_model_for_ep
from src.distributed.parallelism_config import ParallelismConfig
from src.trainers.sft import DistributedSFTTrainer
from tests.common.distributed import world_spread
from tests.common.models import QWEN3_0_6B, TINY_GPTOSS_CONFIG, TINY_QWEN3_CONFIG
from tests.common.utils import log, max_or_nan

SEQ_LEN = 64
LOSS_REL_TOL = 2e-2
# Per-expert grads gate against the bf16 ep1-wrapper twin (identical routing decisions + grouped-mm
# kernels — measured ≤0.007 in the pp2×ep1 test), so the gate can be tight. The fp32 raw-HF
# reference cannot serve here: its hidden states differ from the bf16 run relatively, flipping
# near-tied router top-k picks (~1 token/expert), which shifts whole-expert grad mass by ~0.1-0.3
# with no parallelism bug at all. It still anchors the absolute checks (loss, router norm/cosine,
# clip norm) under flip-tolerant bands.
EXPERT_GRAD_REL_TOL = 2e-2
ROUTER_NORM_RATIO_BAND = (0.7, 1.4)
ROUTER_COSINE_MIN = 0.9
# Router grads vs the bf16 twin. The ×16-scaled (decisive) router saturates its softmax, so router
# grads are tiny and bf16-noise-dominated: the fp32 cosine is data-dependent (0.98 on the 4-row
# batch, 0.84 on one layer of the 8-row batch with no parallelism bug), while the twin — identical
# kernels and routing — separates precision noise from a real sync bug. rel_err < 5e-2 implies
# cosine > 0.9987, far tighter than any usable fp32 band.
ROUTER_TWIN_REL_TOL = 5e-2
CLIP_NORM_REL_TOL = 5e-2  # vs fp32: the flipped tokens' grads genuinely differ between the two runs


@dataclass
class UnsplitReference:
    """One unsplit model's full-batch backward, packaged for the equivalence checks.

    ``slice_losses[r]`` is DP replica ``r``'s expected loss ``S_r / (G / dp)`` — the ragged label
    prefixes make the per-replica token counts unequal on purpose, and the PP trainer normalizes
    every replica by the DP-global count ``G/dp``.

    ``expert_grads`` / ``router_grads`` / ``expert_param_names`` are populated only when the
    reference model carries EP wrappers: expert parameters have no sharding-independent name (ETP
    splits the fused ``gate_up_proj``, grouped-GEMM splits it again), so they are compared through
    :meth:`~src.distributed.expert_parallel.base_layer.EPMoELayerBase.gather_expert_grads` — the
    same canonical layout on either side — and excluded from the name-keyed comparison.
    """

    grads: dict[str, torch.Tensor]
    param_names: set[str]
    slice_losses: list[float]
    total_norm: float
    total_tokens: int
    expert_grads: dict[str, dict[str, torch.Tensor]]  # {EP module: gather_expert_grads()}
    router_grads: dict[str, torch.Tensor]  # {EP module: cat of sorted router param grads, fp32}
    expert_param_names: set[str]


def build_dense_model(tokenizer=None, *, vocab_size: int | None = None) -> Qwen3ForCausalLM:
    """Tiny dense Qwen3, identical on every rank (fixed seed).

    pad/eos ids must live inside the model's vocab: TRL copies the tokenizer's ids into the model
    config, and the saved config would otherwise fail ``from_pretrained``'s padding_idx assert. With
    no ``tokenizer`` the tiny config's own 1024-token vocab and ids 0/1 are used; a suite that
    tokenizes with a real tokenizer passes it together with that checkpoint's ``vocab_size``.
    """
    torch.manual_seed(1234)
    config = TINY_QWEN3_CONFIG if vocab_size is None else {**TINY_QWEN3_CONFIG, "vocab_size": vocab_size}
    return Qwen3ForCausalLM(
        Qwen3Config(
            **config,
            pad_token_id=0 if tokenizer is None else tokenizer.pad_token_id,
            eos_token_id=1 if tokenizer is None else tokenizer.eos_token_id,
        )
    )


def build_moe_model(dtype: torch.dtype = torch.bfloat16) -> GptOssForCausalLM:
    """Tiny GptOss MoE, identical on every rank (fixed seed)."""
    torch.manual_seed(1234)
    model = GptOssForCausalLM(GptOssConfig(**TINY_GPTOSS_CONFIG, pad_token_id=0, eos_token_id=1))
    # Decisive routing: at init the router logits are near-tied, so the bf16 run and the reference
    # flip top-k selections for a few tokens — a data-dependent divergence that shifts whole-expert
    # grad mass and has nothing to do with the parallelism under test. Scaling the router weights
    # widens the logit gaps far past bf16 noise so both runs route identically; the math compared
    # (dispatch/combine/sync) is unchanged.
    with torch.no_grad():
        for layer in model.model.layers:
            layer.mlp.router.weight.mul_(16.0)
    return model.to(dtype)


def build_patched_moe_model(config: ParallelismConfig, dtype: torch.dtype = torch.bfloat16) -> GptOssForCausalLM:
    """The MoE model carrying the config's EP wrappers + prepared (non-collective) EP buffers."""
    model = build_moe_model(dtype)
    patch_moe_model_for_ep(model, config.create_ep_config())
    return model


def count_ep_layers(model: torch.nn.Module) -> int:
    return sum(isinstance(m, EPMoELayerBase) for m in model.modules())


def build_ragged_batch(global_batch: int, seq_len: int = SEQ_LEN, seed: int = 7) -> tuple[torch.Tensor, torch.Tensor]:
    """Fixed random rows with ragged ignore prefixes: replicas hold unequal loss-token counts."""
    g = torch.Generator().manual_seed(seed)
    rows = torch.randint(4, TINY_GPTOSS_CONFIG["vocab_size"], (global_batch, seq_len), generator=g)
    labels = rows.clone()
    for i in range(global_batch):
        labels[i, : 4 + 3 * i] = -100
    return rows, labels


def token_ce_sums(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Per-row summed next-token CE in fp32 (the S_r pieces of the global S)."""
    shift_logits = logits[:, :-1].float()
    shift_labels = labels[:, 1:]
    ce = F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1), ignore_index=-100, reduction="none"
    )
    return ce.view(shift_labels.shape).sum(dim=1)


def rel_err(test: torch.Tensor, ref: torch.Tensor) -> float:
    denom = ref.float().norm().clamp_min(1e-12)
    return float((test.float() - ref.float()).norm() / denom)


def inert_ep1_config(rank: int, use_grouped_gemm: bool = True) -> EPConfig:
    """A single-process ep1 EPConfig whose hooks and dispatch stay inert.

    ``stage_world_size=1`` / ``stage_base_rank=global_rank`` keeps its group creation collective-consistent
    (every rank creates every single-rank group in the same order).
    """
    return EPConfig(
        ep_size=1,
        stage_world_size=1,
        stage_base_rank=rank,
        node_local=True,
        nvlink_domain_size=1,
        use_grouped_gemm=use_grouped_gemm,
        experts_fsdp_managed=True,
    )


def run_unsplit_reference(model, rows: torch.Tensor, labels: torch.Tensor, device, dp_size: int) -> UnsplitReference:
    """Full-global-batch forward+backward on an unsplit model; grads keyed by global param name.

    An EP-wrapped model additionally yields its experts through ``gather_expert_grads`` (no
    collective at ``ep_group_size == 1``) so a distributed run can be compared shard-for-shard in
    one canonical layout, and the expert parameter names so the name-keyed comparison can skip them.
    """
    model.train()
    out = model(input_ids=rows.to(device), attention_mask=torch.ones_like(rows).to(device))
    row_sums = token_ce_sums(out.logits, labels.to(device))
    total_tokens = int((labels[:, 1:] != -100).sum())
    loss = row_sums.sum() / total_tokens  # S / G
    loss.backward()
    grads = {name: p.grad.detach().clone() for name, p in model.named_parameters() if p.grad is not None}
    param_names = {name for name, p in model.named_parameters() if p.requires_grad}
    per_replica = rows.shape[0] // dp_size
    slice_losses = [
        float(row_sums[r * per_replica : (r + 1) * per_replica].sum() / (total_tokens / dp_size))
        for r in range(dp_size)
    ]  # S_r / (G / dp)
    total_norm = float(torch.sqrt(sum(g.float().norm() ** 2 for g in grads.values())))

    ep_modules = [(name, m) for name, m in model.named_modules() if isinstance(m, EPMoELayerBase)]
    expert_grads = {name: m.gather_expert_grads(device=device) for name, m in ep_modules}
    router_grads = {
        name: torch.cat(
            [grads[f"{name}.router.{p}"].float().flatten() for p in sorted(n for n, _ in m.router.named_parameters())]
        )
        for name, m in ep_modules
    }
    expert_param_names = {f"{name}.{attr}" for name, m in ep_modules for attr, _ in m.expert_named_params()}

    del out
    model.zero_grad(set_to_none=True)
    return UnsplitReference(
        grads, param_names, slice_losses, total_norm, total_tokens, expert_grads, router_grads, expert_param_names
    )


def fp32_reference(rows: torch.Tensor, labels: torch.Tensor, device, dp_size: int) -> UnsplitReference:
    """fp32 unsplit raw-HF reference on the same bf16-rounded weights (absolute-scale anchor)."""
    return run_unsplit_reference(build_moe_model().float().to(device), rows, labels, device, dp_size)


def ep1_reference(
    rows: torch.Tensor,
    labels: torch.Tensor,
    device,
    dp_size: int,
    rank: int,
    dtype: torch.dtype = torch.bfloat16,
    use_grouped_gemm: bool = True,
) -> UnsplitReference:
    """Unsplit reference carrying the same ep1 EP wrappers, single-process.

    At ``dtype=torch.float32`` with ``use_grouped_gemm=False`` this is the exact-arithmetic
    reference an fp32 distributed run is gated against: the expert compute is the per-expert loop
    over plain torch ops, so no kernel is asked to run outside the dtype it supports.
    """
    model = build_moe_model(dtype)
    patch_moe_model_for_ep(model, inert_ep1_config(rank, use_grouped_gemm))
    return run_unsplit_reference(model.to(device), rows, labels, device, dp_size)


def bf16_twin_reference(ctx, rows: torch.Tensor, labels: torch.Tensor) -> UnsplitReference:
    """bf16 ep1-wrapper twin: identical routing, grouped-mm kernels and expert-output rounding → tight
    per-expert/router gates. Pins the process to the unfused expert-output sum first
    (:func:`_pin_unfused_expert_sum`), which the EP run under test then shares."""
    _pin_unfused_expert_sum(ctx)
    return ep1_reference(rows, labels, ctx.device, dp_size=1, rank=ctx.rank)


class _UnfusedWeightedUnpermute:
    """:class:`~src.kernels.moe_permute.MoEWeightedUnpermute`'s contract with each weighted expert row
    rounded to the run dtype before an ``index_add`` sums a token's rows."""

    @staticmethod
    def apply(expert_out, weights, sorted_token_idx, inv_map):
        weighted = expert_out * weights.unsqueeze(-1)
        return weighted.new_zeros((inv_map.shape[0], weighted.shape[-1])).index_add(0, sorted_token_idx, weighted)


def _pin_unfused_expert_sum(ctx) -> None:
    """Run every EP layer in this process on the unfused expert-output sum, which rounds each weighted
    expert output to bf16 before adding. With two experts per token those two adds commute, so the ep1
    twin and an EP run that splits a token's experts across ranks (each partial rounded, then summed by
    DeepEP's combine) round alike; three or more would not.

    The fused un-permute accumulates a rank's weighted rows in fp32 and rounds once, so the two runs
    would round differently and near-tied top-k picks flip downstream. The permute's gather-reduce
    backward needs no pin: it sums gradient rows that are already bf16, which two-way commutes the same.
    The fused kernels' own numerics are gated by ``test_moe_permute`` and the family kernel-stack numerics.
    """
    patcher = mock.patch.object(base_layer, "MoEWeightedUnpermute", _UnfusedWeightedUnpermute)
    patcher.start()
    ctx.on_teardown(patcher.stop)


def build_pp_sft_trainer(
    ctx,
    config: ParallelismConfig,
    model,
    rows: torch.Tensor,
    labels: torch.Tensor,
    *,
    per_device_batch: int,
    n_microbatches: int,
    max_steps: int = 2,
    seq_len: int = SEQ_LEN,
    gradient_checkpointing: bool = False,
    moe_balancing: str = "auto",
    bf16: bool = True,
):
    """The real DistributedSFTTrainer over the PP config (the trainer splits the model itself)."""
    args = SFTConfig(
        output_dir=ctx.output_dir,
        max_steps=max_steps,
        per_device_train_batch_size=per_device_batch,
        gradient_accumulation_steps=n_microbatches,  # PP consumes accumulation as the microbatch count
        max_length=seq_len,
        save_strategy="no",
        report_to=[],
        logging_nan_inf_filter=False,
        seed=42,
        bf16=bf16,
        gradient_checkpointing=gradient_checkpointing,  # TRL defaults it True; off keeps the base run kernel-plain
    )
    dataset = Dataset.from_dict(
        {
            "input_ids": rows.tolist(),
            "labels": labels.tolist(),
            "attention_mask": torch.ones_like(rows).tolist(),
        }
    )
    return DistributedSFTTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        processing_class=AutoTokenizer.from_pretrained(QWEN3_0_6B),
        data_collator=default_data_collator,
        parallelism_config=config,
        moe_balancing=moe_balancing,
    )


def dp_slice_batch(rows: torch.Tensor, labels: torch.Tensor, dp_rank: int, per_device_batch: int) -> dict:
    """This DP replica's slice of the global batch (every rank of a chain gets the same slice)."""
    sl = slice(dp_rank * per_device_batch, (dp_rank + 1) * per_device_batch)
    return {
        "input_ids": rows[sl],
        "labels": labels[sl],
        "attention_mask": torch.ones_like(rows[sl]),
    }


def check_loss_vs_reference(
    loss, reference: UnsplitReference, dp_rank: int, tol: float = LOSS_REL_TOL
) -> tuple[dict, dict]:
    """Per-replica loss vs the reference slice loss."""
    expected = reference.slice_losses[dp_rank]
    loss_err = abs(float(loss) - expected) / abs(expected)
    return {"loss_matches_reference": loss_err < tol}, {"loss_rel_err": loss_err}


def check_router_grads(
    stage,
    ep_modules: list,
    reference: UnsplitReference,
    stage_group,
    band: tuple[float, float] = ROUTER_NORM_RATIO_BAND,
    cos_min: float = ROUTER_COSINE_MIN,
    twin_router_grads: dict[str, torch.Tensor] | None = None,
    twin_tol: float = ROUTER_TWIN_REL_TOL,
) -> tuple[dict, dict]:
    """Router grads: bit-identical across the stage's ranks, plus a gate vs the references.

    Without a twin, the fp32 reference carries both scale and direction (norm-ratio band +
    cosine). With ``twin_router_grads``, direction moves to the tight twin rel-err gate
    (``router_grads_match_twin``) and the fp32 reference keeps only the absolute norm-ratio band —
    the fp32 cosine on the saturated router's tiny grads is bf16-noise-dominated and
    data-dependent (see :data:`ROUTER_TWIN_REL_TOL`); it stays reported as a metric.
    """
    router_ok, router_identical, twin_ok = True, True, True
    worst_ratio, worst_cos, worst_twin_err = 1.0, 1.0, 0.0
    for local_name, module in ep_modules:
        router_param_names = sorted(name for name, _ in module.router.named_parameters())
        router_params = dict(module.router.named_parameters())
        grads = torch.cat([router_params[n].grad.float().flatten() for n in router_param_names])
        gathered = [torch.zeros_like(grads) for _ in range(dist.get_world_size(stage_group))]
        dist.all_gather(gathered, grads, group=stage_group)
        router_identical &= all(torch.equal(grads, g) for g in gathered)

        global_name = stage.global_parameter_name(local_name)
        ref = torch.cat([reference.grads[f"{global_name}.router.{n}"].flatten() for n in router_param_names])
        ratio = float(grads.norm() / ref.norm().clamp_min(1e-12))
        cos = float(F.cosine_similarity(grads.unsqueeze(0), ref.unsqueeze(0)))
        worst_ratio = ratio if abs(ratio - 1) > abs(worst_ratio - 1) else worst_ratio
        worst_cos = min(worst_cos, cos)
        if twin_router_grads is None:
            router_ok &= band[0] < ratio < band[1] and cos > cos_min
        else:
            router_ok &= band[0] < ratio < band[1]
            twin_err = rel_err(grads, twin_router_grads[global_name])
            worst_twin_err = max(worst_twin_err, twin_err) if math.isfinite(twin_err) else math.inf
            twin_ok &= twin_err < twin_tol
    checks = {
        "router_grads_bitwise_identical_across_stage": router_identical,
        "router_grads_match_reference": router_ok,
    }
    metrics = {"router_worst_norm_ratio": worst_ratio, "router_worst_cosine": worst_cos}
    if twin_router_grads is not None:
        checks["router_grads_match_twin"] = twin_ok
        metrics["router_worst_twin_rel_err"] = worst_twin_err
    return checks, metrics


def check_expert_grads_vs_reference(
    ctx,
    stage,
    ep_modules: list,
    reference: UnsplitReference,
    *,
    tol: float = EXPERT_GRAD_REL_TOL,
) -> tuple[dict, dict]:
    """Expert grads: reassemble this rank's shards with the layer's own gather, compare per expert.

    Both sides go through ``gather_expert_grads``, so the comparison follows the layout the family
    actually holds instead of a hardcoded attribute list — which under ETP names attributes that do
    not exist and silently compares nothing. The gather is collective over the expert-TP and
    dispatch groups; ``down_proj_bias`` lives on expert-TP rank 0 only, so a rank's key set can be a
    strict subset of the reference's and coverage is pinned world-wide rather than per rank.
    """
    num_experts = TINY_GPTOSS_CONFIG["num_local_experts"]
    expert_ok, compared_tensors, experts_covered = True, 0, 0
    errs, failures, compared_keys = [], [], set()
    for local_name, module in ep_modules:
        global_name = stage.global_parameter_name(local_name)
        ref_grads = reference.expert_grads[global_name]
        for key, full_grad in module.gather_expert_grads(device=ctx.device).items():
            ref = ref_grads[key]
            if full_grad.shape != ref.shape:
                log(f"[rank {ctx.rank}] shape mismatch {global_name}.{key}: {full_grad.shape} vs {ref.shape}")
                expert_ok = False
                continue
            compared_tensors += 1
            compared_keys.add(f"{global_name}.{key}")
            for e in range(ref.shape[0]):
                if float(ref[e].float().norm()) == 0.0:
                    # A dead expert (no tokens routed to it) would make this comparison vacuous.
                    log(f"[rank {ctx.rank}] reference grad is all-zero: {global_name}.{key}[{e}]")
                    expert_ok = False
                    continue
                experts_covered += 1
                err = rel_err(full_grad[e], ref[e])
                errs.append(err)
                if err >= tol:
                    failures.append((err, f"{global_name}.{key}", e))
                expert_ok &= err < tol
    for err, name, e in sorted(failures, reverse=True)[:10]:
        log(f"[rank {ctx.rank}] expert grad off vs reference: {name}[{e}] rel_err={err:.4f}")

    gathered: list = [None] * ctx.world_size
    dist.all_gather_object(gathered, compared_keys)
    covered = set().union(*gathered)
    expected = {f"{module}.{key}" for module, grads in reference.expert_grads.items() for key in grads}
    if covered != expected:
        log(f"[rank {ctx.rank}] expert key diff: extra={covered - expected} missing={expected - covered}")
    checks = {
        "expert_grads_match_reference": expert_ok,
        "expert_comparison_not_vacuous": compared_tensors > 0
        and experts_covered == compared_tensors * num_experts
        and covered == expected,
    }
    metrics = {
        "worst_expert_grad_rel_err": max_or_nan(errs, default=1.0),
        "expert_tensors_compared": compared_tensors,
    }
    return checks, metrics


def compare_stage_grads_vs_reference(
    ctx, stage, reference: UnsplitReference, tol: float, *, exclude_experts: bool = False
) -> tuple[dict, dict]:
    """Every stage param's grad vs the reference via global names (+ world-wide name-union cover).

    DTensor grads are gathered with ``full_tensor()`` (collective over their mesh); the union of
    stage param names across the world must cover the reference exactly — nothing dropped by the
    split, nothing double-owned under a wrong name.

    ``exclude_experts`` drops the EP layers' expert weights from both sides (the reference's own
    ``expert_param_names``), leaving every other parameter — routers included — compared by name.
    Required whenever the run shards the experts along an axis the reference does not, since then
    the expert parameters have neither a matching name nor a matching shape;
    :func:`check_expert_grads_vs_reference` covers them instead.
    """
    skip_ids = (
        {id(p) for m in stage.modules() if isinstance(m, EPMoELayerBase) for _, p in m.expert_named_params()}
        if exclude_experts
        else set()
    )
    expected_names = reference.param_names - (reference.expert_param_names if exclude_experts else set())
    worst_err, worst_name, compared, my_names = 0.0, "", 0, set()
    grads_ok = True
    for local_name, param in stage.named_parameters():
        if not param.requires_grad or id(param) in skip_ids:
            continue
        global_name = stage.global_parameter_name(local_name)
        my_names.add(global_name)
        grad = param.grad
        full = grad.full_tensor() if isinstance(grad, DTensor) else grad  # collective over the mesh
        ref = reference.grads.get(global_name)
        if ref is None or full.shape != ref.shape:
            log(f"[rank {ctx.rank}] missing/mismatched reference grad for {global_name}")
            grads_ok = False
            continue
        err = rel_err(full, ref)
        # A NaN compares false against every bound, so a non-finite error counts as infinite.
        err = err if math.isfinite(err) else math.inf
        if err > worst_err:
            worst_err, worst_name = err, global_name
        compared += 1
        grads_ok &= err < tol
    log(f"[rank {ctx.rank}] worst grad rel err {worst_err:.4f} at {worst_name}")

    gathered_names: list = [None] * ctx.world_size
    dist.all_gather_object(gathered_names, my_names)
    union = set().union(*gathered_names)
    if union != expected_names:
        log(f"[rank {ctx.rank}] name diff: extra={union - expected_names} missing={expected_names - union}")
    local_expected = sum(1 for _, p in stage.named_parameters() if p.requires_grad and id(p) not in skip_ids)
    checks = {
        "all_stage_grads_match_reference": grads_ok,
        "every_stage_param_compared": compared == local_expected,
        "stage_param_names_cover_reference": union == expected_names,
    }
    return checks, {"worst_grad_rel_err": worst_err, "params_compared": compared}


def check_clip_norm(ctx, clip_norm, reference: UnsplitReference, tol: float = CLIP_NORM_REL_TOL) -> tuple[dict, dict]:
    """Clip norm: one identical value on every rank, matching the reference total norm."""
    norm_t = clip_norm.detach().float().reshape(1).to(ctx.device)
    spread = world_spread(float(norm_t), device=ctx.device)
    ref_total_norm = reference.total_norm
    checks = {
        "clip_norm_rank_uniform": spread < 1e-5 * max(1.0, ref_total_norm),
        "clip_norm_matches_reference": abs(float(norm_t) - ref_total_norm) / ref_total_norm < tol,
    }
    return checks, {"clip_norm": float(norm_t), "clip_norm_ref": ref_total_norm, "clip_norm_spread": spread}


def count_checkpoint_calls(stage) -> list[int]:
    """Wrap each stage layer's installed checkpoint function with a counter; returns ``[count]``.

    Call after the trainer enabled GC. The counter is the anti-vacuity gate for every GC leg: HF
    only checkpoints a layer that is training with grad enabled, so a silently-skipped gate would
    otherwise turn a GC run into a plain re-test of the non-GC path.
    """
    calls = [0]
    for layer in stage.model.layers:
        inner = layer._gradient_checkpointing_func

        def counting(*args, _inner=inner, **kwargs):
            calls[0] += 1
            return _inner(*args, **kwargs)

        layer._gradient_checkpointing_func = counting
    return calls


def check_gc_engaged(stage, calls: list[int], n_microbatches: int) -> tuple[dict, dict]:
    """Every stage layer checkpointed on every microbatch forward (torch's one-time shape-inference
    pass may add one more sweep)."""
    n_layers = len(stage.model.layers)
    lo = n_layers * n_microbatches
    return (
        {"gc_engaged_every_layer_every_microbatch": lo <= calls[0] <= n_layers * (n_microbatches + 1)},
        {"gc_calls": calls[0], "gc_calls_expected_min": lo},
    )


def check_expert_load_counts(
    stage, ep_modules: list, calls: list[int], tokens_per_forward: int, n_microbatches: int
) -> tuple[dict, dict]:
    """Expert-load accounting: exactly one count per (token, top-k slot) of every training forward.

    The checkpoint recompute re-runs the router on the same tokens, so a recompute that is not
    recognised as one doubles every layer's counter — invisible in the loss, corrupting the
    ``moe/*`` metrics and, under bias_update balancing, the balancing update itself. The pipeline's
    one-time shape-inference forward runs under ``no_grad`` and drives no optimizer step, so it
    must not count — the same rule that keeps a KL-reference forward out of the balancing signal —
    which is why the expected total is calibrated on the schedule's grad-driven microbatches while
    ``calls`` (one checkpoint call per layer per forward, shape inference included) separately
    proves how many forwards ran at all. Requires :meth:`enable_bias_balancing` on the layers
    before the step.
    """
    total_forwards = calls[0] // len(stage.model.layers)
    expected = tokens_per_forward * TINY_GPTOSS_CONFIG["num_experts_per_tok"] * n_microbatches
    counts = {name: float(module.expert_load_counter.sum()) for name, module in ep_modules}
    return (
        {
            "expert_load_counted_once": all(count == expected for count in counts.values()),
            "expert_load_not_vacuous": bool(counts) and expected > 0,
        },
        {
            "expert_load_per_layer": sorted(set(counts.values())),
            "expert_load_expected": expected,
            "total_forwards_incl_shape_inference": total_forwards,
        },
    )


def second_step_dispatch_works(trainer, batch: dict, first_loss) -> bool:
    """A second step proves the kept layers' DeepEP dispatch still works after the stage split."""
    for p in trainer.model.parameters():
        p.grad = None
    trainer.state.global_step = 1  # the deferred sweep's double-run guard keys on the step
    loss2 = trainer.training_step(trainer.model, dict(batch))
    trainer.accelerator.clip_grad_norm_(trainer.model.parameters(), 1e6)
    return bool(torch.isfinite(loss2)) and abs(float(loss2) - float(first_loss)) < 1.0
