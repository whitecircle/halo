#!/usr/bin/env python
"""TP=2 on dense Qwen3-0.6B reproduces the single-GPU loss and total grad norm.

A TP grad norm that drops lm_head/embedding or mis-aggregates across the TP axis must fail against the unsharded
baseline.
"""

import math

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.distributed.tensor_parallel import gather_state_dict_for_save
from trl import SFTConfig

from halo.distributed.mesh import MeshDim, mesh_dim_names, model_tp_mesh, tp_group
from halo.distributed.parallelism_config import ParallelismConfig
from halo.model_loading.parallel.loader import load_distributed_model
from halo.runtime.collectives import barrier
from halo.runtime.dtensor import materialize_dtensor
from halo.trainers.sft import DistributedSFTTrainer
from tests.common.datasets import create_sft_dataset
from tests.common.distributed import ensure_model_downloaded, group_max_abs_diff, world_mean, world_spread
from tests.common.ep_reference import fixed_chat_batch
from tests.common.harness import gpu_test_main
from tests.common.memory import cleanup_memory, gpu_mem_gb
from tests.common.models import QWEN3_0_6B
from tests.common.numerics import max_or_nan
from tests.common.rank_log import log, log_all
from tests.common.tolerances import TOL

MODEL_NAME = QWEN3_0_6B
TP_SIZE = 2
SEQ_LEN = 128
SEED = 42

# Tight: TP is a math-exact rearrangement of the same compute, so only bf16 reduction-order jitter remains.
LOSS_ABS_TOL = 0.02
LOSS_REL_TOL = 0.01
GRAD_NORM_REL_TOL = 0.03

# One parameter per TP style, each reduced by a different rule: colwise/rowwise are disjoint plain slices reassembled
# by the plan; `*_norm` weights are `replicated_with_grad_allreduce` (replicated storage, head-partial gradient, needs
# a TP SUM); layernorm/embed are truly replicated. A wrong rule is invisible in the global norm on a small tensor.
COMPARED_PARAMS = (
    "model.layers.0.self_attn.q_proj.weight",
    "model.layers.0.self_attn.k_proj.weight",
    "model.layers.0.self_attn.v_proj.weight",
    "model.layers.0.self_attn.o_proj.weight",
    "model.layers.0.self_attn.q_norm.weight",
    "model.layers.0.self_attn.k_norm.weight",
    "model.layers.0.mlp.gate_proj.weight",
    "model.layers.0.mlp.down_proj.weight",
    "model.layers.0.input_layernorm.weight",
    "model.embed_tokens.weight",
)
# Loose next to the 3% aggregate: a single bf16 gradient carries far more reassociation noise than the whole-model sum.
# Still decisive, since a missing TP sum lands at ratio 0.5 and averaging two disjoint slices at rel_err ~1.0.
GRAD_TENSOR_REL_TOL = 0.20
GRAD_TENSOR_RATIO_TOL = 0.20


def compute_baseline_loss(tokenizer, device):
    """Rank 0's loss and fp32 grad norm with no TP, plus its gradient of each ``COMPARED_PARAMS`` entry: a `q_norm`
    weight is head_dim-sized, so even a `tp_size`x error moves the global norm far less than the bf16 tolerance."""
    rank = dist.get_rank()
    baseline_grads: dict[str, torch.Tensor] = {}
    result_tensor = torch.zeros(2, device=device)

    if rank == 0:
        log("  Loading baseline model (no TP) on rank 0...")
        log(f"  GPU memory before load: {gpu_mem_gb():.2f} GB")

        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
            device_map={"": 0},
        )
        model.eval()

        log(f"  GPU memory after load: {gpu_mem_gb():.2f} GB")

        input_ids, attention_mask, labels = fixed_chat_batch(tokenizer, SEQ_LEN, device, seed=SEED)

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            result_tensor[0] = outputs.loss
        del outputs

        model.zero_grad(set_to_none=True)
        loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).loss
        loss.backward()
        grad_sq = torch.zeros((), device=device, dtype=torch.float32)
        for p in model.parameters():
            if p.grad is not None:
                grad_sq += p.grad.norm(dtype=torch.float32) ** 2
        result_tensor[1] = grad_sq.sqrt()

        named = dict(model.named_parameters())
        for name in COMPARED_PARAMS:
            param = named.get(name)
            assert param is not None and param.grad is not None, f"{MODEL_NAME} has no gradient for {name}"
            baseline_grads[name] = param.grad.detach().float().cpu()

        log(f"  Baseline loss: {result_tensor[0].item():.6f}")
        log(f"  Baseline grad norm (fp32, all params): {result_tensor[1].item():.6f}")
        log(f"  Baseline per-param grads kept: {len(baseline_grads)}")

        model.zero_grad(set_to_none=True)
        del model, loss, named
        cleanup_memory()
        log(f"  GPU memory after cleanup: {gpu_mem_gb():.2f} GB")

    barrier()
    dist.broadcast(result_tensor, src=0)
    return result_tensor[0].item(), result_tensor[1].item(), baseline_grads


def compute_tp_loss_and_grad_norm(tokenizer, local_rank, output_dir):
    """Forward loss and TP-aware grad norm at TP=2 through ``_sync_tp_replicated_grads`` +
    ``_compute_global_grad_norm``, the path the TP clip calls in production. Means and spreads are over the world."""
    device = f"cuda:{local_rank}"

    log("\n  Loading TP=2 model...")

    parallelism_config = ParallelismConfig(tp_size=TP_SIZE)
    log(f"  Config: {parallelism_config.summary()}")

    model, _ = load_distributed_model(
        model_name_or_path=MODEL_NAME,
        parallelism_config=parallelism_config,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )

    # A severed tie is two Parameter objects here, long before it surfaces as a 0.63 rel_err on the embedding gradient.
    if model.config.tie_word_embeddings:
        assert model.get_output_embeddings().weight is model.get_input_embeddings().weight, (
            "tie_word_embeddings=True but embed_tokens and lm_head are two independent parameters"
        )

    train_dataset = create_sft_dataset(8, tokenizer, seed=SEED)
    sft_config = SFTConfig(
        output_dir=output_dir,
        max_steps=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=2e-5,
        bf16=True,
        gradient_checkpointing=False,
        use_liger_kernel=False,
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        logging_nan_inf_filter=False,
        max_length=SEQ_LEN,
        dataloader_drop_last=True,
        dataloader_num_workers=0,
        max_grad_norm=1.0,
    )
    trainer = DistributedSFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        parallelism_config=parallelism_config,
    )
    assert trainer.parallelism_config.is_tp_mode, "expected TP mode"

    log(f"  GPU memory after TP load: {gpu_mem_gb():.2f} GB")

    input_ids, attention_mask, labels = fixed_chat_batch(tokenizer, SEQ_LEN, device, seed=SEED, broadcast=True)

    tp_model = trainer.model

    tp_model.eval()
    with torch.no_grad():
        outputs = tp_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        tp_loss_local = outputs.loss.item()
    del outputs
    log_all(f"  TP loss: {tp_loss_local:.6f}")

    tp_model.train()
    tp_model.zero_grad(set_to_none=True)
    loss = tp_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).loss
    loss.backward()

    params = [p for p in tp_model.parameters() if p.grad is not None]
    # Mirrors the TP clip.
    trainer._sync_tp_replicated_grads(params)
    tp_synced_identical, tp_synced_count = assert_tp_synced_grads_identical(trainer, tp_model)
    tp_grad_norm_local = float(trainer._compute_global_grad_norm(params))
    log_all(f"  TP grad norm: {tp_grad_norm_local:.6f}")

    tp_loss = (world_mean(tp_loss_local), world_spread(tp_loss_local))
    tp_grad_norm = (world_mean(tp_grad_norm_local), world_spread(tp_grad_norm_local))

    tp_grads = reassemble_tp_grads(tp_model)

    tp_model.zero_grad(set_to_none=True)
    del model, trainer, tp_model, loss
    cleanup_memory()

    return tp_loss, tp_grad_norm, tp_grads, (tp_synced_identical, tp_synced_count)


def assert_tp_synced_grads_identical(trainer, tp_model) -> tuple[bool, int]:
    """COLLECTIVE. After ``_sync_tp_replicated_grads``, every param without a ``tp`` mesh dim has a bit-identical
    gradient across the TP group. Selected by mesh dims, independent of the trainer's own buckets, so a sweep skipping
    a class of params fails here. The identical batch keeps the walk order rank-uniform."""
    unwrapped = tp_model.module if hasattr(tp_model, "module") else tp_model
    group = tp_group(trainer._device_mesh)
    checked, diffs = 0, []
    for _name, p in unwrapped.named_parameters():
        if p.grad is None:
            continue
        if isinstance(p.data, DTensor) and MeshDim.TP in mesh_dim_names(p.data.device_mesh):
            continue  # Plan-sharded: ranks legitimately hold different slices.
        grad = p.grad.to_local() if isinstance(p.grad, DTensor) else p.grad
        checked += 1
        diffs.append(group_max_abs_diff(grad, group))
    max_diff = max_or_nan(diffs, default=0.0)
    log_all(f"  TP-synced grads: {checked} params checked, max cross-TP diff {max_diff:.3e}")
    return max_diff == 0.0 and checked > 0, checked


def reassemble_tp_grads(tp_model) -> dict[str, torch.Tensor]:
    """COLLECTIVE. The unsharded gradient of each ``COMPARED_PARAMS`` entry through ``gather_state_dict_for_save``, the
    call the TP checkpoint save makes, so a plan the save would mis-gather is mis-gathered here too."""
    unwrapped = tp_model.module if hasattr(tp_model, "module") else tp_model
    named = dict(unwrapped.named_parameters())
    local = {}
    for name in COMPARED_PARAMS:
        param = named.get(name)
        assert param is not None and param.grad is not None, f"no live gradient for {name} under TP"
        local[name] = materialize_dtensor(param.grad)  # Collective under TP+DP; identity at dp=1.
    full = gather_state_dict_for_save(local, unwrapped._tp_plan, model_tp_mesh(unwrapped), TP_SIZE)
    return {name: tensor.detach().float().cpu() for name, tensor in full.items()}


def run(ctx):
    log(f"\n{'#' * 70}")
    log("  Tensor Parallelism Correctness Test")
    log(f"  World size: {ctx.world_size}, TP size: {TP_SIZE}")
    log(f"  Model: {MODEL_NAME}")
    log(f"  Seq len: {SEQ_LEN}, Seed: {SEED}")
    log(f"  GPU: {torch.cuda.get_device_name(ctx.local_rank)}")
    log(f"{'#' * 70}")

    # 2 GPUs: pure TP, every planned parameter a `tp` DTensor. 4: TP+DP, where FSDP2 makes the TP replicas 1D `dp`
    # DTensors, so the mesh alone no longer separates them. An identical batch on every rank keeps the baseline norm
    # valid.
    if ctx.world_size % TP_SIZE != 0:
        raise ValueError(f"world size {ctx.world_size} must be a multiple of TP size {TP_SIZE}")
    log(f"  Topology: tp={TP_SIZE}, dp={ctx.world_size // TP_SIZE}")

    log("\nEnsuring model is downloaded...")
    ensure_model_downloaded(MODEL_NAME, ctx.rank)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    log(f"\n{'=' * 70}")
    log("PHASE 1: Single-GPU Baseline (No TP)")
    log(f"{'=' * 70}")

    baseline_loss, baseline_grad_norm, baseline_grads = compute_baseline_loss(tokenizer, str(ctx.device))

    barrier()
    cleanup_memory()

    log(f"\n{'=' * 70}")
    log("PHASE 2: TP=2 Forward Pass + Grad Norm")
    log(f"{'=' * 70}")

    (tp_loss_avg, tp_spread), (tp_gn_avg, gn_spread), tp_grads, tp_synced = compute_tp_loss_and_grad_norm(
        tokenizer, ctx.local_rank, ctx.output_dir
    )

    barrier()

    checks: dict[str, bool] = {}
    metrics: dict[str, float] = {}
    checks["tp_synced_grads_bit_identical"] = tp_synced[0]
    if ctx.rank == 0:
        log(f"  TP-synced grad bit-identity over {tp_synced[1]} params: {'PASS' if tp_synced[0] else 'FAIL'}")
        log(f"\n{'=' * 70}")
        log("VALIDATION")
        log(f"{'=' * 70}")

        # world_spread is inf when any rank's value is non-finite, so a finite spread covers every rank.
        tp_finite = math.isfinite(tp_spread)
        checks["tp_losses_finite"] = tp_finite
        log(f"\n  TP losses finite (all ranks): {'PASS' if tp_finite else 'FAIL'}")

        tp_consistent = tp_spread < TOL.all_reduced_rank_spread_abs
        checks["tp_rank_consistency"] = tp_consistent
        log(f"  TP rank consistency (spread={tp_spread:.8f}): {'PASS' if tp_consistent else 'FAIL'}")

        baseline_finite = math.isfinite(baseline_loss)
        checks["baseline_finite"] = baseline_finite
        log(f"  Baseline loss finite ({baseline_loss:.6f}): {'PASS' if baseline_finite else 'FAIL'}")

        abs_diff = abs(tp_loss_avg - baseline_loss)
        rel_diff = abs_diff / max(abs(baseline_loss), 1e-10)

        loss_match = abs_diff < LOSS_ABS_TOL and rel_diff < LOSS_REL_TOL
        checks["loss_match"] = loss_match
        log("\n  --- TP vs Baseline Loss Comparison ---")
        log(f"  Baseline loss: {baseline_loss:.6f}")
        log(f"  TP loss (avg): {tp_loss_avg:.6f}")
        log(f"  Abs diff:      {abs_diff:.6f} (tol: {LOSS_ABS_TOL})")
        log(f"  Rel diff:      {rel_diff:.4%} (tol: {LOSS_REL_TOL:.4%})")
        log(f"  Match (abs AND rel): {'PASS' if loss_match else 'FAIL'}")

        gn_finite = math.isfinite(gn_spread)
        checks["tp_grad_norm_finite"] = gn_finite
        log(f"\n  TP grad norms finite (all ranks): {'PASS' if gn_finite else 'FAIL'}")

        gn_consistent = gn_spread < TOL.all_reduced_rank_spread_abs
        checks["tp_grad_norm_consistency"] = gn_consistent
        log(f"  TP grad norm consistency (spread={gn_spread:.8f}): {'PASS' if gn_consistent else 'FAIL'}")

        # Equality against the unsharded norm: HF-native TP shards projections into plain slices, so a classifier
        # keying on tensor type reads them as replicas and averages. That lands at ratio ~0.5 with every rank in
        # agreement.
        gn_ratio = tp_gn_avg / max(abs(baseline_grad_norm), 1e-10)
        gn_matches = abs(gn_ratio - 1.0) <= GRAD_NORM_REL_TOL
        checks["grad_norm_matches_baseline"] = gn_matches
        log("\n  --- TP vs Baseline Grad Norm Comparison ---")
        log(f"  Baseline grad norm: {baseline_grad_norm:.6f}")
        log(f"  TP grad norm (avg): {tp_gn_avg:.6f}")
        log(f"  Ratio (TP/baseline): {gn_ratio:.4f} (tol: {GRAD_NORM_REL_TOL:.1%})")
        log(f"  Match: {'PASS' if gn_matches else 'FAIL'}")

        # The big projections dominate the global norm, so mis-reducing `q_norm`/`k_norm` hides inside its tolerance.
        worst_rel_name, worst_rel = "", 0.0
        worst_ratio_name, worst_ratio_dev = "", 0.0
        per_param_lines = []
        for name in COMPARED_PARAMS:
            ref, got = baseline_grads[name], tp_grads[name]
            assert ref.shape == got.shape, f"{name}: reassembled {tuple(got.shape)} != reference {tuple(ref.shape)}"
            ref_norm = max(ref.norm().item(), 1e-12)
            rel = (got - ref).norm().item() / ref_norm
            ratio = got.norm().item() / ref_norm
            per_param_lines.append(f"    {name}: rel_err={rel:.2e} norm_ratio={ratio:.4f}")
            # A NaN compares false against the trackers below.
            rel = rel if math.isfinite(rel) else math.inf
            ratio_dev = abs(ratio - 1.0) if math.isfinite(ratio) else math.inf
            if rel > worst_rel:
                worst_rel_name, worst_rel = name, rel
            if ratio_dev > worst_ratio_dev:
                worst_ratio_name, worst_ratio_dev = name, ratio_dev
        grads_match = worst_rel <= GRAD_TENSOR_REL_TOL and worst_ratio_dev <= GRAD_TENSOR_RATIO_TOL
        checks["per_param_grads_match_baseline"] = grads_match
        log("\n  --- Per-parameter gradient comparison (reassembled via the TP plan) ---")
        for line in per_param_lines:
            log(line)
        log(f"  Worst rel_err:   {worst_rel_name} {worst_rel:.2e} (tol: {GRAD_TENSOR_REL_TOL:.0%})")
        log(f"  Worst |ratio-1|: {worst_ratio_name} {worst_ratio_dev:.2e} (tol: {GRAD_TENSOR_RATIO_TOL:.0%})")
        log(f"  Match: {'PASS' if grads_match else 'FAIL'}")

        metrics = {
            "baseline_loss": baseline_loss,
            "tp_loss_avg": tp_loss_avg,
            "baseline_grad_norm": baseline_grad_norm,
            "tp_grad_norm_avg": tp_gn_avg,
            "grad_norm_ratio": gn_ratio,
            "worst_grad_rel_err": worst_rel,
        }

    # Only rank 0 holds the baseline, so its verdict is what every rank reports.
    return {"checks": ctx.broadcast_checks(checks), "metrics": metrics}


main = gpu_test_main(min_world_size=TP_SIZE, prefix="test_tp_correctness")(run)

if __name__ == "__main__":
    main()
