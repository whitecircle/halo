#!/usr/bin/env python
"""
Tensor Parallelism (TP) correctness test on a dense model.

Validates that TP=2 produces the same forward pass loss AND the same total
gradient L2 norm as a single-GPU baseline on Qwen3-0.6B (dense model). This
tests DTensor weight sharding correctness for attention and MLP layers, and
guards the TP grad-norm aggregation path: a TP grad-norm that silently drops
the lm_head/embedding contribution or mis-aggregates across the TP axis (the
"grad-norm spread 31→0" class) must fail the grad-norm check.

Rank 0 records the baseline loss and total fp32 grad L2 norm with no TP; all ranks
then run the SAME deterministic input through a TP=2 DistributedSFTTrainer, and both
the loss (tight conjunctive tolerance) and the TP-aware grad norm
(trainer._compute_tp_grad_norm, the production clipping path) are compared against
that baseline within a tight relative tolerance.

Run with 2 GPUs:
    torchrun --nproc_per_node=2 \
        tests/gpu/parallelism/tp/test_tp_correctness.py

Requirements:
    - 2x GPUs
    - Model: Qwen/Qwen3-0.6B (auto-downloaded)
"""

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.distributed.tensor_parallel import gather_state_dict_for_save
from trl import SFTConfig

from src.distributed.loading.model_loading import load_distributed_model
from src.distributed.mesh import MeshDim, mesh_dim_names
from src.distributed.parallelism_config import ParallelismConfig
from src.distributed.runtime import barrier, materialize_dtensor
from src.distributed.tensor_parallel.state_dict import get_tp_mesh
from src.trainers.sft import DistributedSFTTrainer
from tests.common.datasets import create_sft_dataset
from tests.common.distributed import ensure_model_downloaded
from tests.common.ep_reference import fixed_chat_batch
from tests.common.harness import gpu_test_main
from tests.common.models import QWEN3_0_6B
from tests.common.utils import cleanup_memory, gpu_mem_gb, log, log_all

MODEL_NAME = QWEN3_0_6B
TP_SIZE = 2
SEQ_LEN = 128
SEED = 42

# Loss match is CONJUNCTIVE (both bounds) and tight — TP is a math-exact rearrangement of the same
# compute, so only bf16 reduction-order jitter remains. Same for the grad norm: same batch/seed.
LOSS_ABS_TOL = 0.02
LOSS_REL_TOL = 0.01
GRAD_NORM_REL_TOL = 0.03

# One representative parameter per TP *style*, because each is reduced by a different rule and a
# wrong rule is invisible in the global norm when the tensor is small: colwise/rowwise are disjoint
# plain slices reassembled by the plan; `*_norm` weights are `replicated_with_grad_allreduce`
# (replicated storage, HEAD-PARTIAL gradient, needs a TP SUM); layernorm/embed are truly replicated.
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
# Loose next to the 3% aggregate because a single bf16 gradient tensor carries far more reassociation
# noise than the whole-model sum (5-9% here, uniform across sharded and replicated tensors alike).
# Still decisive: every failure mode is a whole FACTOR — a missing TP sum lands at ratio 0.5,
# averaging two disjoint slices at rel_err ~1.0.
GRAD_TENSOR_REL_TOL = 0.20
GRAD_TENSOR_RATIO_TOL = 0.20


def compute_baseline_loss(tokenizer, device):
    """Compute forward loss AND total fp32 grad L2 norm on a single GPU (no TP).

    Only rank 0 loads and runs the model. It runs one forward (loss) and one
    forward+backward, then sums every parameter's grad L2 norm in fp32 to get the
    reference total grad norm. Both scalars are broadcast to all ranks so every
    rank can participate in the comparison.

    The grad norm is the unsharded ground truth: the TP phase must reproduce it
    (a TP grad-norm that drops lm_head/embedding or mis-aggregates the TP axis
    diverges from this).

    It also keeps the unsharded gradient of every parameter in ``COMPARED_PARAMS`` (rank 0 only) so
    Phase 3 can compare them one by one. The global norm alone cannot: a `q_norm` weight is a
    head_dim-sized vector, so even a `tp_size`x error in it moves the total norm by far less than
    the bf16 tolerance.

    Returns:
        tuple[float, float, dict]: (baseline_loss, baseline_grad_norm, per-param grads on rank 0).
    """
    rank = dist.get_rank()
    baseline_grads: dict[str, torch.Tensor] = {}
    result_tensor = torch.zeros(2, device=device)  # [loss, grad_norm]

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
    """Compute forward loss AND TP-aware total grad norm with TP=2.

    All ranks build a ``DistributedSFTTrainer`` with ``tp_size=2`` — this is what
    sets ``trainer._device_mesh`` and installs the TP grad-norm machinery
    (``_sync_tp_replicated_grads`` + ``_compute_tp_grad_norm``, the exact path
    ``tp_clip_grad_norm_`` calls in production). We then run, on the SAME fixed
    batch used by the baseline:

      * a forward (no_grad) for the loss, and
      * a forward+backward for the gradients, after which we call the trainer's
        ``_sync_tp_replicated_grads`` then ``_compute_tp_grad_norm`` — i.e. the
        production TP grad-norm, which must reproduce the unsharded baseline.

    Returns:
        tuple[list[float], list[float]]: (per-rank loss, per-rank TP grad norm).
    """
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

    # Decisive at the source: a severed tie is two Parameter objects here, long before it surfaces
    # downstream as a 0.63 rel_err on the reassembled embedding gradient.
    if model.config.tie_word_embeddings:
        assert model.get_output_embeddings().weight is model.get_input_embeddings().weight, (
            "tie_word_embeddings=True but embed_tokens and lm_head are two independent parameters"
        )

    # The trainer supplies the TP-aware grad-norm path (device mesh + _sync_tp_replicated_grads /
    # _compute_tp_grad_norm) and the production FSDP2-for-TP wrapping; forward/backward is manual.
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
        max_length=SEQ_LEN,
        dataloader_drop_last=True,
        dataloader_num_workers=0,
        max_grad_norm=1.0,
        fsdp="",
    )
    trainer = DistributedSFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        parallelism_config=parallelism_config,
    )
    assert trainer.is_tp_mode, "expected TP mode"

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
    # Mirrors tp_clip_grad_norm_ — sync replicated grads across the TP axis, then take the global norm.
    trainer._sync_tp_replicated_grads(params)
    tp_synced_identical, tp_synced_count = assert_tp_synced_grads_identical(trainer, tp_model)
    # float(), not the tensor: all_gather_object unpickles onto each rank's OWN device, so a gathered
    # list would mix cuda:0 and cuda:1 and every comparison below would raise.
    tp_grad_norm_local = float(trainer._compute_tp_grad_norm(params))
    log_all(f"  TP grad norm: {tp_grad_norm_local:.6f}")

    all_tp_losses = [None] * dist.get_world_size()
    all_tp_grad_norms = [None] * dist.get_world_size()
    dist.all_gather_object(all_tp_losses, tp_loss_local)
    dist.all_gather_object(all_tp_grad_norms, tp_grad_norm_local)

    tp_grads = reassemble_tp_grads(tp_model)

    tp_model.zero_grad(set_to_none=True)
    del model, trainer, tp_model, loss
    cleanup_memory()

    return all_tp_losses, all_tp_grad_norms, tp_grads, (tp_synced_identical, tp_synced_count)


def assert_tp_synced_grads_identical(trainer, tp_model) -> tuple[bool, int]:
    """After ``_sync_tp_replicated_grads``, every param WITHOUT a ``tp`` mesh dim must carry a
    bit-identical gradient across the TP group: replicas are AVGed, per-head norms SUMmed, and both
    collectives hand every rank the same result. Selection is structural (mesh dims, not the
    trainer's own bucket sets), so a sweep that silently skips a class of params fails here.
    Collective on every rank — the identical fixed batch keeps the walk order and grad-presence
    pattern rank-uniform."""
    unwrapped = tp_model.module if hasattr(tp_model, "module") else tp_model
    tp_group = trainer._get_tp_process_group()
    tp_world = dist.get_world_size(group=tp_group)
    checked, max_diff = 0, 0.0
    for _name, p in unwrapped.named_parameters():
        if p.grad is None:
            continue
        if isinstance(p.data, DTensor) and MeshDim.TP in mesh_dim_names(p.data.device_mesh):
            continue  # plan-sharded: ranks legitimately hold different slices
        grad = p.grad.to_local() if isinstance(p.grad, DTensor) else p.grad
        gathered = [torch.empty_like(grad) for _ in range(tp_world)]
        dist.all_gather(gathered, grad.contiguous(), group=tp_group)
        checked += 1
        max_diff = max(max_diff, max((g - gathered[0]).abs().max().item() for g in gathered))
    log_all(f"  TP-synced grads: {checked} params checked, max cross-TP diff {max_diff:.3e}")
    return max_diff == 0.0 and checked > 0, checked


def reassemble_tp_grads(tp_model) -> dict[str, torch.Tensor]:
    """Rebuild the UNSHARDED gradient of each ``COMPARED_PARAMS`` entry. Collective on every rank.

    Uses the production reassembly seam — HF's own plan-keyed ``gather_state_dict_for_save``, the
    same call the TP checkpoint save makes — so a plan the save would mis-gather is mis-gathered
    here too rather than papered over by a hand-rolled all-gather. Under TP+DP the grads are
    FSDP2 DTensors first, so materialize each over its dp mesh before the plan gather.
    """
    unwrapped = tp_model.module if hasattr(tp_model, "module") else tp_model
    named = dict(unwrapped.named_parameters())
    local = {}
    for name in COMPARED_PARAMS:
        param = named.get(name)
        assert param is not None and param.grad is not None, f"no live gradient for {name} under TP"
        local[name] = materialize_dtensor(param.grad)  # collective under TP+DP; identity at dp=1
    full = gather_state_dict_for_save(local, unwrapped._tp_plan, get_tp_mesh(unwrapped), TP_SIZE)
    return {name: tensor.detach().float().cpu() for name, tensor in full.items()}


def run(ctx):
    log(f"\n{'#' * 70}")
    log("  Tensor Parallelism Correctness Test")
    log(f"  World size: {ctx.world_size}, TP size: {TP_SIZE}")
    log(f"  Model: {MODEL_NAME}")
    log(f"  Seq len: {SEQ_LEN}, Seed: {SEED}")
    log(f"  GPU: {torch.cuda.get_device_name(ctx.local_rank)}")
    log(f"{'#' * 70}")

    # 2 GPUs → pure TP, where transformers distributes every planned parameter as a `tp` DTensor;
    # 4 → TP+DP, where FSDP2 additionally makes the TP replicas 1D `dp` DTensors, so the mesh alone
    # no longer separates the two. Identical batch on every rank, so the DP average reproduces the
    # single-batch gradient and the same baseline norm.
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

    tp_losses, tp_grad_norms, tp_grads, tp_synced = compute_tp_loss_and_grad_norm(
        tokenizer, ctx.local_rank, ctx.output_dir
    )

    barrier()

    checks: dict[str, bool] = {}
    metrics: dict[str, float] = {}
    # Bit-identity of TP-synced grads is asserted on every rank (each gathered its own view).
    checks["tp_synced_grads_bit_identical"] = tp_synced[0]
    if ctx.rank == 0:
        log(f"  TP-synced grad bit-identity over {tp_synced[1]} params: {'PASS' if tp_synced[0] else 'FAIL'}")
        log(f"\n{'=' * 70}")
        log("VALIDATION")
        log(f"{'=' * 70}")

        tp_finite = all(not (torch.isnan(torch.tensor(l)) or torch.isinf(torch.tensor(l))) for l in tp_losses)
        checks["tp_losses_finite"] = tp_finite
        log(f"\n  TP losses finite: {'PASS' if tp_finite else 'FAIL'}")
        log(f"    TP losses per rank: {[f'{l:.6f}' for l in tp_losses]}")

        tp_spread = max(tp_losses) - min(tp_losses)
        tp_consistent = tp_spread < 1e-4
        checks["tp_rank_consistency"] = tp_consistent
        log(f"  TP rank consistency (spread={tp_spread:.8f}): {'PASS' if tp_consistent else 'FAIL'}")

        baseline_finite = not (torch.isnan(torch.tensor(baseline_loss)) or torch.isinf(torch.tensor(baseline_loss)))
        checks["baseline_finite"] = baseline_finite
        log(f"  Baseline loss finite ({baseline_loss:.6f}): {'PASS' if baseline_finite else 'FAIL'}")

        tp_loss_avg = sum(tp_losses) / len(tp_losses)
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

        loss_reasonable = 0 < tp_loss_avg < 100
        checks["loss_reasonable"] = loss_reasonable
        log(f"  Loss in reasonable range (0 < {tp_loss_avg:.4f} < 100): {'PASS' if loss_reasonable else 'FAIL'}")

        gn_finite = all(not (torch.isnan(torch.tensor(g)) or torch.isinf(torch.tensor(g))) for g in tp_grad_norms)
        checks["tp_grad_norm_finite"] = gn_finite
        log(f"\n  TP grad norms finite: {'PASS' if gn_finite else 'FAIL'}")
        log(f"    TP grad norms per rank: {[f'{g:.6f}' for g in tp_grad_norms]}")

        # The TP-aware norm is a global all-reduced scalar, so every TP rank must agree.
        gn_spread = max(tp_grad_norms) - min(tp_grad_norms)
        gn_consistent = gn_spread < 1e-4
        checks["tp_grad_norm_consistency"] = gn_consistent
        log(f"  TP grad norm consistency (spread={gn_spread:.8f}): {'PASS' if gn_consistent else 'FAIL'}")

        # Equality against the UNSHARDED norm, not a band: HF-native TP shards planned projections
        # into PLAIN slices, so a classifier keying on tensor type reads them as TP replicas and
        # averages across the group. That lands at ratio ~0.5 while keeping every rank in
        # agreement — invisible to the cross-rank check above, inside any order-of-magnitude band.
        tp_gn_avg = sum(tp_grad_norms) / len(tp_grad_norms)
        gn_ratio = tp_gn_avg / max(abs(baseline_grad_norm), 1e-10)
        gn_matches = abs(gn_ratio - 1.0) <= GRAD_NORM_REL_TOL
        checks["grad_norm_matches_baseline"] = gn_matches
        log("\n  --- TP vs Baseline Grad Norm Comparison ---")
        log(f"  Baseline grad norm: {baseline_grad_norm:.6f}")
        log(f"  TP grad norm (avg): {tp_gn_avg:.6f}")
        log(f"  Ratio (TP/baseline): {gn_ratio:.4f} (tol: {GRAD_NORM_REL_TOL:.1%})")
        log(f"  Match: {'PASS' if gn_matches else 'FAIL'}")

        # The global norm is dominated by the big projections, so a small tensor reduced by the
        # wrong rule hides inside its tolerance — mis-reducing `q_norm`/`k_norm` moves the total
        # by well under 1%. Only a per-tensor comparison sees it; failures are whole factors.
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
            if rel > worst_rel:
                worst_rel_name, worst_rel = name, rel
            if abs(ratio - 1.0) > worst_ratio_dev:
                worst_ratio_name, worst_ratio_dev = name, abs(ratio - 1.0)
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
