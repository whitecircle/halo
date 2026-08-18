#!/usr/bin/env python
"""Per-head attention norms must receive the SUM of their gradient across the TP group — ONCE.

``q_norm``/``k_norm`` are replicated ``(head_dim,)`` parameters applied AFTER the colwise q/k
projection. ``ColwiseParallel`` defaults ``use_local_output=True``, so the projection returns a plain
tensor holding only this rank's heads and the DTensor graph ends there — each rank therefore
accumulates the norm's gradient over its own heads only, and the true gradient is the SUM over the
group. Nothing in the TP graph performs it.

Two failure modes, both silent, and this test separates them by accumulating TWO micro-batches
before the reduction (the shipped configs use ``gradient_accumulation_steps > 1``):

* **No reduction** — the norms train on 1/tp_size of their gradient (or, at dp>1, diverge across the
  group from step one).
* **Reduction per backward** — a ``register_full_backward_hook`` (transformers'
  ``ReplicatedWithGradAllReduce``) re-reduces whatever is already in ``.grad``, so each earlier
  micro-batch's contribution is multiplied by ``tp_size`` again on every later micro-step.

Both land far from the reference; a correct once-per-step SUM lands on it.

This test is on the TOOLKIT path (a MoE model), which builds its own plan — the existing
``test_tp_correctness.py`` uses dense Qwen3, where transformers supplies its own handling.

Run:
    torchrun --nproc_per_node=2 \
        tests/gpu/parallelism/tp/test_tp_attention_norm_grad.py
"""

from pathlib import Path

import torch
import torch.distributed as dist
from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM

from src.distributed.loading.model_loading import load_distributed_model
from src.distributed.mesh import get_tp_submesh
from src.distributed.parallelism_config import ParallelismConfig
from src.trainers.mixins.base import DistributedTrainerMixin
from tests.common.distributed import shared_scratch_dir
from tests.common.harness import gpu_test_main
from tests.common.utils import cleanup_memory, log, log_all

# num_attention_heads must stay divisible by tp_size (2 and 4 both run here).
TINY_CONFIG_KWARGS = {
    "vocab_size": 512,
    "hidden_size": 128,
    "intermediate_size": 256,
    "moe_intermediate_size": 64,
    "num_experts": 8,
    "num_experts_per_tok": 2,
    "num_hidden_layers": 2,
    "num_attention_heads": 8,
    "num_key_value_heads": 4,
    "head_dim": 16,
    "max_position_embeddings": 512,
    "tie_word_embeddings": False,
}

GRAD_ACCUM_MICRO_BATCHES = 2

# Every failure mode here is a clean factor of tp_size and the bf16 reduce-order noise is ~1e-1, so
# 0.35 separates them decisively.
NORM_GRAD_TOL = 0.35

# SDPA, not the auto-selected FA4: its CuTe backward fails IR verification at this model's tiny
# head_dim. The gradient reduction under test is attention-backend independent.
ATTN_IMPL = "sdpa"


class _StepSync:
    """Drives the PRODUCTION reduction (the trainer's step-time sync) against a bare model.

    The methods below are the real ones off ``DistributedTrainerMixin`` — a stand-in for the trainer
    only in what it supplies them (the model and the TP group), never in what they do.
    """

    def __init__(self, model, tp_group, parallelism_config):
        self._model = model
        self._tp_group = tp_group
        self.parallelism_config = parallelism_config

    def _top_level_model(self):
        return self._model

    def _get_tp_process_group(self):
        return self._tp_group

    _tp_sharded_plain_param_ids = DistributedTrainerMixin._tp_sharded_plain_param_ids
    _tp_per_head_norm_param_ids = DistributedTrainerMixin._tp_per_head_norm_param_ids
    _sync_tp_replicated_grads = DistributedTrainerMixin._sync_tp_replicated_grads


def _norm_param_names(model) -> list[str]:
    """Attention norms whose gradient is per-head-partial under colwise sharding."""
    return [n for n, _ in model.named_parameters() if n.endswith(("q_norm.weight", "k_norm.weight"))]


def _accumulate(model, batches) -> None:
    """Backward every micro-batch without zeroing — what ``gradient_accumulation_steps`` does."""
    for ids, labels in batches:
        model(input_ids=ids, labels=labels, use_cache=False).loss.backward()


def _reference_grads(ckpt_dir: str, batches, device: str) -> dict:
    """Single-GPU accumulation over the same micro-batches and weights (rank 0 only)."""
    model = Qwen3MoeForCausalLM.from_pretrained(ckpt_dir, dtype=torch.bfloat16, attn_implementation=ATTN_IMPL).to(
        device
    )
    model.train()
    _accumulate(model, batches)
    grads = {n: p.grad.detach().float().clone() for n, p in model.named_parameters() if n in _norm_param_names(model)}
    del model
    cleanup_memory()
    return grads


def run(ctx):
    device = f"cuda:{ctx.local_rank}"
    checks: dict[str, bool] = {}
    metrics: dict[str, float] = {}

    ckpt_dir = shared_scratch_dir("tp_norm_grad")
    if ctx.rank == 0:
        torch.manual_seed(0)
        Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
        Qwen3MoeForCausalLM(Qwen3MoeConfig(**TINY_CONFIG_KWARGS)).to(torch.bfloat16).save_pretrained(ckpt_dir)
    ctx.barrier()

    torch.manual_seed(1)
    batches = []
    for _ in range(GRAD_ACCUM_MICRO_BATCHES):
        ids = torch.randint(0, TINY_CONFIG_KWARGS["vocab_size"], (2, 64), device=device)
        dist.broadcast(ids, src=0)
        batches.append((ids, ids.clone()))

    ref: dict[str, torch.Tensor] = {}
    if ctx.rank == 0:
        ref = _reference_grads(ckpt_dir, batches, device)
        log(f"  reference norms: {len(ref)} params, first grad norm {next(iter(ref.values())).norm():.6e}")
    names = sorted(ref) if ctx.rank == 0 else []
    obj = [names]
    dist.broadcast_object_list(obj, src=0)
    names = obj[0]
    checks["found_per_head_norms"] = len(names) > 0  # anti-vacuity: a model without them proves nothing
    for name in names:
        buf = ref[name] if ctx.rank == 0 else torch.zeros(TINY_CONFIG_KWARGS["head_dim"], device=device)
        dist.broadcast(buf, src=0)
        ref[name] = buf

    pc = ParallelismConfig(tp_size=ctx.world_size, max_concurrent_loading=0)
    model, _ = load_distributed_model(
        model_name_or_path=ckpt_dir,
        parallelism_config=pc,
        dtype=torch.bfloat16,
        attn_implementation=ATTN_IMPL,
    )
    model.train()

    registered = set(getattr(model, "_tp_per_head_norm_params", None) or ())
    # The reduction is driven off this registry, so an empty one would make the comparison vacuous.
    checks["norms_registered_for_step_sync"] = registered.issuperset(names)

    _accumulate(model, batches)

    # Exactly one reduction, after all micro-batches — where the trainer runs it.
    tp_group = get_tp_submesh(model._device_mesh).get_group()
    _StepSync(model, tp_group, pc)._sync_tp_replicated_grads(list(model.parameters()))

    live = dict(model.named_parameters())
    worst = 0.0
    for name in names:
        grad = live[name].grad
        local = grad.to_local() if hasattr(grad, "to_local") else grad
        rel = (local.detach().float() - ref[name]).norm().item() / max(ref[name].norm().item(), 1e-12)
        worst = max(worst, rel)
        log_all(f"  {name}: rel_err={rel:.4e}")
    metrics["worst_norm_grad_rel_err"] = worst
    checks["norm_grads_match_reference"] = worst < NORM_GRAD_TOL

    # The whole failure mode at dp>1 is the ranks disagreeing, so pin agreement directly.
    spread = torch.tensor([worst], device=device)
    gathered = [torch.zeros_like(spread) for _ in range(ctx.world_size)]
    dist.all_gather(gathered, spread)
    checks["ranks_agree"] = max(abs(g.item() - gathered[0].item()) for g in gathered) < 1e-6

    return {"checks": checks, "metrics": metrics}


@gpu_test_main(prefix="tp_norm_grad", min_world_size=2)
def _run(ctx):
    return run(ctx)


if __name__ == "__main__":
    _run()
