#!/usr/bin/env python
"""Per-head attention norms receive the sum of their gradient across the TP group, once, on the toolkit's MoE plan.

Each rank accumulates only its heads' norm gradient and nothing in the TP graph sums them. With no reduction they train
on 1/tp_size of it; a per-backward hook re-multiplies earlier micro-batches by ``tp_size``, so two micro-batches
accumulate before the reduce.
"""

from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM

from halo.distributed.parallelism_config import ParallelismConfig
from halo.distributed.tensor_parallel.plain_params import per_head_norm_param_names
from halo.model_loading.parallel.loader import load_distributed_model
from halo.trainers.distributed_trainer import DistributedTrainerMixin
from tests.common.distributed import shared_scratch_dir, world_spread
from tests.common.harness import gpu_test_main
from tests.common.memory import cleanup_memory
from tests.common.numerics import max_or_nan
from tests.common.rank_log import log, log_all

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

# Every failure is a clean factor of tp_size and the bf16 reduce-order noise is ~1e-1.
NORM_GRAD_TOL = 0.35

# FA4's CuTe backward fails IR verification at this model's tiny head_dim; the reduction is backend independent.
ATTN_IMPL = "sdpa"


class _StepSync:
    """The real methods off ``DistributedTrainerMixin``, supplied only the model and its TP mesh."""

    def __init__(self, model, parallelism_config):
        self._model = model
        self._device_mesh = model._device_mesh
        self.parallelism_config = parallelism_config
        self._param_id_caches = {}
        self.state = SimpleNamespace(global_step=0)

    def _top_level_model(self):
        return self._model

    _memoized_param_ids = DistributedTrainerMixin._memoized_param_ids
    _tp_sharded_plain_param_ids = DistributedTrainerMixin._tp_sharded_plain_param_ids
    _tp_per_head_norm_param_ids = DistributedTrainerMixin._tp_per_head_norm_param_ids
    _sync_tp_replicated_grads = DistributedTrainerMixin._sync_tp_replicated_grads


def _norm_param_names(model) -> list[str]:
    return [n for n, _ in model.named_parameters() if n.endswith(("q_norm.weight", "k_norm.weight"))]


def _accumulate(model, batches) -> None:
    """Backward every micro-batch without zeroing, as ``gradient_accumulation_steps`` does."""
    for ids, labels in batches:
        model(input_ids=ids, labels=labels, use_cache=False).loss.backward()


def _reference_grads(ckpt_dir: str, batches, device: str) -> dict:
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
    checks["found_per_head_norms"] = len(names) > 0  # Anti-vacuity: a model without them proves nothing.
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
        trust_remote_code=True,
    )
    model.train()

    registered = set(per_head_norm_param_names(model))
    # The reduction is driven off this registry, so an empty one would make the comparison vacuous.
    checks["norms_registered_for_step_sync"] = registered.issuperset(names)

    _accumulate(model, batches)

    # Exactly one reduction, after all micro-batches, where the trainer runs it.
    _StepSync(model, pc)._sync_tp_replicated_grads(list(model.parameters()))

    live = dict(model.named_parameters())
    rels = []
    for name in names:
        grad = live[name].grad
        local = grad.to_local() if hasattr(grad, "to_local") else grad
        rel = (local.detach().float() - ref[name]).norm().item() / max(ref[name].norm().item(), 1e-12)
        rels.append(rel)
        log_all(f"  {name}: rel_err={rel:.4e}")
    worst = max_or_nan(rels, default=0.0)
    metrics["worst_norm_grad_rel_err"] = worst
    checks["norm_grads_match_reference"] = worst < NORM_GRAD_TOL

    # The whole failure mode at dp>1 is the ranks disagreeing.
    checks["ranks_agree"] = world_spread(worst) < 1e-6

    return {"checks": checks, "metrics": metrics}


@gpu_test_main(prefix="tp_norm_grad", min_world_size=2)
def _run(ctx):
    return run(ctx)


if __name__ == "__main__":
    _run()
