#!/usr/bin/env python
"""PP equivalence with real padding masks, sliding-window layers, and per-impl attention paths.

The base equivalence gate (test_pp_vs_single_gpu.py) passes no attention mask, so transformers'
whole mask machinery — 4D mask construction from ``inputs_embeds`` on mid stages, sliding vs full
selection by enumerate position, FA varlen unpadding — is dead code there. Production PP always has
mask zeros (the fixed-shape collator pads with 0), so this test drives ragged right-padded batches
through a tiny Qwen3 with an alternating sliding/full ``layer_types`` pattern, or a tiny Qwen3.5
hybrid, and compares loss + every gradient against the unsplit model under the same attention
implementation.

Modes (manifest ``args_matrix``):
    (none)                      eager, fp32 — exact gate
    --attn sdpa                 SDPA, fp32 — exact gate (mask content decides the all-ones skip)
    --positions                 eager, fp32, explicit position_ids — proves kwarg chunking stays
                                row-aligned with the batch split
    --attn flash_attention_4    FA4, bf16 — varlen unpad path, loose gate; skips off Blackwell
    --family qwen3_5            Qwen3.5 hybrid GatedDeltaNet (period-4 layer_types), eager, fp32 —
                                exact gate, the recurrent decay params under their own bound

Run: torchrun --nproc_per_node=4 tests/gpu/parallelism/pp/test_pp_ragged_mask_vs_single_gpu.py
"""

import math
import sys

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard
from transformers import Qwen3Config, Qwen3ForCausalLM
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

from src.distributed.parallelism_config import ParallelismConfig
from src.distributed.pipeline_parallel.groups import build_pipeline_group
from src.distributed.pipeline_parallel.runner import PipelineRunner
from src.distributed.pipeline_parallel.stage import build_pipeline_stage
from src.distributed.pipeline_parallel.stage_losses import TokenLoss
from tests.common.harness import gpu_test_main, skip_unless_blackwell
from tests.common.models import TINY_QWEN3_CONFIG, TINY_QWEN35_CONFIG
from tests.common.utils import log, log_all

GLOBAL_BATCH = 8
SEQ_LEN = 32

ATTN_IMPL = "eager"
EXPLICIT_POSITIONS = False
# qwen3: sliding/full alternation through Qwen3ForCausalLM. qwen3_5: the hybrid GatedDeltaNet
# family — period-4 L,L,L,F layer_types, partial interleaved mrope, linear-attn conv state — the
# strongest "does the generic split really inherit a family's own machinery" probe.
FAMILY = "qwen3"

# Sliding/full alternation makes the enumerate-position mask selection load-bearing: a rebase bug
# hands sliding layers a full mask (or vice versa) and the grads diverge. head_dim 64 so the FA4
# mode runs a supported kernel shape.
SLIDING_OVERRIDES = {
    "head_dim": 64,
    "use_sliding_window": True,
    "sliding_window": 8,
    "layer_types": ["sliding_attention", "full_attention"] * (TINY_QWEN3_CONFIG["num_hidden_layers"] // 2),
}


def _dtype() -> torch.dtype:
    return torch.bfloat16 if ATTN_IMPL == "flash_attention_4" else torch.float32


def _tolerances() -> tuple[float, float]:
    """(loss_rel, grad_rel): exact in fp32; bf16 kernels get the EP-test-style loose gate."""
    return (1e-2, 3e-2) if _dtype() == torch.bfloat16 else (1e-4, 2e-3)


# The recurrent decay parameter of a linear-attention family (Qwen3.5 GatedDeltaNet ``A_log``) is
# ill-conditioned by construction: its gradient flows through an exponentiated cumulative product over
# the whole sequence scan, so reordering that accumulation — which is exactly what splitting a batch
# into microbatches does — moves it far more than any ordinary weight. Measured over repeated runs:
# every other parameter in the stage lands at ~1e-4 while A_log on the deepest layers sits at
# 1.7e-3..3.4e-3, the loss staying exact to 6.8e-8. Hence its own bound, not a blanket loosening.
DECAY_GRAD_TOL = 1e-2

_DECAY_PARAM_SUFFIXES = ("A_log", "dt_bias")


def _is_recurrent_decay_param(fqn: str) -> bool:
    return fqn.endswith(_DECAY_PARAM_SUFFIXES)


def build_model(device):
    torch.manual_seed(1234)
    if FAMILY == "qwen3_5":
        config = Qwen3_5TextConfig(**TINY_QWEN35_CONFIG, attn_implementation=ATTN_IMPL)
        return Qwen3_5ForCausalLM(config).to(device=device, dtype=_dtype())
    config = Qwen3Config(**{**TINY_QWEN3_CONFIG, **SLIDING_OVERRIDES}, attn_implementation=ATTN_IMPL)
    return Qwen3ForCausalLM(config).to(device=device, dtype=_dtype())


def make_batch(device, vocab_size):
    """Ragged true lengths right-padded to SEQ_LEN: mask zeros + -100 labels on every pad."""
    g = torch.Generator(device="cpu").manual_seed(99)
    input_ids = torch.randint(4, vocab_size, (GLOBAL_BATCH, SEQ_LEN), generator=g)
    attention_mask = torch.zeros(GLOBAL_BATCH, SEQ_LEN, dtype=torch.long)
    for i in range(GLOBAL_BATCH):
        true_len = 10 + 2 * i  # 10..24 — every row shorter than SEQ_LEN, all lengths distinct
        attention_mask[i, :true_len] = 1
        input_ids[i, true_len:] = 0
    labels = input_ids.masked_fill(attention_mask == 0, -100)
    position_ids = torch.arange(SEQ_LEN).unsqueeze(0).expand(GLOBAL_BATCH, -1).contiguous()
    return (t.to(device) for t in (input_ids, labels, attention_mask, position_ids))


def summed_ce(logits, labels):
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)).float(),
        shift_labels.view(-1),
        ignore_index=-100,
        reduction="sum",
    )


@gpu_test_main(min_world_size=4, prefix="pp_ragged_mask")
def run(ctx):
    device = torch.device("cuda", ctx.local_rank)
    pp_size = ctx.world_size // 2
    n_microbatches = max(2, pp_size)
    config = ParallelismConfig(world_size=ctx.world_size, gpus_per_node=2, nvlink_domain_size=2, pp_size=pp_size)

    input_ids, labels, attention_mask, position_ids = make_batch(device, TINY_QWEN3_CONFIG["vocab_size"])
    global_count = (labels[:, 1:] != -100).sum().float()

    # Baseline: unsplit model, full batch, same attention implementation and dtype.
    model = build_model(device)
    kwargs = {"position_ids": position_ids} if EXPLICIT_POSITIONS else {}
    ref_loss = summed_ce(model(input_ids=input_ids, attention_mask=attention_mask, **kwargs).logits, labels)
    ref_loss = ref_loss / global_count
    ref_loss.backward()
    ref_grads = {n: p.grad.detach().float().clone() for n, p in model.named_parameters() if p.grad is not None}
    del model
    torch.cuda.empty_cache()
    log(f"[baseline:{ATTN_IMPL}] loss={ref_loss.item():.6f}")

    mesh = init_device_mesh("cuda", (pp_size, config.stage_world_size), mesh_dim_names=("pp", "dp"))
    dp_mesh = mesh["dp"]
    pp_group = build_pipeline_group(config)

    stage_module = build_pipeline_stage(build_model(device), config.pp_rank, pp_size)
    for layer in stage_module.model.layers:
        fully_shard(layer, mesh=dp_mesh, reshard_after_forward=False)
    fully_shard(stage_module, mesh=dp_mesh, reshard_after_forward=False)
    runner = PipelineRunner(stage_module, config, pp_group, device, n_microbatches, TokenLoss())

    # Ragged rows make DP replicas hold unequal counts — the exact case the rank-local default
    # normalizer gets wrong. Pass the trainer's G/dp so per-replica losses average to S_total/G.
    dp_rank = config.get_data_parallel_rank()
    local_bs = GLOBAL_BATCH // config.data_parallel_size
    sl = slice(dp_rank * local_bs, (dp_rank + 1) * local_bs)
    loss = runner.step(
        input_ids[sl],
        labels[sl],
        attention_mask=attention_mask[sl],
        position_ids=position_ids[sl] if EXPLICIT_POSITIONS else None,
        num_items_in_batch=global_count / config.data_parallel_size,
    )

    loss_tol, grad_tol = _tolerances()
    checks, metrics = {}, {}
    loss_err_t = torch.zeros(1, device=device)
    if config.is_last_pp_stage:
        mean_loss = loss.detach().float().clone()
        dist.all_reduce(mean_loss, op=dist.ReduceOp.AVG, group=dp_mesh.get_group())
        loss_err_t[0] = abs(mean_loss.item() - ref_loss.item()) / abs(ref_loss.item())
        log(f"[pp:{ATTN_IMPL}] loss={mean_loss.item():.6f} baseline={ref_loss.item():.6f} rel={loss_err_t.item():.3e}")
    dist.all_reduce(loss_err_t, op=dist.ReduceOp.MAX)
    checks["loss_matches_baseline"] = loss_err_t.item() < loss_tol
    metrics["loss_rel_err"] = loss_err_t.item()

    worst_name, worst_err, n_checked, missing = None, 0.0, 0, []
    decay_name, decay_err = None, 0.0
    for name, param in stage_module.named_parameters():
        fqn = stage_module.global_parameter_name(name)
        if fqn not in ref_grads or param.grad is None:
            missing.append(fqn)
            continue
        grad = param.grad
        grad = grad.full_tensor() if isinstance(grad, torch.distributed.tensor.DTensor) else grad
        ref = ref_grads[fqn]
        err = ((grad.float() - ref).abs().max() / ref.abs().max().clamp_min(1e-12)).item()
        n_checked += 1
        # A NaN compares false against every bound, so a non-finite error counts as infinite.
        err = err if math.isfinite(err) else math.inf
        if _is_recurrent_decay_param(fqn):
            if err > decay_err:
                decay_name, decay_err = fqn, err
        elif err > worst_err:
            worst_name, worst_err = fqn, err
    # log_all, not log: the per-stage worst offender is exactly what a failure needs, and only
    # rank 0 reporting it makes another stage's mismatch undiagnosable.
    log_all(
        f"[rank {ctx.rank} stage {config.pp_rank}] checked={n_checked} worst={worst_err:.3e} @ {worst_name}"
        + (f" | decay {decay_err:.3e} @ {decay_name}" if decay_name else "")
    )

    expected = len(list(stage_module.named_parameters()))
    checks["all_stage_params_compared"] = n_checked == expected and not missing
    checks["grads_match_baseline"] = n_checked > 0 and worst_err < grad_tol
    checks["decay_grads_match_baseline"] = decay_err < DECAY_GRAD_TOL
    metrics["worst_grad_rel_err"] = worst_err
    metrics["worst_decay_grad_rel_err"] = decay_err

    # The mask must have actually carried zeros into the stage — otherwise this run silently
    # degenerated to the unmasked gate.
    checks["mask_has_padding"] = bool((attention_mask[sl] == 0).any().item())

    stage_param_count = torch.tensor([float(n_checked)], device=device)
    dist.all_reduce(stage_param_count, op=dist.ReduceOp.MIN)
    checks["every_stage_has_params"] = stage_param_count.item() > 0
    return {"checks": checks, "metrics": metrics}


if __name__ == "__main__":
    if "--positions" in sys.argv:
        EXPLICIT_POSITIONS = True
        sys.argv.remove("--positions")
    if "--attn" in sys.argv:
        i = sys.argv.index("--attn")
        ATTN_IMPL = sys.argv[i + 1]
        del sys.argv[i : i + 2]
    if ATTN_IMPL == "flash_attention_4":
        skip_unless_blackwell("the --attn flash_attention_4 row")
    if "--family" in sys.argv:
        i = sys.argv.index("--family")
        FAMILY = sys.argv[i + 1]
        del sys.argv[i : i + 2]
    sys.exit(run())
