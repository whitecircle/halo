#!/usr/bin/env python
"""CP's fused head loss on GPU: Liger's fused linear cross-entropy kernel, which the CPU tier replaces with a
plain-torch stand-in, against the fp32 logits loss the CP wrapper runs without ``fused_linear_cross_entropy``.

Three checks per launch, on each of the two ranks:

  * **Kernel.** :func:`fused_head_ce_sum` against ``transform.project`` and fp32 cross-entropy for every head
    transform field (hidden scale, logit scale, softcap, vocabulary cut): the summed loss, the argmax of every
    supervised row and the hidden and weight gradients.
  * **Memory.** At Qwen3's vocabulary the fused loss's forward-and-backward peak stays under a quarter of the
    logits loss's, which holds the bf16 logits, their fp32 copy and the saved log-softmax.
  * **Model.** The family's tiny model (``tests/common/equivalence/cp_equivalence.py``) under CP=2, once per path
    on one seeded weight set and one batch whose rank-0 chunk supervises its boundary target alone in one row: the
    rank-averaged loss and gradients, the router aux term, the chunk accuracy the fused forward returns in place
    of logits, and the head projecting one position on the fused path. Bailing's remote-code forward takes no
    ``logits_to_keep`` and must keep the logits loss.
  * **Wrapper memory.** Where the fused path runs, the same tiny model at Qwen3's vocabulary: the fused training step's peak stays under the
    logits step's and under the bf16 logits plane alone, which a forward building that plane would reach.

Run: torchrun --nproc_per_node=2 tests/gpu/parallelism/cp/test_cp_fused_head_loss.py --family qwen3
"""

import argparse
import inspect

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from halo.distributed.context_parallel.sequence_chunks import cp_local_targets
from halo.distributed.context_parallel.ulysses_model import UlyssesModel
from halo.distributed.expert_parallel.parallelize import apply_ep_layers
from halo.distributed.parallelism_config import ParallelismConfig
from halo.kernels.logprobs import LABEL_IGNORE_INDEX
from halo.model_loading.liger.cross_entropy import fused_head_ce_sum
from halo.model_loading.liger.dispatch import LIGER_CP_FUSED_LOSS_ATTR
from halo.models.config_access import set_config_field_run_scoped
from halo.models.head_transform import HeadTransform
from tests.common.cp_model import head_input_lengths
from tests.common.distributed import world_mean
from tests.common.ep_reference import random_token_batch
from tests.common.equivalence.cp_equivalence import BATCH, CP_SIZE, FAMILIES, SEED, SEQ, tiny_model
from tests.common.first_step import parameter_gradient_checks
from tests.common.harness import gpu_test_main
from tests.common.memory import cleanup_memory
from tests.common.numerics import cos_sim
from tests.common.parallelism import single_process_ep_config
from tests.common.rank_log import log, log_all

KERNEL_TRANSFORMS = {
    "identity": HeadTransform(),
    "hidden_scale": HeadTransform(hidden_scale=0.5),
    "logit_scale": HeadTransform(logit_scale=0.0625),
    "softcap": HeadTransform(softcap=30.0),
    "vocab_cut": HeadTransform(vocab_size=4089),
    "all": HeadTransform(hidden_scale=2.0, logit_scale=0.25, softcap=30.0, vocab_size=4089),
}
KERNEL_SHAPE = (2048, 256, 4096)  # tokens, hidden, vocab
# Both sides round the same bf16 logits; the fused kernel softcaps in fp32 where the projection softcaps in bf16.
KERNEL_LOSS_RTOL = 2e-3
KERNEL_GRAD_COSINE_MIN = 0.9999
KERNEL_GRAD_NORM_RTOL = 1e-2
# Qwen3's vocabulary at a modest width: the logits loss holds ~10 B per logit at its peak, the fused one a chunk.
MEMORY_SHAPE = (8192, 1024, 151936)
MEMORY_RATIO_MAX = 0.25
# A wrapper step whose bf16 logits plane (2 B x local tokens x vocab) dwarfs the tiny model's activations, gradients
# and the fused kernel's fp32 weight gradient; the tiny configs' 512 positions bound the sequence.
WRAPPER_MEMORY_VOCAB = 151936
WRAPPER_MEMORY_BATCH, WRAPPER_MEMORY_SEQ = 16, 512
MOE_AUX_OVERRIDES = {"output_router_logits": True, "router_aux_loss_coef": 0.1}
# Two bf16 paths over one backbone: only the head rounds differently (one GEMM against per-chunk GEMMs).
MODEL_LOSS_RTOL = 2e-3


def _relative_gap(got: torch.Tensor, want: torch.Tensor) -> float:
    return ((got.float() - want.float()).abs().max() / want.float().abs().max().clamp(min=1e-12)).item()


def kernel_checks(device, metrics: dict) -> dict[str, bool]:
    tokens, hidden_size, vocab = KERNEL_SHAPE
    generator = torch.Generator(device=device).manual_seed(SEED)
    checks = {}
    for name, transform in KERNEL_TRANSFORMS.items():
        head = nn.Linear(hidden_size, vocab, bias=False, device=device, dtype=torch.bfloat16)
        hidden = torch.randn(tokens, hidden_size, generator=generator, device=device, dtype=torch.bfloat16)
        hidden.requires_grad_()
        targets = torch.randint(0, transform.vocab_size or vocab, (tokens,), generator=generator, device=device)
        targets[::7] = LABEL_IGNORE_INDEX

        logits = transform.project(head, hidden).float()
        reference = F.cross_entropy(logits, targets, ignore_index=LABEL_IGNORE_INDEX, reduction="sum")
        reference.backward()
        want = {"hidden": hidden.grad.clone(), "weight": head.weight.grad.clone()}
        hidden.grad, head.weight.grad = None, None

        loss, predicted = fused_head_ce_sum(head, transform, hidden, targets)
        loss.backward()
        got = {"hidden": hidden.grad, "weight": head.weight.grad}
        loss_gap = _relative_gap(loss, reference)
        # Liger writes -1 at an ignored target, and a -1 on a supervised row is a miss. The bf16 softcap can round a
        # near-tie into an exact tie that argmax breaks by index, so the kernel's token must hold its row's maximal
        # logit.
        supervised = targets != LABEL_IGNORE_INDEX
        rows, tokens_at = logits[supervised], predicted[supervised]
        picked = rows.gather(-1, tokens_at.clamp(min=0)[:, None]).squeeze(-1)
        argmax_agree = ((tokens_at >= 0) & (picked == rows.amax(dim=-1))).float().mean().item()
        grads_hold = all(
            cos_sim(got[key], want[key], label=f"{name} d{key}") >= KERNEL_GRAD_COSINE_MIN
            and abs(got[key].float().norm() / want[key].float().norm() - 1) <= KERNEL_GRAD_NORM_RTOL
            for key in want
        )
        log(f"  kernel {name}: loss gap {loss_gap:.2e}, argmax agreement {argmax_agree:.4f}")
        metrics |= {f"kernel_{name}_loss_gap": loss_gap, f"kernel_{name}_argmax_agreement": argmax_agree}
        checks[f"kernel_{name}_loss"] = loss_gap <= KERNEL_LOSS_RTOL
        checks[f"kernel_{name}_argmax"] = argmax_agree >= 0.999
        checks[f"kernel_{name}_grads"] = grads_hold
    return checks


def _peak_bytes(device, step) -> int:
    cleanup_memory()
    torch.cuda.synchronize(device)
    before = torch.cuda.memory_allocated(device)
    torch.cuda.reset_peak_memory_stats(device)
    step()
    torch.cuda.synchronize(device)
    return torch.cuda.max_memory_allocated(device) - before


def memory_checks(device, metrics: dict) -> dict[str, bool]:
    tokens, hidden_size, vocab = MEMORY_SHAPE
    head = nn.Linear(hidden_size, vocab, bias=False, device=device, dtype=torch.bfloat16)
    hidden = torch.randn(tokens, hidden_size, device=device, dtype=torch.bfloat16, requires_grad=True)
    targets = torch.randint(0, vocab, (tokens,), device=device)

    def logits_loss():
        logits = head(hidden).float()
        F.cross_entropy(logits, targets, ignore_index=LABEL_IGNORE_INDEX, reduction="sum").backward()

    def fused_loss():
        fused_head_ce_sum(head, HeadTransform(), hidden, targets)[0].backward()

    logits_peak, fused_peak = _peak_bytes(device, logits_loss), _peak_bytes(device, fused_loss)
    ratio = fused_peak / logits_peak
    log(f"  peak: logits loss {logits_peak / 2**30:.2f} GiB, fused {fused_peak / 2**30:.2f} GiB ({ratio:.3f})")
    metrics |= {"logits_loss_peak_gib": logits_peak / 2**30, "fused_loss_peak_gib": fused_peak / 2**30}
    return {"fused_peak_under_a_quarter": ratio <= MEMORY_RATIO_MAX}


def _seeded_model(family: str, device, *, fused: bool, **overrides) -> nn.Module:
    model = tiny_model(family, "gpu", device, **(MOE_AUX_OVERRIDES if FAMILIES[family].moe else {}), **overrides)
    for tensor in [*model.parameters(), *model.buffers()]:
        if tensor.is_floating_point():
            dist.broadcast(tensor.data, src=0)
    if FAMILIES[family].moe:
        apply_ep_layers(model, single_process_ep_config(use_grouped_gemm=True))
    if fused:
        set_config_field_run_scoped(model.config, LIGER_CP_FUSED_LOSS_ATTR, True)
    return model


def _cp_step(model: nn.Module, cp_config, ids, labels) -> dict:
    wrapper = UlyssesModel(model, cp_config).train()
    with head_input_lengths(model) as head_lengths:
        outputs = wrapper(input_ids=ids, labels=labels, use_cache=False)
    outputs.loss.backward()
    grads = {}
    for name, param in wrapper.named_parameters():
        if param.grad is not None:
            dist.all_reduce(param.grad, op=dist.ReduceOp.AVG)
            grads[name] = param.grad
    aux = getattr(outputs, "aux_loss", None)
    return {
        "head_lengths": head_lengths,
        "loss": outputs.loss.item(),
        "logits": outputs.logits,
        "token_accuracy": getattr(outputs, "token_accuracy", None),
        "aux": None if aux is None else aux.item(),
        "grads": grads,
    }


def model_checks(family: str, device, metrics: dict) -> dict[str, bool]:
    cp_config = ParallelismConfig(cp_size=CP_SIZE).cp_config()
    logits_model = _seeded_model(family, device, fused=False)
    vocab = logits_model.get_input_embeddings().num_embeddings
    ids, labels = random_token_batch(vocab, BATCH, SEQ, device, seed=SEED)
    labels[0, : SEQ // CP_SIZE] = LABEL_IGNORE_INDEX
    labels[0, SEQ // CP_SIZE] = ids[0, SEQ // CP_SIZE]
    takes_logits_to_keep = "logits_to_keep" in inspect.signature(logits_model.forward).parameters
    logits_run = _cp_step(logits_model, cp_config, ids, labels)
    fused_run = _cp_step(_seeded_model(family, device, fused=True), cp_config, ids, labels)

    log_all(f"  loss: logits path {logits_run['loss']:.6f}, fused path {fused_run['loss']:.6f}")
    logits_mean, fused_mean = world_mean(logits_run["loss"]), world_mean(fused_run["loss"])
    loss_gap = abs(fused_mean - logits_mean) / abs(logits_mean)
    grads_match, grad_metrics = parameter_gradient_checks(fused_run["grads"], logits_run["grads"])
    metrics |= {"model_loss_gap": loss_gap, **{f"model_{k}": v for k, v in grad_metrics.items()}}
    checks = {
        "model_loss_matches": loss_gap <= MODEL_LOSS_RTOL,
        "model_grads_cover": fused_run["grads"].keys() == logits_run["grads"].keys(),
        "model_grads_match": grads_match,
    }
    if logits_run["aux"] is not None or fused_run["aux"] is not None:
        aux_pair = (logits_run["aux"], fused_run["aux"])
        log_all(f"  aux: logits path {aux_pair[0]}, fused path {aux_pair[1]}")
        checks["model_aux_on_both_paths"] = None not in aux_pair and abs(aux_pair[0] - aux_pair[1]) <= 1e-3 * abs(
            aux_pair[0]
        )
    if not takes_logits_to_keep:
        checks["model_without_logits_to_keep_keeps_the_logits_loss"] = fused_run["logits"] is not None
        return checks

    targets = cp_local_targets(labels, cp_config.cp_rank, CP_SIZE)
    supervised = targets != LABEL_IGNORE_INDEX
    correct = ((logits_run["logits"].argmax(dim=-1) == targets) & supervised).sum()
    expected = (correct / supervised.sum().clamp(min=1)).item()
    accuracy = fused_run["token_accuracy"]
    gap = abs(accuracy.item() - expected) if accuracy is not None else float("inf")
    log_all(f"  chunk accuracy: fused {accuracy}, from logits {expected:.4f}")
    # A bf16 near-tie may pick another argmax on either side; allow two of the chunk's targets.
    checks["model_fused_returns_no_logits"] = fused_run["logits"] is None
    checks["model_logits_path_projects_the_chunk"] = logits_run["head_lengths"] == [SEQ // CP_SIZE]
    checks["model_fused_path_projects_one_position"] = fused_run["head_lengths"] == [1]
    checks["model_chunk_accuracy"] = gap <= 2 / max(int(supervised.sum()), 1)
    return checks | wrapper_memory_checks(family, device, metrics)


def wrapper_memory_checks(family: str, device, metrics: dict) -> dict[str, bool]:
    """The fused and logits training steps' peaks through the CP wrapper over one family model each."""
    cp_config = ParallelismConfig(cp_size=CP_SIZE).cp_config()
    ids, labels = random_token_batch(WRAPPER_MEMORY_VOCAB, WRAPPER_MEMORY_BATCH, WRAPPER_MEMORY_SEQ, device, seed=SEED)
    peaks = {}
    for path in ("logits", "fused"):
        model = _seeded_model(family, device, fused=path == "fused", vocab_size=WRAPPER_MEMORY_VOCAB)
        wrapper = UlyssesModel(model, cp_config).train()

        def step(wrapper=wrapper):
            wrapper(input_ids=ids, labels=labels, use_cache=False).loss.backward()

        peaks[path] = _peak_bytes(device, step)
        del model, wrapper
    plane = 2 * WRAPPER_MEMORY_BATCH * (WRAPPER_MEMORY_SEQ // CP_SIZE) * WRAPPER_MEMORY_VOCAB
    log_all(
        f"  wrapper peak: logits path {peaks['logits'] / 2**30:.2f} GiB, fused path {peaks['fused'] / 2**30:.2f} GiB, "
        f"bf16 logits plane {plane / 2**30:.2f} GiB"
    )
    metrics |= {f"wrapper_{path}_peak_gib": peak / 2**30 for path, peak in peaks.items()}
    return {
        "wrapper_fused_peak_below_logits_peak": peaks["fused"] < peaks["logits"],
        "wrapper_fused_peak_below_the_logits_plane": peaks["fused"] < plane,
    }


@gpu_test_main(exact_world_size=CP_SIZE, prefix="cp_fused_head_loss")
def run(ctx):
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=sorted(FAMILIES), required=True)
    family = parser.parse_args().family
    metrics: dict[str, float] = {}
    checks = kernel_checks(ctx.device, metrics)
    checks |= memory_checks(ctx.device, metrics)
    log(f"\n--- {family} under CP={CP_SIZE} ---")
    checks |= model_checks(family, ctx.device, metrics)
    return {"checks": checks, "metrics": metrics}


if __name__ == "__main__":
    run()
