#!/usr/bin/env python
"""The gate under packed GRPO micro-batches (``packed_microbatch_tokens``): several rows flattened into one ``[1, N]``
varlen forward score and train as each row alone. Held per family, per emission (what the row hands the model besides
``input_ids``), per attention backend and for an ``N`` aligned to 128 and not.

``padding_free`` is refused on M-RoPE decoders because a flattened M-RoPE row faulted the flash varlen kernels, while
SFT ``packing`` ships on the same decoders; which input faulted was never isolated, so each emission runs here. A CUDA
fault poisons the process, hence one launch per family and emission. The verdict this file records decides
``PACKED_MICROBATCH_ADMITS_MROPE`` in ``halo/trainers/grpo/environmental/environmental_rules.py``.
"""

import argparse
import copy

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

from halo.hardware import is_blackwell_gpu
from halo.model_loading.dtype import configure_float32_matmul_precision
from halo.model_loading.model_preparation import apply_family_patches
from halo.models.segment_markers import packed_row_segment_kwargs, packed_row_varlen_kwargs, segment_markers_for
from tests.common.attention_kernels import kernel_calls
from tests.common.harness import gpu_test_main, skip_unless_blackwell
from tests.common.numerics import max_abs_rel_err
from tests.common.rank_log import log
from tests.common.tiny_models import (
    TINY_DENSE_ATTENTION_FAMILIES,
    TINY_DENSE_FAMILIES,
    TINY_MOE_FAMILIES,
    TINY_MOE_VLM_FAMILIES,
    TINY_QWEN3_5_COMPOSITE_FAMILY,
    tiny_family_model,
)

SEED = 0
# Documents of one micro-batch: N = 1024 is a multiple of 128, N = 648 is not.
ALIGNED_DOCUMENTS = (384, 384, 256)
UNALIGNED_DOCUMENTS = (300, 211, 137)
# A packed row is held to the per-row bf16 forward's own distance from the fp32 reference, times this: two bf16 kernels
# accumulate in different orders, and the document set changes the grouped GEMM's tile choice.
PACKED_ERROR_RATIO = 2.0
# Of the reference's peak, where the per-row error is below what a ratio can resolve.
ERROR_FLOOR = 2e-3

_GQA_256 = {"num_attention_heads": 16, "num_key_value_heads": 2, "head_dim": 256}
_ALL_EIGHT_EXPERTS = {"num_experts": 8, "num_experts_per_tok": 8}
# Three GatedDeltaNet layers and a full-attention layer: the delta rule, the conv and the attention all see the row.
_HYBRID = {
    "num_hidden_layers": 4,
    "layer_types": ["linear_attention", "linear_attention", "linear_attention", "full_attention"],
}
_GDN = {**_GQA_256, **_HYBRID}
FAMILIES = {
    "qwen3_5": (TINY_DENSE_FAMILIES["qwen3_5"], _GDN),
    "qwen3_5_moe_text": (TINY_MOE_FAMILIES["qwen3_5_moe_text"], {**_GDN, **_ALL_EIGHT_EXPERTS}),
    "qwen3_5_composite": (TINY_QWEN3_5_COMPOSITE_FAMILY, _GDN),
    "qwen3_5_moe_composite": (TINY_MOE_VLM_FAMILIES["qwen3_5_moe"], {**_GDN, **_ALL_EIGHT_EXPERTS}),
    "qwen3_next": (TINY_MOE_FAMILIES["qwen3_next"], {**_GDN, **_ALL_EIGHT_EXPERTS}),
    "qwen3_moe": (TINY_MOE_FAMILIES["qwen3_moe"], {"num_hidden_layers": 2, **_ALL_EIGHT_EXPERTS}),
    "llama": (TINY_DENSE_ATTENTION_FAMILIES["llama"], {"num_hidden_layers": 2}),
    "nemotron_h": (TINY_MOE_FAMILIES["nemotron_h"], {}),
    "lfm2_moe": (TINY_MOE_FAMILIES["lfm2_moe"], {}),
    "gpt_oss": (TINY_MOE_FAMILIES["gpt_oss"], {"num_hidden_layers": 2, **_ALL_EIGHT_EXPERTS}),
}
# Live sinks ride only the FA4 varlen entry (halo/models/attention/gpt_oss_sinks.py), which a Blackwell GPU alone has.
FA4_ONLY_FAMILIES = frozenset({"gpt_oss"})
ATTENTION_BACKENDS = ("flash_attention_2", "flash_attention_4")


def _emission_kwargs(emission: str, position_ids: torch.Tensor, config) -> dict:
    """What the micro-batch hands the model beside ``input_ids``: ``position_ids`` alone; those with the family's
    segment markers (what SFT ``packing`` emits); the flash varlen set alone (what ``padding_free`` emits); the markers
    with the M-RoPE positions spelled out as the ``[3, 1, N]`` plane the decoder would build itself."""
    if emission == "position_ids":
        return {"position_ids": position_ids}
    if emission == "markers":
        return {"position_ids": position_ids, **packed_row_segment_kwargs(position_ids, segment_markers_for(config))}
    if emission == "flash_varlen":
        return {"position_ids": position_ids, **packed_row_varlen_kwargs(position_ids)}
    if emission == "mrope_plane":
        return {
            "position_ids": position_ids.expand(3, -1, -1).contiguous(),
            **packed_row_segment_kwargs(position_ids, segment_markers_for(config)),
        }
    raise ValueError(f"unknown emission {emission!r}")


def _set_attention(model, attn_implementation: str) -> None:
    resolved = apply_family_patches(model.config, attn_implementation)
    for config in (model.config, getattr(model.config, "text_config", None)):
        if config is not None:
            config._attn_implementation = resolved


def _documents(lengths: tuple[int, ...], vocab_size: int, device) -> list[torch.Tensor]:
    generator = torch.Generator().manual_seed(SEED)
    return [torch.randint(1, vocab_size, (1, length), generator=generator).to(device) for length in lengths]


def _targets(documents: list[torch.Tensor]) -> torch.Tensor:
    """Each document's next token, -100 at its last position, which has none."""
    return torch.cat([F.pad(doc[0, 1:], (0, 1), value=-100) for doc in documents])


def _step(model, documents: list[torch.Tensor], emission: str | None):
    """Logits over all documents' positions, in order, and the parameter gradients of their summed cross entropy:
    one forward per document, or one flattened forward under ``emission``."""
    model.zero_grad(set_to_none=True)
    if emission is None:
        logits = torch.cat([model(input_ids=doc, use_cache=False).logits[0] for doc in documents])
    else:
        ids = torch.cat(documents, dim=1)
        positions = torch.cat([torch.arange(doc.size(1)) for doc in documents]).unsqueeze(0).to(ids.device)
        logits = model(input_ids=ids, use_cache=False, **_emission_kwargs(emission, positions, model.config)).logits[0]
    F.cross_entropy(logits.float(), _targets(documents), ignore_index=-100, reduction="sum").backward()
    grads = {name: p.grad.detach().float() for name, p in model.named_parameters() if p.grad is not None}
    return logits.detach().float(), grads


def _grad_error(grads: dict[str, torch.Tensor], reference: dict[str, torch.Tensor]) -> float:
    return max(max_abs_rel_err(grads[name], ref) for name, ref in reference.items() if name in grads)


def _check_attention(family_name, attn, emission, lengths, label, models, checks, metrics) -> None:
    reference_model, model = models
    documents = _documents(
        lengths, reference_model.config.get_text_config().vocab_size, next(model.parameters()).device
    )
    reference_logits, reference_grads = _step(reference_model, documents, None)
    _set_attention(model, attn)
    with kernel_calls() as per_row_calls:
        per_row_logits, per_row_grads = _step(model, documents, None)
    with kernel_calls() as packed_calls:
        packed_logits, packed_grads = _step(model, documents, emission)
    floor = max(max_abs_rel_err(per_row_logits, reference_logits), ERROR_FLOOR)
    grad_floor = max(_grad_error(per_row_grads, reference_grads), ERROR_FLOOR)
    packed = max_abs_rel_err(packed_logits, reference_logits)
    packed_grad = _grad_error(packed_grads, reference_grads)
    log(
        f"  [{family_name}/{attn}/{emission}/{label} N={sum(lengths)}] logits {packed:.3e} (per-row {floor:.3e}) "
        f"grads {packed_grad:.3e} (per-row {grad_floor:.3e}) packed kernels {dict(packed_calls)}"
    )
    key = f"{attn}_{label}"
    finite = torch.isfinite(packed_logits).all().item() and all(
        torch.isfinite(g).all().item() for g in packed_grads.values()
    )
    checks[f"{key}_finite"] = bool(finite)
    checks[f"{key}_reaches_the_kernel"] = set(packed_calls) == {attn} == set(per_row_calls)
    checks[f"{key}_logits_match_per_row"] = packed <= PACKED_ERROR_RATIO * floor
    checks[f"{key}_grads_match_per_row"] = packed_grad <= PACKED_ERROR_RATIO * grad_floor
    metrics[f"{key}_logits_error"], metrics[f"{key}_logits_error_per_row"] = packed, floor
    metrics[f"{key}_grad_error"], metrics[f"{key}_grad_error_per_row"] = packed_grad, grad_floor


def _isolation_check(model, attn: str, emission: str, lengths, checks) -> None:
    """Changing one document leaves every other document's logits where they were, up to kernel noise."""
    _set_attention(model, attn)
    device = next(model.parameters()).device
    documents = _documents(lengths, model.config.get_text_config().vocab_size, device)
    changed = [documents[0], (documents[1] + 1) % model.config.get_text_config().vocab_size, *documents[2:]]
    with torch.no_grad():
        before, _ = _step(model, documents, emission)
        after, _ = _step(model, changed, emission)
    first = lengths[0]
    untouched = max_abs_rel_err(after[:first], before[:first])
    third_start = lengths[0] + lengths[1]
    checks[f"{attn}_a_neighbour_does_not_move_the_documents_around_it"] = (
        untouched <= ERROR_FLOOR and max_abs_rel_err(after[third_start:], before[third_start:]) <= ERROR_FLOOR
    )


@gpu_test_main(exact_world_size=1, partial_state=False, prefix="packed_rl_forward")
def run(ctx):
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=sorted(FAMILIES), required=True)
    parser.add_argument(
        "--emission", choices=("position_ids", "markers", "flash_varlen", "mrope_plane"), default="markers"
    )
    parser.add_argument("--attn", choices=ATTENTION_BACKENDS)
    parser.add_argument(
        "--checkpoint", help="a real checkpoint to truncate to --layers layers and run in place of the tiny family"
    )
    parser.add_argument("--layers", type=int, default=4)
    args = parser.parse_args()
    family, overrides = FAMILIES[args.family]
    backends = (args.attn,) if args.attn else ATTENTION_BACKENDS
    if args.family in FA4_ONLY_FAMILIES:
        backends = tuple(b for b in backends if b == "flash_attention_4")
    if args.family in FA4_ONLY_FAMILIES:
        skip_unless_blackwell("the live-sink varlen row")
    configure_float32_matmul_precision()

    torch.manual_seed(SEED)
    if args.checkpoint:
        base = AutoModelForCausalLM.from_pretrained(
            args.checkpoint, num_hidden_layers=args.layers, torch_dtype=torch.bfloat16
        )
    else:
        base = tiny_family_model(family, overrides=overrides)
    reference_model = copy.deepcopy(base).to(ctx.device, torch.float32)
    reference_model.config._attn_implementation = "sdpa"
    model = copy.deepcopy(base).to(ctx.device, torch.bfloat16)

    checks, metrics = {}, {}
    for attn in backends:
        if attn == "flash_attention_4" and not is_blackwell_gpu():
            log(f"  [{args.family}] FA4 skipped: not a Blackwell GPU")
            continue
        for label, lengths in (("aligned", ALIGNED_DOCUMENTS), ("unaligned", UNALIGNED_DOCUMENTS)):
            _check_attention(
                args.family, attn, args.emission, lengths, label, (reference_model, model), checks, metrics
            )
        _isolation_check(model, attn, args.emission, UNALIGNED_DOCUMENTS, checks)
    return {"checks": checks, "metrics": metrics}


if __name__ == "__main__":
    run()
