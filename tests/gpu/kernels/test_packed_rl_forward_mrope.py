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
from transformers import AutoModelForCausalLM

from halo.hardware import is_blackwell_gpu
from halo.model_loading.dtype import configure_float32_matmul_precision
from tests.common.attention_kernels import kernel_calls
from tests.common.harness import gpu_test_main, skip_unless_blackwell
from tests.common.numerics import max_abs_rel_err
from tests.common.packed_rows import (
    ERROR_FLOOR,
    SEED,
    build_documents,
    check_isolation,
    grad_error,
    row_step,
    set_attention,
)
from tests.common.rank_log import log
from tests.common.tiny_models import (
    TINY_DENSE_ATTENTION_FAMILIES,
    TINY_DENSE_FAMILIES,
    TINY_MOE_FAMILIES,
    TINY_MOE_VLM_FAMILIES,
    TINY_QWEN3_5_COMPOSITE_FAMILY,
    tiny_family_model,
)

# Documents of one micro-batch: N = 1024 is a multiple of 128, N = 648 is not.
ALIGNED_DOCUMENTS = (384, 384, 256)
UNALIGNED_DOCUMENTS = (300, 211, 137)
# A packed row is held to the per-row bf16 forward's own distance from the fp32 reference, times this: two bf16 kernels
# accumulate in different orders, and the document set changes the grouped GEMM's tile choice.
PACKED_ERROR_RATIO = 2.0

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


def _check_attention(family_name, attn, emission, lengths, label, models, checks, metrics) -> None:
    reference_model, model = models
    documents = build_documents(
        lengths, reference_model.config.get_text_config().vocab_size, next(model.parameters()).device
    )
    reference_logits, reference_grads = row_step(reference_model, documents, None)
    set_attention(model, attn)
    with kernel_calls() as per_row_calls:
        per_row_logits, per_row_grads = row_step(model, documents, None)
    with kernel_calls() as packed_calls:
        packed_logits, packed_grads = row_step(model, documents, emission)
    floor = max(max_abs_rel_err(per_row_logits, reference_logits), ERROR_FLOOR)
    grad_floor = max(grad_error(per_row_grads, reference_grads), ERROR_FLOOR)
    packed = max_abs_rel_err(packed_logits, reference_logits)
    packed_grad = grad_error(packed_grads, reference_grads)
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
        check_isolation(model, attn, args.emission, UNALIGNED_DOCUMENTS, checks)
    return {"checks": checks, "metrics": metrics}


if __name__ == "__main__":
    run()
