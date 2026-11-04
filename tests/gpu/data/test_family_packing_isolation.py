#!/usr/bin/env python
"""Packed rows on the real kernels: the kernel a packed run resolves on each family, and the kernels a run can pin.

Doc A is perturbed and precedes the measured doc B, since causality alone shields an earlier document. The control
withholds the isolation and must move doc B far more, or the zero is vacuous. Ring's Lightning Attention-2 crosses
documents by design. ``--ep`` splits the experts over the ranks, which hold different documents: every number is the
worst rank's, so a token's result reaching the wrong row moves it.
"""

import argparse
from collections.abc import Callable
from dataclasses import dataclass, field

import torch
from transformers.models.nemotron_h.modeling_nemotron_h import NemotronHMamba2Mixer

from halo.configs.script_arguments import ModelArguments
from halo.data.collators.factory import select_data_collator
from halo.distributed.parallelism_config import ParallelismConfig
from halo.entrypoints.model_setup import default_training_rows_attn_implementation
from halo.hardware import is_blackwell_gpu
from halo.model_loading.parallel.loader import load_distributed_model
from halo.models.attention.sdpa_flex_sliding import SDPA_FLEX_SLIDING
from halo.models.attention.selection import is_flash_attn_implementation
from tests.common.attention_kernels import kernel_calls
from tests.common.distributed import world_all, world_max, world_min
from tests.common.harness import gpu_test_main, skip_unless_blackwell
from tests.common.rank_log import log
from tests.common.tiny_models import (
    TINY_BAILING_RING_FAMILY,
    TINY_DENSE_ATTENTION_FAMILIES,
    TINY_DENSE_FAMILIES,
    TINY_LAGUNA_REMOTE_FAMILY,
    TINY_MOE_FAMILIES,
    TINY_MOE_VLM_FAMILIES,
    TinyFamily,
    shared_tiny_family_checkpoint,
    word_level_tokenizer,
)

SEED = 0
DOCUMENTS = (48, 40)
# Each past FlexAttention's one tile, under which a masked call keeps plain SDPA.
FLEX_DOCUMENTS = (192, 160)
INIT_STD = 0.1
# Fraction of doc B's largest |logit|. Isolated, only bf16 reduction order differs; a leak, a dropped causal flag or a
# wrong scale moves the logits by O(1) of their scale.
ISOLATED_REL_MAX = 2e-2
LEAK_REL_MIN = 1e-1
# On a dense label the packed row (masked SDPA) and lone document (causal) take different kernels, whose bf16 rounding
# moves doc B by ~1e-1, onto the leak's bar; at fp32 they agree to ~1e-6.
FP32_ISOLATED_REL_MAX = 1e-4
# Packed and lone doc B run different kernels, and GatedDeltaNet Triton kernels take dots at TF32 even on fp32 inputs,
# so they part at 1.06e-4 of scale while changing doc A moves doc B by exactly 0 and the control by ~1.5.
QWEN4_EXP_FP32_ISOLATED_REL_MAX = 5e-4
FA4 = "flash_attention_4"
FA2 = "flash_attention_2"
# Under both documents, so a window a packed row dropped moves doc B against doc B alone.
SHORT_WINDOW = {"sliding_window": 16}
# Each release's head width: the width FA4 trains on its dedicated kernel, which a packed run leaves for FA2 on these.
_GATED_DELTA_NET_256 = {"head_dim": 256}
# Over ``INIT_STD``: fifty times Mamba's stock ``in_proj``, whose stock init leaves a crossed boundary below rounding.
MAMBA_IN_PROJ_SCALE = 10.0


def _scale_mamba_inputs(model) -> None:
    for module in model.modules():
        if isinstance(module, NemotronHMamba2Mixer):
            module.in_proj.weight.data.mul_(MAMBA_IN_PROJ_SCALE)


@dataclass(frozen=True)
class _FamilyRow:
    family: TinyFamily
    overrides: dict = field(default_factory=dict)
    # What a packed run with attention unset resolves on a Blackwell GPU.
    packed_default: str = FA4
    documents: tuple[int, int] = DOCUMENTS
    isolates: bool = True
    fp32_isolated_rel_max: float = FP32_ISOLATED_REL_MAX
    after_randomize: Callable | None = None


FAMILIES = {
    "bailing_moe": _FamilyRow(TINY_MOE_FAMILIES["bailing_moe"]),
    "bailing_moe_linear": _FamilyRow(TINY_BAILING_RING_FAMILY, isolates=False),
    "cohere2_moe": _FamilyRow(TINY_MOE_FAMILIES["cohere2_moe"], SHORT_WINDOW),
    "gemma4_text": _FamilyRow(
        TINY_MOE_FAMILIES["gemma4_text"],
        {"head_dim": 64, "global_head_dim": 512},
        packed_default=SDPA_FLEX_SLIDING,
        documents=FLEX_DOCUMENTS,
    ),
    # What the Gemma 4 recipe loads: its decoder, through the composite's forward.
    "gemma4": _FamilyRow(
        TINY_MOE_VLM_FAMILIES["gemma4"],
        {"head_dim": 64, "global_head_dim": 512},
        packed_default=SDPA_FLEX_SLIDING,
        documents=FLEX_DOCUMENTS,
    ),
    # The latent split of GLM-4.7-Flash, which stays on FA2 at this width.
    "glm4_moe_lite": _FamilyRow(
        TINY_MOE_FAMILIES["glm4_moe_lite"],
        {
            "num_attention_heads": 20,
            "num_key_value_heads": 20,
            "qk_nope_head_dim": 192,
            "qk_rope_head_dim": 64,
            "v_head_dim": 256,
            "kv_lora_rank": 64,
        },
        packed_default=FA2,
    ),
    # Eight random layers amplify the kernels' bf16 noise past the bar, to 9e-2 between FA4 and SDPA, and
    # top-2 routing flips on it: both would measure something other than the kernel.
    "gpt_oss": _FamilyRow(
        TINY_MOE_FAMILIES["gpt_oss"], {**SHORT_WINDOW, "num_hidden_layers": 2, "num_experts_per_tok": 8}
    ),
    "laguna": _FamilyRow(TINY_MOE_FAMILIES["laguna"], SHORT_WINDOW),
    # What the Laguna recipes load: the pinned hub file, which passes a flash kernel no window, so plain SDPA.
    "laguna_remote": _FamilyRow(TINY_LAGUNA_REMOTE_FAMILY, SHORT_WINDOW, packed_default="sdpa"),
    "lfm2_moe": _FamilyRow(TINY_MOE_FAMILIES["lfm2_moe"]),
    "llama": _FamilyRow(TINY_DENSE_ATTENTION_FAMILIES["llama"], {"head_dim": 64}),
    "mistral4": _FamilyRow(
        TINY_MOE_FAMILIES["mistral4"],
        {"qk_nope_head_dim": 32, "qk_rope_head_dim": 32, "qk_head_dim": 64, "v_head_dim": 64},
    ),
    "mistral": _FamilyRow(TINY_DENSE_ATTENTION_FAMILIES["mistral"], {**SHORT_WINDOW, "head_dim": 64}),
    "qwen2": _FamilyRow(
        TINY_DENSE_ATTENTION_FAMILIES["qwen2"],
        {**SHORT_WINDOW, "hidden_size": 256, "use_sliding_window": True, "max_window_layers": 0},
    ),
    "nemotron_h": _FamilyRow(TINY_MOE_FAMILIES["nemotron_h"], after_randomize=_scale_mamba_inputs),
    # Its text tower through the composite's forward, whose kwargs carry the markers past the tower to the loss too.
    "nemotron_h_omni": _FamilyRow(TINY_MOE_VLM_FAMILIES["nemotron_h_omni"], after_randomize=_scale_mamba_inputs),
    "qwen3": _FamilyRow(TINY_DENSE_FAMILIES["qwen3"], {"head_dim": 64}),
    "qwen3_5": _FamilyRow(TINY_DENSE_FAMILIES["qwen3_5"], _GATED_DELTA_NET_256, packed_default=FA2),
    # What the Qwen3.5/3.6 recipes load: this decoder, through the composite's M-RoPE forward.
    "qwen3_5_moe": _FamilyRow(TINY_MOE_VLM_FAMILIES["qwen3_5_moe"], _GATED_DELTA_NET_256, packed_default=FA2),
    "qwen3_5_moe_text": _FamilyRow(TINY_MOE_FAMILIES["qwen3_5_moe_text"], _GATED_DELTA_NET_256, packed_default=FA2),
    "qwen3_moe": _FamilyRow(TINY_MOE_FAMILIES["qwen3_moe"]),
    "qwen3_next": _FamilyRow(TINY_MOE_FAMILIES["qwen3_next"], _GATED_DELTA_NET_256, packed_default=FA2),
    # Its class declares no flash kernel: SDPA under the tiled QSA.
    "qwen4_exp_text": _FamilyRow(
        TINY_MOE_FAMILIES["qwen4_exp_text"],
        packed_default="sdpa",
        fp32_isolated_rel_max=QWEN4_EXP_FP32_ISOLATED_REL_MAX,
    ),
    "step3p7": _FamilyRow(
        TINY_MOE_FAMILIES["step3p7"], SHORT_WINDOW, packed_default=SDPA_FLEX_SLIDING, documents=FLEX_DOCUMENTS
    ),
    # Zaya's tiny head_dim 16 is below what the FA3/FA4 kernels take.
    "zaya": _FamilyRow(TINY_MOE_FAMILIES["zaya"], {"head_dim": 64}),
}

PARSER = argparse.ArgumentParser()
PARSER.add_argument("--family", choices=sorted(FAMILIES), required=True)
PARSER.add_argument("--attn", default=None, help="attn_implementation; unset = what a packed run resolves")
PARSER.add_argument("--ep", type=int, default=1, help="expert_parallel_size, and the ranks this row takes")
PARSER.add_argument("--liger", action="store_true", help="Liger's kernels, as a packed recipe's defaults run")
ARGS = PARSER.parse_args()


def _randomized(after: Callable | None) -> Callable:
    """Widen the random init so a leak reads well above bf16 noise."""

    def edit(model) -> None:
        for param in model.parameters():
            if param.dim() >= 2:
                torch.nn.init.normal_(param, std=INIT_STD)
        if after is not None:
            after(model)

    return edit


def _packed_run_request(ckpt: str, attn_implementation: str | None) -> str | None:
    """The request ``scripts/training/sft.py`` makes of the loader for a packed run."""
    return default_training_rows_attn_implementation(
        ModelArguments(model_name_or_path=ckpt, attn_implementation=attn_implementation, trust_remote_code=True),
        packing=True,
        padding_free=False,
        per_device_train_batch_size=1,
        dtype=torch.bfloat16,
    )


def _load(ckpt: str, attn_implementation: str | None, dtype: torch.dtype = torch.bfloat16):
    model, _ = load_distributed_model(
        model_name_or_path=ckpt,
        parallelism_config=ParallelismConfig(ep_size=ARGS.ep, max_concurrent_loading=0),
        dtype=dtype,
        trust_remote_code=True,
        attn_implementation=attn_implementation,
        use_liger_kernel=ARGS.liger,
        model_class=FAMILIES[ARGS.family].family.load_class,
    )
    return model.eval()


def _row_inputs(batch, device, keys) -> dict:
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items() if key in keys}


def _doc_b_logits(model, batch, doc_b_start: int, device, *, isolated: bool) -> torch.Tensor:
    input_ids = batch["input_ids"].to(device)
    if isolated:
        kwargs = _row_inputs(batch, device, set(batch) - {"input_ids", "labels"})
    else:
        kwargs = {
            "position_ids": torch.arange(input_ids.shape[1], device=device)[None],
            "attention_mask": torch.ones_like(input_ids),
        }
    with torch.no_grad():
        return model(input_ids=input_ids, use_cache=False, **kwargs).logits[0, doc_b_start:].float()


def _one_document_logits(model, doc, device) -> torch.Tensor:
    with torch.no_grad():
        return model(input_ids=torch.tensor([doc], device=device), use_cache=False).logits[0].float()


def _fp32_doc_b_versus_alone(
    ckpt: str, label: str, batch, doc_b_start: int, doc_b, device
) -> tuple[torch.Tensor, torch.Tensor]:
    model = _load(ckpt, label, dtype=torch.float32)
    return _doc_b_logits(model, batch, doc_b_start, device, isolated=True), _one_document_logits(model, doc_b, device)


def _packed_step_is_finite(model, batch, device) -> bool:
    """A finite loss and finite gradients; an expert no token routed to has none."""
    model.train()
    model.zero_grad(set_to_none=True)
    loss = model(**_row_inputs(batch, device, set(batch)), use_cache=False).loss
    loss.backward()
    grads = [param.grad for param in model.parameters() if param.grad is not None]
    model.eval()
    return bool(torch.isfinite(loss)) and bool(grads) and all(bool(torch.isfinite(grad).all()) for grad in grads)


@gpu_test_main(exact_world_size=ARGS.ep, prefix=f"family_packing_isolation_{ARGS.family}", partial_state=False)
def run(ctx) -> dict:
    if ARGS.attn == FA4:
        skip_unless_blackwell(f"--family {ARGS.family} --attn {FA4}")
    row = FAMILIES[ARGS.family]

    tokenizer = word_level_tokenizer([f"w{i}" for i in range(500)])
    ckpt = shared_tiny_family_checkpoint(
        ctx,
        row.family,
        f"{ARGS.family}_packing_isolation",
        tokenizer,
        SEED,
        edit=_randomized(row.after_randomize),
        overrides=row.overrides,
    )
    model = _load(ckpt, _packed_run_request(ckpt, ARGS.attn))
    label = model.config._attn_implementation
    log(f"=== {ARGS.family} ep{ARGS.ep}: requested {ARGS.attn!r}, resolved {label!r}")

    g = torch.Generator().manual_seed(1 + ctx.rank)
    doc_a_length, doc_b_length = row.documents
    doc_a, doc_a_variant, doc_b = (
        torch.randint(2, len(tokenizer), (n,), generator=g).tolist()
        for n in (doc_a_length, doc_a_length, doc_b_length)
    )
    collator = select_data_collator(
        tokenizer,
        train_on_completions_only=False,
        packing=True,
        model_config=model.config,
        allow_cross_document_linear_attention=not row.isolates,
    )
    batches = [
        collator.torch_call([{"input_ids": a + doc_b, "labels": a + doc_b, "seq_lengths": list(row.documents)}])
        for a in (doc_a, doc_a_variant)
    ]

    alone = _one_document_logits(model, doc_b, ctx.device)
    scale = alone.abs().max().item()
    checks: dict[str, bool] = {}
    metrics: dict[str, float] = {}
    if ARGS.attn or is_blackwell_gpu():
        checks["resolves_the_expected_backend"] = label == (ARGS.attn or row.packed_default)
    if is_flash_attn_implementation(label):
        sdpa_alone = _one_document_logits(_load(ckpt, "sdpa"), doc_b, ctx.device)
        metrics["flash_vs_sdpa_rel"] = world_max((alone - sdpa_alone).abs().max().item() / scale)
        log(f"  one document, {label} vs sdpa: {metrics['flash_vs_sdpa_rel']:.3e} (of max |logit| {scale:.3f})")
        checks["flash_matches_sdpa_on_one_document"] = metrics["flash_vs_sdpa_rel"] <= ISOLATED_REL_MAX

    with kernel_calls() as calls:
        isolated = [_doc_b_logits(model, batch, doc_a_length, ctx.device, isolated=True) for batch in batches]
    log(f"  packed row's attention kernels: {dict(calls)}")
    checks["packed_row_runs_on_that_backend"] = world_all(set(calls) == {label})
    metrics["drift_rel"] = world_max((isolated[0] - isolated[1]).abs().max().item() / scale)
    log(f"  doc B drift when doc A changes: {metrics['drift_rel']:.3e}")
    if row.isolates:
        leaky = [_doc_b_logits(model, batch, doc_a_length, ctx.device, isolated=False) for batch in batches]
        # The rank a control leaks least on, as every rank must show the isolation is doing something.
        metrics["leak_rel"] = world_min((leaky[0] - leaky[1]).abs().max().item() / scale)
        log(f"  control drift {metrics['leak_rel']:.3e}")
        checks["packed_row_leaves_doc_b_unmoved"] = metrics["drift_rel"] <= ISOLATED_REL_MAX
        checks["control_without_isolation_leaks"] = metrics["leak_rel"] >= LEAK_REL_MIN
        # EP refuses fp32, and a dense label's two kernels part at bf16 by the leak's bar: drift alone holds that row.
        versus_alone = None
        if is_flash_attn_implementation(label):
            versus_alone = (isolated[0], alone, ISOLATED_REL_MAX, "bf16")
        elif ARGS.ep == 1:
            fp32 = _fp32_doc_b_versus_alone(ckpt, label, batches[0], doc_a_length, doc_b, ctx.device)
            versus_alone = (*fp32, row.fp32_isolated_rel_max, "fp32")
        if versus_alone is not None:
            packed_b, alone_b, bar, scored_on = versus_alone
            metrics["versus_alone_rel"] = world_max(
                (packed_b - alone_b).abs().max().item() / alone_b.abs().max().item()
            )
            log(f"  packed doc B vs alone {metrics['versus_alone_rel']:.3e} ({scored_on}, bar {bar:.0e})")
            checks["packed_doc_b_matches_doc_b_alone"] = metrics["versus_alone_rel"] <= bar

    checks["packed_backward_is_finite"] = world_all(_packed_step_is_finite(model, batches[0], ctx.device))
    return {"checks": checks, "metrics": metrics}


if __name__ == "__main__":
    run()
