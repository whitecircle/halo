#!/usr/bin/env python
"""Flattened ``padding_free`` rows on an M-RoPE decoder against ``packing`` on the same documents, on the real flash
varlen kernels, for 50 steps of fresh variable-width rows.

Per document, the padding-free logits must sit within the packing floor of the document run alone, with and without the
stacked text + 3-axis ``position_ids`` a composite builds, and every backward must run without a CUDA error. The control
hands over one running ``position_ids`` axis and no varlen arguments, so documents attend across each other and must
drift. The padding-free collator is built directly: the factory's M-RoPE refusal is what this test decides.
"""

import argparse

import torch

from halo.data.collators.factory import select_data_collator
from halo.data.collators.packing import DataCollatorWithFlattening
from halo.distributed.parallelism_config import ParallelismConfig
from halo.model_loading.parallel.loader import load_distributed_model
from halo.models.introspection.config_access import text_config
from halo.models.segment_markers import segment_markers_for
from tests.common.attention_kernels import kernel_calls
from tests.common.harness import gpu_test_main, skip_unless_blackwell
from tests.common.rank_log import log
from tests.common.tiny_models import (
    TINY_DENSE_FAMILIES,
    TINY_QWEN3_VL_FAMILY,
    TinyFamily,
    randomized_weights,
    shared_tiny_family_checkpoint,
    word_level_tokenizer,
)

SEED = 0
STEPS = 50
DOCUMENTS_PER_ROW = 3
DOCUMENT_LENGTHS = (24, 96)
VOCAB_WORDS = 500
# Fraction of a document's largest |logit|. Isolated, only bf16 reduction order differs; a leak moves it by O(1).
ISOLATED_REL_MAX = 2e-2
LEAK_REL_MIN = 1e-1
FA4 = "flash_attention_4"
FA2 = "flash_attention_2"
# Each family's text tower at the width FA4 runs, with the M-RoPE sections summing to half the head.
_HEAD_DIM_64 = {"head_dim": 64, "rope_parameters": {"rope_type": "default", "mrope_section": [16, 8, 8]}}
FAMILIES: dict[str, tuple[TinyFamily, dict]] = {
    "qwen3_5": (TINY_DENSE_FAMILIES["qwen3_5"], {"head_dim": 256}),
    "qwen3_vl": (TINY_QWEN3_VL_FAMILY, _HEAD_DIM_64),
}

PARSER = argparse.ArgumentParser()
PARSER.add_argument("--family", choices=sorted(FAMILIES), required=True)
PARSER.add_argument("--attn", choices=[FA2, FA4], required=True)
ARGS = PARSER.parse_args()


def _rotates_by_mrope(config) -> bool:
    """The text config's class lists ``mrope_section`` among the rope keys transformers' validation skips."""
    return "mrope_section" in (getattr(text_config(config), "ignore_keys_at_rope_validation", None) or ())


def _documents(generator: torch.Generator, vocab: int) -> list[list[int]]:
    low, high = DOCUMENT_LENGTHS
    lengths = torch.randint(low, high + 1, (DOCUMENTS_PER_ROW,), generator=generator).tolist()
    return [torch.randint(2, vocab, (n,), generator=generator).tolist() for n in lengths]


def _forward(model, batch: dict, device, **replace) -> torch.Tensor:
    inputs = {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}
    inputs.pop("labels", None)
    inputs.update(replace)
    with torch.no_grad():
        return model(use_cache=False, **inputs).logits[0].float()


def _per_document_rel(logits: torch.Tensor, alone: list[torch.Tensor]) -> list[float]:
    """Each document's worst logit gap to the document run alone, over that document's largest |logit|."""
    out, start = [], 0
    for lone in alone:
        end = start + lone.shape[0]
        out.append(((logits[start:end] - lone).abs().max() / lone.abs().max()).item())
        start = end
    return out


@gpu_test_main(exact_world_size=1, prefix=f"mrope_padding_free_{ARGS.family}_{ARGS.attn}", partial_state=False)
def run(ctx) -> dict:
    if ARGS.attn == FA4:
        skip_unless_blackwell(f"--family {ARGS.family} --attn {FA4}")
    family, overrides = FAMILIES[ARGS.family]
    tokenizer = word_level_tokenizer([f"w{i}" for i in range(VOCAB_WORDS)])
    ckpt = shared_tiny_family_checkpoint(
        ctx,
        family,
        f"{ARGS.family}_mrope_padding_free",
        tokenizer,
        SEED,
        edit=randomized_weights(),
        overrides=overrides,
    )
    model, _ = load_distributed_model(
        model_name_or_path=ckpt,
        parallelism_config=ParallelismConfig(max_concurrent_loading=0),
        dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation=ARGS.attn,
        use_liger_kernel=False,
        model_class=family.load_class,
    )
    model.eval()
    label = model.config._attn_implementation
    markers = segment_markers_for(model.config)
    log(
        f"=== {ARGS.family}: requested {ARGS.attn}, resolved {label}, M-RoPE decoder {_rotates_by_mrope(model.config)}"
    )

    flattening = DataCollatorWithFlattening(tokenizer=tokenizer, return_seq_idx=markers.seq_idx)
    packing = select_data_collator(
        tokenizer,
        train_on_completions_only=False,
        packing=True,
        model_config=model.config,
        emit_varlen_one_document=True,
    )
    generator = torch.Generator().manual_seed(SEED)
    packing_rel, padding_free_rel, stacked_rel, control_rel = [], [], [], []
    losses_are_finite = grads_are_finite = True
    with kernel_calls() as calls:
        for step in range(STEPS):
            documents = _documents(generator, len(tokenizer))
            alone = [_forward(model, {"input_ids": torch.tensor([doc])}, ctx.device) for doc in documents]
            flat = flattening([{"input_ids": doc, "labels": doc} for doc in documents])
            packed = packing(
                [
                    {
                        "input_ids": [t for doc in documents for t in doc],
                        "labels": [t for doc in documents for t in doc],
                        "seq_lengths": [len(doc) for doc in documents],
                    }
                ]
            )
            positions = flat["position_ids"]
            # Axis 0 is the text axis masks read; the other three rotate by temporal, height and width, equal for text.
            stacked = positions[None].expand(4, -1, -1).to(ctx.device)
            running = torch.arange(positions.shape[1], device=ctx.device)[None]

            packing_rel.append(max(_per_document_rel(_forward(model, packed, ctx.device), alone)))
            padding_free_rel.append(max(_per_document_rel(_forward(model, flat, ctx.device), alone)))
            stacked_rel.append(max(_per_document_rel(_forward(model, flat, ctx.device, position_ids=stacked), alone)))
            unmarked = {"input_ids": flat["input_ids"]}
            leaky = _forward(model, unmarked, ctx.device, position_ids=running)
            control_rel.append(max(_per_document_rel(leaky, alone)[1:]))

            model.train()
            model.zero_grad(set_to_none=True)
            inputs = {key: value.to(ctx.device) if torch.is_tensor(value) else value for key, value in flat.items()}
            loss = model(use_cache=False, **inputs).loss
            loss.backward()
            torch.cuda.synchronize()
            model.eval()
            losses_are_finite &= bool(torch.isfinite(loss))
            grads = [p.grad for p in model.parameters() if p.grad is not None]
            grads_are_finite &= bool(grads) and all(bool(torch.isfinite(g).all()) for g in grads)
            if step % 10 == 0:
                log(
                    f"  step {step}: packing {packing_rel[-1]:.2e} padding_free {padding_free_rel[-1]:.2e} control {control_rel[-1]:.2e}"
                )

    packing_floor = max(max(packing_rel), ISOLATED_REL_MAX)
    metrics = {
        "packing_rel": max(packing_rel),
        "padding_free_rel": max(padding_free_rel),
        "stacked_position_ids_rel": max(stacked_rel),
        "control_rel": min(control_rel),
    }
    log(f"  metrics {metrics}, kernels {dict(calls)}")
    checks = {
        "resolves_the_requested_backend": label == ARGS.attn and set(calls) == {ARGS.attn},
        "decoder_rotates_by_mrope": _rotates_by_mrope(model.config),
        "packing_matches_each_document_alone": metrics["packing_rel"] <= ISOLATED_REL_MAX,
        "padding_free_within_the_packing_floor": metrics["padding_free_rel"] <= packing_floor,
        "stacked_position_ids_within_the_packing_floor": metrics["stacked_position_ids_rel"] <= packing_floor,
        "control_without_per_document_positions_drifts": metrics["control_rel"] >= LEAK_REL_MIN,
        "every_loss_is_finite": losses_are_finite,
        "every_gradient_is_finite": grads_are_finite,
    }
    return {"checks": checks, "metrics": metrics}


if __name__ == "__main__":
    run()
