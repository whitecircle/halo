#!/usr/bin/env python
"""Every TP-capable family's tensor-parallel step against the unsharded model, loss and every gradient.

The dense families take transformers' own TP plan through the loader (Llama, Qwen2, Mistral, Gemma 2's
softcapped attention, Gemma 3's 256-wide heads with q/k norms, Cohere 2 and OLMo 3's sliding layers, Qwen3.5's
gated DeltaNet beside attention); the MoE families take the toolkit's attention plan
(``TP_SHARDABLE_ATTENTION_CLASSES``), alone (``--mode tp2``) or beside EP2 (``--mode ep2tp2``). A plan that
shards a projection the wrong way, drops the q/k-norm gradient sum or sums a replicated gradient twice still
trains to finite losses, so each run is held to the unsharded model of the same checkpoint on the same batch:

  * the loss of one ``DistributedSFTTrainer.training_step`` equals the reference's within
    ``TOL.parallel_vs_baseline_loss_abs``;
  * after the trainer's clip (where the TP replicated-gradient sync and the EP sweep run), every parameter's
    gradient, TP shards gathered and EP experts compared on this rank's slice, matches the reference's in
    direction and norm (:func:`~tests.common.ep_reference.score_ep_grad_pairs`), or, under
    ``TOL.grad_negligible_layer_fraction`` of its layer's largest (Qwen3.5's GatedDeltaNet ``A_log`` / ``dt_bias``
    on a layer whose heads all decay hard, which bf16 leaves no direction even unsharded), in absolute error under
    that floor, while every parameter kind keeps a layer on the direction and norm scores;
  * every decoder layer holding an attention module carries TP-sharded parameters, so a run that left a layer
    replicated cannot pass on the others.

Run: torchrun --nproc_per_node=2 tests/gpu/parallelism/tp/test_tp_family_vs_reference.py --family gemma3_text --mode tp2
"""

import argparse
import os

import torch
from torch.distributed.tensor import DTensor
from transformers import AutoTokenizer
from trl import SFTConfig

from halo.distributed.expert_parallel.moe_layer.layer import named_ep_layers
from halo.distributed.parallelism_config import ParallelismConfig
from halo.model_loading.parallel.loader import load_distributed_model
from halo.trainers.sft import DistributedSFTTrainer
from tests.common.datasets import create_sft_dataset
from tests.common.distributed import pin_deterministic_ep_dispatch
from tests.common.ep_reference import decoder_layer_index, per_layer_grad_floors, score_ep_grad_pairs
from tests.common.harness import gpu_test_main
from tests.common.models import QWEN3_0_6B
from tests.common.parallel_shape import parallel_shape_checks, tp_sharded_param_names
from tests.common.rank_log import log
from tests.common.tiny_models import (
    TINY_DENSE_ATTENTION_FAMILIES,
    TINY_DENSE_FAMILIES,
    TINY_MOE_FAMILIES,
    TINY_SLIDING_DENSE_FAMILIES,
    shared_tiny_family_checkpoint,
)
from tests.common.tolerances import TOL

SEED = 42
ROWS = 2
SEQ_LEN = 64
# Far above any norm here: the clip runs its syncs and scales by exactly 1.
NO_CLIP = 1e9
# The MoE families whose attention the toolkit's TP plan shards; the tiny-family roster test holds this to
# TP_SHARDABLE_ATTENTION_CLASSES.
TP_MOE_FAMILIES = ("cohere2_moe", "glm4_moe_lite", "gpt_oss", "lfm2_moe", "mistral4", "qwen3_5_moe_text", "qwen3_moe")
DENSE_FAMILIES = {
    **TINY_DENSE_ATTENTION_FAMILIES,
    **TINY_SLIDING_DENSE_FAMILIES,
    "qwen3_5": TINY_DENSE_FAMILIES["qwen3_5"],
}
FAMILIES = {**DENSE_FAMILIES, **{name: TINY_MOE_FAMILIES[name] for name in TP_MOE_FAMILIES}}
MODES = {"tp2": {"tp_size": 2}, "ep2tp2": {"ep_size": 2, "tp_size": 2}}
_ATTENTION_ATTRS = ("self_attn", "attention", "attn")

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--family", choices=sorted(FAMILIES), required=True)
parser.add_argument("--mode", choices=sorted(MODES), default="tp2")
ARGS = parser.parse_args()
if ARGS.mode == "ep2tp2" and ARGS.family not in TP_MOE_FAMILIES:
    parser.error("--mode ep2tp2 distributes experts: a MoE family")


def _full(tensor: torch.Tensor) -> torch.Tensor:
    """COLLECTIVE for a DTensor."""
    return (tensor.full_tensor() if isinstance(tensor, DTensor) else tensor).detach().float()


def _batch(vocab: int, device) -> dict[str, torch.Tensor]:
    """One fixed batch every rank holds (TP and EP+TP keep one DP replica on two ranks)."""
    generator = torch.Generator().manual_seed(SEED)
    ids = torch.randint(4, min(vocab, 1024), (ROWS, SEQ_LEN), generator=generator).to(device)
    return {"input_ids": ids, "attention_mask": torch.ones_like(ids), "labels": ids.clone()}


def _expert_slices(model) -> dict[str, slice]:
    """Each EP-sharded expert parameter's name and the expert range this rank holds."""
    names = {id(param): name for name, param in model.named_parameters()}
    return {
        names[id(param)]: slice(layer.expert_start, layer.expert_end)
        for _, layer in named_ep_layers(model)
        if layer.ep_size > 1
        for _, param in layer.expert_named_params()
    }


def _attention_layers_sharded(model) -> tuple[bool, list[int]]:
    """Whether every decoder layer holding an attention module carries TP-sharded parameters; the unsharded ones."""
    layers = next((module for name, module in model.named_modules() if name.endswith("layers") and len(module)), None)
    attention = {index for index, layer in enumerate(layers or ()) if any(hasattr(layer, a) for a in _ATTENTION_ATTRS)}
    sharded = {decoder_layer_index(name) for name in tp_sharded_param_names(model)} - {None}
    missing = sorted(attention - sharded)
    return bool(attention) and not missing, missing


@gpu_test_main(exact_world_size=2, prefix=f"tp_family_vs_reference_{ARGS.family}_{ARGS.mode}")
def run(ctx):
    torch.cuda.set_device(ctx.device)
    if ARGS.mode == "ep2tp2":
        pin_deterministic_ep_dispatch()
    family = FAMILIES[ARGS.family]
    tokenizer = AutoTokenizer.from_pretrained(QWEN3_0_6B)
    checkpoint = shared_tiny_family_checkpoint(ctx, family, f"tp_reference_{ARGS.family}", tokenizer, SEED)
    pc = ParallelismConfig(**MODES[ARGS.mode])
    log(f"{ARGS.family} {ARGS.mode}: {pc.mode_string}")
    load = {"trust_remote_code": family.trust_remote_code, "attn_implementation": "eager", "use_liger_kernel": False}
    model, tokenizer = load_distributed_model(checkpoint, pc, torch.bfloat16, model_class=family.load_class, **load)
    # Unwrapped: the same checkpoint on the same EP layers (a MoE) with no TP and no EP group.
    reference, _ = load_distributed_model(
        checkpoint, ParallelismConfig(), torch.bfloat16, model_class=family.load_class, **load
    )
    trainer = DistributedSFTTrainer(
        model=model,
        args=SFTConfig(
            output_dir=os.path.join(ctx.output_dir, "out"),
            max_steps=1,
            per_device_train_batch_size=ROWS,
            max_grad_norm=NO_CLIP,
            bf16=True,
            gradient_checkpointing=False,
            use_liger_kernel=False,
            save_strategy="no",
            report_to="none",
            logging_nan_inf_filter=False,
            average_tokens_across_devices=False,
        ),
        train_dataset=create_sft_dataset(8, tokenizer, seed=SEED),
        processing_class=tokenizer,
        parallelism_config=pc,
    )
    ctx.on_teardown(trainer.cleanup_ep)
    policy = trainer.model
    checks = parallel_shape_checks(policy, pc)
    checks["every_attention_layer_tp_sharded"], missing = _attention_layers_sharded(policy)
    if missing:
        log(f"  attention layers without a TP-sharded parameter: {missing}")

    batch = _batch(policy.get_input_embeddings().num_embeddings, ctx.device)
    trainer.current_gradient_accumulation_steps = 1
    trainer.accelerator.gradient_state._set_sync_gradients(True)
    loss = float(trainer.training_step(policy, batch))
    trainer.accelerator.clip_grad_norm_(policy.parameters(), NO_CLIP)
    trainable = [(name, param) for name, param in policy.named_parameters() if param.requires_grad]
    grads = {name: None if param.grad is None else _full(param.grad) for name, param in trainable}

    reference.train()
    reference_loss = reference(**batch).loss
    reference_loss.backward()
    reference_grads = {name: param.grad for name, param in reference.named_parameters() if param.grad is not None}
    slices = _expert_slices(policy)
    pairs = {
        name: (grad, reference_grads[name][slices[name]] if name in slices else reference_grads[name])
        for name, grad in grads.items()
        if name in reference_grads and reference_grads[name].any()
    }
    checks["every_reference_gradient_has_a_parallel_one"] = set(reference_grads) <= set(grads)
    checks["grads_compared"] = len(pairs) > 0
    pair_metrics = {}
    floors = per_layer_grad_floors(pairs, TOL.grad_negligible_layer_fraction)
    score_ep_grad_pairs(pairs, checks, pair_metrics, cos_min=TOL.ep_grad_cosine_min, negligible_floors=floors)
    negligible = sum(f"{name}_negligible_abs_error" in pair_metrics for name in pairs)
    loss_error = abs(loss - float(reference_loss))
    checks["loss_matches_unsharded_reference"] = loss_error < TOL.parallel_vs_baseline_loss_abs
    log(
        f"  loss {loss:.6f} vs unsharded {float(reference_loss):.6f} (|Δ| {loss_error:.2e}); {len(pairs)} grads "
        f"compared, {negligible} under their layer's floor"
    )
    metrics = {
        "loss_abs_error": loss_error,
        "grads_compared": float(len(pairs)),
        "grads_negligible": float(negligible),
    }
    return {"checks": checks, "metrics": metrics}


if __name__ == "__main__":
    run()
