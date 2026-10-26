#!/usr/bin/env python
"""Train, ``save_model`` and reload a full checkpoint, for each dense and MoE parallelism mode on 2 GPUs.

``--mode fsdp|tp2|cp2`` trains Qwen3-8B, ``--mode ep2|ep2_tp2|ep2_no_gmm|etp`` gpt-oss-20b. Every rank tears the
process group down before rank 0 reloads the save: reloading two full checkpoints outlasts the NCCL watchdog.
"""

import argparse
import os
from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig

from halo.distributed.parallelism_config import ParallelismConfig
from halo.model_loading.parallel.loader import load_distributed_model
from halo.trainers.sft import DistributedSFTTrainer
from tests.common.checkpoint_io import model_save_checks
from tests.common.datasets import create_sft_dataset
from tests.common.distributed import ensure_model_downloaded, shared_output_dir, teardown_distributed
from tests.common.harness import gpu_test_main
from tests.common.memory import cleanup_memory
from tests.common.models import GPT_OSS_20B, QWEN3_8B
from tests.common.parallel_shape import parallel_shape_checks
from tests.common.rank_log import log
from tests.common.training_checks import training_run_checks

MAX_STEPS = 3
# Per-element bound on the re-save round trip, which must be lossless.
WEIGHT_ATOL = 1e-6


@dataclass(frozen=True)
class RoundTripModel:
    """A checkpoint under test: its attention backend, the parameter count a whole save carries, and whether
    it has gpt-oss's 2D expert biases, which a save could drop and a reload would silently re-initialize."""

    name: str
    attn_implementation: str
    param_range: tuple[float, float]
    expert_biases: bool = False


QWEN3 = RoundTripModel(QWEN3_8B, "flash_attention_2", (7e9, 10e9))
# ~20.7B: the gathered save must carry every expert, not one EP rank's half.
GPT_OSS = RoundTripModel(GPT_OSS_20B, "flex_attention", (18e9, 23e9), expert_biases=True)

# mode -> (model, ParallelismConfig kwargs); no kwargs is plain FSDP2 over the stock transformers model.
MODES = {
    "fsdp": (QWEN3, {}),
    "tp2": (QWEN3, {"tp_size": 2}),
    "cp2": (QWEN3, {"cp_size": 2}),
    "ep2": (GPT_OSS, {"ep_size": 2}),
    "ep2_tp2": (GPT_OSS, {"ep_size": 2, "tp_size": 2}),
    # Off grouped_mm, the only route to the non-grouped expert-bias gather on SM90+.
    "ep2_no_gmm": (GPT_OSS, {"ep_size": 2, "use_grouped_gemm": False}),
    # Pure expert TP: the experts are reassembled across the ETP group before the HF-layout save.
    "etp": (GPT_OSS, {"ep_size": 1, "expert_tp_size": 2}),
}


def train_and_save(ctx, spec: RoundTripModel, parallelism: dict, tokenizer, save_dir: str) -> dict[str, bool]:
    """COLLECTIVE. Train ``MAX_STEPS`` and ``save_model`` to ``save_dir``; the shape and training checks."""
    parallelism_config = ParallelismConfig(**parallelism)
    if parallelism:
        model, _ = load_distributed_model(
            model_name_or_path=spec.name,
            parallelism_config=parallelism_config,
            dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation=spec.attn_implementation,
            use_liger_kernel=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            spec.name, dtype=torch.bfloat16, trust_remote_code=True, attn_implementation=spec.attn_implementation
        )
    trainer = DistributedSFTTrainer(
        model=model,
        args=SFTConfig(
            output_dir=os.path.join(ctx.output_dir, "trainer"),
            max_steps=MAX_STEPS,
            per_device_train_batch_size=1,
            learning_rate=2e-5,
            bf16=True,
            gradient_checkpointing=True,
            # The distributed loader already applied Liger; the stock model gets it from the trainer.
            use_liger_kernel=not parallelism,
            logging_steps=1,
            save_strategy="no",
            report_to="none",
            logging_nan_inf_filter=False,
            max_length=512,
            dataloader_drop_last=True,
            dataloader_num_workers=0,
        ),
        train_dataset=create_sft_dataset(16, tokenizer, seed=42),
        processing_class=tokenizer,
        parallelism_config=parallelism_config,
    )
    checks = parallel_shape_checks(trainer.model, parallelism_config)
    checks |= training_run_checks(trainer.train(), trainer, MAX_STEPS)
    trainer.save_model(save_dir)
    return checks


def inventory_checks(spec: RoundTripModel, model, state: dict[str, torch.Tensor]) -> dict[str, bool]:
    """The whole model's parameter count, no NaN/Inf anywhere, and gpt-oss's expert biases on every MoE layer."""
    num_params = sum(p.numel() for p in model.parameters())
    low, high = spec.param_range
    nan = [name for name, tensor in state.items() if torch.isnan(tensor).any()]
    inf = [name for name, tensor in state.items() if torch.isinf(tensor).any()]
    checks = {"param_count_reasonable": low < num_params < high, "no_nan_weights": not nan, "no_inf_weights": not inf}
    log(f"  {num_params:,} parameters in {len(state)} tensors; NaN in {nan[:5]}, Inf in {inf[:5]}")
    if spec.expert_biases:
        layers = {
            key.split(".")[2]
            for key in state
            if key.startswith("model.layers.") and key.endswith(".mlp.experts.gate_up_proj")
        }
        expected = [
            f"model.layers.{layer}.mlp.experts.{bias}"
            for layer in sorted(layers, key=int)
            for bias in ("gate_up_proj_bias", "down_proj_bias")
        ]
        missing = [key for key in expected if key not in state]
        checks["expert_biases_present"] = bool(layers) and not missing
        log(f"  expert biases over {len(layers)} MoE layers, missing {missing[:3]}")
    return checks


def weights_changed(model_name: str, state: dict[str, torch.Tensor]) -> bool:
    """Whether training moved any parameter off the pretrained checkpoint."""
    pretrained = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16, trust_remote_code=True)
    reference = pretrained.state_dict()
    compared = [key for key in state if key in reference and state[key].shape == reference[key].shape]
    deltas = {key: (state[key].float() - reference[key].float()).abs().max().item() for key in compared}
    changed = [key for key, delta in deltas.items() if delta > 1e-8]
    log(f"  changed vs pretrained: {len(changed)}/{len(compared)}, largest {max(deltas.values(), default=0.0):.2e}")
    return bool(changed)


def forward_checks(model, tokenizer, device) -> dict[str, bool]:
    """A forward of the reloaded model gives logits of the input's shape, finite and in a sane range."""
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": "What is 2 + 2?"}], tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(device)
    model.to(device).eval()
    with torch.no_grad():
        logits = model(**inputs).logits
    model.cpu()
    checks = {
        "forward_shape": logits.shape[:2] == inputs["input_ids"].shape,
        "forward_no_nan": not torch.isnan(logits).any().item(),
        "forward_no_inf": not torch.isinf(logits).any().item(),
        "forward_range": logits.abs().max().item() < 1000,
    }
    log(f"  logits {tuple(logits.shape)} in [{logits.min().item():.2f}, {logits.max().item():.2f}]")
    return checks


def resave_is_lossless(model, state: dict[str, torch.Tensor], roundtrip_dir: str) -> bool:
    """Whether ``save_pretrained`` of the reloaded model reads back to the same tensors."""
    model.save_pretrained(roundtrip_dir, safe_serialization=True)
    again = AutoModelForCausalLM.from_pretrained(roundtrip_dir, dtype=torch.bfloat16, trust_remote_code=True)
    resaved = again.state_dict()
    mismatched = [
        key
        for key, tensor in state.items()
        if key not in resaved
        or resaved[key].shape != tensor.shape
        or (resaved[key].float() - tensor.float()).abs().max().item() > WEIGHT_ATOL
    ]
    log(f"  re-save mismatches: {len(mismatched)}/{len(state)} {mismatched[:3]}")
    return not mismatched


def verify_checkpoint(ctx, spec: RoundTripModel, tokenizer, save_dir: str) -> dict[str, bool]:
    """Rank 0, no collective: what the save holds, read back through a stock ``from_pretrained`` on CPU."""
    checks = model_save_checks(save_dir, ctx.rank)
    if not all(checks.values()):
        return checks
    reloaded = AutoModelForCausalLM.from_pretrained(save_dir, dtype=torch.bfloat16, trust_remote_code=True)
    state = reloaded.state_dict()
    checks |= inventory_checks(spec, reloaded, state)
    checks["weights_changed"] = weights_changed(spec.name, state)
    checks |= forward_checks(reloaded, tokenizer, ctx.device)
    checks["roundtrip_lossless"] = resave_is_lossless(reloaded, state, os.path.join(ctx.output_dir, "roundtrip"))
    return checks


def run(ctx) -> dict:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=MODES)
    mode = parser.parse_args().mode
    spec, parallelism = MODES[mode]
    log(f"  Checkpoint round trip: {mode} on {spec.name}, world size {ctx.world_size}")

    ensure_model_downloaded(spec.name, ctx.rank)
    tokenizer = AutoTokenizer.from_pretrained(spec.name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    save_dir = os.path.join(shared_output_dir(ctx), "checkpoint")
    checks = train_and_save(ctx, spec, parallelism, tokenizer, save_dir)
    cleanup_memory()
    # Down on every rank before the rank-0 reload, so no peer sits in a collective through it. Teardown's
    # own collective is the rendezvous, and the harness's closing barrier and teardown then skip.
    teardown_distributed()
    if ctx.rank == 0 and all(checks.values()):
        checks |= verify_checkpoint(ctx, spec, tokenizer, save_dir)
    return {"checks": checks}


main = gpu_test_main(min_world_size=2, prefix="checkpoint_roundtrip")(run)

if __name__ == "__main__":
    main()
