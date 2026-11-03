#!/usr/bin/env python
"""DPO's adapter-disabled reference under EP with native expert LoRA is the frozen base.

PEFT's ``disable_adapter()`` covers only the attention half: expert adapters drop out only because
``install_ep_aware_disable_adapter`` wraps that context. ``load_reference_model_for_preference`` must hand a mixed run
``None`` rather than refuse it for want of precomputed log-probs.
"""

import contextlib
import math
import random

import torch
from datasets import Dataset
from peft import PeftModel
from peft.tuners.lora import LoraLayer
from trl import DPOConfig
from trl.models.utils import disable_gradient_checkpointing
from trl.trainer.utils import use_adapter

from halo.configs.preference import DPOScriptArguments
from halo.distributed.expert_parallel.moe_layer.expert_lora import model_has_expert_lora
from halo.distributed.expert_parallel.moe_layer.layer import EPMoELayerBase
from halo.distributed.parallelism_config import ParallelismConfig
from halo.env import env_int, env_str
from halo.model_loading.parallel.frozen_models import load_reference_model_for_preference
from halo.trainers.preference.dpo import DistributedDPOTrainer
from tests.common.adapters import load_peft_model_from_config, peft_model_config
from tests.common.distributed import ensure_model_downloaded, world_all, world_min
from tests.common.harness import gpu_test_main
from tests.common.models import GPT_OSS_20B
from tests.common.rank_log import log
from tests.common.training_checks import step_losses

MODEL_NAME = env_str("HALO_TEST_MODEL", GPT_OSS_20B)
# The DPO script's padded default off Blackwell under reset sinks; the contract holds on every backend.
ATTN_IMPL = env_str("HALO_TEST_ATTN", "sdpa")
EP_SIZE = env_int("HALO_TEST_EP", 2)
MAX_STEPS = 3
LEARNING_RATE = 5e-4
NUM_PAIRS = 16
SEED = 42

# Nats; 3 steps move each half by tens of nats.
MIN_HALF_EFFECT = 0.5
# Nats, fp32, max over sequences. Same tokens and kernels, deltas adding exact zeros: measured 0.0. A reference keeping
# the expert half is off by that half's whole effect.
REFERENCE_VS_BASE_ABS = 1e-3


def _pairs(n: int) -> Dataset:
    rng = random.Random(SEED)
    rows = []
    for _ in range(n):
        a, b = rng.randint(1, 50), rng.randint(1, 50)
        rows.append(
            {
                "prompt": [{"role": "user", "content": f"What is {a} + {b}? Explain briefly."}],
                "chosen": [{"role": "assistant", "content": f"Adding {a} and {b} gives {a + b}."}],
                "rejected": [{"role": "assistant", "content": f"The answer is {a + b + rng.randint(1, 9)}."}],
            }
        )
    return Dataset.from_list(rows)


@contextlib.contextmanager
def _zero_adapter_scaling(model, *, attention: bool, experts: bool):
    """The chosen halves stay enabled but add exact zeros, independent of the disable seams under test."""
    peft_layers = [(m, dict(m.scaling)) for m in model.modules() if isinstance(m, LoraLayer)] if attention else []
    ep_layers = (
        [(m, m.expert_lora_scaling) for m in model.modules() if isinstance(m, EPMoELayerBase)] if experts else []
    )
    for layer, _ in peft_layers:
        layer.scaling.update(dict.fromkeys(layer.scaling, 0.0))
    for layer, _ in ep_layers:
        layer.expert_lora_scaling = 0.0
    try:
        yield
    finally:
        for layer, scaling in peft_layers:
            layer.scaling.update(scaling)
        for layer, scaling in ep_layers:
            layer.expert_lora_scaling = scaling


def _sequence_logps(trainer, batch, context) -> torch.Tensor:
    """In fp32, since rounding in TRL's bf16 sum would hide a sub-nat difference; checkpointing off, as TRL's reference
    forward."""
    with (
        torch.no_grad(),
        disable_gradient_checkpointing(trainer.model, trainer.args.gradient_checkpointing_kwargs),
        context,
    ):
        logits = trainer.model(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], use_cache=False
        ).logits.float()
    labels = batch["input_ids"][:, 1:].unsqueeze(-1)
    token_logps = logits[:, :-1].log_softmax(-1).gather(-1, labels).squeeze(-1)
    return (token_logps * batch["completion_mask"][:, 1:]).sum(-1).cpu()


def run(ctx) -> dict:
    ensure_model_downloaded(MODEL_NAME, ctx.rank)
    parallelism_config = ParallelismConfig(ep_size=EP_SIZE)
    model_config = peft_model_config("mixed", parallelism_config, model_name=MODEL_NAME)
    model, tokenizer, peft_config = load_peft_model_from_config(
        model_config, parallelism_config, attn_implementation=ATTN_IMPL, use_liger_kernel=False
    )
    config = DPOConfig(
        output_dir=ctx.output_dir,
        max_steps=MAX_STEPS,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="constant",
        beta=0.1,
        bf16=True,
        gradient_checkpointing=True,
        use_liger_kernel=False,
        precompute_ref_log_probs=False,
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        logging_nan_inf_filter=False,
        max_length=256,
        dataloader_drop_last=True,
        dataloader_num_workers=0,
    )
    # The seam the dpo/kto scripts build the reference through; its gate sees the peeled targets.
    ref_model = load_reference_model_for_preference(
        DPOScriptArguments(),
        model_config,
        config,
        parallelism_config,
        tokenizer,
        is_vlm=False,
        method="DPO",
        attn_default=ATTN_IMPL,
        reset_sinks=True,
    )
    trainer = DistributedDPOTrainer(
        model=model,
        ref_model=ref_model,
        args=config,
        train_dataset=_pairs(NUM_PAIRS),
        processing_class=tokenizer,
        peft_config=peft_config,
        parallelism_config=parallelism_config,
    )
    ctx.on_teardown(trainer.cleanup_ep)

    # The object TRL resolves its reference context on.
    peft_model = trainer.accelerator.unwrap_model(trainer.model)
    checks = {
        "run_is_mixed": isinstance(peft_model, PeftModel)
        and "ref" not in peft_model.peft_config
        and model_has_expert_lora(peft_model),
        "loader_leaves_the_reference_to_the_policy": ref_model is None,
    }

    trainer.train()
    losses = step_losses(trainer)
    checks["in_loop_reference_trains"] = len(losses) == MAX_STEPS and all(math.isfinite(x) for x in losses)
    log(f"  losses: {losses}")

    batch = {k: v.to(ctx.device) for k, v in next(iter(trainer.get_train_dataloader())).items() if torch.is_tensor(v)}
    policy = _sequence_logps(trainer, batch, contextlib.nullcontext())
    base = _sequence_logps(trainer, batch, _zero_adapter_scaling(peft_model, attention=True, experts=True))
    attention_only = _sequence_logps(trainer, batch, _zero_adapter_scaling(peft_model, attention=False, experts=True))
    experts_only = _sequence_logps(trainer, batch, _zero_adapter_scaling(peft_model, attention=True, experts=False))
    reference = _sequence_logps(trainer, batch, use_adapter(peft_model, adapter_name=None))

    expert_effect = (policy - attention_only).abs().max().item()
    attention_effect = (policy - experts_only).abs().max().item()
    reference_error = (reference - base).abs().max().item()
    log(
        f"  expert-half effect {expert_effect:.4f}, attention-half effect {attention_effect:.4f}, "
        f"reference vs base {reference_error:.2e} (tol {REFERENCE_VS_BASE_ABS})"
    )
    # Every rank scores its own rows, so the worst rank decides.
    checks["expert_half_moves_policy"] = world_min(expert_effect) > MIN_HALF_EFFECT
    checks["attention_half_moves_policy"] = world_min(attention_effect) > MIN_HALF_EFFECT
    checks["reference_is_frozen_base"] = world_all(reference_error <= REFERENCE_VS_BASE_ABS)
    return {"checks": checks, "metrics": ctx.metrics(trainer)}


main = gpu_test_main(min_world_size=2, prefix="pref_ep_expert_lora_reference")(run)

if __name__ == "__main__":
    main()
