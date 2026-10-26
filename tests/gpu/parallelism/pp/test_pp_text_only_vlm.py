#!/usr/bin/env python
"""A text-only run of a multimodal wrapper through the real DistributedSFTTrainer under pipeline parallelism.

The stage split keeps the text backbone and the task head; a composite wrapper's vision tower is a
sibling of ``language_model`` and lands in no stage. For a text-only run that is exactly right —
the tower is dead weight — provided (1) a run that does feed images is still refused, loudly, and
(2) the export stays a complete wrapper-layout checkpoint: the untouched vision tensors are stashed
at the split on the save rank and re-emitted by every PP save, so ``from_pretrained`` of the
composite class reloads with no missing keys and serves with no reattach step.

Pinned here on a tiny GLM-5 composite (``Glm5NextForConditionalGeneration`` — a family that ships
no text-only CausalLM sibling, so the wrapper is the only class there is):

  * the image-bearing refusal, on every rank, before any collective;
  * construction admits the text-only run; no stage carries a vision parameter; the stash holds
    exactly the vision tower (save rank) or nothing (every other rank);
  * the step-1 loss equals the unsplit composite's causal-LM loss on the captured batch;
  * training moves the text weights, losses are finite and rank-identical;
  * the save reloads as the composite class with no missing keys, its vision tensors bitwise equal
    to the init ones, its text tensors moved, and its eval loss reproducing the logged one.

Topology: 4 GPUs as 2 simulated nodes of 2, pp=2 x dp=2 (stage FSDP over 2 ranks).

Run: torchrun --nproc_per_node=4 tests/gpu/parallelism/pp/test_pp_text_only_vlm.py [--family glm5_next|step3p7]
(the same pins on the Step-3.7 composite, whose tower and projector sit under ``model.vision_model`` /
``model.multi_modal_projector``)
"""

import sys

import torch
import torch.distributed as dist
import torch.nn.functional as F
from transformers import (
    AutoTokenizer,
    Glm5NextConfig,
    Glm5NextForConditionalGeneration,
    Step3p7Config,
    Step3p7ForConditionalGeneration,
    default_data_collator,
)
from trl import SFTConfig

from src.distributed.parallelism_config import ParallelismConfig
from src.trainers.parallelism.pipeline_trainer import wrapper_state_outside_stages
from src.trainers.sft import DistributedSFTTrainer
from tests.common.harness import gpu_test_main
from tests.common.models import (
    QWEN3_0_6B,
    TINY_GLM5_CONFIG,
    TINY_GLM5_VISION_CONFIG,
    TINY_STEP3P7_CONFIG,
    TINY_STEP3P7_VISION_CONFIG,
)
from tests.common.utils import log
from tests.gpu.parallelism.pp.test_pp_sft_trainer_e2e import build_dataset

PP_SIZE = 2
SEQ_LEN = 64
N_STEPS = 8
BATCH = 4
PP_SPLIT = [4, 4]
# Eight layers so both layer kinds of each family's interleave land on both stages of the [4, 4]
# split (GLM-5's KDA/DSA, Step-3.7's full/sliding attention).
_FAMILIES = {
    "glm5_next": {
        "config_cls": Glm5NextConfig,
        "model_cls": Glm5NextForConditionalGeneration,
        "text": {**TINY_GLM5_CONFIG, "num_hidden_layers": 8, "mlp_layer_types": ["dense"] + ["sparse"] * 7},
        "vision": TINY_GLM5_VISION_CONFIG,
        "composite": {},
    },
    "step3p7": {
        "config_cls": Step3p7Config,
        "model_cls": Step3p7ForConditionalGeneration,
        "text": {
            **TINY_STEP3P7_CONFIG,
            "num_hidden_layers": 8,
            "layer_types": ["full_attention", "sliding_attention"] * 4,
            "mlp_layer_types": ["dense"] + ["sparse"] * 7,
            "swiglu_limits": [0, 0] + [0.5] * 6,
            "swiglu_limits_shared": [0, 0] + [0.4] * 6,
        },
        "vision": TINY_STEP3P7_VISION_CONFIG,
        "composite": {"image_token_id": 2000},
    },
}
FAMILY = "glm5_next"
if "--family" in sys.argv:
    _index = sys.argv.index("--family")
    FAMILY = sys.argv[_index + 1]
    del sys.argv[_index : _index + 2]
if FAMILY not in _FAMILIES:
    raise SystemExit(f"--family must be one of {sorted(_FAMILIES)}, got {FAMILY!r}")
TEXT_CONFIG = _FAMILIES[FAMILY]["text"]
# Step-1 loss vs the unsplit fp32 composite: the trainer runs bf16, the reference fp32.
LOSS_REL_TOL = 2e-2
# Reloaded eval vs the logged one: the run evaluates its fp32 masters under bf16 autocast, the artifact
# holds them rounded to bf16, and the precision a bf16 reload is evaluated in alone moves the eval of
# this tiny model by 3e-4 relative. A missing or misplaced tensor moves it by orders of magnitude more.
EVAL_REL_TOL = 1e-3


def build_composite(dtype=torch.float32):
    """Identical weights on every rank — the split model and the unsplit reference must start equal."""
    torch.manual_seed(1234)
    family = _FAMILIES[FAMILY]
    config = family["config_cls"](
        text_config=dict(TEXT_CONFIG),
        vision_config=dict(family["vision"]),
        attn_implementation="sdpa",
        **family["composite"],
    )
    return family["model_cls"](config).to(dtype)


def mean_ce(model, batch: dict) -> float:
    """Per-batch mean CE exactly as the pipeline computes it (sum CE / non-ignored count)."""
    with torch.no_grad():
        logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
    shift_logits, shift_labels = logits[:, :-1, :], batch["labels"][:, 1:]
    total = F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)).float(),
        shift_labels.reshape(-1),
        ignore_index=-100,
        reduction="sum",
    )
    return float(total / (shift_labels != -100).sum().clamp(min=1))


def make_args(output_dir: str, **overrides) -> SFTConfig:
    base = {
        "output_dir": output_dir,
        "max_steps": N_STEPS,
        "per_device_train_batch_size": BATCH,
        "per_device_eval_batch_size": BATCH,
        "gradient_accumulation_steps": 2,  # becomes the pipeline's microbatch count
        "learning_rate": 5e-3,
        "lr_scheduler_type": "constant",
        "max_length": SEQ_LEN,
        "logging_steps": 1,
        "save_strategy": "no",
        "report_to": [],
        "logging_nan_inf_filter": False,
        "seed": 42,
        "bf16": True,
        "eval_strategy": "steps",
        "eval_steps": N_STEPS,
    }
    base.update(overrides)
    return SFTConfig(**base)


@gpu_test_main(exact_world_size=4, prefix="pp_text_only_vlm")
def run(ctx):
    config = ParallelismConfig(
        world_size=ctx.world_size, gpus_per_node=2, nvlink_domain_size=2, pp_size=PP_SIZE, pp_split=PP_SPLIT
    )
    tokenizer = AutoTokenizer.from_pretrained(QWEN3_0_6B)
    vocab = TEXT_CONFIG["vocab_size"]
    train_dataset = build_dataset(vocab)
    eval_dataset = build_dataset(vocab, n_rows=8, seed=23)
    checks, metrics = {}, {}

    # ── An image-bearing run of the same wrapper stays refused, on every rank, before any collective.
    image_dataset = train_dataset.add_column("images", [["not-an-image"]] * len(train_dataset))
    try:
        DistributedSFTTrainer(
            model=build_composite(),
            args=make_args(ctx.output_dir + "/images"),
            train_dataset=image_dataset,
            processing_class=tokenizer,
            data_collator=default_data_collator,
            parallelism_config=config,
            moe_balancing="none",
        )
        checks["image_bearing_run_refused"] = False
    except ValueError as exc:
        checks["image_bearing_run_refused"] = "Vision-language training is not supported" in str(exc)
        if not checks["image_bearing_run_refused"]:
            log(f"[refusal] raised without the mechanism: {exc}")

    # ── The text-only run ─────────────────────────────────────────────────────────────────────────
    trainer = DistributedSFTTrainer(
        model=build_composite(),
        args=make_args(ctx.output_dir),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        data_collator=default_data_collator,
        parallelism_config=config,
        # The HF aux-loss path is severed by the split (a documented PP constraint on MoE families).
        moe_balancing="none",
    )
    stage = trainer.model
    init_vision = wrapper_state_outside_stages(build_composite())
    tower_roots = {key.split(".")[1] for key in init_vision}
    checks["no_stage_carries_a_vision_parameter"] = bool(tower_roots) and not any(
        root in name.split(".") for name, _ in stage.named_parameters() for root in tower_roots
    )
    stash = trainer._pp_wrapper_state
    # The stash holds the tower at the artifact's save dtype, so the oracle is the init tensor
    # rounded once to that dtype — a value drift (a stage touching the tower) still fails.
    checks["save_rank_stashes_exactly_the_vision_tower"] = (
        set(stash) == set(init_vision)
        and all(torch.equal(stash[k], init_vision[k].to(stash[k].dtype)) for k in init_vision)
        if ctx.rank == 0
        else stash == {}
    )

    captured: dict = {}
    real_step = trainer._pp_runner.step

    def spying_step(input_ids, labels, **kw):
        if "input_ids" not in captured:
            captured.update(
                {
                    "input_ids": input_ids.clone(),
                    "labels": labels.clone(),
                    "attention_mask": kw["attention_mask"].clone(),
                    "normalizer": float(kw["num_items_in_batch"]),
                }
            )
        return real_step(input_ids, labels, **kw)

    trainer._pp_runner.step = spying_step
    trainer.train()

    losses = [e["loss"] for e in trainer.state.log_history if "loss" in e]
    checks["ran_all_steps"] = trainer.state.global_step == N_STEPS and len(losses) == N_STEPS
    checks["losses_finite"] = bool(torch.isfinite(torch.tensor(losses)).all())
    checks["loss_decreased"] = losses[-1] < losses[0] - 0.05
    metrics["first_loss"], metrics["last_loss"] = losses[0], losses[-1]

    # Step 1 on init weights == the unsplit composite's token-mean CE over this replica's batch,
    # normalized like the pipeline (this replica's summed CE over the DP-global count / dp).
    reference_model = build_composite().to(ctx.device).eval()
    with torch.no_grad():
        logits = reference_model(input_ids=captured["input_ids"], attention_mask=captured["attention_mask"]).logits
    shift_logits, shift_labels = logits[:, :-1, :], captured["labels"][:, 1:]
    summed = F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)).float(),
        shift_labels.reshape(-1),
        ignore_index=-100,
        reduction="sum",
    )
    replica_loss = summed / captured["normalizer"]
    expected = replica_loss.clone()
    dist.all_reduce(expected, op=dist.ReduceOp.AVG)
    metrics["step1_loss"], metrics["step1_reference"] = losses[0], float(expected)
    checks["step1_loss_matches_unsplit_composite"] = abs(losses[0] - float(expected)) < LOSS_REL_TOL * float(expected)

    local = torch.tensor(losses, device=ctx.device)
    gathered = [torch.zeros_like(local) for _ in range(ctx.world_size)]
    dist.all_gather(gathered, local)
    checks["losses_identical_across_ranks"] = all(torch.allclose(local, g, atol=1e-6) for g in gathered)
    eval_losses = [e["eval_loss"] for e in trainer.state.log_history if "eval_loss" in e]
    checks["eval_ran_finite"] = len(eval_losses) == 1 and bool(torch.isfinite(torch.tensor(eval_losses)).all())

    # ── Save → the composite class reloads whole: nothing missing, vision untouched, text moved.
    dirs: list = [None] * ctx.world_size
    dist.all_gather_object(dirs, ctx.output_dir)
    save_dir = dirs[0]
    # TRL copied the full-size tokenizer's special-token ids into the tiny config; restore ids inside
    # the tiny vocab so from_pretrained reflects PP mechanics, not the test's size mismatch.
    text_config = trainer.model.config.get_text_config()
    text_config.pad_token_id, text_config.eos_token_id = 0, 1
    trainer.model.config.pad_token_id, trainer.model.config.eos_token_id = 0, 1
    trainer.save_model(save_dir)
    dist.barrier()
    reload_ok = True
    if ctx.rank == 0:
        try:
            reloaded, info = _FAMILIES[FAMILY]["model_cls"].from_pretrained(
                save_dir, dtype=torch.bfloat16, output_loading_info=True
            )
            if info["missing_keys"]:
                log(f"[reload] MISSING (random-initialized): {sorted(info['missing_keys'])[:8]}")
                reload_ok = False
            state = reloaded.state_dict()
            reference_keys = set(build_composite().state_dict())
            if set(state) != reference_keys:
                log(f"[reload] key set differs: {sorted(set(state) ^ reference_keys)[:8]}")
                reload_ok = False
            init_bf16 = build_composite(torch.bfloat16).state_dict()
            vision_intact = all(torch.equal(state[k], init_bf16[k]) for k in init_vision)
            text_keys = [k for k in reference_keys if k.startswith("model.language_model.layers.")]
            moved = sum(not torch.equal(state[k], init_bf16[k]) for k in text_keys)
            log(f"[reload] vision intact={vision_intact} text params moved: {moved}/{len(text_keys)}")
            if not vision_intact or moved < len(text_keys) * 0.9:
                reload_ok = False
            reloaded = reloaded.to(ctx.device).eval()
            eval_batches = [eval_dataset[i : i + BATCH] for i in (0, BATCH)]
            shard_means = [
                mean_ce(reloaded, {k: torch.tensor(v, device=ctx.device) for k, v in batch.items()})
                for batch in eval_batches
            ]
            reference_eval = sum(shard_means) / len(shard_means)
            log(f"[reload] reference eval={reference_eval:.6f} logged={eval_losses[-1]:.6f}")
            if abs(reference_eval - eval_losses[-1]) / abs(reference_eval) > EVAL_REL_TOL:
                reload_ok = False
        except Exception as exc:
            log(f"[reload] FAILED: {exc!r}")
            reload_ok = False
    flag = torch.tensor([1.0 if reload_ok else 0.0], device=ctx.device)
    dist.broadcast(flag, src=0)
    checks["save_reloads_as_the_complete_wrapper"] = bool(flag.item())

    return {"checks": checks, "metrics": metrics}


if __name__ == "__main__":
    sys.exit(run())
