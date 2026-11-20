#!/usr/bin/env python
"""Quantization-aware distillation: a ``lowp_precision`` student fit to its own checkpoint as a frozen bf16 teacher.

The student loads through the SFT loader, which fake-quantizes its expert GEMMs; the teacher loads through
``load_frozen_auxiliary_model``, which never does. So at step 0 the two differ only by the quantization, and the KL
between them is what the run trains down. At this size they start almost on top of each other (KL about 1e-5), and the
first Adam step moves every bf16 weight by about the learning rate whatever its gradient, so it knocks the student off
the teacher. The distillation check is that the steps after it pull the student back. ``--ep 1`` puts the experts under FSDP2, which pins their version counter,
so the weight round trip runs uncached; ``--ep 2`` keeps them EP-sharded, where it is cached per optimizer step and the
cached entry must quantize the current weight. ``--expert_lora`` trains native expert adapters over frozen
fake-quantized experts. ``--chunked`` scores the loss with ``use_chunked_distillation_loss``, which projects both
models' output embeddings from their last hidden states. Every row pins the first step's logged ``distillation_loss``
to the KL of both models' full logits on that batch, so a chunked row matches its full-logits row through the same
reference, and a chunked student forward that skipped the fake-quant would log a KL near 0.
"""

import argparse
import math
import os

import torch
import torch.nn.functional as F
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback

from halo.configs.distillation import TeacherDistillationConfig
from halo.data.collators.padded import DataCollatorForCausalLMWithPadding
from halo.distributed.parallelism_config import ParallelismConfig
from halo.kernels import expert_gemm
from halo.kernels.logprobs import LABEL_IGNORE_INDEX
from halo.kernels.lowp import quantization
from halo.kernels.lowp.spec import LOWP_RECIPES, LowpSpec
from halo.model_loading.parallel.frozen_models import load_frozen_auxiliary_model
from halo.model_loading.parallel.loader import load_distributed_model
from halo.model_loading.parallel.lowp_compute import lowp_converted_modules
from halo.models.moe.expert_lora_spec import ExpertLoraSpec
from halo.trainers.distillation.divergences import logits_forward_inputs
from halo.trainers.distillation.teacher_distillation import DistributedTeacherDistillationTrainer
from tests.common.adapters import freeze_base_keep_expert_adapters
from tests.common.datasets import create_sft_dataset
from tests.common.distributed import pin_deterministic_ep_dispatch, world_all, world_mean
from tests.common.ep_reference import ep_layers
from tests.common.harness import gpu_test_main
from tests.common.models import QWEN3_0_6B
from tests.common.rank_log import log
from tests.common.tiny_models import TINY_MOE_FAMILIES, shared_tiny_family_checkpoint
from tests.common.tolerances import TOL
from tests.common.training_checks import step_losses

# Numerically identical to the compiled path, without a per-rank compile warm-up.
quantization._compile_failed = True

SEED = 42
TOTAL_STEPS = 6
MAX_LENGTH = 128
NUM_SAMPLES = 64
LEARNING_RATE = 1e-3
LORA_R, LORA_ALPHA = 8, 16
# The teacher must match a plain bf16 forward far more closely than the fake-quant student does: a teacher that
# picked up the student's quantization would sit as far from it as the student.
TEACHER_TO_STUDENT_DIFF_RATIO = 0.05

parser = argparse.ArgumentParser()
parser.add_argument("--family", choices=("qwen3_moe", "lfm2_moe"), required=True)
parser.add_argument("--precision", choices=("fp4", "mxfp4"), required=True)
parser.add_argument("--ep", type=int, choices=(1, 2), required=True)
parser.add_argument("--expert_lora", action="store_true", help="Native expert-LoRA adapters over frozen experts")
parser.add_argument("--chunked", action="store_true", help="use_chunked_distillation_loss: no [tokens, vocab] logits")
ARGS = parser.parse_args()
QUANT_FORMAT = LOWP_RECIPES[ARGS.precision].quant_format
# Expert weights are stored ``[E, K, N]``.
CONTRACTION_AXIS = 1


def _tokenized_rows(tokenizer) -> Dataset:
    rows = []
    for text in create_sft_dataset(NUM_SAMPLES, tokenizer, seed=SEED)["text"]:
        ids = tokenizer(text, truncation=True, max_length=MAX_LENGTH)["input_ids"]
        rows.append({"input_ids": ids, "attention_mask": [1] * len(ids), "labels": list(ids)})
    return Dataset.from_list(rows)


def _count_calls(module, name: str) -> list[int]:
    """Replace ``module.name`` for the process with a wrapper counting its calls."""
    real, calls = getattr(module, name), [0]

    def counted(*args, **kwargs):
        calls[0] += 1
        return real(*args, **kwargs)

    setattr(module, name, counted)
    return calls


class _CacheIsCurrent(TrainerCallback):
    """Before each optimizer write, whatever the step's forwards cached for an expert weight must be the quantization of
    that weight as it is now. A rank with no token for an expert leaves no entry, so ``compared`` keeps the check from
    passing on nothing. Base expert weights only: ``expert_named_params`` also returns the LoRA adapters, which the
    round trip never quantizes and which train under ``--expert_lora``."""

    def __init__(self, model):
        self.weights = [param for layer in ep_layers(model) for _, param in layer._base_expert_named_params()]
        self.compared = 0
        self.stale = 0

    @torch.no_grad()
    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        for weight in self.weights:
            entry = quantization._weight_dequant_cache.get(weight, (weight._version, QUANT_FORMAT, CONTRACTION_AXIS))
            if entry is None:
                continue
            self.compared += 1
            fresh = quantization._block_round_trip(weight, QUANT_FORMAT, CONTRACTION_AXIS).to(weight.dtype)
            self.stale += not torch.equal(entry, fresh)


@torch.no_grad()
def _probe(model, batch) -> torch.Tensor:
    """Fp32 logits of one fixed batch; every rank runs it, since the EP and FSDP2 forwards are collective."""
    was_training = model.training
    model.eval()
    logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits.float()
    model.train(was_training)
    return logits


def _kl(teacher_logits, student_logits, mask) -> float:
    """Token-mean ``KL(teacher || student)`` over the batch's real tokens."""
    per_token = F.kl_div(
        F.log_softmax(student_logits, dim=-1), F.log_softmax(teacher_logits, dim=-1), log_target=True, reduction="none"
    ).sum(-1)
    return float((per_token * mask).sum() / mask.sum())


class _KlAfterFirstStep(TrainerCallback):
    """The probe KL right after the first optimizer step, the point the remaining steps must improve on (see the module
    docstring). Every rank runs the probe, as it is collective."""

    def __init__(self, model, teacher_logits, batch, mask):
        self.model, self.teacher_logits, self.batch, self.mask = model, teacher_logits, batch, mask
        self.kl = None

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step == 1:
            self.kl = _kl(self.teacher_logits, _probe(self.model, self.batch), self.mask)


@torch.no_grad()
def _full_logits_kl(trainer, inputs) -> float:
    """``KL(teacher || student)`` at temperature 1, the token mean over the batch's supervised positions, from both
    models' full logits: what the trainer's first ``distillation_loss`` must equal on either loss path. The student
    runs in train mode, as the trainer's forward does. Collective, so every rank runs it."""
    forward_inputs = logits_forward_inputs(inputs)
    student_logits = trainer.model(**forward_inputs).logits[..., :-1, :].float()
    teacher_logits = trainer.teacher_model(**forward_inputs).logits[..., :-1, :].float()
    supervised = (inputs["labels"][..., 1:] != LABEL_IGNORE_INDEX).float()
    return _kl(teacher_logits, student_logits, supervised)


def _adapter_params(model) -> dict[str, torch.Tensor]:
    return {
        f"{index}.{attr}_lora_{which}": getattr(layer, f"{attr}_lora_{which}")
        for index, layer in enumerate(ep_layers(model))
        for attr in layer._expert_lora_attrs
        for which in ("A", "B")
    }


def _local_snapshot(params) -> list[torch.Tensor]:
    return [(p.to_local() if hasattr(p, "to_local") else p).detach().float().clone() for p in params]


@gpu_test_main(
    exact_world_size=2,
    prefix=f"lowp_teacher_distill_{ARGS.family}_{ARGS.precision}_ep{ARGS.ep}"
    f"{'_expert_lora' if ARGS.expert_lora else ''}{'_chunked' if ARGS.chunked else ''}",
)
def run(ctx):
    torch.cuda.set_device(ctx.device)
    pin_deterministic_ep_dispatch()
    checks: dict[str, bool] = {}
    tokenizer = AutoTokenizer.from_pretrained(QWEN3_0_6B)
    checkpoint = shared_tiny_family_checkpoint(
        ctx, TINY_MOE_FAMILIES[ARGS.family], f"lowp_teacher_distill_{ARGS.family}", tokenizer, SEED
    )

    pc = ParallelismConfig(ep_size=ARGS.ep, lowp=LowpSpec(precision=ARGS.precision))
    if ARGS.expert_lora:
        # Before the load, so the EP layers build their adapters at init.
        pc.expert_lora = ExpertLoraSpec(r=LORA_R, alpha=LORA_ALPHA, projections=frozenset({"gate", "up", "down"}))
    student, _ = load_distributed_model(
        checkpoint, pc, torch.bfloat16, trust_remote_code=False, attn_implementation="eager", use_liger_kernel=False
    )
    if ARGS.expert_lora:
        freeze_base_keep_expert_adapters(student)
    teacher = load_frozen_auxiliary_model(
        checkpoint, dtype=torch.bfloat16, trust_remote_code=False, attn_implementation="eager", download_tag=None
    )

    layers = ep_layers(student)
    expect_cacheable = not pc.experts_fsdp_managed
    checks["setup_matches_the_row"] = (ARGS.ep == 1) == pc.experts_fsdp_managed
    checks["student_experts_carry_the_format"] = bool(layers) and all(
        layer._lowp_quant_format == QUANT_FORMAT and layer._lowp_weight_cacheable is expect_cacheable
        for layer in layers
    )
    checks["teacher_holds_no_converted_module"] = lowp_converted_modules(teacher) == []

    dataset = _tokenized_rows(tokenizer)
    collator = DataCollatorForCausalLMWithPadding(tokenizer=tokenizer, mlm=False)
    trainer = DistributedTeacherDistillationTrainer(
        student_model=student,
        teacher_model=teacher,
        teacher_tokenizer=tokenizer,
        args=TeacherDistillationConfig(
            output_dir=os.path.join(ctx.output_dir, "run"),
            max_steps=TOTAL_STEPS,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=1,
            learning_rate=LEARNING_RATE,
            lr_scheduler_type="constant",
            max_length=MAX_LENGTH,
            bf16=True,
            gradient_checkpointing=False,
            use_liger_kernel=False,
            logging_steps=1,
            save_strategy="no",
            eval_strategy="no",
            report_to=[],
            logging_nan_inf_filter=False,
            dataloader_drop_last=True,
            dataloader_num_workers=0,
            dataset_num_proc=1,
            seed=SEED,
            distill_loss="kl_divergence",
            distill_alpha=1.0,
            use_chunked_distillation_loss=ARGS.chunked,
        ),
        train_dataset=dataset,
        data_collator=collator,
        processing_class=tokenizer,
        parallelism_config=pc,
    )
    ctx.on_teardown(trainer.cleanup_ep)

    # One fixed batch every rank feeds to every probe, from the rows the run trains on.
    batch = {k: v.to(ctx.device) for k, v in collator([dataset[i] for i in range(4)]).items()}
    mask = batch["attention_mask"].float()
    plain = AutoModelForCausalLM.from_pretrained(checkpoint, dtype=torch.bfloat16, attn_implementation="eager")
    plain_logits = _probe(plain.to(ctx.device), batch)
    del plain
    teacher_logits = _probe(trainer.teacher_model, batch)
    student_before = _probe(trainer.model, batch)
    teacher_diff = float((teacher_logits - plain_logits).abs().max())
    student_diff = float((student_before - plain_logits).abs().max())
    checks["teacher_matches_a_plain_bf16_forward"] = world_all(
        student_diff > 0.0 and teacher_diff <= TEACHER_TO_STUDENT_DIFF_RATIO * student_diff
    )
    kl_before = _kl(teacher_logits, student_before, mask)
    checks["step0_kl_is_the_quantization"] = world_all(kl_before > 0.0 and math.isfinite(kl_before))

    cache_calls = _count_calls(expert_gemm, "cached_fake_quant")
    uncached_calls = _count_calls(expert_gemm, "fake_quant")
    cache_check = _CacheIsCurrent(trainer.model)
    trainer.add_callback(cache_check)
    first_step = _KlAfterFirstStep(trainer.model, teacher_logits, batch, mask)
    trainer.add_callback(first_step)
    first_batch_kl: list[float] = []
    real_compute_loss = trainer.compute_loss

    def spying_compute_loss(model, inputs, *args, **kwargs):
        # The first step's batch, before the trainer's own forward and any optimizer write.
        if not first_batch_kl:
            first_batch_kl.append(_full_logits_kl(trainer, inputs))
        return real_compute_loss(model, inputs, *args, **kwargs)

    trainer.compute_loss = spying_compute_loss
    trainable = [p for p in trainer.model.parameters() if p.requires_grad]
    adapters_before = _local_snapshot(_adapter_params(trainer.model).values()) if ARGS.expert_lora else []
    experts_before = _local_snapshot(cache_check.weights)

    trainer.train()

    losses = step_losses(trainer)
    student_after = _probe(trainer.model, batch)
    kl_after = _kl(teacher_logits, student_after, mask)
    checks["ran_every_step_finite"] = len(losses) == TOTAL_STEPS and all(math.isfinite(loss) for loss in losses)
    kl_step1 = first_step.kl
    # The logged metric is the world mean of each rank's token mean, so the reference is reduced the same way.
    logged_step0 = next((e.get("distillation_loss") for e in trainer.state.log_history if "loss" in e), None)
    reference_step0 = world_mean(first_batch_kl[0], ctx.device) if first_batch_kl else float("nan")
    checks["step0_loss_is_the_full_logits_kl"] = (
        logged_step0 is not None
        and reference_step0 > 0.0
        and abs(logged_step0 - reference_step0) <= TOL.exact_objective_rel * reference_step0
    )
    checks["kl_fell_after_the_first_step"] = world_all(kl_step1 is not None and kl_after < kl_step1)
    # The weight round trip took the path the layout requires on every rank.
    if expect_cacheable:
        checks["expert_weights_round_trip_cached"] = world_all(cache_calls[0] > 0)
        checks["cached_entries_were_compared"] = world_all(cache_check.compared > 0)
        checks["cached_entries_quantize_the_current_weight"] = world_all(cache_check.stale == 0)
    else:
        checks["expert_weights_round_trip_uncached"] = world_all(cache_calls[0] == 0 and uncached_calls[0] > 0)
    experts_after = _local_snapshot(cache_check.weights)
    experts_moved = any(not torch.equal(a, b) for a, b in zip(experts_before, experts_after, strict=True))
    if ARGS.expert_lora:
        adapters_after = _local_snapshot(_adapter_params(trainer.model).values())
        adapter_ids = {id(a) for a in _adapter_params(trainer.model).values()}
        checks["only_adapters_train"] = bool(adapter_ids) and {id(p) for p in trainable} == adapter_ids
        checks["adapters_moved"] = world_all(
            any(not torch.equal(a, b) for a, b in zip(adapters_before, adapters_after, strict=True))
        )
        checks["frozen_experts_unchanged"] = world_all(not experts_moved)
    else:
        checks["expert_masters_moved"] = world_all(experts_moved)

    step1_text = "missing" if kl_step1 is None else f"{kl_step1:.3g}"
    log(
        f"  {ARGS.family} {ARGS.precision} ep{ARGS.ep}{' expert_lora' if ARGS.expert_lora else ''}"
        f"{' chunked' if ARGS.chunked else ''}: logged step-0 distillation_loss {logged_step0!r}, full-logits KL "
        f"{reference_step0:.6g}; KL step 0 {kl_before:.3g}, step 1 {step1_text}, step {TOTAL_STEPS} {kl_after:.3g}; max |logit - bf16|: teacher {teacher_diff:.4g}, student "
        f"{student_diff:.4g}; losses {losses}; cached calls {cache_calls[0]}, fake_quant calls {uncached_calls[0]}, "
        f"cache entries compared {cache_check.compared}, stale {cache_check.stale}"
    )
    return {
        "checks": checks,
        "metrics": {
            "kl_before": kl_before,
            "logged_step0_distillation_loss": float("nan") if logged_step0 is None else logged_step0,
            "full_logits_step0_kl": reference_step0,
            "kl_step1": kl_step1,
            "kl_after": kl_after,
            "teacher_logit_diff": teacher_diff,
            "student_logit_diff": student_diff,
        },
    }


if __name__ == "__main__":
    run()
