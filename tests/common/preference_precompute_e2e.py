"""The DPO / KTO ``precompute_ref_log_probs`` resume body the precompute-resume GPU suites run.

TRL computes the reference log-probs inside ``__init__`` over ``self.ref_model or self.model``, and a
Path-B resume (EP, TP, ETP, and FSDP2 under the default ``use_grouped_gemm``) builds the policy from
the checkpoint before the trainer exists. A sweep there scores the trained weights, so the reference
equals the policy and every log-ratio is zero. Each checkpoint therefore carries the untrained
columns (``reference_logps.pt``), per split, and a trainer handed the resume checkpoint at
construction attaches them instead of sweeping. A LoRA resume builds the policy from the base, so
there the sweep scores untrained weights and stays correct. The policy is a family's tiny model
(:data:`~tests.common.tiny_models.TINY_MOE_FAMILIES`, or the dense
:data:`~tests.common.tiny_models.TINY_DENSE_FAMILY`) on a two-rank layout (``MODES``), and the
train split rides with two named eval splits, which TRL precomputes under their ``eval_dataset`` keys.

One checkpoint, four phases, the resume taken the production way (the policy source from
``resolve_resume_weights_source``, the trainer told the checkpoint and whether the policy came from it):

  1. **Continuous**: ``TOTAL_STEPS`` from the base, checkpointing at ``SAVE_AT_STEP``.
  2. **Resume**: every split's reference columns equal phase 1's bit for bit, and the first resumed
     step's loss equals the continuous run's at that step within ``TOL.replayed_resume_loss_abs``.
  3. **Control**: the same resume with the trainer not told. Where the policy came from the
     checkpoint its sweep scores the trained weights, and its columns must miss phase 1's by more
     than ``CONTROL_MIN_LOGP_DELTA`` and its first-step loss by more than
     ``TOL.control_min_loss_shift`` of the resume bound, which is what gives phase 2 its teeth. Where
     it came from the base the sweep must reproduce phase 1's columns instead.
  4. **Without the sidecar**: the checkpoint copied without ``reference_logps.pt`` refuses on every
     rank where the policy came from it, and sweeps back phase 1's columns where it came from the base.
"""

import argparse
import math
import os
import shutil
from collections.abc import Iterable
from types import SimpleNamespace

import torch
from datasets import Dataset, load_from_disk
from transformers import AutoTokenizer

from src.checkpoint.format import REFERENCE_LOGPS_FILE
from src.distributed.loading.model_loading import load_distributed_model
from src.distributed.parallelism_config import ParallelismConfig
from src.training.environment import resolve_resume_weights_source
from tests.common.distributed import shared_scratch_dir
from tests.common.models import QWEN3_0_6B
from tests.common.peft_helpers import load_peft_model
from tests.common.preference_precompute import TRAINERS, column
from tests.common.tiny_models import TINY_DENSE_FAMILY, TINY_MOE_FAMILIES, shared_tiny_family_checkpoint
from tests.common.tolerances import TOL
from tests.common.utils import cleanup_memory, finish_phase, log

# The layouts: plain FSDP2 DP (dp2, the dense model's), plain per-rank experts (ep2), FSDP-sharded
# DTensor experts over DP=2 (ep1), attention and MLP sharded over the TP mesh (tp2), the expert FFN
# sharded (etp2), per-rank experts under attention TP (ep2tp2), and per-rank experts each split two ways
# (ep2etp2, an expert group of four).
MODES = {
    "dp2": {},
    "ep2": {"ep_size": 2},
    "ep1": {"ep_size": 1},
    "tp2": {"tp_size": 2},
    "etp2": {"ep_size": 1, "expert_tp_size": 2},
    "ep2tp2": {"ep_size": 2, "tp_size": 2},
    "ep2etp2": {"ep_size": 2, "expert_tp_size": 2},
}
DENSE = "dense"
# KTO's own loss computes a KL term from mismatched completions; ``apo_zero_unpaired`` has none.
KTO_LOSSES = ("kto", "apo_zero_unpaired")
SEED = 42
N_ROWS = 16
# Named eval splits and their sizes; TRL precomputes each under its dict key.
EVAL_SPLITS = {"alpha": 6, "beta": 4}
BATCH_SIZE = 2
TOTAL_STEPS = 4
SAVE_AT_STEP = 2
MAX_LENGTH = 96
# Large enough that two steps move the policy's log-probs well off the reference, so a resume that
# re-derived the reference from the trained weights lands visibly elsewhere.
LEARNING_RATE = 2e-3
# The control's sweep of the trained policy misses the base run's reference columns by at least 1 nat
# on every family and layout, so half of that separates it from a restore, which is bit for bit.
CONTROL_MIN_LOGP_DELTA = 0.5


def world_size(mode: str) -> int:
    """Ranks ``mode`` runs on: its expert group, never fewer than two."""
    layout = MODES[mode]
    return max(2, layout.get("ep_size", 1) * layout.get("expert_tp_size", 1))


def precompute_resume_parser(families: Iterable[str], *, world: int = 2) -> argparse.ArgumentParser:
    """The CLI a precompute-resume suite takes, over the ``families`` and the layouts of ``world`` ranks it runs."""
    modes = sorted(mode for mode in MODES if world_size(mode) == world)
    parser = argparse.ArgumentParser()
    parser.add_argument("--trainer", choices=sorted(TRAINERS), default="dpo")
    parser.add_argument("--family", choices=sorted(families), required=True)
    parser.add_argument("--mode", choices=modes, default="ep2" if "ep2" in modes else modes[0])
    parser.add_argument("--peft", action="store_true")
    parser.add_argument("--kto-loss", choices=KTO_LOSSES, default="kto")
    return parser


def _build_rows(kind: str, n: int, start: int = 0) -> dict:
    """Text rows with ragged completions, so rows (and a swapped reference) score differently;
    ``start`` offsets the questions, so each split holds its own rows."""
    ids = range(start, start + n)
    prompts = [f"Question {i}: what is {i} plus {i}?" for i in ids]
    answers = [" The answer is " + " ".join(str(2 * i + j) for j in range(1 + i % 5)) + "." for i in ids]
    if kind == "dpo":
        rejected = [" Perhaps " + " ".join(str(3 * i + j) for j in range(1 + (i + 3) % 5)) + " ok." for i in ids]
        return {"prompt": prompts, "chosen": answers, "rejected": rejected}
    return {"prompt": prompts, "completion": answers, "label": [i % 2 == 0 for i in ids]}


def _training_args(kind: str, output_dir: str, *, save_at: int | None, kto_loss: str):
    loss = {"loss_type": kto_loss} if kind == "kto" else {}
    return TRAINERS[kind][1](
        output_dir=output_dir,
        max_steps=TOTAL_STEPS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        eval_strategy="no",
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="constant",
        logging_steps=1,
        logging_nan_inf_filter=False,
        save_strategy="steps" if save_at else "no",
        save_steps=save_at or 0,
        report_to=[],
        seed=SEED,
        # Phase 2 replays phase 1 exactly: DeepEP dispatch and the per-expert loop's scatter both order
        # their sums by atomics unless deterministic algorithms are on.
        full_determinism=True,
        bf16=True,
        gradient_checkpointing=False,
        precompute_ref_log_probs=True,
        dataloader_num_workers=0,
        ddp_find_unused_parameters=True,  # EP leaves inactive experts gradient-free
        fsdp="",  # the mixin owns FSDP wrapping
        max_length=MAX_LENGTH,
        **loss,
    )


def _reference_columns(trainer) -> dict[str, dict[str, torch.Tensor]]:
    """Each split's attached reference columns, by split name then column."""
    splits = {"train": trainer.train_dataset, **trainer.eval_dataset}
    return {
        name: {key: column(dataset, key) for key in trainer._required_ref_logps_columns()}
        for name, dataset in splits.items()
    }


def _precomputed(trainer, columns: dict[str, dict[str, torch.Tensor]]) -> bool:
    """Whether each split's :func:`_reference_columns` are what a sweep writes: one finite, negative
    sequence log-prob per row."""
    splits = {"train": trainer.train_dataset, **trainer.eval_dataset}
    return all(
        bool(columns[name])
        and all(
            len(values) == len(dataset) and bool(torch.isfinite(values).all() and (values < 0).all())
            for values in columns[name].values()
        )
        for name, dataset in splits.items()
    )


def _losses_by_step(trainer) -> dict[int, float]:
    return {entry["step"]: entry["loss"] for entry in trainer.state.log_history if "loss" in entry}


def _split_deltas(actual: dict, reference: dict) -> dict[str, float]:
    """Per split, the largest |difference| over its reference columns; a split missing from
    ``actual`` counts as infinitely far."""
    return {
        name: max(float((actual[name][key] - values).abs().max()) for key, values in columns.items())
        if name in actual
        else math.inf
        for name, columns in reference.items()
    }


def _columns_equal(actual: dict, reference: dict) -> bool:
    return sorted(actual) == sorted(reference) and all(
        torch.equal(actual[name][key], values)
        for name, columns in reference.items()
        for key, values in columns.items()
    )


def run_precompute_resume(ctx, *, trainer: str, family: str, mode: str, peft: bool, kto_loss: str) -> dict:
    """The four phases above for one trainer, family and layout; returns the harness result.

    ``family`` is a :data:`TINY_MOE_FAMILIES` key or ``"dense"``; ``peft`` trains attention LoRA over
    a frozen base, whose disabled adapters are the reference.
    """
    kind = trainer
    tiny = TINY_DENSE_FAMILY if family == DENSE else TINY_MOE_FAMILIES[family]
    checks: dict[str, bool] = {}
    metrics: dict[str, float] = {}
    label = f"{kind}_{family}_{mode}{'_lora' if peft else ''}{'_' + kto_loss if kind == 'kto' else ''}"
    shared = shared_scratch_dir(f"pref_precompute_resume_{label}")
    tiny_dir = shared_tiny_family_checkpoint(
        ctx, tiny, f"pref_precompute_resume_{label}_tiny_model", AutoTokenizer.from_pretrained(QWEN3_0_6B), SEED
    )
    dataset_dirs = {name: os.path.join(shared, f"dataset_{name}") for name in ("train", *EVAL_SPLITS)}
    train_out = os.path.join(shared, "train_out")
    ckpt_dir = os.path.join(train_out, f"checkpoint-{SAVE_AT_STEP}")
    bare_ckpt_dir = os.path.join(shared, "checkpoint_without_sidecar")
    if ctx.rank == 0:
        # A standalone rerun gets the same MASTER_PORT-keyed dir; phase 4 copies into a fixed path.
        shutil.rmtree(shared, ignore_errors=True)
        ctx.on_teardown(lambda: shutil.rmtree(shared, ignore_errors=True))
        # Built once for every rank and phase, so each phase tokenizes the same rows.
        Dataset.from_dict(_build_rows(kind, N_ROWS)).save_to_disk(dataset_dirs["train"])
        start = N_ROWS
        for name, n in EVAL_SPLITS.items():
            Dataset.from_dict(_build_rows(kind, n, start)).save_to_disk(dataset_dirs[name])
            start += n
    ctx.barrier()
    pc = ParallelismConfig(**MODES[mode])
    log(f"  {label}: {pc.mode_string or 'dp'}, data_parallel_size={pc.data_parallel_size}")
    first_resumed_step = SAVE_AT_STEP + 1

    def make_trainer(weights_dir: str, output_dir: str, *, save_at=None, **resume_context):
        """The trainer over a policy loaded from ``weights_dir`` the way the entry scripts load it, on
        the ``train`` split and the named eval splits."""
        if peft:
            model, tokenizer, peft_config = load_peft_model(
                "lora", pc, model_name=weights_dir, attn_implementation="sdpa", use_liger_kernel=False
            )
        else:
            model, tokenizer = load_distributed_model(
                model_name_or_path=weights_dir,
                parallelism_config=pc,
                dtype=torch.bfloat16,
                attn_implementation="sdpa",
                use_liger_kernel=False,
                trust_remote_code=tiny.trust_remote_code,
            )
            peft_config = None
        built = TRAINERS[kind][0](
            model=model,
            args=_training_args(kind, output_dir, save_at=save_at, kto_loss=kto_loss),
            train_dataset=load_from_disk(dataset_dirs["train"]),
            eval_dataset={name: load_from_disk(dataset_dirs[name]) for name in EVAL_SPLITS},
            processing_class=tokenizer,
            parallelism_config=pc,
            peft_config=peft_config,
            **resume_context,
        )
        ctx.on_teardown(built.cleanup_ep)
        return built

    def resume_context(checkpoint: str) -> tuple[str, dict]:
        """Where the entry scripts load the policy from on this resume, and what they tell the trainer."""
        source = resolve_resume_weights_source(checkpoint, SimpleNamespace(model_name_or_path=tiny_dir), pc)
        return source, {"resume_checkpoint": checkpoint, "policy_from_checkpoint": source == checkpoint}

    log(f"\n--- Phase 1 ({label}): continuous {TOTAL_STEPS} steps, checkpoint at {SAVE_AT_STEP} ---")
    built = make_trainer(tiny_dir, train_out, save_at=SAVE_AT_STEP)
    base_columns = _reference_columns(built)
    checks["every_split_was_precomputed"] = set(base_columns) == {"train", *EVAL_SPLITS} and _precomputed(
        built, base_columns
    )
    built.train()
    continuous = _losses_by_step(built)
    finish_phase(built)
    ctx.barrier()
    checks["continuous_ran_all_steps"] = sorted(continuous) == list(range(1, TOTAL_STEPS + 1))
    checks["checkpoint_carries_the_sidecar"] = os.path.isfile(os.path.join(ckpt_dir, REFERENCE_LOGPS_FILE))
    if not (checks["continuous_ran_all_steps"] and checks["checkpoint_carries_the_sidecar"]):
        return {"checks": checks, "metrics": metrics}
    metrics["continuous_first_resumed_step_loss"] = continuous[first_resumed_step]

    source, told = resume_context(ckpt_dir)
    from_checkpoint = told["policy_from_checkpoint"]
    checks["policy_source_matches_the_layout"] = from_checkpoint is not peft
    log(f"\n--- Phase 2 ({label}): resume from checkpoint-{SAVE_AT_STEP}, policy from {source} ---")
    built = make_trainer(source, os.path.join(shared, "resume_out"), **told)
    resumed_columns = _reference_columns(built)
    built.train(resume_from_checkpoint=ckpt_dir)
    resumed = _losses_by_step(built)
    finish_phase(built)
    ctx.barrier()
    checks["resumed_reference_columns_equal_the_base_run"] = _columns_equal(resumed_columns, base_columns)
    resumed_delta = abs(resumed.get(first_resumed_step, math.inf) - continuous[first_resumed_step])
    metrics["resumed_first_step_loss"] = resumed.get(first_resumed_step, math.nan)
    metrics["resumed_first_step_loss_delta"] = resumed_delta
    checks["resumed_first_step_loss_matches_continuous"] = resumed_delta < TOL.replayed_resume_loss_abs

    log(f"\n--- Phase 3 ({label}): the same resume, trainer NOT told (control) ---")
    built = make_trainer(source, os.path.join(shared, "control_out"))
    control_columns = _reference_columns(built)
    built.train(resume_from_checkpoint=ckpt_dir)
    control = _losses_by_step(built)
    finish_phase(built)
    ctx.barrier()
    control_deltas = _split_deltas(control_columns, base_columns)
    # The smallest split's miss: every split, eval ones included, is re-swept off the policy.
    metrics["control_reference_max_logp_delta"] = min(control_deltas.values())
    control_delta = abs(control.get(first_resumed_step, math.inf) - continuous[first_resumed_step])
    metrics["control_first_step_loss_delta"] = control_delta
    if from_checkpoint:
        checks["control_reference_misses_the_base_run"] = (
            metrics["control_reference_max_logp_delta"] > CONTROL_MIN_LOGP_DELTA
        )
        checks["control_first_step_loss_misses_continuous"] = math.isfinite(control_delta) and (
            control_delta > TOL.control_min_loss_shift(TOL.replayed_resume_loss_abs)
        )
    else:
        checks["control_sweep_of_the_base_reproduces_the_base_run"] = _columns_equal(control_columns, base_columns)

    log(f"\n--- Phase 4 ({label}): resume from a copy without {REFERENCE_LOGPS_FILE} ---")
    if ctx.rank == 0:
        shutil.copytree(ckpt_dir, bare_ckpt_dir, ignore=shutil.ignore_patterns(REFERENCE_LOGPS_FILE))
    ctx.barrier()
    bare_source, bare_told = resume_context(bare_ckpt_dir)
    try:
        built = make_trainer(bare_source, os.path.join(shared, "refusal_out"), **bare_told)
        refusal = None
    except RuntimeError as exc:
        built, refusal = None, str(exc)
    if from_checkpoint:
        checks["resume_without_the_sidecar_refuses"] = (
            refusal is not None and "TRAINED weights as the reference" in refusal
        )
    else:
        checks["resume_without_the_sidecar_sweeps_the_base"] = built is not None and _columns_equal(
            _reference_columns(built), base_columns
        )
    if built is not None:
        finish_phase(built)
    cleanup_memory()

    log(
        f"continuous step-{first_resumed_step} loss {continuous[first_resumed_step]:.6f}; resumed "
        f"{metrics['resumed_first_step_loss']:.6f} (|delta| {resumed_delta:.2e}); control |delta| "
        f"{control_delta:.2e}; control reference max |delta logp| {metrics['control_reference_max_logp_delta']:.3f}"
    )
    return {"checks": checks, "metrics": metrics}
