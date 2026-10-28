#!/usr/bin/env python
"""Save, exact resume and export of every PP trainer beyond SFT, through the real trainer at pp2 x dp2.

``--trainer`` picks SMPO, DPO, KTO, the Bradley-Terry reward trainer, classification, or offline GRPO
without (``offline_grpo``) or with (``offline_grpo_kl``) its KL term, each configured as the suite that pins
its pipeline loss configures it. The body is :func:`tests.common.resume_export.run_resume_export`: one run
trains ``TOTAL_STEPS`` from a saved tiny base, checkpointing every ``SAVE_STEP``; a trainer built from the
midway checkpoint, as the training scripts resume, trains the rest and must restore the optimizer bit for bit
and replay the unbroken run; the final checkpoint must load through the trainer's export class. The seams
each trainer adds to the SFT resume (``test_pp_resume.py``), each failing one of those checks when it breaks:

  * the pair and row layouts and per-step normalizers replay across the resume, which lands mid-epoch for
    every trainer here (``SAVE_STEP`` is coprime with each one's steps per epoch), and SMPO's scheduled margin
    follows the restored step;
  * the reward and classification score heads are written by the last stage, read back by the stage loader
    and served through ``AutoModelForSequenceClassification``;
  * offline GRPO's grouped sampler resumes on the rows the unbroken run trained;
  * with ``kl_beta > 0`` the run-start reference rides every checkpoint as ``reference_logps.pt``, and a
    trainer built from the trained checkpoint attaches it rather than scoring its own weights.

Topology: 4 GPUs as 2 simulated nodes of 2 (NVLink domain = 2), pp2 x dp2.

Run: torchrun --nproc_per_node=4 tests/gpu/parallelism/pp/test_pp_trainer_checkpoints.py --trainer smpo
"""

import argparse
import os
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial

import torch
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer

from halo.checkpoint.format import REFERENCE_LOGPS_FILE
from halo.data.collators.offline_grpo import REF_PER_TOKEN_LOGPS_COLUMN
from halo.distributed.parallelism_config import ParallelismConfig
from halo.trainers.grpo.offline.trainer import OfflineGRPOTrainer
from halo.trainers.preference.smpo import SMPOTrainer
from halo.trainers.reward.bradley_terry import DistributedRewardTrainer
from halo.trainers.reward.classification import ClassificationTrainer
from tests.common.distributed import shared_output_dir
from tests.common.harness import gpu_test_main
from tests.common.models import QWEN3_0_6B, TINY_QWEN3_CONFIG
from tests.common.pp_equivalence import build_dense_model
from tests.common.pp_trainers import build_seq_cls_model, tiny_vocab_tokenizer
from tests.common.resume_export import ResumeExportRoute, run_resume_export, save_base_checkpoint
from tests.gpu.parallelism.pp import test_pp_classification as classification
from tests.gpu.parallelism.pp import test_pp_offline_grpo as offline_grpo
from tests.gpu.parallelism.pp import test_pp_offline_grpo_kl as offline_grpo_kl
from tests.gpu.parallelism.pp import test_pp_preference_precompute as preference
from tests.gpu.parallelism.pp import test_pp_reward_bt as reward
from tests.gpu.parallelism.pp import test_pp_smpo as smpo

PP_SIZE = 2
# Coprime with every trainer's steps per epoch (2, 3 or 4 here), so the resume lands mid-epoch and the
# trainer's sampler has to skip to the unbroken run's next batch.
SAVE_STEP = 5
TOTAL_STEPS = 2 * SAVE_STEP


def no_resume_context(checkpoint: str | None) -> dict:
    """The constructor kwargs of a trainer the training scripts tell nothing about the resume."""
    return {}


def reference_resume_context(checkpoint: str | None) -> dict:
    """What the DPO and KTO scripts tell their trainer: the checkpoint, and that the policy was built from it."""
    return {"resume_checkpoint": checkpoint, "policy_from_checkpoint": checkpoint is not None}


def offline_grpo_resume_context(checkpoint: str | None) -> dict:
    """What the offline GRPO script tells its trainer: the checkpoint its policy was built from."""
    return {"resume_checkpoint": checkpoint}


def reference_columns(trainer) -> dict[str, torch.Tensor]:
    """The run-start reference log-probs each split carries, flattened in row order."""
    return {
        split: torch.tensor([value for row in dataset[REF_PER_TOKEN_LOGPS_COLUMN] for value in row])
        for split, dataset in (("training", trainer.train_dataset), ("evaluation", trainer.eval_dataset))
    }


@dataclass(frozen=True)
class Case:
    """One trainer's run, on the configuration of the suite that pins its pipeline loss.

    ``args(output_dir, *, max_steps, save_every)`` builds its training arguments and ``datasets()`` its dataset
    kwargs; ``resume_context(checkpoint)`` is what its training script passes the constructor about a resume.
    """

    trainer_cls: type
    tokenizer: Callable
    base_model: Callable
    export_class: type
    args: Callable
    datasets: Callable
    trainer_kwargs: dict = field(default_factory=dict)
    resume_context: Callable = no_resume_context
    sidecars: tuple[str, ...] = ()
    run_state: Callable | None = None


def _qwen_tokenizer():
    return AutoTokenizer.from_pretrained(QWEN3_0_6B)


def _preference_case(kind: str) -> Case:
    spec = preference.SPECS[kind]
    return Case(
        trainer_cls=spec.trainer_cls,
        tokenizer=_qwen_tokenizer,
        base_model=partial(build_dense_model, vocab_size=preference.VOCAB_SIZE),
        export_class=AutoModelForCausalLM,
        args=partial(preference.build_args, spec),
        datasets=lambda: {"train_dataset": preference.build_rows(spec, spec.train_examples)},
        resume_context=reference_resume_context,
    )


CASES = {
    "smpo": Case(
        trainer_cls=SMPOTrainer,
        tokenizer=tiny_vocab_tokenizer,
        base_model=lambda tokenizer: build_dense_model(),
        export_class=AutoModelForCausalLM,
        args=smpo.build_args,
        datasets=lambda: {"train_dataset": smpo.build_pref_dataset(TINY_QWEN3_CONFIG["vocab_size"])},
    ),
    "dpo": _preference_case("dpo"),
    "kto": _preference_case("kto"),
    "reward": Case(
        trainer_cls=DistributedRewardTrainer,
        tokenizer=tiny_vocab_tokenizer,
        base_model=lambda tokenizer: build_seq_cls_model(num_labels=1),
        export_class=AutoModelForSequenceClassification,
        args=reward.build_args,
        datasets=lambda: {"train_dataset": reward.build_dataset(TINY_QWEN3_CONFIG["vocab_size"], n_rows=32, seed=7)},
    ),
    "classification": Case(
        trainer_cls=ClassificationTrainer,
        tokenizer=tiny_vocab_tokenizer,
        base_model=lambda tokenizer: build_seq_cls_model(classification.NUM_LABELS),
        export_class=AutoModelForSequenceClassification,
        args=partial(classification.base_args, loss_type="cross_entropy", class_weights=classification.CLASS_WEIGHTS),
        datasets=lambda: {"train_dataset": classification.build_single_label_dataset(24, seed=7)},
        trainer_kwargs={"is_binary": False, "is_multi_label": False},
    ),
    "offline_grpo": Case(
        trainer_cls=OfflineGRPOTrainer,
        tokenizer=_qwen_tokenizer,
        base_model=partial(build_dense_model, vocab_size=offline_grpo.VOCAB_SIZE),
        export_class=AutoModelForCausalLM,
        args=offline_grpo.build_args,
        datasets=lambda: {"train_dataset": offline_grpo.build_dataset()},
        resume_context=offline_grpo_resume_context,
    ),
    "offline_grpo_kl": Case(
        trainer_cls=OfflineGRPOTrainer,
        tokenizer=_qwen_tokenizer,
        base_model=partial(build_dense_model, vocab_size=offline_grpo.VOCAB_SIZE),
        export_class=AutoModelForCausalLM,
        args=partial(offline_grpo.build_args, kl_beta=offline_grpo_kl.KL_BETA),
        # The evaluation split is scored at construction too, so the sidecar carries both splits.
        datasets=lambda: {
            "train_dataset": offline_grpo.build_dataset(sizes=offline_grpo_kl.TRAIN_GROUP_SIZES),
            "eval_dataset": offline_grpo.build_dataset(sizes=offline_grpo.EVAL_GROUP_SIZES, prompt_offset=100),
        },
        resume_context=offline_grpo_resume_context,
        sidecars=(REFERENCE_LOGPS_FILE,),
        run_state=reference_columns,
    ),
}
CASE = None  # CASES[--trainer], set before the harness runs


@gpu_test_main(exact_world_size=4, prefix="pp_trainer_checkpoints")
def run(ctx):
    config = ParallelismConfig(world_size=ctx.world_size, gpus_per_node=2, nvlink_domain_size=2, pp_size=PP_SIZE)
    tokenizer = CASE.tokenizer()
    # The PP checkpoint and offline GRPO's reference cache are multi-writer directories: one root for all ranks.
    root = shared_output_dir(ctx)
    base = save_base_checkpoint(ctx, CASE.base_model(tokenizer), tokenizer, os.path.join(root, "base"))

    def build_trainer(model, output_dir, *, max_steps, save_every, resume_checkpoint):
        return CASE.trainer_cls(
            model=model,
            args=CASE.args(output_dir, max_steps=max_steps, save_every=save_every),
            processing_class=tokenizer,
            parallelism_config=config,
            **CASE.datasets(),
            **CASE.trainer_kwargs,
            **CASE.resume_context(resume_checkpoint),
        )

    route = ResumeExportRoute(
        build_trainer,
        CASE.export_class,
        dtype=torch.float32,
        attn_implementation="sdpa",
        sidecars=CASE.sidecars,
        run_state=CASE.run_state,
    )
    checks, metrics = run_resume_export(
        ctx, route, config, base=base, root=root, save_step=SAVE_STEP, total_steps=TOTAL_STEPS
    )
    return {"checks": checks, "metrics": metrics}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trainer", required=True, choices=sorted(CASES))
    CASE = CASES[parser.parse_args().trainer]
    sys.exit(run())
