#!/usr/bin/env python
"""Native dense hybrid CP2 SFT: evaluate, save, exact resume and native-class export on a local tiny checkpoint."""

import argparse
import math
import os

import torch
from datasets import Dataset
from transformers import DataCollatorForSeq2Seq, Qwen3_5ForCausalLM, TrainerCallback
from trl import SFTConfig

from halo.distributed.parallelism_config import ParallelismConfig
from halo.trainers.sft import DistributedSFTTrainer
from tests.common.checkpoint_io import save_schedule
from tests.common.distributed import shared_output_dir
from tests.common.efficiency_observation import ObservedEfficiencyCallback
from tests.common.harness import gpu_test_main
from tests.common.hybrid_cp import build_hybrid_model
from tests.common.offline_grpo import make_offline_tokenizer
from tests.common.resume_export import ResumeExportRoute, run_resume_export, save_base_checkpoint


class EvaluationRecorder(TrainerCallback):
    def __init__(self):
        self.losses = []
        self.steps = []

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        self.losses.append(metrics["eval_loss"])
        self.steps.append(state.global_step)


def _dataset(vocab_size, rows):
    records = []
    for row in range(rows):
        tokens = [3 + (row + token) % (vocab_size - 3) for token in range(18 + row % 3)]
        records.append({"input_ids": tokens, "attention_mask": [1] * len(tokens), "labels": [-100] * 10 + tokens[10:]})
    return Dataset.from_list(records)


def run(ctx):
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, choices=(1, 3), default=1)
    parser.add_argument("--gc", action="store_true")
    args = parser.parse_args()
    ctx.broadcast_seed(114)
    parallelism = ParallelismConfig(cp_size=2)
    tokenizer = make_offline_tokenizer()
    tokenizer.padding_side = "right"
    root = shared_output_dir(ctx)
    base = save_base_checkpoint(
        ctx,
        build_hybrid_model(
            vocab_size=len(tokenizer),
            hidden_size=128,
            intermediate_size=256,
            head_dim=32,
            linear_key_head_dim=32,
            linear_value_head_dim=32,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        ).to(torch.bfloat16),
        tokenizer,
        os.path.join(root, "base"),
    )
    training, evaluation = _dataset(len(tokenizer), 18), _dataset(len(tokenizer), 6)
    observed = []
    efficiencies = []

    def build_trainer(model, output_dir, *, max_steps, save_every, resume_checkpoint):
        recorder = EvaluationRecorder()
        observed.append(recorder)
        # The resumed arm has two steps; one warms the restored token-count baseline.
        efficiency = ObservedEfficiencyCallback(parallelism, n_warmup_steps=1)
        efficiencies.append(efficiency)
        return DistributedSFTTrainer(
            model=model,
            args=SFTConfig(
                output_dir=output_dir,
                max_steps=max_steps,
                **save_schedule(save_every),
                per_device_train_batch_size=args.batch_size,
                per_device_eval_batch_size=args.batch_size,
                learning_rate=1e-3,
                bf16=True,
                gradient_checkpointing=args.gc,
                gradient_checkpointing_kwargs={"use_reentrant": True},
                logging_steps=1,
                include_num_input_tokens_seen="all",
                logging_nan_inf_filter=False,
                report_to=[],
                eval_strategy="steps",
                eval_steps=1,
                max_length=32,
                packing=False,
                padding_free=False,
                dataset_kwargs={"skip_prepare_dataset": True},
                dataloader_num_workers=0,
                dataloader_drop_last=True,
                seed=114,
                data_seed=114,
                resume_from_checkpoint=resume_checkpoint,
            ),
            train_dataset=training,
            eval_dataset=evaluation,
            processing_class=tokenizer,
            data_collator=DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=2),
            parallelism_config=parallelism,
            callbacks=[recorder, efficiency],
        )

    route = ResumeExportRoute(
        build_trainer, Qwen3_5ForCausalLM, torch.bfloat16, "flash_attention_2", trust_remote_code=False
    )
    checks, metrics = run_resume_export(ctx, route, parallelism, base=base, root=root, save_step=2, total_steps=4)
    checks["unbroken_and_resumed_eval_finite"] = len(observed) == 2 and all(
        recorder.losses and all(math.isfinite(loss) for loss in recorder.losses) for recorder in observed
    )
    checks["unbroken_and_resumed_eval_steps"] = [recorder.steps for recorder in observed] == [[1, 2, 3, 4], [3, 4]]
    checks["efficiency_phases_recorded"] = len(efficiencies) == 2
    for phase, efficiency, expected_steps in zip(("unbroken", "resumed"), efficiencies, (3, 1), strict=False):
        checks |= {f"{phase}/{name}": passed for name, passed in efficiency.measurement_checks(expected_steps).items()}
        snapshot = ctx.metrics(efficiency)
        checks[f"{phase}/efficiency_metrics_recorded"] = (
            math.isfinite(snapshot["tokens_per_second"])
            and snapshot["tokens_per_second"] > 0
            and math.isfinite(snapshot["peak_allocated_gb"])
            and snapshot["peak_allocated_gb"] >= 0
        )
        metrics |= {f"{phase}_{name}": value for name, value in snapshot.items()}
        metrics |= {
            f"{phase}_measured_steps": len(efficiency.measured_step_times),
            f"{phase}_token_estimate_calls": efficiency.token_estimate_calls,
            f"{phase}_memory_readings": len(efficiency.memory_readings),
        }
        if phase == "resumed":
            metrics |= snapshot
    return {"checks": checks, "metrics": metrics}


main = gpu_test_main(exact_world_size=2, prefix="sft_hybrid_cp")(run)

if __name__ == "__main__":
    main()
