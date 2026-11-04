#!/usr/bin/env python
"""DPO / KTO reference-log-prob precompute under EP (ep2, tiny Qwen3-MoE), on rank-private dataset directories.

The only way to run a full-finetune DPO/KTO under EP. TRL hands columns over through an Arrow cache beside the dataset,
which fails wherever ranks don't share that directory, so ``TRLPreferenceMixin`` runs the sweep itself. A rank failing
on a missing file leaves its peers in the next collective, so the manifest timeout is the assertion; keep it tight.
"""

import argparse
import os
import sys

import torch
import torch.distributed as dist
from datasets import Dataset, load_from_disk
from transformers import AutoTokenizer, Qwen3MoeConfig, Qwen3MoeForCausalLM

from halo.distributed.expert_parallel.parallelize import apply_ep_layers
from halo.distributed.parallelism_config import ParallelismConfig
from tests.common.harness import gpu_test_main
from tests.common.models import QWEN3_0_6B, TINY_QWEN3_MOE_CONFIG
from tests.common.preference_precompute import TRAINERS, column
from tests.common.rank_log import log
from tests.common.training_checks import step_losses, training_run_checks

EP_SIZE = 2
N_ROWS = 16
N_STEPS = 4
BATCH_SIZE = 2
MAX_LENGTH = 96

# The real Qwen tokenizer tokenizes the text rows, so the tiny model must span its full id range.
VOCAB_SIZE = 151936

# bf16 kernel and reduction-order noise against the dense reference: measured max relative error 3.1e-3 on sums of
# ~-1e2. A swapped or permuted row moves the sum by ~1.0 relative, two orders above this.
LOGP_RTOL = 1e-2
CONTROL_MIN_RTOL = 0.05


def build_model() -> Qwen3MoeForCausalLM:
    torch.manual_seed(1234)
    config = Qwen3MoeConfig(
        **{**TINY_QWEN3_MOE_CONFIG, "vocab_size": VOCAB_SIZE},
        pad_token_id=0,
        eos_token_id=1,
    )
    return Qwen3MoeForCausalLM(config).to(torch.bfloat16)


def build_rows(kind: str) -> dict:
    """Ragged lengths give the swap control its teeth: equal-length completions on an untrained model sum to nearly the
    same."""
    rows: dict[str, list] = {"prompt": []}
    for i in range(N_ROWS):
        rows["prompt"].append(f"Question {i}: what is {i} plus {i}?")
    if kind == "dpo":
        rows["chosen"], rows["rejected"] = [], []
        for i in range(N_ROWS):
            rows["chosen"].append(" The answer is " + " ".join(str(2 * i + j) for j in range(1 + i % 5)) + ".")
            rows["rejected"].append(" Perhaps " + " ".join(str(3 * i + j) for j in range(1 + (i + 3) % 5)) + " ok.")
    else:
        rows["completion"], rows["label"] = [], []
        for i in range(N_ROWS):
            rows["completion"].append(" The answer is " + " ".join(str(2 * i + j) for j in range(1 + i % 5)) + ".")
            rows["label"].append(i % 2 == 0)
    return rows


def rank_private_dataset(kind: str, ctx, subdir: str) -> Dataset:
    """From this rank's own arrow directory, as per-node storage leaves it, so another rank's cache file is
    invisible."""
    path = os.path.join(ctx.output_dir, subdir, f"{kind}_dataset")
    Dataset.from_dict(build_rows(kind)).save_to_disk(path)
    return load_from_disk(path)


def sequence_logprobs(model, prompt_ids: list[list[int]], completion_ids: list[list[int]], device) -> torch.Tensor:
    """TRL's reference log-prob per row on a dense model, independent of ``halo/``. Both TRL collators right-pad, so
    under causal attention one row per forward is numerically the same as any batch."""
    values = []
    for prompt, completion in zip(prompt_ids, completion_ids, strict=True):
        ids = torch.tensor([prompt + completion], device=device)
        mask = torch.tensor([[0] * len(prompt) + [1] * len(completion)], device=device)
        with torch.no_grad():
            logits = model(input_ids=ids, attention_mask=torch.ones_like(ids)).logits
        token_logps = torch.log_softmax(logits[:, :-1].float(), dim=-1)
        token_logps = token_logps.gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)
        values.append(float((token_logps * mask[:, 1:]).sum()))
    return torch.tensor(values)


def max_relative_error(actual: torch.Tensor, reference: torch.Tensor) -> float:
    return float(((actual - reference).abs() / reference.abs().clamp(min=1e-6)).max())


def build_args(kind: str, output_dir: str):
    common = {
        "output_dir": output_dir,
        "max_steps": N_STEPS,
        "per_device_train_batch_size": BATCH_SIZE,
        "learning_rate": 1e-4,
        "lr_scheduler_type": "constant",
        "logging_steps": 1,
        "logging_nan_inf_filter": False,
        "save_strategy": "no",
        "report_to": [],
        "seed": 42,
        "bf16": True,
        "gradient_checkpointing": False,
        "precompute_ref_log_probs": True,
        "dataloader_num_workers": 0,
    }
    _, config_cls = TRAINERS[kind]
    return config_cls(max_length=MAX_LENGTH, **common)


def all_ranks_agree(value, ctx) -> bool:
    gathered: list = [None] * ctx.world_size
    dist.all_gather_object(gathered, value)
    return all(peer == gathered[0] for peer in gathered)


@gpu_test_main(exact_world_size=2, prefix="ep_preference_precompute")
def run(ctx):
    parser = argparse.ArgumentParser()
    parser.add_argument("--trainer", choices=sorted(TRAINERS), default="dpo")
    kind = parser.parse_args().trainer
    trainer_cls, _ = TRAINERS[kind]

    checks, metrics = {}, {}
    tokenizer = AutoTokenizer.from_pretrained(QWEN3_0_6B)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    config = ParallelismConfig(ep_size=EP_SIZE)
    model = build_model().to(ctx.device)
    apply_ep_layers(model, config.ep_config())

    trainer = trainer_cls(
        model=model,
        args=build_args(kind, ctx.output_dir),
        train_dataset=rank_private_dataset(kind, ctx, "train"),
        processing_class=tokenizer,
        parallelism_config=config,
    )
    ctx.on_teardown(trainer.cleanup_ep)

    prepared = trainer.train_dataset
    columns = prepared.column_names
    reference_model = build_model().to(ctx.device).eval()
    prompt_ids = prepared["prompt_ids"]

    if kind == "dpo":
        completions = {"ref_chosen_logps": prepared["chosen_ids"], "ref_rejected_logps": prepared["rejected_ids"]}
        # Control: each row's rejected completion against the chosen column, and vice versa.
        controls = {"ref_chosen_logps": prepared["rejected_ids"], "ref_rejected_logps": prepared["chosen_ids"]}
    else:
        completions = {"ref_logps": prepared["completion_ids"]}
        # Control: the KL column comes from the mismatched completions TRL cycles in a batch, so true completions must
        # miss.
        controls = {"ref_logps": prepared["KL_completion_ids"]} if trainer.calculate_KL else {}
        if trainer.calculate_KL:
            completions["ref_KL_logps"] = prepared["KL_completion_ids"]
            controls["ref_KL_logps"] = prepared["completion_ids"]
    expected_columns = tuple(completions)

    checks["ref_columns_present"] = all(name in columns for name in expected_columns)
    if not checks["ref_columns_present"]:
        log(f"prepared columns: {columns}")
        return {"checks": checks, "metrics": metrics}

    for name in expected_columns:
        actual = column(prepared, name)
        reference = sequence_logprobs(reference_model, prompt_ids, completions[name], ctx.device)
        error = max_relative_error(actual, reference)
        metrics[f"{name}_max_rel_err"] = error
        checks[f"{name}_matches_dense_reference"] = error < LOGP_RTOL
        # Anti-vacuity: an all-zero or constant column would match a broken reference trivially.
        checks[f"{name}_finite"] = bool(torch.isfinite(actual).all())
        checks[f"{name}_nondegenerate"] = bool(actual.std() > 1.0 and actual.abs().min() > 1.0)
        # Byte for byte, each from its own gather.
        checks[f"{name}_identical_across_ranks"] = all_ranks_agree(actual.tolist(), ctx)
        if name in controls:
            control_error = max_relative_error(
                actual, sequence_logprobs(reference_model, prompt_ids, controls[name], ctx.device)
            )
            metrics[f"{name}_control_rel_err"] = control_error
            checks[f"{name}_control_misses"] = control_error > CONTROL_MIN_RTOL

    # No deadlock through the training loop (the sweep leaves the EP buffers live).
    checks.update(training_run_checks(trainer.train(), trainer, N_STEPS, grad_norms=True))
    losses = step_losses(trainer)
    checks["every_step_logged"] = len(losses) == N_STEPS
    metrics["first_loss"], metrics["last_loss"] = losses[0], losses[-1]

    # Pre-sharded data + one gathered set would train every non-zero rank on rank 0's log-probs, so this must raise.
    try:
        trainer_cls(
            model=model,
            args=build_args(kind, ctx.output_dir + "/presharded"),
            train_dataset=rank_private_dataset(kind, ctx, "presharded"),
            processing_class=tokenizer,
            parallelism_config=config,
            dataset_presharded=True,
        )
        checks["presharded_precompute_rejected"] = False
    except ValueError as exc:
        checks["presharded_precompute_rejected"] = "pre-sharded" in str(exc)

    return {"checks": checks, "metrics": metrics}


if __name__ == "__main__":
    sys.exit(run())
