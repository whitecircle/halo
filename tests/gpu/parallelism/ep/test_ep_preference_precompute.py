#!/usr/bin/env python
"""DPO / KTO reference-log-prob PRECOMPUTE under Expert Parallelism (ep2, tiny Qwen3-MoE).

``precompute_ref_log_probs`` is the only way to run a *full-finetune* DPO/KTO under EP: the
reference model cannot be parallelized, so TRL sweeps the EP-patched policy once inside
``__init__`` and caches the log-probs as dataset columns. TRL keys that disk cache on
``Hasher.hash((dataset._fingerprint, hash_module(model)))`` — and under EP ``hash_module`` hashes a
RANK-SHARDED expert state dict, so every rank derives a DIFFERENT cache path. TRL writes the file on
the main process only and has every rank read it back, so a divergent key means the non-main ranks
block on a path nobody wrote and the next EP collective deadlocks with no exception anywhere.
``PrecomputeRefLogpsRankConsistentMixin`` collapses both key inputs to rank 0's value.

This test pins that seam three ways, and fails when any of them breaks:

  * **the mechanism** — the cache path every rank derives inside the sweep must be identical, and
    the file must exist on every rank afterwards. Its anti-vacuity control is the raw
    ``hash_module`` of the EP-patched model, which must DIVERGE across ranks: that proves the guard
    is load-bearing here rather than trivially satisfied by a rank-identical model.
  * **the values** — the cached columns must equal per-sequence reference log-probs recomputed on a
    DENSE, un-EP-patched copy of the same seeded weights, row by row, under TRL's exact shifted
    completion-mask convention. This catches a sweep that silently ran on the wrong rows (a
    mis-ordered ``gather_for_metrics``), on a stale cache from another model, or that copied rank
    0's log-probs onto rank r's rows. Anti-vacuity: the same comparison with chosen/rejected swapped
    (DPO) or against the mismatched KL completions (KTO) must MISS by orders of magnitude.
  * **no deadlock** — construction and a short ``train()`` complete. A re-deadlock does not raise,
    it wedges, so the manifest timeout is the assertion; keep it tight.

Also pins the pre-sharded rejection (per-rank shards + one rank-0-authoritative cache = every
non-zero rank training on rank 0's log-probs).

Run: torchrun --nproc_per_node=2 tests/gpu/parallelism/ep/test_ep_preference_precompute.py \
         --trainer dpo
"""

import argparse
import os
import sys

import torch
import torch.distributed as dist
import trl.experimental.kto.kto_trainer as trl_kto_module
import trl.trainer.dpo_trainer as trl_dpo_module
from datasets import Dataset, load_from_disk
from transformers import AutoTokenizer, Qwen3MoeConfig, Qwen3MoeForCausalLM
from trl import DPOConfig, KTOConfig

from src.distributed.expert_parallel.patching import create_ep_buffers, patch_moe_model_for_ep
from src.distributed.parallelism_config import ParallelismConfig
from tests.common.harness import gpu_test_main, log
from tests.common.models import QWEN3_0_6B, TINY_QWEN3_MOE_CONFIG

# The module whose ``hash_module`` symbol keys the cache — the one the guard patches per trainer.
TRL_MODULES = {"dpo": trl_dpo_module, "kto": trl_kto_module}

EP_SIZE = 2
N_ROWS = 16
N_STEPS = 4
BATCH_SIZE = 2
MAX_LENGTH = 96

# The REAL Qwen tokenizer tokenizes the text rows, so the tiny model must span its full id range.
VOCAB_SIZE = 151936

# Per-sequence log-prob sums run to ~-1e2 (untrained model over a 151936-way vocab). The EP path
# routes the expert FFN through DeepEP dispatch/combine in bf16 while the reference runs the dense
# HF experts, so the two agree only to bf16 kernel/reduction-order noise: measured max relative
# error on this model is 3.1e-3. 1e-2 clears that floor without flaking, and still sits two orders
# of magnitude under the smallest real bug this can see — a swapped or permuted row moves the sum
# by ~1.0 relative (measured by the controls below).
LOGP_RTOL = 1e-2
# The controls must MISS by far more than the tolerance, or the comparison proves nothing.
CONTROL_MIN_RTOL = 0.05


def build_model() -> Qwen3MoeForCausalLM:
    """The seeded tiny MoE. Called identically on every rank and for the dense reference."""
    torch.manual_seed(1234)
    config = Qwen3MoeConfig(
        **{**TINY_QWEN3_MOE_CONFIG, "vocab_size": VOCAB_SIZE},
        pad_token_id=0,
        eos_token_id=1,
    )
    return Qwen3MoeForCausalLM(config).to(torch.bfloat16)


def build_rows(kind: str) -> dict:
    """Text preference rows with RAGGED completion lengths.

    Ragged chosen/rejected lengths are what give the swap control its teeth: two completions of
    equal length over an untrained model would sum to nearly the same log-prob, and a swapped-row
    comparison would then pass by accident.
    """
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


def shared_dataset(kind: str, ctx) -> Dataset:
    """A dataset backed by ONE arrow directory every rank loads from.

    Required, not incidental: TRL's cache lives beside the dataset's own arrow files, and the whole
    contract under test is that rank 0 writes a file the other ranks then read. An in-memory dataset
    gets a random cache path per call and a per-rank directory gets a private one, so either would
    make the sweep unreachable rather than exercising it — the production shape is a shared
    ``HF_DATASETS_CACHE``.
    """
    directories: list = [None] * ctx.world_size
    dist.all_gather_object(directories, ctx.output_dir)
    path = os.path.join(directories[0], f"{kind}_dataset")
    if ctx.rank == 0:
        Dataset.from_dict(build_rows(kind)).save_to_disk(path)
    dist.barrier()
    return load_from_disk(path)


def sequence_logprobs(model, prompt_ids: list[list[int]], completion_ids: list[list[int]], device) -> torch.Tensor:
    """TRL's reference log-prob per row, on a DENSE model, independent of ``src/``.

    Mirrors ``compute_ref_log_probs``: forward prompt⧺completion, shift by one, score the tokens the
    shifted completion mask marks, sum. Both TRL collators right-pad only, so under causal attention
    a row's value does not depend on the batch it was collated with — one row per forward is
    numerically the same batch of one.
    """
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
        "save_strategy": "no",
        "report_to": [],
        "seed": 42,
        "bf16": True,
        "gradient_checkpointing": False,
        "precompute_ref_log_probs": True,
        "dataloader_num_workers": 0,
        "ddp_find_unused_parameters": True,  # EP leaves inactive experts gradient-free
        "fsdp": "",  # the mixin owns FSDP wrapping
    }
    return (DPOConfig if kind == "dpo" else KTOConfig)(max_length=MAX_LENGTH, **common)


def spying_trainer_class(kind: str, recorded: list[str]):
    """The real trainer with the cache path it derives inside the sweep recorded.

    ``Dataset._get_cache_file_path`` is spied only for the duration of the sweep so the surrounding
    ``map`` calls (which legitimately use their own fingerprints) stay out of the record. The
    subclass is a subclass of ``PrecomputeRefLogpsRankConsistentMixin``, which is exactly the class
    set ``_defining_module`` skips — so wrapping here cannot displace the guard it is measuring.
    """
    from src.trainers.preference.dpo import DistributedDPOTrainer
    from src.trainers.preference.kto import DistributedKTOTrainer

    base = DistributedDPOTrainer if kind == "dpo" else DistributedKTOTrainer

    class SpyingTrainer(base):
        def _precompute_ref_logps(self, dataset, name, batch_size):
            original = Dataset._get_cache_file_path

            def spy(dataset_self, fingerprint):
                path = original(dataset_self, fingerprint)
                recorded.append(path)
                return path

            Dataset._get_cache_file_path = spy
            try:
                return super()._precompute_ref_logps(dataset, name, batch_size)
            finally:
                Dataset._get_cache_file_path = original

    return SpyingTrainer


def all_ranks_agree(value, ctx) -> bool:
    gathered: list = [None] * ctx.world_size
    dist.all_gather_object(gathered, value)
    return all(peer == gathered[0] for peer in gathered)


def any_rank_differs(value, ctx) -> bool:
    gathered: list = [None] * ctx.world_size
    dist.all_gather_object(gathered, value)
    return any(peer != gathered[0] for peer in gathered)


@gpu_test_main(exact_world_size=2, prefix="ep_preference_precompute")
def run(ctx):
    parser = argparse.ArgumentParser()
    parser.add_argument("--trainer", choices=("dpo", "kto"), default="dpo")
    kind = parser.parse_args().trainer

    checks, metrics = {}, {}
    tokenizer = AutoTokenizer.from_pretrained(QWEN3_0_6B)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    config = ParallelismConfig(ep_size=EP_SIZE)
    model = build_model().to(ctx.device)
    patch_moe_model_for_ep(model, config.create_ep_config())
    create_ep_buffers(model)

    # ── Anti-vacuity for the mechanism check: the raw cache-key input MUST diverge across ranks.
    # If it did not (a dense model, or EP that never sharded the experts), the identical-path check
    # below would pass with the guard deleted and prove nothing.
    trl_module = TRL_MODULES[kind]
    hash_module = trl_module.hash_module
    checks["raw_model_hash_diverges_across_ranks"] = any_rank_differs(hash_module(model), ctx)

    dataset = shared_dataset(kind, ctx)
    args = build_args(kind, ctx.output_dir)
    recorded: list[str] = []
    trainer = spying_trainer_class(kind, recorded)(
        model=model,
        args=args,
        train_dataset=dataset,
        processing_class=tokenizer,
        parallelism_config=config,
    )
    ctx.on_teardown(trainer.cleanup_ep)

    # ── The mechanism: one cache path, derived identically everywhere, written by rank 0, present
    # for all. This is the deadlock precondition; with the guard removed the paths fork on the
    # sharded expert hash and the non-main ranks read a file nobody wrote.
    checks["sweep_derived_a_cache_path"] = bool(recorded)
    cache_path = recorded[0] if recorded else ""
    checks["precompute_cache_path_identical_across_ranks"] = all_ranks_agree(cache_path, ctx)
    checks["precompute_cache_file_exists_on_every_rank"] = os.path.exists(cache_path)
    # The patched symbol is restored, so nothing downstream inherits a broadcasting hash.
    checks["hash_module_restored_after_sweep"] = trl_module.hash_module is hash_module

    # ── The values: the cached columns vs a DENSE reference over the trainer's OWN tokenized rows.
    prepared = trainer.train_dataset
    columns = prepared.column_names
    reference_model = build_model().to(ctx.device).eval()
    prompt_ids = prepared["prompt_ids"]

    if kind == "dpo":
        expected_columns = ("ref_chosen_logps", "ref_rejected_logps")
        completions = {"ref_chosen_logps": prepared["chosen_ids"], "ref_rejected_logps": prepared["rejected_ids"]}
        # Control: score each row's REJECTED completion against the CHOSEN column (and vice versa).
        controls = {"ref_chosen_logps": prepared["rejected_ids"], "ref_rejected_logps": prepared["chosen_ids"]}
    else:
        expected_columns = ("ref_logps", "ref_KL_logps") if trainer.calculate_KL else ("ref_logps",)
        completions = {"ref_logps": prepared["completion_ids"]}
        # Control: the KL column comes from the MISMATCHED completions TRL cycles within a batch, so
        # scoring the true completions against it must miss.
        controls = {"ref_logps": prepared["KL_completion_ids"]} if trainer.calculate_KL else {}
        if trainer.calculate_KL:
            completions["ref_KL_logps"] = prepared["KL_completion_ids"]
            controls["ref_KL_logps"] = prepared["completion_ids"]

    checks["ref_columns_present"] = all(column in columns for column in expected_columns)
    if not checks["ref_columns_present"]:
        log(f"prepared columns: {columns}")
        return {"checks": checks, "metrics": metrics}

    for column in expected_columns:
        actual = torch.tensor(prepared[column], dtype=torch.float32)
        reference = sequence_logprobs(reference_model, prompt_ids, completions[column], ctx.device)
        error = max_relative_error(actual, reference)
        metrics[f"{column}_max_rel_err"] = error
        checks[f"{column}_matches_dense_reference"] = error < LOGP_RTOL
        # Anti-vacuity: real, non-degenerate values (an all-zero or constant column would match a
        # broken reference trivially).
        checks[f"{column}_finite"] = bool(torch.isfinite(actual).all())
        checks[f"{column}_nondegenerate"] = bool(actual.std() > 1.0 and actual.abs().min() > 1.0)
        # Every rank must hold rank 0's values byte-for-byte (they all read one cache file).
        checks[f"{column}_identical_across_ranks"] = all_ranks_agree(actual.tolist(), ctx)
        if column in controls:
            control_error = max_relative_error(
                actual, sequence_logprobs(reference_model, prompt_ids, controls[column], ctx.device)
            )
            metrics[f"{column}_control_rel_err"] = control_error
            checks[f"{column}_control_misses"] = control_error > CONTROL_MIN_RTOL

    # ── No deadlock through the training loop either (the sweep leaves the EP buffers live).
    trainer.train()
    losses = [entry["loss"] for entry in trainer.state.log_history if "loss" in entry]
    checks["ran_all_steps"] = trainer.state.global_step == N_STEPS and len(losses) == N_STEPS
    checks["losses_finite"] = bool(torch.isfinite(torch.tensor(losses)).all())
    local = torch.tensor(losses, device=ctx.device)
    gathered = [torch.zeros_like(local) for _ in range(ctx.world_size)]
    dist.all_gather(gathered, local)
    checks["losses_identical_across_ranks"] = all(torch.allclose(local, peer, atol=1e-4) for peer in gathered)
    grad_norms = [entry["grad_norm"] for entry in trainer.state.log_history if "grad_norm" in entry]
    checks["grad_norm_logged_finite"] = bool(grad_norms) and bool(torch.isfinite(torch.tensor(grad_norms)).all())
    metrics["first_loss"], metrics["last_loss"] = losses[0], losses[-1]

    # ── Pre-sharded data + one rank-0-authoritative cache would train every non-zero rank on rank
    # 0's log-probs; that must raise rather than silently mis-train.
    try:
        spying_trainer_class(kind, [])(
            model=model,
            args=build_args(kind, ctx.output_dir + "/presharded"),
            train_dataset=shared_dataset(kind, ctx),
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
