#!/usr/bin/env python
"""Packed SFT under TP2: each document's varlen bounds reach the TP-sharded attention on both ranks, and bind it.

The packing collator the factory builds for a flash backend hands the model the documents' cumulative lengths
(``cu_seq_lens_q`` / ``max_length_q``), and under TP each rank's attention runs its own heads over the same packed
row, so both ranks have to receive the same bounds. On Qwen3-0.6B at ``tp_size=2``, through
``DistributedSFTTrainer`` with the factory's packing collator:

  * every forward of the first decoder layer's attention receives ``cu_seq_lens_q`` and ``max_length_q``, the same
    on both TP ranks, describing several documents and ending at the row's token count;
  * on the first batch the model trained on, the logits under its own bounds repeat bit for bit, and moving one inner
    document boundary by one token changes them: the kernel reads the bounds rather than attending across the row.

The trainer is built as ``scripts/training/sft.py`` builds it: rows tokenized as the script tokenizes a rendered
conversation and packed by its packing pass (``coordinated_pack``), so each row carries the ``seq_lengths`` the
collator bounds its documents by, and TRL's own packing and dataset preparation turned off
(``disable_trl_dataset_prep``). TRL's default "bfd" packing turns on its ``padding_free``, which refuses the collator.

Run: torchrun --nproc_per_node=2 tests/gpu/trainers/sft/test_sft_tp_packed_varlen.py
"""

import torch
import torch.distributed as dist
from trl import SFTConfig

from halo.data.collators.factory import select_data_collator
from halo.data.pipeline.coordinated_map import coordinated_pack
from halo.distributed.parallelism_config import ParallelismConfig
from halo.entrypoints.data_setup import disable_trl_dataset_prep
from halo.model_loading.parallel.loader import load_distributed_model
from halo.trainers.sft import DistributedSFTTrainer
from tests.common.datasets import create_tokenized_sft_dataset
from tests.common.harness import gpu_test_main
from tests.common.models import QWEN3_0_6B
from tests.common.parallel_shape import parallel_shape_checks
from tests.common.rank_log import log

SEED = 42
STEPS = 2
MAX_LENGTH = 2048
_VARLEN_KEYS = ("cu_seq_lens_q", "max_length_q")
# The collated model inputs the replay feeds back. transformers' flash path takes handed-over bounds only with
# ``max_length_q/k`` beside them (host ints); without them it re-derives the bounds from ``position_ids``.
_BATCH_KEYS = ("input_ids", "position_ids", "cu_seq_lens_q", "cu_seq_lens_k", "max_length_q", "max_length_k")


def _shift_inner_boundary(batch: dict) -> dict:
    """``batch``'s varlen set with its first inner document boundary one token later, the longest document re-read
    so the kernel still covers every one."""
    shifted = batch["cu_seq_lens_q"].clone()
    shifted[1] += 1
    longest = int(shifted.diff().max())
    return {"cu_seq_lens_q": shifted, "cu_seq_lens_k": shifted, "max_length_q": longest, "max_length_k": longest}


@gpu_test_main(exact_world_size=2, prefix="sft_tp_packed_varlen")
def run(ctx):
    torch.cuda.set_device(ctx.device)
    pc = ParallelismConfig(tp_size=2)
    model, tokenizer = load_distributed_model(
        QWEN3_0_6B, pc, torch.bfloat16, trust_remote_code=False, attn_implementation="flash_attention_2"
    )
    config = SFTConfig(
        output_dir=ctx.output_dir,
        max_steps=STEPS,
        per_device_train_batch_size=1,
        learning_rate=1e-5,
        bf16=True,
        gradient_checkpointing=False,
        use_liger_kernel=False,
        packing=True,
        max_length=MAX_LENGTH,
        logging_steps=1,
        save_strategy="no",
        report_to=[],
        logging_nan_inf_filter=False,
        dataloader_num_workers=0,
        seed=SEED,
    )
    collator = select_data_collator(
        tokenizer,
        train_on_completions_only=False,
        packing=config.packing,
        model_config=model.config,
        per_device_train_batch_size=config.per_device_train_batch_size,
        emit_varlen_one_document=True,
    )
    rows = create_tokenized_sft_dataset(256, tokenizer, seed=SEED)
    train_dataset = coordinated_pack(rows, config.max_length, config.packing_strategy)
    disable_trl_dataset_prep(config)
    trainer = DistributedSFTTrainer(
        model=model,
        args=config,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        data_collator=collator,
        parallelism_config=pc,
    )
    checks = parallel_shape_checks(trainer.model, pc)
    checks["trainer_trains_on_the_factory_collator"] = trainer.data_collator is collator
    seen: list[dict] = []
    batch: dict = {}

    def capture(module, args, kwargs):
        seen.append({key: kwargs.get(key) for key in _VARLEN_KEYS})

    def record_first_batch(module, args, kwargs):
        if not batch:
            batch.update(
                {
                    key: value.detach().clone() if torch.is_tensor(value) else value
                    for key, value in kwargs.items()
                    if key in _BATCH_KEYS
                }
            )

    handles = [
        trainer.model.model.layers[0].self_attn.register_forward_pre_hook(capture, with_kwargs=True),
        trainer.model.register_forward_pre_hook(record_first_batch, with_kwargs=True),
    ]
    trainer.train()
    for handle in handles:
        handle.remove()

    checks["attention_saw_every_forward"] = bool(seen)
    checks["every_forward_carried_the_varlen_bounds"] = bool(seen) and all(
        all(call[key] is not None for key in _VARLEN_KEYS) for call in seen
    )
    first = seen[0] if seen else {}
    bounds = first.get("cu_seq_lens_q")
    mine = None if bounds is None else bounds.detach().cpu().tolist()
    gathered: list = [None] * ctx.world_size
    dist.all_gather_object(gathered, mine)
    checks["both_tp_ranks_received_the_same_bounds"] = mine is not None and all(view == mine for view in gathered)
    tokens = batch["input_ids"].shape[-1] if "input_ids" in batch else -1
    checks["bounds_describe_several_documents_of_the_row"] = (
        mine is not None and len(mine) > 2 and mine[0] == 0 and mine[-1] == tokens
    )

    if "cu_seq_lens_q" in batch:
        trainer.model.eval()
        with torch.no_grad():
            logits = trainer.model(**batch).logits
            again = trainer.model(**batch).logits
            moved = trainer.model(**(batch | _shift_inner_boundary(batch))).logits
        checks["logits_repeat_under_the_same_bounds"] = torch.equal(logits, again)
        checks["a_shifted_boundary_changes_the_logits"] = not torch.equal(logits, moved)
        log(
            f"  bounds {mine[:6] if mine else None}..., {len(seen)} attention forwards; shifted max|Δ| "
            f"{(logits.float() - moved.float()).abs().max().item():.3e}"
        )
    else:
        checks["collator_emitted_varlen_bounds"] = False
    return {"checks": checks}


if __name__ == "__main__":
    run()
