#!/usr/bin/env python
"""Packed SFT under TP2: each document's varlen bounds reach the TP-sharded attention on both ranks, and bind it.

The packing collator the factory builds for a flash backend hands the model the documents' cumulative lengths
(``cu_seq_lens_q`` / ``max_length_q``), and under TP each rank's attention runs its own heads over the same packed
row, so both ranks have to receive the same bounds. On Qwen3-0.6B at ``tp_size=2``, through
``DistributedSFTTrainer`` with the factory's packing collator:

  * every forward of the first decoder layer's attention receives ``cu_seq_lens_q`` and ``max_length_q``, the same
    on both TP ranks, describing several documents and ending at the row's token count;
  * on one captured packed batch, the logits under its own bounds repeat bit for bit, and moving one inner document
    boundary by one token changes them: the kernel reads the bounds rather than attending across the row.

Run: torchrun --nproc_per_node=2 tests/gpu/trainers/sft/test_sft_tp_packed_varlen.py
"""

import torch
import torch.distributed as dist
from trl import SFTConfig

from halo.data.collators.factory import select_data_collator
from halo.distributed.parallelism_config import ParallelismConfig
from halo.model_loading.parallel.loader import load_distributed_model
from halo.trainers.sft import DistributedSFTTrainer
from tests.common.datasets import create_sft_dataset
from tests.common.harness import gpu_test_main
from tests.common.models import QWEN3_0_6B
from tests.common.parallel_shape import parallel_shape_checks
from tests.common.rank_log import log

SEED = 42
STEPS = 2
MAX_LENGTH = 2048
_VARLEN_KEYS = ("cu_seq_lens_q", "max_length_q")


class _RecordingCollator:
    """The factory's collator, keeping the first batch it produced."""

    def __init__(self, collator):
        self.collator = collator
        self.first = None

    def __call__(self, features):
        batch = self.collator(features)
        if self.first is None:
            self.first = {key: value.clone() for key, value in batch.items() if torch.is_tensor(value)}
        return batch


def _shift_inner_boundary(cu_seq_lens: torch.Tensor) -> torch.Tensor:
    """``cu_seq_lens`` with its first inner boundary moved one token later."""
    shifted = cu_seq_lens.clone()
    shifted[1] += 1
    return shifted


@gpu_test_main(exact_world_size=2, prefix="sft_tp_packed_varlen")
def run(ctx):
    torch.cuda.set_device(ctx.device)
    pc = ParallelismConfig(tp_size=2)
    model, tokenizer = load_distributed_model(
        QWEN3_0_6B, pc, torch.bfloat16, trust_remote_code=False, attn_implementation="flash_attention_2"
    )
    collator = _RecordingCollator(
        select_data_collator(tokenizer, train_on_completions_only=False, packing=True, model_config=model.config)
    )
    trainer = DistributedSFTTrainer(
        model=model,
        args=SFTConfig(
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
        ),
        train_dataset=create_sft_dataset(256, tokenizer, seed=SEED),
        processing_class=tokenizer,
        data_collator=collator,
        parallelism_config=pc,
    )
    checks = parallel_shape_checks(trainer.model, pc)
    seen: list[dict] = []
    attention = trainer.model.model.layers[0].self_attn

    def capture(module, args, kwargs):
        seen.append({key: kwargs.get(key) for key in _VARLEN_KEYS})

    handle = attention.register_forward_pre_hook(capture, with_kwargs=True)
    trainer.train()
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
    batch = collator.first or {}
    tokens = batch["input_ids"].shape[-1] if "input_ids" in batch else -1
    checks["bounds_describe_several_documents_of_the_row"] = (
        mine is not None and len(mine) > 2 and mine[0] == 0 and mine[-1] == tokens
    )

    if "cu_seq_lens_q" in batch:
        inputs = {key: value.to(ctx.device) for key, value in batch.items() if key != "labels"}
        trainer.model.eval()
        with torch.no_grad():
            logits = trainer.model(**inputs).logits
            again = trainer.model(**inputs).logits
            shifted = dict(inputs)
            for key in ("cu_seq_lens_q", "cu_seq_lens_k"):
                if key in shifted:
                    shifted[key] = _shift_inner_boundary(shifted[key])
            moved = trainer.model(**shifted).logits
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
