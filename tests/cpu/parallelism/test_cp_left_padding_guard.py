"""CP refuses a left-padded batch on the host, before it reaches the device. Ulysses never uses ``attention_mask``,
so leading pads are attended and the loss is silently wrong."""

from types import SimpleNamespace

import pytest
import torch
from datasets import Dataset
from torch.utils.data import DataLoader, SequentialSampler

from halo.data.collators.offline_grpo import OfflineGRPOCPDataCollatorWithPadding
from halo.data.collators.smpo import DataCollatorForSMPO, concatenate_pairs
from halo.distributed.context_parallel.sequence_chunks import LeftPaddingGuard, reject_left_padding
from halo.trainers.grpo.offline.trainer import OfflineGRPOTrainer
from halo.trainers.preference.smpo import SMPOTrainer
from halo.trainers.sft import DistributedSFTTrainer

PAD = 0
# Equal-length prompts leave trailing pads only, the batch a cached verdict would wrongly settle on.
TRAILING_PADS = torch.tensor([[1, 1, 1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1]])
LEFT_PADDED = torch.tensor([[0, 0, 1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1]])
# Laid out by the collator below as [[0, 0, 5, 6], [7, 8, 9, 10]].
RAGGED_ROWS = [{"input_ids": [5, 6]}, {"input_ids": [7, 8, 9, 10]}]


def _left_padding_collate(rows):
    width = max(len(row["input_ids"]) for row in rows)
    input_ids = torch.full((len(rows), width), PAD)
    attention_mask = torch.zeros((len(rows), width), dtype=torch.long)
    for index, row in enumerate(rows):
        input_ids[index, width - len(row["input_ids"]) :] = torch.tensor(row["input_ids"])
        attention_mask[index, width - len(row["input_ids"]) :] = 1
    return {"input_ids": input_ids, "attention_mask": attention_mask}


def test_left_padding_in_any_row_raises():
    with pytest.raises(ValueError, match="LEFT-padded"):
        reject_left_padding(torch.tensor([[1, 1, 1, 1], [1, 1, 1, 1], [0, 1, 1, 1]]))


@pytest.mark.parametrize("dtype", [torch.long, torch.bool])
def test_a_tokenless_row_is_not_left_padding(dtype):
    """A row without tokens has no real token to attend its pads."""
    tokenless, right_padded, left_padded = [0, 0, 0, 0], [1, 1, 0, 0], [0, 1, 1, 1]
    reject_left_padding(torch.tensor([tokenless, right_padded], dtype=dtype))
    with pytest.raises(ValueError, match="LEFT-padded"):
        reject_left_padding(torch.tensor([tokenless, left_padded], dtype=dtype))


def test_smpo_collator_shape_is_what_the_guard_catches():
    """Unequal prompts left-padded, then right-padded completions, carry leading pads."""
    # Prompt lengths 2 and 4, left-padded to 4; completions right-padded to 3.
    prompt_mask = torch.tensor([[0, 0, 1, 1], [1, 1, 1, 1]])
    completion_mask = torch.tensor([[1, 1, 0], [1, 1, 1]])
    with pytest.raises(ValueError, match="LEFT-padded"):
        reject_left_padding(torch.cat([prompt_mask, completion_mask], dim=1))


def test_equal_length_prompts_produce_no_leading_pad():
    """This is the workaround the error message recommends."""
    prompt_mask = torch.tensor([[1, 1, 1, 1]])
    completion_mask = torch.tensor([[1, 1, 0]])
    reject_left_padding(torch.cat([prompt_mask, completion_mask], dim=1))


@pytest.mark.parametrize(
    "first", [TRAILING_PADS, torch.ones(2, 8, dtype=torch.long)], ids=["trailing-pads", "unpadded"]
)
def test_every_batch_is_checked_whatever_preceded_it(first):
    """An equal-length batch certifies nothing about the next one."""
    loader = DataLoader(
        [first[0], first[1], LEFT_PADDED[0], LEFT_PADDED[1]],
        batch_size=2,
        collate_fn=LeftPaddingGuard(lambda rows: {"attention_mask": torch.stack(rows)}),
    )
    batches = iter(loader)
    assert torch.equal(next(batches)["attention_mask"], first)
    with pytest.raises(ValueError, match="LEFT-padded"):
        next(batches)


def _trainer_stub(trainer_cls, *, cp: bool):
    """HF's seams and the accelerate prepare are identities, so the built ``DataLoader`` is what the assertions
    read."""
    trainer = object.__new__(trainer_cls)
    trainer.parallelism_config = SimpleNamespace(
        is_cp_mode=cp,
        is_pp_mode=False,
        non_dp_replication_factor=2,
        data_parallel_size=1,
        get_data_parallel_rank=lambda: 0,
    )
    trainer._dataset_presharded = False
    trainer.data_collator = _left_padding_collate
    trainer.train_dataset = trainer.eval_dataset = Dataset.from_list(RAGGED_ROWS)
    trainer._cached_train_group_ids = trainer._cached_eval_group_ids = [0, 1]
    trainer._train_batch_size = len(RAGGED_ROWS)
    trainer.args = SimpleNamespace(
        remove_unused_columns=False,
        train_sampling_strategy="random",
        per_device_train_batch_size=len(RAGGED_ROWS),
        per_device_eval_batch_size=len(RAGGED_ROWS),
        eval_batch_size=len(RAGGED_ROWS),
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        dataloader_persistent_workers=False,
        dataloader_multiprocessing_context=None,
        dataloader_prefetch_factor=None,
        dataloader_in_order=True,
        dataloader_drop_last=False,
        data_seed=None,
        seed=0,
    )
    trainer._remove_unused_columns = lambda dataset, description=None: dataset
    trainer._get_train_sampler = lambda: SequentialSampler(trainer.train_dataset)
    trainer._get_eval_sampler = SequentialSampler
    trainer._prepare_dataloader = lambda loader, **_geometry: loader
    return trainer


def _reference_sweep_loader(trainer):
    """Its ``(batch, real rows)`` items cut to the batch."""
    loader = trainer._reference_sweep_loader(trainer.train_dataset, "training", 1, 0)
    return (batch for batch, _real_rows in loader)


GETTERS = {
    "train": lambda trainer: trainer.get_train_dataloader(),
    "eval": lambda trainer: trainer.get_eval_dataloader(),
    "test": lambda trainer: trainer.get_test_dataloader(Dataset.from_list(RAGGED_ROWS)),
}
CP_LOADERS = {
    f"{name}-{split}": (trainer_cls, build)
    for name, trainer_cls in (
        ("sft", DistributedSFTTrainer),
        ("smpo", SMPOTrainer),
        ("offline-grpo", OfflineGRPOTrainer),
    )
    for split, build in GETTERS.items()
}
CP_LOADERS["offline-grpo-reference-sweep"] = (OfflineGRPOTrainer, _reference_sweep_loader)


@pytest.mark.parametrize("name", list(CP_LOADERS))
def test_every_cp_trainer_loader_refuses_a_left_padded_batch(name):
    trainer_cls, build = CP_LOADERS[name]
    with pytest.raises(ValueError, match="LEFT-padded"):
        next(iter(build(_trainer_stub(trainer_cls, cp=True))))


@pytest.mark.parametrize("name", list(CP_LOADERS))
def test_loaders_outside_cp_keep_left_padding(name):
    """Attention there honours the mask."""
    trainer_cls, build = CP_LOADERS[name]
    batch = next(iter(build(_trainer_stub(trainer_cls, cp=False))))
    assert sorted(batch["attention_mask"].tolist()) == [[0, 0, 1, 1], [1, 1, 1, 1]]


def _offline_row(prompt, completion):
    return {
        "prompt_input_ids": prompt,
        "completion_input_ids": completion,
        "group_id": 0,
        "group_size": 2,
        "advantage": 1.0,
    }


def test_offline_grpo_cp_rows_pass_the_guard():
    """Offline GRPO's CP collator right-pads each joined row."""
    collator = LeftPaddingGuard(OfflineGRPOCPDataCollatorWithPadding(pad_token_id=PAD, cp_size=2))
    batch = collator([_offline_row([5], [6]), _offline_row([9, 10, 11], [])])
    assert torch.equal(batch["attention_mask"], torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]]))


def test_smpo_batches_pass_the_guard_and_reach_the_forward_right_padded():
    """SMPO's batch carries no ``attention_mask``, and its concat flushes every pad behind the tokens."""
    rows = [
        {"prompt_input_ids": [5], "chosen_input_ids": [6, 7], "rejected_input_ids": [8]},
        {"prompt_input_ids": [9, 10, 11], "chosen_input_ids": [12], "rejected_input_ids": [13, 14]},
    ]
    batch = LeftPaddingGuard(DataCollatorForSMPO(pad_token_id=PAD))(rows)
    with pytest.raises(ValueError, match="LEFT-padded"):
        reject_left_padding(batch["prompt_attention_mask"])
    reject_left_padding(concatenate_pairs(batch, PAD, cp_size=2)["attention_mask"])


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
