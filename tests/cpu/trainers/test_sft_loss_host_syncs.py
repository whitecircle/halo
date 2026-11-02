#!/usr/bin/env python
"""An SFT micro-batch issues no metric collective and reads nothing back to the host.

TRL's ``SFTTrainer.compute_loss`` gathers its token count, entropy and accuracy across the world and reads each
back with ``.item()`` on every micro-batch, and HF's exact input-token count (``include_num_input_tokens_seen``,
which the efficiency metrics turn on) gathers one more; at 512 ranks that is a collective and a host stall per
metric per micro-batch. The trainer records the metrics on the device and folds them once per log, and adds the
input tokens up on the host until a log or a save sums them. This drives the real trainer for a few steps on CPU
and counts, outside the log calls, every accelerate gather or reduce and every ``.item()``/``.tolist()`` the toolkit
or TRL issues.

Run: python tests/cpu/trainers/test_sft_loss_host_syncs.py
"""

import sys
from collections import Counter

import pytest
import torch
from accelerate import Accelerator
from datasets import Dataset
from transformers import Qwen3Config, Qwen3ForCausalLM, TrainerCallback
from trl import SFTConfig

from halo.trainers.sft import DistributedSFTTrainer
from tests.common.offline_grpo import OFFLINE_VOCAB, make_offline_tokenizer
from tests.common.parallelism import make_parallelism_config

SEQ_LEN = 24
BATCH = 2
GRAD_ACCUM = 2
# Optimizer steps whose micro-batches are counted: past the first step's one-off checks, none a log step's flush.
COUNTED_STEPS = range(2, 6)
LOGGING_STEPS = 3
# Packages whose host reads count: the trainer stack this test owns, not torch's or transformers' internals.
_COUNTED_PACKAGES = ("/halo/", "/trl/")


class _CountWindow(TrainerCallback):
    """Turns the counters on for the micro-batches of :data:`COUNTED_STEPS`; HF logs after ``on_step_end``."""

    def __init__(self):
        self.step, self.active = 0, False

    def on_step_begin(self, args, state, control, **kwargs):
        self.active = self.step in COUNTED_STEPS

    def on_step_end(self, args, state, control, **kwargs):
        self.step, self.active = self.step + 1, False


@pytest.fixture
def counted(monkeypatch):
    """``(window, counts)``: collectives and host reads made while ``window.active``."""
    window, counts = _CountWindow(), Counter()

    def counting_collective(name, original):
        def wrapper(self, *args, **kwargs):
            if window.active:
                counts[f"accelerator.{name}"] += 1
            return original(self, *args, **kwargs)

        return wrapper

    def counting_read(name, original):
        def wrapper(self, *args, **kwargs):
            caller = sys._getframe(1).f_code.co_filename
            if window.active and any(package in caller for package in _COUNTED_PACKAGES):
                counts[f"{name} at {caller.rsplit('/', 2)[-2]}"] += 1
            return original(self, *args, **kwargs)

        return wrapper

    for name in ("gather", "gather_for_metrics", "reduce"):
        monkeypatch.setattr(Accelerator, name, counting_collective(name, getattr(Accelerator, name)))
    for name in ("item", "tolist"):
        monkeypatch.setattr(torch.Tensor, name, counting_read(name, getattr(torch.Tensor, name)))
    return window, counts


def _trainer(tmp_path, window, input_tokens: str) -> DistributedSFTTrainer:
    torch.manual_seed(0)
    vocab = len(OFFLINE_VOCAB)
    model = Qwen3ForCausalLM(
        Qwen3Config(
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            vocab_size=vocab,
            pad_token_id=0,
            eos_token_id=1,
        )
    )
    rows = [[3 + (i + j) % (vocab - 3) for j in range(SEQ_LEN)] for i in range(64)]
    args = SFTConfig(
        output_dir=str(tmp_path),
        use_cpu=True,
        bf16=False,
        max_length=SEQ_LEN,
        per_device_train_batch_size=BATCH,
        gradient_accumulation_steps=GRAD_ACCUM,
        max_steps=COUNTED_STEPS.stop,
        logging_steps=LOGGING_STEPS,
        gradient_checkpointing=False,
        use_liger_kernel=False,
        include_num_input_tokens_seen=input_tokens,
        dataloader_num_workers=0,
        save_strategy="no",
        report_to="none",
        disable_tqdm=True,
    )
    return DistributedSFTTrainer(
        model=model,
        args=args,
        train_dataset=Dataset.from_dict({"input_ids": rows}),
        processing_class=make_offline_tokenizer(),
        parallelism_config=make_parallelism_config(world_size=1, gpus_per_node=1),
        callbacks=[window],
    )


@pytest.mark.parametrize("input_tokens", ["no", "all"])
def test_micro_batches_issue_no_metric_collective_or_host_read(tmp_path, counted, input_tokens):
    """The input-token count on or off, a step between logs reads nothing back; the count still lands whole."""
    window, counts = counted
    trainer = _trainer(tmp_path, window, input_tokens)
    trainer.train()

    assert window.step == COUNTED_STEPS.stop, "the counting window never closed"
    assert dict(counts) == {}, f"collectives / host reads over {len(COUNTED_STEPS)} steps"
    logged = next(entry for entry in trainer.state.log_history if entry.get("step") == LOGGING_STEPS)
    if input_tokens == "all":
        assert trainer.state.num_input_tokens_seen == COUNTED_STEPS.stop * GRAD_ACCUM * BATCH * SEQ_LEN
        assert logged["num_input_tokens_seen"] == LOGGING_STEPS * GRAD_ACCUM * BATCH * SEQ_LEN
    assert logged["num_attended_tokens_seen"] == LOGGING_STEPS * GRAD_ACCUM * BATCH * SEQ_LEN
    assert 0.0 <= logged["mean_token_accuracy"] <= 1.0
    assert logged["entropy"] > 0.0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
