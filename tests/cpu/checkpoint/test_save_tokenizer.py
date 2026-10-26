"""The tokenizer a checkpoint loads with is written once per save, before its weights, on every saver.

A save stopped after its index names the weight files is a loadable checkpoint, so the tokenizer must
already be beside it; and a tokenizer write that fails on the save rank must stop every rank rather than
strand the peers in the next collective.

    python tests/cpu/checkpoint/test_save_tokenizer.py
"""

import datetime
import os

import pytest
import torch.nn as nn

from halo.checkpoint.parallel.save import save_checkpoint, save_tokenizer
from halo.checkpoint.writing import StreamingShardWriter
from tests.common.checkpoint_io import save_context
from tests.common.gloo import run_gloo_ranks
from tests.common.tiny_models import word_level_tokenizer

WORLD_SIZE = 2
PG_TIMEOUT_SEC = 15


class _Killed(BaseException):
    """The process dying: not an ``Exception``, so no deferred-failure guard records it."""


def test_the_tokenizer_is_on_disk_when_the_save_dies_after_its_index(tmp_path, monkeypatch):
    close = StreamingShardWriter.close_as_hf_checkpoint

    def close_then_die(writer):
        close(writer)
        raise _Killed

    monkeypatch.setattr(StreamingShardWriter, "close_as_hf_checkpoint", close_then_die)
    ctx = save_context(nn.Linear(4, 4), toolkit_fsdp2=True, is_save_rank=True, tokenizer=word_level_tokenizer())

    with pytest.raises(_Killed):
        save_checkpoint(ctx, str(tmp_path))

    assert (tmp_path / "model.safetensors").is_file(), "premise: the weights and their index were written"
    assert (tmp_path / "tokenizer_config.json").is_file(), "a checkpoint that loads ships without its tokenizer"


class _FailingTokenizer:
    def save_pretrained(self, output_dir):
        raise OSError(28, "No space left on device")


def _worker(rank: int, tmp_dir: str) -> None:
    ctx = save_context(nn.Linear(4, 4), is_save_rank=rank == 0, tokenizer=_FailingTokenizer())
    try:
        save_tokenizer(ctx, tmp_dir)
        outcome = "NO RAISE"
    except BaseException as error:
        outcome = f"{type(error).__name__}: {error}"
    with open(os.path.join(tmp_dir, f"result_{rank}.txt"), "w") as fh:
        fh.write(outcome)


def test_a_tokenizer_write_failure_raises_on_every_rank(tmp_path):
    run_gloo_ranks(_worker, WORLD_SIZE, str(tmp_path), pg_timeout=datetime.timedelta(seconds=PG_TIMEOUT_SEC))

    for rank in range(WORLD_SIZE):
        outcome = (tmp_path / f"result_{rank}.txt").read_text()
        assert outcome != "NO RAISE", f"rank {rank} sailed past the save rank's failed tokenizer write"
        assert "No space left on device" in outcome, f"rank {rank} does not name the cause: {outcome}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
