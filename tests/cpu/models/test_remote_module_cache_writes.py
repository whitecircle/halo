#!/usr/bin/env python
"""A remote-code module a peer is still copying into ``HF_MODULES_CACHE`` is never imported half-written.

A node's ranks share the dynamic-module cache, and transformers copies a module into it by truncating
and rewriting the file under a lock that is per process. A rank that finds the file present
mid-copy imports the cut module, and a config cut mid-``__init__`` loads with no error but without
its later attributes. ``src.models.patches.remote_code_hooks`` makes the copies land whole; here one
process's copy stops halfway while a second process resolves the same remote config into the same
cold cache.

Run: python tests/cpu/models/test_remote_module_cache_writes.py
"""

import multiprocessing
import shutil
from pathlib import Path

import pytest
from transformers import AutoConfig

import src.models.patches.remote_code_hooks  # noqa: F401 - the whole-file copies under test
from tests.common.models import BAILING_MOE_LING_MINI
from tests.common.tokenizers import load_cached_config

# Bound on every cross-process wait: a stuck peer fails the test instead of hanging it.
WAIT_S = 120.0


def _copy_stalled_halfway(half_written, reader_done):
    """A ``shutil.copyfile`` whose writer is descheduled after half the bytes, until the reader is done."""
    real = shutil.copyfile

    def copyfile(src, dst, *args, **kwargs):
        shutil.copyfile = real
        data = Path(src).read_bytes()
        with open(dst, "wb") as fh:
            fh.write(data[: len(data) // 2])
            fh.flush()
            half_written.set()
            reader_done.wait(WAIT_S)
            fh.write(data[len(data) // 2 :])
        return dst

    return copyfile


def _writer(half_written, reader_done) -> None:
    shutil.copyfile = _copy_stalled_halfway(half_written, reader_done)
    AutoConfig.from_pretrained(BAILING_MOE_LING_MINI, trust_remote_code=True)


def _reader(half_written, reader_done, result_path: str) -> None:
    try:
        if not half_written.wait(WAIT_S):
            raise TimeoutError("the writer never reached its copy")
        config = AutoConfig.from_pretrained(BAILING_MOE_LING_MINI, trust_remote_code=True)
        result = f"num_attention_heads={config.num_attention_heads}"
    except Exception as e:
        result = f"{type(e).__name__}: {e}"
    finally:
        reader_done.set()
    with open(result_path, "w") as fh:
        fh.write(result)


def test_a_peer_mid_copy_never_hands_out_a_cut_module(tmp_path, monkeypatch):
    load_cached_config(BAILING_MOE_LING_MINI, trust_remote_code=True)
    # Inherited by the spawned processes, which import transformers after it is set.
    monkeypatch.setenv("HF_MODULES_CACHE", str(tmp_path / "modules"))
    ctx = multiprocessing.get_context("spawn")
    half_written, reader_done = ctx.Event(), ctx.Event()
    result_path = str(tmp_path / "reader.txt")
    writer = ctx.Process(target=_writer, args=(half_written, reader_done))
    reader = ctx.Process(target=_reader, args=(half_written, reader_done, result_path))
    writer.start()
    reader.start()
    for process in (reader, writer):
        process.join(WAIT_S)
        if process.is_alive():
            process.kill()
            process.join()

    assert writer.exitcode == 0, f"the stalled writer failed or hung (exit {writer.exitcode})"
    assert reader.exitcode == 0, f"the reader died or hung (exit {reader.exitcode})"
    with open(result_path) as fh:
        result = fh.read()
    assert result.startswith("num_attention_heads="), f"the reader imported the module mid-copy: {result}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
