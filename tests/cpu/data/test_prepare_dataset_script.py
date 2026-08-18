#!/usr/bin/env python
"""``prepare_dataset.py`` must run the invocations the docs publish, and fail fast when it cannot.

Three things this drives through the real ``main()`` in a fresh subprocess (so the accelerate
singleton starts unset, as in a real CLI run):

* **The documented raw-text pre-training command.** The shared data-loading helpers log through
  accelerate's logger, which raises unless ``PartialState()`` exists — and the completion-masking
  default must resolve per ``--mode``: a flat default-on is rejected by ``PreprocessingConfig`` as
  inapplicable to ``mode='text'``, which would exit every documented pre-training invocation on a
  flag its user never passed.
* **An explicit masking flag in text mode still fails.** The mode-aware default must not turn the
  applicability check into a no-op — asking for something ``mode='text'`` cannot do stays loud.
* **Chat mode with no response marker fails BEFORE the input is fetched.** The labels are baked
  into the artifact here, so masking without a marker is unbuildable; the config that owns the pair
  is only constructed after the whole corpus is in hand, which is minutes of S3 download to report
  a two-flag mistake.

Plus the artifact contract, unstubbed: a ``--num-shards N`` publish must be what the training-side
sharded loader hands ``N`` data-parallel ranks — every row exactly once, none twice.

    python tests/cpu/data/test_prepare_dataset_script.py
"""

import json
import os
import subprocess
import sys
import tempfile

import pytest
from accelerate import PartialState
from datasets import Dataset
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_SCRIPT = os.path.join(REPO, "scripts", "before_training", "prepare_dataset.py")

# A real fast tokenizer built in-process, so the unstubbed run needs no download: one id per word.
_WORDS = ("doc", *(str(i) for i in range(32)))

# Runs in a subprocess: load the real script as a module, stub only the heavy/network steps
# (tokenizer load + preprocessing) so the offline load path is exercised, then call main().
# Reaching the stubbed preprocess proves load_input_dataset() ran past the accelerate logger; the
# stubbed load reports whether a rejection came before or after the input was fetched.
_DRIVER = """
import sys, traceback, types

from tests.common.utils import load_script_module

mod = load_script_module("scripts/before_training/prepare_dataset.py")


class _ReachedPreprocess(Exception):
    pass


mod.setup_tokenizer = lambda args: object()  # never used (preprocess is stubbed) — avoids any download

real_load = mod.load_input_dataset


def _load(args):
    print("REACHED_LOAD")
    return real_load(args)


def _boom(**kwargs):
    raise _ReachedPreprocess()


mod.load_input_dataset = _load
mod.preprocess_dataset = _boom
{prelude}
sys.argv = ["prepare_dataset.py"] + {argv!r}
try:
    mod.main()
except _ReachedPreprocess:
    print("REACHED_PREPROCESS")
except RuntimeError as e:
    if "accelerate state" in str(e):
        print("ACCELERATE_STATE_ERROR")
    else:
        raise
except SystemExit as e:
    # Mirror the interpreter: a string payload is the message of an exit-status-1 exit.
    print(f"EXIT:1:{{e.code}}" if isinstance(e.code, str) else f"EXIT:{{e.code}}")
except Exception as e:
    print(f"RAISED:{{type(e).__name__}}:{{e}}")
    traceback.print_exc()
"""


def _run(argv: list[str], prelude: str = "") -> str:
    """Drive ``main()`` with ``argv`` over a two-row local dataset; returns everything it printed.

    ``prelude`` is extra driver code run against the loaded module before ``main()``.
    """
    with tempfile.TemporaryDirectory() as directory:
        raw = os.path.join(directory, "raw")
        out = os.path.join(directory, "out")  # not created; the local-save branch is never reached
        Dataset.from_dict(
            {
                "prompt": [[{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}]],
                "text": ["a raw document"],
            }
        ).save_to_disk(raw)

        driver = _DRIVER.format(argv=["--input", raw, "--output", out, *argv], prelude=prelude)
        env = dict(os.environ)
        env["PYTHONPATH"] = REPO + os.pathsep + env.get("PYTHONPATH", "")
        proc = subprocess.run([sys.executable, "-c", driver], capture_output=True, text=True, env=env, cwd=REPO)
        return proc.stdout + proc.stderr


# The published pre-training command (agent-docs/training-methods/pretraining.md), minus the S3 paths:
# no masking flags, because raw text has no turns to mask.
_DOC_TEXT_ARGV = [
    "--model-name",
    "dummy-model",
    "--mode",
    "text",
    "--text-field",
    "text",
    "--max-length",
    "8192",
    "--pack-sequences",
    "--packing-strategy",
    "bfd",
    "--num-shards",
    "2",
]

# The published SFT pre-processing command (agent-docs/training-methods/sft.md, launch-recipes.md,
# scripts-reference.md): chat mode, no marker.
_DOC_CHAT_ARGV = ["--model-name", "dummy-model", "--max-length", "8192", "--num-shards", "2", "--pack-sequences"]


def test_the_documented_pretraining_invocation_runs():
    """It reaches preprocessing: the accelerate state is initialized, and no flag the user never
    passed is carried into a mode that rejects it."""
    output = _run(_DOC_TEXT_ARGV)

    assert "ACCELERATE_STATE_ERROR" not in output, (
        "prepare_dataset.py crashed on the accelerate logger before preprocessing — "
        "PartialState()/Accelerator() is not initialized in main():\n" + output
    )
    assert "REACHED_PREPROCESS" in output, (
        f"the documented pre-training command did not reach preprocessing:\n{output}"
    )


def test_asking_for_completion_masking_in_text_mode_still_fails():
    """The mode-aware default must not silence the applicability check: raw text renders no chat
    template, so there are no user turns to mask and the request is a mistake worth naming."""
    output = _run([*_DOC_TEXT_ARGV, "--train-on-completions-only", "--assistant-message-template", "<a>"])

    assert "REACHED_PREPROCESS" not in output, output
    assert "not applicable to mode='text'" in output, output


def test_chat_mode_without_a_response_marker_fails_before_the_input_is_fetched():
    """Masking is on by default in chat mode (matching training), and the labels are baked here —
    so the marker is required, and the run must be refused while it still costs nothing."""
    output = _run(_DOC_CHAT_ARGV)

    assert "assistant-message-template" in output, output
    assert "REACHED_LOAD" not in output, f"the refusal came after the input dataset was fetched:\n{output}"
    assert "REACHED_PREPROCESS" not in output, output


def test_the_documented_sft_invocation_runs_once_the_marker_is_supplied():
    """The other side of that refusal: with the marker, the same command runs — the guard is about
    an unbuildable pair, not about chat mode."""
    output = _run([*_DOC_CHAT_ARGV, "--assistant-message-template", "<|im_start|>assistant\n"])

    assert "REACHED_PREPROCESS" in output, output


def test_a_multi_process_launch_is_refused_before_the_input_is_fetched():
    """Under torchrun every rank would run the publish against the same destination — one S3 key,
    local directory or Hub repo — so a launch with more than one process is refused up front."""
    output = _run(_DOC_TEXT_ARGV, prelude="mod.PartialState = lambda: types.SimpleNamespace(num_processes=2)")

    assert "EXIT:1" in output, output
    assert "single-process tool" in output, output
    assert "REACHED_LOAD" not in output, f"the refusal came after the input dataset was fetched:\n{output}"


def _save_tiny_tokenizer(directory: str) -> None:
    vocab = {"<unk>": 0, "<eos>": 1, **{word: i + 2 for i, word in enumerate(_WORDS)}}
    backend = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="<unk>", eos_token="<eos>", pad_token="<eos>"
    ).save_pretrained(directory)


def test_a_sharded_publish_is_partitioned_disjointly_across_data_parallel_ranks():
    """The real script, unstubbed, on a local raw-text corpus with ``--num-shards 4``: the published
    artifact must load through ``load_preprocessed_dataset`` for each of 4 data-parallel ranks, and
    the four slices must be a partition of the corpus — a gap silently drops data, an overlap silently
    double-trains it, and neither shows up as an error at training time."""
    from src.data.sources.loading import load_preprocessed_dataset

    PartialState()  # the loaders log through accelerate's rank-aware logger
    num_shards, num_docs = 4, 16
    with tempfile.TemporaryDirectory() as directory:
        raw, tokenizer_dir, out = (os.path.join(directory, name) for name in ("raw", "tok", "out"))
        _save_tiny_tokenizer(tokenizer_dir)
        # Each document tokenizes to a distinct id sequence, so a row is identifiable after tokenization.
        Dataset.from_dict({"text": [f"doc {i}" for i in range(num_docs)]}).save_to_disk(raw)
        argv = [
            sys.executable,
            _SCRIPT,
            *("--input", raw, "--output", out, "--model-name", tokenizer_dir),
            *("--mode", "text", "--text-field", "text", "--max-length", "8"),
            *("--num-shards", str(num_shards), "--test-size", "0.25"),
        ]
        env = {**os.environ, "PYTHONPATH": REPO + os.pathsep + os.environ.get("PYTHONPATH", "")}
        proc = subprocess.run(argv, capture_output=True, text=True, env=env, cwd=REPO)
        assert proc.returncode == 0, proc.stdout + proc.stderr
        with open(os.path.join(out, "metadata.json")) as f:
            assert json.load(f)["num_shards"] == num_shards

        seen: dict[str, list[tuple[int, ...]]] = {"train": [], "test": []}
        for rank in range(num_shards):
            loaded = load_preprocessed_dataset(out, data_parallel_rank=rank, data_parallel_size=num_shards)
            for split in seen:
                rows = [tuple(ids) for ids in loaded[split]["input_ids"]]
                assert rows, f"rank {rank} was handed an empty {split} split"
                seen[split].extend(rows)
        for split, rows in seen.items():
            assert len(rows) == len(set(rows)), f"{split}: a row reached two ranks"
        assert len(seen["train"]) + len(seen["test"]) == num_docs, "the ranks' slices do not add up to the corpus"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
