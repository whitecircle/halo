#!/usr/bin/env python
"""``prepare_dataset.py`` must run the invocations the docs publish, and fail fast when it cannot.

What this drives through the real ``main()`` in a fresh subprocess (so the accelerate singleton
starts unset, as in a real CLI run):

* **The documented raw-text pre-training command.** The completion-masking default must resolve per
  ``--mode``: a flat default-on is rejected by ``PreprocessingConfig`` as inapplicable to
  ``mode='text'``, which would exit every documented pre-training invocation on a flag its user never
  passed.
* **An explicit masking flag in text mode still fails.** The mode-aware default must not turn the
  applicability check into a no-op — asking for something ``mode='text'`` cannot do stays loud.
* **Chat mode with no response marker fails BEFORE the input is fetched.** The labels are baked
  into the artifact here, so masking without a marker is unbuildable, and the config that owns the
  pair is built from the command line before any download.
* **A column flag naming no input column fails before tokenization**, naming the flag and the
  columns the input carries.

Plus the artifact contract, unstubbed: a ``--num_shards N`` publish must be what the training-side
sharded loader hands ``N`` data-parallel ranks — every row exactly once, none twice; and the bake
keeps no intermediate map cache behind and stores labels at the width of ``input_ids``.

    python tests/cpu/data/test_prepare_dataset_script.py
"""

import json
import os
import subprocess
import sys
import tempfile

import pytest
from datasets import Dataset, DatasetDict

from halo.data.preprocessed import publishing
from halo.data.preprocessed.metadata import PreprocessingConfig
from halo.data.sources.loading import load_preprocessed_dataset
from tests.common.repo import REPO_ROOT
from tests.common.tiny_models import word_level_tokenizer

_SCRIPT = REPO_ROOT / "scripts/before_training/prepare_dataset.py"

# A real fast tokenizer built in-process, so the unstubbed run needs no download: one id per word.
_WORDS = ("doc", *(str(i) for i in range(32)))

# Runs in a subprocess: load the real script as a module, stub only the heavy/network steps
# (tokenizer load + preprocessing) so the offline load path is exercised, then call main().
# Reaching the stubbed preprocess proves load_input_dataset() ran; the stubbed load reports whether a
# rejection came before or after the input was fetched.
_DRIVER = """
import sys, traceback, types

from halo.data.preprocessed import publishing
from tests.common.repo import load_script_module

mod = load_script_module("scripts/before_training/prepare_dataset.py")


class _ReachedPreprocess(Exception):
    pass


publishing.setup_tokenizer = lambda args: object()  # never used (preprocess is stubbed) — avoids any download
publishing.AutoConfig = types.SimpleNamespace(
    from_pretrained=lambda *args, **kwargs: types.SimpleNamespace(model_type="qwen3")
)

real_load = publishing.load_input_dataset


def _load(args):
    print("REACHED_LOAD")
    return real_load(args)


def _boom(**kwargs):
    raise _ReachedPreprocess()


publishing.load_input_dataset = _load
publishing.preprocess_dataset = _boom
{prelude}
sys.argv = ["prepare_dataset.py"] + {argv!r}
try:
    mod.main()
except _ReachedPreprocess:
    print("REACHED_PREPROCESS")
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
        env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
        proc = subprocess.run([sys.executable, "-c", driver], capture_output=True, text=True, env=env, cwd=REPO_ROOT)
        return proc.stdout + proc.stderr


# The published pre-training command (agent-docs/training-methods/pretraining.md), minus the S3 paths:
# no masking flags, because raw text has no turns to mask.
_DOC_TEXT_ARGV = [
    "--model_name_or_path",
    "dummy-model",
    "--mode",
    "text",
    "--text_field",
    "text",
    "--max_length",
    "8192",
    "--pack_sequences",
    "--packing_strategy",
    "bfd",
    "--num_shards",
    "2",
]

# The published SFT pre-processing command (agent-docs/training-methods/sft.md, launch-recipes.md,
# scripts-reference.md): chat mode, no marker.
_DOC_CHAT_ARGV = [
    "--model_name_or_path",
    "dummy-model",
    "--max_length",
    "8192",
    "--num_shards",
    "2",
    "--pack_sequences",
]


def test_the_documented_pretraining_invocation_runs():
    """It reaches preprocessing: no flag the user never passed is carried into a mode that rejects it."""
    output = _run(_DOC_TEXT_ARGV)

    assert "REACHED_PREPROCESS" in output, (
        f"the documented pre-training command did not reach preprocessing:\n{output}"
    )


def test_asking_for_completion_masking_in_text_mode_still_fails():
    """The mode-aware default must not silence the applicability check: raw text renders no chat
    template, so there are no user turns to mask and the request is a mistake worth naming."""
    output = _run([*_DOC_TEXT_ARGV, "--train_on_completions_only", "--assistant_message_template", "<a>"])

    assert "REACHED_PREPROCESS" not in output, output
    assert "not applicable to mode='text'" in output, output


def test_chat_mode_without_a_response_marker_fails_before_the_input_is_fetched():
    """Masking is on by default in chat mode (matching training), and the labels are baked here —
    so the marker is required, and the run must be refused while it still costs nothing."""
    output = _run(_DOC_CHAT_ARGV)

    assert "requires assistant_message_template" in output, output
    assert "REACHED_LOAD" not in output, f"the refusal came after the input dataset was fetched:\n{output}"
    assert "REACHED_PREPROCESS" not in output, output


def test_the_documented_sft_invocation_runs_once_the_marker_is_supplied():
    """The other side of that refusal: with the marker, the same command runs — the guard is about
    an unbuildable pair, not about chat mode."""
    output = _run([*_DOC_CHAT_ARGV, "--assistant_message_template", "<|im_start|>assistant\n"])

    assert "REACHED_PREPROCESS" in output, output


@pytest.mark.parametrize(
    ("argv", "knob", "column"),
    [
        (
            [*_DOC_CHAT_ARGV, "--assistant_message_template", "<a>", "--conversation_field", "messages"],
            "conversation_field",
            "messages",
        ),
        ([*_DOC_TEXT_ARGV, "--text_field", "body"], "text_field", "body"),
        ([*_DOC_CHAT_ARGV, "--assistant_message_template", "<a>", "--tools_field", "tools"], "tools_field", "tools"),
    ],
    ids=["conversation", "text", "tools"],
)
def test_a_column_flag_naming_no_input_column_is_refused_before_tokenization(argv, knob, column):
    """A ``messages`` corpus under the default ``--conversation_field prompt`` (and its text-mode and
    tools twins) otherwise dies as a bare ``KeyError`` in a map worker, or bakes every row without the
    column; the refusal names the knob, the column and what the input does carry."""
    output = _run(argv)

    assert "REACHED_LOAD" in output, output
    assert "REACHED_PREPROCESS" not in output, f"the input reached tokenization with {knob}={column}:\n{output}"
    assert f"{knob}='{column}' names a column" in output, output
    assert "available columns: ['prompt', 'text']" in output, output


@pytest.mark.parametrize(
    ("splits", "refused"), [(("train", "validation"), True), (("train", "test", "validation"), False)]
)
def test_the_held_out_split_the_artifact_bakes_is_checked(splits, refused):
    """A source with no ``test`` split bakes its ``validation`` split as the test split, so a column
    missing only there would still die in that split's map; a ``validation`` split beside a ``test``
    one is never baked, so its schema is not checked."""
    conversation = [[{"role": "user", "content": "hi"}]]
    dataset = DatasetDict(
        {name: Dataset.from_dict({"messages" if name == "validation" else "prompt": conversation}) for name in splits}
    )
    config = PreprocessingConfig(model_name_or_path="m")

    if not refused:
        publishing.require_declared_columns(dataset, config, "src")
        return
    with pytest.raises(ValueError, match=r"conversation_field='prompt' .*\['validation'\]"):
        publishing.require_declared_columns(dataset, config, "src")


def test_a_multi_process_launch_is_refused_before_the_input_is_fetched():
    """Under torchrun every rank would run the publish against the same destination — one S3 key,
    local directory or Hub repo — so a launch with more than one process is refused up front."""
    output = _run(_DOC_TEXT_ARGV, prelude="publishing.PartialState = lambda: types.SimpleNamespace(num_processes=2)")

    assert "EXIT:1" in output, output
    assert "single-process tool" in output, output
    assert "REACHED_LOAD" not in output, f"the refusal came after the input dataset was fetched:\n{output}"


def test_packing_a_compressed_kv_model_is_refused_before_the_input_is_fetched():
    """An offline-packed artifact for a model whose compressor windows cut by row index trains
    nothing the collator would accept, so ``--pack_sequences`` is refused off the model config."""
    prelude = (
        "from transformers.models.deepseek_v4 import DeepseekV4Config\n"
        "from tests.common.models import TINY_DSV4_CONFIG\n"
        "publishing.AutoConfig = types.SimpleNamespace(\n"
        "    from_pretrained=lambda *a, **k: DeepseekV4Config(**TINY_DSV4_CONFIG)\n"
        ")"
    )
    output = _run(_DOC_TEXT_ARGV, prelude=prelude)

    assert "--pack_sequences is refused" in output, output
    assert "REACHED_LOAD" not in output, f"the refusal came after the input dataset was fetched:\n{output}"


def test_a_sharded_publish_is_partitioned_disjointly_across_data_parallel_ranks():
    """The real script, unstubbed, on a local raw-text corpus with ``--num_shards 4``: the published
    artifact must load through ``load_preprocessed_dataset`` for each of 4 data-parallel ranks, and
    the four slices must be a partition of the corpus — a gap silently drops data, an overlap silently
    double-trains it, and neither shows up as an error at training time."""
    num_shards, num_docs = 4, 16
    with tempfile.TemporaryDirectory() as directory:
        raw, tokenizer_dir, out = (os.path.join(directory, name) for name in ("raw", "tok", "out"))
        word_level_tokenizer(_WORDS).save_pretrained(tokenizer_dir)
        # Each document tokenizes to a distinct id sequence, so a row is identifiable after tokenization.
        Dataset.from_dict({"text": [f"doc {i}" for i in range(num_docs)]}).save_to_disk(raw)
        argv = [
            sys.executable,
            _SCRIPT,
            *("--input", raw, "--output", out, "--model_name_or_path", tokenizer_dir),
            *("--mode", "text", "--text_field", "text", "--max_length", "8"),
            *("--num_shards", str(num_shards), "--test_size", "0.25"),
        ]
        env = {**os.environ, "PYTHONPATH": str(REPO_ROOT) + os.pathsep + os.environ.get("PYTHONPATH", "")}
        proc = subprocess.run(argv, capture_output=True, text=True, env=env, cwd=REPO_ROOT)
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


def test_the_bake_leaves_no_map_cache_behind_and_stores_labels_at_input_width():
    """Each tokenize/filter/label/pack map writes a full copy of the corpus; the single-process tool must not
    leave them in HF_DATASETS_CACHE, where nothing reuses or removes them. Labels inferred as int64 beside
    int32 ``input_ids`` would be most of the artifact's bytes."""
    with tempfile.TemporaryDirectory() as directory:
        raw, tokenizer_dir, out, cache = (os.path.join(directory, name) for name in ("raw", "tok", "out", "cache"))
        word_level_tokenizer(_WORDS).save_pretrained(tokenizer_dir)
        # --pack_sequences reads the model config (the compressed-KV refusal); a stock family's suffices.
        with open(os.path.join(tokenizer_dir, "config.json"), "w") as f:
            json.dump({"model_type": "qwen3"}, f)
        Dataset.from_dict({"text": [f"doc {i}" for i in range(16)]}).save_to_disk(raw)
        os.makedirs(cache)
        argv = [
            sys.executable,
            _SCRIPT,
            *("--input", raw, "--output", out, "--model_name_or_path", tokenizer_dir),
            *("--mode", "text", "--text_field", "text", "--max_length", "8", "--pack_sequences"),
        ]
        env = {
            **os.environ,
            "PYTHONPATH": str(REPO_ROOT) + os.pathsep + os.environ.get("PYTHONPATH", ""),
            "HF_DATASETS_CACHE": cache,
        }
        proc = subprocess.run(argv, capture_output=True, text=True, env=env, cwd=REPO_ROOT)
        assert proc.returncode == 0, proc.stdout + proc.stderr

        assert os.listdir(cache) == [], f"the bake left map caches behind: {os.listdir(cache)}"
        features = load_preprocessed_dataset(out)["train"].features
        assert features["labels"] == features["input_ids"], features


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
