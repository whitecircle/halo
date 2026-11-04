#!/usr/bin/env python
"""``prepare_dataset.py`` runs the invocations the docs publish and fails fast when it cannot, through the real
``main()`` in a fresh subprocess so the accelerate singleton starts unset. A ``--num_shards N`` publish must be
what the sharded loader hands ``N`` DP ranks, every row exactly once."""

import json
import os
import subprocess
import sys
import tempfile

import pytest
from datasets import Dataset, DatasetDict

from halo.data.preprocessed import prepare_dataset_tool
from halo.data.preprocessed.metadata import PreprocessingConfig
from halo.data.sources.loading import load_preprocessed_dataset
from tests.common.repo import REPO_ROOT
from tests.common.tiny_models import word_level_tokenizer

_SCRIPT = REPO_ROOT / "scripts/before_training/prepare_dataset.py"

# A real fast tokenizer built in-process, so the unstubbed run needs no download: one id per word.
_WORDS = ("doc", *(str(i) for i in range(32)))

# Runs in a subprocess: stub only the tokenizer load and preprocessing. Reaching the stubbed preprocess proves the load
# ran; the stubbed load reports whether a rejection came before the input was fetched.
_DRIVER = """
import sys, traceback, types

from halo.data.preprocessed import prepare_dataset_tool
from tests.common.repo import load_script_module

mod = load_script_module("scripts/before_training/prepare_dataset.py")


class _ReachedPreprocess(Exception):
    pass


prepare_dataset_tool.load_tokenizer = lambda args: object()  # never used (preprocess is stubbed) — avoids any download
prepare_dataset_tool.AutoConfig = types.SimpleNamespace(
    from_pretrained=lambda *args, **kwargs: types.SimpleNamespace(model_type="qwen3")
)

real_load = prepare_dataset_tool.load_input_dataset


def _load(args):
    print("REACHED_LOAD")
    return real_load(args)


def _boom(**kwargs):
    raise _ReachedPreprocess()


prepare_dataset_tool.load_input_dataset = _load
prepare_dataset_tool.preprocess_dataset = _boom
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
    """Drive ``main()`` with ``argv`` over a two-row local dataset and return what it printed; ``prelude`` runs
    against the loaded module first."""
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


# The published pre-training command (agent-docs/training-methods/pretraining.md), minus the S3 paths.
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

# The published SFT pre-processing command (agent-docs/training-methods/sft.md): chat mode, no marker.
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
    """Raw text renders no chat template, so there are no user turns to mask and the request is a mistake."""
    output = _run([*_DOC_TEXT_ARGV, "--train_on_completions_only", "--assistant_message_template", "<a>"])

    assert "REACHED_PREPROCESS" not in output, output
    assert "not applicable to mode='text'" in output, output


def test_chat_mode_without_a_response_marker_fails_before_the_input_is_fetched():
    """The labels are baked here, so the marker is required, and the run is refused while that costs nothing."""
    output = _run(_DOC_CHAT_ARGV)

    assert "requires assistant_message_template" in output, output
    assert "REACHED_LOAD" not in output, f"the refusal came after the input dataset was fetched:\n{output}"
    assert "REACHED_PREPROCESS" not in output, output


def test_the_documented_sft_invocation_runs_once_the_marker_is_supplied():
    """With the marker, the same command runs: the guard is about an unbuildable pair."""
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
    """Otherwise a bare ``KeyError`` in a map worker, or every row baked without the column."""
    output = _run(argv)

    assert "REACHED_LOAD" in output, output
    assert "REACHED_PREPROCESS" not in output, f"the input reached tokenization with {knob}={column}:\n{output}"
    assert f"{knob}='{column}' names a column" in output, output
    assert "available columns: ['prompt', 'text']" in output, output


@pytest.mark.parametrize(
    ("splits", "refused"), [(("train", "validation"), True), (("train", "test", "validation"), False)]
)
def test_the_held_out_split_the_artifact_bakes_is_checked(splits, refused):
    """A source with no ``test`` bakes its ``validation`` as the test split, so a column missing only there is
    caught; a ``validation`` beside a ``test`` is never baked."""
    conversation = [[{"role": "user", "content": "hi"}]]
    dataset = DatasetDict(
        {name: Dataset.from_dict({"messages" if name == "validation" else "prompt": conversation}) for name in splits}
    )
    config = PreprocessingConfig(model_name_or_path="m")

    if not refused:
        prepare_dataset_tool.require_declared_columns(dataset, config, "src")
        return
    with pytest.raises(ValueError, match=r"conversation_field='prompt' .*\['validation'\]"):
        prepare_dataset_tool.require_declared_columns(dataset, config, "src")


def test_a_multi_process_launch_is_refused_before_the_input_is_fetched():
    """Every rank would run the publish against the same destination."""
    output = _run(
        _DOC_TEXT_ARGV, prelude="prepare_dataset_tool.PartialState = lambda: types.SimpleNamespace(num_processes=2)"
    )

    assert "EXIT:1" in output, output
    assert "single-process tool" in output, output
    assert "REACHED_LOAD" not in output, f"the refusal came after the input dataset was fetched:\n{output}"


def test_packing_a_compressed_kv_model_is_refused_before_the_input_is_fetched():
    """A compressor whose windows cut by row index trains nothing the collator would accept from an offline-packed
    artifact."""
    prelude = (
        "from transformers.models.deepseek_v4 import DeepseekV4Config\n"
        "from tests.common.models import TINY_DSV4_CONFIG\n"
        "prepare_dataset_tool.AutoConfig = types.SimpleNamespace(\n"
        "    from_pretrained=lambda *a, **k: DeepseekV4Config(**TINY_DSV4_CONFIG)\n"
        ")"
    )
    output = _run(_DOC_TEXT_ARGV, prelude=prelude)

    assert "--pack_sequences is refused" in output, output
    assert "REACHED_LOAD" not in output, f"the refusal came after the input dataset was fetched:\n{output}"


_RING_PRELUDE = (
    "from tests.common.tiny_models import TINY_BAILING_RING_FAMILY, tiny_family_model\n"
    "prepare_dataset_tool.AutoConfig = types.SimpleNamespace(\n"
    "    from_pretrained=lambda *a, **k: tiny_family_model(TINY_BAILING_RING_FAMILY).config\n"
    ")"
)


def test_packing_a_model_training_refuses_is_refused_at_the_bake_unless_opted_in():
    refused = _run(_DOC_TEXT_ARGV, prelude=_RING_PRELUDE)
    assert "allow_cross_document_linear_attention" in refused, refused
    assert "REACHED_LOAD" not in refused, f"the refusal came after the input dataset was fetched:\n{refused}"

    opted_in = _run([*_DOC_TEXT_ARGV, "--allow_cross_document_linear_attention", "true"], prelude=_RING_PRELUDE)
    assert "REACHED_LOAD" in opted_in or "REACHED_PREPROCESS" in opted_in, opted_in


def test_the_bake_opt_in_needs_packed_rows():
    argv = [arg for arg in _DOC_TEXT_ARGV if arg != "--pack_sequences"]
    output = _run([*argv, "--allow_cross_document_linear_attention", "true"])
    assert "needs --pack_sequences" in output, output


def test_a_sharded_publish_is_partitioned_disjointly_across_data_parallel_ranks():
    """The real script, unstubbed, with ``--num_shards 4``: a gap silently drops data and an overlap double-trains
    it."""
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


def test_the_bake_leaves_no_map_cache_behind_and_stores_no_full_sequence_labels():
    """Each map writes a full corpus copy that nothing reuses in HF_DATASETS_CACHE. A packed full-sequence
    artifact's collator derives labels and mask from ``input_ids``, so stored copies would double it."""
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
        assert set(features) == {"input_ids", "seq_lengths"}, features


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
