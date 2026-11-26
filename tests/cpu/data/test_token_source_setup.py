#!/usr/bin/env python
"""The token_sources surface of the SFT script: the config's parsing and refusals, the training options a token run
needs, the mix laid out for the run's length and rows per step and held to the fingerprint the resumed checkpoint's
trainer state carries, the held-out folders, the one token width, the s3:// sources fetched once per node's input scope
with their errors named, the vocabulary check and tokenizer warning, every data refusal landing before the model load,
and a run without token_sources refused a token run's checkpoint."""

import contextlib
import datetime
import hashlib
import logging
import os
import re
import shutil
import sys
import types
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
from transformers import Qwen3Config, TrainerState
from trl import SFTConfig

from halo.callbacks.token_mix_checkpoint import TokenMixCheckpointCallback
from halo.checkpoint.format import TRAINER_STATE_FILE
from halo.configs.parser import HaloArgumentParser
from halo.configs.sft import SFTScriptArguments
from halo.data.sources import token_shards
from halo.data.sources.dataset_cache import compute_etag_fingerprint
from halo.data.sources.token_mix import TokenSource
from halo.entrypoints import data_setup
from halo.entrypoints.data_setup import (
    load_token_sources,
    reject_unsupported_token_source_options,
    verify_token_sources_fit_model,
)
from tests.common.datatrove_shards import write_token_folder
from tests.common.gloo import run_gloo_ranks
from tests.common.repo import load_script_module

MAX_LENGTH = 8
_MODEL_ID = "stub/qwen3"
PG_TIMEOUT = datetime.timedelta(seconds=120)


def _folder(path, tokens: int = 200, seed: int = 0):
    rng = np.random.default_rng(seed)
    documents = [rng.integers(3, 90, size=size).tolist() for size in (tokens // 4, tokens // 4, tokens // 2)]
    return write_token_folder(path, {"00000.ds": documents}, token_bytes=2)


def _args(**fields) -> SFTScriptArguments:
    return SFTScriptArguments(train_on_completions_only=False, **fields)


def _training(tmp_path, **fields) -> SFTConfig:
    clean = {
        "max_steps": 5,
        "train_sampling_strategy": "sequential",
        "packing": True,
        "max_length": MAX_LENGTH,
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 3,
    }
    return SFTConfig(output_dir=str(tmp_path / "out"), use_cpu=True, bf16=False, **(clean | fields))


def test_token_sources_parse_from_yaml_into_sources(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text(
        "train_on_completions_only: false\n"
        "token_sources:\n- {name: web, path: /data/web, weight: 7}\n- {name: code, path: s3://b/code, weight: 3}\n"
        "eval_token_sources:\n- {name: web-val, path: /data/web-val}\n"
    )
    (args,) = HaloArgumentParser((SFTScriptArguments,)).parse_yaml_file(str(config))
    assert args.token_source_specs == [TokenSource("web", "/data/web", 7), TokenSource("code", "s3://b/code", 3)]
    assert args.eval_token_source_specs == [TokenSource("web-val", "/data/web-val")]


_CONFIG_REFUSALS = {
    "not a list": ({"token_sources": {"name": "a", "path": "/a"}}, "must be a non-empty list of mappings"),
    "no source": ({"token_sources": []}, "must be a non-empty list of mappings"),
    "not a mapping": ({"token_sources": ["/a"]}, r"token_sources\[0\] must be a mapping"),
    "unknown key": ({"token_sources": [{"name": "a", "path": "/a", "weigth": 1}]}, r"unknown keys \['weigth'\]"),
    "no path": ({"token_sources": [{"name": "a"}]}, r"token_sources\[0\] lacks \['path'\]"),
    "blank name": ({"token_sources": [{"name": " ", "path": "/a"}]}, r"token_sources\[0\]\.name must be a non-blank"),
    "blank path": ({"token_sources": [{"name": "a", "path": ""}]}, r"token_sources\[0\]\.path must be a non-blank"),
    "some weighted": (
        {"token_sources": [{"name": "a", "path": "/a", "weight": 1}, {"name": "b", "path": "/b"}]},
        r"weighs \['a'\] and leaves the others unweighted",
    ),
    "repeated name": (
        {"token_sources": [{"name": "a", "path": "/a"}, {"name": "a", "path": "/b"}]},
        r"repeats the names \['a'\]",
    ),
    "repeated eval name": (
        {
            "token_sources": [{"name": "a", "path": "/a"}],
            "eval_token_sources": [{"name": "v", "path": "/v"}, {"name": "v", "path": "/w"}],
        },
        r"eval_token_sources repeats the names \['v'\]",
    ),
    "zero weight": ({"token_sources": [{"name": "a", "path": "/a", "weight": 0}]}, r"\[0\]\.weight must be > 0"),
    "nan weight": (
        {"token_sources": [{"name": "a", "path": "/a", "weight": float("nan")}]},
        r"\[0\]\.weight must be a finite number, got nan",
    ),
    "bool weight": (
        {"token_sources": [{"name": "a", "path": "/a", "weight": True}]},
        r"\[0\]\.weight must be a finite number, got True",
    ),
    "eval alone": ({"eval_token_sources": [{"name": "v", "path": "/v"}]}, "set token_sources"),
    "weighted eval": (
        {
            "token_sources": [{"name": "a", "path": "/a"}],
            "eval_token_sources": [{"name": "v", "path": "/v", "weight": 2}],
        },
        r"eval_token_sources \['v'\] carry a weight, but a held-out folder is read whole at every evaluation",
    ),
    "a dataset beside": (
        {"token_sources": [{"name": "a", "path": "/a"}], "dataset": "org/data"},
        r"silently ignored\): \['dataset'\]",
    ),
    "a conversation column": (
        {"token_sources": [{"name": "a", "path": "/a"}], "conversation_field": "messages"},
        r"\['conversation_field'\]",
    ),
    "tree packing": ({"token_sources": [{"name": "a", "path": "/a"}], "tree_packing": True}, r"\['tree_packing'\]"),
    "a vision field": ({"token_sources": [{"name": "a", "path": "/a"}], "min_pixels": 3136}, r"\['min_pixels'\]"),
    "a template column": (
        {"token_sources": [{"name": "a", "path": "/a"}], "tools_field": "tools"},
        r"\['tools_field'\]",
    ),
}


def test_a_token_source_config_the_mix_cannot_honor_is_refused(subtests):
    for label, (fields, message) in _CONFIG_REFUSALS.items():
        with subtests.test(label), pytest.raises(ValueError, match=message):
            _args(**fields)
    with subtests.test("completion masking left on"), pytest.raises(ValueError, match="no completion marks"):
        SFTScriptArguments(token_sources=[{"name": "a", "path": "/a"}])


_EVALUATION_REQUIREMENT = "eval_token_sources when the run evaluates, by eval_strategy, eval_on_start or the MTP"
_GATE = {"mtp_loss_weight": 0.1, "mtp_train_scope": "head_only"}


def _dist(context_parallel_size: int = 1):
    return types.SimpleNamespace(context_parallel_size=context_parallel_size)


def test_the_training_options_a_token_run_needs_are_named_when_missing(tmp_path):
    token_run = _args(token_sources=[{"name": "a", "path": "/a"}])
    reject_unsupported_token_source_options(token_run, _training(tmp_path), _dist())
    unmet = _training(
        tmp_path,
        max_steps=-1,
        train_sampling_strategy="random",
        packing=False,
        packing_strategy="wrapped",
        eval_packing=False,
        padding_free=True,
        ignore_data_skip=True,
        auto_find_batch_size=True,
        accelerator_config={"split_batches": True},
        eval_strategy="steps",
        metric_for_best_model="loss",
    )
    with pytest.raises(ValueError) as refused:
        reject_unsupported_token_source_options(token_run, unmet, _dist(context_parallel_size=2))
    for requirement in (
        "max_steps set",
        "sequential",
        "packing: true",
        "packing_strategy",
        "eval_packing",
        "padding_free off",
        "ignore_data_skip off",
        "auto_find_batch_size off",
        "dispatch_batches and split_batches off and even_batches on",
        "context_parallel_size: 1",
        _EVALUATION_REQUIREMENT,
        "metric_for_best_model unset or naming a held-out folder's eval_<name>_loss",
    ):
        assert requirement in str(refused.value), requirement


def test_each_loader_knob_that_breaks_the_mix_is_refused_alone(tmp_path, subtests):
    """dispatch_batches and split_batches would leave a rank reading other rows than rows_per_step and the resume skip
    count; even_batches off hangs the evaluation of a held-out folder that does not split evenly over the ranks."""
    token_run = _args(token_sources=[{"name": "a", "path": "/a"}])
    requirement = "split_batches off and even_batches on .*the resume skip count.*hangs the evaluation"
    for knob in ({"dispatch_batches": True}, {"split_batches": True}, {"even_batches": False}):
        with subtests.test(str(knob)), pytest.raises(ValueError, match=requirement):
            reject_unsupported_token_source_options(token_run, _training(tmp_path, accelerator_config=knob), _dist())


def test_the_best_model_metric_must_be_a_held_out_folder_loss(tmp_path, subtests):
    """HF's post-init sets it to loss under load_best_model_at_end; a token run logs only eval_<name>_loss."""
    held_out = _args(token_sources=[{"name": "a", "path": "/a"}], eval_token_sources=[{"name": "v", "path": "/v"}])
    evaluated = {"eval_strategy": "steps", "save_strategy": "steps"}
    for metric in ("eval_v_loss", "v_loss"):
        with subtests.test(metric):
            training = _training(tmp_path, metric_for_best_model=metric, **evaluated)
            reject_unsupported_token_source_options(held_out, training, _dist())
    with subtests.test("load_best_model_at_end"), pytest.raises(ValueError, match="metric_for_best_model unset"):
        training = _training(tmp_path, load_best_model_at_end=True, **evaluated)
        reject_unsupported_token_source_options(held_out, training, _dist())


def test_a_token_run_evaluates_only_on_held_out_folders(tmp_path, subtests):
    held_out = {"eval_token_sources": [{"name": "v", "path": "/v"}]}
    for label, fields, training in (
        ("eval_strategy", {}, {"eval_strategy": "steps"}),
        ("eval_on_start", {}, {"eval_on_start": True}),
        ("the acceptance gate", _GATE, {}),
    ):
        args = _args(token_sources=[{"name": "a", "path": "/a"}], **fields)
        with subtests.test(label), pytest.raises(ValueError, match=_EVALUATION_REQUIREMENT):
            reject_unsupported_token_source_options(args, _training(tmp_path, **training), _dist())
        with subtests.test(f"{label} with held-out folders"):
            args = _args(token_sources=[{"name": "a", "path": "/a"}], **held_out, **fields)
            reject_unsupported_token_source_options(args, _training(tmp_path, **training), _dist())


def test_the_acceptance_gate_takes_one_held_out_folder(tmp_path):
    """The gate sets the last reading of its key against the first, and each folder logs under the same key."""
    two = [{"name": "v", "path": "/v"}, {"name": "w", "path": "/w"}]
    ungated = _args(token_sources=[{"name": "a", "path": "/a"}], eval_token_sources=two)
    reject_unsupported_token_source_options(ungated, _training(tmp_path, eval_strategy="steps"), _dist())
    gated = _args(token_sources=[{"name": "a", "path": "/a"}], eval_token_sources=two, **_GATE)
    with pytest.raises(ValueError, match="one eval_token_sources folder under the MTP acceptance gate"):
        reject_unsupported_token_source_options(gated, _training(tmp_path), _dist())


def _parallelism(data_parallel_size: int = 2, pp_size: int = 1):
    return types.SimpleNamespace(data_parallel_size=data_parallel_size, pp_size=pp_size)


def test_the_mix_covers_the_run_and_draws_with_the_data_seed(tmp_path):
    args = _args(token_sources=[{"name": "a", "path": str(_folder(tmp_path / "a"))}])
    mix, held_out = load_token_sources(args, _training(tmp_path), _parallelism(), resume_checkpoint=None)
    assert mix.rows_per_step == 2 * 3 * 2 and len(mix) == 5 * mix.rows_per_step
    assert mix.max_length == MAX_LENGTH and held_out is None
    assert mix.seed == _training(tmp_path).seed
    seeded, _ = load_token_sources(args, _training(tmp_path, data_seed=99), _parallelism(), resume_checkpoint=None)
    assert seeded.seed == 99


def test_a_pipeline_step_reads_the_per_device_batch_split_into_microbatches(tmp_path):
    """Under pipeline parallelism the accumulation becomes the microbatch count and the trainer accumulates once."""
    args = _args(token_sources=[{"name": "a", "path": str(_folder(tmp_path / "a"))}])
    mix, _ = load_token_sources(args, _training(tmp_path), _parallelism(pp_size=2), resume_checkpoint=None)
    assert mix.rows_per_step == 2 * 2 and len(mix) == 5 * mix.rows_per_step


def test_held_out_folders_are_read_whole_by_name(tmp_path):
    args = _args(
        token_sources=[{"name": "a", "path": str(_folder(tmp_path / "a"))}],
        eval_token_sources=[{"name": "a-val", "path": str(_folder(tmp_path / "v", tokens=40, seed=1))}],
    )
    _, held_out = load_token_sources(args, _training(tmp_path), _parallelism(), resume_checkpoint=None)
    assert list(held_out) == ["a-val"] and len(held_out["a-val"]) == 40 // MAX_LENGTH


def test_an_s3_source_is_read_from_its_cached_copy(tmp_path, monkeypatch):
    local = {
        "s3://bucket/tok/a": _folder(tmp_path / "cached"),
        "s3://bucket/tok/a-val": _folder(tmp_path / "cached-val", tokens=40, seed=1),
    }
    fetched = []

    def fetch(uri):
        fetched.append(uri)
        return local[uri]

    monkeypatch.setattr(data_setup, "fetch_s3_token_folder", fetch)
    args = _args(
        token_sources=[{"name": "a", "path": "s3://bucket/tok/a"}],
        eval_token_sources=[{"name": "a-val", "path": "s3://bucket/tok/a-val"}],
    )
    mix, held_out = load_token_sources(args, _training(tmp_path), _parallelism(), resume_checkpoint=None)
    assert fetched == list(local)
    assert mix.shards[0].folder == local["s3://bucket/tok/a"]
    assert held_out["a-val"].folder == local["s3://bucket/tok/a-val"]


def test_folders_of_two_token_widths_are_refused_as_two_tokenizers(tmp_path, subtests):
    """datatrove takes the width from the vocabulary size alone, so a trained or held-out folder of another width holds
    another tokenizer's ids."""
    narrow = {"name": "a", "path": str(_folder(tmp_path / "a"))}
    wide = str(write_token_folder(tmp_path / "wide", {"00000.ds": [[5] * 50, [6] * 50]}, token_bytes=4))
    cases = {
        "trained": ({"token_sources": [narrow, {"name": "b", "path": wide}]}, "token_sources 'b'"),
        "held out": (
            {"token_sources": [narrow], "eval_token_sources": [{"name": "v", "path": wide}]},
            "eval_token_sources 'v'",
        ),
    }
    for label, (fields, listed) in cases.items():
        refusal = (
            rf"different widths: token_sources 'a' 2 bytes, {listed} 4 bytes\. datatrove writes 2-byte ids up to a "
            r"vocabulary of 65,536 and 4-byte ids past it.*Re-tokenize every folder"
        )
        with subtests.test(label), pytest.raises(ValueError, match=refusal):
            load_token_sources(_args(**fields), _training(tmp_path), _parallelism(), resume_checkpoint=None)


def test_an_s3_source_that_cannot_be_fetched_is_named_in_the_error(tmp_path, monkeypatch):
    """S3's own errors, such as botocore's missing credentials, name neither the source nor its prefix."""

    class NoCredentialsError(Exception):
        pass

    def fetch(uri):
        raise NoCredentialsError("Unable to locate credentials")

    monkeypatch.setattr(data_setup, "fetch_s3_token_folder", fetch)
    args = _args(token_sources=[{"name": "web", "path": "s3://bucket/tok/web"}])
    with pytest.raises(
        RuntimeError, match=r"token source 'web' \(s3://bucket/tok/web\): NoCredentialsError: Unable to locate"
    ):
        load_token_sources(args, _training(tmp_path), _parallelism(), resume_checkpoint=None)


class _NodeBucket:
    """S3 at its client boundary: the prefix is a local folder of this rank's node bucket, every download logged."""

    root: Path

    def __init__(self, bucket: str):
        self.bucket = bucket

    def download_folder(self, key: str, local_path: str) -> str:
        with open(self.root.parent / "downloads.log", "a") as log:
            log.write(f"{os.environ['RANK']}\n")
        shutil.copytree(self.root / key, local_path)
        return local_path

    def content_fingerprint(self, key: str) -> str:
        files = sorted((self.root / key).iterdir())
        return compute_etag_fingerprint([(path.name, hashlib.md5(path.read_bytes()).hexdigest(), 0) for path in files])


def _s3_rank(rank: int, root: str, bucket_per_node: bool) -> None:
    """One rank of two imitated nodes of two ranks, each node with its own S3 cache and, under ``bucket_per_node``,
    its own view of the bucket."""
    node = rank // 2
    os.environ["LOCAL_RANK"] = str(rank % 2)
    _NodeBucket.root = Path(root) / f"bucket{node if bucket_per_node else 0}"
    args = _args(token_sources=[{"name": "web", "path": "s3://bucket/tok/web"}])
    training = types.SimpleNamespace(
        max_length=MAX_LENGTH,
        max_steps=5,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1,
        data_seed=None,
        seed=0,
    )
    with (
        mock.patch.object(token_shards, "S3Client", _NodeBucket),
        mock.patch.object(token_shards, "HALO_S3_DATASET_CACHE_DIR", f"{root}/cache{node}"),
    ):
        try:
            mix, _ = load_token_sources(args, training, _parallelism(4), resume_checkpoint=None)
            verdict = str(mix.shards[0].folder)
        except ValueError as error:
            verdict = str(error)
    Path(root, f"rank{rank}.txt").write_text(verdict)


@pytest.mark.parametrize("bucket_per_node", [False, True], ids=["one bucket", "node 1 sees another bucket state"])
def test_each_node_fetches_its_s3_copy_once_and_a_node_that_saw_other_files_fails_every_rank(
    tmp_path, bucket_per_node
):
    """Per-node input storage: each node's load rank downloads into its own cache, and its peer reads that copy."""
    rng = np.random.default_rng(0)
    documents = [rng.integers(3, 90, size=size).tolist() for size in (60, 40, 100)]
    write_token_folder(tmp_path / "bucket0" / "tok" / "web", {"00000.ds": documents}, token_bytes=2)
    write_token_folder(tmp_path / "bucket1" / "tok" / "web", {"00000.ds": documents[::-1]}, token_bytes=2)
    run_gloo_ranks(
        _s3_rank,
        4,
        str(tmp_path),
        bucket_per_node,
        pg_timeout=PG_TIMEOUT,
        env={"LOCAL_WORLD_SIZE": "2", "DIST_INPUT_SHARED_FILESYSTEM": "0"},
    )
    verdicts = [(tmp_path / f"rank{rank}.txt").read_text() for rank in range(4)]
    assert sorted((tmp_path / "downloads.log").read_text().split()) == ["0", "2"], "one download per node"
    if not bucket_per_node:
        assert [Path(verdict).relative_to(tmp_path).parts[0] for verdict in verdicts] == ["cache0"] * 2 + [
            "cache1"
        ] * 2
        return
    assert len(set(verdicts)) == 1
    assert "under token_sources: ranks [2, 3] (nodes [1]) differ from rank 0 in source 'web' content" in verdicts[0]


def _saved_checkpoint(path, fingerprint: dict | None) -> str:
    """A checkpoint directory holding the trainer state a save writes, with ``fingerprint`` exported by the run's
    callback as the base save collects it, or no callback state."""
    path.mkdir()
    exported = [] if fingerprint is None else [TokenMixCheckpointCallback(fingerprint)]
    TrainerState(stateful_callbacks=exported).save_to_json(str(path / TRAINER_STATE_FILE))
    return str(path)


def _checkpoint(tmp_path, args) -> str:
    """A checkpoint of a run of ``args`` at 2 × 3 rows on each of 2 data-parallel ranks, carrying its mix."""
    mix, _ = load_token_sources(args, _training(tmp_path), _parallelism(), resume_checkpoint=None)
    return _saved_checkpoint(tmp_path / "checkpoint-3", mix.fingerprint())


def test_a_resume_is_held_to_the_mix_its_checkpoint_trained_on(tmp_path, subtests):
    args = _args(token_sources=[{"name": "a", "path": str(_folder(tmp_path / "a"))}])
    unmarked = {
        "a trainer state without the mix": _saved_checkpoint(tmp_path / "checkpoint-1", None),
        "no trainer state": str(tmp_path / "absent"),
    }
    for label, checkpoint in unmarked.items():
        with (
            subtests.test(label),
            pytest.raises(ValueError, match=f"carries no token mix: its {TRAINER_STATE_FILE} holds no "),
        ):
            load_token_sources(args, _training(tmp_path), _parallelism(), resume_checkpoint=checkpoint)
    checkpoint = _checkpoint(tmp_path, args)
    # Four ranks of 1 × 3 rows keep the checkpoint's 12 rows per step; a longer run reads the same rows at each step.
    regrouped = _training(tmp_path, max_steps=9, per_device_train_batch_size=1)
    load_token_sources(args, regrouped, _parallelism(4), resume_checkpoint=checkpoint)
    with pytest.raises(ValueError, match="checkpoint was trained on: seed 42 → 7"):
        load_token_sources(args, _training(tmp_path, seed=7), _parallelism(), resume_checkpoint=checkpoint)


def test_a_resume_onto_another_global_batch_is_refused_with_the_layout_remedy(tmp_path):
    """Four ranks at the checkpoint's 2 × 3 rows read 24 rows a step: the resume would skip 24 per trained step."""
    args = _args(token_sources=[{"name": "a", "path": str(_folder(tmp_path / "a"))}])
    checkpoint = _checkpoint(tmp_path, args)
    with pytest.raises(ValueError, match=r"rows_per_step 12 → 24\..*halve one factor and double another"):
        load_token_sources(args, _training(tmp_path), _parallelism(4), resume_checkpoint=checkpoint)


def _warnings(caplog, mix, held_out, tokenizer_name: str) -> list[str]:
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger=data_setup.logger.name):
        verify_token_sources_fit_model(mix, held_out, 90, tokenizer_name)
    return [record.getMessage() for record in caplog.records if record.name == data_setup.logger.name]


def test_ids_past_the_vocabulary_are_refused(tmp_path):
    args = _args(token_sources=[{"name": "a", "path": str(_folder(tmp_path / "a"))}])
    mix, _ = load_token_sources(args, _training(tmp_path), _parallelism(), resume_checkpoint=None)
    highest = max(token for index in range(mix.shards[0].num_windows) for token in mix.shards[0][index]["input_ids"])
    verify_token_sources_fit_model(mix, None, highest + 1, "synthetic")
    with pytest.raises(ValueError, match=f"holds token id {highest}, past the model's vocabulary of {highest}"):
        verify_token_sources_fit_model(mix, None, highest, "synthetic")


def test_a_held_out_folder_alone_holding_an_id_past_the_vocabulary_is_refused(tmp_path):
    """The mix's ids stay under 90; the held-out folder's are all 95."""
    high = write_token_folder(tmp_path / "v", {"00000.ds": [[95] * MAX_LENGTH] * 3}, token_bytes=2)
    args = _args(
        token_sources=[{"name": "a", "path": str(_folder(tmp_path / "a"))}],
        eval_token_sources=[{"name": "v", "path": str(high)}],
    )
    mix, held_out = load_token_sources(args, _training(tmp_path), _parallelism(), resume_checkpoint=None)
    verify_token_sources_fit_model(mix, None, 90, "synthetic")
    with pytest.raises(
        ValueError, match=rf"{re.escape(str(high))}: window 0 holds token id 95, past the model's vocabulary of 90"
    ):
        verify_token_sources_fit_model(mix, held_out, 90, "synthetic")


def test_folders_naming_another_tokenizer_or_two_are_warned_naming_each_source(tmp_path, caplog):
    other = write_token_folder(
        tmp_path / "b", {"00000.ds": [[5] * 50, [6] * 50]}, token_bytes=2, tokenizer="other/tokenizer"
    )
    args = _args(
        token_sources=[{"name": "a", "path": str(_folder(tmp_path / "a"))}],
        eval_token_sources=[{"name": "v", "path": str(_folder(tmp_path / "v", tokens=40, seed=1))}],
    )
    mix, held_out = load_token_sources(args, _training(tmp_path), _parallelism(), resume_checkpoint=None)
    assert _warnings(caplog, mix, held_out, "synthetic") == []
    (renamed,) = _warnings(caplog, mix, held_out, "Qwen/Qwen3-0.6B")
    assert "names ['synthetic'], not only this run's tokenizer 'Qwen/Qwen3-0.6B'" in renamed
    assert "token_sources 'a' names ['synthetic']; eval_token_sources 'v' names ['synthetic']" in renamed
    two = _args(
        token_sources=[{"name": "a", "path": str(tmp_path / "a")}, {"name": "b", "path": str(other)}],
        eval_token_sources=[{"name": "v", "path": str(tmp_path / "v")}],
    )
    mix, held_out = load_token_sources(two, _training(tmp_path), _parallelism(), resume_checkpoint=None)
    (mixed,) = _warnings(caplog, mix, held_out, "synthetic")
    assert "names ['other/tokenizer', 'synthetic'], not only this run's tokenizer 'synthetic'" in mixed
    assert "token_sources 'b' names ['other/tokenizer']" in mixed


class _PolicyLoadReached(Exception):
    """Raised by the stubbed policy load: main() ran every data check before it."""


def _runtime(resume_checkpoint=None):
    return types.SimpleNamespace(
        parallelism_config=types.SimpleNamespace(cp_size=1, is_cp_mode=False, pp_size=1, data_parallel_size=1),
        model_source=_MODEL_ID,
        mode_suffix="",
        local_rank=0,
        resume_checkpoint=resume_checkpoint,
        policy_from_checkpoint=False,
    )


def _run_sft_main(tmp_path, yaml_body: str, *, resume_checkpoint=None, **stubs) -> list[dict]:
    """The policy loads ``sft.py:main()`` reaches from a YAML holding ``yaml_body``; ``stubs`` replace more of the
    script's names."""
    module = load_script_module("scripts/training/sft.py", "halo_test_sft_token_sources")
    config = tmp_path / "config.yaml"
    config.write_text(
        f"model_name_or_path: {_MODEL_ID}\noutput_dir: {tmp_path / 'out'}\nbf16: false\nuse_cpu: true\n"
        f"max_length: {MAX_LENGTH}\ntrain_on_completions_only: false\ntrain_sampling_strategy: sequential\n"
        f"packing: true\n{yaml_body}"
    )
    loads = []

    def load(_context, *_args, **kwargs):
        loads.append(kwargs)
        raise _PolicyLoadReached

    checkpoint_config = Qwen3Config(hidden_size=64, num_attention_heads=4, num_key_value_heads=2, vocab_size=128)
    with (
        mock.patch.object(module, "init_training_script", return_value=_runtime(resume_checkpoint)),
        mock.patch.object(module, "with_distributed_teardown", lambda fn: fn),
        mock.patch.object(module, "probe_checkpoint", return_value=(checkpoint_config, False)),
        mock.patch.object(module, "load_policy_model", side_effect=load),
        mock.patch.multiple(module, **stubs) if stubs else contextlib.nullcontext(),
        mock.patch("halo.entrypoints.script.install_log_tee"),
        mock.patch.object(sys, "argv", ["prog", str(config)]),
        contextlib.suppress(_PolicyLoadReached),
    ):
        module.main()
    return loads


def test_the_script_reaches_the_model_load_with_a_text_run_of_packed_rows(tmp_path):
    folder = _folder(tmp_path / "a")
    loads = _run_sft_main(tmp_path, f"max_steps: 4\ntoken_sources:\n- {{name: a, path: {folder}}}\n")
    assert [load["vlm_run"] for load in loads] == [False]


@pytest.mark.parametrize(
    ("yaml_body", "message"),
    [
        ("token_sources:\n- {name: a, path: /absent}\n", "max_steps set"),
        ("max_steps: 4\ntoken_sources:\n- {name: a, path: /absent}\n", "Token source folder /absent does not exist"),
        ("max_steps: 4\neval_strategy: steps\ntoken_sources:\n- {name: a, path: /absent}\n", _EVALUATION_REQUIREMENT),
        (
            "max_steps: 4\ncontext_parallel_size: 2\ntoken_sources:\n- {name: a, path: /absent}\n",
            "context_parallel_size: 1",
        ),
    ],
    ids=[
        "no max_steps",
        "a missing folder",
        "evaluation without held-out folders",
        "context parallelism before any folder is read",
    ],
)
def test_a_token_run_that_cannot_train_fails_before_the_model_load(tmp_path, yaml_body, message):
    with pytest.raises(ValueError, match=message):
        _run_sft_main(tmp_path, yaml_body)


def test_a_resume_onto_a_changed_mix_fails_before_the_model_load(tmp_path):
    folder = _folder(tmp_path / "a")
    checkpoint = _saved_checkpoint(tmp_path / "checkpoint-2", {"order_version": 1, "max_length": 16, "sources": []})
    with pytest.raises(ValueError, match="max_length 16 → 8"):
        _run_sft_main(
            tmp_path,
            f"max_steps: 4\ntoken_sources:\n- {{name: a, path: {folder}}}\n",
            resume_checkpoint=checkpoint,
        )


def _datasets_loaded(*_args, **_kwargs):
    raise AssertionError("the datasets loaded before the refusal")


def test_a_run_without_token_sources_refuses_a_token_runs_checkpoint_before_the_load(tmp_path, subtests):
    """Its resumed trainer state would carry the mix's fingerprint into every checkpoint it saves, which a later token
    run would resume from."""
    token_run = _saved_checkpoint(tmp_path / "checkpoint-2", {"order_version": 1, "sources": []})
    with (
        subtests.test("a token run's checkpoint"),
        pytest.raises(
            ValueError,
            match="was saved by a token_sources run.*start a new run from its weights.*merge_ep_shards.py first if it "
            "is EP-sharded",
        ),
    ):
        _run_sft_main(
            tmp_path,
            "max_steps: 4\ndataset: org/data\n",
            resume_checkpoint=token_run,
            load_script_datasets=_datasets_loaded,
        )
    with subtests.test("another run's checkpoint"):
        other_run = _saved_checkpoint(tmp_path / "checkpoint-1", None)
        loads = _run_sft_main(
            tmp_path,
            "max_steps: 4\ndataset: org/data\n",
            resume_checkpoint=other_run,
            load_script_datasets=lambda *_args, **_kwargs: ((None, False), False),
            resolve_vlm_run=lambda *_args, **_kwargs: False,
        )
        assert len(loads) == 1


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
