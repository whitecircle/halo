#!/usr/bin/env python3
"""``prepare_dataset.py`` publishing and defaults. ``metadata.json`` is the completion marker, so the local path
bakes into a sibling and swaps it in, and the Hub path clears it first and uploads it last. A remote output is checked
before the input loads and outlives a failed upload; one output takes one bake at a time. CLI defaults match
training's, and tokenizer overrides reach ``PreprocessingConfig``."""

import os
import re
import sys
import types

import pytest
from datasets import Dataset, DatasetDict
from filelock import FileLock
from huggingface_hub.utils import filter_repo_objects

from halo import hardware
from halo.checkpoint import atomic
from halo.data.pipeline import coordinated_map
from halo.data.preprocessed import prepare_dataset_tool
from halo.model_loading import tokenizer_setup
from scripts.before_training import prepare_dataset


def _fill(directory) -> str:
    """Write an output tree shaped like the real one (payload + completion marker) into ``directory``."""
    os.makedirs(directory)
    with open(os.path.join(directory, "metadata.json"), "w") as f:
        f.write('{"preprocessed": true}')
    os.makedirs(os.path.join(directory, "shard_0000"))
    with open(os.path.join(directory, "shard_0000", "data.arrow"), "w") as f:
        f.write("new-payload")
    return str(directory)


def _staged(tmp_path, name="staged"):
    return _fill(tmp_path / name)


def _publish(dest, overwrite: bool) -> None:
    prepare_dataset_tool.publish_local_output(
        _fill(prepare_dataset_tool.stage_local_output(str(dest), overwrite)), str(dest)
    )


def test_local_publish_replaces_an_existing_dataset_on_success(tmp_path):
    dest = tmp_path / "out"
    dest.mkdir()
    (dest / "stale.arrow").write_text("old")

    _publish(dest, overwrite=True)

    assert (dest / "shard_0000" / "data.arrow").read_text() == "new-payload"
    assert not (dest / "stale.arrow").exists(), "the swap must replace the tree, not merge into it"
    assert not list(tmp_path.glob(f"*{atomic.PUBLISH_STAGING_SUFFIX}")), "staging directory leaked"
    assert not list(tmp_path.glob(f"*{atomic.DISPLACED_SUFFIX}")), "replaced directory leaked"


def _stub_main(monkeypatch, tmp_path, output: str, preprocess, overwrite: bool = True) -> None:
    """``main()`` on ``output`` with everything but the bake's placement and the publish stubbed."""
    args = types.SimpleNamespace(
        input="in",
        output=output,
        overwrite=overwrite,
        verbose=False,
        dry_run=False,
        private=False,
        allow_cross_document_linear_attention=False,
    )
    config = types.SimpleNamespace(
        pack_sequences=False, is_vlm=False, model_name_or_path="model", max_length=8, num_shards=1
    )
    monkeypatch.setenv("HF_DATASETS_CACHE", str(tmp_path / "datasets_cache"))
    monkeypatch.setattr(prepare_dataset_tool, "PartialState", lambda: types.SimpleNamespace(num_processes=1))
    monkeypatch.setattr(prepare_dataset, "parse_args", lambda: (args, config))
    monkeypatch.setattr(prepare_dataset_tool, "load_tokenizer", lambda config: object())
    monkeypatch.setattr(prepare_dataset_tool, "load_input_dataset", lambda args: DatasetDict())
    monkeypatch.setattr(prepare_dataset_tool, "require_declared_columns", lambda *a: None)
    monkeypatch.setattr(prepare_dataset_tool, "preprocess_dataset", preprocess)
    monkeypatch.setattr(
        prepare_dataset_tool.shutil, "copytree", lambda *a, **k: pytest.fail("the baked output was copied")
    )


def test_a_local_output_is_baked_beside_its_destination_and_written_once(tmp_path, monkeypatch):
    dest = tmp_path / "out"
    dest.mkdir()
    (dest / "stale.arrow").write_text("old")
    baked_into = []

    def _preprocess(*, dataset, tokenizer_or_processor, config, output_dir):
        baked_into.append(output_dir)
        _fill(output_dir)
        return {
            "metadata": types.SimpleNamespace(
                total_train_examples=1, total_test_examples=0, num_shards=1, packed=False
            )
        }

    _stub_main(monkeypatch, tmp_path, str(dest), _preprocess)
    prepare_dataset.main()

    assert baked_into == [f"{dest}{atomic.PUBLISH_STAGING_SUFFIX}"], (
        "the bake must land on the destination's filesystem"
    )
    assert (dest / "shard_0000" / "data.arrow").read_text() == "new-payload"
    assert not (dest / "stale.arrow").exists()
    assert not list(tmp_path.glob(f"*{atomic.PUBLISH_STAGING_SUFFIX}")), "staging directory leaked"


def test_a_failed_bake_keeps_the_old_dataset_and_leaves_no_staging(tmp_path, monkeypatch):
    """An --overwrite that rmtrees first leaves nothing when the bake dies; the old dataset must survive."""
    dest = tmp_path / "out"
    dest.mkdir()
    (dest / "metadata.json").write_text('{"preprocessed": true}')
    (dest / "data.arrow").write_text("old-payload")

    def _preprocess(*, output_dir, **kwargs):
        _fill(output_dir)
        raise OSError("disk full")

    _stub_main(monkeypatch, tmp_path, str(dest), _preprocess)
    with pytest.raises(OSError, match="disk full"):
        prepare_dataset.main()

    assert (dest / "data.arrow").read_text() == "old-payload", "a failed run destroyed the previous dataset"
    assert not list(tmp_path.glob(f"*{atomic.PUBLISH_STAGING_SUFFIX}")), "staging directory leaked after the failure"


def test_a_rerun_reaps_the_map_copies_a_killed_bake_left(tmp_path, monkeypatch):
    """A SIGKILLed bake leaves every intermediate map's corpus copy in the datasets cache, and the scratch directory
    is named after the destination, so the next bake removes them."""
    dest = tmp_path / "out"
    seen = []

    def _preprocess(*, output_dir, **kwargs):
        scratch = coordinated_map.ensure_cache_dir()
        seen.append((scratch, sorted(os.listdir(scratch))))
        _fill(output_dir)
        return {
            "metadata": types.SimpleNamespace(
                total_train_examples=1, total_test_examples=0, num_shards=1, packed=False
            )
        }

    _stub_main(monkeypatch, tmp_path, str(dest), _preprocess)
    prepare_dataset.main()
    ((scratch, _),) = seen
    assert scratch.startswith(str(tmp_path / "datasets_cache")) and not os.path.exists(scratch)

    os.makedirs(scratch)
    with open(os.path.join(scratch, "cache-killed.arrow"), "w") as f:
        f.write("a killed bake's map copy")
    prepare_dataset.main()
    assert seen[1] == (scratch, []), "the rerun baked beside the killed bake's map copies"
    assert not os.path.exists(scratch)


def test_local_publish_restores_the_old_dataset_if_the_swap_fails(tmp_path, monkeypatch):
    """The destination was rotated aside; if the final rename fails, the old dataset goes back and does not stay
    under a scratch name."""
    dest = tmp_path / "out"
    dest.mkdir()
    (dest / "data.arrow").write_text("old-payload")

    real_rename = os.rename
    calls = {"n": 0}

    def _fail_second_rename(src, dst):
        calls["n"] += 1
        if calls["n"] == 2:
            raise OSError("rename failed")
        real_rename(src, dst)

    monkeypatch.setattr(prepare_dataset_tool.os, "rename", _fail_second_rename)

    with pytest.raises(OSError, match="rename failed"):
        _publish(dest, overwrite=True)

    assert (dest / "data.arrow").read_text() == "old-payload", "the rotated-aside dataset was not restored"


def test_local_publish_refuses_an_existing_destination_without_overwrite(tmp_path):
    dest = tmp_path / "out"
    dest.mkdir()
    with pytest.raises(FileExistsError, match="--overwrite"):
        prepare_dataset_tool.stage_local_output(str(dest), overwrite=False)


def test_local_publish_creates_a_fresh_destination(tmp_path):
    dest = tmp_path / "brand-new"
    _publish(dest, overwrite=False)
    assert (dest / "metadata.json").exists()


def test_hub_publish_clears_then_uploads_the_completion_marker_last(tmp_path, monkeypatch):
    """A repo carrying metadata.json during the upload looks complete to a reader, with shards mixing old and new."""
    calls = []

    class _FakeApi:
        def repo_exists(self, repo_id, repo_type=None):
            return True

        def create_repo(self, repo_id, repo_type=None, private=False, exist_ok=False):
            calls.append(("create_repo", repo_id))

        def delete_file(self, path_in_repo, repo_id=None, repo_type=None):
            calls.append(("delete_metadata", path_in_repo))

        def upload_file(self, path_or_fileobj=None, path_in_repo=None, repo_id=None, repo_type=None):
            calls.append(("upload_metadata", path_in_repo))

    def _fake_upload_folder(
        folder_path=None, repo_id=None, repo_type=None, ignore_patterns=None, delete_patterns=None
    ):
        calls.append(("upload_folder", tuple(ignore_patterns or ())))

    monkeypatch.setattr(prepare_dataset_tool, "HfApi", _FakeApi)
    monkeypatch.setattr(prepare_dataset_tool, "upload_folder", _fake_upload_folder)

    prepare_dataset_tool.save_to_hf_hub(str(_staged(tmp_path)), "org/name")

    ordered = [c[0] for c in calls]
    assert ordered.index("delete_metadata") < ordered.index("upload_folder") < ordered.index("upload_metadata"), calls
    folder_call = next(c for c in calls if c[0] == "upload_folder")
    assert prepare_dataset_tool.METADATA_FILE in folder_call[1], (
        "the payload upload must exclude the completion marker"
    )


def test_a_hub_overwrite_removes_the_old_bakes_files_and_keeps_the_card(tmp_path, monkeypatch):
    """upload_folder adds and replaces but deletes nothing: a sharded bake overwritten by an unsharded one would keep
    its shard index, which the loader reads first."""
    deleted = []

    class _FakeApi:
        def create_repo(self, *args, **kwargs):
            pass

        def delete_file(self, *args, **kwargs):
            pass

        def upload_file(self, *args, **kwargs):
            pass

    def _fake_upload_folder(delete_patterns=None, **kwargs):
        old_tree = ["README.md", ".gitattributes", "dataset_dict.json", "train/shard_index.json"]
        old_tree += ["train/shard_0000/data-00000-of-00001.arrow", "test/shard_0000/state.json"]
        deleted.extend(filter_repo_objects(old_tree, allow_patterns=delete_patterns) if delete_patterns else [])

    monkeypatch.setattr(prepare_dataset_tool, "HfApi", _FakeApi)
    monkeypatch.setattr(prepare_dataset_tool, "upload_folder", _fake_upload_folder)
    prepare_dataset_tool.save_to_hf_hub(str(_staged(tmp_path)), "org/name")

    assert sorted(deleted) == [
        "dataset_dict.json",
        "test/shard_0000/state.json",
        "train/shard_0000/data-00000-of-00001.arrow",
        "train/shard_index.json",
    ]


def _fake_s3(existing: bool = False, push_error: Exception | None = None):
    """An S3 client class whose ``exists`` answers ``existing`` and whose push raises ``push_error`` if set, and the
    list of the folders it pushed."""
    pushed = []

    class _Client:
        def __init__(self, bucket):
            pass

        def exists(self, key):
            return existing

        def push_folder(self, local_path, key, **kwargs):
            if push_error is not None:
                raise push_error
            pushed.append(local_path)

    return _Client, pushed


def _bake_into(output_dir, **kwargs):
    _fill(output_dir)
    return {
        "metadata": types.SimpleNamespace(total_train_examples=1, total_test_examples=0, num_shards=1, packed=False)
    }


def test_an_existing_remote_output_is_refused_before_the_input_loads(tmp_path, monkeypatch):
    """Checked after the bake, the refusal would cost the whole bake and remove it."""
    _stub_main(monkeypatch, tmp_path, "s3://bucket/ds", _bake_into, overwrite=False)
    monkeypatch.setattr(prepare_dataset_tool, "S3Client", _fake_s3(existing=True)[0])
    monkeypatch.setattr(prepare_dataset_tool, "load_input_dataset", lambda args: pytest.fail("the input loaded"))
    with pytest.raises(FileExistsError, match="s3://bucket/ds.*--overwrite"):
        prepare_dataset.main()


def test_a_failed_upload_keeps_the_bake_and_names_it(tmp_path, monkeypatch):
    """Only the upload is left to redo, so the bake must outlive it, on the scratch's volume."""
    _stub_main(monkeypatch, tmp_path, "s3://bucket/ds", _bake_into)
    monkeypatch.setattr(prepare_dataset_tool, "S3Client", _fake_s3(push_error=ConnectionError("unreachable"))[0])
    with pytest.raises(RuntimeError, match=r"kept at \S+ until the next run.*s3\.py push") as excinfo:
        prepare_dataset.main()

    staged = re.search(r"kept at (\S+) until", str(excinfo.value)).group(1)
    assert staged.startswith(str(tmp_path / "datasets_cache"))
    with open(os.path.join(staged, "shard_0000", "data.arrow")) as f:
        assert f.read() == "new-payload"


def test_a_published_remote_bake_leaves_no_staging(tmp_path, monkeypatch):
    _stub_main(monkeypatch, tmp_path, "s3://bucket/ds", _bake_into)
    client, pushed = _fake_s3()
    monkeypatch.setattr(prepare_dataset_tool, "S3Client", client)
    prepare_dataset.main()

    (staged,) = pushed
    assert not os.path.exists(staged)


def test_a_second_bake_of_one_output_exits_while_the_first_holds_it(tmp_path, monkeypatch):
    """Both bakes name their scratch and staging after the output, so a second would remove the first's and both
    would die."""
    dest = tmp_path / "out"
    _stub_main(monkeypatch, tmp_path, str(dest), lambda **kwargs: pytest.fail("the second bake ran"))
    scratch = prepare_dataset_tool.bake_scratch_path(str(dest))
    os.makedirs(scratch)
    with open(os.path.join(scratch, "cache-first-bake.arrow"), "w"):
        pass

    with FileLock(f"{scratch}.lock"), pytest.raises(SystemExit, match="Another prepare_dataset.py is baking"):
        prepare_dataset.main()
    assert os.listdir(scratch) == ["cache-first-bake.arrow"], "the second bake removed the first's scratch"
    assert not dest.exists()


def test_a_trailing_slash_spelling_of_an_s3_output_takes_the_same_lock(tmp_path, monkeypatch):
    """Both spellings name one prefix, so two bakes under two locks would push their shards over each other."""
    _stub_main(monkeypatch, tmp_path, "s3://bucket/ds/", lambda **kwargs: pytest.fail("the second bake ran"))
    monkeypatch.setattr(prepare_dataset_tool, "S3Client", _fake_s3()[0])
    scratch = prepare_dataset_tool.bake_scratch_path("s3://bucket/ds")

    with FileLock(f"{scratch}.lock"), pytest.raises(SystemExit, match="Another prepare_dataset.py is baking"):
        prepare_dataset.main()


def _parse(argv_extra):
    argv = ["prepare_dataset.py", "--input", "in", "--output", "out", "--model_name_or_path", "model", *argv_extra]
    old = sys.argv
    sys.argv = argv
    try:
        return prepare_dataset.parse_args()[1]
    finally:
        sys.argv = old


def test_test_size_writes_nothing_into_the_input_directory(tmp_path, monkeypatch):
    """``datasets`` writes a split's index files beside the source's Arrow files, which fails on a read-only input."""
    monkeypatch.setenv("HF_DATASETS_CACHE", str(tmp_path / "datasets_cache"))
    source = tmp_path / "input"
    Dataset.from_dict({"text": [f"doc {i}" for i in range(20)]}).save_to_disk(source)
    before = sorted(os.listdir(source))
    args = types.SimpleNamespace(input=str(source), test_size=0.25)

    split = prepare_dataset_tool.load_input_dataset(args)

    assert (len(split["train"]), len(split["test"])) == (15, 5)
    assert sorted(os.listdir(source)) == before


def test_a_token_is_refused_on_the_command_line(capsys):
    """argv is world-readable through ``ps``; the Hub client reads ``HF_TOKEN`` or the stored login itself."""
    with pytest.raises(SystemExit):
        _parse(["--hf_token", "hf_secret"])
    assert "--hf_token" in capsys.readouterr().err


def test_conversation_field_default_matches_the_training_side():
    """Rendering ``conversation`` instead makes the default pairing raise."""
    assert _parse(["--assistant_message_template", "<a>"]).conversation_field == "prompt"


def test_train_on_completions_only_defaults_on_and_is_still_switchable():
    """Training refuses a dataset whose baked labels disagree, so masking defaulted off yields an artifact the
    training defaults reject."""
    assert _parse(["--assistant_message_template", "<a>"]).train_on_completions_only is True
    assert _parse(["--train_on_completions_only", "false"]).train_on_completions_only is False
    assert _parse(["--mode", "text"]).train_on_completions_only is False


def test_the_bake_takes_every_usable_cpu_but_a_reserve_unless_told_otherwise(monkeypatch, tmp_path):
    monkeypatch.delenv("HALO_DATASET_NUM_PROC", raising=False)
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: set(range(192)))
    monkeypatch.setattr(hardware, "_CGROUP_ROOT", str(tmp_path))  # no CPU quota
    assert _parse(["--mode", "text"]).num_proc == 192 - prepare_dataset_tool._BAKE_RESERVED_CPUS
    assert _parse(["--mode", "text", "--num_proc", "3"]).num_proc == 3
    monkeypatch.setenv("HALO_DATASET_NUM_PROC", "7")
    assert _parse(["--mode", "text"]).num_proc == 7


def test_apply_tokenizer_overrides_sets_every_recorded_knob():
    tokenizer = types.SimpleNamespace(pad_token=None, eos_token="</s>", bos_token=None, chat_template=None)
    args = types.SimpleNamespace(pad_token="<pad>", eos_token="<eos>", bos_token="<s>", chat_template=None)
    prepare_dataset_tool.apply_tokenizer_overrides(tokenizer, args)
    assert (tokenizer.pad_token, tokenizer.eos_token, tokenizer.bos_token) == ("<pad>", "<eos>", "<s>")


def test_tokenizer_overrides_reach_the_recorded_config(tmp_path):
    """Otherwise the artifact is indistinguishable from an unmutated one and the compatibility check passes on
    another tokenizer's ids. A template file is recorded as its resolved text."""
    template = tmp_path / "chat.jinja"
    template.write_text("{{ messages }}")
    overrides = [
        "--pad_token",
        "<pad>",
        "--eos_token",
        "<eos>",
        "--bos_token",
        "<s>",
        "--chat_template",
        str(template),
    ]
    config = _parse(["--train_on_completions_only", "false", *overrides])
    assert (config.pad_token, config.eos_token, config.bos_token) == ("<pad>", "<eos>", "<s>")
    assert config.chat_template == "{{ messages }}"


def test_script_uses_the_shared_chat_template_resolver():
    """Asserted on the bound object: a re-implementation keeping the import would satisfy a string check."""
    assert prepare_dataset.load_chat_template is tokenizer_setup.load_chat_template


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
