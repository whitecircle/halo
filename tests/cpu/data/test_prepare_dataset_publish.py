#!/usr/bin/env python3
"""``prepare_dataset.py`` publishing and defaults.

Three contracts pinned here:

* **Destructive publish.** An ``--overwrite`` that removes the destination before writing (local
  ``rmtree`` then write; the Hub upload replacing files in place) lets a run that dies mid-write
  destroy the only copy AND leave a partial tree whose ``metadata.json`` makes
  ``is_preprocessed_dataset`` accept it — training then runs on a silently truncated dataset.
  ``metadata.json`` is the completion marker, so the local path bakes into a sibling of the
  destination and swaps it in (one write, no copy), and the Hub path clears it first and uploads it
  last.
* **Defaults contradicting training.** ``--conversation_field`` must default to ``prompt`` — the
  training side and the documented SFT shape — not ``conversation``, and
  ``--train_on_completions_only`` must default on in chat mode as training does; otherwise preparing
  with defaults produces a dataset the training defaults reject.
* **A bake idling the host.** The tool runs alone, so unset its worker count takes every CPU the process may run
  on but a small reserve, not the training default's three-quarter share capped at 64.
* **Unrecorded tokenizer mutations.** ``--pad_token/--eos_token/--bos_token/--chat_template`` change
  the ids that get baked, so they must reach ``PreprocessingConfig`` or the compatibility check cannot
  see them.

Run: pytest tests/cpu/data/test_prepare_dataset_publish.py
"""

import os
import sys
import types

import pytest
from datasets import DatasetDict

from halo import hardware
from halo.checkpoint import atomic
from halo.data.pipeline import coordinated_map
from halo.data.preprocessed import publishing
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
    publishing.publish_local_output(_fill(publishing.stage_local_output(str(dest), overwrite)), str(dest))


# Local publish: bake beside the destination, then swap


def test_local_publish_replaces_an_existing_dataset_on_success(tmp_path):
    dest = tmp_path / "out"
    dest.mkdir()
    (dest / "stale.arrow").write_text("old")

    _publish(dest, overwrite=True)

    assert (dest / "shard_0000" / "data.arrow").read_text() == "new-payload"
    assert not (dest / "stale.arrow").exists(), "the swap must replace the tree, not merge into it"
    assert not list(tmp_path.glob(f"*{atomic.PUBLISH_STAGING_SUFFIX}")), "staging directory leaked"
    assert not list(tmp_path.glob(f"*{atomic.DISPLACED_SUFFIX}")), "replaced directory leaked"


def _stub_main(monkeypatch, tmp_path, output: str, preprocess) -> None:
    """``main()`` on a local ``output`` with everything but the bake's placement and the publish stubbed."""
    args = types.SimpleNamespace(
        input="in", output=output, overwrite=True, verbose=False, dry_run=False, private=False, hf_token=None
    )
    config = types.SimpleNamespace(
        pack_sequences=False, is_vlm=False, model_name_or_path="model", max_length=8, num_shards=1
    )
    monkeypatch.setenv("HF_DATASETS_CACHE", str(tmp_path / "datasets_cache"))
    monkeypatch.setattr(prepare_dataset, "parse_args", lambda: (args, config))
    monkeypatch.setattr(publishing, "PartialState", lambda: types.SimpleNamespace(num_processes=1))
    monkeypatch.setattr(publishing, "setup_tokenizer", lambda config: object())
    monkeypatch.setattr(publishing, "load_input_dataset", lambda args: DatasetDict())
    monkeypatch.setattr(publishing, "require_declared_columns", lambda *a: None)
    monkeypatch.setattr(publishing, "preprocess_dataset", preprocess)
    monkeypatch.setattr(publishing.shutil, "copytree", lambda *a, **k: pytest.fail("the baked output was copied"))


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
    """The load-bearing case: an --overwrite that rmtrees the destination FIRST leaves nothing at
    all when the bake dies. The old dataset must survive a failed run untouched."""
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
    """A SIGKILLed bake runs no cleanup, leaving every intermediate map's full corpus copy in the datasets cache:
    the scratch directory is named after the destination, so the next bake to it starts by removing them."""
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
    """The staged copy succeeded and the destination was rotated aside; if the final rename fails,
    the old dataset must go BACK to the destination rather than stay under a scratch name."""
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

    monkeypatch.setattr(publishing.os, "rename", _fail_second_rename)

    with pytest.raises(OSError, match="rename failed"):
        _publish(dest, overwrite=True)

    assert (dest / "data.arrow").read_text() == "old-payload", "the rotated-aside dataset was not restored"


def test_local_publish_refuses_an_existing_destination_without_overwrite(tmp_path):
    dest = tmp_path / "out"
    dest.mkdir()
    with pytest.raises(FileExistsError, match="--overwrite"):
        publishing.stage_local_output(str(dest), overwrite=False)


def test_local_publish_creates_a_fresh_destination(tmp_path):
    """Anti-over-rejection: the ordinary first-run publish still works."""
    dest = tmp_path / "brand-new"
    _publish(dest, overwrite=False)
    assert (dest / "metadata.json").exists()


# Hub publish: completion marker cleared first, uploaded last


def test_hub_publish_clears_then_uploads_the_completion_marker_last(tmp_path, monkeypatch):
    """While the payload uploads, the repo must not carry a metadata.json — otherwise a reader (or a
    run that dies mid-upload) sees a complete-looking dataset whose shards are a mix of the old and
    the partly-uploaded new one."""
    calls = []

    class _FakeApi:
        def __init__(self, token=None):
            pass

        def repo_exists(self, repo_id, repo_type=None):
            return True

        def create_repo(self, repo_id, repo_type=None, private=False, exist_ok=False):
            calls.append(("create_repo", repo_id))

        def delete_file(self, path_in_repo, repo_id=None, repo_type=None):
            calls.append(("delete_metadata", path_in_repo))

        def upload_file(self, path_or_fileobj=None, path_in_repo=None, repo_id=None, repo_type=None):
            calls.append(("upload_metadata", path_in_repo))

    def _fake_upload_folder(folder_path=None, repo_id=None, repo_type=None, token=None, ignore_patterns=None):
        calls.append(("upload_folder", tuple(ignore_patterns or ())))

    monkeypatch.setattr(publishing, "HfApi", _FakeApi)
    monkeypatch.setattr(publishing, "upload_folder", _fake_upload_folder)

    publishing.save_to_hf_hub(str(_staged(tmp_path)), "org/name", overwrite=True)

    ordered = [c[0] for c in calls]
    assert ordered.index("delete_metadata") < ordered.index("upload_folder") < ordered.index("upload_metadata"), calls
    folder_call = next(c for c in calls if c[0] == "upload_folder")
    assert publishing.METADATA_FILE in folder_call[1], "the payload upload must exclude the completion marker"


# CLI defaults line up with the training side


def _parse(argv_extra):
    """The PreprocessingConfig the command line builds."""
    argv = ["prepare_dataset.py", "--input", "in", "--output", "out", "--model_name_or_path", "model", *argv_extra]
    old = sys.argv
    sys.argv = argv
    try:
        return prepare_dataset.parse_args()[1]
    finally:
        sys.argv = old


def test_conversation_field_default_matches_the_training_side():
    """Preparing with defaults must render the field training reads, 'prompt' — the documented SFT
    shape. Rendering 'conversation' instead makes the default pairing KeyError or (post-check) raise."""
    assert _parse(["--assistant_message_template", "<a>"]).conversation_field == "prompt"


def test_train_on_completions_only_defaults_on_and_is_still_switchable():
    """Training defaults masking ON and refuses a dataset whose baked labels disagree, so preparing
    with masking defaulted off produces an artifact the training defaults reject."""
    assert _parse(["--assistant_message_template", "<a>"]).train_on_completions_only is True
    assert _parse(["--train_on_completions_only", "false"]).train_on_completions_only is False
    assert _parse(["--mode", "text"]).train_on_completions_only is False


def test_the_bake_takes_every_usable_cpu_but_a_reserve_unless_told_otherwise(monkeypatch, tmp_path):
    monkeypatch.delenv("HALO_DATASET_NUM_PROC", raising=False)
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: set(range(192)))
    monkeypatch.setattr(hardware, "_CGROUP_ROOT", str(tmp_path))  # no CPU quota
    assert _parse(["--mode", "text"]).num_proc == 192 - publishing._BAKE_RESERVED_CPUS
    assert _parse(["--mode", "text", "--num_proc", "3"]).num_proc == 3
    monkeypatch.setenv("HALO_DATASET_NUM_PROC", "7")
    assert _parse(["--mode", "text"]).num_proc == 7


def test_bos_token_override_exists():
    """bos must be overridable alongside pad/eos; without it one tokenizer mutation stays outside
    anything the recorded config can describe."""
    assert _parse(["--train_on_completions_only", "false", "--bos_token", "<s>"]).bos_token == "<s>"


def test_apply_tokenizer_overrides_sets_every_recorded_knob():
    tokenizer = types.SimpleNamespace(pad_token=None, eos_token="</s>", bos_token=None, chat_template=None)
    args = types.SimpleNamespace(pad_token="<pad>", eos_token="<eos>", bos_token="<s>", chat_template=None)
    publishing.apply_tokenizer_overrides(tokenizer, args)
    assert (tokenizer.pad_token, tokenizer.eos_token, tokenizer.bos_token) == ("<pad>", "<eos>", "<s>")


def test_tokenizer_overrides_reach_the_recorded_config(tmp_path):
    """The knobs mutate the tokenizer that bakes the ids; if they do not reach PreprocessingConfig
    the artifact is byte-indistinguishable from an unmutated one and the training-time
    compatibility check passes on a dataset tokenized by a different tokenizer. A template file is
    recorded as its resolved text, which is what the training side compares."""
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
    """A second, differently-behaving copy of the path-vs-text resolver would make the recorded
    template and the training-side one disagree on what counts as a file. Asserted on the bound OBJECT, not
    on an import line: a re-implementation that kept the import would satisfy the string."""
    assert prepare_dataset.load_chat_template is tokenizer_setup.load_chat_template


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
