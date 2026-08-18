#!/usr/bin/env python3
"""``prepare_dataset.py`` publishing and defaults.

Three contracts pinned here:

* **Destructive publish.** An ``--overwrite`` that removes the destination before writing (local
  ``rmtree`` then ``copytree``; the Hub upload replacing files in place) lets a run that dies
  mid-write destroy the only copy AND leave a partial tree whose ``metadata.json`` makes
  ``is_preprocessed_dataset`` accept it — training then runs on a silently truncated dataset.
  ``metadata.json`` is the completion marker, so the local path stages-then-swaps and the Hub path
  clears it first and uploads it last.
* **Defaults contradicting training.** ``--conversation-field`` must default to ``prompt`` — the
  training side and the documented SFT shape — not ``conversation``, and
  ``--train-on-completions-only`` must default on as training does; otherwise preparing with
  defaults produces a dataset the training defaults reject.
* **Unrecorded tokenizer mutations.** ``--pad/--eos/--bos/--chat-template`` change the ids that get
  baked, so they must reach ``PreprocessingConfig`` or the compatibility check cannot see them.

Run: pytest tests/cpu/data/test_prepare_dataset_publish.py
"""

import ast
import os
import sys
import types
from pathlib import Path

import pytest
from accelerate import PartialState

# The script logs through accelerate's rank-aware logger, which requires an initialized state.
PartialState()

from scripts.before_training import prepare_dataset as mod
from src.checkpoint import tool_io
from src.models.loading import tokenizer_setup

_SCRIPT = os.path.join(os.path.dirname(__file__), "..", "..", "..", "scripts", "before_training", "prepare_dataset.py")


def _staged(tmp_path, name="staged"):
    """A staged output directory shaped like the real one (payload + completion marker)."""
    staged = tmp_path / name
    staged.mkdir()
    (staged / "metadata.json").write_text('{"preprocessed": true}')
    (staged / "shard_0000").mkdir()
    (staged / "shard_0000" / "data.arrow").write_text("new-payload")
    return staged


# Local publish: stage, then swap


def test_local_publish_replaces_an_existing_dataset_on_success(tmp_path):
    dest = tmp_path / "out"
    dest.mkdir()
    (dest / "stale.arrow").write_text("old")

    mod.save_to_local(str(_staged(tmp_path)), str(dest), overwrite=True)

    assert (dest / "shard_0000" / "data.arrow").read_text() == "new-payload"
    assert not (dest / "stale.arrow").exists(), "the swap must replace the tree, not merge into it"
    assert not list(tmp_path.glob(f"*{tool_io.STAGING_SUFFIX}")), "staging directory leaked"
    assert not list(tmp_path.glob(f"*{tool_io.DISPLACED_SUFFIX}")), "replaced directory leaked"


def test_local_publish_keeps_the_old_dataset_when_the_copy_fails(tmp_path, monkeypatch):
    """The load-bearing case: an --overwrite that rmtrees the destination FIRST leaves nothing at
    all when the copy dies. The old dataset must survive a failed publish untouched."""
    dest = tmp_path / "out"
    dest.mkdir()
    (dest / "metadata.json").write_text('{"preprocessed": true}')
    (dest / "data.arrow").write_text("old-payload")

    def _boom(src, dst, *a, **kw):
        raise OSError("disk full")

    monkeypatch.setattr(mod.shutil, "copytree", _boom)

    with pytest.raises(OSError, match="disk full"):
        mod.save_to_local(str(_staged(tmp_path)), str(dest), overwrite=True)

    assert (dest / "data.arrow").read_text() == "old-payload", "a failed publish destroyed the previous dataset"
    assert not list(tmp_path.glob(f"*{tool_io.STAGING_SUFFIX}")), "staging directory leaked after the failure"


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

    monkeypatch.setattr(mod.os, "rename", _fail_second_rename)

    with pytest.raises(OSError, match="rename failed"):
        mod.save_to_local(str(_staged(tmp_path)), str(dest), overwrite=True)

    assert (dest / "data.arrow").read_text() == "old-payload", "the rotated-aside dataset was not restored"


def test_local_publish_refuses_an_existing_destination_without_overwrite(tmp_path):
    dest = tmp_path / "out"
    dest.mkdir()
    with pytest.raises(FileExistsError, match="--overwrite"):
        mod.save_to_local(str(_staged(tmp_path)), str(dest), overwrite=False)


def test_local_publish_creates_a_fresh_destination(tmp_path):
    """Anti-over-rejection: the ordinary first-run publish still works."""
    dest = tmp_path / "brand-new"
    mod.save_to_local(str(_staged(tmp_path)), str(dest), overwrite=False)
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

    monkeypatch.setattr(mod, "HfApi", _FakeApi)
    monkeypatch.setattr(mod, "upload_folder", _fake_upload_folder)

    mod.save_to_hf_hub(str(_staged(tmp_path)), "org/name", overwrite=True)

    ordered = [c[0] for c in calls]
    assert ordered.index("delete_metadata") < ordered.index("upload_folder") < ordered.index("upload_metadata"), calls
    folder_call = next(c for c in calls if c[0] == "upload_folder")
    assert mod.METADATA_FILE in folder_call[1], "the payload upload must exclude the completion marker"


# CLI defaults line up with the training side


def _parse(argv_extra):
    argv = ["prepare_dataset.py", "-i", "in", "-o", "out", "-m", "model", *argv_extra]
    old = sys.argv
    sys.argv = argv
    try:
        return mod.parse_args()
    finally:
        sys.argv = old


def test_conversation_field_default_matches_the_training_side():
    """Preparing with defaults must render the field training reads, 'prompt' — the documented SFT
    shape. Rendering 'conversation' instead makes the default pairing KeyError or (post-check) raise."""
    assert _parse([]).conversation_field == "prompt"


def test_train_on_completions_only_defaults_on_and_is_still_switchable():
    """Training defaults masking ON and refuses a dataset whose baked labels disagree, so preparing
    with masking defaulted off produces an artifact the training defaults reject."""
    assert _parse([]).train_on_completions_only is True
    assert _parse(["--no-train-on-completions-only"]).train_on_completions_only is False


def test_bos_token_override_exists():
    """bos must be overridable alongside pad/eos; without it one tokenizer mutation stays outside
    anything the recorded config can describe."""
    assert _parse(["--bos-token", "<s>"]).bos_token == "<s>"


def test_apply_tokenizer_overrides_sets_every_recorded_knob():
    tokenizer = types.SimpleNamespace(pad_token=None, eos_token="</s>", bos_token=None, chat_template=None)
    args = types.SimpleNamespace(pad_token="<pad>", eos_token="<eos>", bos_token="<s>", chat_template=None)
    mod.apply_tokenizer_overrides(tokenizer, args)
    assert (tokenizer.pad_token, tokenizer.eos_token, tokenizer.bos_token) == ("<pad>", "<eos>", "<s>")


def test_tokenizer_overrides_reach_the_recorded_config():
    """The knobs mutate the tokenizer that bakes the ids; if they do not reach PreprocessingConfig
    the artifact is byte-indistinguishable from an unmutated one and the training-time
    compatibility check passes on a dataset tokenized by a different tokenizer."""
    tree = ast.parse(Path(_SCRIPT).read_text(encoding="utf-8"))
    config_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "PreprocessingConfig"
    ]
    assert len(config_calls) == 1, f"expected one PreprocessingConfig construction, found {len(config_calls)}"
    passed = {kw.arg for kw in config_calls[0].keywords}
    for knob in ("pad_token", "eos_token", "bos_token", "chat_template"):
        assert knob in passed, f"{knob} mutates the tokenizer but is not recorded in the metadata config"


def test_script_uses_the_shared_chat_template_resolver():
    """A second, differently-behaving copy of the path-vs-text resolver made the recorded template
    and the training-side one disagree on what counts as a file. Asserted on the bound OBJECT, not
    on an import line: a re-implementation that kept the import would satisfy the string."""
    assert mod.load_chat_template is tokenizer_setup.load_chat_template


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
