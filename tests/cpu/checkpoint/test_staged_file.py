"""``staged_file`` publishes only a block that finished, and never leaves its stage behind.

A raise mid-write (a kill, a failed verification) must keep the previous file under the final name:
under ``reset_sinks --in_place`` that file is the only copy of the weights.

    python tests/cpu/checkpoint/test_staged_file.py
"""

import pytest

from src.checkpoint.atomic import is_atomic_staging_file, staged_file


@pytest.mark.parametrize("durable", [True, False])
def test_a_finished_block_replaces_the_file(tmp_path, durable):
    (tmp_path / "card.md").write_text("old")

    with staged_file(tmp_path, "card.md", durable=durable) as staged:
        staged.write_text("new")

    assert (tmp_path / "card.md").read_text() == "new"
    assert [path.name for path in tmp_path.iterdir()] == ["card.md"]


@pytest.mark.parametrize("durable", [True, False])
def test_a_failed_block_keeps_the_previous_file_and_drops_its_stage(tmp_path, durable):
    (tmp_path / "card.md").write_text("old")

    with pytest.raises(RuntimeError, match="verification failed"):
        with staged_file(tmp_path, "card.md", durable=durable) as staged:
            staged.write_text("half")
            assert is_atomic_staging_file(staged.name)
            raise RuntimeError("verification failed")

    assert (tmp_path / "card.md").read_text() == "old"
    assert [path.name for path in tmp_path.iterdir()] == ["card.md"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
