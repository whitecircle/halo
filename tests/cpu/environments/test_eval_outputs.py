"""``write_eval_outputs`` runs once every episode has been paid for, so the ``--output`` dump must
not be the step that loses the run: a missing parent directory is created, and the values the
trajectory JSONL writer serializes are accepted here too.

    python tests/cpu/environments/test_eval_outputs.py
"""

import argparse
import json

import pytest

from scripts.environments._common import write_eval_outputs


def _write(output: str, results: list[dict]) -> None:
    args = argparse.Namespace(output=output, model="m", dataset="d", config=None, split="test")
    write_eval_outputs(
        args,
        results,
        env=None,
        traj_path=None,
        env_type="qa_search",
        max_turns=None,
        max_tokens=16,
        temperature=0.0,
        num_samples=1,
    )


def test_the_output_dump_creates_its_parent_directory(tmp_path):
    output = tmp_path / "runs" / "eval" / "results.json"
    results = [{"group": None, "id": 1, "samples": [{"reward": 1.0, "success": True, "stats": {}}]}]

    _write(str(output), results)

    assert json.loads(output.read_text()) == results


def test_the_output_dump_accepts_what_the_trajectory_writer_accepts(tmp_path):
    """A value only ``str`` can serialize (an env-supplied set in the episode info) reaches both
    writers from the same record; the JSONL writer takes it, so the dump must not raise on it."""
    output = tmp_path / "results.json"
    info = {"seen": {"a"}}
    results = [{"group": None, "id": 1, "samples": [{"reward": 0.0, "success": False, "trajectory": {"info": info}}]}]

    _write(str(output), results)

    assert json.loads(output.read_text())[0]["samples"][0]["trajectory"]["info"]["seen"] == str(info["seen"])


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
