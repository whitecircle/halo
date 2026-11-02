"""The checkpoint, EP and CP comparison harnesses compare only the families both trees' rosters hold, read through
the one roster helper (``revisions.common_families``): a family one commit adds, drops or renames never reaches the
``--families`` of a tree without it, whose parser refuses it as an invalid choice, and the evidence names it under
``skipped`` even when no request named it.

Each tree here is a stub of the harness module: its ``FAMILIES`` roster and a parser that refuses any other
family, run through the harness's real producer launches.

Run: python tests/cpu/conventions/test_harness_family_rosters.py
"""

import logging
from pathlib import Path

import pytest

from tests.common.equivalence import checkpoint_artifacts, cp_equivalence, ep_bitwise
from tests.common.sample_tree import write_sample_tree

HARNESSES = [
    pytest.param(module, id=module.__name__.rpartition(".")[2])
    for module in (ep_bitwise, checkpoint_artifacts, cp_equivalence)
]
PACKAGES = ("tests", "tests/common", "tests/common/equivalence")
STUB = """
    import argparse

    FAMILIES = {families!r}

    if __name__ == "__main__":
        parser = argparse.ArgumentParser()
        parser.add_argument("--families", nargs="+", choices=FAMILIES, required=True)
        parser.parse_known_args()
"""


def _stub_trees(monkeypatch, harness, rosters: dict[str, list[str]]) -> dict:
    """Point ``harness`` at one stub tree per commit label in ``rosters`` and return the evidence it writes."""
    module_path = f"{harness.__name__.replace('.', '/')}.py"

    def export_tree(repo, commit: str, destination: Path) -> Path:
        files = {f"{package}/__init__.py": "" for package in PACKAGES}
        write_sample_tree(destination, files | {module_path: STUB.format(families=tuple(rosters[commit]))})
        return destination

    evidence = {}

    def write_evidence(harness_name: str, **record) -> Path:
        evidence.update(record)
        return Path(f"{harness_name}.json")

    monkeypatch.setattr(harness, "resolve_commit", lambda repo, ref: ref)
    monkeypatch.setattr(harness, "export_tree", export_tree)
    monkeypatch.setattr(harness, "write_evidence", write_evidence)
    return evidence


@pytest.mark.parametrize("request_from", ["checkout", "head"])
@pytest.mark.parametrize("harness", HARNESSES)
def test_a_family_one_tree_lacks_never_reaches_its_families(harness, request_from, monkeypatch, caplog):
    """The base drops the roster's last family and the head its first. Requested from the checkout's wider
    roster, or from the head's own (the default when the checkout is the head, which never names the family the
    head dropped), both families are skipped and named."""
    roster = list(harness.FAMILIES)
    base_only, head_only = roster[0], roster[-1]
    rosters = {"base": roster[:-1], "head": roster[1:]}
    evidence = _stub_trees(monkeypatch, harness, rosters)
    requested = ["--families", *rosters["head"]] if request_from == "head" else []

    with caplog.at_level(logging.WARNING):
        assert harness.main(["--base-ref", "base", "--head-ref", "head", *requested]) == 0

    assert evidence["findings"]["families"] == roster[1:-1]
    assert sorted(evidence["findings"]["skipped"]) == sorted([base_only, head_only])
    assert f"the base tree's roster lacks {[head_only]}" in caplog.text
    assert f"the head tree's roster lacks {[base_only]}" in caplog.text


@pytest.mark.parametrize("harness", HARNESSES)
def test_rosters_sharing_no_family_fail_before_any_producer_runs(harness, monkeypatch):
    roster = list(harness.FAMILIES)
    _stub_trees(monkeypatch, harness, {"base": roster[:1], "head": roster[1:2]})

    with pytest.raises(ValueError, match="is in every tree's roster"):
        harness.main(["--base-ref", "base", "--head-ref", "head", "--families", *roster[:2]])


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
