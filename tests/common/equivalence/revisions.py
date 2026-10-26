"""Two commits of the repository side by side, for the harnesses that compare a change with its base: a
commit's tree exported and run, the renames git finds, how a producer records a case that raised, the
entry-by-entry comparison of two outputs with its accepted deltas and negative control, and the evidence
record a harness writes per head commit. Standard library only, so the move kit runs wherever git does.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import logging
import os
import shlex
import subprocess
import sys
import tarfile
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path

from tests.common.ports import free_port
from tests.common.repo import REPO_ROOT

# Git-ignored; each run writes <harness>-<head commit>.json here.
EVIDENCE_DIR = REPO_ROOT / "evidence"
# The similarity git pairs a deleted file with an added one at; a declared rename must clear the
# caller's own, higher bar.
RENAME_DETECTION_PERCENT = 50
DIGEST_HEX_CHARS = 16
# How many differing entries one output entry's delta lists before it stops.
DELTA_ENTRIES = 20
# The key a producer records a raising case's exception under, in place of the case's output.
RAISED = "raised"
# Spelled into a raised message in place of the output root, which differs between the two runs.
OUTPUT_ROOT_MARK = "<output>"


def _git(repo: Path, *args: str) -> bytes:
    result = subprocess.run(["git", *args], cwd=repo, capture_output=True, check=False)
    if result.returncode != 0:
        stderr = result.stderr.decode(errors="replace").strip()
        raise RuntimeError(f"git {shlex.join(args)} failed in {repo}: {stderr}")
    return result.stdout


def resolve_commit(repo: Path, ref: str) -> str:
    """The full SHA of the commit ``ref`` names in ``repo``."""
    return _git(repo, "rev-parse", "--verify", f"{ref}^{{commit}}").decode().strip()


def export_tree(repo: Path, commit: str, destination: Path) -> Path:
    """Write the files ``commit`` tracks under ``destination`` and return it; untracked and ignored files of
    the checkout never reach it."""
    archive = _git(repo, "archive", "--format=tar", commit)
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        tar.extractall(destination, filter="data")
    return destination


@dataclass(frozen=True)
class Rename:
    """A file git pairs across two commits, and how similar the two versions are, in percent."""

    old: str
    new: str
    similarity: int


def detect_renames(repo: Path, base: str, head: str) -> list[Rename]:
    """Every rename ``git diff -M50%`` finds from ``base`` to ``head``, with no cap on the files it compares."""
    fields = _git(repo, "diff", f"-M{RENAME_DETECTION_PERCENT}%", "-l0", "--name-status", "-z", base, head).decode()
    entries = fields.split("\0")
    renames, index = [], 0
    while index < len(entries) - 1:
        status = entries[index]
        if status[0] in "RC":
            if status[0] == "R":
                renames.append(Rename(entries[index + 1], entries[index + 2], int(status[1:])))
            index += 3
        else:
            index += 2
    return renames


def _run_module(
    tree: Path, module: str, args: tuple[str, ...], env: Mapping[str, str] | None, nproc: int | None = None, **kwargs
):
    environment = {**os.environ, **(env or {}), "PYTHONPATH": str(tree)}
    launcher = [sys.executable]
    if nproc is not None:
        launcher += ["-m", "torch.distributed.run", f"--nproc_per_node={nproc}"]
        launcher += ["--master_addr=127.0.0.1", f"--master_port={free_port()}"]
    return subprocess.run([*launcher, "-m", module, *args], cwd=tree, env=environment, **kwargs)


def run_in_tree(
    tree: Path, module: str, *args: str, env: Mapping[str, str] | None = None, nproc: int | None = None
) -> None:
    """Run ``python -m module args`` as the repository at ``tree``: its working directory and its only
    first-party import path, so the module and everything it imports are that tree's own. ``nproc`` runs it
    under ``torchrun`` as that many ranks on this host instead. ``env`` adds to the environment; a non-zero
    exit raises."""
    _run_module(tree, module, args, env, nproc, check=True)


def probe_in_tree(tree: Path, module: str, *args: str) -> subprocess.CompletedProcess[str]:
    """:func:`run_in_tree` for a module whose exit status is the answer: how it exited, with its output."""
    return _run_module(tree, module, args, None, check=False, capture_output=True, text=True)


def digest(data: bytes) -> str:
    """The short content digest every harness compares by."""
    return hashlib.sha256(data).hexdigest()[:DIGEST_HEX_CHARS]


def raised_description(error: Exception, output_root: Path | str) -> dict[str, str]:
    """What a case raised, as a producer records it: the exception's type and its message with
    ``output_root`` spelled :data:`OUTPUT_ROOT_MARK`."""
    return {"type": type(error).__qualname__, "message": str(error).replace(str(output_root), OUTPUT_ROOT_MARK)}


@contextlib.contextmanager
def quiet_logs() -> Iterator[None]:
    """Every log record below ERROR dropped while inside, so a producer's output carries only its failures."""
    logging.disable(logging.WARNING)
    try:
        yield
    finally:
        logging.disable(logging.NOTSET)


def json_differences(base, head, prefix: str = "") -> list[str]:
    """Every leaf by which two JSON-able values differ, one line each naming its key path; ``[]`` when equal."""
    if base == head:
        return []
    if isinstance(base, Mapping) and isinstance(head, Mapping):
        found = []
        for key in sorted(base.keys() | head.keys(), key=str):
            where = f"{prefix}/{key}" if prefix else str(key)
            if key not in head:
                found.append(f"{where}: only in base")
            elif key not in base:
                found.append(f"{where}: only in head")
            else:
                found += json_differences(base[key], head[key], where)
        return found
    return [f"{prefix}: {json.dumps(base)[:200]} != {json.dumps(head)[:200]}"]


def compare_outputs[T](
    base: Mapping[str, T], head: Mapping[str, T], differences: Callable[[T, T], list[str]] = json_differences
) -> list[dict]:
    """Two outputs entry by entry, each keyed by its path in the output: every path one holds and the other
    lacks, and every path whose ``differences`` are not empty, with the first :data:`DELTA_ENTRIES` of them."""
    deltas = []
    for path in sorted(base.keys() | head.keys()):
        if path not in head:
            deltas.append({"path": path, "delta": "only in base"})
        elif path not in base:
            deltas.append({"path": path, "delta": "only in head"})
        elif entries := differences(base[path], head[path]):
            deltas.append({"path": path, "delta": "differs", "entries": entries[:DELTA_ENTRIES]})
    return deltas


def split_accepted(deltas: list[dict], accepted: Iterable[str]) -> tuple[list[dict], list[dict]]:
    """``deltas`` as (failing, accepted): a path ``accepted`` names covers itself and every path under it,
    and its deltas, a flagged change's declared effect, do not fail the verdict."""
    prefixes = tuple(accepted)
    failing, declared = [], []
    for delta in deltas:
        path = delta["path"]
        covered = any(path == prefix or path.startswith(f"{prefix}/") for prefix in prefixes)
        (declared if covered else failing).append(delta)
    return failing, declared


def unseen_controls(control_deltas: list[dict], control_paths: Iterable[str], raised: Iterable[str]) -> list[str]:
    """The negative-control paths that went unseen: each one no ``differs`` delta names, and each one whose
    case raised, since a perturbation that raises proves nothing about what the comparison sees."""
    seen = {delta["path"] for delta in control_deltas if delta["delta"] == "differs"}
    return sorted((set(control_paths) - seen) | set(raised))


def write_evidence(
    harness: str,
    *,
    base: str,
    head: str,
    image: str | None,
    verdict: str,
    findings: dict,
    directory: Path = EVIDENCE_DIR,
) -> Path:
    """Write ``<directory>/<harness>-<head>.json`` and return its path: the commits compared, the image the
    run used, its command line and verdict, and the harness's ``findings``."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{harness}-{head}.json"
    record = {
        "harness": harness,
        "base": base,
        "head": head,
        "image": image,
        "command": shlex.join(sys.orig_argv),
        "verdict": verdict,
        **findings,
    }
    path.write_text(json.dumps(record, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return path
