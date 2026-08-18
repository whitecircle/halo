"""A coordinated dataset operation must never sit inside a main-first ordering block.

``coordinated_dataset_operation`` IS the rank ordering: it barriers in ``ensure_cache_dir`` and then
joins twice on the c10d store through ``store_reject_across_ranks``. A main-first block orders its
body by holding the peers OUTSIDE it — accelerate's ``_goes_first`` barriers before the body on
non-main ranks and after it on main — so the body's coordination is shifted by one position between
them: the main rank's ``ensure_cache_dir`` barrier pairs with a peer's entry barrier, and from there
the ranks disagree on which join they are in — one side waits out a store key its peer never writes
(the ``_StorePhase`` equal-entry divergence), the other times out in a gloo barrier nobody matches.
Against ``fs_aware_main_first`` the peers sit on the outer store key and are in no collective at
all, so the main rank's barrier blocks alone.

Three parts: the gloo run proves the mechanism on a real process group, the repo scan is the
regression detector over ``src/`` and ``scripts/``, and the synthetic modules prove the scan is not
vacuous by making it fire on the exact shape above. The scan follows calls by bare name —
the module's own definitions first, then any name defined exactly once tree-wide — so it sees a body
that reaches the op through a helper here or in another file. It does not model imports, so a name
defined in several modules is followed only within its own, and a manager bound to a variable before
the ``with`` is not recognized.

    python tests/cpu/data/test_coordinated_op_not_main_first.py
"""

import ast
import contextlib
import datetime
import os
import sys

import pytest
import torch.distributed as dist
import torch.multiprocessing as mp
from accelerate import PartialState
from datasets import Dataset

from src.data.pipeline.processing import coordinated_map
from src.distributed import filesystem
from tests.common.ports import free_port

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
SCANNED_ROOTS = ("src", "scripts")

WORLD_SIZE = 2
# One port per test — a reused port collides with the previous run's lingering TCPStore.

# Well under any real dataset map: a diverged sequence can only end in a gloo timeout, and the suite
# must not sit on one.
PG_TIMEOUT_SEC = 15

# Context managers that order ranks by holding the peers outside the body.
MAIN_FIRST_MANAGERS = frozenset({"main_process_first", "local_main_process_first", "fs_aware_main_first"})

# Helpers that issue a collective or a world-ordered store join, so no main-first body may reach
# them (the ``fs_aware_main_first`` docstring states the rule; the coordinated ops carry their own
# ordering on top of it). A store join is as position-sensitive as a collective: it counts tag
# entries, so reaching one from a main-first body puts that rank permanently off-by-one.
COLLECTIVE_CALLEES = frozenset(
    {
        "coordinated_dataset_operation",
        "coordinated_map",
        "coordinated_filter",
        "process_dataset_with_map_and_filter",
        "pack_dataset_coordinated",
        "reject_across_ranks",
        "store_reject_across_ranks",
        "ensure_cache_dir",
        "fs_aware_makedirs",
        "broadcast_from_rank0",
        "resolve_shared_filesystem_consensus",
        "barrier",
    }
)


def _called_names(node: ast.AST) -> set[str]:
    """Bare callee names invoked anywhere under ``node`` (``a.b.c(...)`` contributes ``c``)."""
    names = set()
    for call in ast.walk(node):
        if not isinstance(call, ast.Call):
            continue
        func = call.func
        if isinstance(func, ast.Name):
            names.add(func.id)
        elif isinstance(func, ast.Attribute):
            names.add(func.attr)
    return names


def _reachable_names(node: ast.AST, local_defs: dict[str, ast.AST], global_defs: dict[str, ast.AST]) -> set[str]:
    """Callee names reachable from ``node``, expanding each one that resolves to a definition.

    The module's own definitions win over the tree-wide index, so a name several modules define
    (``_prepare_dataset``) still resolves to the one this file means.
    """
    seen: set[str] = set()
    pending = list(_called_names(node))
    while pending:
        name = pending.pop()
        if name in seen:
            continue
        seen.add(name)
        body = local_defs.get(name) or global_defs.get(name)
        if body is not None:
            pending.extend(_called_names(body))
    return seen


def _parse_tree(path: str) -> ast.AST:
    with open(path, encoding="utf-8") as fh:
        return ast.parse(fh.read(), filename=path)


def _python_files(roots) -> list[str]:
    paths = []
    for root in roots:
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in {"__pycache__", ".git"}]
            paths.extend(os.path.join(dirpath, f) for f in sorted(filenames) if f.endswith(".py"))
    return sorted(paths)


def _module_defs(tree: ast.AST) -> dict[str, ast.AST]:
    return {node.name: node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)}


def _unambiguous_defs(module_defs) -> dict[str, ast.AST]:
    """Bare function name → its definition, across every scanned module.

    Lets a ``with`` body that calls a helper in ANOTHER file be followed. A name defined in more than
    one module is dropped rather than guessed at: resolving it would need a real import graph, and
    picking one arbitrarily invents call edges. Callers fall back to this only after the module's own
    definitions, which is where a colliding name is resolved correctly.
    """
    defs: dict[str, ast.AST] = {}
    ambiguous: set[str] = set()
    for per_module in module_defs:
        for name, node in per_module.items():
            if name in defs:
                ambiguous.add(name)
            defs[name] = node
    for name in ambiguous:
        del defs[name]
    return defs


def find_collective_in_main_first(*roots: str) -> list[str]:
    """Report every ``with <main-first>:`` block under ``roots`` whose body can reach a collective."""
    trees = {path: _parse_tree(path) for path in _python_files(roots)}
    defs_by_path = {path: _module_defs(tree) for path, tree in trees.items()}
    global_defs = _unambiguous_defs(defs_by_path.values())

    offenders = []
    for path, tree in trees.items():
        local_defs = defs_by_path[path]
        for node in ast.walk(tree):
            if not isinstance(node, ast.With | ast.AsyncWith):
                continue
            managers = set().union(*(_called_names(item.context_expr) for item in node.items))
            if managers.isdisjoint(MAIN_FIRST_MANAGERS):
                continue
            for stmt in node.body:
                hits = sorted(_reachable_names(stmt, local_defs, global_defs) & COLLECTIVE_CALLEES)
                if hits:
                    offenders.append(f"{os.path.relpath(path, REPO_ROOT)}:{node.lineno} body reaches {hits}")
                    break
    return offenders


def test_no_collective_inside_a_main_first_block():
    """The regression detector: re-adding the wrapper around SMPO's dataset prep fails here."""
    offenders = find_collective_in_main_first(*(os.path.join(REPO_ROOT, name) for name in SCANNED_ROOTS))

    assert not offenders, (
        "a main-first block holds its peers outside the body, so a collective inside it lands at a "
        "different sequence position on main than on its peers:\n  " + "\n  ".join(offenders)
    )


def test_detector_fires_on_the_smpo_shape(tmp_path):
    """Guard the guard: the scan must catch a coordinated op reached through a helper, in this file
    or another one, and must not fire on a body that only names a manager in a string."""
    (tmp_path / "trainer_like.py").write_text(
        "from accelerate import PartialState\n"
        "\n"
        "class T:\n"
        "    def build(self, ds):\n"
        "        with PartialState().local_main_process_first():\n"
        "            ds = self._prepare_dataset(ds)\n"
        "        return ds\n"
        "\n"
        "    def _prepare_dataset(self, ds):\n"
        "        return tokenize_rows(ds)\n",
        encoding="utf-8",
    )
    (tmp_path / "helpers.py").write_text(
        "from src.data.pipeline.processing import coordinated_map\n"
        "\n"
        "def tokenize_rows(ds):\n"
        "    return coordinated_map(ds, lambda r: r)\n"
        "\n"
        "def unrelated():\n"
        "    return open('notes_on_main_process_first.txt')\n",
        encoding="utf-8",
    )

    offenders = find_collective_in_main_first(str(tmp_path))

    assert len(offenders) == 1, offenders
    assert "trainer_like.py" in offenders[0] and "coordinated_map" in offenders[0]


def _init(rank: int, port: str) -> None:
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=port, RANK=str(rank), WORLD_SIZE=str(WORLD_SIZE))
    os.environ.update(LOCAL_RANK=str(rank), LOCAL_WORLD_SIZE=str(WORLD_SIZE), ACCELERATE_USE_CPU="1")
    os.environ["HF_DATASETS_CACHE"] = os.environ.get("TMPDIR", "/tmp") + "/main_first_op_cache"
    # Bound the store joins like the process group: a diverged sequence must end in a loud timeout
    # in seconds, not DIST_STORE_TIMEOUT_HOURS (hours-granular, min 1) of suite stall. Rebound on
    # `filesystem`, which owns every store wait and resolves the getter in its own namespace.
    filesystem.get_store_timeout = lambda timeout_hours=None: datetime.timedelta(seconds=PG_TIMEOUT_SEC + 5)
    dist.init_process_group(
        "gloo", rank=rank, world_size=WORLD_SIZE, timeout=datetime.timedelta(seconds=PG_TIMEOUT_SEC)
    )


def _identity(row):
    return row


def _worker(rank: int, tmp_dir: str, wrapped: bool, port: str) -> None:
    """Run one coordinated map, optionally inside a main-first block; record the outcome."""
    _init(rank, port)
    dataset = Dataset.from_dict({"text": ["a", "b"]})

    def run():
        coordinated_map(dataset, _identity, desc="main-first probe", num_proc=1)

    try:
        if wrapped:
            with PartialState().local_main_process_first():
                run()
        else:
            run()
        outcome = "OK"
    except BaseException as e:
        outcome = f"{type(e).__name__}: {e}"
    with open(os.path.join(tmp_dir, f"outcome_{rank}.txt"), "w") as fh:
        fh.write(outcome)
    with contextlib.suppress(Exception):
        dist.destroy_process_group()


def _outcomes(tmp_path, wrapped: bool, port: str) -> list[str]:
    with contextlib.suppress(Exception):
        mp.start_processes(
            _worker, args=(str(tmp_path), wrapped, port), nprocs=WORLD_SIZE, join=True, start_method="spawn"
        )
    out = []
    for rank in range(WORLD_SIZE):
        path = tmp_path / f"outcome_{rank}.txt"
        out.append(path.read_text() if path.exists() else "KILLED")
    return out


def test_unwrapped_coordinated_map_completes_on_every_rank(tmp_path):
    """The control: on its own the coordinated op orders the ranks and both finish."""
    assert _outcomes(tmp_path, wrapped=False, port=str(free_port())) == ["OK", "OK"]


def test_main_first_wrapper_diverges_the_collective_sequence(tmp_path):
    """The mechanism: the same op wrapped in ``local_main_process_first`` cannot complete cleanly."""
    outcomes = _outcomes(tmp_path, wrapped=True, port=str(free_port()))

    assert all(outcome != "OK" for outcome in outcomes), (
        f"a rank completed the wrapped run ({outcomes}), so either accelerate stopped barriering "
        f"around the body or the coordinated op stopped issuing ordered coordination — re-derive "
        f"the ordering before trusting the repo scan above"
    )
    # How a diverged sequence surfaces: a barrier nobody matches (gloo timeout), a store key nobody
    # writes (the _StorePhase diagnostic), or a rank unpickling the wrong buffer.
    assert any("Timed out" in o or "Ran out of input" in o or "Store coordination" in o for o in outcomes), (
        f"the wrapped run failed for some reason other than a diverged coordination sequence: {outcomes}"
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
