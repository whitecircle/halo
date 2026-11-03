"""Move kit: the proof that a commit moves code without changing it.

``python -m tests.common.equivalence.move_kit --base-ref <sha>`` (``make verify-move BASE=<sha>``) reads the
base and head commits from git and writes ``evidence/move_kit-<head sha>.json``, one verdict per check:

* ``renames``: each file rename the change declares shows in ``git diff -M50%`` at ``--min-similarity``
  percent or more;
* ``definitions``: every module docstring, function, class header, method and module-level assignment
  hashes the same in both trees once each first-party name it reads is spelled in head terms, through every
  module rename git finds, every declared symbol rename, and every definition that moved to another module
  under its own name; a head definition no base one reaches fails unless the change declares it added;
* ``public_names``: no public top-level name appears or disappears, every base one placed in head terms;
* ``string_literals`` and ``raise_templates``: the multisets of string literals (docstrings aside) and of
  raised expressions over every module;
* ``side_effects``: each module's ordered statements that run first-party code at import or do more than
  bind a name: bare calls, assignments that call first-party code or set an attribute or item, first-party
  decorators, control flow, and the imports a ``noqa: F401`` or ``E402`` marks as run for their effect or
  where they stand;
* ``goldens``: every committed golden file is byte-identical but those the change declares, and, from the
  command line, each one's generator passes its check against head's own code in a fresh interpreter;
* ``patch_targets`` (review only): each patch in head on a name that a touched module only imports.

A string naming a renamed file, or a directory every Python file of which moved to one new directory, is
spelled in head terms too. A renamed method's callers through an instance (``self.old()``) read as changed:
an attribute is not resolved. Comments are not compared; docstrings are, unless ``--ignore-docstrings`` (a
prose-only change). :func:`compare_trees` imports nothing it compares, so it runs on the standard library
and git.
"""

from __future__ import annotations

import argparse
import ast
import copy
import dataclasses
import json
import re
import tempfile
import tomllib
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path, PurePosixPath

import tests.common.goldens
from scripts.docs.references import (
    FIRST_PARTY,
    PACKAGE,
    UNKNOWN_FIELD,
    RepoIndex,
    block_statements,
    dotted_name,
    module_bindings,
    own_bindings,
    patch_call,
    string_constants,
)
from tests.common.equivalence.revisions import (
    Rename,
    detect_renames,
    digest,
    export_tree,
    probe_in_tree,
    resolve_commit,
    write_evidence,
)
from tests.common.repo import REPO_ROOT

CHECKS = ("renames", "definitions", "public_names", "string_literals", "raise_templates", "side_effects", "goldens")
DEFAULT_MIN_SIMILARITY = 90
TOOL_ROOTS = ("scripts",)
# Where a dotted name or a repo path in a string may start, besides the wheel's packages.
TEXT_ROOTS = tuple(root for root in FIRST_PARTY if root != PACKAGE)
GOLDENS_DIR = Path(tests.common.goldens.__file__).parent.relative_to(REPO_ROOT)
# The qualname a module's docstring is compared under.
MODULE_DOCSTRING = "__doc__"
_SIDE_EFFECT_IMPORT = re.compile(r"#\s*noqa:[^#\n]*\b(?:F401|E402)\b")
_DEFINITIONS = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
# A real name never holds a space.
_IMPORT_MARK = "import "


@dataclass(frozen=True)
class RenameMap:
    """What a change declares beyond what git and definitions show: file renames, symbol renames and added definitions
    (``module:qualname``), and changed goldens."""

    files: Mapping[str, str] = field(default_factory=dict)
    symbols: Mapping[str, str] = field(default_factory=dict)
    added: tuple[str, ...] = ()
    goldens: tuple[str, ...] = ()

    @classmethod
    def read(cls, path: Path) -> RenameMap:
        """Parse a JSON object of these fields; any other key is refused."""
        declared = json.loads(path.read_text(encoding="utf-8"))
        known = [entry.name for entry in dataclasses.fields(cls)]
        unknown = set(declared) - set(known)
        if unknown:
            raise ValueError(f"{path}: unknown keys {sorted(unknown)}; a rename map holds {known}")
        return cls(
            declared.get("files", {}),
            declared.get("symbols", {}),
            tuple(declared.get("added", ())),
            tuple(declared.get("goldens", ())),
        )


def module_name(relative: Path) -> str:
    parts = relative.with_suffix("").parts
    return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)


@dataclass(frozen=True)
class Checkout:
    """One commit's tree on disk: its Python under the compared roots, each module by dotted name and file."""

    root: Path
    packages: tuple[str, ...]
    modules: Mapping[str, Path]

    @classmethod
    def read(cls, root: Path, roots: tuple[str, ...] | None = None) -> Checkout:
        """The checkout at ``root``; ``roots`` defaults to the wheel's packages and :data:`TOOL_ROOTS`."""
        pyproject = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
        packages = tuple(pyproject["tool"]["hatch"]["build"]["targets"]["wheel"]["packages"])
        modules = {
            module_name(path.relative_to(root)): path
            for top in roots or (*packages, *TOOL_ROOTS)
            for path in sorted((root / top).rglob("*.py"))
        }
        return cls(root, packages, modules)

    @cached_property
    def index(self) -> RepoIndex:
        return RepoIndex(self.root)

    @cached_property
    def python_files(self) -> frozenset[str]:
        """Every Python file under the packages and :data:`TEXT_ROOTS`, by repo path: what a path in a string may
        name."""
        return frozenset(
            path.relative_to(self.root).as_posix()
            for top in (*self.packages, *TEXT_ROOTS)
            for path in (self.root / top).rglob("*.py")
        )


def _directory_renames(path_renames: Mapping[str, str], base_files: Iterable[str]) -> dict[str, str]:
    """Each directory whose ``base_files`` all move under one new directory, which a string may name by its path."""
    candidates = set()
    for old, new in path_renames.items():
        old_parts, new_parts = PurePosixPath(old).parts, PurePosixPath(new).parts
        depth = 1
        while depth < min(len(old_parts), len(new_parts)) and old_parts[-depth] == new_parts[-depth]:
            candidates.add(("/".join(old_parts[:-depth]), "/".join(new_parts[:-depth])))
            depth += 1
    renamed = {}
    for old_dir, new_dir in candidates:
        inside = [path for path in base_files if path.startswith(f"{old_dir}/")]
        if inside and all(path_renames.get(path) == new_dir + path[len(old_dir) :] for path in inside):
            renamed[old_dir] = new_dir
    return renamed


@dataclass(frozen=True)
class _Spelling:
    """How one tree's first-party names read in head terms: as they are for the head, through the renames for the
    base."""

    modules: frozenset[str]
    text_roots: tuple[str, ...]
    module_renames: Mapping[str, str] = field(default_factory=dict)
    symbol_renames: Mapping[str, str] = field(default_factory=dict)
    path_renames: Mapping[str, str] = field(default_factory=dict)
    directory_renames: Mapping[str, str] = field(default_factory=dict)

    @cached_property
    def _dotted_pattern(self) -> re.Pattern:
        return re.compile(rf"(?<![\w./$-])(?:{'|'.join(map(re.escape, self.text_roots))})(?:\.\w+)+")

    @cached_property
    def _path_pattern(self) -> re.Pattern:
        return re.compile(rf"(?<![\w./$-])(?:{'|'.join(map(re.escape, self.text_roots))})/[\w./-]+")

    @cached_property
    def _directories(self) -> list[str]:
        return sorted(self.directory_renames, key=len, reverse=True)

    def symbol(self, dotted: str) -> str:
        """``dotted`` as ``module:qualname`` in head terms, a bare module as its name, an outside name as it is."""
        parts = dotted.split(".")
        cut = next((cut for cut in range(len(parts), 0, -1) if ".".join(parts[:cut]) in self.modules), 0)
        if not cut:
            return dotted
        module, qualname = ".".join(parts[:cut]), parts[cut:]
        for end in range(len(qualname), 0, -1):
            renamed = self.symbol_renames.get(f"{module}:{'.'.join(qualname[:end])}")
            if renamed is not None:
                return ".".join([renamed, *qualname[end:]])
        module = self.module_renames.get(module, module)
        return f"{module}:{'.'.join(qualname)}" if qualname else module

    def path(self, path: str) -> str:
        """``path`` in head terms: an exact file rename first, else the longest renamed directory it lies in. A
        sentence's full stop is ignored."""
        named = path.rstrip(".")
        if named in self.path_renames:
            return self.path_renames[named] + path[len(named) :]
        for old in self._directories:
            if named == old or named.startswith(f"{old}/"):
                return self.directory_renames[old] + path[len(old) :]
        return path

    def text(self, text: str) -> str:
        text = self._dotted_pattern.sub(lambda match: self.symbol(match.group(0)).replace(":", "."), text)
        return self._path_pattern.sub(lambda match: self.path(match.group(0)), text)


class _Canonical(ast.NodeTransformer):
    """A module's syntax with first-party names, strings and imports in head terms, so two trees' copies of the same
    code dump alike. A name a function, lambda or comprehension binds hides the module-level one inside it."""

    def __init__(self, bindings: Mapping[str, str], import_sources: Mapping[int, str], spelling: _Spelling):
        self.bindings = bindings
        self.import_sources = import_sources
        self.spelling = spelling
        self.hidden: frozenset[str] = frozenset()

    def _scoped(self, node: ast.AST) -> ast.AST:
        outer, self.hidden = self.hidden, self.hidden | own_bindings(node)
        try:
            return self.generic_visit(node)
        finally:
            self.hidden = outer

    visit_FunctionDef = visit_AsyncFunctionDef = visit_Lambda = _scoped
    visit_ListComp = visit_SetComp = visit_DictComp = visit_GeneratorExp = _scoped

    def _bound(self, name: str) -> str | None:
        return None if name in self.hidden else self.bindings.get(name)

    def visit_Name(self, node: ast.Name) -> ast.AST:
        bound = self._bound(node.id) if isinstance(node.ctx, ast.Load) else None
        return ast.Name(self.spelling.symbol(bound), node.ctx) if bound else node

    def visit_Attribute(self, node: ast.Attribute) -> ast.AST:
        dotted = dotted_name(node) if isinstance(node.ctx, ast.Load) else None
        if dotted and (bound := self._bound(dotted.partition(".")[0])):
            return ast.Name(self.spelling.symbol(f"{bound}.{dotted.partition('.')[2]}"), node.ctx)
        return self.generic_visit(node)

    def visit_Constant(self, node: ast.Constant) -> ast.AST:
        return ast.Constant(self.spelling.text(node.value)) if isinstance(node.value, str) else node

    def _import(self, node: ast.Import | ast.ImportFrom, targets: Iterable[str]) -> ast.AST:
        mark = ast.Expr(ast.Name(_IMPORT_MARK + ", ".join(map(self.spelling.symbol, targets)), ast.Load()))
        return ast.copy_location(mark, node)

    def visit_Import(self, node: ast.Import) -> ast.AST:
        return self._import(node, (alias.name for alias in node.names))

    def visit_ImportFrom(self, node: ast.ImportFrom) -> ast.AST:
        return self._import(node, (f"{self.import_sources[id(node)]}.{alias.name}" for alias in node.names))


def _has_docstring(node: ast.AST) -> bool:
    if not isinstance(node, (ast.Module, *_DEFINITIONS)) or not node.body:
        return False
    first = node.body[0]
    return isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str)


def _import_mark(statement: ast.stmt) -> str | None:
    if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Name):
        return statement.value.id if statement.value.id.startswith(_IMPORT_MARK) else None
    return None


def _definitions(statements: list[ast.stmt], prefix: str = "") -> Iterator[tuple[str, ast.AST]]:
    """``(qualname, what its digest covers)`` for each function, method, class header and, at module level, each bound
    name."""
    for node in block_statements(statements):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            unnamed = copy.copy(node)
            unnamed.name = ""
            yield prefix + node.name, unnamed
        elif isinstance(node, ast.ClassDef):
            header = copy.copy(node)
            header.name = ""
            header.body = [statement for statement in node.body if not isinstance(statement, _DEFINITIONS)]
            yield prefix + node.name, header
            yield from _definitions(node.body, f"{prefix}{node.name}.")
        elif not prefix and isinstance(node, (ast.Assign, ast.AnnAssign)) and node.value is not None:
            covered = ast.Tuple([node.value, getattr(node, "annotation", None) or ast.Constant(None)], ast.Load())
            for target in node.targets if isinstance(node, ast.Assign) else [node.target]:
                elements = target.elts if isinstance(target, (ast.Tuple, ast.List)) else [target]
                yield from ((element.id, covered) for element in elements if isinstance(element, ast.Name))


def _module_definitions(module: ast.Module) -> Iterator[tuple[str, ast.AST]]:
    """The module's docstring as :data:`MODULE_DOCSTRING`, then its :func:`_definitions`, numbered where a qualname
    repeats (``name#2``)."""
    pairs = ([(MODULE_DOCSTRING, module.body[0])] if _has_docstring(module) else []) + list(_definitions(module.body))
    seen = Counter()
    for qualname, node in pairs:
        seen[qualname] += 1
        yield (qualname if seen[qualname] == 1 else f"{qualname}#{seen[qualname]}"), node


def _calls_first_party(node: ast.AST) -> bool:
    """Whether evaluating ``node`` calls first-party code, which the canonical form spells ``module:qualname``."""
    return any(isinstance(call, ast.Call) and ":" in (dotted_name(call.func) or "") for call in ast.walk(node))


def _side_effect(statement: ast.stmt, source_lines: list[str]) -> str | None:
    """Canonical text of a module-level statement doing more than binding a name, else None; only first-party code may
    register as it runs."""
    if (mark := _import_mark(statement)) is not None:
        lines = "\n".join(source_lines[statement.lineno - 1 : statement.end_lineno])
        return mark if _SIDE_EFFECT_IMPORT.search(lines) else None
    if isinstance(statement, _DEFINITIONS):
        registering = [
            f"@{ast.unparse(decorator)}"
            for decorator in statement.decorator_list
            if ":" in (dotted_name(decorator) or "") or _calls_first_party(decorator)
        ]
        return f"{' '.join(registering)} {type(statement).__name__} {statement.name}" if registering else None
    if isinstance(statement, (ast.Assign, ast.AnnAssign)):
        targets = statement.targets if isinstance(statement, ast.Assign) else [statement.target]
        runs_toolkit_code = statement.value is not None and _calls_first_party(statement.value)
        mutates = any(not isinstance(target, (ast.Name, ast.Tuple, ast.List)) for target in targets)
        return ast.unparse(statement) if runs_toolkit_code or mutates else None
    if isinstance(statement, ast.If) and (dotted_name(statement.test) or "").endswith("TYPE_CHECKING"):
        return None
    return ast.unparse(statement)


def _parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


@dataclass
class _ModuleFacts:
    definitions: dict[str, str] = field(default_factory=dict)
    strings: Counter = field(default_factory=Counter)
    raises: Counter = field(default_factory=Counter)
    side_effects: list[str] = field(default_factory=list)


def _module_facts(
    checkout: Checkout, module: str, spelling: _Spelling, *, ignore_docstrings: bool = False
) -> _ModuleFacts:
    """What ``module`` defines, says, raises and runs at import, every first-party name spelled through
    ``spelling``."""
    path = checkout.modules[module]
    source = path.read_text(encoding="utf-8")
    syntax = ast.parse(source, filename=str(path))
    import_sources = {
        id(node): checkout.index.import_source(node, path)
        for node in ast.walk(syntax)
        if isinstance(node, ast.ImportFrom)
    }
    bindings = module_bindings(syntax, module, checkout.index.package_of(path))
    canonical = _Canonical(bindings, import_sources, spelling).visit(syntax)
    documented = [node for node in ast.walk(canonical) if _has_docstring(node)]
    skipped = {id(node.body[0].value) for node in documented}
    module_docstring = canonical.body[0] if _has_docstring(canonical) else None
    if ignore_docstrings:
        for node in documented:
            node.body = node.body[1:]
    skipped |= {id(part) for node in ast.walk(canonical) if isinstance(node, ast.JoinedStr) for part in node.values}

    facts = _ModuleFacts()
    facts.definitions = {
        qualname: digest(ast.dump(node).encode()) for qualname, node in _module_definitions(canonical)
    }
    for node in ast.walk(canonical):
        if isinstance(node, ast.JoinedStr):
            facts.strings[ast.unparse(node)] += 1
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in skipped:
            facts.strings[node.value] += 1
        elif isinstance(node, ast.Raise) and node.exc is not None:
            facts.raises[ast.unparse(node.exc)] += 1
    lines = source.splitlines()
    facts.side_effects = [
        text
        for statement in canonical.body
        if statement is not module_docstring and (text := _side_effect(statement, lines)) is not None
    ]
    return facts


def _qualnames(checkout: Checkout) -> list[tuple[str, str]]:
    return [
        (module, qualname)
        for module, path in checkout.modules.items()
        for qualname, _ in _module_definitions(_parse(path))
    ]


def _split_symbol(symbol: str) -> tuple[str, str]:
    module, separator, qualname = symbol.partition(":")
    if not separator or not module or not qualname:
        raise ValueError(f"symbol {symbol!r} is not spelled module:qualname")
    return module, qualname


def _spelled(key: tuple[str, str]) -> str:
    return f"{key[0]}:{key[1]}"


def _pair_definitions(
    base: Checkout, head: Checkout, module_renames: Mapping[str, str], renames: RenameMap
) -> tuple[dict[tuple[str, str], tuple[str, str]], dict[tuple[str, str], tuple[str, str]]]:
    """Where each base definition lives in head, and which placements are moves. Declared symbol renames first, else
    the module's renamed location, else the only unplaced head definition of its qualname."""
    base_keys, head_keys = _qualnames(base), set(_qualnames(head))
    declared_keys = {_split_symbol(old): _split_symbol(new) for old, new in renames.symbols.items()}
    absent = sorted(set(map(_spelled, declared_keys)) - set(map(_spelled, base_keys)))
    in_head = {*declared_keys.values(), *map(_split_symbol, renames.added)}
    absent += sorted(set(map(_spelled, in_head)) - set(map(_spelled, head_keys)))
    if absent:
        raise ValueError(f"the declared renames and additions name definitions the trees lack: {absent}")
    placed = {key: declared_keys.get(key, (module_renames.get(key[0], key[0]), key[1])) for key in base_keys}
    reached = set(placed.values())
    unplaced, unreached = defaultdict(list), defaultdict(list)
    for key, target in placed.items():
        if target not in head_keys:
            unplaced[key[1]].append(key)
    for key in head_keys - reached:
        unreached[key[1]].append(key)
    moves = {
        keys[0]: unreached[qualname][0]
        for qualname, keys in unplaced.items()
        if len(keys) == 1 and len(unreached[qualname]) == 1
    }
    return placed | moves, moves


def _verdict(failed: bool) -> str:
    return "fail" if failed else "pass"


def _multiset_check(base: Counter, head: Counter) -> dict:
    only_base, only_head = base - head, head - base
    return {
        "verdict": _verdict(bool(only_base or only_head)),
        "only_base": dict(sorted(only_base.items())),
        "only_head": dict(sorted(only_head.items())),
    }


def _renames_check(detected: list[Rename], declared: Mapping[str, str], min_similarity: int) -> dict:
    found = {(rename.old, rename.new): rename.similarity for rename in detected}
    rows = [{"old": old, "new": new, "similarity": found.get((old, new))} for old, new in sorted(declared.items())]
    short = [row for row in rows if row["similarity"] is None or row["similarity"] < min_similarity]
    return {
        "verdict": _verdict(bool(short)),
        "min_similarity": min_similarity,
        "declared": rows,
        "below_minimum": short,
        "detected": [vars(rename) for rename in detected],
    }


def _definitions_check(
    before: Mapping[str, _ModuleFacts],
    after: Mapping[str, _ModuleFacts],
    placements: Mapping[tuple[str, str], tuple[str, str]],
    moves: Mapping[tuple[str, str], tuple[str, str]],
    declared_added: Iterable[str],
) -> dict:
    head_digests = {
        (module, qualname): hashed for module, facts in after.items() for qualname, hashed in facts.definitions.items()
    }
    changed, removed = [], []
    for module, facts in before.items():
        for qualname, hashed in facts.definitions.items():
            target = placements[(module, qualname)]
            if target not in head_digests:
                removed.append(_spelled((module, qualname)))
            elif head_digests[target] != hashed:
                changed.append(f"{_spelled((module, qualname))} -> {_spelled(target)}")
    added = head_digests.keys() - set(placements.values())
    # A declaration naming a docstring --ignore-docstrings leaves uncompared is moot.
    declared = set(map(_split_symbol, declared_added)) & head_digests.keys()
    return {
        "verdict": _verdict(bool(changed or removed or added ^ declared)),
        "compared": len(placements),
        "moved": sorted(f"{_spelled(old)} -> {_spelled(new)}" for old, new in moves.items()),
        "changed": sorted(changed),
        "removed": sorted(removed),
        "added": sorted(map(_spelled, added - declared)),
        "declared_added": sorted(map(_spelled, added & declared)),
        "declared_not_added": sorted(map(_spelled, declared - added)),
    }


def _public_names_check(base_public: Iterable[tuple[str, str]], head_public: Iterable[tuple[str, str]]) -> dict:
    def by_module(keys: Iterable[tuple[str, str]]) -> dict[str, set[str]]:
        modules = defaultdict(set)
        for module, qualname in keys:
            modules[module].add(qualname)
        return modules

    before, after = by_module(base_public), by_module(head_public)
    modules = {
        module: {
            "missing": sorted(before[module] - after[module]),
            "unexpected": sorted(after[module] - before[module]),
        }
        for module in sorted(before.keys() | after.keys())
        if before[module] != after[module]
    }
    return {"verdict": _verdict(bool(modules)), "modules": modules}


def _side_effects_check(
    before: Mapping[str, _ModuleFacts], after: Mapping[str, _ModuleFacts], module_renames: Mapping[str, str]
) -> dict:
    sequences = defaultdict(lambda: {"base": [], "head": []})
    for module, facts in before.items():
        sequences[module_renames.get(module, module)]["base"] = facts.side_effects
    for module, facts in after.items():
        sequences[module]["head"] = facts.side_effects
    differing = {module: pair for module, pair in sorted(sequences.items()) if pair["base"] != pair["head"]}
    return {"verdict": _verdict(bool(differing)), "modules": differing}


def run_golden_checks(root: Path) -> dict[str, str]:
    """Each golden checked by its generator against the tree's own code, in a fresh interpreter so no other import
    order reaches it."""
    outcomes = {}
    for golden in sorted((root / GOLDENS_DIR).glob("*.json")):
        generator = golden.with_suffix(".py")
        if not generator.is_file():
            outcomes[golden.name] = "no generator"
            continue
        result = probe_in_tree(root, module_name(generator.relative_to(root)))
        reported = result.stderr.strip().splitlines() or [f"exit status {result.returncode}"]
        outcomes[golden.name] = "pass" if result.returncode == 0 else reported[-1].replace(f"{root}/", "")
    return outcomes


def _goldens_check(
    base: Checkout, head: Checkout, declared: Iterable[str], generator_checks: Mapping[str, str] | None
) -> dict:
    def files(checkout: Checkout) -> dict[str, bytes]:
        return {path.name: path.read_bytes() for path in sorted((checkout.root / GOLDENS_DIR).glob("*.json"))}

    before, after = files(base), files(head)
    differing = {name for name in before.keys() | after.keys() if before.get(name) != after.get(name)}
    declared = set(declared)
    undeclared = differing - declared
    failing = {name: outcome for name, outcome in (generator_checks or {}).items() if outcome != "pass"}
    return {
        "verdict": _verdict(bool(undeclared or declared - differing or failing)),
        "changed": sorted(name for name in undeclared if name in before and name in after),
        "removed": sorted(name for name in undeclared if name not in after),
        "added": sorted(name for name in undeclared if name not in before),
        "declared": sorted(declared & differing),
        "declared_unchanged": sorted(declared - differing),
        "generator_checks": generator_checks,
    }


def _patch_review(head: Checkout, touched: set[str]) -> dict:
    """Each patch in head's Python on a name a touched module binds only by import: it replaces that binding, not the
    defining module's."""
    index = head.index
    rows = []
    files = {*head.modules.values(), *(path for top in TEXT_ROOTS for path in (head.root / top).rglob("*.py"))}
    for path in sorted(files):
        syntax = _parse(path)
        constants = string_constants(syntax)
        owners = index.owners(path, syntax, constants)
        for node in ast.walk(syntax):
            patch = patch_call(node, owners, constants) if isinstance(node, ast.Call) else None
            if patch is None or UNKNOWN_FIELD in patch.target:
                continue
            parts = patch.target.split(".")
            cut = next((cut for cut in range(len(parts) - 1, 0, -1) if index.module_source(".".join(parts[:cut]))), 0)
            module = ".".join(parts[:cut])
            if cut and module in touched and (bound_from := index.import_binding(module, parts[cut])):
                rows.append(
                    {
                        "file": str(path.relative_to(head.root)),
                        "line": node.lineno,
                        "target": patch.target,
                        "bound_from": bound_from,
                    }
                )
    return {"verdict": "review", "targets": rows}


def compare_trees(
    base: Checkout,
    head: Checkout,
    detected: list[Rename],
    renames: RenameMap | None = None,
    *,
    min_similarity: int = DEFAULT_MIN_SIMILARITY,
    ignore_docstrings: bool = False,
    golden_checks: Mapping[str, str] | None = None,
) -> dict[str, dict]:
    """Every check of the module docstring from ``base`` to ``head``: each check's verdict and findings."""
    renames = renames or RenameMap()
    path_renames = {rename.old: rename.new for rename in detected} | dict(renames.files)
    module_renames = {
        module_name(Path(old)): module_name(Path(new))
        for old, new in path_renames.items()
        if old.endswith(".py") and new.endswith(".py")
    }
    module_renames = {old: new for old, new in module_renames.items() if old in base.modules and new in head.modules}
    placements, moves = _pair_definitions(base, head, module_renames, renames)
    symbol_renames = dict(renames.symbols) | {_spelled(old): _spelled(new) for old, new in moves.items()}
    base_spelling = _Spelling(
        frozenset(base.modules),
        (*base.packages, *TEXT_ROOTS),
        module_renames,
        symbol_renames,
        path_renames,
        _directory_renames(path_renames, base.python_files),
    )
    head_spelling = _Spelling(frozenset(head.modules), (*head.packages, *TEXT_ROOTS))
    before = {
        module: _module_facts(base, module, base_spelling, ignore_docstrings=ignore_docstrings)
        for module in base.modules
    }
    after = {
        module: _module_facts(head, module, head_spelling, ignore_docstrings=ignore_docstrings)
        for module in head.modules
    }

    def is_public(qualname: str) -> bool:
        return "." not in qualname and "#" not in qualname and not qualname.startswith("_")

    additions = set(map(_split_symbol, renames.added)) - set(placements.values())
    base_public = (placements[key] for key in placements if is_public(key[1]))
    head_public = (
        (module, qualname)
        for module, facts in after.items()
        for qualname in facts.definitions
        if is_public(qualname) and (module, qualname) not in additions
    )
    touched = {
        *module_renames.values(),
        *(new[0] for new in moves.values()),
        *(module_renames.get(old[0], old[0]) for old in moves),
    }
    return {
        "renames": _renames_check(detected, renames.files, min_similarity),
        "definitions": _definitions_check(before, after, placements, moves, renames.added),
        "public_names": _public_names_check(base_public, head_public),
        "string_literals": _multiset_check(
            sum((facts.strings for facts in before.values()), Counter()),
            sum((facts.strings for facts in after.values()), Counter()),
        ),
        "raise_templates": _multiset_check(
            sum((facts.raises for facts in before.values()), Counter()),
            sum((facts.raises for facts in after.values()), Counter()),
        ),
        "side_effects": _side_effects_check(before, after, module_renames),
        "goldens": _goldens_check(base, head, renames.goldens, golden_checks),
        "patch_targets": _patch_review(head, touched),
    }


def verify_move(
    repo: Path,
    base: str,
    head: str,
    renames: RenameMap | None = None,
    *,
    min_similarity: int = DEFAULT_MIN_SIMILARITY,
    ignore_docstrings: bool = False,
    roots: tuple[str, ...] | None = None,
    check_goldens: bool = False,
) -> dict[str, dict]:
    """:func:`compare_trees` over two commits of ``repo``; ``check_goldens`` imports head's code and so needs the
    image."""
    with tempfile.TemporaryDirectory() as scratch:
        base_root = export_tree(repo, base, Path(scratch, "base"))
        head_root = export_tree(repo, head, Path(scratch, "head"))
        return compare_trees(
            Checkout.read(base_root, roots),
            Checkout.read(head_root, roots),
            detect_renames(repo, base, head),
            renames,
            min_similarity=min_similarity,
            ignore_docstrings=ignore_docstrings,
            golden_checks=run_golden_checks(head_root) if check_goldens else None,
        )


def overall_verdict(checks: Mapping[str, dict], required: Iterable[str] = CHECKS) -> str:
    return _verdict(any(checks[name]["verdict"] != "pass" for name in required))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prove that one commit only moves another's code.")
    parser.add_argument("--base-ref", required=True, help="the commit the change is compared with")
    parser.add_argument("--head-ref", default="HEAD", help="the commit under test (default: HEAD)")
    parser.add_argument(
        "--renames",
        type=Path,
        help='a JSON file under the repo: {"files": {old: new}, "symbols": {"mod:qual": "mod:qual"}, '
        '"added": ["mod:qual"], "goldens": ["name.json"]}',
    )
    parser.add_argument(
        "--min-similarity", type=int, default=DEFAULT_MIN_SIMILARITY, help="percent a declared rename needs"
    )
    parser.add_argument(
        "--ignore-docstrings", action="store_true", help="compare definitions without their docstrings"
    )
    parser.add_argument(
        "--roots", nargs="+", help="top-level directories to compare (default: the packages and scripts)"
    )
    parser.add_argument("--require", nargs="+", choices=CHECKS, default=list(CHECKS), help="checks the verdict needs")
    parser.add_argument("--image", help="the image the run used, recorded in the evidence")
    args = parser.parse_args(argv)

    base, head = resolve_commit(REPO_ROOT, args.base_ref), resolve_commit(REPO_ROOT, args.head_ref)
    renames = RenameMap.read(args.renames) if args.renames else RenameMap()
    checks = verify_move(
        REPO_ROOT,
        base,
        head,
        renames,
        min_similarity=args.min_similarity,
        ignore_docstrings=args.ignore_docstrings,
        roots=tuple(args.roots) if args.roots else None,
        check_goldens=True,
    )
    verdict = overall_verdict(checks, args.require)
    path = write_evidence(
        "move_kit",
        base=base,
        head=head,
        image=args.image,
        verdict=verdict,
        findings={"required": args.require, "renames": dataclasses.asdict(renames), "checks": checks},
    )
    for name, check in checks.items():
        print(f"{name:18} {check['verdict']}")
    print(f"move kit: {verdict} ({path})")
    return 0 if verdict == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
