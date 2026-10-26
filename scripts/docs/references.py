"""First-party references in text, resolved against the repository without importing anything.

A reference is a dotted name in a :data:`FIRST_PARTY` package (``src.env.env_flag``, a ``module:attr``
entry point, each name of a ``from … import`` line) or a repo-root path under one of the
:data:`REFERENCE_ROOTS`, with its ``#heading`` when it names one on a markdown page. A placeholder
(``<x>``, ``{}``, ``*``, ``$``, ``…``) cuts a reference short: only the part before it is checked. A
path that wraps after a ``/`` onto the next comment line is read whole. Names resolve off the source's
syntax trees (:class:`RepoIndex`); parsing the package's grammar needs Python 3.12.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Literal, NamedTuple

from scripts.docs.markdown import fragment_page, load_document

# The import package; every reference to it, dotted or by path, is spelled from this name.
PACKAGE = "src"
REPO_ROOT = Path(__file__).resolve().parents[2]
# Top-level packages a dotted reference may start with.
FIRST_PARTY = (PACKAGE, "scripts", "tests")
# Top-level directories a doc or a message names files under by repo-root path.
REFERENCE_ROOTS = (
    *FIRST_PARTY,
    "examples",
    "launcher-configs",
    "jinja-templates",
    "docker",
    "agent-docs",
    "human-docs",
    "skills",
    ".github",
)

# A path character or ``$`` before a reference makes it part of a longer name, path or shell variable.
_DOTTED_NAME = re.compile(rf"(?<![\w./$-])(?:{'|'.join(FIRST_PARTY)})(?:\.\w+)+(?::\w+)?")
_FROM_IMPORT = re.compile(rf"\bfrom ((?:{'|'.join(FIRST_PARTY)})(?:\.\w+)+) import ([\w, ]+)")
# The line break and the next line's comment marker where a path wraps after a ``/``; what follows
# must read as a path (a dot or a slash in it), not as prose.
_PATH_WRAP = re.compile(r"\n[ \t]*(?:#+|//+|\*)?[ \t]*")
_REPO_PATH = re.compile(
    rf"(?<![\w./$-])(?:\./)?((?:{'|'.join(map(re.escape, REFERENCE_ROOTS))})/[\w./-]*"
    rf"(?:(?<=/){_PATH_WRAP.pattern}[\w-]+(?:\.\w|/)[\w./-]*)?)(#[\w-]+)?"
)
_PLACEHOLDER_STARTS = ("<", "{", "*", "$", "[", "…", "...")
# What every module has without binding it.
_MODULE_DUNDERS = frozenset(
    {"__dict__", "__doc__", "__file__", "__loader__", "__name__", "__package__", "__path__", "__spec__"}
)
# A base from outside the repo is not read, so an attribute only it defines that first-party code names
# on a subclass is listed here, under the base's dotted name.
_EXTERNAL_BASE_ATTRIBUTES = {
    "torch.autograd.Function": frozenset({"apply"}),
    "trl.GRPOTrainer": frozenset({"_compute_loss"}),
    "typing.NamedTuple": frozenset({"_asdict", "_field_defaults", "_fields", "_make", "_replace"}),
}


class Reference(NamedTuple):
    """A first-party dotted name or a repo-root path some text names, from offset ``start`` of it."""

    kind: Literal["name", "path"]
    text: str
    why: str | None  # why it names nothing; None when it resolves
    start: int


class LoadedModule(NamedTuple):
    """What a ``load_script_module("<path>.py")`` or ``import_module("<dotted>")`` call loads."""

    name: str  # dotted; a load by path reads the file's directories as its packages
    path: str | None  # the repo-relative file a load by path runs; None for a load by name


class _Binding(NamedTuple):
    """What a name is bound to, as far as the source fixes it."""

    # "module" and "class" are first-party; "import" re-binds the first-party name ``target``;
    # "external" is bound by an import from outside the repo, "opaque" by an assignment or a def.
    kind: Literal["module", "class", "import", "external", "opaque"]
    # the module ("module"), the defining module ("class"), the re-bound name ("import", "external")
    target: str = ""
    node: ast.ClassDef | None = None


_OPAQUE = _Binding("opaque")


class _Scope(NamedTuple):
    names: dict[str, _Binding]
    # A module ``__getattr__`` or a star import answers names the source does not list.
    is_open: bool


def _first_party(dotted: str) -> bool:
    return dotted.partition(".")[0] in FIRST_PARTY


def _has_content(directory: Path) -> bool:
    """Whether ``directory`` holds anything but the bytecode caches a moved-out package leaves behind."""
    return any(
        child.name != "__pycache__" and (child.is_file() or (child.is_dir() and _has_content(child)))
        for child in directory.iterdir()
    )


def _directory_before_placeholder(path: str) -> str:
    """The directory part of ``path`` that a ``...`` in it, or a placeholder after it, leaves named."""
    return path.partition("...")[0].rpartition("/")[0] + "/"


def _block_statements(statements: list[ast.stmt]):
    """``statements`` and those nested in their blocks (``if``/``try``/``for``/``with``/``match``), not in
    function or class bodies: every statement that binds a name at this level."""
    for node in statements:
        yield node
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        for block in ("body", "orelse", "finalbody"):
            yield from _block_statements(getattr(node, block, []))
        for clause in [*getattr(node, "handlers", []), *getattr(node, "cases", [])]:
            yield from _block_statements(clause.body)


def _target_names(target: ast.expr):
    if isinstance(target, ast.Name):
        yield target.id
    elif isinstance(target, (ast.Tuple, ast.List)):
        for element in target.elts:
            yield from _target_names(element)
    elif isinstance(target, ast.Starred):
        yield from _target_names(target.value)


def _assignment_targets(node: ast.AST) -> list[ast.expr]:
    if isinstance(node, ast.Assign):
        return node.targets
    if isinstance(node, (ast.AnnAssign, ast.AugAssign, ast.For, ast.AsyncFor)):
        return [node.target]
    if isinstance(node, (ast.With, ast.AsyncWith)):
        return [item.optional_vars for item in node.items if item.optional_vars is not None]
    if isinstance(node, ast.TypeAlias):
        return [node.name]
    return []


def _import_source(node: ast.ImportFrom, package: str) -> str:
    """The absolute module a ``from … import`` reads, ``package`` being the importing file's package."""
    if not node.level:
        return node.module or ""
    parts = package.split(".")
    base = parts[: len(parts) - node.level + 1]
    return ".".join([*base, node.module] if node.module else base)


def _scope(statements: list[ast.stmt], module: str, package: str) -> _Scope:
    """The names ``statements`` bind at their own level, in ``module`` of ``package``."""
    names: dict[str, _Binding] = {}
    is_open = False
    for node in _block_statements(statements):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            names[node.name] = _OPAQUE
            is_open |= node.name == "__getattr__"
        elif isinstance(node, ast.ClassDef):
            names[node.name] = _Binding("class", module, node)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                bound = alias.asname or alias.name.partition(".")[0]
                imported = alias.name if alias.asname else bound
                names[bound] = _Binding("import" if _first_party(imported) else "external", imported)
        elif isinstance(node, ast.ImportFrom):
            source = _import_source(node, package)
            for alias in node.names:
                if alias.name == "*":
                    is_open = True
                else:
                    kind = "import" if _first_party(source) else "external"
                    names[alias.asname or alias.name] = _Binding(kind, f"{source}.{alias.name}")
        else:
            for target in _assignment_targets(node):
                names.update(dict.fromkeys(_target_names(target), _OPAQUE))
    return _Scope(names, is_open)


def _instance_attributes(cls: ast.ClassDef) -> set[str]:
    """The attributes ``cls``'s methods assign on ``self`` or ``cls``."""
    return {
        target.attr
        for node in ast.walk(cls)
        for target in _assignment_targets(node)
        if isinstance(target, ast.Attribute)
        and isinstance(target.value, ast.Name)
        and target.value.id in ("self", "cls")
    }


def dotted_name(node: ast.expr) -> str | None:
    """``a.b.c`` for a name or an attribute chain on one; None for any other expression."""
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    return ".".join([node.id, *reversed(parts)]) if isinstance(node, ast.Name) else None


def string_constants(tree: ast.Module) -> dict[str, str]:
    """The module-level names ``tree`` binds to a string literal."""
    constants = {}
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)) and isinstance(node.value, ast.Constant):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if isinstance(node.value.value, str):
                constants.update((target.id, node.value.value) for target in targets if isinstance(target, ast.Name))
    return constants


def loaded_module(call: ast.Call, constants: dict[str, str]) -> LoadedModule | None:
    """What ``call`` loads when it is a ``load_script_module`` or ``import_module`` call, bare or through
    its module, on a string literal or a name ``constants`` binds to one; None for any other call."""
    callee = (dotted_name(call.func) or "").rpartition(".")[2]
    argument = call.args[0] if call.args else None
    if isinstance(argument, ast.Constant) and isinstance(argument.value, str):
        target = argument.value
    elif isinstance(argument, ast.Name) and argument.id in constants:
        target = constants[argument.id]
    else:
        return None
    if callee == "load_script_module" and target.endswith(".py"):
        return LoadedModule(target.removesuffix(".py").replace("/", "."), target)
    return LoadedModule(target, None) if callee == "import_module" else None


class RepoIndex:
    """A repository's files and the names its first-party modules bind, read off disk and syntax trees,
    never imported.

    A dotted name resolves when its longest module prefix exists and the next part is a submodule or a
    name that module binds at its top level, in any block (``TYPE_CHECKING`` included); a submodule
    wins over a name its package binds, as ``import`` and ``patch()`` resolve it. A name an import
    binds is followed to its source. A part after a class must be one of its attributes: its body's,
    its ``self.`` assignments' and its first-party bases'. A base from outside the repo adds only what
    ``_EXTERNAL_BASE_ATTRIBUTES`` lists for it, so a renamed method stays caught under ``nn.Module`` or a
    TRL trainer; a base the source does not fix (an assignment, a call) leaves the class's attributes
    unchecked, as any other binding leaves the parts after it. A directory holding only bytecode caches
    is no package and no path.
    """

    def __init__(self, root: Path) -> None:
        self.root = root
        self._sources: dict[str, Path | None] = {}
        self._scopes: dict[str, _Scope] = {}
        self._class_attributes: dict[int, frozenset[str] | None] = {}

    def module_source(self, module: str) -> Path | None:
        """The file defining ``module``, or its directory for a namespace package; None when there is none."""
        if module not in self._sources:
            path = self.root.joinpath(*module.split("."))
            files = [file for file in (path.with_name(f"{path.name}.py"), path / "__init__.py") if file.is_file()]
            namespace = path if path.is_dir() and _has_content(path) else None
            self._sources[module] = files[0] if files else namespace
        return self._sources[module]

    def package_of(self, file: Path) -> str:
        """The dotted package a file under the root belongs to: its directory's."""
        return ".".join(file.relative_to(self.root).parent.parts)

    def import_source(self, node: ast.ImportFrom, file: Path) -> str:
        """The absolute module the ``from … import`` ``node`` in ``file`` reads."""
        return _import_source(node, self.package_of(file))

    def from_import_module(self, source: str, name: str) -> str:
        """The module ``from source import name`` loads: the submodule ``source.name`` where the repo has one,
        else ``source``."""
        return f"{source}.{name}" if self.module_source(f"{source}.{name}") is not None else source

    def imported_names(self, file: Path, tree: ast.Module) -> dict[str, str]:
        """The first-party name each top-level name of ``file`` (parsed as ``tree``) is bound to by an import."""
        scope = _scope(tree.body, "", self.package_of(file))
        return {name: binding.target for name, binding in scope.names.items() if binding.kind == "import"}

    def import_binding(self, module: str, name: str) -> str | None:
        """The dotted name an import in ``module`` binds ``name`` to, first-party or not; None when ``name`` is
        a submodule of ``module``, or ``module`` binds it otherwise or not at all."""
        if self.module_source(module) is None or self.module_source(f"{module}.{name}") is not None:
            return None
        binding = self._module_scope(module).names.get(name)
        return binding.target if binding is not None and binding.kind in ("import", "external") else None

    def unresolved_name(self, dotted: str) -> str | None:
        """Why the first-party name ``dotted`` names nothing, or None when it resolves."""
        found = self._lookup(dotted.split("."), frozenset())
        return found if isinstance(found, str) else None

    def unresolved_path(self, reference: str) -> str | None:
        """Why the repo-root path ``reference`` (a page may carry a ``#heading``) names nothing, or None."""
        path, _, fragment = reference.partition("#")
        target = self.root / path
        if not (target.is_file() or (target.is_dir() and _has_content(target))):
            return "no such file or directory"
        page = fragment_page(target) if fragment else None
        if page and fragment not in load_document(page.resolve()).anchors:
            return f"{page.name} has no heading #{fragment}"
        return None

    def references(self, text: str) -> list[Reference]:
        """Each first-party dotted name, ``from … import`` name and repo-root path ``text`` names, resolved."""
        found = []
        for match in _DOTTED_NAME.finditer(text):
            name = match.group(0).replace(":", ".")
            if text.startswith(_PLACEHOLDER_STARTS, match.end()):
                name = name.rpartition(".")[0]
            if "." in name:
                found.append(Reference("name", match.group(0), self.unresolved_name(name), match.start()))
        for match in _FROM_IMPORT.finditer(text):
            module, imported = match.groups()
            # A module that does not resolve is reported once, as the dotted name above.
            if self.unresolved_name(module) is None:
                found.extend(
                    Reference(
                        "name",
                        f"from {module} import {name}",
                        self.unresolved_name(f"{module}.{name}"),
                        match.start(),
                    )
                    for name in (alias.split()[0] for alias in imported.split(",") if alias.strip())
                )
        for match in _REPO_PATH.finditer(text):
            path, fragment = _PATH_WRAP.sub("", match.group(1)), match.group(2) or ""
            if "..." in path or text.startswith(_PLACEHOLDER_STARTS, match.end(1)):
                path, fragment = _directory_before_placeholder(path), ""
            # A path that ends a sentence carries its full stop.
            named = path.rstrip(".") + fragment
            found.append(Reference("path", named, self.unresolved_path(named), match.start()))
        return found

    def _module_scope(self, module: str) -> _Scope:
        if module not in self._scopes:
            source = self.module_source(module)
            if source.is_dir():
                self._scopes[module] = _Scope({}, False)
                return self._scopes[module]
            tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
            scope = _scope(tree.body, module, self.package_of(source))
            # A ``global`` statement binds the name at module level whenever its function runs.
            for node in ast.walk(tree):
                if isinstance(node, ast.Global):
                    for name in node.names:
                        scope.names.setdefault(name, _OPAQUE)
            self._scopes[module] = scope
        return self._scopes[module]

    def _lookup(self, parts: list[str], followed: frozenset[str]) -> _Binding | str:
        """The binding ``parts`` reaches, or why it reaches none; a binding past which nothing can be
        checked ends the walk, leaving the parts after it unread."""
        module, rest = parts[0], parts[1:]
        if module not in FIRST_PARTY or self.module_source(module) is None:
            return f"{module} is not a first-party package here"
        while rest:
            name, rest = rest[0], rest[1:]
            if self.module_source(f"{module}.{name}") is not None:
                module = f"{module}.{name}"
                continue
            scope = self._module_scope(module)
            binding = scope.names.get(name)
            if binding is None:
                if scope.is_open or name in _MODULE_DUNDERS:
                    return _OPAQUE
                return f"{module} has no {name!r}"
            if binding.kind == "import":
                if binding.target in followed:
                    return f"{binding.target} is re-exported in a cycle"
                return self._lookup([*binding.target.split("."), *rest], followed | {binding.target})
            if binding.kind == "class" and rest:
                return self._class_attribute(binding, rest[0])
            if binding.kind == "external":
                return _Binding("external", ".".join([binding.target, *rest]))
            return binding
        return _Binding("module", module)

    def _class_attribute(self, cls: _Binding, attribute: str) -> _Binding | str:
        attributes = self._attributes(cls)
        is_dunder = attribute.startswith("__") and attribute.endswith("__")
        if attributes is None or attribute in attributes or is_dunder:
            return _OPAQUE
        return f"{cls.target}.{cls.node.name} defines no {attribute!r} in its first-party hierarchy"

    def _attributes(self, cls: _Binding) -> frozenset[str] | None:
        """The attributes a first-party class defines or inherits; None when its body or a base the
        source does not fix may add any."""
        key = id(cls.node)
        if key not in self._class_attributes:
            # A hierarchy that loops back to this class reads it as open.
            self._class_attributes[key] = None
            self._class_attributes[key] = self._own_and_inherited_attributes(cls)
        return self._class_attributes[key]

    def _own_and_inherited_attributes(self, cls: _Binding) -> frozenset[str] | None:
        scope = _scope(cls.node.body, cls.target, self.package_of(self.module_source(cls.target)))
        if scope.is_open:
            return None
        attributes = set(scope.names) | _instance_attributes(cls.node)
        for base in cls.node.bases:
            # A subscripted base (``Generic[T]``) inherits from what it subscripts.
            name = dotted_name(base.value if isinstance(base, ast.Subscript) else base)
            found = self._lookup([*cls.target.split("."), *name.split(".")], frozenset()) if name else _OPAQUE
            # A builtin base (``Exception``) names nothing the index reads.
            if isinstance(found, str):
                continue
            if found.kind == "external":
                attributes |= _EXTERNAL_BASE_ATTRIBUTES.get(found.target, frozenset())
                continue
            inherited = self._attributes(found) if found.kind == "class" else None
            if inherited is None:
                return None
            attributes |= inherited
        return frozenset(attributes)
