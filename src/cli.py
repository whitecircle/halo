"""Halo command-line launcher.

Two entry points wrap the repository's runnable scripts so a method or tool can be
launched by name instead of spelling out ``python``/``torchrun`` and a file path:

* ``halo launch`` — config-driven training methods under ``scripts/training``
  (run with ``python``, ``torchrun`` via ``--nproc``, or ``accelerate launch`` via ``--accelerate``).
* ``halo run`` — flag-driven tools under every other ``scripts/`` subtree (after-training,
  inference, before-training data preparation, environments, profiling, ...), discovered
  from the tree rather than a hand-maintained list.
"""

from __future__ import annotations

import os
import shlex
import sys
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Annotated

import click
import typer

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = Path("scripts")
TRAINING_ROOT = SCRIPTS_ROOT / "training"

# Excluded from ``halo run``: ``training`` belongs to ``halo launch``, and ``diagrams`` holds
# doc-asset generators that mostly lack a ``__main__`` guard, so indexing it would surface only the
# one runnable generator.
_NON_TOOL_DIRS = frozenset({TRAINING_ROOT.name, "diagrams"})

_EXTRA_ARGS = {"allow_extra_args": True, "ignore_unknown_options": True}

# torchrun's multi-node / rendezvous flags. ``halo launch`` builds a single-node command line and
# passes everything after the config to the script, whose parser would reject these as unknown
# config fields. Multi-node runs launch one torchrun per node directly.
_TORCHRUN_MULTINODE_FLAGS = frozenset(
    {
        "--nnodes",
        "--node_rank",
        "--node-rank",
        "--master_addr",
        "--master-addr",
        "--master_port",
        "--master-port",
        "--rdzv_backend",
        "--rdzv-backend",
        "--rdzv_endpoint",
        "--rdzv-endpoint",
        "--rdzv_id",
        "--rdzv-id",
    }
)

app = typer.Typer(no_args_is_help=True, add_completion=False)


def _tool_roots(repo_root: Path) -> tuple[Path, ...]:
    """Every top-level ``scripts/`` subtree except the ones in :data:`_NON_TOOL_DIRS`.

    Discovered from the tree, so a new subtree (e.g. ``profiling``) indexes without a code change;
    directories holding no runnable scripts drop out in :func:`_index`.
    """

    scripts_root = repo_root / SCRIPTS_ROOT
    if not scripts_root.exists():
        return ()
    return tuple(
        SCRIPTS_ROOT / entry.name
        for entry in sorted(scripts_root.iterdir())
        if entry.is_dir() and entry.name not in _NON_TOOL_DIRS and not entry.name.startswith("_")
    )


def _slug(text: str) -> str:
    return text.replace("_", "-")


def _lookup_key(name: str) -> str:
    """Match key for a name typed on the command line.

    The index is hyphenated (``offline-grpo``) while the script and its config directory spell the
    method ``offline_grpo``; both spellings must resolve to the same entry.
    """
    return _slug(name).lower()


def _stem_alias_key(stem: str) -> str:
    """Aliasing key that folds singular/plural stem variants together.

    A bare-stem alias must identify exactly one script. ``reward`` and ``rewards`` fold to the same
    key, so neither gets an alias and the qualified name is required.
    """
    return stem.removesuffix("s")


def _is_runnable(path: Path) -> bool:
    """A script is launchable if it has a ``__main__`` guard."""
    return "__main__" in path.read_text(encoding="utf-8")


def _index(roots: Iterable[Path], repo_root: Path, name_base: Path) -> dict[str, Path]:
    """Map slugged names to runnable scripts found under ``roots``.

    Each script is indexed by its path relative to ``name_base`` (e.g. ``preference/smpo`` or
    ``after-training/merge-ep-shards``), plus a short alias of its file stem when that stem, folded
    with its singular/plural variant (:func:`_stem_alias_key`), is unambiguous across the index.
    """

    discovered: list[tuple[str, str, Path]] = []
    for root in roots:
        script_root = repo_root / root
        if not script_root.exists():
            continue
        for script in sorted(script_root.rglob("*.py")):
            if not _is_runnable(script):
                continue
            parts = [_slug(part) for part in script.relative_to(repo_root / name_base).with_suffix("").parts]
            discovered.append(("/".join(parts), _slug(script.stem), script))

    index = {name: script for name, _, script in discovered}
    stem_counts = Counter(_stem_alias_key(short) for _, short, _ in discovered)
    for _, short, script in discovered:
        if stem_counts[_stem_alias_key(short)] == 1:
            index.setdefault(short, script)
    return index


def training_methods(root: Path | None = None) -> dict[str, Path]:
    """Index training methods from ``scripts/training`` file paths (``sft``, ``preference/smpo``, ...).

    Path-derived names are registered before stem aliases, so an ambiguous stem cannot overwrite one.
    """
    return _index((TRAINING_ROOT,), root or REPO_ROOT, TRAINING_ROOT)


def tool_commands(root: Path | None = None) -> dict[str, Path]:
    """Index the non-training tools (after-training, inference, before-training, environments,
    profiling, ...) from every ``scripts/`` subtree :data:`_NON_TOOL_DIRS` does not exclude."""

    repo_root = root or REPO_ROOT
    return _index(_tool_roots(repo_root), repo_root, SCRIPTS_ROOT)


def command(
    script: Path,
    config: Path | None,
    *,
    nproc: int,
    args: list[str],
    accelerate_config: Path | None = None,
    port: int | None = None,
) -> list[str]:
    """Build the process argv.

    Picks the launcher: ``accelerate launch`` when ``accelerate_config`` is given (standard
    FSDP data-parallel), ``torchrun`` when ``nproc > 1`` (EP/CP/TP and any multi-process run),
    otherwise plain Python. ``nproc`` maps to ``--num_processes`` under accelerate and
    ``--nproc_per_node`` under torchrun; ``port`` to ``--main_process_port`` / ``--master_port``
    (concurrent multi-process launches collide on the default rendezvous port without it).
    """

    if accelerate_config is not None:
        launcher = ["accelerate", "launch", "--config_file", str(accelerate_config)]
        if nproc > 1:
            launcher += ["--num_processes", str(nproc)]
        if port is not None:
            launcher += ["--main_process_port", str(port)]
    elif nproc > 1:
        launcher = ["torchrun", f"--nproc_per_node={nproc}"]
        if port is not None:
            launcher.append(f"--master_port={port}")
    else:
        if port is not None:
            raise typer.BadParameter("--port only applies to multi-process launches (--nproc > 1 or --accelerate)")
        launcher = [sys.executable]
    positionals = [str(script)] if config is None else [str(script), str(config)]
    return [*launcher, *positionals, *args]


def resolve_config(config: Path, root: Path) -> Path:
    resolved = config if config.is_absolute() or config.exists() else root / config
    if not resolved.exists():
        raise typer.BadParameter(f"config not found: {config} (resolved to {resolved})")
    # Always absolute: _execute chdirs to --root before exec, so a path validated against the
    # caller's cwd would otherwise re-resolve there and select a different file.
    return resolved.resolve()


def _extra_args(ctx: typer.Context) -> list[str]:
    return ctx.args[1:] if ctx.args[:1] == ["--"] else ctx.args


def reject_multinode_flags(args: list[str]) -> None:
    """Reject torchrun's multi-node flags among the pass-through args (``_TORCHRUN_MULTINODE_FLAGS``)."""
    stray = [arg for arg in args if arg.split("=", 1)[0] in _TORCHRUN_MULTINODE_FLAGS]
    if stray:
        raise typer.BadParameter(
            f"{stray} are torchrun flags, and halo launch is single-node (they would reach the training "
            f"script as unknown config fields). Launch one torchrun per node directly — "
            f"agent-docs/parallelism/launch-recipes.md."
        )


def _show(index: dict[str, Path], root: Path) -> None:
    if not index:
        typer.echo(f"(none found under {root})")
        return
    width = max(len(name) for name in index)
    for name, script in sorted(index.items()):
        typer.echo(f"{name:<{width}}  {script}")


def _lookup(index: dict[str, Path], name: str, kind: str, root: Path) -> Path:
    """Resolve a typed name against the hyphenated index (see :func:`_lookup_key`)."""

    # An empty index means a wrong --root (or an install without the scripts tree), not a bad name.
    if not index:
        raise typer.BadParameter(f"no {kind}s found under {root} — is --root a Halo checkout?")
    entries = {_lookup_key(indexed): indexed for indexed in index}
    key = _lookup_key(name)
    if key in entries:
        return index[entries[key]]
    # A stem dropped for ambiguity still names real scripts; list them rather than report "unknown".
    stem = _stem_alias_key(key)
    candidates = sorted(n for n in index if _stem_alias_key(_lookup_key(n).rsplit("/", 1)[-1]) == stem)
    if candidates:
        scripts = {index[n] for n in candidates}
        # A qualified name and its own alias both match the stem; a single script is not ambiguous.
        if len(scripts) == 1:
            return scripts.pop()
        raise typer.BadParameter(f"{kind} '{name}' is ambiguous; use a qualified name: {', '.join(candidates)}")
    known = ", ".join(sorted(index)[:20])
    raise typer.BadParameter(f"unknown {kind} '{name}'. Known {kind}s: {known} (--list for all)")


def _execute(argv: list[str], cwd: Path | None, *, dry_run: bool) -> None:
    """Exec ``argv``, from ``cwd`` when one is given.

    ``halo launch`` passes the repo root so a training config's relative paths (dataset dirs, output
    dirs, accelerate configs) resolve there whatever the caller's cwd; :func:`resolve_config`
    absolutizes the config for the same reason.

    ``halo run`` passes ``None``: a tool's path flags were typed against the caller's cwd, and
    chdir'ing would re-root a relative ``--output_dir ./out`` under the repo, which lives on the
    small root filesystem. No chdir is needed — the indexed script path is already absolute and the
    image puts ``src`` on ``PYTHONPATH``.
    """
    if dry_run:
        typer.echo(" ".join(shlex.quote(part) for part in argv))
        return
    if cwd is not None:
        os.chdir(cwd)
    os.execvp(argv[0], argv)


@app.callback()
def halo() -> None:
    """Halo command line tools."""


@app.command(context_settings=_EXTRA_ARGS)
def launch(
    ctx: typer.Context,
    method: Annotated[str | None, typer.Argument(help="Method name, e.g. sft or preference/smpo")] = None,
    config: Annotated[Path | None, typer.Argument(help="YAML config path")] = None,
    nproc: Annotated[
        int,
        typer.Option(
            "--nproc", "-n", min=1, help="Process count (torchrun --nproc_per_node, or accelerate --num_processes)"
        ),
    ] = 1,
    accelerate: Annotated[
        Path | None,
        typer.Option(
            "--accelerate",
            "-a",
            help="Launch via `accelerate launch` with this config (e.g. launcher-configs/accelerate/fsdp2_gradop_config.yaml)",
        ),
    ] = None,
    port: Annotated[
        int | None,
        typer.Option("--port", "-p", help="Rendezvous port (torchrun --master_port / accelerate --main_process_port)"),
    ] = None,
    list_methods: Annotated[bool, typer.Option("--list", help="List indexed training methods")] = False,
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Print command without executing")] = False,
    root: Annotated[Path, typer.Option("--root", help="Repository root")] = REPO_ROOT,
) -> None:
    """Launch a training method without spelling out Python, torchrun/accelerate, or script paths.

    Defaults to Python (single process) or torchrun (``--nproc > 1``, required for EP/CP/TP).
    Pass ``--accelerate <config>`` for standard FSDP data-parallel via ``accelerate launch``.

    The run executes from ``--root``, so relative paths in the config (and in CLI overrides) resolve
    against the repository root, not the caller's cwd.
    """

    root = root.resolve()
    index = training_methods(root)
    if list_methods:
        _show(index, root)
        return
    if method is None or config is None:
        raise click.UsageError("halo launch requires METHOD and CONFIG")
    script = _lookup(index, method, "method", root)
    accelerate_config = resolve_config(accelerate, root) if accelerate is not None else None
    extra = _extra_args(ctx)
    reject_multinode_flags(extra)
    argv = command(
        script,
        resolve_config(config, root),
        nproc=nproc,
        args=extra,
        accelerate_config=accelerate_config,
        port=port,
    )
    _execute(argv, root, dry_run=dry_run)


@app.command(context_settings=_EXTRA_ARGS)
def run(
    ctx: typer.Context,
    tool: Annotated[
        str | None, typer.Argument(help="Tool name, e.g. merge-ep-shards or inference/reward-model/rm-scoring")
    ] = None,
    nproc: Annotated[int, typer.Option("--nproc", "-n", min=1, help="Use torchrun with this many processes")] = 1,
    port: Annotated[
        int | None,
        typer.Option("--port", "-p", help="Rendezvous port (torchrun --master_port), for concurrent runs"),
    ] = None,
    list_tools: Annotated[bool, typer.Option("--list", help="List indexed tools")] = False,
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Print command without executing")] = False,
    root: Annotated[Path, typer.Option("--root", help="Repository root")] = REPO_ROOT,
) -> None:
    """Run a non-training tool (after-training, inference, before-training, environments,
    profiling, ...) by name.

    Flags after the tool name pass through to the underlying script, except the launcher's own
    (--nproc/-n, --port/-p, --list, --dry-run, --root), which the CLI consumes first. Put ``--``
    before a tool flag that collides (``halo run prepare-dataset -- --dry-run``).

    The tool runs in the caller's cwd (unlike ``halo launch``), so a relative path flag means what it
    would mean had the script been invoked directly.
    """

    root = root.resolve()
    index = tool_commands(root)
    if list_tools:
        _show(index, root)
        return
    if tool is None:
        raise click.UsageError("halo run requires a TOOL name (use --list to see options)")
    script = _lookup(index, tool, "tool", root)
    argv = command(script, None, nproc=nproc, args=_extra_args(ctx), port=port)
    _execute(argv, None, dry_run=dry_run)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
