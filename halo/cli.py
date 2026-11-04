"""``halo launch`` runs a training method, ``halo run`` a tool from any other ``scripts/`` subtree."""

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
from torch.numa.binding import AffinityMode

REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS_ROOT = Path("scripts")
TRAINING_ROOT = _SCRIPTS_ROOT / "training"

# ``training`` belongs to ``halo launch``; ``diagrams`` holds doc generators that mostly lack a ``__main__`` guard.
_NON_TOOL_DIRS = frozenset({TRAINING_ROOT.name, "diagrams"})

_EXTRA_ARGS = {"allow_extra_args": True, "ignore_unknown_options": True}
_LAUNCH_USAGE = "halo launch METHOD CONFIG [launcher flags] [--field=value ...]"
_RUN_USAGE = "halo run TOOL [launcher flags] [tool flags]"

# ``halo launch`` is single-node and forwards the rest to the script, whose parser rejects these as unknown fields.
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
    """Every top-level ``scripts/`` subtree but :data:`_NON_TOOL_DIRS`; one with no runnable script drops out later."""
    scripts_root = repo_root / _SCRIPTS_ROOT
    if not scripts_root.exists():
        return ()
    return tuple(
        _SCRIPTS_ROOT / entry.name
        for entry in sorted(scripts_root.iterdir())
        if entry.is_dir() and entry.name not in _NON_TOOL_DIRS and not entry.name.startswith("_")
    )


def _slug(text: str) -> str:
    return text.replace("_", "-")


def _lookup_key(name: str) -> str:
    """Match key for a typed name: the index is hyphenated (``reward/reward-model``), scripts ``reward_model``."""
    return _slug(name).lower()


def _stem_alias_key(stem: str) -> str:
    """Folds singular and plural stems together, so ``reward`` and ``rewards`` both lose their bare-stem alias."""
    return stem.removesuffix("s")


def _index(roots: Iterable[Path], repo_root: Path, name_base: Path) -> dict[str, Path]:
    """Map slugged names to the scripts with a ``__main__`` guard under ``roots``, by path relative to ``name_base``,
    plus a file-stem alias where unambiguous."""
    discovered: list[tuple[str, str, Path]] = []
    for root in roots:
        script_root = repo_root / root
        if not script_root.exists():
            continue
        for script in sorted(script_root.rglob("*.py")):
            if "__main__" not in script.read_text(encoding="utf-8"):
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
    """Index ``scripts/training`` by path; paths register before stem aliases, so an ambiguous stem overwrites none."""
    return _index((TRAINING_ROOT,), root or REPO_ROOT, TRAINING_ROOT)


def tool_commands(root: Path | None = None) -> dict[str, Path]:
    """Index the tools in every ``scripts/`` subtree :data:`_NON_TOOL_DIRS` does not exclude."""
    repo_root = root or REPO_ROOT
    return _index(_tool_roots(repo_root), repo_root, _SCRIPTS_ROOT)


def build_launch_command(
    script: Path,
    config: Path | None,
    *,
    nproc: int,
    args: list[str],
    accelerate_config: Path | None = None,
    port: int | None = None,
    numa_binding: AffinityMode | None = None,
) -> list[str]:
    """Build the process argv: ``accelerate launch`` with ``accelerate_config``, ``torchrun`` when ``nproc > 1``, else
    plain Python. Concurrent multi-process launches collide on the default rendezvous port without ``port``.
    ``numa_binding`` is torchrun's ``--numa-binding``, refused for the other two launchers."""
    if numa_binding is not None and (accelerate_config is not None or nproc <= 1):
        raise typer.BadParameter(
            "--numa-binding is a torchrun option and needs a torchrun launch (--nproc > 1 without "
            "--accelerate): accelerate launch and python have no NUMA binding"
        )
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
        if numa_binding is not None:
            launcher.append(f"--numa-binding={numa_binding.value}")
    else:
        if port is not None:
            raise typer.BadParameter(
                "--port only applies to multi-process launches (--nproc > 1 or --accelerate); to hand a "
                "script its own --port, put it after a standalone `--`"
            )
        launcher = [sys.executable]
    positionals = [str(script)] if config is None else [str(script), str(config)]
    return [*launcher, *positionals, *args]


def resolve_config(config: Path, root: Path) -> Path:
    resolved = config if config.is_absolute() or config.exists() else root / config
    if not resolved.exists():
        raise typer.BadParameter(f"config not found: {config} (resolved to {resolved})")
    # Absolute: _execute chdirs to --root, so a path checked against the caller's cwd would re-resolve to another file.
    return resolved.resolve()


def _reject_option_as_positional(value: str | Path | None, name: str, usage: str) -> None:
    """Refuse a flag parsed as positional ``name``: unknown options pass through, surfacing as a missing file."""
    if value is not None and str(value).startswith("-"):
        raise click.UsageError(f"{name} must come before any flag, but got '{value}'. Usage: {usage}")


def _show(index: dict[str, Path], root: Path) -> None:
    if not index:
        typer.echo(f"(none found under {root})")
        return
    width = max(len(name) for name in index)
    for name, script in sorted(index.items()):
        typer.echo(f"{name:<{width}}  {script}")


def _lookup(index: dict[str, Path], name: str, kind: str, root: Path) -> Path:
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


def _execute(argv: list[str], root: Path, *, cwd: Path | None, dry_run: bool) -> None:
    """Exec ``argv`` with ``root`` first on ``PYTHONPATH``, since ``python script.py`` puts only the script's directory
    on ``sys.path``. ``halo launch`` passes the repo root as ``cwd`` so a config's relative paths resolve there;
    ``halo run`` passes ``None`` so a tool's path flags keep the caller's cwd."""
    if dry_run:
        typer.echo(" ".join(shlex.quote(part) for part in argv))
        return
    if cwd is not None:
        os.chdir(cwd)
    python_path = os.pathsep.join(filter(None, [str(root), os.environ.get("PYTHONPATH")]))
    os.execvpe(argv[0], argv, {**os.environ, "PYTHONPATH": python_path})


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
    numa_binding: Annotated[
        AffinityMode | None,
        typer.Option(
            "--numa-binding",
            "--numa_binding",
            help="torchrun --numa-binding: run each rank on CPUs near its GPU (torchrun launches only)",
        ),
    ] = None,
    list_methods: Annotated[bool, typer.Option("--list", help="List indexed training methods")] = False,
    dry_run: Annotated[bool, typer.Option("--dry_run", help="Print command without executing")] = False,
    root: Annotated[Path, typer.Option("--root", help="Repository root")] = REPO_ROOT,
) -> None:
    """Launch a training method without spelling out Python, torchrun/accelerate, or script paths.

    Defaults to Python (single process) or torchrun (``--nproc > 1``, required for EP/CP/TP/ETP/PP); pass
    ``--accelerate <config>`` for standard FSDP data-parallel via ``accelerate launch``.

    Config-field overrides follow the config and reach the training script as typed
    (``halo launch sft config.yaml -n 8 --learning_rate=1e-5``); a standalone ``--`` is needed only
    before a flag that collides with the launcher's own. The run executes from ``--root``, so relative
    paths in the config and its overrides resolve against the repository root, not the caller's cwd.
    """
    root = root.resolve()
    index = training_methods(root)
    if list_methods:
        _show(index, root)
        return
    _reject_option_as_positional(method, "METHOD", _LAUNCH_USAGE)
    _reject_option_as_positional(config, "CONFIG", _LAUNCH_USAGE)
    if method is None or config is None:
        raise click.UsageError("halo launch requires METHOD and CONFIG")
    script = _lookup(index, method, "method", root)
    accelerate_config = resolve_config(accelerate, root) if accelerate is not None else None
    stray = [arg for arg in ctx.args if arg.split("=", 1)[0] in _TORCHRUN_MULTINODE_FLAGS]
    if stray:
        raise typer.BadParameter(
            f"{stray} are torchrun flags, and halo launch is single-node (they would reach the training "
            f"script as unknown config fields). Launch one torchrun per node directly — "
            f"agent-docs/parallelism/launch-recipes.md."
        )
    argv = build_launch_command(
        script,
        resolve_config(config, root),
        nproc=nproc,
        args=ctx.args,
        accelerate_config=accelerate_config,
        port=port,
        numa_binding=numa_binding,
    )
    _execute(argv, root, cwd=root, dry_run=dry_run)


@app.command(context_settings=_EXTRA_ARGS)
def run(
    ctx: typer.Context,
    tool: Annotated[
        str | None, typer.Argument(help="Tool name, e.g. merge-ep-shards or inference/reward-model/score")
    ] = None,
    list_tools: Annotated[bool, typer.Option("--list", help="List indexed tools")] = False,
    dry_run: Annotated[bool, typer.Option("--dry_run", help="Print command without executing")] = False,
    root: Annotated[Path, typer.Option("--root", help="Repository root")] = REPO_ROOT,
) -> None:
    """Run a non-training tool (after-training, inference, before-training, environments,
    profiling, ...) by name, as one process: every tool is single-process, and N copies of a
    checkpoint tool would race on one output directory.

    Flags after the tool name pass through to the underlying script, except the launcher's own
    (--list, --dry_run, --root), which the CLI consumes first. Put ``--`` before a tool flag that
    collides (``halo run prepare-dataset -- --help``).

    The tool runs in the caller's cwd (unlike ``halo launch``), so a relative path flag means what it
    would mean had the script been invoked directly.
    """
    root = root.resolve()
    index = tool_commands(root)
    if list_tools:
        _show(index, root)
        return
    _reject_option_as_positional(tool, "TOOL", _RUN_USAGE)
    if tool is None:
        raise click.UsageError("halo run requires a TOOL name (use --list to see options)")
    script = _lookup(index, tool, "tool", root)
    argv = build_launch_command(script, None, nproc=1, args=ctx.args)
    _execute(argv, root, cwd=None, dry_run=dry_run)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
