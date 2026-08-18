#!/usr/bin/env python
"""One flag surface across the standalone checkpoint tools: ``--trust_remote_code``,
``--max_shard_size``, ``--model_id``/``--revision``, and one source/destination spelling.

These tools are used interchangeably on one artifact — patch the vocabulary, train, merge the
adapter, convert to bf16, reset the sinks. When each spells a flag its own way, the same
remote-code checkpoint (Bailing/Ling, Laguna, a gpt-oss derivative) loads through one tool and is
refused by the next, and the failure surfaces as an unrelated ``ValueError`` from transformers deep
inside a multi-hour conversion. The shared ``add_trust_remote_code_arg`` helper exists so there is
exactly one flag spelling, and one default per *input source*: a local checkpoint or adapter the
operator already produced defaults on (the remote-code families do not load without it), a
Hub-capable ``--model_id`` source defaults off (a freshly downloaded third-party repo must not
execute its own code because a tool was pointed at it). These assertions are what keeps a tool from
drifting off that — a ``store_true`` that silently defaults OFF a local-source tool, a tool that
never exposes the opt-out, or a tool that parses the flag and then hardcodes
``trust_remote_code=True`` at the load, which reads identically from ``--help``.

The same reasoning converged the source/destination flags on ``--input_dir`` / ``--output_dir``
(``--model_id`` where the source may be a Hub repo); retired spellings are removed, not aliased. It
converges the shard cap and the Hub source the same way: each is added by one helper
(``add_max_shard_size_arg`` / ``add_hub_source_args``), and the assertions below read the sentence
that helper writes out of ``--help`` — a re-typed flag lands off it, quietly shipping a different
default or promising a pin the tool never threads. Which tools those two assertions cover is swept
off the tree and read off each parser, so a new tool is covered by existing.

Parser-level for the surface — no model is loaded, so this stays a fast CPU test that reads the exact
surface a user gets from ``--help`` — plus one load-site check per default, where the tool runs as far
as its first checkpoint load with that load stubbed out, and one subprocess check that the module
those spellings live in is reachable to a bare interpreter at all.

Run: ``pytest -m cpu tests/cpu/checkpoint/test_cli_trust_remote_code_policy.py``
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts._common import HUB_SOURCE_HELP, add_hub_source_args, add_max_shard_size_arg
from src.checkpoint.format import DEFAULT_MAX_SHARD_SIZE
from tests.common.utils import REPO_ROOT, load_script_module

# (tool, the default its input source earns). Every standalone tool that executes a checkpoint's own
# modeling/config/tokenizer code is here, including convert_deepseek_v4_bf16 and reattach_vision_tower
# (both reach from_pretrained/AutoConfig, which imports a remote config module). The three whose source
# is a Hub-capable --model_id are opt-in; the rest read a local checkpoint/adapter, or the run's
# tokenizer, and default on. Tools that only stream safetensors (shard merges, unfuse_moe_experts,
# quantize_to_lowp, convert_mistral4_bf16, convert_glm5_bf16) load no model code and expose no flag.
_TOOLS = (
    ("scripts/after_training/convert_to_bf16.py", True),
    ("scripts/after_training/merge_models.py", True),
    ("scripts/after_training/merge_peft_adapters.py", True),
    ("scripts/after_training/reattach_vision_tower.py", False),
    ("scripts/after_training/reset_sinks.py", True),
    ("scripts/before_training/convert_deepseek_v4_bf16.py", False),
    ("scripts/before_training/patch_vocab.py", False),
    ("scripts/before_training/prepare_dataset.py", True),
)

# Hub-source tools that thread no revision, so they must not advertise the pin. The helper's own
# default is ``revision=True``: a tool is required to thread one until it is listed here.
_NO_REVISION_TOOLS = frozenset({"scripts/before_training/patch_vocab.py"})


def _empty_source_dir(tmp_path: Path) -> str:
    """A source directory that carries no weights: every gate before the load passes it through."""
    source = tmp_path / "source"
    source.mkdir()
    return str(source)


def _minimal_adapter_dir(tmp_path: Path) -> str:
    """A PEFT adapter directory that reads far enough for the merge to reach its first load."""
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text(
        json.dumps(
            {
                "peft_type": "LORA",
                "task_type": "CAUSAL_LM",
                "base_model_name_or_path": "acme/base-model",
                "r": 8,
                "lora_alpha": 16,
                "target_modules": ["q_proj"],
            }
        )
    )
    return str(adapter)


# (tool, its source flag, a builder for the source it is pointed at, the module attribute its FIRST
# checkpoint load goes through). One row per default: a tool that parses the flag and then hardcodes
# trust_remote_code at the load has the same --help as one that threads it, so the surface tests
# above cannot tell them apart. The load is stubbed out, so no weights are read.
_LOAD_SITES = (
    ("scripts/before_training/patch_vocab.py", "--model_id", _empty_source_dir, "load_processing_class"),
    (
        "scripts/before_training/convert_deepseek_v4_bf16.py",
        "--model_id",
        _empty_source_dir,
        "source_config",
    ),
    (
        "scripts/after_training/merge_peft_adapters.py",
        "--adapter_dir",
        _minimal_adapter_dir,
        "merge_adapter_into_base",
    ),
)


class _ParserBuilt(Exception):
    """Carries the parser out of the tool, before it can consume ``sys.argv``."""

    def __init__(self, parser: argparse.ArgumentParser):
        super().__init__("parser captured")
        self.parser = parser


def _tool_parser(relative_path: str) -> argparse.ArgumentParser:
    """The parser a ``scripts/`` CLI builds, captured at its own ``parse_args`` call.

    The tools split between a module-level ``parse_args()`` and a ``main()`` that builds the parser
    inline, so the entry point is whichever one the module exposes; intercepting
    ``ArgumentParser.parse_args`` catches both without running any of the work behind them.
    """
    module = load_script_module(relative_path, f"{Path(relative_path).stem}_trc")
    entry = getattr(module, "parse_args", None) or module.main

    def _capture(self, *args, **kwargs):
        raise _ParserBuilt(self)

    with patch.object(argparse.ArgumentParser, "parse_args", _capture), patch.object(sys, "argv", [relative_path]):
        try:
            entry()
        except _ParserBuilt as built:
            return built.parser
    raise AssertionError(f"{relative_path} built no parser — its flag surface cannot be checked")


def _tool_paths() -> tuple[str, ...]:
    """Every standalone checkpoint tool, swept off the tree rather than listed.

    A tool added to either subtree joins the rosters below by existing, so it cannot ship a re-typed
    flag — or none at all — under a hand-maintained list that nobody remembered to extend.
    """
    return tuple(
        sorted(
            f"scripts/{subtree}/{path.name}"
            for subtree in ("before_training", "after_training")
            for path in (REPO_ROOT / "scripts" / subtree).glob("*.py")
            if not path.name.startswith("_")
        )
    )


def _tool_options(relative_path: str) -> set[str]:
    """Every flag spelling the tool's parser carries."""
    return {spelling for action in _tool_parser(relative_path)._actions for spelling in action.option_strings}


# Membership off the PARSER, not off an import sweep: a tool that re-types a shared flag with its own
# add_argument still lands in the roster, and the assertions on its default and its wording catch it.
_MAX_SHARD_SIZE_TOOLS = tuple(path for path in _tool_paths() if "--max_shard_size" in _tool_options(path))
# (tool, whether it also threads a --revision) for every tool whose source may be a Hub repo, so its
# source flag is --model_id rather than --input_dir.
_HUB_SOURCE_TOOLS = tuple(
    (path, path not in _NO_REVISION_TOOLS) for path in _tool_paths() if "--model_id" in _tool_options(path)
)


def test_the_derived_rosters_cover_the_tool_tree():
    """Anti-vacuity: a sweep that stopped finding tools would leave the parametrizations below empty
    and passing. The trust roster is hand-declared (a per-source POLICY, not a surface) and must name
    only tools that still exist."""
    paths = _tool_paths()

    assert len(paths) >= 10, sorted(paths)
    assert set(dict(_TOOLS)) <= set(paths), (
        f"the trust roster names tools that are gone: {sorted(set(dict(_TOOLS)) - set(paths))}"
    )
    assert len(_MAX_SHARD_SIZE_TOOLS) >= 8, _MAX_SHARD_SIZE_TOOLS
    assert len(_HUB_SOURCE_TOOLS) >= 4, _HUB_SOURCE_TOOLS
    assert set(paths) >= _NO_REVISION_TOOLS, sorted(_NO_REVISION_TOOLS - set(paths))


def test_the_shared_flag_module_is_reachable_to_a_bare_interpreter(tmp_path: Path):
    """``scripts._common`` must import for a plain ``python scripts/<subtree>/<tool>.py``.

    Every tool below takes its flag spellings from a ``scripts.*`` module, so the repo root has to
    be on a bare interpreter's ``sys.path`` — the editable install puts it there, and the images
    also set ``PYTHONPATH=/workspace``. Under pytest the root conftest has already inserted it,
    which is exactly why the parser tests below cannot catch a packaging change: a subprocess run
    from outside the repo with ``PYTHONPATH`` cleared is what reproduces an operator's invocation,
    where a missing repo root is ``ModuleNotFoundError: scripts`` before a single flag is parsed.
    """
    env = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
    result = subprocess.run(
        [sys.executable, "-c", "import scripts._common"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, (
        f"`import scripts._common` fails for a bare interpreter outside the repo: "
        f"{result.stderr.strip().splitlines()[-1:] or ['(no output)']}. The checkpoint tools take "
        f"their flags from that module, so every one of them would die on startup."
    )


@pytest.mark.parametrize(("relative_path", "expected_default"), _TOOLS)
def test_trust_remote_code_default_follows_the_input_source(relative_path: str, expected_default: bool):
    parser = _tool_parser(relative_path)
    help_text = parser.format_help()

    assert parser.get_default("trust_remote_code") is expected_default, (
        f"{relative_path} defaults --trust_remote_code to {parser.get_default('trust_remote_code')!r}, "
        f"expected {expected_default!r} for its input source; pass the default through "
        f"add_trust_remote_code_arg(parser, default=...) rather than a per-tool add_argument"
    )
    for spelling in ("--trust_remote_code", "--no-trust_remote_code"):
        assert spelling in help_text, (
            f"{relative_path} exposes no {spelling}, so the operator cannot override the default for "
            f"a source whose trust does not match it"
        )


def _flat_help(parser: argparse.ArgumentParser) -> str:
    """``--help`` with its line wrapping flattened, so a helper's sentence matches whole."""
    return " ".join(parser.format_help().split())


def _canonical_flag_help(add_arg: Callable[[argparse.ArgumentParser], object], flag: str) -> str:
    """The description ``--help`` renders for ``flag`` on a parser carrying only what ``add_arg`` adds.

    Read off the rendered help rather than the source string, so it is the wording a user sees; the
    argparse scaffolding is cut at the flag's own ``--flag METAVAR`` heading, which is the last thing
    before its description on a single-flag (or trailing-flag) parser.
    """
    parser = argparse.ArgumentParser(add_help=False)
    add_arg(parser)
    heading = f"{flag} {flag.lstrip('-').upper()} "
    return " ".join(parser.format_help().split()).split(heading)[-1].strip()


@pytest.mark.parametrize("relative_path", _MAX_SHARD_SIZE_TOOLS)
def test_the_shard_cap_is_the_shared_flag(relative_path: str):
    """One default and one description: a re-typed ``--max_shard_size`` is how two tools in the same
    chain end up capping their shards differently, and how a tool's default drifts off
    DEFAULT_MAX_SHARD_SIZE without anything failing."""
    parser = _tool_parser(relative_path)

    assert parser.get_default("max_shard_size") == DEFAULT_MAX_SHARD_SIZE, (
        f"{relative_path} defaults --max_shard_size to {parser.get_default('max_shard_size')!r}, not the "
        f"toolkit-wide {DEFAULT_MAX_SHARD_SIZE!r}; add it with add_max_shard_size_arg(parser)"
    )
    canonical = _canonical_flag_help(add_max_shard_size_arg, "--max_shard_size")
    assert canonical in _flat_help(parser), (
        f"{relative_path} re-types --max_shard_size with its own help text instead of calling "
        f"add_max_shard_size_arg(parser) — expected it to carry {canonical!r}"
    )


@pytest.mark.parametrize(("relative_path", "threads_revision"), _HUB_SOURCE_TOOLS)
def test_the_hub_source_flags_are_the_shared_ones(relative_path: str, threads_revision: bool):
    """``--model_id`` must describe what ``resolve_checkpoint_source`` actually accepts, and
    ``--revision`` must appear only where the tool threads one — an advertised pin a tool ignores
    silently converts a checkpoint from whatever the Hub's default branch holds today."""
    help_text = _flat_help(_tool_parser(relative_path))

    assert HUB_SOURCE_HELP in help_text, (
        f"{relative_path} re-types --model_id with its own description instead of calling "
        f"add_hub_source_args(parser, ...) — expected it to carry {HUB_SOURCE_HELP!r}"
    )
    revision_help = _canonical_flag_help(lambda p: add_hub_source_args(p, source="x"), "--revision")
    if threads_revision:
        assert revision_help in help_text, f"{relative_path} re-types --revision: expected {revision_help!r}"
    else:
        assert "--revision" not in help_text, (
            f"{relative_path} advertises --revision but threads none — it would promise a pin the conversion ignores"
        )


@pytest.mark.parametrize(("relative_path", "source_flag", "build_source", "load_site"), _LOAD_SITES)
def test_the_parsed_trust_remote_code_reaches_the_first_load(
    relative_path: str,
    source_flag: str,
    build_source: Callable[[Path], str],
    load_site: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    module = load_script_module(relative_path, f"{Path(relative_path).stem}_wiring")
    assert hasattr(module, load_site), (
        f"{relative_path} no longer loads through {load_site} — the row is stale, repoint it at the "
        f"call that now reads the flag"
    )
    expected_default = dict(_TOOLS)[relative_path]
    source, output_dir = build_source(tmp_path), str(tmp_path / "out")

    class _LoadReached(Exception):
        """Carries control out of the tool at its first checkpoint load."""

    seen: dict[str, object] = {}

    def _capture(*args, **kwargs):
        seen.update(kwargs)
        raise _LoadReached

    monkeypatch.setattr(module, load_site, _capture)

    for extra, expected in (
        ([], expected_default),
        (["--trust_remote_code"], True),
        (["--no-trust_remote_code"], False),
    ):
        seen.clear()
        monkeypatch.setattr(sys, "argv", [relative_path, source_flag, source, "--output_dir", output_dir, *extra])
        with pytest.raises(_LoadReached):
            module.main()
        assert seen.get("trust_remote_code") is expected, (
            f"{relative_path} {' '.join(extra) or '(default)'} reached {load_site} with "
            f"trust_remote_code={seen.get('trust_remote_code')!r}, expected {expected!r}: the flag is "
            f"parsed but not threaded into the load, so --help promises a policy the tool ignores"
        )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
