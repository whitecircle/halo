#!/usr/bin/env python
"""One flag surface across the standalone checkpoint tools. A local source defaults ``--trust_remote_code`` on, a Hub
``--model_id`` off, and an adapter follows the base it names, since a third-party repo must not run its code because a
tool was pointed at it."""

from __future__ import annotations

import argparse
import inspect
import json
import os
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

import pytest
import torch
from datasets import DatasetDict
from safetensors.torch import save_file

from halo.checkpoint.adapters import EXPERT_LORA_PEFT_TYPE
from halo.checkpoint.format import ADAPTER_SAFETENSORS_FILE, DEFAULT_MAX_SHARD_SIZE
from halo.checkpoint.tools import adapter_merge, bf16_conversion, deepseek_v4_dequant, vocab_patch
from halo.data.preprocessed import prepare_dataset_tool, tokenization
from scripts.cli_arguments import (
    HUB_SOURCE_HELP,
    add_device_map_arg,
    add_dtype_arg,
    add_hub_source_args,
    add_max_shard_size_arg,
)
from tests.common.repo import REPO_ROOT, load_script_module

# (tool, the default its input source earns, None where the adapter's base decides it). Tools that only stream
# safetensors load no model code and expose no flag.
_TOOLS = (
    ("scripts/after_training/convert_to_bf16.py", None),
    ("scripts/after_training/merge_models.py", True),
    ("scripts/after_training/merge_peft_adapters.py", None),
    ("scripts/after_training/reattach_vision_tower.py", False),
    ("scripts/before_training/convert_deepseek_v4_bf16.py", False),
    ("scripts/before_training/deduplicate_dataset.py", False),
    ("scripts/before_training/patch_vocab.py", False),
    ("scripts/before_training/prepare_dataset.py", False),
    ("scripts/inference/reward_model/score.py", False),
    ("scripts/inference/reward_model/rejection_sample.py", False),
)

# The helper's default is ``revision=True``: a tool must thread one until it is listed here.
_NO_REVISION_TOOLS = frozenset(
    {"scripts/before_training/deduplicate_dataset.py", "scripts/before_training/patch_vocab.py"}
)


def _empty_source_dir(tmp_path: Path) -> str:
    """Every gate before the load passes it through."""
    source = tmp_path / "source"
    source.mkdir()
    return str(source)


def _adapter_dir(tmp_path: Path, base_model: str) -> str:
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text(
        json.dumps(
            {
                "peft_type": "LORA",
                "task_type": "CAUSAL_LM",
                "base_model_name_or_path": base_model,
                "r": 8,
                "lora_alpha": 16,
                "target_modules": ["q_proj"],
            }
        )
    )
    return str(adapter)


def _adapter_on_a_hub_base(tmp_path: Path) -> str:
    return _adapter_dir(tmp_path, "acme/base-model")


def _adapter_on_a_local_base(tmp_path: Path) -> str:
    return _adapter_dir(tmp_path, _empty_source_dir(tmp_path))


# (tool, source flag, source builder, driver module and attribute of its first load, the default that load sees). A
# tool that hardcodes trust_remote_code at the load has the same --help as one threading it.
_LOAD_SITES = (
    (
        "scripts/before_training/patch_vocab.py",
        "--model_id",
        _empty_source_dir,
        vocab_patch,
        "load_processing_class",
        False,
    ),
    (
        "scripts/before_training/convert_deepseek_v4_bf16.py",
        "--model_id",
        _empty_source_dir,
        deepseek_v4_dequant,
        "source_config",
        False,
    ),
    (
        "scripts/after_training/merge_peft_adapters.py",
        "--adapter_dir",
        _adapter_on_a_hub_base,
        adapter_merge,
        "merge_adapter_into_base",
        False,
    ),
    (
        "scripts/after_training/merge_peft_adapters.py",
        "--adapter_dir",
        _adapter_on_a_local_base,
        adapter_merge,
        "merge_adapter_into_base",
        True,
    ),
    (
        "scripts/after_training/convert_to_bf16.py",
        "--input_dir",
        _empty_source_dir,
        bf16_conversion,
        "load_processing_class",
        True,
    ),
)


class _ParserBuilt(Exception):
    """Carries the parser out of the tool, before it can consume ``sys.argv``."""

    def __init__(self, parser: argparse.ArgumentParser):
        super().__init__("parser captured")
        self.parser = parser


def _tool_parser(relative_path: str) -> argparse.ArgumentParser:
    """Intercepting ``ArgumentParser.parse_args`` catches module-level and inline parsers without running the work
    behind them."""
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
    """A tool added to either subtree joins the rosters by existing."""
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


# Membership off the parser: a tool re-typing a shared flag still lands in the roster.
_MAX_SHARD_SIZE_TOOLS = tuple(path for path in _tool_paths() if "--max_shard_size" in _tool_options(path))
# (tool, whether it also threads a --revision).
_HUB_SOURCE_TOOLS = tuple(
    (path, path not in _NO_REVISION_TOOLS) for path in _tool_paths() if "--model_id" in _tool_options(path)
)
# Any spelling, so a re-spelled one (``--torch_dtype``) is swept too.
_DTYPE_TOOLS = tuple(path for path in _tool_paths() if any("dtype" in option for option in _tool_options(path)))
_DEVICE_MAP_TOOLS = tuple(path for path in _tool_paths() if "--device_map" in _tool_options(path))


def test_the_derived_rosters_cover_the_tool_tree():
    """Anti-vacuity: a sweep that stopped finding tools would leave the parametrizations empty and passing."""
    paths = _tool_paths()

    assert len(paths) >= 10, sorted(paths)
    on_disk = {str(path.relative_to(REPO_ROOT)) for path in (REPO_ROOT / "scripts").rglob("*.py")}
    assert set(dict(_TOOLS)) <= on_disk, (
        f"the trust roster names tools that are gone: {sorted(set(dict(_TOOLS)) - on_disk)}"
    )
    assert len(_MAX_SHARD_SIZE_TOOLS) >= 8, _MAX_SHARD_SIZE_TOOLS
    assert len(_HUB_SOURCE_TOOLS) >= 4, _HUB_SOURCE_TOOLS
    assert len(_DTYPE_TOOLS) >= 3, _DTYPE_TOOLS
    assert len(_DEVICE_MAP_TOOLS) >= 3, _DEVICE_MAP_TOOLS
    assert set(paths) >= _NO_REVISION_TOOLS, sorted(_NO_REVISION_TOOLS - set(paths))


def test_the_shared_flag_module_is_reachable_to_a_bare_interpreter(tmp_path: Path):
    """The root conftest hides a packaging change, so a subprocess outside the repo with ``PYTHONPATH`` cleared
    reproduces an operator's invocation."""
    env = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
    result = subprocess.run(
        [sys.executable, "-c", "import scripts.cli_arguments"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, (
        f"`import scripts.cli_arguments` fails for a bare interpreter outside the repo: "
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
    """Line wrapping flattened, so a helper's sentence matches whole."""
    return " ".join(parser.format_help().split())


def _canonical_flag_help(add_arg: Callable[[argparse.ArgumentParser], object], flag: str) -> str:
    """Read off the rendered help, so it is the wording a user sees."""
    parser = argparse.ArgumentParser(add_help=False)
    add_arg(parser)
    heading = f"{flag} {flag.lstrip('-').upper()} "
    return " ".join(parser.format_help().split()).split(heading)[-1].strip()


@pytest.mark.parametrize("relative_path", _MAX_SHARD_SIZE_TOOLS)
def test_the_shard_cap_is_the_shared_flag(relative_path: str):
    """A re-typed flag lets a tool's default drift off DEFAULT_MAX_SHARD_SIZE without anything failing."""
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
    """An advertised ``--revision`` a tool ignores silently converts whatever the Hub's default branch holds today."""
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


def _declared_action(parser: argparse.ArgumentParser, flag: str) -> argparse.Action:
    return next(action for action in parser._actions if flag in action.option_strings)


def _shared_action(add_arg: Callable[[argparse.ArgumentParser], object], flag: str) -> argparse.Action:
    """On an otherwise empty parser."""
    parser = argparse.ArgumentParser(add_help=False)
    add_arg(parser)
    return _declared_action(parser, flag)


@pytest.mark.parametrize(
    ("relative_path", "flag", "add_arg"),
    [(path, "--dtype", add_dtype_arg) for path in _DTYPE_TOOLS]
    + [(path, "--device_map", add_device_map_arg) for path in _DEVICE_MAP_TOOLS],
)
def test_the_load_flags_are_the_shared_ones(relative_path: str, flag: str, add_arg: Callable):
    """A re-typed one drifts on each (``--torch_dtype``, a ``bf16`` default beside a ``bfloat16`` one)."""
    parser = _tool_parser(relative_path)
    name = flag.lstrip("-")
    spellings = sorted(option for action in parser._actions for option in action.option_strings if name in option)
    assert spellings == [flag], f"{relative_path} spells its {name} flag {spellings}; add it with {add_arg.__name__}"

    declared, shared = _declared_action(parser, flag), _shared_action(add_arg, flag)
    for field in ("default", "choices", "help"):
        assert getattr(declared, field) == getattr(shared, field), (
            f"{relative_path} re-types {flag} ({field}={getattr(declared, field)!r}, shared "
            f"{getattr(shared, field)!r}); add it with {add_arg.__name__}(parser)"
        )


@pytest.mark.parametrize(
    ("relative_path", "source_flag", "build_source", "driver", "load_site", "expected_default"), _LOAD_SITES
)
def test_the_parsed_trust_remote_code_reaches_the_first_load(
    relative_path: str,
    source_flag: str,
    build_source: Callable[[Path], str],
    driver: ModuleType,
    load_site: str,
    expected_default: bool,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    module = load_script_module(relative_path, f"{Path(relative_path).stem}_wiring")
    assert hasattr(driver, load_site), (
        f"{driver.__name__} no longer loads through {load_site} — the row is stale, repoint it at the "
        f"call that now reads the flag"
    )
    source, output_dir = build_source(tmp_path), str(tmp_path / "out")

    class _LoadReached(Exception):
        """Carries control out of the tool at its first checkpoint load."""

    seen: dict[str, object] = {}

    def _capture(*args, **kwargs):
        seen.update(kwargs)
        raise _LoadReached

    monkeypatch.setattr(driver, load_site, _capture)

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


class _LoadReached(Exception):
    """Carries control out of a tool once the loads under test ran."""


@pytest.mark.parametrize(
    ("build_adapter", "extra", "expected"),
    [
        (_adapter_on_a_hub_base, [], False),
        (_adapter_on_a_hub_base, ["--trust_remote_code"], True),
        (_adapter_on_a_local_base, [], True),
        (_adapter_on_a_local_base, ["--no-trust_remote_code"], False),
    ],
    ids=["hub-default", "hub-explicit", "local-default", "local-refused"],
)
def test_every_load_of_a_merge_runs_under_the_trust_its_base_earns(
    build_adapter: Callable[[Path], str],
    extra: list[str],
    expected: bool,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """The base's own model and processor, and the adapter's processor, whose ``auto_map`` may name the same repo."""
    module = load_script_module("scripts/after_training/merge_peft_adapters.py", "merge_peft_adapters_loads")
    adapter = build_adapter(tmp_path)
    seen: dict[str, object] = {}

    def _load_base(path, task, **kwargs):
        seen["base model"] = kwargs["trust_remote_code"]
        return torch.nn.Linear(1, 1)

    def _load_processor(path, *, trust_remote_code):
        seen["base processor" if path != adapter else "adapter processor"] = trust_remote_code
        if path != adapter:
            raise _LoadReached

    monkeypatch.setattr(adapter_merge, "adapter_module_paths", lambda _adapter: set())
    monkeypatch.setattr(adapter_merge, "load_task_model", _load_base)
    monkeypatch.setattr(adapter_merge, "load_processing_class", _load_processor)
    argv = ["merge", "--adapter_dir", adapter, "--output_dir", str(tmp_path / "out"), *extra]
    monkeypatch.setattr(sys, "argv", argv)

    with pytest.raises(_LoadReached):
        module.main()

    assert seen == {"base model": expected, "adapter processor": expected, "base processor": expected}


@pytest.mark.parametrize(
    ("extra", "expected"), [([], False), (["--trust_remote_code"], True)], ids=["default", "explicit"]
)
def test_a_converted_adapters_inference_check_holds_a_hub_base_to_the_same_rule(
    extra: list[str], expected: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """``--peft`` loads no base to convert, so ``--check_inference`` is where the adapter's repo would first run."""
    module = load_script_module("scripts/after_training/convert_to_bf16.py", "convert_to_bf16_peft_check")
    adapter = _adapter_on_a_hub_base(tmp_path)
    save_file({"lora_A.weight": torch.zeros(1)}, os.path.join(adapter, ADAPTER_SAFETENSORS_FILE))
    seen: dict[str, object] = {}

    def _load_processor(path, *, trust_remote_code):
        is_base = path == "acme/base-model"
        seen["base processor" if is_base else "adapter processor"] = trust_remote_code
        return object() if is_base else None

    def _load_base(path, task, **kwargs):
        seen["base model"] = kwargs["trust_remote_code"]
        raise _LoadReached

    monkeypatch.setattr(adapter_merge, "adapter_module_paths", lambda _adapter: set())
    monkeypatch.setattr(bf16_conversion, "load_processing_class", _load_processor)
    monkeypatch.setattr(bf16_conversion, "load_task_model", _load_base)
    argv = ["convert", "--input_dir", adapter, "--output_dir", str(tmp_path / "out"), "--peft", "--check_inference"]
    monkeypatch.setattr(sys, "argv", [*argv, *extra])

    with pytest.raises(_LoadReached):
        module.main()

    assert seen == {"adapter processor": expected, "base processor": expected, "base model": expected}


@pytest.mark.parametrize("extra", [[], ["--trust_remote_code"]], ids=["default", "explicit"])
def test_a_full_conversion_refuses_a_hub_repo_before_any_load(
    extra: list[str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Its default is on because the operator made the checkpoint, which a Hub repo does not show."""
    module = load_script_module("scripts/after_training/convert_to_bf16.py", "convert_to_bf16_hub_refusal")

    def _reached(*_args, **_kwargs):
        pytest.fail("a load ran on a refused source")

    monkeypatch.setattr(bf16_conversion, "load_processing_class", _reached)
    monkeypatch.setattr(bf16_conversion, "load_model", _reached)
    monkeypatch.setattr(
        sys, "argv", ["convert", "--input_dir", "acme/base-model", "--output_dir", str(tmp_path / "out"), *extra]
    )

    with pytest.raises(
        ValueError, match=r"not a local directory\. Download the repo first \(hf download acme/base-model "
    ):
        module.main()


def test_an_expert_lora_adapter_still_reaches_its_own_refusal_with_the_flag_unset(tmp_path: Path):
    """Resolving the default reads the adapter first, and PEFT's config class raises a bare KeyError on this type."""
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text(
        json.dumps({"peft_type": EXPERT_LORA_PEFT_TYPE, "base_model_name_or_path": "acme/base-model"})
    )

    with pytest.raises(ValueError, match="merge_expert_lora_on_save"):
        adapter_merge.merge_peft_adapter(str(adapter), str(tmp_path / "out"))


@pytest.mark.parametrize(
    "loader",
    [adapter_merge.merge_adapter_into_base, adapter_merge.load_task_base_model, bf16_conversion.run_test_inference],
    ids=lambda loader: loader.__name__,
)
def test_below_the_clis_an_unstated_trust_stays_off(loader: Callable):
    """Only the tools resolve a default off the source, so a direct caller of these must state the trust it gives."""
    assert inspect.signature(loader).parameters["trust_remote_code"].default is False


def test_prepare_dataset_threads_the_flag_into_the_label_bake_config_read(tmp_path: Path, monkeypatch):
    """The completion-only bake reads the model config for its eos set, so a flag stopping at the tokenizer fails
    here."""
    module = load_script_module("scripts/before_training/prepare_dataset.py", "prepare_dataset_config_read")
    seen: list[object] = []

    class _ConfigRead(Exception):
        """Carries control out of the bake at its config read."""

    def _record(*_args, **kwargs):
        seen.append(kwargs.get("trust_remote_code"))
        raise _ConfigRead

    def _bake(*, dataset, tokenizer_or_processor, config, output_dir):
        tokenization.resolve_label_eos_token_ids(config, tokenizer_or_processor)

    monkeypatch.setattr(tokenization.AutoConfig, "from_pretrained", _record)
    monkeypatch.setattr(prepare_dataset_tool, "load_tokenizer", lambda args: object())
    monkeypatch.setattr(prepare_dataset_tool, "load_input_dataset", lambda args: DatasetDict())
    monkeypatch.setattr(prepare_dataset_tool, "preprocess_dataset", _bake)
    argv = ["--input", "org/raw", "--output", str(tmp_path / "out"), "--model_name_or_path", "org/model"]
    argv += ["--assistant_message_template", "<|im_start|>assistant\n"]

    for extra, expected in (([], False), (["--trust_remote_code"], True), (["--no-trust_remote_code"], False)):
        seen.clear()
        monkeypatch.setattr(sys, "argv", ["prepare_dataset.py", *argv, *extra])
        with pytest.raises(_ConfigRead):
            module.main()
        assert seen == [expected], f"{extra or '(default)'}: the config read ran with trust_remote_code={seen}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
