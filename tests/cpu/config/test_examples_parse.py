#!/usr/bin/env python
"""Every shipped ``examples/`` YAML must parse through its own entry script's parser.

``H4ArgumentParser`` runs with ``allow_extra_keys=False``, so a key that matches no field of the
script's dataclass tuple is rejected — which makes "does it parse?" a real gate: it catches a
renamed/removed knob left behind in a config, a typo that would otherwise be silently dropped, a
value outside a ``Literal``'s choices, and any ``__post_init__`` guard the config now violates.
Each config is parsed with the *imported* entry-script module so import-time registrations the real
run depends on are in place (``optim: flash_adamw`` is only valid because importing the trainer
extends HF's ``OptimizerNames``).

Run: pytest tests/cpu/config/test_examples_parse.py
"""

import ast
import dataclasses
from pathlib import Path

import pytest
import transformers.training_args
import yaml

from src.distributed.expert_parallel.config import GIN_MAX_TOKENS_PER_RANK
from src.training.parser import H4ArgumentParser
from tests.common.utils import load_script_module

PROJECT_ROOT = Path(__file__).resolve().parents[3]
EXAMPLES_ROOT = PROJECT_ROOT / "examples"
TRAINING_ROOT = PROJECT_ROOT / "scripts" / "training"

# Which entry script owns a config, keyed by (directory prefix, file-name prefix). A directory maps
# to one script except where several methods share it (preference/, distillation/),
# which the file-name prefix disambiguates; ``test_every_example_is_mapped`` fails if a config lands
# outside every rule, so a new method cannot quietly go unchecked.
_SCRIPT_RULES = (
    ("examples/sft", "", "sft.py"),
    ("examples/classification", "", "classification.py"),
    ("examples/embedding", "", "embedding.py"),
    ("examples/reward", "", "preference/rewards.py"),
    ("examples/preference", "dpo-", "preference/dpo.py"),
    ("examples/preference", "smpo-", "preference/smpo.py"),
    ("examples/preference", "kto-", "preference/kto.py"),
    ("examples/distillation", "self-distill-", "distillation/self_distill.py"),
    ("examples/distillation", "distill-", "distillation/teacher_distill.py"),
    ("examples/grpo/offline", "", "offline_grpo.py"),
    ("examples/grpo/online", "", "online_grpo/rlvr.py"),
    ("examples/grpo/environmental", "", "environmental_grpo.py"),
)

_EXAMPLES = sorted(EXAMPLES_ROOT.rglob("*.yaml"))


def declares_cross_node_ep(config: Path) -> bool:
    """Does ``config`` declare an EP group meant to span NVLink domains over RDMA?

    ``ep_scope`` is the declaration: ``node`` (the default) confines the group to one NVLink domain,
    where dispatch is validated far above the cross-node ceiling, while ``global`` lets it span
    domains and ride Gin. Read off the raw YAML rather than the parsed config because resolving
    ``ep_scope`` through ``ParallelismConfig`` needs a live launcher topology.
    """
    raw = yaml.safe_load(config.read_text(encoding="utf-8")) or {}
    return raw.get("ep_scope") == "global" and (raw.get("expert_parallel_size") or 1) > 1


_CROSS_NODE_EP_EXAMPLES = [config for config in _EXAMPLES if declares_cross_node_ep(config)]

_module_cache: dict[str, object] = {}


def script_for(config: Path) -> str | None:
    """Entry script (relative to ``scripts/training``) that consumes ``config``, or None."""
    relative = config.relative_to(PROJECT_ROOT).as_posix()
    matches = [
        (dir_prefix, script)
        for dir_prefix, name_prefix, script in _SCRIPT_RULES
        if relative.startswith(dir_prefix) and config.name.startswith(name_prefix)
    ]
    if not matches:
        return None
    return max(matches, key=lambda match: len(match[0]))[1]


def _script_module(script: str):
    """Import an entry script by path (``scripts/training`` is not a package root)."""
    if script not in _module_cache:
        name = "halo_examples_" + script.replace("/", "_").removesuffix(".py")
        _module_cache[script] = load_script_module(f"scripts/training/{script}", name, register=True)
    return _module_cache[script]


def parsed_field(parsed: tuple, name: str):
    """Value of ``name`` from whichever dataclass of a parsed config tuple declares it."""
    for config_object in parsed:
        if hasattr(config_object, name):
            return getattr(config_object, name)
    raise AssertionError(f"no parsed config object declares {name!r}")


def parser_for(script: str) -> H4ArgumentParser:
    """The parser the script builds, read off its own ``H4ArgumentParser((...))`` call."""
    module = _script_module(script)
    tree = ast.parse((TRAINING_ROOT / script).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "H4ArgumentParser":
            names = [element.id for element in node.args[0].elts]
            return H4ArgumentParser(tuple(getattr(module, name) for name in names))
    raise AssertionError(f"{script} builds no H4ArgumentParser")


@pytest.fixture(autouse=True)
def _bf16_capable(monkeypatch):
    """The CPU test container exposes no GPU, and ``bf16`` (a toolkit default) is checked against
    one in ``TrainingArguments.__post_init__``. Report a bf16-capable device so the configs are
    validated as they are on a training host instead of all failing that one check."""
    monkeypatch.setattr(transformers.training_args, "is_torch_bf16_gpu_available", lambda: True)


def test_examples_tree_is_not_empty():
    """A glob that stops matching would make every parametrized case vanish silently."""
    assert len(_EXAMPLES) > 50, f"expected the shipped examples tree, found {len(_EXAMPLES)} configs"


def test_every_example_is_mapped():
    unmapped = [c.relative_to(PROJECT_ROOT).as_posix() for c in _EXAMPLES if script_for(c) is None]
    assert not unmapped, (
        "these configs match no entry-script rule, so nothing parse-checks them; add a rule to "
        "_SCRIPT_RULES:\n  " + "\n  ".join(unmapped)
    )


@pytest.mark.parametrize("config", _EXAMPLES, ids=lambda c: c.relative_to(EXAMPLES_ROOT).as_posix())
def test_example_config_parses(config):
    """Parsing is the contract: an unknown key, a stale knob, or an invalid value raises here."""
    parser_for(script_for(config)).parse_yaml_file(str(config))


@pytest.mark.parametrize("config", _EXAMPLES, ids=lambda c: c.relative_to(EXAMPLES_ROOT).as_posix())
def test_example_output_dir_is_repo_relative(config):
    """A shipped config must not hardcode a host path.

    ``/mnt`` is not guaranteed large (on some hosts it shares the small root device), and only the
    repo-relative spelling is what ``.gitignore`` and ``make clean`` know about; an operator
    redirects with ``--output_dir=`` or by launching from the scratch volume.
    """
    output_dir = yaml.safe_load(config.read_text(encoding="utf-8"))["output_dir"]
    assert not output_dir.startswith("/"), (
        f"{config.name}: output_dir={output_dir!r} hardcodes an absolute host path — ship checkpoints/<run>"
    )


def test_a_cross_node_ep_example_exists():
    """The ceiling check is parametrized over this list — an empty one would assert nothing."""
    assert _CROSS_NODE_EP_EXAMPLES, (
        "no shipped example declares ep_scope: global, so the cross-node dispatch ceiling below is "
        "checked against nothing"
    )


@pytest.mark.parametrize("config", _CROSS_NODE_EP_EXAMPLES, ids=lambda c: c.relative_to(EXAMPLES_ROOT).as_posix())
def test_cross_node_ep_example_fits_the_gin_dispatch_ceiling(config):
    """A cross-node EP example must not ship a token budget its first MoE dispatch would refuse.

    Cross-node dispatch wedges rather than errors above ``HALO_DEEPEP_GIN_MAX_TOKENS_PER_RANK``
    tokens/rank, so the dispatcher rejects a buffer sized past it. ``per_device_train_batch_size ×
    max_length`` is the most one MoE forward can present, and packing makes it the usual case — an
    over-budget example dies on step 1, after every rank has loaded the model.
    """
    parsed = parser_for(script_for(config)).parse_yaml_file(str(config))
    max_length = parsed_field(parsed, "max_length")
    assert max_length, (
        f"{config.name} declares cross-node EP but pins no max_length, leaving its per-rank token "
        f"budget unknown until the model loads; set it at or below {GIN_MAX_TOKENS_PER_RANK}"
    )
    tokens_per_rank = parsed_field(parsed, "per_device_train_batch_size") * max_length
    assert tokens_per_rank <= GIN_MAX_TOKENS_PER_RANK, (
        f"{config.name} presents {tokens_per_rank} tokens/rank per MoE forward "
        f"(per_device_train_batch_size × max_length), over the validated cross-node ceiling "
        f"{GIN_MAX_TOKENS_PER_RANK}. Lower either, or move the example to node-local EP "
        f"(ep_scope: node), which dispatches over NVLink and is not bound by it."
    )


def test_an_unknown_key_is_rejected(tmp_path):
    """Pins the property the parametrized cases rely on — without it they assert nothing.

    Goes through ``parse_yaml_file`` (not ``parse_dict``) so the rename/deprecation stage, which
    could plausibly swallow an unrecognized key, is covered too."""
    parser = parser_for("sft.py")
    declared = {f.name for t in parser.dataclass_types for f in dataclasses.fields(t)}
    assert "bogus_typo_key" not in declared
    config = tmp_path / "typo.yaml"
    config.write_text("model_name_or_path: gpt2\noutput_dir: /tmp/x\nbogus_typo_key: 3\n")
    with pytest.raises(ValueError, match="bogus_typo_key"):
        parser.parse_yaml_file(str(config))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
