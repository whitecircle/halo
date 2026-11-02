#!/usr/bin/env python
"""Every shipped ``examples/`` YAML must parse through its own entry script's parser.

``parse_script_args`` parses with ``allow_extra_keys=False``, so a key that matches no field of the
script's dataclass tuple is rejected — which makes "does it parse?" a real gate: it catches a
renamed/removed knob left behind in a config, a typo that would otherwise be silently dropped, a
value outside a ``Literal``'s choices, and any ``__post_init__`` guard the config now violates.
Each config is parsed with the *imported* entry-script module so import-time registrations the real
run depends on are in place (``optim: flash_adamw`` is only valid because importing the trainer
extends HF's ``OptimizerNames``). A config also sets no key the parser would give the same value anyway, since a
reader takes every line for a choice.

Run: pytest tests/cpu/config/test_examples_parse.py
"""

import ast
import dataclasses
from pathlib import Path

import pytest
import transformers.training_args
import yaml

from halo.configs.parser import _TOOLKIT_DEFAULTS, HaloArgumentParser, _load_yaml
from halo.distributed.expert_parallel.deepep.capacity import GIN_MAX_TOKENS_PER_RANK
from tests.common.repo import REPO_ROOT, load_script_module

EXAMPLES_ROOT = REPO_ROOT / "examples"
TRAINING_ROOT = REPO_ROOT / "scripts" / "training"

# Which entry script owns a config, keyed by (directory prefix, file-name prefix). A directory maps
# to one script except where several methods share it (preference/, distillation/),
# which the file-name prefix disambiguates; ``test_every_example_is_mapped`` fails if a config lands
# outside every rule, so a new method cannot quietly go unchecked.
_SCRIPT_RULES = (
    ("examples/sft", "", "sft.py"),
    ("examples/classification", "", "classification.py"),
    ("examples/embedding", "", "embedding.py"),
    ("examples/reward", "", "reward/reward_model.py"),
    ("examples/preference", "dpo-", "preference/dpo.py"),
    ("examples/preference", "smpo-", "preference/smpo.py"),
    ("examples/preference", "kto-", "preference/kto.py"),
    ("examples/distillation", "self-distill-", "distillation/self_distill.py"),
    ("examples/distillation", "distill-", "distillation/teacher_distill.py"),
    ("examples/grpo/offline", "", "grpo/offline.py"),
    ("examples/grpo/online", "", "grpo/online.py"),
    ("examples/grpo/environmental", "", "grpo/environmental.py"),
)

_EXAMPLES = sorted(EXAMPLES_ROOT.rglob("*.yaml"))

# The keys a config may set to their default, because the line states the run's intent: what it trains on, adapter or
# full fine-tune, the parallel layout, step size, batch shape and length budget, and the objective and sampling knobs,
# whose defaults TRL owns or a GRPO arm sets for itself.
_INTENT_KEYS = frozenset(
    {
        "environment_type",
        "rewards",
        "use_peft",
        "expert_parallel_size",
        "fsdp_reshard_after_forward",
        "learning_rate",
        "per_device_train_batch_size",
        "gradient_accumulation_steps",
        "max_length",
        "max_prompt_length",
        "packing",
        "beta",
        "loss_type",
        "epsilon",
        "num_generations",
        "num_iterations",
        "scale_rewards",
        "drop_degenerate_groups",
        "top_entropy_quantile",
        "mask_truncated_completions",
        "enable_prefetch",
        "desirable_weight",
        "undesirable_weight",
        "temperature",
        "rollout_temperature",
    }
)


def uses_expert_parallelism(raw: dict) -> bool:
    """Does the raw YAML ask for an EP group (``expert_parallel_size`` above 1)?"""
    return (raw.get("expert_parallel_size") or 1) > 1


def declares_cross_node_ep(config: Path) -> bool:
    """Does ``config`` declare an EP group meant to span NVLink domains over RDMA?

    ``ep_scope`` is the declaration: ``node`` (the default) confines the group to one NVLink domain,
    where dispatch is validated far above the cross-node ceiling, while ``global`` lets it span
    domains and ride Gin. Read off the raw YAML rather than the parsed config because resolving
    ``ep_scope`` through ``ParallelismConfig`` needs a live launcher topology.
    """
    raw = yaml.safe_load(config.read_text(encoding="utf-8")) or {}
    return raw.get("ep_scope") == "global" and uses_expert_parallelism(raw)


_CROSS_NODE_EP_EXAMPLES = [config for config in _EXAMPLES if declares_cross_node_ep(config)]

_module_cache: dict[str, object] = {}


def script_for(config: Path) -> str | None:
    """Entry script (relative to ``scripts/training``) that consumes ``config``, or None."""
    relative = config.relative_to(REPO_ROOT).as_posix()
    matches = [
        (dir_prefix, script)
        for dir_prefix, name_prefix, script in _SCRIPT_RULES
        if relative.startswith(dir_prefix) and config.name.startswith(name_prefix)
    ]
    if not matches:
        return None
    return max(matches, key=lambda match: len(match[0]))[1]


ENV_GRPO_SCRIPT = "grpo/environmental.py"
ENV_GRPO_EXAMPLES = [config for config in _EXAMPLES if script_for(config) == ENV_GRPO_SCRIPT]


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


def parser_for(script: str) -> HaloArgumentParser:
    """The parser over the dataclasses the script hands its own ``parse_script_args(...)`` call."""
    module = _script_module(script)
    tree = ast.parse((TRAINING_ROOT / script).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "parse_script_args":
            names = [argument.id for argument in node.args]
            assert len(names) >= 4, f"{script} parses only {names}: the script args, configs and ParallelismArguments"
            return HaloArgumentParser(tuple(getattr(module, name) for name in names))
    raise AssertionError(f"{script} calls no parse_script_args")


def _unset_values(parser: HaloArgumentParser, name: str, values: dict) -> list:
    """What each dataclass declaring ``name`` takes when ``values`` leave it unset: the toolkit default where its
    guard holds, else the field's own default (``MISSING`` for a required field)."""
    toolkit = _TOOLKIT_DEFAULTS.get(name)
    if toolkit is not None and (toolkit[1] is None or toolkit[1](values)):
        return [toolkit[0]]
    return [
        field.default_factory() if field.default_factory is not dataclasses.MISSING else field.default
        for dataclass_type in parser.dataclass_types
        for field in dataclasses.fields(dataclass_type)
        if field.init and field.name == name
    ]


def noop_keys(parser: HaloArgumentParser, values: dict) -> list[str]:
    """The keys of config ``values`` that change nothing: each set to what every declaring field takes unset, plus
    ``num_train_epochs`` beside a positive ``max_steps``, where the Trainer derives the epochs from the steps."""
    keys = []
    for name, value in values.items():
        rest = {key: other for key, other in values.items() if key != name}
        unset = _unset_values(parser, name, rest)
        if unset and all(default is not dataclasses.MISSING and value == default for default in unset):
            keys.append(name)
    if "num_train_epochs" in values and (values.get("max_steps") or -1) > 0:
        keys.append("num_train_epochs")
    return keys


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
    unmapped = [c.relative_to(REPO_ROOT).as_posix() for c in _EXAMPLES if script_for(c) is None]
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


@pytest.mark.parametrize("config", _EXAMPLES, ids=lambda c: c.relative_to(EXAMPLES_ROOT).as_posix())
def test_example_sets_no_key_that_changes_nothing(config):
    """A line restating the parser's default reads as a deliberate choice, and goes stale silently when the default
    moves; only a key in ``_INTENT_KEYS`` may stand at its default."""
    noop = noop_keys(parser_for(script_for(config)), _load_yaml(str(config)))
    idle = [key for key in noop if key not in _INTENT_KEYS]
    assert not idle, (
        f"{config.name} sets {idle} to what the parser gives them anyway: drop the lines, or add a key that states "
        "the run's intent to _INTENT_KEYS"
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


def test_a_key_that_changes_nothing_is_caught():
    """Pins what the sweep relies on: a field default, a guarded toolkit default and an epoch count under max_steps
    are found; a value off its default is not."""
    values = {
        "model_name_or_path": "gpt2",
        "output_dir": "checkpoints/x",
        "seed": 42,
        "bf16": True,
        "learning_rate": 3.0e-4,
        "max_steps": 5,
        "num_train_epochs": 1,
    }
    assert noop_keys(parser_for("sft.py"), values) == ["seed", "bf16", "num_train_epochs"]
    assert noop_keys(parser_for("sft.py"), {**values, "fp16": True}) == ["seed", "num_train_epochs"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
