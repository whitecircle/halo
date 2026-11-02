#!/usr/bin/env python
"""``lora_target_modules`` means what its spelling says, at any length.

A list names modules: PEFT adapts every module whose path is an entry or ends in ``.<entry>``, so an entry matches
whole trailing path segments. A bare string is PEFT's own spelling, the ``all-linear`` sentinel or a regex
fullmatched against each full module path. TRL's ``ModelConfig`` collapses a one-entry list to its bare entry,
which PEFT would then read as that regex, so ``[o_proj]`` would match no module and fail at injection;
``ModelArguments`` keeps the list. Driven through the scripts' parser and the production ``build_peft_config`` to
the adapted model:

* one named target adapts that module on every layer, whichever way the list reaches the config (a YAML list, a
  CLI override over a YAML, a flags-only command line, a tuple built in code), as a path suffix, and with DoRA too;
* it does so on every tiny family, dense and MoE, under an attention target its LoRA rows name;
* a one-entry ``[all-linear]`` is still PEFT's sentinel, and a bare YAML string is still a regex;
* a regex written into the list, from the YAML or the command line, which PEFT would compare to module paths
  verbatim and match to nothing (alone, or silently beside an entry that matches), is refused when the config is
  parsed, for each character a regex spells and a module path does not.

Run: pytest tests/cpu/peft/test_lora_target_spelling.py
"""

import pytest
import torch
from peft import get_peft_model
from peft.tuners.tuners_utils import BaseTunerLayer

from halo.configs.parser import HaloArgumentParser
from halo.configs.script_arguments import ModelArguments
from halo.model_loading.parallel.peft_setup import build_peft_config
from tests.common.tiny_models import TINY_DENSE_FAMILIES, TINY_DENSE_FAMILY, TINY_MOE_FAMILIES, tiny_family_model

TARGET = "o_proj"
# The attention projection a family's rows adapt where it names none: LFM-2 spells its attention output out_proj.
DEFAULT_TARGETS = ("o_proj", "out_proj")
ROSTER = {**TINY_DENSE_FAMILIES, **TINY_MOE_FAMILIES}
YAML_HEAD = "model_name_or_path: dummy/model\nuse_peft: true\n"
# The characters a regex spells and a module path does not, written out apart from the refusal's own set.
REGEX_CHARACTERS = {
    "backslash": "\\",
    "star": "*",
    "plus": "+",
    "question": "?",
    "open-paren": "(",
    "close-paren": ")",
    "open-bracket": "[",
    "close-bracket": "]",
    "open-brace": "{",
    "close-brace": "}",
    "pipe": "|",
    "caret": "^",
    "dollar": "$",
}


def _parse(tmp_path, argv: list[str], yaml_body: str | None = None) -> ModelArguments:
    """``ModelArguments`` as the training scripts parse it: ``yaml_body`` (when given) is the config ``argv``
    overrides, else ``argv`` is a flags-only command line."""
    if yaml_body is not None:
        config = tmp_path / "config.yaml"
        config.write_text(YAML_HEAD + yaml_body)
        argv = [str(config), *argv]
    parser = HaloArgumentParser((ModelArguments,))
    (model_config,) = parser.parse_dict(parser.load_values(argv))
    return model_config


def _named(model: torch.nn.Module, leaf: str) -> set[str]:
    return {name for name, _ in model.named_modules() if name.rsplit(".", 1)[-1] == leaf}


def _adapted(model: torch.nn.Module, model_config: ModelArguments) -> set[str]:
    """The module paths the run's adapter wraps, built as the scripts build it."""
    peft_model = get_peft_model(model, build_peft_config(model, model_config))
    return {
        name.removeprefix("base_model.model.")
        for name, module in peft_model.named_modules()
        if isinstance(module, BaseTunerLayer)
    }


SOURCES = {
    "yaml-flow-list": ([], f"lora_target_modules: [{TARGET}]\n"),
    "yaml-block-list": ([], f"lora_target_modules:\n  - {TARGET}\n"),
    "cli-override": ([f"--lora_target_modules={TARGET}"], "lora_target_modules: [q_proj, v_proj]\n"),
    "flags-only": (
        ["--model_name_or_path", "dummy/model", "--use_peft", "true", "--lora_target_modules", TARGET],
        None,
    ),
}


@pytest.mark.parametrize("source", sorted(SOURCES))
def test_one_named_target_adapts_that_module_on_every_layer(source, tmp_path):
    argv, yaml_body = SOURCES[source]
    model_config = _parse(tmp_path, argv, yaml_body)
    model = tiny_family_model(TINY_DENSE_FAMILY)
    expected = _named(model, TARGET)
    assert len(expected) == model.config.num_hidden_layers, "premise: one o_proj per layer"

    assert model_config.lora_target_modules == [TARGET]
    assert _adapted(model, model_config) == expected


def test_one_target_names_a_module_by_its_path_suffix():
    model_config = ModelArguments(
        model_name_or_path="dummy/model", use_peft=True, lora_target_modules=["self_attn.o_proj"]
    )
    model = tiny_family_model(TINY_DENSE_FAMILY)

    assert _adapted(model, model_config) == _named(model, TARGET)


def test_a_one_entry_tuple_built_in_code_names_its_module():
    """TRL collapses any one-entry sequence, so a tuple a caller builds the config with is kept as names too."""
    model_config = ModelArguments(model_name_or_path="dummy/model", use_peft=True, lora_target_modules=(TARGET,))
    model = tiny_family_model(TINY_DENSE_FAMILY)

    assert model_config.lora_target_modules == [TARGET]
    assert _adapted(model, model_config) == _named(model, TARGET)


def test_one_named_target_takes_dora():
    model_config = ModelArguments(
        model_name_or_path="dummy/model", use_peft=True, use_dora=True, lora_target_modules=[TARGET]
    )
    model = tiny_family_model(TINY_DENSE_FAMILY)
    expected = _named(model, TARGET)

    peft_model = get_peft_model(model, build_peft_config(model, model_config))

    dora = {
        name.removeprefix("base_model.model.")
        for name, module in peft_model.named_modules()
        if isinstance(module, BaseTunerLayer) and "default" in module.lora_magnitude_vector
    }
    assert dora == expected


@pytest.mark.parametrize("family", sorted(ROSTER))
def test_one_named_target_adapts_that_module_in_every_family(family):
    """One attention target a family's LoRA rows name (Zaya's only one, ``o_proj``) adapts every module of that
    name, as a longer list would."""
    tiny = ROSTER[family]
    torch.manual_seed(0)
    model = tiny_family_model(tiny)
    target = next(name for name in tiny.attention_targets or DEFAULT_TARGETS if _named(model, name))
    expected = _named(model, target)
    model_config = ModelArguments(model_name_or_path="dummy/model", use_peft=True, lora_target_modules=[target])

    assert _adapted(model, model_config) == expected


@pytest.mark.parametrize(
    "argv, yaml_body",
    [([], "lora_target_modules: [all-linear]\n"), (["--lora_target_modules=ALL-LINEAR"], "")],
    ids=["yaml-list", "cli-override"],
)
def test_a_one_entry_all_linear_list_is_peft_sentinel(argv, yaml_body, tmp_path):
    """PEFT reads ``all-linear`` only as a bare string; the list a user writes reaches it as that sentinel and
    adapts every linear layer but the output head."""
    model_config = _parse(tmp_path, argv, yaml_body)
    model = tiny_family_model(TINY_DENSE_FAMILY)
    expected = {
        name for name, module in model.named_modules() if isinstance(module, torch.nn.Linear) and name != "lm_head"
    }

    assert _adapted(model, model_config) == expected


def test_a_bare_yaml_string_is_peft_regex(tmp_path):
    """A string is fullmatched against each full module path: a pattern selects the paths it spells, and a bare
    module name, which spells no full path, matches nothing."""
    pattern = _parse(tmp_path, [], f"lora_target_modules: '.*\\.0\\.self_attn\\.{TARGET}'\n")
    assert _adapted(tiny_family_model(TINY_DENSE_FAMILY), pattern) == {f"model.layers.0.self_attn.{TARGET}"}

    bare_name = _parse(tmp_path, [], f"lora_target_modules: {TARGET}\n")
    assert bare_name.lora_target_modules == TARGET
    with pytest.raises(ValueError, match=f"Target modules {TARGET} not found"):
        _adapted(tiny_family_model(TINY_DENSE_FAMILY), bare_name)


@pytest.mark.parametrize(
    "argv, yaml_body",
    [
        ([], f"lora_target_modules: ['.*\\.{TARGET}']\n"),
        ([], "lora_target_modules: [q_proj, '(k|v)_proj']\n"),
        ([f"--lora_target_modules=.*\\.{TARGET}"], ""),
    ],
    ids=["one-entry", "beside-a-name", "cli"],
)
def test_a_regex_in_the_list_is_refused_when_parsed(argv, yaml_body, tmp_path):
    """PEFT compares a list entry to module paths verbatim, so a regex there adapts nothing: alone the run would
    fail at injection after the load, beside a name it would drop out unnoticed. The command line always gives a
    list, so a regex there is refused too."""
    with pytest.raises(ValueError, match="are regexes, but a list names modules"):
        _parse(tmp_path, argv, yaml_body)


@pytest.mark.parametrize("character", REGEX_CHARACTERS.values(), ids=REGEX_CHARACTERS.keys())
def test_an_entry_spelling_any_regex_character_is_refused(character, tmp_path):
    """No module path spells the character, so an entry carrying it matches nothing; each entry carries one such
    character alone, so a character dropped from the refusal fails its own case."""
    entry = f"{TARGET}{character}"
    assert sum(entry.count(other) for other in REGEX_CHARACTERS.values()) == 1, "premise: one regex character"

    with pytest.raises(ValueError, match="are regexes, but a list names modules"):
        _parse(tmp_path, [], f"lora_target_modules: [q_proj, '{entry}']\n")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
