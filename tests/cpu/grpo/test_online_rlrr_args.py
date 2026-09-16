#!/usr/bin/env python
"""The RLVR script arguments' RLRR block: declared once, validated at parse time, inert values refused.

* The ``rlrr_*`` fields are DERIVED from :class:`RLRRConfig` — same type, default and help under the
  YAML spelling ``rlrr_arg_name`` gives — so a tunable added to the config reaches the CLI and the
  two rosters cannot drift.
* Every RLRR invariant (the clip band, positive finite τ/λ, finite band and threshold) fails when
  the args are built — the parse step — gate on or off, instead of inside the trainer after the model
  load.
* ``AdvantageShaping`` refuses a non-finite scale, which the normalizer's ``nan_to_num`` would
  otherwise turn into all-zero advantages.
* ``rlrr_*`` values without ``use_rlrr`` and ``sdpg_*`` / ``opd_positive_advantage_only`` values
  without ``use_sdpg`` are refused by the script rather than silently ignored.

    python tests/cpu/grpo/test_online_rlrr_args.py
"""

import ast
import dataclasses
import math
from pathlib import Path

import pytest

from src.args.mixins import AdvantageShaping, RLRRArguments, RLRRConfig, rlrr_arg_name
from src.args.rlvr_online_grpo_args import RLVROnlineGRPOScriptArguments
from src.training.parser import H4ArgumentParser
from src.training.script_runner import reject_non_default_args
from tests.common.utils import REPO_ROOT

RLVR_SCRIPT = REPO_ROOT / "scripts/training/online_grpo/rlvr.py"

NON_DEFAULT_CONFIG = RLRRConfig(
    mode="prr",
    tau=0.25,
    lam=512.0,
    xi_pos=0.05,
    xi_neg=-0.02,
    std_normalize=True,
    length_rerank=False,
    correctness_clip=False,
    correctness_threshold=0.75,
)


def _script_fields() -> dict[str, dataclasses.Field]:
    return {f.name: f for f in dataclasses.fields(RLVROnlineGRPOScriptArguments)}


def test_every_rlrr_script_field_is_derived_from_the_config_field():
    """Type, default and help all come from RLRRConfig; the only thing the script adds is the spelling."""
    script = _script_fields()
    for config_field in dataclasses.fields(RLRRConfig):
        arg = script[rlrr_arg_name(config_field.name)]
        assert arg.type == config_field.type, config_field.name
        assert arg.default == config_field.default, config_field.name
        assert arg.metadata["help"] == config_field.metadata["help"], config_field.name
    assert set(RLRRArguments.TUNABLES) == {rlrr_arg_name(f.name) for f in dataclasses.fields(RLRRConfig)}
    assert "rlrr_lambda" in RLRRArguments.TUNABLES and "rlrr_lam" not in script


def test_forwarded_config_equals_the_explicit_one():
    args = RLVROnlineGRPOScriptArguments(
        use_rlrr=True,
        **{rlrr_arg_name(f.name): getattr(NON_DEFAULT_CONFIG, f.name) for f in dataclasses.fields(RLRRConfig)},
    )
    assert args.build_rlrr_config() == NON_DEFAULT_CONFIG
    assert RLVROnlineGRPOScriptArguments().build_rlrr_config() is None


def test_the_parser_surfaces_the_derived_fields_with_their_choices():
    """A make_dataclass field must reach the parser like a hand-written one: parsed, and its Literal gated."""
    parser = H4ArgumentParser((RLVROnlineGRPOScriptArguments,))
    (args,) = parser.parse_dict({"use_rlrr": True, "rlrr_mode": "prr", "rlrr_lambda": 64.0})
    assert args.build_rlrr_config() == RLRRConfig(mode="prr", lam=64.0)
    with pytest.raises(ValueError, match="rlrr_mode"):
        parser.parse_dict({"rlrr_mode": "linear"})


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"rlrr_xi_pos": -1.0, "rlrr_xi_neg": 1.0}, "rlrr_xi_neg <= rlrr_xi_pos"),
        ({"rlrr_tau": 0.0}, "rlrr_tau"),
        ({"rlrr_tau": float("nan")}, "rlrr_tau"),
        ({"rlrr_lambda": -1.0}, "rlrr_lambda"),
        ({"rlrr_lambda": float("inf")}, "rlrr_lambda"),
        ({"rlrr_xi_pos": float("nan")}, "rlrr_xi_pos"),
        ({"rlrr_xi_neg": float("-inf")}, "rlrr_xi_neg"),
        ({"rlrr_correctness_threshold": float("nan")}, "rlrr_correctness_threshold"),
    ],
)
@pytest.mark.parametrize("use_rlrr", [True, False])
def test_every_rlrr_invariant_fails_when_the_args_are_built(kwargs, match, use_rlrr):
    with pytest.raises(ValueError, match=match):
        RLVROnlineGRPOScriptArguments(use_rlrr=use_rlrr, **kwargs)


def test_an_inverted_band_fails_on_the_cli_override_path(tmp_path):
    """``--key=value`` overrides bypass ``__post_init__``; the cooperative ``_validate_ranges`` re-run
    must reach the RLRR block too."""
    yaml_path = tmp_path / "rlvr.yaml"
    yaml_path.write_text("use_rlrr: true\n")
    parser = H4ArgumentParser((RLVROnlineGRPOScriptArguments,))
    with pytest.raises(ValueError, match="rlrr_xi_neg <= rlrr_xi_pos"):
        parser.parse_yaml_and_args(str(yaml_path), ["--rlrr_xi_neg=1.0"])


@pytest.mark.parametrize("kwargs", [{"neg_scale": float("nan")}, {"pos_scale": float("inf")}, {"neg_scale": -0.1}])
def test_advantage_shaping_refuses_a_non_finite_or_negative_scale(kwargs):
    with pytest.raises(ValueError, match="finite value >= 0"):
        AdvantageShaping(mode="asymmetric", **kwargs)


def test_a_nan_scale_is_refused_at_the_script_args_even_at_the_mean_mode():
    """Built eagerly, so the value fails at parse time whether or not the mode would use it."""
    with pytest.raises(ValueError, match="neg_scale"):
        RLVROnlineGRPOScriptArguments(advantage_neg_scale=math.nan)


# --- Tunables set beside a closed gate are refused, not ignored ---


def test_rlrr_tunables_without_the_gate_are_refused_by_the_default_comparing_guard():
    reject_non_default_args(
        "RLVR Online GRPO with use_rlrr off", RLVROnlineGRPOScriptArguments(), *RLRRArguments.TUNABLES
    )
    with pytest.raises(ValueError, match=r"\['rlrr_tau'\]"):
        reject_non_default_args(
            "RLVR Online GRPO with use_rlrr off", RLVROnlineGRPOScriptArguments(rlrr_tau=0.3), *RLRRArguments.TUNABLES
        )


def test_sdpg_tunables_cover_the_shared_block_plus_the_rlvr_only_gate():
    assert set(RLVROnlineGRPOScriptArguments.SDPG_TUNABLES) == {
        "sdpg_hint_template",
        "sdpg_loss",
        "sdpg_temperature",
        "sdpg_beta_base",
        "sdpg_beta_warmup_steps",
        "sdpg_beta_decay_steps",
        "opd_positive_advantage_only",
    }
    # opd_positive_advantage_only defaults to True: only the default-comparing form can guard it.
    with pytest.raises(ValueError, match=r"\['opd_positive_advantage_only'\]"):
        reject_non_default_args(
            "RLVR Online GRPO with use_sdpg off",
            RLVROnlineGRPOScriptArguments(opd_positive_advantage_only=False),
            *RLVROnlineGRPOScriptArguments.SDPG_TUNABLES,
        )


def _guards_by_gate(script: Path) -> dict[str, set[str]]:
    """``{gate flag: {starred rosters}}`` for each ``if not args.<gate>: reject_non_default_args(...)``."""
    found: dict[str, set[str]] = {}
    for node in ast.walk(ast.parse(script.read_text())):
        if not (isinstance(node, ast.If) and isinstance(node.test, ast.UnaryOp) and isinstance(node.test.op, ast.Not)):
            continue
        gate = ast.unparse(node.test.operand)
        for call in ast.walk(node):
            if isinstance(call, ast.Call) and ast.unparse(call.func) == "reject_non_default_args":
                found.setdefault(gate, set()).update(
                    ast.unparse(arg.value) for arg in call.args if isinstance(arg, ast.Starred)
                )
    return found


def test_the_rlvr_script_guards_both_blocks_behind_their_gates():
    guards = _guards_by_gate(RLVR_SCRIPT)
    assert guards.get("args.use_rlrr") == {"RLRRArguments.TUNABLES"}, guards
    assert guards.get("args.use_sdpg") == {"args.SDPG_TUNABLES"}, guards


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
