#!/usr/bin/env python
"""Scripts with padded batches must request the padded-workload backend, under the sinks policy.

Every GRPO batch is right-padded (prompts left-padded, completions right-padded — none of these
scripts packs). A YAML that pins no ``attn_implementation`` reaches the loader with the padded
default, the same request ``teacher_distill.py`` / ``classification.py`` / the reward scripts make:
nothing on Blackwell, where the loader's FA4 outruns SDPA on these batches, except for a remote-code
checkpoint whose class the resolver cannot read (Bailing's refuses every flash label at build) and a
softcapped one, whose padded batches would reach FA4's varlen kernels (the softcap then lands the run on
FA2); ``sdpa`` on every other GPU.

It is a *request*, not a pin: with live sinks (``reset_sinks: false`` — the on-policy gpt-oss flow)
the backend resolver refuses a sink-dropping impl, so forcing SDPA there would reject the run
outright. The default must therefore be dropped exactly in that case — which is the whole reason
the exemption lives in ``default_padded_attn_implementation`` rather than in a ternary each script
copies: four copies is four chances for one of them to keep requesting SDPA, and the run that
notices is an on-policy gpt-oss run failing at model load.

Run: python tests/cpu/config/test_grpo_padded_attn_default.py  (or pytest)
"""

import ast
import contextlib
import sys
import types
from pathlib import Path
from unittest import mock

import pytest
from datasets import DatasetDict
from transformers import Gemma2Config, GemmaConfig, PretrainedConfig, Qwen3Config

from halo.distributed.parallelism_config import ParallelismConfig
from halo.entrypoints import model_setup
from halo.entrypoints.model_setup import default_padded_attn_implementation
from halo.models.attention import selection
from tests.common.offline_grpo import offline_grpo_dataset
from tests.common.repo import REPO_ROOT, load_script_module

_TRAINING_DIR = REPO_ROOT / "scripts" / "training"

# The three GRPO entry points; each pads its batches and loads through load_model_and_tokenizer.
_GRPO_SCRIPTS = ["grpo/offline.py", "grpo/environmental.py", "grpo/online.py"]

_HELPER = "default_padded_attn_implementation"


class _StopAtModelLoad(Exception):
    """Raised by the stubbed loader: main() is only run as far as the seam under test."""


def _load_script_module(rel_path: str):
    name = "halo_test_attn_" + rel_path.replace("/", "_").removesuffix(".py")
    return load_script_module(f"scripts/training/{rel_path}", name)


class _RemoteCodeConfig(PretrainedConfig):
    """A config no Auto mapping knows, as a remote-code checkpoint's is (Bailing's)."""

    model_type = "halo_test_remote_code"


@contextlib.contextmanager
def _blackwell(checkpoint_config):
    """A Blackwell host with an FA2 build whose policy checkpoint has ``checkpoint_config``; yields the config
    fetch."""
    with (
        mock.patch.object(selection, "is_blackwell_gpu", lambda: True),
        mock.patch.object(selection, "is_flash_attn_2_available", lambda: True),
        mock.patch.object(model_setup, "is_flash_attn_2_available", lambda: True),
        mock.patch.object(model_setup, "fetch_checkpoint_config", return_value=checkpoint_config) as fetch,
    ):
        yield fetch


def _requested_attn(
    rel_path: str, yaml_body: str, tmp_path: Path, parallelism_config=None, checkpoint_config=None
) -> str | None:
    """The attention implementation the script's model load would actually resolve under, on a Blackwell host
    whose checkpoint has ``checkpoint_config`` when one is given.

    ``load_model_and_tokenizer`` falls back to ``model_config.attn_implementation`` when the script passes
    none, so the effective request — not the kwarg alone — is what the run behaves on.
    """
    module = _load_script_module(rel_path)
    config = tmp_path / "config.yaml"
    config.write_text(
        # bf16/use_cpu keep TrainingArguments constructible on a GPU-less runner.
        f"model_name_or_path: dummy/model\noutput_dir: {tmp_path / 'out'}\nbf16: false\nuse_cpu: true\n{yaml_body}"
    )
    captured: dict = {}

    def fake_load(context, training_config, model_config, dist_args, **kwargs):
        captured["requested"] = model_config.attn_implementation or kwargs.get("attn_default")
        raise _StopAtModelLoad

    runtime = types.SimpleNamespace(
        parallelism_config=parallelism_config or ParallelismConfig(),
        model_source="dummy/model",
        policy_from_checkpoint=False,
        mode_suffix="",
        local_rank=0,
    )
    loaders = {"load_model_and_tokenizer": fake_load, "load_policy_model": fake_load}
    with contextlib.ExitStack() as stack:
        stack.enter_context(mock.patch.object(module, "init_training_script", return_value=runtime))
        for name, fake in loaders.items():
            if hasattr(module, name):
                stack.enter_context(mock.patch.object(module, name, fake))
        # SMPO reads its data before the model load to decide the run's modality: a text run here.
        if hasattr(module, "resolve_preference_vlm_run"):
            stack.enter_context(mock.patch.object(module, "load_script_datasets", return_value=(None, False)))
            stack.enter_context(mock.patch.object(module, "resolve_preference_vlm_run", return_value=False))
        # Offline GRPO refuses splits its run-start reference cannot read before the model load.
        elif rel_path == "grpo/offline.py":
            splits = DatasetDict(train=offline_grpo_dataset(2), test=offline_grpo_dataset(2, offset=2))
            stack.enter_context(mock.patch.object(module, "load_script_datasets", return_value=(splits, False)))
        if checkpoint_config is not None:
            stack.enter_context(_blackwell(checkpoint_config))
        # The log tee redirects the process's stdout/stderr fds — keep it out of the test process.
        stack.enter_context(mock.patch("halo.entrypoints.script.install_log_tee"))
        stack.enter_context(mock.patch.object(sys, "argv", ["prog", str(config)]))
        stack.enter_context(pytest.raises(_StopAtModelLoad))
        module.main()
    return captured["requested"]


@pytest.mark.parametrize("script", _GRPO_SCRIPTS)
def test_unpinned_config_requests_the_padded_workload_backend(script, tmp_path):
    """An unpinned YAML must not reach the loader with nothing: padded GRPO batches would land on
    FA4's varlen path."""
    assert _requested_attn(script, "", tmp_path) == "sdpa"


_PADDED_DEFAULT_SCRIPTS = [*_GRPO_SCRIPTS, "preference/smpo.py"]


# Gemma 2 27B's attention: 128-wide heads, which FA4's general kernel runs, and a logit softcap.
_SOFTCAPPED = Gemma2Config(hidden_size=4608, num_attention_heads=32, num_key_value_heads=16, head_dim=128)


@pytest.mark.parametrize("script", _PADDED_DEFAULT_SCRIPTS)
@pytest.mark.parametrize(
    ("checkpoint_config", "expected"),
    [
        (Qwen3Config(), None),
        (GemmaConfig(), None),
        (_RemoteCodeConfig(), "sdpa"),
        (_SOFTCAPPED, "flash_attention_2"),
    ],
    ids=["native", "head_dim-256", "remote", "softcapped"],
)
def test_blackwell_leaves_the_kernel_to_the_loader_unless_the_checkpoint_rules_it_out(
    script, checkpoint_config, expected, tmp_path
):
    """On Blackwell an unpinned padded run takes the loader's pick, a head_dim-256 checkpoint's included (Gemma 1
    has neither a softcap nor a window, so FA4's dedicated kernel trains it). A remote-code checkpoint keeps SDPA,
    since the resolver cannot read its flags and Bailing's refuses a flash label only at the model build; a
    softcapped one, whose padded batches would reach FA4's varlen kernels, requests FA2, which trains its softcap."""
    assert _requested_attn(script, "", tmp_path, checkpoint_config=checkpoint_config) == expected


@pytest.mark.parametrize("script", _GRPO_SCRIPTS)
def test_pinned_attention_wins_over_the_padded_default(script, tmp_path):
    assert _requested_attn(script, "attn_implementation: flash_attention_4\n", tmp_path) == "flash_attention_4"


@pytest.mark.parametrize("script", _GRPO_SCRIPTS)
def test_live_sinks_drop_the_padded_default(script, tmp_path):
    """reset_sinks: false keeps the pretrained sinks live, and the resolver then accepts only a
    sink-carrying impl — requesting SDPA would turn an on-policy gpt-oss run into a hard rejection."""
    assert _requested_attn(script, "reset_sinks: false\n", tmp_path) is None


# A CP shape ParallelismConfig would refuse on a GPU-less host (CP must fit an NVLink domain); the
# script reads only this verdict before its model load.
_CP_PARALLELISM = types.SimpleNamespace(is_cp_mode=True)


# The CP-capable scripts that request the padded default.
_CP_SCRIPTS = ["grpo/offline.py", "preference/smpo.py"]


@pytest.mark.parametrize("script", _CP_SCRIPTS)
@pytest.mark.parametrize(
    ("yaml_body", "expected"),
    [("", None), ("attn_implementation: flash_attention_2\n", "flash_attention_2")],
    ids=["unpinned", "pinned"],
)
def test_cp_scripts_leave_the_kernel_to_the_loader(script, yaml_body, expected, tmp_path):
    """Ulysses calls FlashAttention itself, and the CP validator refuses an SDPA label for most
    families, so the padded default must not reach a CP run's load."""
    assert _requested_attn(script, yaml_body, tmp_path, _CP_PARALLELISM) == expected


def test_smpo_off_cp_keeps_the_padded_default(tmp_path):
    assert _requested_attn("preference/smpo.py", "", tmp_path) == "sdpa"


# --- the consolidated seam -------------------------------------------------------------------


@pytest.mark.parametrize(
    ("pinned", "sinks_reset", "context_parallel", "expected"),
    [
        (None, True, False, "sdpa"),
        (None, False, False, None),
        (None, True, True, None),
        (None, False, True, None),
        ("flash_attention_4", True, False, "flash_attention_4"),
        ("flash_attention_4", False, False, "flash_attention_4"),
        ("eager", True, True, "eager"),
    ],
)
def test_the_seam_owns_the_padded_default_exemptions(pinned, sinks_reset, context_parallel, expected):
    """The whole decision in one place: pin wins; unpinned takes the padded default only while the
    sinks are being reset and the run is not context-parallel."""
    model_config = types.SimpleNamespace(attn_implementation=pinned)
    with mock.patch.object(selection, "is_blackwell_gpu", lambda: False):
        assert (
            default_padded_attn_implementation(
                model_config, sinks_reset=sinks_reset, context_parallel=context_parallel
            )
            == expected
        )


@pytest.mark.parametrize(
    ("pinned", "sinks_reset", "context_parallel", "expected"),
    [("eager", True, False, "eager"), (None, False, False, None), (None, True, True, None)],
)
def test_the_blackwell_seam_fetches_no_config_it_does_not_read(pinned, sinks_reset, context_parallel, expected):
    """The checkpoint is read only to choose between the loader's pick and SDPA; every other branch decides
    without the hub."""
    model_config = types.SimpleNamespace(attn_implementation=pinned)
    with _blackwell(Qwen3Config()) as fetch:
        assert (
            default_padded_attn_implementation(
                model_config, sinks_reset=sinks_reset, context_parallel=context_parallel
            )
            == expected
        )
    fetch.assert_not_called()


@pytest.mark.parametrize(
    ("policy_config", "companion_config", "expected"),
    [
        (Qwen3Config(), Qwen3Config(), None),
        (Qwen3Config(), _RemoteCodeConfig(), "sdpa"),
        (_RemoteCodeConfig(), Qwen3Config(), "sdpa"),
        (Qwen3Config(), GemmaConfig(), None),
        (Qwen3Config(), _SOFTCAPPED, "sdpa"),
        (_SOFTCAPPED, Qwen3Config(), "flash_attention_2"),
    ],
    ids=["native-native", "native-remote", "remote-native", "native-hd256", "native-softcapped", "softcapped-native"],
)
def test_the_blackwell_seam_decides_on_every_config_it_loads(policy_config, companion_config, expected):
    """A companion model loaded on the same request (a distillation teacher) can rule out the loader's pick
    for both loads, and so can the policy's fetched config: a forward-only teacher's softcap counts, since it stays
    off FA4's varlen kernels too."""
    model_config = types.SimpleNamespace(attn_implementation=None)
    with _blackwell(policy_config) as fetch:
        assert (
            default_padded_attn_implementation(model_config, sinks_reset=True, companion_configs=(companion_config,))
            == expected
        )
    fetch.assert_called_once()


def test_the_blackwell_seam_reads_the_config_it_is_handed():
    """A handed policy config is read in place of a fetch."""
    model_config = types.SimpleNamespace(attn_implementation=None)
    with _blackwell(Qwen3Config()) as fetch:
        assert (
            default_padded_attn_implementation(model_config, sinks_reset=True, checkpoint_config=_RemoteCodeConfig())
            == "sdpa"
        )
    fetch.assert_not_called()


def _helper_calls(path: Path) -> list[ast.Call]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == _HELPER
    ]


# The training scripts, and the preference load DPO and KTO share.
_HELPER_CALL_SITES = sorted(
    path
    for path in (*_TRAINING_DIR.rglob("*.py"), REPO_ROOT / "halo/entrypoints/model_setup.py")
    if _HELPER in path.read_text(encoding="utf-8")
)


def test_the_call_site_sweep_is_not_empty():
    """Guard the derivation: an empty sweep would pass the assertion below vacuously."""
    assert len(_HELPER_CALL_SITES) >= 8, _HELPER_CALL_SITES


@pytest.mark.parametrize("script", _HELPER_CALL_SITES, ids=lambda p: p.stem)
def test_every_call_site_wires_the_exemption_to_the_runs_own_flag(script):
    """A literal here is a claim about the MODEL ("these batches are never sink-carrying") made at a
    site that cannot know it: the flag is per run. Every caller must pass the run's reset_sinks."""
    for call in _helper_calls(script):
        wired = [kw.value for kw in call.keywords if kw.arg == "sinks_reset"]
        assert wired, f"{script.name} calls {_HELPER} without sinks_reset"
        assert not isinstance(wired[0], ast.Constant), (
            f"{script.name} hardcodes sinks_reset; it must come from the run's reset_sinks flag"
        )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
