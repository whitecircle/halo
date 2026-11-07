#!/usr/bin/env python
"""Frozen auxiliary models share one loader and one sinks policy.
A per-site copy drifts silently: a pin read off an object that cannot carry it, or sdpa over sinks never reset."""

import ast
import inspect
import pathlib
import types
from unittest.mock import patch

import pytest
import torch
from torch import nn
from transformers import AutoModelForImageTextToText, GenerationConfig, Qwen3Config

import halo.model_loading.parallel.frozen_models as frozen_models
import scripts.training.distillation.self_distill as self_distill_script
import scripts.training.distillation.teacher_distill as distill_script
import scripts.training.grpo.offline as offline_grpo_script
from halo.configs.script_arguments import ModelArguments
from halo.distributed.parallelism_config import ParallelismConfig
from halo.model_loading import model_preparation
from halo.model_loading.parallel.frozen_models import (
    load_frozen_auxiliary_model,
    load_reference_model_for_on_policy_grpo,
    load_reference_model_for_preference,
)
from halo.model_loading.parallel.model_source import ModelSource
from halo.models.moe.expert_lora_spec import ExpertLoraSpec
from tests.common.ep_stubs import StubEPLayerBase
from tests.common.frozen_loader import STUB_CONFIG, STUB_RESOLVED_ATTN, captured_load, stub_frozen_loader
from tests.common.repo import REPO_ROOT

AUX_MODEL = "org/auxiliary"

# Resolving the backend and applying the sinks policy are one decision (a neutralized ``SinksPolicy`` approves a
# sink-dropping backend), so a site binding either again owns a second copy.
FROZEN_LOAD_STEPS = ("resolve_attn_implementation", "apply_sinks_policy")

# Allowlist: the sweep covers the whole tree, so a new copy is caught by default. loader.py re-runs the resolver after
# the parallelism overrides; the checkpoint modules apply a recorded or explicit policy to a checkpoint on disk.
FROZEN_LOAD_STEP_OWNERS = frozenset(
    {
        "halo/models/attention/selection.py",
        "halo/models/attention/gpt_oss_sinks.py",
        "halo/model_loading/model_preparation.py",
        "halo/model_loading/parallel/frozen_models.py",
        "halo/model_loading/parallel/loader.py",
        "scripts/training/embedding.py",
        "halo/checkpoint/sidecars.py",
        "halo/checkpoint/tools/vocab_patch.py",
    }
)


def _token_setup_args():
    return types.SimpleNamespace(
        eos_token=None,
        bos_token=None,
        pad_token=None,
        chat_template=None,
        force_chat_template=False,
        added_special_tokens=None,
        tokenizer_backend="hf",
    )


def _tokenizer():
    # Every ``<special>_token_id`` accompanies its token, as ``PreTrainedTokenizerBase.__getattr__`` resolves them.
    return types.SimpleNamespace(
        eos_token="<e>",
        eos_token_id=1,
        bos_token="<b>",
        bos_token_id=2,
        pad_token="<p>",
        pad_token_id=3,
        chat_template="{{ messages }}",
    )


def _preference_reference(*, reset_sinks, attn_default=None, attn_implementation=None):
    return load_reference_model_for_preference(
        _token_setup_args(),
        ModelArguments(model_name_or_path=AUX_MODEL, attn_implementation=attn_implementation),
        types.SimpleNamespace(bf16=True, fp16=False, precompute_ref_log_probs=False),
        ParallelismConfig(),
        _tokenizer(),
        is_vlm=False,
        method="DPO",
        reset_sinks=reset_sinks,
        attn_default=attn_default,
    )


def _script_teacher(*, reset_sinks):
    """The script preloads the teacher for the trainer."""
    return distill_script._load_distill_teacher(
        args=types.SimpleNamespace(teacher_model=AUX_MODEL, teacher_model_revision=None, reset_sinks=reset_sinks),
        model_config=types.SimpleNamespace(trust_remote_code=False),
        training_config=types.SimpleNamespace(bf16=True, fp16=False),
        requested_attn=None,
        is_vlm=False,
        local_rank=0,
    )


def _sdpg_reference(*, reset_sinks):
    args = _token_setup_args()
    args.reference_kl_coef, args.reference_model_name_or_path, args.reset_sinks = 0.1, AUX_MODEL, reset_sinks
    return self_distill_script._load_sdpg_reference(
        args=args,
        model_config=ModelArguments(model_name_or_path=AUX_MODEL),
        sft_config=types.SimpleNamespace(bf16=True, fp16=False),
        tokenizer=_tokenizer(),
        is_vlm=False,
        attn_default=None,
    )


def _offline_grpo_reference(*, reset_sinks):
    """Native expert LoRA uses a live unadapted base rather than full-FT's run-start score cache."""
    args = _token_setup_args()
    args.reset_sinks = reset_sinks
    return offline_grpo_script._load_kl_reference(
        args,
        types.SimpleNamespace(
            parallelism_config=ParallelismConfig(expert_lora=ExpertLoraSpec(r=2, alpha=4)), model_source=AUX_MODEL
        ),
        types.SimpleNamespace(kl_beta=0.1, bf16=True, fp16=False),
        ModelArguments(model_name_or_path=AUX_MODEL),
        policy=nn.Sequential(StubEPLayerBase()),
        tokenizer=_tokenizer(),
        peft_config=None,
        attn_default=None,
    )


def _on_policy_grpo_reference(*, reset_sinks):
    """The online and environmental GRPO KL reference, handed to TRL in place of its own."""
    return load_reference_model_for_on_policy_grpo(
        _token_setup_args(),
        ModelArguments(model_name_or_path=AUX_MODEL),
        types.SimpleNamespace(beta=0.1, bf16=True, fp16=False),
        ParallelismConfig(),
        _tokenizer(),
        peft_config=None,
        reset_sinks=reset_sinks,
        attn_default=None,
    )


FROZEN_LOAD_SITES = {
    "preference-reference": _preference_reference,
    "on-policy-grpo-reference": _on_policy_grpo_reference,
    "distillation-script": _script_teacher,
    "sdpg-reference": _sdpg_reference,
    "offline-grpo-reference": _offline_grpo_reference,
}

# Where DPO and KTO load both a policy and a live reference, so the two loads must agree.
PREFERENCE_LOADS = REPO_ROOT / "halo/entrypoints/model_setup.py"


@pytest.mark.parametrize("site", FROZEN_LOAD_SITES.values(), ids=list(FROZEN_LOAD_SITES))
def test_every_frozen_load_site_reaches_the_shared_loader(site):
    """Only the shared loader is stubbed, so a site with its own fetch would reach the hub and record nothing."""
    with stub_frozen_loader() as caps:
        model = site(reset_sinks=True)
    captured = captured_load(caps)
    assert model is captured.model
    assert captured.load_positional[0] == AUX_MODEL
    # A pinned checkpoint paired with hub-main's config would load the wrong model.
    assert captured.config["revision"] == captured.load["revision"]
    assert captured.load["attn_implementation"] == STUB_RESOLVED_ATTN
    assert captured.load["dtype"] is torch.bfloat16


def _repo_tokenizers(vocabs):
    fetched = {}

    def from_pretrained(path, revision=None, **_):
        fetched[path] = revision
        return types.SimpleNamespace(get_vocab=lambda: dict(vocabs[path]))

    return types.SimpleNamespace(from_pretrained=from_pretrained), fetched


POLICY_CONFIG = ModelArguments(model_name_or_path=AUX_MODEL, model_revision="abc123")


def _anchor_args(reference):
    args = _token_setup_args()
    args.reference_kl_coef, args.reference_model_name_or_path, args.reset_sinks = 0.1, reference, True
    return args


@pytest.mark.parametrize(("reference", "pinned"), [(AUX_MODEL, "abc123"), ("org/other-reference", None)])
def test_the_sdpg_reference_takes_the_policy_pin_only_from_the_policy_repo(reference, pinned):
    """The pin names a commit in the policy's repo; sent to another repo it 404s."""
    with stub_frozen_loader() as caps:
        self_distill_script._load_sdpg_reference(
            args=_anchor_args(reference),
            model_config=POLICY_CONFIG,
            sft_config=types.SimpleNamespace(bf16=True, fp16=False),
            tokenizer=_tokenizer(),
            is_vlm=False,
            attn_default=None,
        )
    captured = captured_load(caps)
    assert captured.load_positional[0] == reference
    assert captured.config["revision"] == captured.load["revision"] == pinned


def test_a_separate_reference_repo_of_another_tokenizer_is_refused():
    """The reference is keyed to the policy's tokenizer, so a different id map scores other tokens."""
    auto_tokenizer, fetched = _repo_tokenizers({AUX_MODEL: {"a": 0, "b": 1}, "org/other-reference": {"a": 1, "b": 0}})
    with patch.object(self_distill_script, "AutoTokenizer", auto_tokenizer):
        with pytest.raises(ValueError, match="disagree on"):
            self_distill_script._require_reference_token_ids(_anchor_args("org/other-reference"), POLICY_CONFIG)
        assert fetched == {AUX_MODEL: "abc123", "org/other-reference": None}
        fetched.clear()
        self_distill_script._require_reference_token_ids(_anchor_args(AUX_MODEL), POLICY_CONFIG)
    assert fetched == {}


@pytest.mark.parametrize("reset_sinks", [True, False])
@pytest.mark.parametrize("site", FROZEN_LOAD_SITES.values(), ids=list(FROZEN_LOAD_SITES))
def test_every_frozen_load_site_gets_the_sinks_policy(site, reset_sinks):
    """The policy the backend was resolved under must land on the returned model, or every logprob shifts by nats."""
    with stub_frozen_loader() as caps:
        model = site(reset_sinks=reset_sinks)
    captured = captured_load(caps)

    applied, skipped = (
        (captured.reset_sinks, captured.freeze_sinks) if reset_sinks else (captured.freeze_sinks, captured.reset_sinks)
    )
    applied.assert_called_once()
    skipped.assert_not_called()
    assert applied.call_args.args[0] is model, "the policy must land on the model that is returned"
    assert applied.call_args.args[1] == STUB_CONFIG, "against the frozen model's OWN config"


def _module_binds(path: pathlib.Path, name: str) -> bool:
    """Binding rather than a text match: function-local imports are banned, and a docstring mention is not a copy."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom | ast.Import) and any(
            alias.name == name or alias.asname == name for alias in node.names
        ):
            return True
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return True
    return False


@pytest.mark.parametrize("step", FROZEN_LOAD_STEPS)
def test_the_resolve_and_sinks_steps_live_only_in_their_owners(step):
    """Any module outside the owners that binds one of these steps is a new copy until proven otherwise."""
    roots = [REPO_ROOT / directory for directory in ("halo", "scripts")]
    assert all(root.is_dir() for root in roots), f"the sweep lost a root: {roots}"
    offenders = sorted(
        str(path.relative_to(REPO_ROOT))
        for root in roots
        for path in root.rglob("*.py")
        if str(path.relative_to(REPO_ROOT)) not in FROZEN_LOAD_STEP_OWNERS and _module_binds(path, step)
    )
    assert not offenders, (
        f"{offenders} bind '{step}' themselves. Resolving a backend and applying the sinks policy "
        f"are one decision — load_frozen_auxiliary_model owns it for every frozen reference/teacher, "
        f"and load_distributed_model owns it for the policy. A third copy is free to drift from both."
    )


def test_the_frozen_model_is_finalized_like_the_policy():
    """``save_pretrained`` refuses sampling params with ``do_sample=False``, which a re-saved teacher would hit."""
    with stub_frozen_loader() as caps:
        model = caps.auto_load.return_value
        model.generation_config = GenerationConfig(temperature=0.7, do_sample=False)
        load_frozen_auxiliary_model(AUX_MODEL, dtype=torch.bfloat16, trust_remote_code=False, download_tag=None)

    assert model.generation_config.do_sample is True


def test_reset_sinks_default_matches_the_policy_loaders():
    assert inspect.signature(load_frozen_auxiliary_model).parameters["reset_sinks"].default is True


def test_the_reference_requests_the_policy_s_attention_default():
    """Off Blackwell DPO/KTO request ``attn_default="sdpa"``. A reference sending ``None`` auto-detects a flash kernel,
    so every logratio differences two kernels' logprobs, which biases the objective instead of crashing."""
    with stub_frozen_loader() as caps:
        _preference_reference(reset_sinks=True, attn_default="sdpa")
    assert captured_load(caps).resolver["attn_implementation"] == "sdpa"

    # An explicit config still wins over the default, for the reference as for the policy.
    with stub_frozen_loader() as caps:
        _preference_reference(reset_sinks=True, attn_default="sdpa", attn_implementation="eager")
    assert captured_load(caps).resolver["attn_implementation"] == "eager"


def test_both_loads_of_a_preference_run_share_one_attention_default():
    """It defaults to None on both loaders, so passing it to only one silently reopens the split above."""
    tree = ast.parse(PREFERENCE_LOADS.read_text(encoding="utf-8"))
    (function,) = (node for node in ast.walk(tree) if getattr(node, "name", None) == "load_preference_models")
    script = PREFERENCE_LOADS
    defaults = {}
    for node in ast.walk(function):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            continue
        if node.func.id not in ("load_policy_model", "load_reference_model_for_preference"):
            continue
        passed = [kw.value for kw in node.keywords if kw.arg == "attn_default"]
        assert passed, f"{script.name}: {node.func.id} does not pass attn_default"
        defaults[node.func.id] = ast.dump(passed[0])

    assert len(defaults) == 2, f"{script.name}: expected both loaders, found {sorted(defaults)}"
    assert len(set(defaults.values())) == 1, (
        f"{script.name}: policy and reference are given different attention defaults: {defaults}"
    )


def test_vlm_load_pins_the_image_text_class_and_placement():
    """Resolving the class off the config drops the vision tower for a wrapper the mapping does not name."""
    with stub_frozen_loader() as caps:
        load_frozen_auxiliary_model(
            AUX_MODEL,
            dtype=torch.bfloat16,
            is_vlm=True,
            device_map={"": 2},
            download_tag="aux",
            trust_remote_code=False,
        )
    captured = captured_load(caps)
    assert captured.load["model_class"] is AutoModelForImageTextToText
    assert captured.load["device_map"] == {"": 2}
    captured.fetch_scope.assert_called_once_with(AUX_MODEL, None, tag="aux")


@pytest.mark.parametrize("is_vlm", [False, True], ids=["text", "vlm"])
def test_the_frozen_load_keeps_the_task_head(is_vlm):
    """The coverage gate excuses an added head because a run that trains it supplies it. This model trains nothing,
    so an absent head is random weights producing plausible numbers on one side of the objective."""
    with stub_frozen_loader() as caps:
        load_frozen_auxiliary_model(
            AUX_MODEL, dtype=torch.bfloat16, is_vlm=is_vlm, trust_remote_code=False, download_tag=None
        )
    assert captured_load(caps).load["excuse_task_head"] is False


def test_the_frozen_load_applies_the_remote_code_shims_before_the_config_fetch():
    """A teacher or reference preloaded ahead of the policy is the process's first remote-code load, and the config
    fetch alone can import a remote modeling file."""
    order = []
    with (
        stub_frozen_loader(),
        patch.object(frozen_models, "resolve_build_config", model_preparation.resolve_build_config),
        patch.object(model_preparation, "apply_remote_code_compat_shims", lambda: order.append("shims")),
        patch.object(model_preparation, "AutoConfig") as auto_config,
    ):
        auto_config.from_pretrained.side_effect = lambda *args, **kwargs: order.append("config") or Qwen3Config()
        load_frozen_auxiliary_model(AUX_MODEL, dtype=torch.bfloat16, trust_remote_code=False, download_tag=None)
    assert order == ["shims", "config"]


def test_the_frozen_load_warms_the_fa4_kernels_after_the_load():
    """Left cold, it pays a ~10s compile in a training step on one rank while its peers run into the next collective.
    The warm-up barriers, so it must follow the source resolution and the weight load, never sit inside either."""
    order: list = []

    with (
        stub_frozen_loader() as caps,
        patch.object(
            frozen_models,
            "warm_attention_kernels",
            lambda model, *, dtype: order.append(("warmup", dtype)),
        ),
    ):
        caps.fetch_scope.side_effect = lambda path, revision, *, tag: order.append(("resolve", tag)) or ModelSource(
            path, revision, path
        )
        caps.auto_load.side_effect = lambda *args, **kwargs: order.append("weights") or caps.auto_load.return_value
        load_frozen_auxiliary_model(AUX_MODEL, dtype=torch.bfloat16, download_tag="teacher", trust_remote_code=False)

    assert order == [("resolve", "teacher"), "weights", ("warmup", torch.bfloat16)]


def test_an_untagged_fetch_opens_no_coordination_scope():
    """``fs_aware_main_first`` needs every rank to reach a tag equally often, so a scope on a partial load hangs."""
    with stub_frozen_loader() as caps:
        load_frozen_auxiliary_model(AUX_MODEL, dtype=torch.bfloat16, trust_remote_code=False, download_tag=None)
    caps.fetch_scope.assert_not_called()


def test_the_on_policy_reference_load_checks_its_fit_then_warns_before_loading(monkeypatch):
    order = []
    monkeypatch.setattr(frozen_models, "reject_unfit_reference", lambda *args, **kwargs: order.append("fit"))
    monkeypatch.setattr(frozen_models, "warn_unparallelized_reference", lambda *args: order.append("warn"))
    monkeypatch.setattr(frozen_models, "load_frozen_reference_model", lambda *args, **kwargs: order.append("load"))

    _on_policy_grpo_reference(reset_sinks=True)

    assert order == ["fit", "warn", "load"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
