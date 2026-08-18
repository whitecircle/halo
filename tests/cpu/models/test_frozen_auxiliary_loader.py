#!/usr/bin/env python
"""Consolidation contract for the frozen auxiliary models: one loader, one sinks policy.

The DPO/KTO reference, the SDPG KL anchor and the distillation teacher all load an unparallelized
frozen model whose logprobs are the other half of the objective. A per-site copy of
that load buys a silent numerical bug the moment it drifts — a pin read off an object that cannot
carry it (a "pinned" teacher on hub ``main``), or a backend resolved under ``sinks_reset=True``
whose sink reset is then never applied (a GptOss teacher running sdpa over live sinks, every
logprob shifted by nats). Both are "this copy is missing a step the others have".

These tests pin the property that removes that class of bug: every call site reaches
``load_frozen_auxiliary_model``, the sinks policy is applied there rather than by each caller, and no
module outside the two owning loaders binds those steps at all.

Run: python tests/cpu/models/test_frozen_auxiliary_loader.py  (or pytest)
"""

import ast
import contextlib
import inspect
import pathlib
import sys
import types
from unittest.mock import patch

import pytest
import torch
from accelerate import PartialState
from transformers import AutoModelForImageTextToText, GenerationConfig
from trl import ModelConfig

import scripts.training.distillation.self_distill as self_distill_script
import scripts.training.distillation.teacher_distill as distill_script
import src.distributed.loading.frozen_models as frozen_models
from src.distributed.loading.frozen_models import load_frozen_auxiliary_model, load_reference_model_for_preference
from src.distributed.parallelism_config import ParallelismConfig
from tests.common.frozen_loader import STUB_CONFIG, STUB_RESOLVED_ATTN, captured_load, stub_frozen_loader

# The SDPG reference load logs through accelerate's logger, which requires an initialized state.
PartialState()

AUX_MODEL = "org/auxiliary"

# The steps no call site may own itself: resolving the backend and applying the sinks policy are one
# decision (``sinks_reset=True`` is what APPROVES a sink-dropping backend), so a site that binds
# either of them again owns a second copy of it, free to drift from the loaders that own the first.
# The two sink primitives are not swept: only their definer binds them now, and every caller reaches
# them through ``apply_sinks_policy`` — which is what a drifting copy would have to bind.
FROZEN_LOAD_STEPS = ("resolve_attn_implementation", "apply_sinks_policy")

# Allowlist, not a list of known offenders: the sweep below is over the whole tree, so a new copy is
# caught by default. attention.py defines the resolver and gpt_oss_sinks.py the policy;
# model_preparation.py owns ``finalize_run_model``, the one post-load application BOTH loaders run;
# frozen_models.py is the frozen (unparallelized) loader; model_loading.py is the POLICY loader,
# re-running the resolver AFTER the parallelism-specific overrides (CP's flex→FA switch, EP's
# flex-compile disable); embedding.py applies the policy to the backbone SentenceTransformer returns;
# tool_io.py, patch_vocab.py and reset_sinks.py apply a RECORDED or explicit policy to a
# checkpoint on disk — reset_sinks.py exists to apply exactly one, so it reaches the shared seam
# rather than filling ``.sinks`` by name behind it.
FROZEN_LOAD_STEP_OWNERS = frozenset(
    {
        "src/models/patches/attention.py",
        "src/models/patches/gpt_oss_sinks.py",
        "src/models/loading/model_preparation.py",
        "src/distributed/loading/frozen_models.py",
        "src/distributed/loading/model_loading.py",
        "scripts/training/embedding.py",
        "src/checkpoint/tool_io.py",
        "scripts/before_training/patch_vocab.py",
        "scripts/after_training/reset_sinks.py",
    }
)

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]


def _preference_reference(*, reset_sinks, attn_default=None, attn_implementation=None):
    """The DPO/KTO full-finetune reference load, driven through its real call site."""
    args = types.SimpleNamespace(
        eos_token=None,
        bos_token=None,
        pad_token=None,
        chat_template=None,
        force_chat_template=False,
        added_special_tokens=None,
        tokenizer_backend="hf",
    )
    # Every ``<special>_token_id`` accompanies its token: ``PreTrainedTokenizerBase.__getattr__``
    # resolves them, so no real tokenizer can carry one without the other, and the setup seam the
    # loader runs records the pad id on the model.
    tokenizer = types.SimpleNamespace(
        eos_token="<e>",
        eos_token_id=1,
        bos_token="<b>",
        bos_token_id=2,
        pad_token="<p>",
        pad_token_id=3,
        chat_template="{{ messages }}",
    )
    return load_reference_model_for_preference(
        args,
        ModelConfig(model_name_or_path=AUX_MODEL, attn_implementation=attn_implementation),
        types.SimpleNamespace(bf16=True, fp16=False, precompute_ref_log_probs=False),
        ParallelismConfig(),
        tokenizer,
        is_vlm=False,
        method="DPO",
        reset_sinks=reset_sinks,
        attn_default=attn_default,
    )


def _script_teacher(*, reset_sinks):
    """The shipped distillation launch path (the script preloads the teacher for the trainer)."""
    return distill_script._load_distill_teacher(
        args=types.SimpleNamespace(teacher_model=AUX_MODEL, teacher_model_revision=None),
        model_config=types.SimpleNamespace(attn_implementation=None, trust_remote_code=False),
        training_config=types.SimpleNamespace(bf16=True, fp16=False),
        dist_args=types.SimpleNamespace(reset_sinks=reset_sinks),
        is_vlm=False,
        local_rank=0,
    )


def _sdpg_reference(*, reset_sinks):
    """The SDPG self-distillation ``L_ref`` KL anchor."""
    return self_distill_script._load_sdpg_reference(
        args=types.SimpleNamespace(reference_kl_coef=0.1, reference_model_name_or_path=AUX_MODEL),
        model_config=ModelConfig(model_name_or_path=AUX_MODEL),
        sft_config=types.SimpleNamespace(bf16=True, fp16=False),
        dist_args=types.SimpleNamespace(reset_sinks=reset_sinks),
        is_vlm=False,
    )


FROZEN_LOAD_SITES = {
    "preference-reference": _preference_reference,
    "distillation-script": _script_teacher,
    "sdpg-reference": _sdpg_reference,
}

# The scripts that load a policy AND a live reference, i.e. the ones whose two loads must agree.
PREFERENCE_SCRIPTS = tuple(
    pathlib.Path(__file__).resolve().parents[3] / "scripts/training/preference" / f"{name}.py"
    for name in ("dpo", "kto")
)


@pytest.mark.parametrize("site", FROZEN_LOAD_SITES.values(), ids=list(FROZEN_LOAD_SITES))
def test_every_frozen_load_site_reaches_the_shared_loader(site):
    """Only ``load_frozen_auxiliary_model``'s namespace is stubbed here.

    A call site that fetched through its own ``AutoConfig``/``auto_load_model`` would miss these stubs
    entirely — it would reach the hub for a repo that does not exist, and record no fetch — so this is
    a routing check, not a smoke test.
    """
    with stub_frozen_loader() as caps:
        model = site(reset_sinks=True)
    captured = captured_load(caps)
    assert model is captured.model
    assert captured.load_positional[0] == AUX_MODEL
    # Whatever pin a site resolves must reach BOTH fetches — a pinned checkpoint paired with
    # hub-main's config is its own way of loading the wrong model.
    assert captured.config["revision"] == captured.load["revision"]
    assert captured.load["attn_implementation"] == STUB_RESOLVED_ATTN
    assert captured.load["dtype"] is torch.bfloat16


@pytest.mark.parametrize("reset_sinks", [True, False])
@pytest.mark.parametrize("site", FROZEN_LOAD_SITES.values(), ids=list(FROZEN_LOAD_SITES))
def test_every_frozen_load_site_gets_the_sinks_policy(site, reset_sinks):
    """The policy the backend was resolved UNDER must land on the model every site returns.

    ``resolve_attn_implementation(..., sinks_reset=True)`` approves sdpa for a GptOss model only on
    the premise that the caller then neutralizes the sinks; ``sinks_reset=False`` keeps them live and
    they must be frozen instead. Skipping the pairing shifts every logprob of the frozen model by nats
    against the policy it scores, with nothing logged.
    """
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
    """Whether a module imports or defines ``name``.

    Binding is the check rather than a text match: CLAUDE.md bans function-local imports, so a
    module-level import is the only way a call site can reach one of these, while a prose mention in
    a docstring (the CP GptOss layer explains when the reset runs relative to its wrap) is not a copy.
    """
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
    """Sweep the whole tree, not a list of the copies we happen to know about.

    Drift takes the form of a copy that looks finished — a pin read off an object that cannot carry
    it, a backend resolved under a premise never applied. A hand-enumerated call-site list misses
    each such copy at the moment it is written, so the assertion is inverted: any module outside the
    owners that binds one of these steps is a new copy until proven otherwise.
    """
    offenders = sorted(
        str(path.relative_to(REPO_ROOT))
        for directory in ("src", "scripts")
        for path in (REPO_ROOT / directory).rglob("*.py")
        if str(path.relative_to(REPO_ROOT)) not in FROZEN_LOAD_STEP_OWNERS and _module_binds(path, step)
    )
    assert not offenders, (
        f"{offenders} bind '{step}' themselves. Resolving a backend and applying the sinks policy "
        f"are one decision — load_frozen_auxiliary_model owns it for every frozen reference/teacher, "
        f"and load_distributed_model owns it for the policy. A third copy is free to drift from both."
    )


def test_the_frozen_model_is_finalized_like_the_policy():
    """Both loaders end in ``finalize_run_model``, so the frozen copy carries the same fixups.

    ``sanitize_generation_config`` is the one the frozen path did NOT run: a config shipping sampling
    params with ``do_sample=False`` is what ``save_pretrained`` refuses, and a teacher exported or
    re-saved by a tool would have hit it while the policy never does.
    """
    with stub_frozen_loader() as caps:
        model = caps.auto_load.return_value
        model.generation_config = GenerationConfig(temperature=0.7, do_sample=False)
        load_frozen_auxiliary_model(AUX_MODEL, dtype=torch.bfloat16)

    assert model.generation_config.do_sample is True


def test_reset_sinks_default_matches_the_policy_loaders():
    """A caller that omits the policy gets the off-policy fine-tuning one, like every sibling loader."""
    assert inspect.signature(load_frozen_auxiliary_model).parameters["reset_sinks"].default is True


def test_the_reference_requests_the_policy_s_attention_default():
    """The reference must inherit the POLICY's fallback backend, not auto-detect its own.

    DPO/KTO pin the policy to ``attn_default="sdpa"`` (padded batches run FA4's slow varlen path).
    A reference that passes only ``model_config.attn_implementation`` sends ``None`` — the request
    that means "auto-detect" — and lands on flash_attention_4 on Blackwell. Every logratio is then
    a difference of logprobs computed by two different kernels, which biases the objective instead
    of crashing. The unset case is the common one: no shipped preference YAML sets the field.
    """
    with stub_frozen_loader() as caps:
        _preference_reference(reset_sinks=True, attn_default="sdpa")
    assert captured_load(caps).resolver["attn_implementation"] == "sdpa"

    # An explicit config still wins over the default, for the reference as for the policy.
    with stub_frozen_loader() as caps:
        _preference_reference(reset_sinks=True, attn_default="sdpa", attn_implementation="eager")
    assert captured_load(caps).resolver["attn_implementation"] == "eager"


@pytest.mark.parametrize("script", PREFERENCE_SCRIPTS, ids=lambda p: p.stem)
def test_both_loads_in_a_preference_script_share_one_attention_default(script):
    """``attn_default`` defaults to None on both loaders, so a call site that passes it to only one
    silently reopens the split above. Pin that each script hands both the SAME expression."""
    tree = ast.parse(script.read_text(encoding="utf-8"))
    defaults = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            continue
        if node.func.id not in ("load_model_for_training", "load_reference_model_for_preference"):
            continue
        passed = [kw.value for kw in node.keywords if kw.arg == "attn_default"]
        assert passed, f"{script.name}: {node.func.id} does not pass attn_default"
        defaults[node.func.id] = ast.dump(passed[0])

    assert len(defaults) == 2, f"{script.name}: expected both loaders, found {sorted(defaults)}"
    assert len(set(defaults.values())) == 1, (
        f"{script.name}: policy and reference are given different attention defaults: {defaults}"
    )


def test_vlm_load_pins_the_image_text_class_and_placement():
    """``is_vlm`` must pin ``AutoModelForImageTextToText``: resolving the class off the config drops
    the vision tower for a wrapper the mapping does not name."""
    with stub_frozen_loader() as caps:
        load_frozen_auxiliary_model(
            AUX_MODEL, dtype=torch.bfloat16, is_vlm=True, device_map={"": 2}, download_tag="aux"
        )
    captured = captured_load(caps, is_vlm=True)
    assert captured.load_positional[0] is AutoModelForImageTextToText
    assert captured.load["device_map"] == {"": 2}
    captured.fetch_scope.assert_called_once_with("aux")


@pytest.mark.parametrize("is_vlm", [False, True], ids=["text", "vlm"])
def test_the_frozen_load_keeps_the_task_head(is_vlm):
    """The coverage gate excuses whatever the architecture adds on top of a base checkpoint, because
    a run that TRAINS that head supplies it. This model never trains anything — every tensor it
    carries is consumed to score the policy — so an absent head is not an excusable absence but a
    randomly initialized one producing plausible numbers on one side of the objective. The reward
    scorers already load this way (``scripts/inference/reward_model/_common.py``).
    """
    with stub_frozen_loader() as caps:
        load_frozen_auxiliary_model(AUX_MODEL, dtype=torch.bfloat16, is_vlm=is_vlm)
    assert captured_load(caps, is_vlm=is_vlm).load["excuse_task_head"] is False


def test_the_frozen_load_applies_the_remote_code_shims_before_the_config_fetch():
    """A teacher or reference preloaded ahead of the policy is the first remote-code load of the
    process, so it cannot inherit the shims the policy loader applies — and the config fetch is
    already enough to import a remote modeling file, so the order matters as much as the call.
    """
    order = []
    with (
        stub_frozen_loader() as caps,
        patch.object(frozen_models, "apply_remote_code_compat_shims", lambda: order.append("shims")),
    ):
        caps.auto_config.from_pretrained.side_effect = lambda *args, **kwargs: order.append("config") or STUB_CONFIG
        load_frozen_auxiliary_model(AUX_MODEL, dtype=torch.bfloat16)
    assert order == ["shims", "config"]


def test_the_frozen_load_warms_the_fa4_kernels_outside_the_coordination_scope():
    """This model's first scoring forward must not be the one that JIT-compiles FA4.

    The policy loader warms them; a frozen teacher/reference left cold pays a ~10s compile inside a
    training step, on whichever rank reaches it first, while its peers run ahead into the next
    collective — the desync the warm-up exists to prevent, moved onto the auxiliary model. The
    warm-up barriers, so it must sit OUTSIDE the main-first block, where the peers sit on a store
    key rather than in a matching collective.
    """
    order: list = []

    @contextlib.contextmanager
    def _recording_scope(*args, **kwargs):
        order.append("scope-enter")
        yield
        order.append("scope-exit")

    with (
        stub_frozen_loader() as caps,
        patch.object(
            frozen_models,
            "warm_attention_kernels",
            lambda model, *, dtype: order.append(("warmup", dtype)),
        ),
    ):
        caps.fetch_scope.side_effect = _recording_scope
        load_frozen_auxiliary_model(AUX_MODEL, dtype=torch.bfloat16, download_tag="teacher")

    assert order == ["scope-enter", "scope-exit", ("warmup", torch.bfloat16)]


def test_an_untagged_fetch_opens_no_coordination_scope():
    """``fs_aware_main_first`` demands every rank reach a tag equally often — an unconditional scope
    around a load only some ranks perform would hang the ones that do."""
    with stub_frozen_loader() as caps:
        load_frozen_auxiliary_model(AUX_MODEL, dtype=torch.bfloat16)
    caps.fetch_scope.assert_not_called()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
