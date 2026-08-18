#!/usr/bin/env python
"""Contracts every training entry script must uphold.

- Loader-kwarg threading: any module calling ``load_distributed_model`` (the scripts, and the
  shared ``script_runner.load_script_model`` they route through) must thread
  ``revision=`` and ``quantization_config=`` (or explicitly reject quantization via a
  ``get_quantization_config`` guard, like embedding.py). A missing kwarg silently loads hub
  ``main`` instead of a pinned ``model_revision``, or trains full-precision under a QLoRA YAML.
  The script set is derived from the ``scripts/training`` tree (via the ``halo`` CLI index), so
  new scripts are covered without touching this file.
- Preprocessed completion-masking: ``sft.py`` must reject a preprocessed dataset whose baked
  ``train_on_completions_only`` disagrees with the runtime flag (labels are baked at prep time).
- Fail-loud rejection of silently-ignored fields: ``self_distill.py`` rejects
  ``train_on_last_assistant_only``.
- Collator selection: no script may construct a completion-only collator directly — the
  null-marker raise and ``verify_marker_renders_in_chat_template`` live in ``select_data_collator``,
  and a direct build trains on every token while the log still says completion-only masking.
- PEFT wrapping: no script may call bare ``get_peft_model`` — self-wrapping scripts
  (``teacher_distill.py``) must route through ``prepare_peft_model`` (k-bit prep).

Run: pytest tests/cpu/config/test_training_script_contracts.py
"""

import ast
import sys
from pathlib import Path
from unittest import mock

import pytest

from src.cli import training_methods
from src.data.collators import completions_only
from tests.common.utils import load_script_module

_REPO_ROOT = Path(__file__).resolve().parents[3]
_TRAINING_DIR = _REPO_ROOT / "scripts" / "training"

_TRAINING_SCRIPTS = sorted(set(training_methods(_REPO_ROOT).values()))


def _load_script_module(rel_path: str):
    name = "halo_test_script_" + rel_path.replace("/", "_").removesuffix(".py")
    return load_script_module(f"scripts/training/{rel_path}", name)


def _named_calls(tree: ast.AST, func_name: str) -> list[ast.Call]:
    calls = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)
            if name == func_name:
                calls.append(node)
    return calls


# Loader-kwarg threading (revision + quantization_config)


# Every training script now loads through ``script_runner.load_script_model`` or
# ``vlm_setup.load_model_for_training``, both of which route the one ``load_distributed_model`` call
# through ``vlm_setup.load_model_consuming_init_kwargs``; a script may still call the loader directly,
# so all are held to the same contract. The shared helper is listed explicitly because a vacuous
# parametrization (all scripts skipping) would silently stop testing anything —
# ``test_loader_contract_is_not_vacuous`` pins that.
_LOADER_CALL_SITES = [*_TRAINING_SCRIPTS, _REPO_ROOT / "src" / "distributed" / "loading" / "vlm_setup.py"]


def _threads_loader_kwargs(script: Path) -> bool:
    """Whether ``script`` calls ``load_distributed_model`` (asserting the contract on each call)."""
    source = script.read_text(encoding="utf-8")
    calls = _named_calls(ast.parse(source), "load_distributed_model")
    for call in calls:
        kwargs = {kw.arg for kw in call.keywords}
        assert "revision" in kwargs, (
            f"{script}: load_distributed_model call does not thread revision= — a pinned "
            f"model_revision would silently load hub main."
        )
        assert "quantization_config" in kwargs or "get_quantization_config" in source, (
            f"{script}: load_distributed_model call neither threads quantization_config= nor "
            f"rejects quantization via a get_quantization_config guard — a QLoRA YAML would "
            f"silently train full-precision."
        )
    return bool(calls)


def test_loader_contract_is_not_vacuous():
    """At least one call site must exist, else the parametrized test below passes by skipping."""
    threading = [p for p in _LOADER_CALL_SITES if _threads_loader_kwargs(p)]
    assert threading, (
        "No load_distributed_model call site found across the training scripts or "
        "src/distributed/loading/vlm_setup.py — the loader-kwarg contract is no longer tested. "
        "Add the module that now owns the load."
    )


@pytest.mark.parametrize(
    "script",
    _LOADER_CALL_SITES,
    ids=[str(p.relative_to(_REPO_ROOT)) for p in _LOADER_CALL_SITES],
)
def test_load_distributed_model_threads_revision_and_quantization(script: Path):
    source = script.read_text(encoding="utf-8")
    calls = _named_calls(ast.parse(source), "load_distributed_model")
    if not calls:
        pytest.skip("module does not call load_distributed_model directly")
    for call in calls:
        kwargs = {kw.arg for kw in call.keywords}
        assert "revision" in kwargs, (
            f"{script}: load_distributed_model call does not thread revision= — a pinned "
            f"model_revision would silently load hub main."
        )
        assert "quantization_config" in kwargs or "get_quantization_config" in source, (
            f"{script}: load_distributed_model call neither threads quantization_config= nor "
            f"rejects quantization via a get_quantization_config guard — a QLoRA YAML would "
            f"silently train full-precision."
        )


def test_training_script_index_is_nonempty():
    """The parametrization source itself must not silently go empty."""
    assert len(_TRAINING_SCRIPTS) >= 10


# Modality detection must read the SAME commit the weights load from

_VLM_PROBE_CALL_SITES = [
    *_TRAINING_SCRIPTS,
    _REPO_ROOT / "src" / "distributed" / "loading" / "vlm_setup.py",
]


def _vlm_probe_calls(path: Path) -> list[ast.Call]:
    """Both entry points into the modality probe — ``is_vlm_run`` forwards its pin to
    ``is_vlm_model``, so an unpinned call to either loads hub main's config."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return _named_calls(tree, "is_vlm_model") + _named_calls(tree, "is_vlm_run")


def test_vlm_probe_contract_is_not_vacuous():
    """At least one probe must exist, else the parametrized test below passes by skipping."""
    assert [p for p in _VLM_PROBE_CALL_SITES if _vlm_probe_calls(p)], (
        "No is_vlm_model call site found — the revision-pinning contract is no longer tested."
    )


@pytest.mark.parametrize(
    "path",
    _VLM_PROBE_CALL_SITES,
    ids=[str(p.relative_to(_REPO_ROOT)) for p in _VLM_PROBE_CALL_SITES],
)
def test_is_vlm_model_probe_is_revision_pinned(path: Path):
    """Every probe must pin the revision (or be handed an already-loaded config).

    ``is_vlm_model`` fetches ``AutoConfig`` when given no config, and hub ``main`` can name a
    different modality — or a different tensor namespace — than the commit ``model_revision`` pins
    (Zaya). The answer picks the ``Auto*`` class, the processing class, the collator and the run
    prefix, so an unpinned probe routes a pinned run down the wrong path with no error.
    """
    calls = _vlm_probe_calls(path)
    if not calls:
        pytest.skip("module does not probe the modality")
    for call in calls:
        kwargs = {kw.arg for kw in call.keywords}
        assert "revision" in kwargs or "config" in kwargs, (
            f"{path}: modality probe pins neither revision= nor an already-loaded config= — "
            f"a model_revision-pinned run would be routed by hub main's config."
        )


# sft.py: preprocessed completion-masking mismatch


@pytest.fixture(scope="module")
def metadata_cls():
    from src.data.pipeline.preprocessed_metadata import PreprocessedDatasetMetadata

    return PreprocessedDatasetMetadata


def _validate_masking(metadata_cls, baked: bool | None, runtime: bool):
    from src.data.pipeline.preprocessed_metadata import validate_preprocessing_compatibility

    config = {} if baked is None else {"train_on_completions_only": baked}
    metadata = metadata_cls(max_length=4096, train_on_completions_only=bool(baked), config=config)
    validate_preprocessing_compatibility(
        metadata, required_max_length=4096, required_train_on_completions_only=runtime
    )


def test_preprocessed_completion_masking_mismatch_raises(metadata_cls):
    with pytest.raises(ValueError, match="train_on_completions_only"):
        _validate_masking(metadata_cls, baked=False, runtime=True)


def test_preprocessed_completion_masking_reverse_mismatch_raises(metadata_cls):
    with pytest.raises(ValueError, match="train_on_completions_only"):
        _validate_masking(metadata_cls, baked=True, runtime=False)


def test_preprocessed_completion_masking_match_passes(metadata_cls):
    _validate_masking(metadata_cls, baked=True, runtime=True)


def test_preprocessed_completion_masking_old_metadata_warns_not_crashes(metadata_cls):
    # Metadata written before the field existed carries no signal — must not raise.
    _validate_masking(metadata_cls, baked=None, runtime=True)


def test_sft_forwards_the_runtime_masking_flag_to_the_preprocessed_check():
    """The cases above drive the leaf helper; this pins the wiring that reaches it.

    ``required_train_on_completions_only=None`` makes the check skip entirely, so dropping the
    kwarg from either call site turns a baked/runtime mismatch back into a silent wrong-masking run
    — and none of the helper-level cases above would notice.
    """
    tree = ast.parse((_TRAINING_DIR / "sft.py").read_text(encoding="utf-8"))
    calls = _named_calls(tree, "validate_preprocessing_compatibility")
    assert len(calls) == 2, f"sft.py makes {len(calls)} preprocessed-metadata checks; expected the text + VLM paths"
    for call in calls:
        value = {kw.arg: kw.value for kw in call.keywords}.get("required_train_on_completions_only")
        assert isinstance(value, ast.Attribute), (
            "sft.py must pass the live args.train_on_completions_only; a literal (or an omitted "
            "kwarg) degrades the baked/runtime mismatch check to a no-op."
        )


def test_sft_forwards_render_args_and_packing_to_the_preprocessed_check():
    """``render_args=None`` / ``required_packing=None`` skip those checks entirely, so dropping
    either kwarg from a call site silently reverts render-knob verification (baked rows vs YAML)
    and the unpacked-artifact-with-packing warning."""
    tree = ast.parse((_TRAINING_DIR / "sft.py").read_text(encoding="utf-8"))
    calls = _named_calls(tree, "validate_preprocessing_compatibility")
    assert len(calls) == 2, f"sft.py makes {len(calls)} preprocessed-metadata checks; expected the text + VLM paths"
    for call in calls:
        kwargs = {kw.arg: kw.value for kw in call.keywords}
        assert isinstance(kwargs.get("render_args"), ast.Name), (
            "sft.py must pass the live script args as render_args; omitting it skips the baked-render verification."
        )
        assert isinstance(kwargs.get("required_packing"), ast.Attribute), (
            "sft.py must pass the live sft_config.packing; omitting it skips the unpacked-artifact warning."
        )


# Fail-loud rejection of silently-ignored fields


def _run_main_with_yaml(module, yaml_body: str, tmp_path: Path):
    config = tmp_path / "config.yaml"
    config.write_text(
        # bf16/use_cpu keep TrainingArguments constructible on a GPU-less test runner.
        f"model_name_or_path: dummy/model\noutput_dir: {tmp_path / 'out'}\nbf16: false\nuse_cpu: true\n{yaml_body}"
    )
    # The log tee redirects the process's stdout/stderr fds — keep it out of the test process.
    with (
        mock.patch("src.training.parser.install_log_tee"),
        mock.patch.object(sys, "argv", ["prog", str(config)]),
    ):
        module.main()


def test_self_distill_rejects_train_on_last_assistant_only(tmp_path):
    module = _load_script_module("distillation/self_distill.py")
    with pytest.raises(ValueError, match="train_on_last_assistant_only"):
        _run_main_with_yaml(module, "train_on_last_assistant_only: true\n", tmp_path)


# PEFT wrapping: scripts must route through prepare_peft_model (k-bit prep)


@pytest.mark.parametrize(
    "script",
    _TRAINING_SCRIPTS,
    ids=[str(p.relative_to(_TRAINING_DIR)) for p in _TRAINING_SCRIPTS],
)
def test_no_script_calls_bare_get_peft_model(script: Path):
    """A bare ``get_peft_model`` skips ``prepare_peft_model``'s k-bit prep (layernorm fp32 upcast,
    gradient-checkpointing input hooks) that every sibling trainer applies — QLoRA would silently
    mis-train. Scripts that wrap PEFT themselves must call ``prepare_peft_model``."""
    source = script.read_text(encoding="utf-8")
    assert not _named_calls(ast.parse(source), "get_peft_model"), (
        f"{script}: calls bare get_peft_model — route through prepare_peft_model "
        f"(src/distributed/loading/model_loading.py) like SMPO/classification/offline-GRPO."
    )


def test_teacher_distill_wraps_peft_via_prepare_peft_model():
    """Keeps the blanket get_peft_model ban above non-vacuous: teacher_distill.py wraps the student
    itself (plain Trainer, no peft_config kwarg) and must do so through prepare_peft_model."""
    source = (_TRAINING_DIR / "distillation/teacher_distill.py").read_text(encoding="utf-8")
    assert _named_calls(ast.parse(source), "prepare_peft_model"), (
        "teacher_distill.py no longer calls prepare_peft_model — its PEFT wrap lost the k-bit prep."
    )


_TOKENIZER_SEAMS = ("apply_max_length", "apply_prompt_completion_window", "setup_model_and_tokenizer")

# SentenceTransformers owns its own tokenizer pipeline end to end, so embedding.py resolves the
# length against the ST transformer module instead of a toolkit seam. Listed rather than skipped so
# a script that quietly stops calling a seam cannot hide here.
_NO_TOKENIZER_SEAM = frozenset({"embedding.py"})


def _assigned_targets(tree: ast.AST, call: ast.Call) -> list[ast.expr] | None:
    """Targets of the Assign whose value is ``call``, or None when the result is discarded."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and node.value is call:
            return node.targets
        if isinstance(node, ast.AnnAssign) and node.value is call:
            return [node.target]
    return None


@pytest.mark.parametrize(
    "script",
    _TRAINING_SCRIPTS,
    ids=[str(p.relative_to(_TRAINING_DIR)) for p in _TRAINING_SCRIPTS],
)
def test_scripts_bind_the_tokenizer_their_length_seam_returns(script: Path):
    """The length seams RESOLVE the tokenizer (``tokenizer_backend`` swaps in a gigatoken proxy) and
    return it. Dropping the return value leaves the script holding the pre-resolution object, so the
    knob parses, validates, logs — and does nothing. That is invisible at runtime, which is how one
    dropped return sits unnoticed on three GRPO scripts at once.

    ``apply_prompt_completion_window`` returns ``(tokenizer, window)``, so a tuple target counts.
    """
    tree = ast.parse(script.read_text(encoding="utf-8"))
    seen = 0
    for seam in _TOKENIZER_SEAMS:
        for call in _named_calls(tree, seam):
            seen += 1
            targets = _assigned_targets(tree, call)
            assert targets is not None, (
                f"{script}: {seam}() result is discarded — tokenizer_backend is a silent no-op there."
            )
            names = {n.id for t in targets for n in ast.walk(t) if isinstance(n, ast.Name)}
            assert "tokenizer" in names, (
                f"{script}: {seam}() result is bound to {sorted(names)}, not `tokenizer` — the "
                f"backend-resolved tokenizer never reaches the trainer."
            )
    if script.name in _NO_TOKENIZER_SEAM:
        assert not seen, f"{script}: now calls a length seam — drop it from _NO_TOKENIZER_SEAM."
    else:
        assert seen, f"{script}: calls no tokenizer length seam ({', '.join(_TOKENIZER_SEAMS)})."


_INSTALLS_PROCESSING_CLASS = (
    "sft.py",
    "preference/dpo.py",
    "preference/kto.py",
    "preference/smpo.py",
    "distillation/self_distill.py",
    "distillation/teacher_distill.py",
)


@pytest.mark.parametrize("rel_path", _INSTALLS_PROCESSING_CLASS)
def test_vlm_capable_scripts_install_the_resolved_tokenizer(rel_path: str):
    """A VLM processor holds its own inner tokenizer, so binding `tokenizer` is not enough — the
    resolved object must be installed into the trainer's ``processing_class`` on both the VLM and the
    text branch. ``install_resolved_tokenizer`` is the one seam that does it; a script that stops
    calling it silently hands the trainer the pre-resolution tokenizer.
    """
    tree = ast.parse((_TRAINING_DIR / rel_path).read_text(encoding="utf-8"))
    calls = _named_calls(tree, "install_resolved_tokenizer")
    assert calls, f"{rel_path}: never calls install_resolved_tokenizer — processing_class is unresolved."
    for call in calls:
        targets = _assigned_targets(tree, call)
        names = {n.id for t in (targets or []) for n in ast.walk(t) if isinstance(n, ast.Name)}
        assert "processing_class" in names, (
            f"{rel_path}: install_resolved_tokenizer() result is bound to {sorted(names)}, "
            f"not `processing_class` — the resolved tokenizer never reaches the trainer."
        )


# Collator selection: the completion-masking guards live in the factory


_COMPLETION_COLLATORS = sorted(
    name
    for name, obj in vars(completions_only).items()
    if isinstance(obj, type) and obj.__module__ == completions_only.__name__
)


def test_the_completion_collator_roster_is_derived_and_nonempty():
    assert "DataCollatorForCompletionOnlyLM" in _COMPLETION_COLLATORS


@pytest.mark.parametrize(
    "script",
    _TRAINING_SCRIPTS,
    ids=[str(p.relative_to(_TRAINING_DIR)) for p in _TRAINING_SCRIPTS],
)
def test_no_script_builds_a_completion_only_collator_directly(script: Path):
    """``select_data_collator`` owns the guards a completion-only collator is unusable without: the
    template must exist, and the marker must actually render in the chat template. Built directly,
    a marker no template emits masks every token out and the run trains on nothing — with the same
    log line a correct run prints."""
    tree = ast.parse(script.read_text(encoding="utf-8"))
    direct = sorted(
        {
            node.func.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _COMPLETION_COLLATORS
        }
    )
    assert not direct, (
        f"{script.relative_to(_TRAINING_DIR)}: constructs {direct} directly, bypassing "
        f"select_data_collator's assistant-marker guards."
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
