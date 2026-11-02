#!/usr/bin/env python
"""Contracts every training entry script must uphold.

- Loader-kwarg threading: any module calling ``load_distributed_model`` (the scripts, and the
  shared ``model_setup.load_model_and_tokenizer`` they route through) must thread
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
- PEFT wrapping: no script may call bare ``get_peft_model`` — a self-wrapping script must route
  through ``build_peft_model`` (k-bit prep) — and none may call bare ``get_peft_config``, which
  would skip ``build_peft_config``'s exclusion of target-name matches PEFT cannot adapt.
- TRL's dataset preparation: a script building a trainer derived from TRL's ``SFTTrainer`` runs
  ``disable_trl_dataset_prep`` on the config it hands over, before the build — TRL refuses the
  script's packing or padding-free collator under its own ``packing`` / ``padding_free``.

Run: pytest tests/cpu/config/test_training_script_contracts.py
"""

import ast
import importlib
import sys
from pathlib import Path
from unittest import mock

import pytest
from trl import SFTTrainer

from halo.cli import training_methods
from halo.data.collators import padded
from halo.data.preprocessed.metadata import PreprocessedDatasetMetadata, verify_preprocessed_dataset
from tests.common.repo import REPO_ROOT, load_script_module

_TRAINING_DIR = REPO_ROOT / "scripts" / "training"

_TRAINING_SCRIPTS = sorted(set(training_methods(REPO_ROOT).values()))


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


def _seam_trees(script: Path) -> list[ast.AST]:
    """The script's tree, plus the shared preference load when the script delegates its tokenizer seams to
    it, so a seam the shared load drops is charged to every script that calls it."""
    tree = ast.parse(script.read_text(encoding="utf-8"))
    if not _named_calls(tree, "load_preference_models"):
        return [tree]
    shared = ast.parse((REPO_ROOT / "halo/entrypoints/model_setup.py").read_text(encoding="utf-8"))
    (load,) = (node for node in shared.body if getattr(node, "name", None) == "load_preference_models")
    return [tree, load]


# Loader-kwarg threading (revision + quantization_config)


# Every training script loads through ``model_setup.load_model_and_tokenizer``, directly or through
# ``model_setup.load_policy_model``, and so makes the one ``load_distributed_model`` call there; a script may still call
# the loader directly, so all are held to the same contract. The shared helper is listed explicitly because a vacuous
# parametrization (all scripts skipping) would silently stop testing anything — ``test_loader_contract_is_not_vacuous``
# pins that.
_LOADER_CALL_SITES = [*_TRAINING_SCRIPTS, REPO_ROOT / "halo" / "entrypoints" / "model_setup.py"]


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
        "halo/entrypoints/model_setup.py — the loader-kwarg contract is no longer tested. "
        "Add the module that now owns the load."
    )


@pytest.mark.parametrize(
    "script",
    _LOADER_CALL_SITES,
    ids=[str(p.relative_to(REPO_ROOT)) for p in _LOADER_CALL_SITES],
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

# data_setup.py holds the scripts' ``resolve_vlm_run``, whose ``probe_checkpoint`` call is where their
# pin is threaded.
_VLM_PROBE_CALL_SITES = [
    *_TRAINING_SCRIPTS,
    REPO_ROOT / "halo" / "entrypoints" / "model_setup.py",
    REPO_ROOT / "halo" / "entrypoints" / "data_setup.py",
]


def _vlm_probe_calls(path: Path) -> list[ast.Call]:
    """Every entry point into the modality verdict: ``probe_checkpoint`` reads the config, and
    ``is_vlm_model`` / ``is_vlm_run`` take one already loaded or the probe's verdict, so an unpinned
    probe loads hub main's config."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return sum((_named_calls(tree, name) for name in ("is_vlm_model", "is_vlm_run", "probe_checkpoint")), [])


def test_vlm_probe_contract_is_not_vacuous():
    """At least one probe must exist, else the parametrized test below passes by skipping."""
    assert [p for p in _VLM_PROBE_CALL_SITES if _vlm_probe_calls(p)], (
        "No probe_checkpoint call site found — the revision-pinning contract is no longer tested."
    )


@pytest.mark.parametrize(
    "path",
    _VLM_PROBE_CALL_SITES,
    ids=[str(p.relative_to(REPO_ROOT)) for p in _VLM_PROBE_CALL_SITES],
)
def test_is_vlm_model_probe_is_revision_pinned(path: Path):
    """Every probe must pin the revision (or be handed an already-loaded config or verdict).

    ``probe_checkpoint`` fetches ``AutoConfig``, and hub ``main`` can name a
    different modality — or a different tensor namespace — than the commit ``model_revision`` pins
    (Zaya). The answer picks the ``Auto*`` class, the processing class, the collator and the run
    prefix, so an unpinned probe routes a pinned run down the wrong path with no error.
    """
    calls = _vlm_probe_calls(path)
    if not calls:
        pytest.skip("module does not probe the modality")
    for call in calls:
        kwargs = {kw.arg for kw in call.keywords}
        assert kwargs & {"revision", "config", "vlm_checkpoint"}, (
            f"{path}: modality probe pins neither revision= nor an already-loaded config= or vlm_checkpoint= — "
            f"a model_revision-pinned run would be routed by hub main's config."
        )


@pytest.mark.parametrize(
    "path",
    _VLM_PROBE_CALL_SITES,
    ids=[str(p.relative_to(REPO_ROOT)) for p in _VLM_PROBE_CALL_SITES],
)
def test_is_vlm_model_probe_threads_trust_remote_code(path: Path):
    """A probe that fetches the config must pass the run's ``trust_remote_code``: left at its
    default it refuses a remote-code config the run trusted, and forced on it would execute code the
    run never trusted, on the run's first hub contact."""
    calls = _vlm_probe_calls(path)
    if not calls:
        pytest.skip("module does not probe the modality")
    for call in calls:
        kwargs = {kw.arg for kw in call.keywords}
        assert kwargs & {"config", "vlm_checkpoint", "trust_remote_code"}, (
            f"{path}: modality probe fetches a config without the run's trust_remote_code="
        )


def _forced_remote_code_lines(path: Path) -> list[int]:
    """Lines where ``path`` passes a literal ``trust_remote_code=True``."""
    return [
        node.lineno
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
        if isinstance(node, ast.Call)
        and any(
            kw.arg == "trust_remote_code" and isinstance(kw.value, ast.Constant) and kw.value.value is True
            for kw in node.keywords
        )
    ]


def test_no_src_load_forces_trust_remote_code():
    """Every load in ``halo/`` takes the run's ``trust_remote_code``. Forced on, it runs a
    checkpoint's remote modules the run never trusted, and on a repo that ships a native class
    beside an ``auto_map`` it swaps the native config or processor for the remote one."""
    src_files = sorted((REPO_ROOT / "halo").rglob("*.py"))
    assert len(src_files) > 100, f"the halo/ sweep collapsed to {len(src_files)} files"
    forced = [
        f"{path.relative_to(REPO_ROOT)}:{line}" for path in src_files for line in _forced_remote_code_lines(path)
    ]
    assert not forced, f"trust_remote_code=True hardcoded at {forced}; pass the run's model_config.trust_remote_code"


# sft.py: preprocessed completion-masking mismatch


@pytest.fixture(scope="module")
def metadata_cls():
    return PreprocessedDatasetMetadata


def _validate_masking(metadata_cls, baked: bool | None, runtime: bool):
    config = {} if baked is None else {"train_on_completions_only": baked}
    metadata = metadata_cls(max_length=4096, train_on_completions_only=bool(baked), config=config)
    verify_preprocessed_dataset(metadata, required_max_length=4096, required_train_on_completions_only=runtime)


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
    kwarg from the call site turns a baked/runtime mismatch back into a silent wrong-masking run
    — and none of the helper-level cases above would notice.
    """
    tree = ast.parse((_TRAINING_DIR / "sft.py").read_text(encoding="utf-8"))
    calls = _named_calls(tree, "verify_preprocessed_dataset")
    assert len(calls) == 1, f"sft.py makes {len(calls)} preprocessed-metadata checks; expected one for both paths"
    functions = {node.name: node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
    checker = next(name for name, node in functions.items() if _named_calls(node, "verify_preprocessed_dataset"))
    for path in ("_prepare_text_data", "_prepare_vlm_data"):
        assert _named_calls(functions[path], checker), f"sft.py {path} skips the preprocessed-metadata check"
    for call in calls:
        value = {kw.arg: kw.value for kw in call.keywords}.get("required_train_on_completions_only")
        assert isinstance(value, ast.Attribute), (
            "sft.py must pass the live args.train_on_completions_only; a literal (or an omitted "
            "kwarg) degrades the baked/runtime mismatch check to a no-op."
        )


def test_sft_forwards_render_args_and_packing_to_the_preprocessed_check():
    """``render_args=None`` / ``required_packing=None`` skip those checks entirely, so dropping
    either kwarg from the call site silently reverts render-knob verification (baked rows vs YAML)
    and the unpacked-artifact-with-packing warning."""
    tree = ast.parse((_TRAINING_DIR / "sft.py").read_text(encoding="utf-8"))
    calls = _named_calls(tree, "verify_preprocessed_dataset")
    assert len(calls) == 1, f"sft.py makes {len(calls)} preprocessed-metadata checks; expected one for both paths"
    for call in calls:
        kwargs = {kw.arg: kw.value for kw in call.keywords}
        assert isinstance(kwargs.get("render_args"), ast.Name), (
            "sft.py must pass the live script args as render_args; omitting it skips the baked-render verification."
        )
        assert isinstance(kwargs.get("required_packing"), ast.Attribute), (
            "sft.py must pass the live sft_config.packing; omitting it skips the unpacked-artifact warning."
        )


def _passed_name(call: ast.Call, func: ast.FunctionDef, param: str) -> str | None:
    """The variable ``call`` passes for ``func``'s parameter ``param``, positionally or by keyword."""
    value = {kw.arg: kw.value for kw in call.keywords}.get(param)
    if value is None:
        index = [a.arg for a in func.args.args].index(param)
        value = call.args[index] if index < len(call.args) else None
    return value.id if isinstance(value, ast.Name) else None


def test_sft_holds_the_preprocessed_rows_to_the_template_the_trainer_exports():
    """``run_processing_class=None`` skips the exported-template check, and checking another object than the one
    the trainer saves (a VLM run's tokenizer instead of its processor) checks a template the checkpoint never
    carries. Traced from the trainer's ``processing_class`` through both data paths to the check."""
    tree = ast.parse((_TRAINING_DIR / "sft.py").read_text(encoding="utf-8"))
    functions = {node.name: node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
    (check,) = _named_calls(tree, "verify_preprocessed_dataset")
    checked = {kw.arg: kw.value for kw in check.keywords}.get("run_processing_class")
    assert isinstance(checked, ast.Name), "sft.py must pass the run's processing class as run_processing_class"
    checker = next(name for name, node in functions.items() if _named_calls(node, "verify_preprocessed_dataset"))
    (trainer,) = _named_calls(functions["main"], "DistributedSFTTrainer")
    exported = {kw.arg: kw.value for kw in trainer.keywords}["processing_class"].id
    for path in ("_prepare_text_data", "_prepare_vlm_data"):
        (inner,) = _named_calls(functions[path], checker)
        path_param = _passed_name(inner, functions[checker], checked.id)
        (outer,) = _named_calls(functions["main"], path)
        assert path_param is not None and _passed_name(outer, functions[path], path_param) == exported, (
            f"sft.py {path} checks another object than the processing_class the trainer exports"
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
        mock.patch("halo.entrypoints.script.install_log_tee"),
        mock.patch.object(sys, "argv", ["prog", str(config)]),
    ):
        module.main()


def test_self_distill_rejects_train_on_last_assistant_only(tmp_path):
    module = _load_script_module("distillation/self_distill.py")
    with pytest.raises(ValueError, match="train_on_last_assistant_only"):
        _run_main_with_yaml(module, "train_on_last_assistant_only: true\n", tmp_path)


def test_self_distill_requires_a_conversation_column(tmp_path):
    """The loader only checks a declared column; an unset one would reach the collators."""
    module = _load_script_module("distillation/self_distill.py")
    with pytest.raises(ValueError, match="conversation_field must name its column"):
        _run_main_with_yaml(module, "conversation_field: null\n", tmp_path)


@pytest.mark.parametrize("max_length", ["null", "0"])
def test_sft_refuses_packing_without_a_length_before_touching_the_checkpoint(tmp_path, max_length):
    """The pack size bounds memory, so it cannot default to the context window; the refusal is pure
    config and lands before the checkpoint probe (``dummy/model`` would fail that first)."""
    module = _load_script_module("sft.py")
    with pytest.raises(ValueError, match="packing=True requires an explicit max_length"):
        _run_main_with_yaml(module, f"packing: true\nmax_length: {max_length}\n", tmp_path)


@pytest.mark.parametrize(
    ("axis_flag", "axis_name"),
    [("context_parallel_size", "Context Parallelism"), ("pipeline_parallel_size", "Pipeline Parallelism")],
)
def test_smpo_refuses_a_vlm_run_under_cp_or_pp_before_the_model_load(tmp_path, axis_flag, axis_name):
    """The trainer refuses both axes for a VLM run too, but only after the weights load. A text run
    of the same shape reaches the load, so the refusal is the VLM verdict's, not the flag's."""
    module = _load_script_module("preference/smpo.py")
    yaml_body = f"{axis_flag}: 2\n"
    with (
        mock.patch.object(module, "init_training_script"),
        mock.patch.object(module, "load_script_datasets", return_value=({}, False)),
        mock.patch.object(module, "load_policy_model", side_effect=AssertionError("the model load ran")),
    ):
        with (
            mock.patch.object(module, "resolve_preference_vlm_run", return_value=True),
            pytest.raises(ValueError, match=f"SMPO VLM mode does not support {axis_name}"),
        ):
            _run_main_with_yaml(module, yaml_body, tmp_path)
        with (
            mock.patch.object(module, "resolve_preference_vlm_run", return_value=False),
            pytest.raises(AssertionError, match="the model load ran"),
        ):
            _run_main_with_yaml(module, yaml_body, tmp_path)


# PEFT wrapping: scripts must route through build_peft_model (k-bit prep)


@pytest.mark.parametrize(
    "script",
    _TRAINING_SCRIPTS,
    ids=[str(p.relative_to(_TRAINING_DIR)) for p in _TRAINING_SCRIPTS],
)
def test_no_script_calls_bare_get_peft_model(script: Path):
    """A bare ``get_peft_model`` skips ``build_peft_model``'s k-bit prep (layernorm fp32 upcast,
    gradient-checkpointing input hooks) that every sibling trainer applies — QLoRA would silently
    mis-train. Scripts that wrap PEFT themselves must call ``build_peft_model``."""
    source = script.read_text(encoding="utf-8")
    assert not _named_calls(ast.parse(source), "get_peft_model"), (
        f"{script}: calls bare get_peft_model — route through build_peft_model "
        f"(halo/model_loading/parallel/peft_setup.py) like SMPO/classification/offline-GRPO."
    )


@pytest.mark.parametrize(
    "script",
    _TRAINING_SCRIPTS,
    ids=[str(p.relative_to(_TRAINING_DIR)) for p in _TRAINING_SCRIPTS],
)
def test_no_script_calls_bare_get_peft_config(script: Path):
    """A bare ``get_peft_config`` reads the YAML alone and never sees the model, so target names that
    also spell a module PEFT cannot adapt (a multimodal tower's wrapped projections) reach injection
    and kill the run. ``build_peft_config`` is the seam that excludes them."""
    source = script.read_text(encoding="utf-8")
    assert not _named_calls(ast.parse(source), "get_peft_config"), (
        f"{script}: calls bare get_peft_config — route through build_peft_config "
        f"(halo/model_loading/parallel/peft_setup.py), directly or via configure_adapter_training."
    )


_TOKENIZER_SEAMS = (
    "apply_max_length",
    "apply_prompt_completion_window",
    "apply_tokenizer_window",
    "configure_run_tokenizer",
)

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
    knob parses, validates, logs — and does nothing. That is invisible at runtime, so a dropped
    return can sit unnoticed on several scripts at once.

    ``apply_prompt_completion_window`` returns ``(tokenizer, window)``, so a tuple target counts.
    """
    seen = 0
    for tree in _seam_trees(script):
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


@pytest.mark.parametrize(
    "script",
    _TRAINING_SCRIPTS,
    ids=[str(p.relative_to(_TRAINING_DIR)) for p in _TRAINING_SCRIPTS],
)
def test_init_training_script_takes_the_parsed_parallelism_arguments(script: Path):
    """``init_training_script`` builds the ``ParallelismConfig`` from its fourth argument, so handing
    it any other parsed dataclass fails the run at the config build, after the process group is up."""
    tree = ast.parse(script.read_text(encoding="utf-8"))
    (parse_call,) = _named_calls(tree, "parse_script_args")
    slot = [argument.id for argument in parse_call.args].index("ParallelismArguments")
    (target,) = _assigned_targets(tree, parse_call)
    parsed_name = target.elts[slot].id
    (init_call,) = _named_calls(tree, "init_training_script")
    assert ast.unparse(init_call.args[3]) == parsed_name, (
        f"{script}: init_training_script gets {ast.unparse(init_call.args[3])!r}, not the parsed "
        f"ParallelismArguments {parsed_name!r}"
    )


@pytest.mark.parametrize("rel_path", ["grpo/online.py", "grpo/environmental.py"])
def test_prompt_dataset_scripts_declare_the_prompt_column_to_the_loader(rel_path: str):
    """The GRPO prompt column is the loader's render column: declared, the loader refuses a dataset
    without it and drops the rows whose prompt is empty, which otherwise render as an empty question
    the policy is trained to answer. The refusal names ``conversation_knob``, so it must be the field
    these scripts read the column from — ``conversation_field`` is no knob of theirs."""
    tree = ast.parse((_TRAINING_DIR / rel_path).read_text(encoding="utf-8"))
    declared = [
        {keyword.arg: ast.unparse(keyword.value) for keyword in call.keywords}
        for call in _named_calls(tree, "load_script_datasets")
    ]
    assert [(call.get("conversation_field"), call.get("conversation_knob")) for call in declared] == [
        ("args.prompt_field", "'prompt_field'")
    ], f"{rel_path}: load_script_datasets declares {declared}"


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
    calls = [
        (tree, call)
        for tree in _seam_trees(_TRAINING_DIR / rel_path)
        for call in _named_calls(tree, "install_resolved_tokenizer")
    ]
    assert calls, f"{rel_path}: never calls install_resolved_tokenizer — processing_class is unresolved."
    for tree, call in calls:
        targets = _assigned_targets(tree, call)
        names = {n.id for t in (targets or []) for n in ast.walk(t) if isinstance(n, ast.Name)}
        assert "processing_class" in names, (
            f"{rel_path}: install_resolved_tokenizer() result is bound to {sorted(names)}, "
            f"not `processing_class` — the resolved tokenizer never reaches the trainer."
        )


# Collator selection: the completion-masking guards live in the factory


_COMPLETION_COLLATORS = sorted(
    name
    for name, obj in vars(padded).items()
    if isinstance(obj, type) and obj.__module__ == padded.__name__ and "CompletionOnly" in name
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


def _sft_trainer_builds(script: Path) -> list[tuple[ast.FunctionDef, ast.Call]]:
    """Each construction in ``script`` of a class it imports from ``halo.trainers`` that derives from TRL's
    ``SFTTrainer``, with the innermost function holding it."""
    tree = ast.parse(script.read_text(encoding="utf-8"))
    trainers = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("halo.trainers"):
            module = importlib.import_module(node.module)
            for alias in node.names:
                obj = getattr(module, alias.name, None)
                if isinstance(obj, type) and issubclass(obj, SFTTrainer):
                    trainers.add(alias.asname or alias.name)
    functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
    builds = []
    for name in sorted(trainers):
        for call in _named_calls(tree, name):
            holder = max((f for f in functions if f.lineno <= call.lineno <= f.end_lineno), key=lambda f: f.lineno)
            builds.append((holder, call))
    return builds


_SFT_TRAINER_SCRIPTS = [script for script in _TRAINING_SCRIPTS if _sft_trainer_builds(script)]


def test_the_sft_trainer_script_roster_is_derived_and_nonempty():
    assert {script.name for script in _SFT_TRAINER_SCRIPTS} >= {"sft.py", "self_distill.py"}


@pytest.mark.parametrize(
    "script",
    _SFT_TRAINER_SCRIPTS,
    ids=[str(p.relative_to(_TRAINING_DIR)) for p in _SFT_TRAINER_SCRIPTS],
)
def test_scripts_turn_trl_dataset_prep_off_before_building_an_sft_trainer(script: Path):
    """TRL's ``SFTTrainer.__init__`` reads ``packing`` and ``padding_free`` again: its ``padding_free``, which its
    default "bfd" packing turns on, refuses the script's collator, so a script that leaves either on fails every
    packed or padding-free run at the trainer build."""
    for function, call in _sft_trainer_builds(script):
        config = next((kw.value for kw in call.keywords if kw.arg == "args"), None)
        assert isinstance(config, ast.Name), f"{script.name}:{call.lineno}: pass the trainer's config as args=<name>"
        seams = [
            seam
            for seam in _named_calls(function, "disable_trl_dataset_prep")
            if seam.lineno < call.lineno and [getattr(arg, "id", None) for arg in seam.args] == [config.id]
        ]
        assert seams, (
            f"{script.name}:{call.lineno}: {function.name}() builds the trainer without "
            f"disable_trl_dataset_prep({config.id}) before it."
        )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
