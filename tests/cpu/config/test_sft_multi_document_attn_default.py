#!/usr/bin/env python
"""SFT's multi-document rows load on FA2 where FA4's head_dim-256 kernel would run them on a GatedDeltaNet family
(under 5% gain over FA2), or would leave them to SDPA: a deterministic run demotes FA4, which runs a packed row
as one dense ``[L, L]`` mask. Tests run deterministic unless they say otherwise."""

import copy
import os
import sys
import types
from unittest import mock

import pytest
import torch
from datasets import Dataset, DatasetDict
from transformers.models.auto.configuration_auto import CONFIG_MAPPING
from trl import SFTConfig

from halo.configs.script_arguments import ModelArguments
from halo.data.collators.collator_rules import reject_unsupported_collator_options
from halo.entrypoints import model_setup, run_setup
from halo.entrypoints.model_setup import default_training_rows_attn_implementation, plan_row_attention
from halo.model_loading.parallel.loader import resolve_policy_attn_implementation
from halo.models import segment_markers
from halo.models.attention import selection
from halo.models.attention.gpt_oss_sinks import SinksPolicy
from halo.models.attention.layout import has_gated_delta_net_layers
from halo.models.modality import is_vlm_model
from tests.common.repo import load_script_module

# The families the default moves to FA2: GatedDeltaNet decoders on FA4's head_dim-256 kernel, every spelling,
# wrappers included.
_GDN_MODEL_TYPES = ("qwen3_5", "qwen3_5_text", "qwen3_5_moe", "qwen3_5_moe_text", "qwen3_next")
# The ones whose decoder rotates by M-RoPE, text-only towers included.
_MROPE_GDN_MODEL_TYPES = ("qwen3_5", "qwen3_5_text", "qwen3_5_moe", "qwen3_5_moe_text")
# Near misses: GLM-4.7-Flash runs the same kernel but has no GatedDeltaNet layers (its FA2 packing is unvalidated);
# the rest keep FA4 and must never see the default.
_OTHER_MODEL_TYPES = ("glm4_moe_lite", "qwen3", "qwen3_moe", "gpt_oss", "llama")

_DETERMINISTIC_ENV = "FLASH_ATTENTION_DETERMINISTIC"
# The segment-aware kernel checks as the images answer them on a GPU host.
_SEGMENT_AWARE_KERNELS_PRESENT = (("causal_conv1d", lambda: True), ("fla>=0.2.2", lambda: True))

# Compute capability per arch, or None for a GPU-less host.
_ARCHS = {"b200": (10, 0), "b300": (10, 3), "h100": (9, 0), "cpu": None}
_FA4_ARCHS = ("b200", "b300")


@pytest.fixture(autouse=True)
def _deterministic_run(monkeypatch):
    monkeypatch.setenv(_DETERMINISTIC_ENV, "1")


@pytest.fixture(params=[True, False], ids=["deterministic", "nondeterministic"])
def determinism(request, _deterministic_run, monkeypatch) -> bool:
    """Runs a test in a deterministic run and outside one; the value says which."""
    if not request.param:
        monkeypatch.delenv(_DETERMINISTIC_ENV)
    return request.param


def _on_arch(arch: str):
    """Patches making the hardware predicates see ``arch``."""
    capability = _ARCHS[arch]
    return (
        mock.patch.object(selection.torch.cuda, "is_available", lambda: capability is not None),
        mock.patch.object(selection.torch.cuda, "get_device_capability", lambda *a, **k: capability),
    )


def _default_for(config, arch: str, *, padding_free: bool, pinned: str | None = None) -> str | None:
    on_arch, capability = _on_arch(arch)
    with (
        on_arch,
        capability,
        mock.patch.object(model_setup, "is_flash_attn_2_available", return_value=True),
        mock.patch.object(selection, "is_flash_attn_2_available", return_value=True),
    ):
        return default_training_rows_attn_implementation(
            ModelArguments(model_name_or_path="stub/model", attn_implementation=pinned),
            packing=not padding_free,
            padding_free=padding_free,
            per_device_train_batch_size=1,
            dtype=torch.bfloat16,
            checkpoint_config=config,
        )


def _default(model_type: str, arch: str, *, padding_free: bool, pinned: str | None = None) -> str | None:
    return _default_for(CONFIG_MAPPING[model_type](), arch, padding_free=padding_free, pinned=pinned)


@pytest.mark.usefixtures("determinism")
@pytest.mark.parametrize("arch", sorted(_ARCHS))
@pytest.mark.parametrize("padding_free", [False, True], ids=["packing", "padding_free"])
@pytest.mark.parametrize("model_type", _GDN_MODEL_TYPES + _OTHER_MODEL_TYPES)
def test_the_default_per_family_collator_and_arch(model_type, padding_free, arch):
    """FA2 exactly where the GPU's own pick is FA4 and the family is a GatedDeltaNet one on FA4's head_dim-256
    kernel, deterministic or not, packed or padding-free."""
    expected = "flash_attention_2" if arch in _FA4_ARCHS and model_type in _GDN_MODEL_TYPES else None
    assert _default(model_type, arch, padding_free=padding_free) == expected


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        (CONFIG_MAPPING["olmo_hybrid"](), "flash_attention_2"),
        (CONFIG_MAPPING["qwen3_next"](layer_types=["full_attention"] * 4, num_hidden_layers=4), None),
        (CONFIG_MAPPING["glm4_moe_lite"](), None),
    ],
    ids=["gated_delta_net_under_another_name", "qwen3_next_without_linear_layers", "glm4_moe_lite"],
)
def test_the_family_gate_reads_gated_delta_net_layers_not_the_model_type(config, expected, monkeypatch):
    """The declared GatedDeltaNet layers alone decide outside a deterministic run: such a decoder under an unlisted
    name takes FA2, while one without such layers (GLM-4.7-Flash's MLA) keeps the loader's pick."""
    monkeypatch.delenv(_DETERMINISTIC_ENV)
    with mock.patch.object(model_setup, "uses_fa4_dedicated_head_dim", return_value=True):
        assert _default_for(config, "b300", padding_free=False) == expected


@pytest.mark.usefixtures("determinism")
@pytest.mark.parametrize("padding_free", [False, True], ids=["packing", "padding_free"])
@pytest.mark.parametrize(
    "config",
    [
        CONFIG_MAPPING["olmo_hybrid"](),
        CONFIG_MAPPING["qwen3_next"](head_dim=128),
        CONFIG_MAPPING["qwen3_5_text"](head_dim=128),
    ],
    ids=["olmo_hybrid", "qwen3_next_head_dim_128", "qwen3_5_text_head_dim_128"],
)
def test_a_gated_delta_net_decoder_off_the_head_dim_256_kernel_keeps_the_loader_pick(config, padding_free):
    assert has_gated_delta_net_layers(config) and not selection.uses_fa4_dedicated_head_dim(config)
    assert _default_for(config, "b300", padding_free=padding_free) is None


@pytest.mark.parametrize("model_type", _MROPE_GDN_MODEL_TYPES)
def test_padding_free_on_a_qwen35_decoder_loads_fa2_and_passes_collator_selection(model_type):
    """The M-RoPE decoder takes the same FA2 default and the same collator rules as any GatedDeltaNet decoder."""
    config = CONFIG_MAPPING[model_type]()
    assert _default(model_type, "b300", padding_free=True) == "flash_attention_2"
    config._attn_implementation = "flash_attention_2"
    with mock.patch.object(segment_markers, "GDN_SEGMENT_AWARE_BACKENDS", _SEGMENT_AWARE_KERNELS_PRESENT):
        reject_unsupported_collator_options(
            types.SimpleNamespace(),
            padding_free=True,
            packing=False,
            train_on_completions_only=False,
            assistant_message_template=None,
            use_context_parallel=False,
            train_on_last_assistant_only=False,
            model_config=config,
            per_device_train_batch_size=1,
            keeps_packed_rows=False,
        )


@pytest.mark.parametrize("model_type", _GDN_MODEL_TYPES)
def test_the_loader_keeps_the_default(model_type, determinism):
    """With the default a packed row resolves to FA2, while an unpinned load keeps the loader's pick, FA4, demoted
    to SDPA in a deterministic run."""
    config = CONFIG_MAPPING[model_type]()
    with mock.patch.object(selection, "_detect_attention_impl", return_value="flash_attention_4"):
        unpinned = selection.resolve_attn_implementation(
            config, None, torch.bfloat16, sinks_policy=SinksPolicy.NEUTRALIZED
        )
        defaulted = selection.resolve_attn_implementation(
            config,
            _default(model_type, "b300", padding_free=False),
            torch.bfloat16,
            sinks_policy=SinksPolicy.NEUTRALIZED,
        )
    assert (unpinned, defaulted) == ("sdpa" if determinism else "flash_attention_4", "flash_attention_2")


@pytest.mark.parametrize("pinned", ["sdpa", "eager"])
def test_a_pinned_attn_implementation_is_kept(pinned):
    """A YAML pin of a backend other than FA4 wins over the default on every family and arch."""
    assert _default("qwen3_next", "b300", padding_free=False, pinned=pinned) == pinned


def test_no_default_without_an_fa2_build():
    """A default never names a backend the image lacks: the loader's own pick stands."""
    config = CONFIG_MAPPING["qwen3_next"]()
    on_arch, capability = _on_arch("b300")
    with on_arch, capability, mock.patch.object(model_setup, "is_flash_attn_2_available", return_value=False):
        assert (
            default_training_rows_attn_implementation(
                ModelArguments(model_name_or_path="stub/model"),
                packing=True,
                padding_free=False,
                per_device_train_batch_size=1,
                dtype=torch.bfloat16,
                checkpoint_config=config,
            )
            is None
        )


# (packing, padding_free, per_device_train_batch_size) of each SFT row shape.
_ROW_SHAPES = {
    "packed": (True, False, 1),
    "padding_free": (False, True, 1),
    "padded": (False, False, 4),
    "single_row": (False, False, 1),
}
_GEMMA2_27B = {"hidden_size": 4608, "num_attention_heads": 32, "num_key_value_heads": 16, "head_dim": 128}


@pytest.mark.usefixtures("determinism")
@pytest.mark.parametrize(
    ("config", "expected"),
    [
        (
            CONFIG_MAPPING["gemma2"](**_GEMMA2_27B, num_hidden_layers=2),
            {"packed": "flash_attention_2", "padding_free": "flash_attention_2", "padded": "flash_attention_2"},
        ),
        (CONFIG_MAPPING["qwen3_5_moe"](), {"packed": "flash_attention_2", "padding_free": "flash_attention_2"}),
        (CONFIG_MAPPING["qwen3_next"](), {"packed": "flash_attention_2", "padding_free": "flash_attention_2"}),
        (CONFIG_MAPPING["qwen3"](), {}),
    ],
    ids=["softcapped_gemma2_27b", "qwen3_5_moe", "qwen3_next", "qwen3"],
)
@pytest.mark.parametrize("rows", sorted(_ROW_SHAPES))
def test_one_default_serves_the_softcap_and_the_packed_gated_delta_net_rows(config, expected, rows):
    """A softcap moves every row reaching FA4's varlen call, while the GatedDeltaNet rule moves only multi-document
    rows; a single unpadded row keeps the loader's pick."""
    packing, padding_free, batch_size = _ROW_SHAPES[rows]
    on_arch, capability = _on_arch("b300")
    with on_arch, capability, mock.patch.object(model_setup, "is_flash_attn_2_available", return_value=True):
        with mock.patch.object(selection, "is_flash_attn_2_available", return_value=True):
            request = default_training_rows_attn_implementation(
                ModelArguments(model_name_or_path="stub/model"),
                packing=packing,
                padding_free=padding_free,
                per_device_train_batch_size=batch_size,
                dtype=torch.bfloat16,
                checkpoint_config=config,
            )
    assert request == expected.get(rows)


# One config per kernel class FA4 runs a family's heads on: its head_dim-256 kernel on a GatedDeltaNet decoder, with
# and without M-RoPE, and on a plain decoder; that kernel refused outright (multi-head latent attention, which the
# resolver moves to FA2 itself); and FA4's general kernel, which a deterministic run keeps.
_KERNEL_CLASSES = {
    "gated_delta_net_hd256_mrope": lambda: CONFIG_MAPPING["qwen3_5_moe"](),
    "gated_delta_net_hd256": lambda: CONFIG_MAPPING["qwen3_next"](),
    "hd256": lambda: CONFIG_MAPPING["qwen3"](head_dim=256),
    "hd256_refused": lambda: CONFIG_MAPPING["glm4_moe_lite"](),
    "general": lambda: CONFIG_MAPPING["qwen3"](),
}


def _expected_backend(kernel_class: str, rows: str, deterministic: bool, pinned: str | None) -> str:
    """The backend a B300 run loads: multi-document rows take FA2 off a deterministic run's FA4 demotion, pinned or
    not, and off an unpinned FA4 on a GatedDeltaNet family; rows of one document keep the demotion to SDPA."""
    if kernel_class == "general":
        return "flash_attention_4"
    if kernel_class == "hd256_refused":
        return "flash_attention_2"
    multi_document = rows in ("packed", "padding_free")
    if multi_document and (deterministic or (pinned is None and kernel_class.startswith("gated_delta_net"))):
        return "flash_attention_2"
    return "sdpa" if deterministic else "flash_attention_4"


@pytest.mark.parametrize("pinned", [None, "flash_attention_4"], ids=["unpinned", "pinned_fa4"])
@pytest.mark.parametrize("rows", sorted(_ROW_SHAPES))
@pytest.mark.parametrize("kernel_class", sorted(_KERNEL_CLASSES))
def test_the_loaded_backend_per_kernel_class_rows_determinism_and_request(kernel_class, rows, determinism, pinned):
    """A deterministic run never hands multi-document rows the SDPA its FA4 demotion would pick, whether the YAML
    pins FA4 or not."""
    packing, padding_free, batch_size = _ROW_SHAPES[rows]
    config = _KERNEL_CLASSES[kernel_class]()
    on_arch, capability = _on_arch("b300")
    with (
        on_arch,
        capability,
        mock.patch.object(model_setup, "is_flash_attn_2_available", return_value=True),
        mock.patch.object(selection, "is_flash_attn_2_available", return_value=True),
        mock.patch.object(selection, "_detect_attention_impl", return_value="flash_attention_4"),
    ):
        request = default_training_rows_attn_implementation(
            ModelArguments(model_name_or_path="stub/model", attn_implementation=pinned),
            packing=packing,
            padding_free=padding_free,
            per_device_train_batch_size=batch_size,
            dtype=torch.bfloat16,
            checkpoint_config=config,
        )
        resolved = selection.resolve_attn_implementation(
            config, request, torch.bfloat16, sinks_policy=SinksPolicy.NEUTRALIZED
        )
    assert resolved == _expected_backend(kernel_class, rows, determinism, pinned)


def test_a_pinned_fa4_moved_to_fa2_says_why(caplog):
    """The pin gives way loudly: the warning names the deterministic backward and the multi-document rows."""
    with caplog.at_level("WARNING", logger=selection.logger.name):
        assert _default("qwen3_next", "b300", padding_free=False, pinned="flash_attention_4") == "flash_attention_2"
    assert "not validated with a deterministic backward" in caplog.text and "several documents" in caplog.text


@pytest.mark.usefixtures("determinism")
@pytest.mark.parametrize(("model_type", "moved"), [("qwen3_5_moe", True), ("qwen3", False)])
def test_an_unpinned_gated_delta_net_moved_to_fa2_says_why(caplog, model_type, moved):
    """The packed GatedDeltaNet default names its reason and the pin that keeps FA4, in either run; a family it
    leaves alone logs nothing."""
    with caplog.at_level("INFO"):
        request = _default(model_type, "b300", padding_free=False)
    lines = [record.getMessage() for record in caplog.records if "GatedDeltaNet" in record.getMessage()]
    assert request == ("flash_attention_2" if moved else None)
    assert len(lines) == int(moved)
    assert all("under 5%" in line and "attn_implementation: flash_attention_4" in line for line in lines)


class _Stop(Exception):
    """Raised by a stubbed seam: ``main()`` runs only as far as it."""


_TEXT_TURNS = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]


def _run_setup_stubs(tmp_path) -> list:
    """Patches for the steps of ``configure_training_run`` that reach past the test process's environment (kernel
    caches, matmul precision, the datasets cache, verbosity, the seed), so the rest runs as the script runs it."""
    return [
        mock.patch.object(run_setup, "ensure_fa4_kernel_cache_env"),
        mock.patch.object(run_setup, "anchor_jit_cache_dir"),
        mock.patch.object(run_setup, "configure_float32_matmul_precision"),
        mock.patch.object(run_setup, "ensure_cache_dir", return_value=str(tmp_path / "datasets")),
        mock.patch.object(run_setup.datasets.config, "HF_DATASETS_CACHE", run_setup.datasets.config.HF_DATASETS_CACHE),
        mock.patch.object(run_setup.transformers.logging, "set_verbosity_info"),
        mock.patch.object(run_setup.transformers.logging, "set_verbosity_error"),
        mock.patch.object(run_setup, "set_seed"),
    ]


def _run_sft_main(
    tmp_path,
    model_type: str,
    yaml_body: str,
    *,
    preprocessed_packed: bool | None = None,
    checkpoint_config=None,
) -> dict:
    """What ``sft.py:main()`` decides on a B300 whose own pick is FA4: ``planned_attn`` (the row plan's backend),
    ``load_attn`` and ``loaded_attn`` (the policy load's request and resolution) and ``tree_attn``.
    ``preprocessed_packed`` stands in a baked corpus; ``None`` keeps it raw."""
    module = load_script_module("scripts/training/sft.py", f"halo_test_sft_attn_{model_type}")
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        f"model_name_or_path: stub/model\ndataset:\n- dummy/dataset\noutput_dir: {tmp_path / 'out'}\n"
        f"bf16: false\nuse_cpu: true\nmax_length: 512\n{yaml_body}"
    )
    if checkpoint_config is None:
        checkpoint_config = CONFIG_MAPPING[model_type]()
    rows = {"prompt": [_TEXT_TURNS]}
    dataset = DatasetDict({"train": Dataset.from_dict(rows), "test": Dataset.from_dict(rows)})
    runtime = types.SimpleNamespace(
        parallelism_config=types.SimpleNamespace(cp_size=1, is_cp_mode=False, pp_size=1),
        model_source="stub/model",
        mode_suffix="",
        local_rank=0,
        resume_checkpoint=None,
        policy_from_checkpoint=False,
    )
    captured = {}

    def init_training_script(args, training_config, *_args, **_kwargs):
        run_setup.configure_training_run(
            args, training_config, "sft-test", overwrite_output_dir=args.overwrite_output_dir
        )
        return runtime

    def plan_rows(model_config, training_args, **kwargs):
        plan = model_setup.plan_row_attention(model_config, training_args, **kwargs)
        captured["planned_attn"] = resolve_policy_attn_implementation(
            copy.deepcopy(checkpoint_config),
            plan.attn_default,
            torch.bfloat16,
            sinks_policy=kwargs["sinks_policy"],
            init_from_scratch=kwargs["init_from_scratch"],
        )
        return plan

    def load_distributed(**kwargs):
        captured["load_attn"] = kwargs["attn_implementation"]
        captured["loaded_attn"] = resolve_policy_attn_implementation(
            copy.deepcopy(checkpoint_config),
            kwargs["attn_implementation"],
            kwargs["dtype"],
            sinks_policy=SinksPolicy.from_flags(reset_sinks=kwargs["reset_sinks"], train_sinks=kwargs["train_sinks"]),
            init_from_scratch=kwargs["init_from_scratch"],
        )
        raise _Stop

    def check_tree_attention(_config, attn_implementation, full_attention, **_tree_options):
        captured["tree_attn"] = attn_implementation
        captured["tree_form"] = full_attention
        raise _Stop

    metadata = types.SimpleNamespace(packed=preprocessed_packed, is_vlm=False)
    on_arch, capability = _on_arch("b300")
    patches = [
        mock.patch.dict(os.environ),
        on_arch,
        capability,
        mock.patch.object(model_setup, "is_flash_attn_2_available", return_value=True),
        mock.patch.object(selection, "is_flash_attn_2_available", return_value=True),
        mock.patch.object(selection, "_detect_attention_impl", return_value="flash_attention_4"),
        mock.patch.object(
            module,
            "probe_checkpoint",
            return_value=(checkpoint_config, is_vlm_model("stub/model", checkpoint_config)),
        ),
        mock.patch.object(segment_markers, "GDN_SEGMENT_AWARE_BACKENDS", _SEGMENT_AWARE_KERNELS_PRESENT),
        mock.patch.object(module, "plan_row_attention", side_effect=plan_rows),
        mock.patch.object(module, "init_training_script", side_effect=init_training_script),
        *_run_setup_stubs(tmp_path),
        mock.patch.object(module, "resolve_training_dtype", return_value=torch.bfloat16),
        mock.patch.object(model_setup, "resolve_training_dtype", return_value=torch.bfloat16),
        mock.patch.object(module, "reject_unsupported_tree_attention", side_effect=check_tree_attention),
        mock.patch.object(
            module, "load_script_datasets", return_value=((dataset, preprocessed_packed is not None), False)
        ),
        mock.patch.object(module, "load_preprocessed_metadata", return_value=metadata),
        mock.patch.object(
            model_setup,
            "probe_checkpoint",
            return_value=(checkpoint_config, is_vlm_model("stub/model", checkpoint_config)),
        ),
        mock.patch.object(model_setup, "load_vlm_processor", return_value=None),
        mock.patch.object(model_setup, "hub_metadata_main_first", return_value=None),
        mock.patch.object(model_setup, "load_distributed_model", side_effect=load_distributed),
        mock.patch("halo.entrypoints.script.install_log_tee"),
        mock.patch.object(sys, "argv", ["prog", str(config_file)]),
    ]
    with mock.patch.object(module, "with_distributed_teardown", lambda fn: fn):
        for patch in patches:
            patch.start()
        try:
            with pytest.raises(_Stop):
                module.main()
        finally:
            for patch in reversed(patches):
                patch.stop()
    return captured


@pytest.mark.usefixtures("determinism")
@pytest.mark.parametrize(
    ("model_type", "yaml_body", "preprocessed_packed", "expected"),
    [
        ("qwen3_5_moe", "packing: true\n", None, "flash_attention_2"),
        ("qwen3_next", "padding_free: true\n", None, "flash_attention_2"),
        ("qwen3_5_moe", "padding_free: true\n", None, "flash_attention_2"),
        ("qwen3_5_moe", "", None, None),
        ("qwen3_5_moe", "packing: true\nattn_implementation: sdpa\n", None, "sdpa"),
        ("glm4_moe_lite", "packing: true\n", None, None),
        ("qwen3", "packing: true\n", None, None),
        ("qwen3_5_moe", "", True, "flash_attention_2"),
        ("qwen3_5_moe", "packing: true\n", False, None),
    ],
    ids=[
        "packing",
        "padding_free_qwen3_next",
        "padding_free_qwen3_5_moe",
        "padded_rows",
        "pinned",
        "glm4_moe_lite",
        "fa4_family",
        "preprocessed_packed",
        "preprocessed_unpacked",
    ],
)
def test_sft_main_hands_the_policy_load_the_default(tmp_path, model_type, yaml_body, preprocessed_packed, expected):
    """A pre-processed corpus by how it was baked, a raw one by ``packing`` / ``padding_free``."""
    captured = _run_sft_main(tmp_path, model_type, yaml_body, preprocessed_packed=preprocessed_packed)
    assert captured["load_attn"] == expected


@pytest.mark.parametrize("model_type", ["qwen3_5_moe", "qwen3_next"])
def test_sft_main_moves_a_pinned_fa4_on_packed_rows_off_the_deterministic_demotion(tmp_path, model_type, determinism):
    """In a deterministic run the pin reaches the load as FA2, not as the pin the loader would demote to SDPA;
    outside one it stays."""
    captured = _run_sft_main(tmp_path, model_type, "packing: true\nattn_implementation: flash_attention_4\n")
    assert captured["load_attn"] == ("flash_attention_2" if determinism else "flash_attention_4")


def test_an_unpacked_row_outside_a_deterministic_run_loads_fa4(tmp_path, monkeypatch):
    """``main()`` hands the load no default, and the resolver keeps the auto-detected FA4, the faster kernel on long
    unpacked rows of these families."""
    monkeypatch.delenv(_DETERMINISTIC_ENV)
    captured = _run_sft_main(tmp_path, "qwen3_5_moe", "")
    assert captured["load_attn"] is None
    with mock.patch.object(selection, "_detect_attention_impl", return_value="flash_attention_4"):
        resolved = selection.resolve_attn_implementation(
            CONFIG_MAPPING["qwen3_5_moe"](),
            captured["load_attn"],
            torch.bfloat16,
            sinks_policy=SinksPolicy.NEUTRALIZED,
        )
    assert resolved == "flash_attention_4"


_TREE_PACKING = "packing: true\ntree_packing: true\ntree_share_markers: ['x']\ntree_full_attention: gather\n"


@pytest.mark.usefixtures("determinism")
@pytest.mark.parametrize(
    ("yaml_body", "expected"),
    [
        (_TREE_PACKING, "flash_attention_2"),
        (_TREE_PACKING + "attn_implementation: sdpa\n", "sdpa"),
    ],
    ids=["unpinned", "pinned"],
)
def test_the_tree_packing_pre_check_sees_the_backend_the_load_takes(tmp_path, yaml_body, expected):
    """On an unpinned Qwen3.5 run the check sees the FA2 default, which it accepts, not the SDPA a deterministic
    demotion picks without it, which tree packing refuses."""
    assert _run_sft_main(tmp_path, "qwen3_5_moe", yaml_body)["tree_attn"] == expected


_UNSET_FORM = "packing: true\ntree_packing: true\ntree_share_markers: ['x']\n"


@pytest.mark.usefixtures("determinism")
@pytest.mark.parametrize(
    ("model_type", "yaml_body", "attn", "form"),
    [
        ("qwen3", "", "flash_attention_4", "split"),
        ("qwen3", "attn_implementation: flash_attention_2\n", "flash_attention_2", "gather"),
        ("qwen3_5_moe", "", "flash_attention_2", "gather"),
        ("qwen3_5_moe", "tree_full_attention: split\n", "flash_attention_2", "split"),
    ],
    ids=["fa4", "pinned_fa2", "qwen3_5_default_fa2", "explicit_stays"],
)
def test_an_unset_tree_form_resolves_off_the_backend_the_load_takes(tmp_path, model_type, yaml_body, attn, form):
    """Resolved once before the load; what is explicit reaches the refusal the backend owes it."""
    captured = _run_sft_main(tmp_path, model_type, _UNSET_FORM + yaml_body)
    assert (captured["tree_attn"], captured["tree_form"]) == (attn, form)


_CROSSING_OPT_IN = "allow_cross_document_linear_attention: true\n"


@pytest.mark.usefixtures("determinism")
def test_the_crossing_opt_in_needs_a_multi_document_row(tmp_path):
    with pytest.raises(ValueError, match=r"without multi-document rows does not support.*allow_cross_document"):
        _run_sft_main(tmp_path, "qwen3", _CROSSING_OPT_IN)


@pytest.mark.usefixtures("determinism")
@pytest.mark.parametrize(
    ("yaml_body", "preprocessed_packed"),
    [("packing: true\n", None), ("padding_free: true\n", None), ("", True)],
    ids=["packing", "padding_free", "baked_packed"],
)
def test_the_crossing_opt_in_stands_on_multi_document_rows(tmp_path, yaml_body, preprocessed_packed):
    """A baked corpus decides by how it was baked, whatever the ``packing`` flag says."""
    captured = _run_sft_main(tmp_path, "qwen3", _CROSSING_OPT_IN + yaml_body, preprocessed_packed=preprocessed_packed)
    assert "load_attn" in captured, "the run stopped short of the model load"


# A checkpoint per attention rule both the row plan and the loader apply: the kernel classes above, a softcap FA4's
# varlen kernels are not validated with, and GptOss sinks.
_ATTENTION_RULES = {
    **_KERNEL_CLASSES,
    "softcap": lambda: CONFIG_MAPPING["gemma2"](**_GEMMA2_27B, num_hidden_layers=2),
    "sinks": lambda: CONFIG_MAPPING["gpt_oss"](),
}
_ROWS_YAML = {
    "packed": "packing: true\n",
    "padding_free": "padding_free: true\n",
    "padded": "per_device_train_batch_size: 2\n",
}


def _determinism_yaml(rows: str, *, full_determinism: bool, pinned: str | None) -> str:
    pin = f"attn_implementation: {pinned}\n" if pinned else ""
    return f"{_ROWS_YAML[rows]}full_determinism: {str(full_determinism).lower()}\n{pin}"


@pytest.mark.parametrize(
    ("rule", "pinned"),
    [("gated_delta_net_hd256_mrope", "flash_attention_4"), ("hd256", "flash_attention_4"), ("hd256", None)],
    ids=["qwen3_5_moe_pinned_fa4", "hd256_pinned_fa4", "hd256_unpinned"],
)
def test_full_determinism_set_in_the_config_moves_packed_rows_to_fa2(tmp_path, monkeypatch, rule, pinned):
    """A plan reading the run as non-deterministic would hand the loader an FA4 it demotes to SDPA, one dense mask
    per packed row."""
    monkeypatch.setenv(_DETERMINISTIC_ENV, "0")
    yaml_body = _determinism_yaml("packed", full_determinism=True, pinned=pinned)
    captured = _run_sft_main(tmp_path, rule, yaml_body, checkpoint_config=_ATTENTION_RULES[rule]())
    assert (captured["planned_attn"], captured["load_attn"], captured["loaded_attn"]) == ("flash_attention_2",) * 3


def _refused_before_the_load(rule: str, pinned: str | None) -> bool:
    """The runs ``main()`` refuses before any weight loads, held by their own test: a pinned FA4 on the softcap its
    varlen kernels are not validated with."""
    return rule == "softcap" and bool(pinned)


@pytest.mark.parametrize(
    ("rule", "rows", "pinned", "full_determinism"),
    [
        pytest.param(
            rule,
            rows,
            pinned,
            full_determinism,
            id=f"{rule}-{rows}-{pinned or 'unpinned'}-{'deterministic' if full_determinism else 'nondeterministic'}",
        )
        for rule in sorted(_ATTENTION_RULES)
        for rows in sorted(_ROWS_YAML)
        for pinned in (None, "flash_attention_4")
        for full_determinism in (True, False)
        if not _refused_before_the_load(rule, pinned)
    ],
)
def test_the_row_plan_and_the_load_resolve_one_backend(tmp_path, monkeypatch, rule, rows, pinned, full_determinism):
    """The backend the plan's request resolves to is the one the load resolves, and no row holding several documents
    lands on SDPA."""
    monkeypatch.setenv(_DETERMINISTIC_ENV, "0")
    yaml_body = _determinism_yaml(rows, full_determinism=full_determinism, pinned=pinned)
    captured = _run_sft_main(tmp_path, rule, yaml_body, checkpoint_config=_ATTENTION_RULES[rule]())
    assert captured["planned_attn"] == captured["loaded_attn"]
    if rows != "padded":
        assert captured["loaded_attn"] != "sdpa"


def test_a_row_plan_refuses_a_full_determinism_config_the_switch_has_not_reached(tmp_path, monkeypatch):
    """A config built without ``parse_script_args`` leaves transformers' switch as launched: the plan refuses it,
    and plans the deterministic run once the switch is on."""
    sft_config = SFTConfig(output_dir=str(tmp_path), full_determinism=True, use_cpu=True, bf16=False, report_to=[])
    on_arch, capability = _on_arch("b300")

    def plan():
        return plan_row_attention(
            ModelArguments(model_name_or_path="stub/model", attn_implementation="flash_attention_4"),
            sft_config,
            sinks_policy=SinksPolicy.NEUTRALIZED,
            packing=True,
            checkpoint_config=CONFIG_MAPPING["qwen3_next"](),
        )

    with (
        on_arch,
        capability,
        mock.patch.object(model_setup, "is_flash_attn_2_available", return_value=True),
        mock.patch.object(selection, "is_flash_attn_2_available", return_value=True),
        mock.patch.object(model_setup, "resolve_training_dtype", return_value=torch.bfloat16),
    ):
        monkeypatch.setenv(_DETERMINISTIC_ENV, "0")
        with pytest.raises(RuntimeError, match="full_determinism is set but FLASH_ATTENTION_DETERMINISTIC is not"):
            plan()
        monkeypatch.setenv(_DETERMINISTIC_ENV, "1")
        assert plan().attn_default == "flash_attention_2"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
