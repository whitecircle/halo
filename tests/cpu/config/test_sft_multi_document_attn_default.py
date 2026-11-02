#!/usr/bin/env python
"""SFT's multi-document rows load on FA2 where FA4's head_dim-256 kernel would run them on a GatedDeltaNet family, or
would leave them to SDPA.

That kernel gains under 5% over FA2 on packed rows, and a deterministic run (``FLASH_ATTENTION_DETERMINISTIC``, set by
``full_determinism``), whose backward it cannot take, demotes FA4, pinned or not, to SDPA, which runs a packed row as
one dense ``[L, L]`` mask with K/V repeated per query head. ``default_training_rows_attn_implementation`` asks for FA2,
which consumes the ``cu_seq_lens`` those rows already carry: on an unpinned GatedDeltaNet family in either run, and on
any family whose FA4 the deterministic run demotes, a pinned FA4 included; rows of one document keep the loader's pick.
Every arm is silent when it breaks — a wrong verdict either keeps the slower path or hands FA2 a shape it faults on —
so the selection is held per (family, collator, arch, determinism, request), through the resolver the loader runs,
and through ``sft.py:main()``. Tests run deterministic unless they say otherwise.

Run: python tests/cpu/config/test_sft_multi_document_attn_default.py  (or pytest)
"""

import sys
import types
from unittest import mock

import pytest
import torch
from datasets import Dataset, DatasetDict
from transformers.models.auto.configuration_auto import CONFIG_MAPPING

from halo.configs.script_arguments import ModelArguments
from halo.data.collators.collator_rules import reject_unsupported_collator_options
from halo.entrypoints import model_setup
from halo.entrypoints.model_setup import default_training_rows_attn_implementation
from halo.models import segment_markers
from halo.models.attention import selection
from halo.models.attention.gpt_oss_sinks import SinksPolicy
from halo.models.attention.layout import has_gated_delta_net_layers, has_mrope_rotary
from halo.models.modality import is_vlm_model
from tests.common.repo import load_script_module

# The families the default moves to FA2: GatedDeltaNet decoders on FA4's head_dim-256 kernel, every spelling,
# wrappers included.
_GDN_MODEL_TYPES = ("qwen3_5", "qwen3_5_text", "qwen3_5_moe", "qwen3_5_moe_text", "qwen3_next")
# The ones whose decoder rotates by M-RoPE, text-only towers included: ``padding_free`` keeps them off FA2.
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
    kernel, deterministic or not; ``padding_free`` additionally leaves out every M-RoPE decoder."""
    expected = (
        "flash_attention_2"
        if arch in _FA4_ARCHS
        and model_type in _GDN_MODEL_TYPES
        and not (padding_free and model_type in _MROPE_GDN_MODEL_TYPES)
        else None
    )
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
    """With the heads on FA4's head_dim-256 kernel, the declared GatedDeltaNet layers alone decide outside a
    deterministic run (whose demotion moves any such decoder, held below): a GatedDeltaNet decoder under a name no
    list holds takes FA2, while a decoder without such layers (a Qwen3-Next config declaring only full attention,
    GLM-4.7-Flash's MLA) keeps the loader's pick."""
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
    """The default is the head_dim-256 kernel's: a GatedDeltaNet decoder whose heads are another width keeps the
    loader's pick, packed or padding-free, deterministic or not."""
    assert has_gated_delta_net_layers(config) and not selection.uses_fa4_dedicated_head_dim(config)
    assert _default_for(config, "b300", padding_free=padding_free) is None


@pytest.mark.parametrize(
    ("model_type", "mrope"),
    [(t, True) for t in _MROPE_GDN_MODEL_TYPES + ("qwen3_vl", "qwen2_5_vl")]
    + [(t, False) for t in ("qwen3_next", "qwen3", "gpt_oss", "llama", "glm4_moe_lite")],
)
def test_mrope_is_read_off_the_decoder_not_the_checkpoint_modality(model_type, mrope):
    """A text-only Qwen3.5 tower runs the same M-RoPE rotary as the multimodal wrapper, with no ``mrope_section``
    in its ``rope_parameters`` (the rotary's default applies); the verdict must hold for both."""
    config = CONFIG_MAPPING[model_type]()
    assert has_mrope_rotary(config) is mrope


@pytest.mark.parametrize("model_type", _MROPE_GDN_MODEL_TYPES)
def test_padding_free_on_a_qwen35_decoder_stays_on_sdpa(model_type):
    """Negative control: the Qwen3.5 decoder's M-RoPE varlen path faults FA2 under ``padding_free``, on a text-only
    checkpoint as on a multimodal one, so the default must stay off and the run keep its loud refusal, which names
    the M-RoPE rotary no backend serves rather than the SDPA it resolved to."""
    config = CONFIG_MAPPING[model_type]()
    assert _default(model_type, "b300", padding_free=True) is None
    with mock.patch.object(selection, "_detect_attention_impl", return_value="flash_attention_4"):
        resolved = selection.resolve_attn_implementation(
            config, None, torch.bfloat16, sinks_policy=SinksPolicy.NEUTRALIZED
        )
    assert resolved == "sdpa"
    with pytest.raises(ValueError, match="rotates by M-RoPE"):
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


@pytest.mark.parametrize("model_type", _MROPE_GDN_MODEL_TYPES)
def test_padding_free_on_a_qwen35_decoder_on_fa4_stops_at_collator_selection(model_type, monkeypatch):
    """Outside a deterministic run the Qwen3.5 decoder resolves FA4, a varlen kernel the dense-backend refusal
    accepts; the M-RoPE refusal is what keeps its flattened row off the varlen path."""
    monkeypatch.delenv(_DETERMINISTIC_ENV)
    config = CONFIG_MAPPING[model_type]()
    config._attn_implementation = "flash_attention_4"
    with pytest.raises(ValueError, match="M-RoPE"):
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
    """The default is only worth anything if the resolver keeps it: with it a packed row resolves to FA2, which the
    collator then accepts, while an unpinned load keeps the loader's own pick, FA4, demoted to SDPA in a
    deterministic run."""
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
        (CONFIG_MAPPING["qwen3_5_moe"](), {"packed": "flash_attention_2"}),
        (CONFIG_MAPPING["qwen3_next"](), {"packed": "flash_attention_2", "padding_free": "flash_attention_2"}),
        (CONFIG_MAPPING["qwen3"](), {}),
    ],
    ids=["softcapped_gemma2_27b", "qwen3_5_moe", "qwen3_next", "qwen3"],
)
@pytest.mark.parametrize("rows", sorted(_ROW_SHAPES))
def test_one_default_serves_the_softcap_and_the_packed_gated_delta_net_rows(config, expected, rows):
    """Both kernel limits answer through the one default: a softcap moves every row that reaches FA4's varlen call,
    a padded batch of several rows included, while the GatedDeltaNet rule moves only multi-document rows (and not
    ``padding_free`` on an M-RoPE decoder); a single unpadded row keeps the loader's pick on every model."""
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
    """The backend a B300 run loads: multi-document rows (but ``padding_free`` on M-RoPE, which collator selection
    refuses) take FA2 off a deterministic run's FA4 demotion, pinned or not, and off an unpinned FA4 on a
    GatedDeltaNet family; rows of one document keep the demotion to SDPA."""
    if kernel_class == "general":
        return "flash_attention_4"
    if kernel_class == "hd256_refused":
        return "flash_attention_2"
    multi_document = rows == "packed" or (rows == "padding_free" and kernel_class != "gated_delta_net_hd256_mrope")
    if multi_document and (deterministic or (pinned is None and kernel_class.startswith("gated_delta_net"))):
        return "flash_attention_2"
    return "sdpa" if deterministic else "flash_attention_4"


@pytest.mark.parametrize("pinned", [None, "flash_attention_4"], ids=["unpinned", "pinned_fa4"])
@pytest.mark.parametrize("rows", sorted(_ROW_SHAPES))
@pytest.mark.parametrize("kernel_class", sorted(_KERNEL_CLASSES))
def test_the_loaded_backend_per_kernel_class_rows_determinism_and_request(kernel_class, rows, determinism, pinned):
    """Through the request and the resolver the loader runs: a deterministic run never hands multi-document rows the
    SDPA its FA4 demotion would pick, whether the YAML pins FA4 or leaves the pick to the GPU."""
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


class _Stop(Exception):
    """Raised by a stubbed seam: ``main()`` runs only as far as it."""


_TEXT_TURNS = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]


def _run_sft_main(tmp_path, model_type: str, yaml_body: str, *, preprocessed_packed: bool | None = None) -> dict:
    """What ``sft.py:main()`` hands its seams for a ``model_type`` checkpoint on a B300 whose own pick is FA4: the
    ``attn_implementation`` the policy load passes ``load_distributed_model`` (the request the script makes, through
    the YAML's pin), or the backend the tree-packing pre-check is given.

    ``preprocessed_packed`` stands the dataset in for a pre-processed corpus baked packed (True) or unpacked
    (False); ``None`` keeps it raw.
    """
    module = load_script_module("scripts/training/sft.py", f"halo_test_sft_attn_{model_type}")
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        f"model_name_or_path: stub/model\ndataset:\n- dummy/dataset\noutput_dir: {tmp_path / 'out'}\n"
        f"bf16: false\nuse_cpu: true\nmax_length: 512\n{yaml_body}"
    )
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

    def load_distributed(**kwargs):
        captured["load_attn"] = kwargs["attn_implementation"]
        raise _Stop

    def check_tree_attention(_config, attn_implementation, _full_attention, **_tree_options):
        captured["tree_attn"] = attn_implementation
        raise _Stop

    metadata = types.SimpleNamespace(packed=preprocessed_packed, is_vlm=False)
    on_arch, capability = _on_arch("b300")
    patches = [
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
        mock.patch.object(module, "init_training_script", return_value=runtime),
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
        "padded_rows",
        "pinned",
        "glm4_moe_lite",
        "fa4_family",
        "preprocessed_packed",
        "preprocessed_unpacked",
    ],
)
def test_sft_main_hands_the_policy_load_the_default(tmp_path, model_type, yaml_body, preprocessed_packed, expected):
    """``sft.py`` must ask for the default exactly when its collator will build multi-document rows: a pre-processed
    corpus by how it was baked, a raw one by ``packing`` / ``padding_free``."""
    captured = _run_sft_main(tmp_path, model_type, yaml_body, preprocessed_packed=preprocessed_packed)
    assert captured == {"load_attn": expected}


@pytest.mark.usefixtures("determinism")
@pytest.mark.parametrize("model_type", ["qwen3_5_moe", "qwen3_5_moe_text"])
def test_sft_main_refuses_padding_free_on_an_mrope_decoder_before_the_load(tmp_path, model_type):
    """Before any weight loads and for the M-RoPE rotary, deterministic run or not: not for the SDPA a deterministic
    run demotes FA4 to on these heads, which the dense-kernel check would name instead."""
    with pytest.raises(ValueError, match="rotates by M-RoPE"):
        _run_sft_main(tmp_path, model_type, "padding_free: true\n")


@pytest.mark.parametrize("model_type", ["qwen3_5_moe", "qwen3_next"])
def test_sft_main_moves_a_pinned_fa4_on_packed_rows_off_the_deterministic_demotion(tmp_path, model_type, determinism):
    """``attn_implementation: flash_attention_4`` on packed rows reaches the policy load as FA2 in a deterministic run,
    not as the pin the loader would demote to SDPA, and as the pin outside one."""
    captured = _run_sft_main(tmp_path, model_type, "packing: true\nattn_implementation: flash_attention_4\n")
    assert captured == {"load_attn": "flash_attention_2" if determinism else "flash_attention_4"}


def test_an_unpacked_row_outside_a_deterministic_run_loads_fa4(tmp_path, monkeypatch):
    """A single-document row gets no default: ``main()`` hands the load none, and the resolver keeps the auto-detected
    FA4, the faster kernel on long unpacked rows of these families."""
    monkeypatch.delenv(_DETERMINISTIC_ENV)
    captured = _run_sft_main(tmp_path, "qwen3_5_moe", "")
    assert captured == {"load_attn": None}
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
    """The tree-attention check runs before any weight loads, against the backend the loader will resolve: on an
    unpinned Qwen3.5 run that is the FA2 default, deterministic or not, which the check accepts, not the SDPA a
    deterministic run's demotion picks without it (which tree packing refuses)."""
    assert _run_sft_main(tmp_path, "qwen3_5_moe", yaml_body) == {"tree_attn": expected}


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
