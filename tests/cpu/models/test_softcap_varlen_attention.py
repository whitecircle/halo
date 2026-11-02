#!/usr/bin/env python
"""CPU tests for where a softcapped model (Gemma 2) trains on FlashAttention-4, per size, row shape and GPU.

FA4 trains a softcap on its dense call only. Its head_dim-256 backward refuses one outright (Gemma 2 2B and 9B), and a
softcap stays off its varlen kernels at any width (not validated there): a packed, padding-free or padded training
batch unpads to that call, so Gemma 2 27B (head_dim 128) trains on FA4 only one unpadded row per batch at a time,
evaluation batches included. Every other row shape takes FA2, chosen before the load; a pinned FA4 there is refused
with the remedy; a host with no FA2 build refuses rather than fall back to a kernel that drops the softcap. Whatever
still reaches FA4's varlen entry point with a softcap (a pinned FA4 on a padded script) raises there, before the kernel
launches.

A stand-in ``flash_attn.cute`` checks the premise against the installed transformers: an unpadded row reaches the
dense entry point, a padded batch and a packed row the varlen one.

Run: ``python tests/cpu/models/test_softcap_varlen_attention.py`` (or ``pytest -m cpu``).
"""

from __future__ import annotations

import contextlib
import sys
import types
from unittest.mock import patch

import pytest
import torch
from transformers import Gemma2Config, Gemma2ForCausalLM, Qwen3Config
from transformers import modeling_flash_attention_utils as flash_utils

from halo.data.collators.collator_rules import (
    evaluation_batch_size,
    reject_fa4_varlen_training,
    rows_reach_varlen_kernel,
)
from halo.data.collators.factory import select_data_collator
from halo.entrypoints import model_setup
from halo.entrypoints.model_setup import default_padded_attn_implementation, default_training_rows_attn_implementation
from halo.model_loading import model_preparation
from halo.models.attention import selection
from halo.models.attention.backend_setup import install_fa4_varlen_softcap_refusal
from halo.models.attention.gpt_oss_sinks import SinksPolicy
from tests.common.models import TINY_GEMMA2_CONFIG
from tests.common.packed_flash_varlen import PACKED_ATTN_IMPLEMENTATIONS, PACKING_FAMILIES, SWEEP_ATTN_IMPLEMENTATION
from tests.common.tiny_models import tiny_family_model

FA2, FA3, FA4, SDPA = "flash_attention_2", "flash_attention_3", "flash_attention_4", "sdpa"
_DETECTED = {"blackwell": FA4, "hopper": FA3}
_UNSET = types.SimpleNamespace(attn_implementation=None)

# The release attention geometry of each Gemma 2 size, at two layers; every size softcaps its scores at 50.
_GEMMA2_SIZES = {
    "2b": {"hidden_size": 2304, "num_attention_heads": 8, "num_key_value_heads": 4, "head_dim": 256},
    "9b": {"hidden_size": 3584, "num_attention_heads": 16, "num_key_value_heads": 8, "head_dim": 256},
    "27b": {"hidden_size": 4608, "num_attention_heads": 32, "num_key_value_heads": 16, "head_dim": 128},
}
# The training rows each SFT collator builds, as (packing, padding_free, per_device_train_batch_size).
_SFT_ROWS = {
    "unpacked-bs1": (False, False, 1),
    "unpacked-bs4": (False, False, 4),
    "packed": (True, False, 1),
    "padding-free": (False, True, 1),
}
# The default's row arguments for an unpacked single row (no varlen call) and a packed one.
_UNPADDED_ROW = {"packing": False, "padding_free": False, "per_device_train_batch_size": 1}
_PACKED_ROW = {"packing": True, "padding_free": False, "per_device_train_batch_size": 1}


def _gemma2(size: str, **overrides) -> Gemma2Config:
    return Gemma2Config(**_GEMMA2_SIZES[size], num_hidden_layers=2, **overrides)


@contextlib.contextmanager
def _on(arch: str, *, fa2: bool = True):
    """A host whose GPU is ``arch``, whose FA4/FA3 build imports, and whose FA2 build does when ``fa2``."""
    with (
        patch.object(selection, "_detect_attention_impl", lambda: _DETECTED[arch]),
        patch.object(selection, "is_blackwell_gpu", lambda: arch == "blackwell"),
        patch.object(selection, "is_hopper_gpu", lambda: arch == "hopper"),
        patch.object(selection, "is_flash_attn_2_available", lambda: fa2),
        patch.object(selection, "_flash_build_imports", lambda _impl: True),
        patch.object(model_setup, "is_flash_attn_2_available", lambda: fa2),
        patch.object(selection.logger, "warning"),
        patch.object(selection.logger, "info"),
    ):
        yield


def _resolve(config, request) -> str:
    return selection.resolve_attn_implementation(config, request, torch.bfloat16, sinks_policy=SinksPolicy.NEUTRALIZED)


def _sft_backend(config, rows: str) -> str:
    """What an unpinned SFT policy load resolves for ``rows``."""
    packing, padding_free, batch_size = _SFT_ROWS[rows]
    request = default_training_rows_attn_implementation(
        _UNSET,
        packing=packing,
        padding_free=padding_free,
        per_device_train_batch_size=batch_size,
        dtype=torch.bfloat16,
        checkpoint_config=config,
    )
    return _resolve(config, request)


def _padded_script_backend(config) -> str:
    """What an unpinned padded script (DPO, GRPO, RM, ...) resolves."""
    request = default_padded_attn_implementation(_UNSET, sinks_reset=True, checkpoint_config=config)
    return _resolve(config, request)


# (size, arch) -> backend per SFT row shape, then the padded scripts'. 2B/9B leave FA4 on every row (its hd256 kernel
# takes no softcap); 27B keeps it on one unpadded row per batch only; Hopper's FA3 trains the softcap on every row.
# The padded scripts request FA2 on Blackwell, and their SDPA request lands on FA3 on Hopper, since SDPA drops the
# softcap.
_EXPECTED = {
    ("2b", "blackwell"): {"unpacked-bs1": FA2, "unpacked-bs4": FA2, "packed": FA2, "padding-free": FA2, "padded": FA2},
    ("9b", "blackwell"): {"unpacked-bs1": FA2, "unpacked-bs4": FA2, "packed": FA2, "padding-free": FA2, "padded": FA2},
    ("27b", "blackwell"): {
        "unpacked-bs1": FA4,
        "unpacked-bs4": FA2,
        "packed": FA2,
        "padding-free": FA2,
        "padded": FA2,
    },
    ("2b", "hopper"): {"unpacked-bs1": FA3, "unpacked-bs4": FA3, "packed": FA3, "padding-free": FA3, "padded": FA3},
    ("9b", "hopper"): {"unpacked-bs1": FA3, "unpacked-bs4": FA3, "packed": FA3, "padding-free": FA3, "padded": FA3},
    ("27b", "hopper"): {"unpacked-bs1": FA3, "unpacked-bs4": FA3, "packed": FA3, "padding-free": FA3, "padded": FA3},
}
_CASES = [(size, arch, rows) for (size, arch), by_rows in _EXPECTED.items() for rows in by_rows]


@pytest.mark.parametrize(("size", "arch", "rows"), _CASES, ids=["-".join(case) for case in _CASES])
def test_the_backend_per_gemma2_size_row_shape_and_gpu(size, arch, rows):
    config = _gemma2(size)
    with _on(arch):
        resolved = _padded_script_backend(config) if rows == "padded" else _sft_backend(config, rows)
    assert resolved == _EXPECTED[(size, arch)][rows]


@pytest.mark.parametrize("rows", ["unpacked-bs4", "packed", "padding-free"])
def test_fa4_keeps_every_row_shape_without_a_softcap(rows):
    """Anti-vacuity: the route reads the softcap, not the family. The 27B geometry without one, and Qwen3, keep FA4
    on every row shape, and on the padded scripts too."""
    for config in (_gemma2("27b", attn_logit_softcapping=None), Qwen3Config()):
        with _on("blackwell"):
            assert _sft_backend(config, rows) == FA4
            assert _padded_script_backend(config) == FA4


@pytest.mark.parametrize("family", sorted(name for name in PACKING_FAMILIES if name.startswith("dense:")))
def test_the_packed_varlen_sweep_runs_each_dense_family_on_its_packed_sft_backend(family):
    """The sweep's flash label per family is the one an unpinned packed SFT run resolves on Blackwell: FA2 for Gemma 2's
    softcap and for Gemma 3's windows on FA4's head_dim-256 kernel, FA4 for the rest."""
    config = tiny_family_model(PACKING_FAMILIES[family]).config
    with _on("blackwell"):
        resolved = _sft_backend(config, "packed")
    assert PACKED_ATTN_IMPLEMENTATIONS.get(family, SWEEP_ATTN_IMPLEMENTATION) == resolved


def test_a_softcapped_forward_only_companion_takes_the_padded_scripts_off_fa4():
    """A distillation teacher only forwards, but a softcap stays off FA4's varlen forward too, so a softcapped teacher
    moves the shared request off the loader's pick as a softcapped policy does; the softcap then lands the teacher on
    FA2."""
    with _on("blackwell"):
        assert (
            default_padded_attn_implementation(
                _UNSET, sinks_reset=True, checkpoint_config=Qwen3Config(), companion_configs=(_gemma2("27b"),)
            )
            == SDPA
        )
        assert _resolve(_gemma2("27b"), SDPA) == FA2


def test_a_softcapped_policy_requests_fa2_on_its_padded_batches():
    """A softcapped policy's padded scripts request FA2 themselves, the softcap named as the reason, rather than an
    SDPA the resolver then refuses for the softcap; without an FA2 build the request stays SDPA, whose refusal routes
    on to a softcap-carrying backend as before."""
    with _on("blackwell"), patch.object(selection.logger, "warning") as warn:
        request = default_padded_attn_implementation(_UNSET, sinks_reset=True, checkpoint_config=_gemma2("27b"))
        assert request == FA2
        assert _resolve(_gemma2("27b"), request) == FA2
    messages = [str(call.args[0]) for call in warn.call_args_list]
    assert any("attention logit softcapping" in message for message in messages), messages
    assert not any("sdpa" in message for message in messages), messages
    with _on("blackwell", fa2=False):
        assert default_padded_attn_implementation(_UNSET, sinks_reset=True, checkpoint_config=_gemma2("27b")) == SDPA


def test_the_sft_default_reads_no_checkpoint_on_the_branches_that_decide_without_it():
    """The checkpoint is fetched (a collective) only where the rows reach the varlen call on an FA4 GPU."""
    with patch.object(model_setup, "fetch_checkpoint_config", return_value=_gemma2("27b")) as fetch:
        with _on("blackwell"):
            assert default_training_rows_attn_implementation(_UNSET, **_UNPADDED_ROW, dtype=torch.bfloat16) is None
            assert default_training_rows_attn_implementation(_UNSET, **_PACKED_ROW, dtype=torch.float32) is None
            assert (
                default_training_rows_attn_implementation(
                    types.SimpleNamespace(attn_implementation=FA4), **_PACKED_ROW, dtype=torch.bfloat16
                )
                == FA4
            )
        with _on("hopper"):
            assert default_training_rows_attn_implementation(_UNSET, **_PACKED_ROW, dtype=torch.bfloat16) is None
        fetch.assert_not_called()
        with _on("blackwell"):
            assert default_training_rows_attn_implementation(_UNSET, **_PACKED_ROW, dtype=torch.bfloat16) == FA2
        fetch.assert_called_once()


@pytest.mark.parametrize(
    ("packing", "padding_free", "batch_size", "eval_batch_size", "expected"),
    [
        (False, False, 1, 0, False),
        (False, False, 1, 1, False),
        (False, False, 2, 0, True),
        (False, False, 1, 8, True),
        (True, False, 1, 0, True),
        (False, True, 1, 0, True),
    ],
)
def test_rows_reach_the_varlen_kernel_packed_padding_free_or_padded(
    packing, padding_free, batch_size, eval_batch_size, expected
):
    """A padded evaluation batch unpads to the varlen call as a padded training batch does."""
    assert (
        rows_reach_varlen_kernel(
            packing=packing,
            padding_free=padding_free,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=eval_batch_size,
        )
        is expected
    )


@pytest.mark.parametrize(("eval_strategy", "expected"), [("no", 0), ("steps", 8), ("epoch", 8)])
def test_the_evaluation_batch_counts_only_where_the_run_evaluates(eval_strategy, expected):
    assert evaluation_batch_size(types.SimpleNamespace(eval_strategy=eval_strategy, per_device_eval_batch_size=8)) == (
        expected
    )


def test_padded_evaluation_batches_take_a_softcapped_run_off_fa4():
    """One unpadded training row would keep 27B on FA4, but an evaluation batch of eight pads and unpads to the
    varlen call, which a softcap stays off."""
    with _on("blackwell"):
        trains_only = default_training_rows_attn_implementation(
            _UNSET, **_UNPADDED_ROW, dtype=torch.bfloat16, checkpoint_config=_gemma2("27b")
        )
        evaluates = default_training_rows_attn_implementation(
            _UNSET,
            **_UNPADDED_ROW,
            per_device_eval_batch_size=8,
            dtype=torch.bfloat16,
            checkpoint_config=_gemma2("27b"),
        )
    assert (trains_only, evaluates) == (None, FA2)


# --- no FA2 build -----------------------------------------------------------------------------------------------


@pytest.mark.parametrize("size", ["2b", "9b"])
def test_a_head_dim_256_softcap_without_flash_attention_2_is_refused_with_the_remedy(size):
    """FA4 cannot run 2B/9B at all; with no FA2 build the resolver raises naming what to set, instead of falling
    through to a label the model build would refuse."""
    with _on("blackwell", fa2=False), pytest.raises(ValueError, match="no FlashAttention-2 build") as error:
        _resolve(_gemma2(size), None)
    assert "attention logit softcapping" in str(error.value)
    assert "flex_attention or eager" in str(error.value)


def test_a_varlen_softcap_route_without_flash_attention_2_is_refused():
    with _on("blackwell", fa2=False), pytest.raises(ValueError, match="no FlashAttention-2 build"):
        _sft_backend(_gemma2("27b"), "packed")


@pytest.mark.parametrize("fa2", [True, False], ids=["fa2", "no-fa2"])
def test_an_fp32_run_leaves_the_softcap_to_the_resolvers_flex_attention(fa2):
    """fp32 runs no flash kernel: the resolver takes flex_attention, which applies the softcap, so the request names
    no FA2 that the resolver would drop, and a host without an FA2 build is not refused."""
    config = _gemma2("27b")
    with _on("blackwell", fa2=fa2):
        request = default_training_rows_attn_implementation(
            _UNSET, **_PACKED_ROW, dtype=torch.float32, checkpoint_config=config
        )
        resolved = selection.resolve_attn_implementation(
            config, request, torch.float32, sinks_policy=SinksPolicy.NEUTRALIZED
        )
    assert (request, resolved) == (None, "flex_attention")


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
def test_a_half_precision_run_on_varlen_rows_requests_fa2(dtype):
    """FA4 runs either half dtype, so the fp32 gate passes both to the FA2 route."""
    with _on("blackwell"):
        request = default_training_rows_attn_implementation(
            _UNSET, **_PACKED_ROW, dtype=dtype, checkpoint_config=_gemma2("27b")
        )
    assert request == FA2


def test_flash_attention_2_absence_moves_nothing_fa4_runs():
    """Anti-vacuity: the refusal is reached only through a route to FA2."""
    with _on("blackwell", fa2=False):
        assert _sft_backend(_gemma2("27b"), "unpacked-bs1") == FA4
        assert _sft_backend(Qwen3Config(), "packed") == FA4


def test_the_route_names_the_kernel_limit():
    """The warning is the only record of why a run lost FA4."""
    with _on("blackwell"), patch.object(selection.logger, "warning") as warn:
        _sft_backend(_gemma2("27b"), "packed")
    assert any(
        "varlen kernels are not validated with attention logit softcapping" in call.args[0]
        for call in warn.call_args_list
    )


# --- the collator rule: a pinned FA4 ----------------------------------------------------------------------------


def _resolved_config(size: str, attn_impl: str) -> Gemma2Config:
    config = _gemma2(size)
    config._attn_implementation = attn_impl
    return config


@pytest.mark.parametrize("rows", ["unpacked-bs4", "packed", "padding-free"])
def test_fa4_on_varlen_rows_of_a_softcapped_model_is_refused_with_the_remedy(rows):
    packing, padding_free, batch_size = _SFT_ROWS[rows]
    with pytest.raises(
        ValueError, match="varlen kernels, which are not validated with attention logit softcapping"
    ) as e:
        select_data_collator(
            tokenizer=types.SimpleNamespace(pad_token_id=0, eos_token_id=1, padding_side="right"),
            train_on_completions_only=False,
            packing=packing,
            padding_free=padding_free,
            per_device_train_batch_size=batch_size,
            model_config=_resolved_config("27b", FA4),
        )
    assert "flash_attention_2" in str(e.value) and "per_device_train_batch_size: 1" in str(e.value)


def test_the_rule_passes_every_run_fa4_trains():
    """Anti-vacuity: one unpadded row per batch, FA2, and an uncapped model all pass."""
    reject_fa4_varlen_training(_resolved_config("27b", FA4), varlen_rows=False)
    reject_fa4_varlen_training(_resolved_config("27b", FA2), varlen_rows=True)
    uncapped = _resolved_config("27b", FA4)
    uncapped.attn_logit_softcapping = None
    reject_fa4_varlen_training(uncapped, varlen_rows=True)


def test_the_collator_rule_refuses_fa4_on_padded_evaluation_batches():
    """The collator serves the evaluation batches too: a single training row does not clear eight padded ones."""
    with pytest.raises(ValueError, match="per_device_eval_batch_size: 1"):
        select_data_collator(
            tokenizer=types.SimpleNamespace(pad_token_id=0, eos_token_id=1, padding_side="right"),
            train_on_completions_only=False,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=8,
            model_config=_resolved_config("27b", FA4),
        )


def test_the_rule_reads_a_handed_backend_over_the_config():
    """Before the load the config carries no resolved backend; the caller hands the one the load resolves."""
    with pytest.raises(ValueError):
        reject_fa4_varlen_training(_gemma2("27b"), varlen_rows=True, attn_impl=FA4)
    reject_fa4_varlen_training(_resolved_config("27b", FA4), varlen_rows=True, attn_impl=FA2)


# --- premise: which flash entry point a training batch reaches --------------------------------------------------

# The transformers module globals the flash path caches the bound kernel and its kwarg map in.
_FLASH_CACHE_GLOBALS = (
    "_loaded_implementation",
    "_flash_fn",
    "_flash_varlen_fn",
    "_flash_with_kvcache_fn",
    "_pad_fn",
    "_unpad_fn",
    "_process_flash_kwargs_fn",
    "_process_with_kvcache_kwargs_fn",
)


class _RecordingCute(types.ModuleType):
    """``flash_attn.cute`` whose two entry points record whether each call was varlen and softcapped, and pass the
    query through."""

    def __init__(self):
        super().__init__("flash_attn.cute")
        self.calls: list[tuple[bool, bool]] = []

        def flash_attn_func(q, k, v, softmax_scale=None, causal=False, window_size=(None, None), softcap=0.0):
            return self._run(q, varlen=False, softcap=softcap)

        def flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q=None,
            cu_seqlens_k=None,
            max_seqlen_q=None,
            max_seqlen_k=None,
            softmax_scale=None,
            causal=False,
            window_size=(None, None),
            softcap=0.0,
        ):
            return self._run(q, varlen=cu_seqlens_q is not None, softcap=softcap)

        self.flash_attn_func = flash_attn_func
        self.flash_attn_varlen_func = flash_attn_varlen_func

    def _run(self, q, *, varlen: bool, softcap):
        self.calls.append((varlen, bool(softcap)))
        return q.clone()


@pytest.fixture
def cute(monkeypatch) -> _RecordingCute:
    """The stand-in installed where transformers' flash path imports FA4 from, with the path's cached binding reset
    (and restored afterwards)."""
    fake = _RecordingCute()
    monkeypatch.setitem(sys.modules, "flash_attn.cute", fake)
    # ``import flash_attn.cute`` reads the package attribute first, which an earlier real import leaves bound.
    monkeypatch.setattr(pytest.importorskip("flash_attn"), "cute", fake, raising=False)
    for name in _FLASH_CACHE_GLOBALS:
        monkeypatch.setattr(flash_utils, name, getattr(flash_utils, name))
    flash_utils._loaded_implementation = None
    return fake


_SEQ = 24
# row shape -> (input_ids rows, the forward's extra inputs, whether FA4's varlen entry point serves it)
_PREMISE_ROWS = {
    "unpacked-bs1": (1, {}, False),
    "unpacked-bs2-padded": (2, {"attention_mask": torch.tensor([[1] * _SEQ, [1] * (_SEQ - 6) + [0] * 6])}, True),
    "packed": (1, {"position_ids": torch.cat([torch.arange(10), torch.arange(_SEQ - 10)]).unsqueeze(0)}, True),
}


@pytest.mark.parametrize("rows", list(_PREMISE_ROWS))
def test_a_padded_or_packed_batch_reaches_fa4s_varlen_call(cute, rows):
    """Premise of the route: a softcapped model's training rows reach FA4's varlen call packed or padded, softcap
    included; one unpadded row reaches the dense call."""
    batch_rows, extra, reaches_varlen = _PREMISE_ROWS[rows]
    torch.manual_seed(0)
    model = Gemma2ForCausalLM(Gemma2Config(**TINY_GEMMA2_CONFIG, attn_logit_softcapping=50.0)).to(torch.bfloat16)
    model.config._attn_implementation = FA4
    ids = torch.randint(3, TINY_GEMMA2_CONFIG["vocab_size"], (batch_rows, _SEQ))
    loss = model(input_ids=ids, labels=ids, use_cache=False, **extra).loss
    assert cute.calls and {varlen for varlen, _ in cute.calls} == {reaches_varlen}
    assert all(softcap for _, softcap in cute.calls)
    assert ((True, True) in cute.calls) is reaches_varlen
    loss.backward()


# --- the varlen entry point's own refusal ----------------------------------------------------------------------


@pytest.mark.parametrize("rows", list(_PREMISE_ROWS))
def test_fa4s_varlen_entry_point_refuses_a_softcapped_call_in_the_forward(cute, rows):
    """Installed, the refusal turns the varlen call a padded or packed batch reaches into an error in the forward,
    before the kernel launches; one unpadded row reaches the dense call, which trains. The premise test above is the
    negative control: without the refusal the same forward reaches the varlen kernel with the softcap."""
    assert install_fa4_varlen_softcap_refusal()
    batch_rows, extra, reaches_varlen = _PREMISE_ROWS[rows]
    torch.manual_seed(0)
    model = Gemma2ForCausalLM(Gemma2Config(**TINY_GEMMA2_CONFIG, attn_logit_softcapping=50.0)).to(torch.bfloat16)
    model.config._attn_implementation = FA4
    ids = torch.randint(3, TINY_GEMMA2_CONFIG["vocab_size"], (batch_rows, _SEQ))
    if reaches_varlen:
        with pytest.raises(RuntimeError, match="reached FlashAttention-4's varlen kernels"):
            model(input_ids=ids, labels=ids, use_cache=False, **extra)
        assert cute.calls == [], "the refused call reached the kernel"
    else:
        model(input_ids=ids, labels=ids, use_cache=False, **extra).loss.backward()
        assert cute.calls and all(softcap and not varlen for varlen, softcap in cute.calls)


def test_the_refusal_passes_an_uncapped_varlen_call(cute):
    """Anti-vacuity: the refusal reads the call's softcap, so a model without one keeps FA4's varlen kernels."""
    install_fa4_varlen_softcap_refusal()
    torch.manual_seed(0)
    model = Gemma2ForCausalLM(Gemma2Config(**TINY_GEMMA2_CONFIG, attn_logit_softcapping=None)).to(torch.bfloat16)
    model.config._attn_implementation = FA4
    ids = torch.randint(3, TINY_GEMMA2_CONFIG["vocab_size"], (2, _SEQ))
    _, extra, _ = _PREMISE_ROWS["unpacked-bs2-padded"]
    model(input_ids=ids, labels=ids, use_cache=False, **extra).loss.backward()
    assert cute.calls and all(varlen and not softcap for varlen, softcap in cute.calls)


@pytest.mark.parametrize(
    ("config", "attn_implementation", "installs"),
    [
        (_gemma2("27b"), FA4, True),
        (_gemma2("27b", attn_logit_softcapping=None), FA4, False),
        (_gemma2("27b"), FA2, False),
    ],
    ids=["softcapped-fa4", "uncapped-fa4", "softcapped-fa2"],
)
def test_every_softcapped_fa4_load_installs_the_refusal(monkeypatch, config, attn_implementation, installs):
    """``apply_family_patches`` runs before every policy, frozen and trainer-path build, so each FA4 load of a softcapped
    model, a padded script's pinned policy or a frozen teacher, takes the refusal; no other load does."""
    calls = []
    monkeypatch.setattr(model_preparation, "install_fa4_varlen_softcap_refusal", lambda: calls.append(1))
    monkeypatch.setattr(model_preparation, "patch_transformers_flash_varlen_int_seqlen", lambda: None)
    model_preparation.apply_family_patches(config, attn_implementation)
    assert bool(calls) is installs


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
