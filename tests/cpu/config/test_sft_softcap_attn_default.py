#!/usr/bin/env python
"""The SFT, self-distillation and embedding scripts request FA2 for a softcapped model whose training or evaluation rows
reach FA4's varlen call, and refuse a pinned FA4 there.

FA4's varlen kernels take no softcap (the backward asserts, and on B300 the forward does not finish), and packed,
padding-free and padded rows (more than one per batch) all reach them, so Gemma 2 27B (head_dim 128, which FA4's
general kernel otherwise runs) trains on FA4 only one unpadded row per batch. Driven through ``main()`` (the embedding EP/TP pipeline through its builder) on a Blackwell host up to the policy
load: the request must be decided on the rows the collator will build (a pre-processed corpus as it was baked), before
any weight loads, and reach every model the run loads on it.

Run: pytest tests/cpu/config/test_sft_softcap_attn_default.py
"""

import ast
import contextlib
import sys
import types
from unittest import mock

import pytest
import torch
from datasets import Dataset, DatasetDict
from transformers import Gemma2Config, Qwen3Config

from halo.entrypoints import model_setup
from halo.entrypoints.model_setup import PolicyLoad
from halo.model_loading.parallel import loader
from halo.models.attention import selection
from tests.common.repo import REPO_ROOT, load_script_module

FA2, FA4 = "flash_attention_2", "flash_attention_4"
_MODEL_ID = "stub/gemma-2-27b"
_TURNS = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]


class _PolicyLoadReached(Exception):
    """Raised by the stubbed policy load, or by the first call after it: main() runs only that far."""


def _gemma2(head_dim: int) -> Gemma2Config:
    """Gemma 2 27B's attention geometry (32:16 heads) at ``head_dim``, softcapped at 50 as every size is."""
    return Gemma2Config(hidden_size=4608, num_attention_heads=32, num_key_value_heads=16, head_dim=head_dim)


def _dataset() -> DatasetDict:
    data = {"prompt": [_TURNS]}
    return DatasetDict({"train": Dataset.from_dict(data), "test": Dataset.from_dict(data)})


def _runtime():
    return types.SimpleNamespace(
        parallelism_config=types.SimpleNamespace(cp_size=1, is_cp_mode=False, pp_size=1),
        model_source=_MODEL_ID,
        mode_suffix="",
        local_rank=0,
        resume_checkpoint=None,
        policy_from_checkpoint=False,
    )


@contextlib.contextmanager
def _blackwell(module, dtype: torch.dtype):
    """A Blackwell host with FA4 and FA2 builds, and ``module`` training in ``dtype`` (bf16, which a GPU-less
    ``TrainingArguments`` cannot construct, or fp32)."""
    with (
        mock.patch.object(selection, "is_blackwell_gpu", lambda: True),
        mock.patch.object(selection, "_detect_attention_impl", lambda: FA4),
        mock.patch.object(selection, "is_flash_attn_2_available", lambda: True),
        mock.patch.object(module, "resolve_training_dtype", lambda _config: dtype),
    ):
        yield


@contextlib.contextmanager
def _blackwell_main(module, tmp_path, yaml_body: str, dtype: torch.dtype = torch.bfloat16):
    """``module.main()`` on a Blackwell host with FA4 and FA2 builds, a ``dtype`` run, from a YAML holding
    ``yaml_body``."""
    config = tmp_path / "config.yaml"
    config.write_text(
        f"model_name_or_path: {_MODEL_ID}\ndataset:\n- dummy/dataset\noutput_dir: {tmp_path / 'out'}\n"
        f"bf16: false\nuse_cpu: true\nmax_length: 512\n{yaml_body}"
    )
    with (
        _blackwell(module, dtype),
        mock.patch.object(module, "init_training_script", return_value=_runtime()),
        mock.patch.object(module, "with_distributed_teardown", lambda fn: fn),
        mock.patch("halo.entrypoints.script.install_log_tee"),
        mock.patch.object(sys, "argv", ["prog", str(config)]),
    ):
        yield


def _sft_request(
    tmp_path, yaml_body: str, checkpoint_config, *, packed_corpus: bool | None = None, dtype=torch.bfloat16
) -> list[dict]:
    """The keyword arguments of every policy load ``sft.py:main()`` reaches; ``packed_corpus`` makes the dataset a
    pre-processed one baked packed (True) or unpacked (False)."""
    module = load_script_module("scripts/training/sft.py", "halo_test_sft_softcap_attn")
    loads = []

    def load(_context, *_args, **kwargs):
        loads.append(kwargs)
        raise _PolicyLoadReached

    preprocessed = packed_corpus is not None
    with (
        _blackwell_main(module, tmp_path, yaml_body, dtype),
        mock.patch.object(module, "probe_checkpoint", return_value=(checkpoint_config, False)),
        mock.patch.object(module, "load_script_datasets", return_value=((_dataset(), preprocessed), False)),
        mock.patch.object(
            module, "load_preprocessed_metadata", return_value=types.SimpleNamespace(packed=packed_corpus)
        ),
        mock.patch.object(module, "load_policy_model", side_effect=load),
        contextlib.suppress(_PolicyLoadReached),
    ):
        module.main()
    return loads


@pytest.mark.parametrize(
    ("yaml_body", "expected"),
    [
        ("per_device_train_batch_size: 1\n", None),
        ("", FA2),
        ("packing: true\n", FA2),
        ("padding_free: true\n", FA2),
    ],
    ids=["unpacked-bs1", "unpacked-default-bs8", "packed", "padding-free"],
)
def test_the_sft_policy_load_takes_fa2_on_rows_reaching_the_varlen_call(tmp_path, yaml_body, expected):
    loads = _sft_request(tmp_path, yaml_body, _gemma2(128))
    assert [load["attn_default"] for load in loads] == [expected]


@pytest.mark.parametrize(
    ("yaml_body", "expected"),
    [
        ("eval_strategy: steps\nper_device_eval_batch_size: 8\n", FA2),
        ("eval_strategy: steps\nper_device_eval_batch_size: 1\n", None),
        ("eval_strategy: 'no'\nper_device_eval_batch_size: 8\n", None),
    ],
    ids=["evaluates-bs8", "evaluates-bs1", "never-evaluates"],
)
def test_padded_evaluation_batches_take_the_sft_policy_off_fa4(tmp_path, yaml_body, expected):
    """One unpadded training row keeps FA4 only when the run's evaluation batches stay unpadded too: an evaluation
    batch of eight unpads to the varlen call, which on B300 would hang at the first evaluation."""
    loads = _sft_request(tmp_path, f"per_device_train_batch_size: 1\n{yaml_body}", _gemma2(128))
    assert [load["attn_default"] for load in loads] == [expected]


def test_a_pinned_fa4_with_padded_evaluation_batches_is_refused_before_the_load(tmp_path):
    yaml_body = f"attn_implementation: {FA4}\nper_device_train_batch_size: 1\neval_strategy: steps\n"
    with pytest.raises(ValueError, match="per_device_eval_batch_size: 1"):
        loads = _sft_request(tmp_path, f"{yaml_body}per_device_eval_batch_size: 8\n", _gemma2(128))
        pytest.fail(f"reached the policy load: {loads}")


@pytest.mark.parametrize(("packed_corpus", "expected"), [(True, FA2), (False, None)])
def test_a_preprocessed_corpus_is_judged_as_it_was_baked(tmp_path, packed_corpus, expected):
    """``packing`` is off in the YAML, but a corpus baked packed reaches the packing collator."""
    loads = _sft_request(tmp_path, "per_device_train_batch_size: 1\n", _gemma2(128), packed_corpus=packed_corpus)
    assert [load["attn_default"] for load in loads] == [expected]


def test_a_pinned_fa4_on_packed_rows_is_refused_before_the_load(tmp_path):
    with pytest.raises(ValueError, match="which take no attention logit softcapping"):
        loads = _sft_request(tmp_path, f"attn_implementation: {FA4}\npacking: true\n", _gemma2(128))
        pytest.fail(f"reached the policy load: {loads}")


@pytest.mark.parametrize(
    ("yaml_body", "head_dim"),
    [
        (f"attn_implementation: {FA4}\nper_device_train_batch_size: 1\n", 128),
        (f"attn_implementation: {FA4}\npacking: true\n", 256),
    ],
    ids=["unpacked-bs1", "hd256-the-loader-demotes"],
)
def test_a_pinned_fa4_the_load_keeps_off_varlen_rows_reaches_the_load(tmp_path, yaml_body, head_dim):
    """Anti-vacuity: one unpadded row per batch trains on FA4, and at head_dim 256 the loader moves the pin to FA2
    itself; neither is refused."""
    assert [load["attn_default"] for load in _sft_request(tmp_path, yaml_body, _gemma2(head_dim))] == [FA4]


@pytest.mark.parametrize(
    ("yaml_body", "checkpoint_config", "resolves"),
    [
        (f"attn_implementation: {FA4}\npacking: true\n", Qwen3Config(), 0),
        (f"attn_implementation: {FA4}\nper_device_train_batch_size: 1\n", _gemma2(128), 0),
        (f"attn_implementation: {FA4}\npacking: true\n", _gemma2(256), 1),
    ],
    ids=["uncapped-packed", "softcapped-unpacked-bs1", "softcapped-packed"],
)
def test_a_pinned_fa4_is_resolved_before_the_load_only_where_it_can_be_refused(
    tmp_path, yaml_body, checkpoint_config, resolves
):
    """The load resolves the backend again and logs every fallback warning a second time, so the pre-load refusal
    resolves only a softcapped model on varlen rows."""
    with mock.patch.object(
        loader, "resolve_policy_attn_implementation", wraps=loader.resolve_policy_attn_implementation
    ) as resolve:
        loads = _sft_request(tmp_path, yaml_body, checkpoint_config)
    assert [load["attn_default"] for load in loads] == [FA4]
    assert resolve.call_count == resolves


# --- self-distillation ------------------------------------------------------------------------------------------


def _self_distill_module():
    return load_script_module("scripts/training/distillation/self_distill.py", "halo_test_self_distill_softcap_attn")


def _self_distill_request(tmp_path, yaml_body: str, *, loaded_backend: str = FA2, dtype=torch.bfloat16) -> list[dict]:
    """The keyword arguments of the policy load ``self_distill.py:main()`` reaches, its checkpoint Gemma 2 27B; the
    stubbed load hands back a model resolved to ``loaded_backend``."""
    module = _self_distill_module()
    loads = []
    loaded = _gemma2(128)
    loaded._attn_implementation = loaded_backend

    def load(_context, *_args, **kwargs):
        loads.append(kwargs)
        return PolicyLoad(types.SimpleNamespace(config=loaded), None, None, False)

    with (
        _blackwell_main(module, tmp_path, f"conversation_field: prompt\nsdpg_beta_base: 0\n{yaml_body}", dtype),
        mock.patch.object(model_setup, "fetch_checkpoint_config", return_value=_gemma2(128)),
        mock.patch.object(module, "load_script_datasets", return_value=(_dataset(), False)),
        mock.patch.object(module, "resolve_vlm_run", return_value=False),
        mock.patch.object(module, "load_policy_model", side_effect=load),
        mock.patch.object(module, "apply_max_length", side_effect=_PolicyLoadReached),
        contextlib.suppress(_PolicyLoadReached),
    ):
        module.main()
    return loads


@pytest.mark.parametrize(
    ("yaml_body", "expected"),
    [("per_device_train_batch_size: 1\n", None), ("per_device_train_batch_size: 2\n", FA2)],
    ids=["bs1", "bs2"],
)
def test_the_self_distillation_policy_takes_fa2_on_padded_rows(tmp_path, yaml_body, expected):
    """Each branch pads one row per conversation, so more than one per batch reaches the varlen call."""
    assert [load["attn_default"] for load in _self_distill_request(tmp_path, yaml_body)] == [expected]


def test_a_self_distillation_policy_loaded_on_fa4_with_padded_rows_is_refused(tmp_path):
    with pytest.raises(ValueError, match="which take no attention logit softcapping"):
        _self_distill_request(
            tmp_path, f"attn_implementation: {FA4}\nper_device_train_batch_size: 2\n", loaded_backend=FA4
        )


def test_the_self_distillation_reference_takes_the_policys_request():
    """The anchor's KL is biased by a reference on another backend than the policy's."""
    module = _self_distill_module()
    args = types.SimpleNamespace(reference_kl_coef=0.5, reference_model_name_or_path=None, reset_sinks=True)
    model_config = types.SimpleNamespace(model_name_or_path=_MODEL_ID)
    with mock.patch.object(module, "load_frozen_reference_model", return_value=object()) as load:
        module._load_sdpg_reference(
            args=args, model_config=model_config, sft_config=None, tokenizer=None, is_vlm=False, attn_default=FA2
        )
    assert load.call_args.kwargs["attn_default"] == FA2


def test_the_self_distillation_policy_and_reference_loads_share_one_request():
    """``main()`` hands both loads the same expression; a reference left on its own default reopens the split."""
    tree = ast.parse((REPO_ROOT / "scripts/training/distillation/self_distill.py").read_text(encoding="utf-8"))
    (main,) = (node for node in tree.body if getattr(node, "name", None) == "main")
    requests = {
        node.func.id: ast.dump(keyword.value)
        for node in ast.walk(main)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        for keyword in node.keywords
        if keyword.arg == "attn_default"
    }
    assert set(requests) == {"load_policy_model", "_load_sdpg_reference"}, requests
    assert len(set(requests.values())) == 1, requests


# --- embedding (the EP/TP pipeline) -----------------------------------------------------------------------------


def _embedding_backbone_loads(
    *,
    batch_size: int,
    pinned: str | None = None,
    loaded_backend: str = FA2,
    dtype=torch.bfloat16,
    head_dim: int = 128,
    eval_strategy: str = "no",
) -> list[dict]:
    """The keyword arguments of the backbone load ``_assemble_preloaded_pipeline`` makes, its checkpoint Gemma 2 27B
    (2B/9B at ``head_dim`` 256); the stubbed load hands back a backbone resolved to ``loaded_backend``."""
    module = load_script_module("scripts/training/embedding.py", "halo_test_embedding_softcap_attn")
    loads = []
    loaded = _gemma2(head_dim)
    loaded._attn_implementation = loaded_backend

    def load(*_args, **kwargs):
        loads.append(kwargs)
        return types.SimpleNamespace(config=loaded), None

    with (
        _blackwell(module, dtype),
        mock.patch.object(model_setup, "fetch_checkpoint_config", return_value=_gemma2(head_dim)),
        mock.patch.object(module, "load_model_and_tokenizer", side_effect=load),
        mock.patch.object(module, "PreloadedTransformer", side_effect=_PolicyLoadReached),
        contextlib.suppress(_PolicyLoadReached),
    ):
        module._assemble_preloaded_pipeline(
            None,
            types.SimpleNamespace(
                per_device_train_batch_size=batch_size, per_device_eval_batch_size=8, eval_strategy=eval_strategy
            ),
            types.SimpleNamespace(attn_implementation=pinned, model_name_or_path=_MODEL_ID),
            None,
        )
    return loads


@pytest.mark.parametrize(("batch_size", "expected"), [(1, None), (2, FA2)], ids=["bs1", "bs2"])
def test_the_embedding_backbone_takes_fa2_on_padded_rows(batch_size, expected):
    """sentence-transformers pads each text column's batch, so more than one row per batch reaches the varlen call."""
    assert [load.get("attn_default") for load in _embedding_backbone_loads(batch_size=batch_size)] == [expected]


def test_padded_evaluation_batches_take_the_embedding_backbone_off_fa4():
    loads = _embedding_backbone_loads(batch_size=1, eval_strategy="steps")
    assert [load.get("attn_default") for load in loads] == [FA2]


def test_an_embedding_backbone_loaded_on_fa4_with_padded_rows_is_refused():
    with pytest.raises(ValueError, match="which take no attention logit softcapping"):
        _embedding_backbone_loads(batch_size=2, pinned=FA4, loaded_backend=FA4)


@pytest.mark.parametrize(
    ("batch_size", "head_dim", "loaded_backend"),
    [(1, 128, FA4), (2, 256, FA2)],
    ids=["unpadded-bs1", "hd256-the-loader-demotes"],
)
def test_a_pinned_fa4_the_backbone_load_keeps_off_varlen_rows_is_not_refused(batch_size, head_dim, loaded_backend):
    """Anti-vacuity: one unpadded row per batch trains on FA4's dense kernel, and at head_dim 256 the loader moves the
    pin to FA2 itself; the rule reads the loaded backbone's backend, not the request."""
    loads = _embedding_backbone_loads(
        batch_size=batch_size, pinned=FA4, loaded_backend=loaded_backend, head_dim=head_dim
    )
    assert [load.get("attn_default") for load in loads] == [FA4]


# --- fp32 -------------------------------------------------------------------------------------------------------


def test_an_fp32_run_on_varlen_rows_requests_nothing_in_every_script(tmp_path):
    """fp32 runs no flash kernel: each script hands the run's own dtype to the default, which then names nothing and
    leaves the softcap to the resolver's flex_attention."""
    fp32 = torch.float32
    sft = _sft_request(tmp_path, "packing: true\n", _gemma2(128), dtype=fp32)
    self_distill = _self_distill_request(tmp_path, "per_device_train_batch_size: 2\n", dtype=fp32)
    embedding = _embedding_backbone_loads(batch_size=2, dtype=fp32)
    assert [load.get("attn_default") for load in (*sft, *self_distill, *embedding)] == [None, None, None]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
