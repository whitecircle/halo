"""``inkling2_relative_flex`` off CUDA: its block mask against the dense mask it stands for, its logits against the
eager path on padded and packed rows, KV-cache decoding, the warm-up's coverage of the first training and scoring
rows, the family's short convolutions on packed rows, and the resolver's choice of the kernel."""

from __future__ import annotations

import contextlib
from unittest import mock

import pytest
import torch
from torch.nn.attention import flex_attention as flex_attention_module
from torch.nn.attention.flex_attention import create_block_mask, create_mask, flex_attention

from halo.model_loading.parallel import whole_model
from halo.models.attention import inkling2_relative_flex as kernel
from halo.models.attention import selection
from halo.models.attention.gpt_oss_sinks import SinksPolicy
from halo.models.attention.inkling2_relative_flex import (
    INKLING2_RELATIVE_FLEX,
    register_inkling2_relative_flex,
    warm_inkling2_relative_flex_kernels,
)
from halo.models.attention.selection import _TOOLKIT_ATTENTION_REGISTRARS, resolve_attn_implementation
from halo.models.families.inkling2 import apply_inkling2_patches
from halo.models.moe.family_specs import MOE_FAMILY_SPECS, find_moe_family_spec
from tests.common.flex_sliding import no_device_check
from tests.common.tiny_models import TINY_MOE_FAMILIES, tiny_family_model

# Past the tiny config's 16-token window and relative extent.
LENGTH = 48
DOCUMENTS = (20, 33, 11)
TOLERANCE = 2e-4


@pytest.fixture(autouse=True)
def patched_family():
    apply_inkling2_patches()
    register_inkling2_relative_flex()


@pytest.fixture
def fresh_dynamo():
    torch._dynamo.reset()
    yield
    torch._dynamo.reset()


def _model(dtype: torch.dtype = torch.float32):
    torch.manual_seed(0)
    return tiny_family_model(TINY_MOE_FAMILIES["inkling2_text"]).to(dtype)


def _ids(model, *shape, seed: int = 1) -> torch.Tensor:
    return torch.randint(3, model.config.vocab_size, shape, generator=torch.Generator().manual_seed(seed))


def _packed_positions(documents=DOCUMENTS) -> torch.Tensor:
    return torch.cat([torch.arange(n) for n in documents]).unsqueeze(0)


@contextlib.contextmanager
def kernel_on_cpu(calls: list | None = None):
    """The kernel on Dynamo's eager backend, which a CPU runs, with FlexAttention's refusal of a CPU backward lifted.
    ``calls`` receives the ``(queries, keys)`` of each compiled call."""
    compiled = torch.compile(flex_attention, dynamic=True, backend="aot_eager")

    def traced(query, key, value, **kwargs):
        if calls is not None:
            calls.append((query.shape[2], key.shape[2]))
        return compiled(query, key, value, **kwargs)

    with (
        mock.patch.object(flex_attention_module, "_validate_device", no_device_check),
        mock.patch.object(kernel, "_compiled_flex", traced),
    ):
        yield


def _logits(model, attn_implementation: str, input_ids, **kwargs) -> torch.Tensor:
    model.config._attn_implementation = attn_implementation
    with torch.no_grad():
        return model(input_ids=input_ids, use_cache=False, **kwargs).logits


def _gap(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a - b).abs().max() / b.abs().max()).item()


def _dense_reference(position_ids, padding_mask, window, batch, length) -> torch.Tensor:
    """``[batch, q_len, kv_len]``, written over the document index of each token as the eager mask reads it."""
    documents = (position_ids.expand(batch, -1) == 0).cumsum(dim=-1)
    index = torch.arange(length)
    keep = (index[:, None] >= index[None, :]) & (documents[:, :, None] == documents[:, None, :])
    if window is not None:
        keep = keep & (index[:, None] - index[None, :] < window)
    if padding_mask is not None:
        keep = keep & padding_mask.bool()[:, None, :]
    return keep


def _left_padded(batch: int, length: int) -> torch.Tensor:
    mask = torch.ones(batch, length, dtype=torch.long)
    mask[-1, :37] = 0
    return mask


def _right_padded(batch: int, length: int) -> torch.Tensor:
    mask = torch.ones(batch, length, dtype=torch.long)
    mask[0, -150:] = 0
    return mask


# Past FlexAttention's 128-token tile, so the block layout has full, partial and empty blocks.
_MASK_LENGTH = 520
_MASK_CASES = {
    "one document": (torch.arange(_MASK_LENGTH).unsqueeze(0), None, None, 1),
    "window": (torch.arange(_MASK_LENGTH).unsqueeze(0), None, 300, 1),
    "packed row": (_packed_positions((260, 219, 41)), None, None, 1),
    "packed row, window": (_packed_positions((260, 219, 41)), None, 300, 1),
    "left padding": (torch.arange(_MASK_LENGTH).unsqueeze(0), _left_padded(2, _MASK_LENGTH), None, 2),
    "right padding, window": (torch.arange(_MASK_LENGTH).unsqueeze(0), _right_padded(2, _MASK_LENGTH), 300, 2),
}


@pytest.mark.parametrize("case", list(_MASK_CASES))
def test_the_block_mask_is_the_dense_mask_of_the_documents_window_and_padding(case):
    position_ids, padding_mask, window, batch = _MASK_CASES[case]
    reference = _dense_reference(position_ids, padding_mask, window, batch, _MASK_LENGTH)

    def reference_mod(b, h, q_idx, kv_idx):
        return reference[b, q_idx, kv_idx]

    expected = create_block_mask(reference_mod, B=batch, H=None, Q_LEN=_MASK_LENGTH, KV_LEN=_MASK_LENGTH, device="cpu")
    block_masks = kernel._row_block_masks(position_ids, padding_mask, window, batch, _MASK_LENGTH)
    assert len(block_masks) == batch
    for row, block_mask in enumerate(block_masks):
        per_score = create_mask(block_mask.mask_mod, 1, 1, _MASK_LENGTH, _MASK_LENGTH, device="cpu")
        assert torch.equal(per_score[0, 0], reference[row])
        assert torch.equal(block_mask.to_dense()[0], expected.to_dense()[row])
        assert torch.equal(block_mask.kv_num_blocks[0], expected.kv_num_blocks[row])
        assert torch.equal(block_mask.full_kv_num_blocks[0], expected.full_kv_num_blocks[row])
    assert expected.full_kv_num_blocks.sum() > 0, "premise: the case has full blocks"


def test_a_row_shorter_than_two_tiles_pads_with_tokens_that_reach_themselves_alone():
    length, padded = 40, kernel._MIN_ROW_LENGTH
    (block_mask,) = kernel._row_block_masks(_packed_positions((25, 15)), None, 10, 1, length)
    per_score = create_mask(block_mask.mask_mod, 1, 1, padded, padded, device="cpu")[0, 0]
    reference = _dense_reference(_packed_positions((25, 15)), None, 10, 1, length)[0]
    assert torch.equal(per_score[:length, :length], reference)
    assert not per_score[:length, length:].any() and not per_score[length:, :length].any()
    assert torch.equal(per_score[length:, length:], torch.eye(padded - length, dtype=torch.bool))


def test_one_forward_builds_each_block_mask_once():
    position_ids = _packed_positions()
    first = kernel._row_block_masks(position_ids, None, 16, 1, sum(DOCUMENTS))
    assert kernel._row_block_masks(position_ids, None, 16, 1, sum(DOCUMENTS)) is first
    assert kernel._row_block_masks(position_ids, None, None, 1, sum(DOCUMENTS)) is not first


def test_a_padded_batch_matches_eager():
    model = _model()
    input_ids = _ids(model, 2, LENGTH)
    attention_mask = torch.ones_like(input_ids)
    attention_mask[1, :7] = 0
    eager = _logits(model, "eager", input_ids, attention_mask=attention_mask)
    calls: list = []
    with kernel_on_cpu(calls):
        flex = _logits(model, INKLING2_RELATIVE_FLEX, input_ids, attention_mask=attention_mask)
    assert len(calls) == 2 * model.config.num_hidden_layers, "one call per row and layer"
    real = attention_mask.bool()
    assert _gap(flex[real], eager[real]) < TOLERANCE


@pytest.mark.parametrize("attn_implementation", ["eager", INKLING2_RELATIVE_FLEX])
def test_a_packed_row_scores_each_document_as_that_document_alone(attn_implementation):
    """On eager the attention reads the documents off its own mask, so the row isolates through the convolution
    patches alone."""
    model = _model()
    input_ids = _ids(model, 1, sum(DOCUMENTS), seed=2)
    with kernel_on_cpu():
        packed = _logits(model, attn_implementation, input_ids, position_ids=_packed_positions())
        alone = torch.cat(
            [_logits(model, attn_implementation, document) for document in input_ids.split(DOCUMENTS, dim=1)], dim=1
        )
        crossed = _logits(model, attn_implementation, input_ids)
    assert _gap(packed, alone) < TOLERANCE
    assert _gap(crossed, alone) > 100 * TOLERANCE, "premise: the row read as one document differs"


def test_packed_gradients_are_the_documents_alone_on_eager():
    """The convolution patches carry the gradient of each document to its own tokens only."""
    model = _model()
    model.config._attn_implementation = "eager"
    input_ids = _ids(model, 1, sum(DOCUMENTS), seed=2)

    def gradients(ids, **kwargs) -> dict[str, torch.Tensor]:
        model.zero_grad(set_to_none=True)
        model(input_ids=ids, use_cache=False, **kwargs).logits.float().square().mean(-1).sum().backward()
        return {name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None}

    packed = gradients(input_ids, position_ids=_packed_positions())
    alone: dict[str, torch.Tensor] = {}
    for document in input_ids.split(DOCUMENTS, dim=1):
        for name, grad in gradients(document).items():
            alone[name] = alone.get(name, 0) + grad
    assert packed.keys() == alone.keys()
    for name, grad in alone.items():
        gap = ((packed[name] - grad).abs().max() / grad.abs().max().clamp(min=1e-6)).item()
        assert gap < TOLERANCE, f"{name}: {gap:.2e}"


@pytest.mark.parametrize("attn_implementation", ["eager", INKLING2_RELATIVE_FLEX])
def test_a_batch_whose_rows_restart_is_refused(attn_implementation):
    """The kernel isolates each row's documents, the short convolutions those of one ``[1, N]`` row only."""
    model = _model()
    input_ids = _ids(model, 2, sum(DOCUMENTS))
    with kernel_on_cpu(), pytest.raises(ValueError, match=r"position_ids of shape \(2, 64\) restart inside a row"):
        _logits(model, attn_implementation, input_ids, position_ids=_packed_positions().expand(2, -1).contiguous())


def test_a_batch_with_one_document_per_row_runs_with_per_row_positions():
    model = _model()
    input_ids = _ids(model, 2, LENGTH)
    position_ids = torch.arange(LENGTH).unsqueeze(0).expand(2, -1).contiguous()
    with kernel_on_cpu():
        with_positions = _logits(model, INKLING2_RELATIVE_FLEX, input_ids, position_ids=position_ids)
        without = _logits(model, INKLING2_RELATIVE_FLEX, input_ids)
    assert torch.equal(with_positions, without)


def test_generation_decodes_on_the_eager_attention():
    """The prefill is a full-sequence forward on the kernel; each decode step has one query over the cached keys."""
    model = _model().eval()
    input_ids = _ids(model, 2, 30)
    attention_mask = torch.ones_like(input_ids)
    attention_mask[1, :5] = 0
    new_tokens = 8

    def generate(attn_implementation: str) -> torch.Tensor:
        model.config._attn_implementation = attn_implementation
        return model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=new_tokens,
            min_new_tokens=new_tokens,
            do_sample=False,
            pad_token_id=0,
        )

    eager = generate("eager")
    calls: list = []
    with kernel_on_cpu(calls):
        flex = generate(INKLING2_RELATIVE_FLEX)
    assert flex.shape == (2, 30 + new_tokens)
    assert torch.equal(flex, eager)
    prefill = [(kernel._MIN_ROW_LENGTH,) * 2] * (2 * model.config.num_hidden_layers)
    assert calls == prefill, "the prefill alone runs the compiled kernel, each row padded past one tile"


def _train_only_output_projections(model) -> None:
    """The first layer's attention inputs then require no gradient while grad mode is on."""
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.endswith("o_proj.weight"))


@pytest.mark.usefixtures("fresh_dynamo")
@pytest.mark.parametrize("frozen", [False, True], ids=["full", "output projections only"])
def test_a_warmed_model_compiles_nothing_on_its_first_rows(frozen):
    """Training and scoring, on a packed ``[1, N]`` row and on a padded ``[B, L]`` batch."""
    model = _model(torch.bfloat16).train()
    model.config._attn_implementation = INKLING2_RELATIVE_FLEX
    if frozen:
        _train_only_output_projections(model)
    padded_ids = _ids(model, 3, 200)
    attention_mask = torch.ones_like(padded_ids)
    attention_mask[1, :9] = 0
    rows = (
        {"input_ids": _ids(model, 1, 301, seed=2), "position_ids": _packed_positions((150, 111, 40))},
        {"input_ids": padded_ids, "attention_mask": attention_mask},
        {"input_ids": _ids(model, 1, 77, seed=3)},
    )
    calls: list = []
    with kernel_on_cpu(calls):
        warm_inkling2_relative_flex_kernels(model, dtype=torch.bfloat16, device=torch.device("cpu"))
        # Sliding and full layers share the tiny config's heads and relative extent: one signature, training and
        # forward-only.
        assert len(calls) == 2
        with mock.patch.object(torch._dynamo.config, "error_on_recompile", True):
            for row in rows:
                for train in (True, False):
                    calls.clear()
                    with torch.set_grad_enabled(train):
                        logits = model(**row, use_cache=False).logits
                    if train:
                        logits.float().square().mean().backward()
                    assert len(calls) == row["input_ids"].shape[0] * model.config.num_hidden_layers


def test_the_warmup_skips_a_model_on_another_attention():
    model = _model()
    model.config._attn_implementation = "eager"
    with mock.patch.object(kernel, "inkling2_relative_flex_attention", side_effect=AssertionError("warmed")):
        warm_inkling2_relative_flex_kernels(model, dtype=torch.float32, device=torch.device("cpu"))
        model.config._attn_implementation = INKLING2_RELATIVE_FLEX
        with pytest.raises(AssertionError, match="warmed"):
            warm_inkling2_relative_flex_kernels(model, dtype=torch.float32, device=torch.device("cpu"))


def test_every_load_warms_the_kernel_behind_the_fence(monkeypatch):
    warmed = []
    monkeypatch.setattr(whole_model, "get_global_world_size", lambda: 2)
    monkeypatch.setattr(whole_model, "barrier_on_exit", lambda _: contextlib.nullcontext())
    for other in ("warm_fa4_kernels", "warm_flex_sliding_kernels", "warm_masked_matmul_kernels"):
        monkeypatch.setattr(whole_model, other, lambda *args, **kwargs: None)
    monkeypatch.setattr(
        whole_model, "warm_inkling2_relative_flex_kernels", lambda model, *, dtype: warmed.append((model, dtype))
    )
    whole_model.warm_attention_kernels("model", dtype=torch.bfloat16)
    assert warmed == [("model", torch.bfloat16)]


def test_every_spec_kernel_has_its_registrar():
    named = {spec.toolkit_attention for spec in MOE_FAMILY_SPECS if spec.toolkit_attention}
    assert named == set(_TOOLKIT_ATTENTION_REGISTRARS)
    assert find_moe_family_spec("inkling2_text").toolkit_attention == INKLING2_RELATIVE_FLEX
    for name, register in _TOOLKIT_ATTENTION_REGISTRARS.items():
        assert register() == name


def _resolve(monkeypatch, pinned: str | None, *, cuda: bool) -> str:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
    monkeypatch.setattr(selection, "_detect_attention_impl", lambda: "flash_attention_2")
    config = _model().config
    return resolve_attn_implementation(config, pinned, torch.bfloat16, sinks_policy=SinksPolicy.LIVE)


@pytest.mark.parametrize(
    ("pinned", "cuda", "expected"),
    [
        (None, True, INKLING2_RELATIVE_FLEX),
        (INKLING2_RELATIVE_FLEX, True, INKLING2_RELATIVE_FLEX),
        ("sdpa", True, INKLING2_RELATIVE_FLEX),
        ("eager", True, "eager"),
        (None, False, "eager"),
        ("sdpa", False, "eager"),
        (INKLING2_RELATIVE_FLEX, False, "eager"),
    ],
)
def test_the_family_resolves_to_its_kernel_on_cuda_unless_eager_is_pinned(monkeypatch, pinned, cuda, expected):
    assert _resolve(monkeypatch, pinned, cuda=cuda) == expected


def test_a_pin_the_kernel_replaces_is_named_in_a_warning(monkeypatch, caplog):
    with caplog.at_level("WARNING", logger=selection.logger.name):
        _resolve(monkeypatch, "sdpa", cuda=True)
        assert "ignoring attn_implementation='sdpa'" in caplog.text
        caplog.clear()
        _resolve(monkeypatch, None, cuda=True)
        _resolve(monkeypatch, "eager", cuda=True)
        assert "ignoring attn_implementation" not in caplog.text


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
