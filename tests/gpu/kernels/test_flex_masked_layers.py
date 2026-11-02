#!/usr/bin/env python
"""The flex-sliding attention on every masked causal layer of a sliding-window model: Step-3.7's full layers
take FlexAttention under their packed (document-causal) or padded mask beside its sliding layers, and Gemma 4's
head_dim-512 global layers past the matmul budget take mem-efficient attention per packed document.

* A tiny Step-3.7 (full and sliding layers, per-layer head counts) on packed rows longer than its window,
  and on a right-padded batch, reproduces the SDPA model's loss and every gradient, with every layer on
  FlexAttention; on a single unpadded document its full layers, which then get no mask, keep SDPA.
* At Step-3.7's real attention shapes (head_dim 128, 12 and 8 query heads per KV head, window 512, bf16),
  the compiled kernel's output and input gradients against a per-document fp64 reference are no worse than
  mem-efficient SDPA's on the dense mask, full and sliding calls alike.
* A packed document's output is bitwise independent of every other document in its row.
* After the load-time warm-up, no packed length, batch, grad mode or ``no_grad`` pass compiles, nor a call
  carrying a tensor-parallel rank's share of a module's heads, which keeps SDPA instead.
* A tiny Gemma 4 with 512-wide global layers, past the matmul budget, reproduces the SDPA model's loss and every
  gradient on packed rows with its global layers on the per-document kernel, compiling nothing after the
  warm-up; a padded row's global layers keep SDPA.
* At Gemma 4's global shapes (head_dim 512, 8 query heads per KV head, ``scaling`` 1.0 over RMS-normed heads,
  bf16) on two packed rows, the per-document kernel's output and input gradients against a per-document fp64
  reference are no worse than mem-efficient SDPA's on the dense mask, and a document's output is bitwise
  independent of the others.

Run: torchrun --nproc_per_node=1 tests/gpu/kernels/test_flex_masked_layers.py
"""

from unittest.mock import patch

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.attention.flex_attention import flex_attention
from transformers import Gemma4ForCausalLM, Gemma4TextConfig, Step3p7Config, Step3p7ForConditionalGeneration

from halo.models.attention import sdpa_flex_sliding
from halo.models.attention.backend_setup import patch_sdpa_for_wide_heads
from halo.models.attention.sdpa_flex_sliding import register_flex_sliding_attention, warm_flex_sliding_kernels
from tests.common.attention_stubs import causal_attention
from tests.common.harness import gpu_test_main, record_check
from tests.common.models import TINY_GEMMA4_WIDE_HEAD_CONFIG, TINY_STEP3P7_CONFIG, TINY_STEP3P7_VISION_CONFIG
from tests.common.numerics import fro_rel_err

WINDOW = TINY_STEP3P7_CONFIG["sliding_window"]
LAYERS = TINY_STEP3P7_CONFIG["num_hidden_layers"]
PACKINGS = ([150, 30, 90], [100, 7, 200])
GEMMA4_LAYER_TYPES = ["sliding_attention", "full_attention"]
GEMMA4_GLOBAL_LAYERS = GEMMA4_LAYER_TYPES.count("full_attention")
GEMMA4_VOCAB = TINY_GEMMA4_WIDE_HEAD_CONFIG["vocab_size"]


def _step3p7(attn: str) -> Step3p7ForConditionalGeneration:
    config = Step3p7Config(text_config=dict(TINY_STEP3P7_CONFIG), vision_config=dict(TINY_STEP3P7_VISION_CONFIG))
    config.text_config._attn_implementation = attn
    config.vision_config._attn_implementation = "sdpa"
    return Step3p7ForConditionalGeneration(config).cuda()


def _counting(function, calls: list):
    """``function``, appending to ``calls`` on each call: proof the path under test ran, where a silent
    fallback to SDPA would reproduce the SDPA model exactly."""
    return lambda *args, **kwargs: calls.append(1) or function(*args, **kwargs)


def _packed(lengths: list[int], vocab: int = TINY_STEP3P7_CONFIG["vocab_size"]) -> dict:
    generator = torch.Generator().manual_seed(sum(lengths))
    ids = torch.randint(3, vocab, (1, sum(lengths)), generator=generator).cuda()
    position_ids = torch.cat([torch.arange(n) for n in lengths]).unsqueeze(0).cuda()
    return {"input_ids": ids, "position_ids": position_ids, "labels": ids}


def _padded(lengths: list[int], vocab: int = TINY_STEP3P7_CONFIG["vocab_size"]) -> dict:
    generator = torch.Generator().manual_seed(sum(lengths) + 1)
    seq = max(lengths)
    ids = torch.randint(3, vocab, (len(lengths), seq), generator=generator).cuda()
    attention_mask = (torch.arange(seq)[None] < torch.tensor(lengths)[:, None]).long().cuda()
    return {"input_ids": ids, "attention_mask": attention_mask, "labels": ids.masked_fill(attention_mask == 0, -100)}


def _loss_and_grads(model, batch: dict):
    model.zero_grad(set_to_none=True)
    out = model(**batch, use_cache=False)
    out.loss.backward()
    return out.loss.detach(), {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None}


def _document_mask(lengths: list[int], window: int | None) -> torch.Tensor:
    document = torch.cat([torch.full((n,), i) for i, n in enumerate(lengths)]).cuda()
    pos = torch.arange(document.numel(), device="cuda")
    mask = (document[:, None] == document[None]) & (pos[None] <= pos[:, None])
    return mask & (pos[:, None] - pos[None] < window) if window else mask


def test_step3p7_matches_sdpa(batch: dict, flex_calls: int):
    """Uncompiled (fp32-exact) FlexAttention in place of the compiled kernel: the model must reproduce the SDPA
    model's loss and every gradient to fp32 round-off, with ``flex_calls`` layers on FlexAttention."""
    name = register_flex_sliding_attention()
    torch.manual_seed(0)
    reference = _step3p7("sdpa").float()
    candidate = _step3p7(name).float()
    candidate.load_state_dict(reference.state_dict())
    calls = []
    with patch.object(sdpa_flex_sliding, "_compiled_flex", _counting(flex_attention, calls)):
        loss, grads = _loss_and_grads(candidate, batch)
    ref_loss, ref_grads = _loss_and_grads(reference, batch)
    assert len(calls) == flex_calls, f"{len(calls)} FlexAttention calls, want {flex_calls}"
    torch.testing.assert_close(loss, ref_loss, rtol=1e-5, atol=1e-6)
    assert grads.keys() == ref_grads.keys()
    for key in ref_grads:
        assert fro_rel_err(grads[key], ref_grads[key]) < 1e-3, key


def _errors(out, grads, expected, expected_grads):
    return [fro_rel_err(out, expected)] + [fro_rel_err(g, e) for g, e in zip(grads, expected_grads, strict=True)]


def test_real_shapes_match_fp64_no_worse_than_sdpa():
    """Step-3.7's attention shapes on a packed 4,096-token row, compiled kernel, bf16: against fp64 per-document
    attention, the output and the query, key and value gradients stay within 1.5x the error of mem-efficient
    SDPA over the same dense mask. Head counts are cut to one KV head per call (the ratios kept) so the fp64
    reference fits; the kernel's numerics do not depend on the count."""
    lengths = [1500, 900, 1200, 496]
    seq = sum(lengths)
    for group, window in ((12, 512), (8, None)):
        generator = torch.Generator(device="cuda").manual_seed(group)
        shape = {"device": "cuda", "dtype": torch.bfloat16, "generator": generator}
        q = torch.randn(1, group, seq, 128, **shape).requires_grad_()
        k = torch.randn(1, 1, seq, 128, **shape).requires_grad_()
        v = torch.randn(1, 1, seq, 128, **shape).requires_grad_()
        grad = torch.randn(1, seq, group, 128, **shape)
        mask = _document_mask(lengths, window)
        scale = 128**-0.5

        q64, k64, v64 = (t.detach().double().requires_grad_() for t in (q, k, v))
        scores = (q64 @ k64.transpose(-1, -2)) * scale
        expected = (torch.softmax(scores.masked_fill(~mask, float("-inf")), -1) @ v64).transpose(1, 2)
        expected.backward(grad.double())
        expected_grads = (q64.grad, k64.grad, v64.grad)
        del scores

        out, _ = sdpa_flex_sliding.flex_sliding_attention(
            causal_attention(group, 1, 128, scale), q, k, v, mask[None, None], scaling=scale, sliding_window=window
        )
        out.backward(grad)
        flex = _errors(out, (q.grad, k.grad, v.grad), expected, expected_grads)

        q2, k2, v2 = (t.detach().clone().requires_grad_() for t in (q, k, v))
        with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION]):
            ref = torch.nn.functional.scaled_dot_product_attention(
                q2, k2.expand(-1, group, -1, -1), v2.expand(-1, group, -1, -1), attn_mask=mask, scale=scale
            ).transpose(1, 2)
        ref.backward(grad)
        sdpa = _errors(ref, (q2.grad, k2.grad, v2.grad), expected, expected_grads)
        for label, got, floor in zip(("output", "dq", "dk", "dv"), flex, sdpa, strict=True):
            assert got <= 1.5 * floor + 1e-4, f"window={window} {label}: flex {got:.2e} against SDPA's {floor:.2e}"


def test_documents_are_isolated_bitwise():
    """Perturbing one packed document's queries, keys and values leaves every other document's output bitwise
    unchanged, on a full call and a sliding call (the premise: the perturbed document's own output moves)."""
    lengths = [700, 300, 500]
    seq, first = sum(lengths), lengths[0]
    for window in (None, 256):
        generator = torch.Generator(device="cuda").manual_seed(seq)
        shape = {"device": "cuda", "dtype": torch.bfloat16, "generator": generator}
        q, k, v = (torch.randn(1, heads, seq, 128, **shape) for heads in (8, 2, 2))
        mask = _document_mask(lengths, window)[None, None]
        with torch.no_grad():
            out, _ = sdpa_flex_sliding.flex_sliding_attention(
                causal_attention(8, 2, 128), q, k, v, mask, scaling=1.0, sliding_window=window
            )
            q[:, :, :first] += 1.0
            k[:, :, :first] -= 1.0
            v[:, :, :first] *= 2.0
            moved, _ = sdpa_flex_sliding.flex_sliding_attention(
                causal_attention(8, 2, 128), q, k, v, mask.clone(), scaling=1.0, sliding_window=window
            )
        assert not torch.equal(moved[:, :first], out[:, :first]), "premise: the perturbed document changed"
        assert torch.equal(moved[:, first:], out[:, first:]), f"window={window}: another document moved"


def test_one_warmup_serves_a_step3p7_run():
    """The warm-up compiles a training and a forward-only graph per call the model's layers make (a full and a
    sliding one here); after it, forwards and backwards over other packed lengths and a padded batch, and
    forward-only passes, compile nothing."""
    name = register_flex_sliding_attention()
    torch.manual_seed(0)
    model = _step3p7(name).to(torch.bfloat16)
    torch._dynamo.reset()
    counters = torch._dynamo.utils.counters
    counters.clear()
    warm_flex_sliding_kernels(model, dtype=torch.bfloat16)
    calls = sdpa_flex_sliding.flex_attention_calls(model)
    assert len(calls) == 2, calls  # premise: one full and one sliding signature
    flex = []
    with (
        patch.object(torch._dynamo.config, "error_on_recompile", True),
        patch.object(sdpa_flex_sliding, "_compiled_flex", _counting(sdpa_flex_sliding._compiled_flex, flex)),
    ):
        for batch in (_packed([300, 150, 40]), _packed([129]), _packed([700]), _padded([260, 140])):
            _loss_and_grads(model, batch)
            with torch.no_grad():
                model(**batch, use_cache=False)
    assert flex, "premise: the run took FlexAttention"
    graphs = counters["stats"]["unique_graphs"]
    assert graphs == 2 * len(calls), f"{graphs} compiled graphs; the warm-up compiles two per call"


def test_a_tensor_parallel_share_of_the_heads_compiles_nothing():
    """Under tensor parallelism a rank's call carries its share of the module's heads (tp=2 here: half the
    query and half the KV heads), which the warm-up, reading the module, does not compile. After the warm-up
    such a call, forward and backward, must compile nothing (Dynamo raises on any compile here) and match SDPA
    over the same mask, while the module's own call runs FlexAttention (the premise)."""
    name = register_flex_sliding_attention()
    torch.manual_seed(0)
    model = _step3p7(name).to(torch.bfloat16)
    torch._dynamo.reset()
    counters = torch._dynamo.utils.counters
    counters.clear()
    warm_flex_sliding_kernels(model, dtype=torch.bfloat16)
    warmed = counters["stats"]["unique_graphs"]
    module, call = next(
        (m, c)
        for m, c in ((m, sdpa_flex_sliding._flex_call(m)) for m in model.modules())
        if c is not None and c.heads % 2 == 0 and c.kv_heads % 2 == 0
    )
    heads, kv_heads, dim, window, scaling = call.heads, call.kv_heads, call.head_dim, call.window, call.scaling
    lengths = [200, 100]
    seq = sum(lengths)
    mask = _document_mask(lengths, window)
    for tp, flex_calls in ((1, 1), (2, 0)):
        generator = torch.Generator(device="cuda").manual_seed(tp)
        shape = {"device": "cuda", "dtype": torch.bfloat16, "generator": generator}
        q = torch.randn(1, heads // tp, seq, dim, **shape).requires_grad_()
        k = torch.randn(1, kv_heads // tp, seq, dim, **shape).requires_grad_()
        v = torch.randn(1, kv_heads // tp, seq, dim, **shape).requires_grad_()
        flex = []
        with (
            patch.object(torch._dynamo.config, "error_on_recompile", True),
            patch.object(sdpa_flex_sliding, "_compiled_flex", _counting(sdpa_flex_sliding._compiled_flex, flex)),
        ):
            out, _ = sdpa_flex_sliding.flex_sliding_attention(
                module, q, k, v, mask[None, None], scaling=scaling, sliding_window=window
            )
            out.sum().backward()
        assert len(flex) == flex_calls, f"tp={tp}: {len(flex)} FlexAttention calls, want {flex_calls}"
        group = q.shape[1] // k.shape[1]
        ref = torch.nn.functional.scaled_dot_product_attention(
            q.detach(),
            k.detach().repeat_interleave(group, 1),
            v.detach().repeat_interleave(group, 1),
            attn_mask=mask,
            scale=scaling,
        ).transpose(1, 2)
        assert fro_rel_err(out.detach(), ref) < 2e-2, f"tp={tp}: differs from SDPA"
    graphs = counters["stats"]["unique_graphs"]
    assert graphs == warmed, f"{graphs} compiled graphs after the warm-up's {warmed}"


def _gemma4(attn: str) -> Gemma4ForCausalLM:
    config = Gemma4TextConfig(**TINY_GEMMA4_WIDE_HEAD_CONFIG, layer_types=GEMMA4_LAYER_TYPES)
    config._attn_implementation = attn
    return Gemma4ForCausalLM(config).cuda()


def test_gemma4_packed_globals_match_sdpa(batch: dict, varlen_calls: int):
    """Past the matmul budget (patched to 0) the 512-wide global layers of a packed row must run the per-document
    kernel, ``varlen_calls`` times, and the model must reproduce the SDPA model's loss and every gradient (fp32;
    both on the mem-efficient kernel, the process pinned as the Gemma 4 loader pins it)."""
    patch_sdpa_for_wide_heads()
    name = register_flex_sliding_attention()
    torch.manual_seed(0)
    reference = _gemma4("sdpa").float()
    candidate = _gemma4(name).float()
    candidate.load_state_dict(reference.state_dict())
    calls = []
    varlen = _counting(sdpa_flex_sliding._VarlenCausalAttention.apply, calls)
    with (
        patch.object(sdpa_flex_sliding, "MATMUL_GLOBAL_BUDGET_BYTES", 0),
        patch.object(sdpa_flex_sliding, "_compiled_flex", flex_attention),
        patch.object(sdpa_flex_sliding._VarlenCausalAttention, "apply", varlen),
    ):
        loss, grads = _loss_and_grads(candidate, batch)
    ref_loss, ref_grads = _loss_and_grads(reference, batch)
    assert len(calls) == varlen_calls, f"{len(calls)} per-document calls, want {varlen_calls}"
    torch.testing.assert_close(loss, ref_loss, rtol=1e-5, atol=1e-6)
    assert grads.keys() == ref_grads.keys()
    for key in ref_grads:
        assert fro_rel_err(grads[key], ref_grads[key]) < 1e-3, key


def test_a_warmed_gemma4_past_the_budget_runs_without_compiling():
    """The per-document kernel compiles nothing: after the warm-up, packed forwards and backwards over several
    lengths and a forward-only pass, with the global layers on it, leave the warm-up's graphs the only ones."""
    patch_sdpa_for_wide_heads()
    name = register_flex_sliding_attention()
    torch.manual_seed(0)
    model = _gemma4(name).to(torch.bfloat16)
    torch._dynamo.reset()
    counters = torch._dynamo.utils.counters
    counters.clear()
    warm_flex_sliding_kernels(model, dtype=torch.bfloat16)
    calls = []
    with (
        patch.object(torch._dynamo.config, "error_on_recompile", True),
        patch.object(sdpa_flex_sliding, "MATMUL_GLOBAL_BUDGET_BYTES", 0),
        patch.object(
            sdpa_flex_sliding._VarlenCausalAttention,
            "apply",
            _counting(sdpa_flex_sliding._VarlenCausalAttention.apply, calls),
        ),
    ):
        for lengths in ([300, 150, 40], [129, 1, 64], [700, 5]):
            batch = _packed(lengths, GEMMA4_VOCAB)
            _loss_and_grads(model, batch)
            with torch.no_grad():
                model(**batch, use_cache=False)
    assert len(calls) == 3 * 2 * GEMMA4_GLOBAL_LAYERS, "premise: every global layer ran the per-document kernel"
    graphs = counters["stats"]["unique_graphs"]
    assert graphs == 2 * len(sdpa_flex_sliding.flex_attention_calls(model)), f"{graphs} compiled graphs"


def _rms_normed(*shape, generator) -> torch.Tensor:
    x = torch.randn(*shape, device="cuda", generator=generator)
    return (x * x.pow(2).mean(-1, keepdim=True).rsqrt()).bfloat16()


def _per_document_reference(q, k, v, rows: list[list[int]], scale: float) -> torch.Tensor:
    """fp64 causal attention run on each document alone (KV repeated per group), ``[B, S, H, D]``."""
    group = q.shape[1] // k.shape[1]
    out = []
    for row, lengths in enumerate(rows):
        documents, start = [], 0
        for n in lengths:
            span = slice(start, start + n)
            documents.append(
                torch.nn.functional.scaled_dot_product_attention(
                    q[row : row + 1, :, span],
                    k[row : row + 1, :, span].repeat_interleave(group, 1),
                    v[row : row + 1, :, span].repeat_interleave(group, 1),
                    is_causal=True,
                    scale=scale,
                )
            )
            start += n
        out.append(torch.cat(documents, dim=2))
    return torch.cat(out).transpose(1, 2)


def _packed_rows_mask(rows: list[list[int]]) -> torch.Tensor:
    return torch.stack([_document_mask(lengths, None) for lengths in rows])[:, None]


def test_gemma4_global_shapes_match_fp64_no_worse_than_sdpa():
    """Two packed 4,096-token rows (different splits, so the boundaries cross the batch), Gemma 4's global shapes,
    bf16, past the budget: against fp64 per-document attention the per-document kernel's output and query, key
    and value gradients stay within 1.5x the error of mem-efficient SDPA over the same dense mask. One KV head
    with its 8 query heads (Gemma 4's ratio) keeps the reference small."""
    rows = [[1500, 900, 1200, 496], [4000, 96]]
    group, seq, dim, scale = 8, 4096, 512, 1.0
    generator = torch.Generator(device="cuda").manual_seed(0)
    q = _rms_normed(2, group, seq, dim, generator=generator).requires_grad_()
    k = _rms_normed(2, 1, seq, dim, generator=generator).requires_grad_()
    v = _rms_normed(2, 1, seq, dim, generator=generator).requires_grad_()
    grad = torch.randn(2, seq, group, dim, device="cuda", generator=generator).bfloat16()
    mask = _packed_rows_mask(rows)

    q64, k64, v64 = (t.detach().double().requires_grad_() for t in (q, k, v))
    expected = _per_document_reference(q64, k64, v64, rows, scale)
    expected.backward(grad.double())
    expected_grads = (q64.grad, k64.grad, v64.grad)

    calls = []
    with (
        patch.object(sdpa_flex_sliding, "MATMUL_GLOBAL_BUDGET_BYTES", 0),
        patch.object(
            sdpa_flex_sliding._VarlenCausalAttention,
            "apply",
            _counting(sdpa_flex_sliding._VarlenCausalAttention.apply, calls),
        ),
    ):
        out, _ = sdpa_flex_sliding.flex_sliding_attention(
            causal_attention(group, 1, dim, scale), q, k, v, mask, scaling=scale
        )
    assert calls == [1], "the call must take the per-document kernel"
    out.backward(grad)
    varlen = _errors(out, (q.grad, k.grad, v.grad), expected, expected_grads)

    q2, k2, v2 = (t.detach().clone().requires_grad_() for t in (q, k, v))
    with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION]):
        ref = torch.nn.functional.scaled_dot_product_attention(
            q2, k2.expand(-1, group, -1, -1), v2.expand(-1, group, -1, -1), attn_mask=mask, scale=scale
        ).transpose(1, 2)
    ref.backward(grad)
    sdpa = _errors(ref, (q2.grad, k2.grad, v2.grad), expected, expected_grads)
    for label, got, floor in zip(("output", "dq", "dk", "dv"), varlen, sdpa, strict=True):
        assert got <= 1.5 * floor + 1e-4, f"{label}: per-document {got:.2e} against SDPA's {floor:.2e}"


def test_gemma4_global_documents_are_isolated_bitwise():
    """Perturbing one packed document leaves every other document's per-document output bitwise unchanged,
    including the second row's (the premise: the perturbed document's own output moves)."""
    rows = [[700, 300, 500], [1000, 500]]
    seq, first, group = 1500, 700, 8
    generator = torch.Generator(device="cuda").manual_seed(1)
    q = _rms_normed(2, group, seq, 512, generator=generator)
    k = _rms_normed(2, 1, seq, 512, generator=generator)
    v = _rms_normed(2, 1, seq, 512, generator=generator)
    mask = _packed_rows_mask(rows)
    with torch.no_grad(), patch.object(sdpa_flex_sliding, "MATMUL_GLOBAL_BUDGET_BYTES", 0):
        out, _ = sdpa_flex_sliding.flex_sliding_attention(causal_attention(group, 1, 512), q, k, v, mask, scaling=1.0)
        q[0, :, :first] *= 2.0
        k[0, :, :first] *= -1.0
        v[0, :, :first] += 1.0
        moved, _ = sdpa_flex_sliding.flex_sliding_attention(
            causal_attention(group, 1, 512), q, k, v, mask.clone(), scaling=1.0
        )
    assert not torch.equal(moved[0, :first], out[0, :first]), "premise: the perturbed document changed"
    assert torch.equal(moved[0, first:], out[0, first:]), "another document of the row moved"
    assert torch.equal(moved[1], out[1]), "the other row moved"


@gpu_test_main(exact_world_size=1, prefix="flex_masked_layers")
def run(ctx) -> dict:
    checks: dict[str, bool] = {}
    for lengths in PACKINGS:
        record_check(
            checks,
            f"step3p7_packed_matches_sdpa[{lengths}]",
            lambda: test_step3p7_matches_sdpa(_packed(lengths), flex_calls=LAYERS),
        )
    record_check(
        checks,
        "step3p7_single_document_full_layers_keep_sdpa",
        lambda: test_step3p7_matches_sdpa(_packed([300]), flex_calls=LAYERS // 2),
    )
    # Each of the two rows runs alone, so every layer calls FlexAttention once per row.
    record_check(
        checks,
        "step3p7_padded_matches_sdpa",
        lambda: test_step3p7_matches_sdpa(_padded([260, 140]), flex_calls=2 * LAYERS),
    )
    record_check(checks, "real_shapes_match_fp64_no_worse_than_sdpa", test_real_shapes_match_fp64_no_worse_than_sdpa)
    record_check(checks, "documents_are_isolated_bitwise", test_documents_are_isolated_bitwise)
    record_check(checks, "one_warmup_serves_a_step3p7_run", test_one_warmup_serves_a_step3p7_run)
    record_check(
        checks,
        "a_tensor_parallel_share_of_the_heads_compiles_nothing",
        test_a_tensor_parallel_share_of_the_heads_compiles_nothing,
    )
    for lengths in PACKINGS:
        record_check(
            checks,
            f"gemma4_packed_globals_match_sdpa[{lengths}]",
            lambda: test_gemma4_packed_globals_match_sdpa(
                _packed(lengths, GEMMA4_VOCAB), varlen_calls=GEMMA4_GLOBAL_LAYERS
            ),
        )
    record_check(
        checks,
        "gemma4_padded_globals_keep_sdpa",
        lambda: test_gemma4_packed_globals_match_sdpa(_padded([260, 140], GEMMA4_VOCAB), varlen_calls=0),
    )
    record_check(
        checks,
        "a_warmed_gemma4_past_the_budget_runs_without_compiling",
        test_a_warmed_gemma4_past_the_budget_runs_without_compiling,
    )
    record_check(
        checks,
        "gemma4_global_shapes_match_fp64_no_worse_than_sdpa",
        test_gemma4_global_shapes_match_fp64_no_worse_than_sdpa,
    )
    record_check(
        checks, "gemma4_global_documents_are_isolated_bitwise", test_gemma4_global_documents_are_isolated_bitwise
    )
    return {"checks": checks}


if __name__ == "__main__":
    run()
