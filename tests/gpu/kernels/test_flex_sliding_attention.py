#!/usr/bin/env python
"""The flex-sliding attention (sliding layers on FlexAttention, mem-efficient-only global layers on matmul
attention within a memory budget) reproduces the SDPA model exactly in what it attends to.

Tiny random models run packed rows (three documents, per-document ``position_ids``, no KV cache so
transformers detects the packing) longer than their sliding window, once under ``sdpa`` and once under the
flex-sliding implementation: Gemma 4 (alternating sliding and global layers, the process pinned to
mem-efficient SDPA) and Mistral (a window on every layer). The loss and every parameter gradient must
agree: a block mask that leaked across a document boundary, dropped the window, or went stale between two
different masks changes them.

Run: torchrun --nproc_per_node=1 tests/gpu/kernels/test_flex_sliding_attention.py
"""

import os
import weakref
from unittest.mock import patch

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.attention.flex_attention import flex_attention
from transformers import Gemma4ForCausalLM, Gemma4TextConfig, GptOssConfig, MistralConfig, MistralForCausalLM

from src.models.patches import flex_sliding_attention
from src.models.patches.attention import head_dim_exceeds_flash, patch_sdpa_for_wide_heads
from src.models.patches.flex_sliding_attention import (
    FLEX_SLIDING,
    register_flex_sliding_attention,
    reject_full_determinism_after_warmup,
    resolve_flex_sliding_attn_implementation,
    warmup_flex_sliding_kernels,
)
from tests.common.harness import gpu_test_main, record_check
from tests.common.utils import fro_rel_err

WINDOW = 32
PACKINGS = ([100, 7, 45], [150])


def _config(attn: str, global_head_dim: int = 128) -> Gemma4TextConfig:
    config = Gemma4TextConfig(
        vocab_size=256,
        hidden_size=128,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
        global_head_dim=global_head_dim,
        num_global_key_value_heads=1,
        sliding_window=WINDOW,
        layer_types=["sliding_attention", "sliding_attention", "full_attention", "full_attention"],
        enable_moe_block=False,
        hidden_size_per_layer_input=0,
        num_kv_shared_layers=0,
    )
    config._attn_implementation = attn
    return config


def _mistral_config(attn: str) -> MistralConfig:
    config = MistralConfig(
        vocab_size=256,
        hidden_size=128,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        sliding_window=WINDOW,
    )
    config._attn_implementation = attn
    return config


def _causal_module(num_key_value_groups: int = 2) -> torch.nn.Module:
    """A stand-in decoder attention module: causal, with the GQA group count SDPA's KV repeat reads."""
    module = torch.nn.Module()
    module.is_causal = True
    module.num_key_value_groups = num_key_value_groups
    return module


def _packed_batch(lengths: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(sum(lengths))
    ids = torch.randint(3, 256, (1, sum(lengths)), generator=generator)
    position_ids = torch.cat([torch.arange(n) for n in lengths]).unsqueeze(0)
    return ids.cuda(), position_ids.cuda()


def _sliding_mask(lengths: list[int], window: int = WINDOW) -> torch.Tensor:
    """The dense ``[Q, K]`` mask of packed documents under a causal sliding window."""
    document = torch.cat([torch.full((n,), i) for i, n in enumerate(lengths)]).cuda()
    pos = torch.arange(document.numel(), device="cuda")
    return (document[:, None] == document[None]) & (pos[None] <= pos[:, None]) & (pos[:, None] - pos[None] < window)


def _loss_and_grads(model, ids, position_ids):
    model.zero_grad(set_to_none=True)
    out = model(input_ids=ids, position_ids=position_ids, labels=ids, use_cache=False)
    out.loss.backward()
    return out.loss.detach(), {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None}


def _counting(function, calls: list):
    """``function``, appending to ``calls`` on each call: proof the path under test ran, where a silent
    fallback to SDPA would reproduce the SDPA model exactly."""
    return lambda *args, **kwargs: calls.append(1) or function(*args, **kwargs)


def _assert_model_matches(reference, candidate, lengths):
    ids, position_ids = _packed_batch(lengths)
    ref_loss, ref_grads = _loss_and_grads(reference, ids, position_ids)
    loss, grads = _loss_and_grads(candidate, ids, position_ids)
    torch.testing.assert_close(loss, ref_loss, rtol=1e-5, atol=1e-6)
    assert grads.keys() == ref_grads.keys()
    for key in ref_grads:
        assert fro_rel_err(grads[key], ref_grads[key]) < 1e-3, key


def test_flex_sliding_wiring_matches_sdpa(lengths):
    """Exact check of what the sliding layers attend to: with the uncompiled (fp32-exact) FlexAttention
    the model must reproduce the SDPA model's loss and every gradient to fp32 round-off."""
    calls = []
    with patch.object(flex_sliding_attention, "_compiled_flex", _counting(flex_attention, calls)):
        patch_sdpa_for_wide_heads()
        name = register_flex_sliding_attention()
        torch.manual_seed(0)
        reference = Gemma4ForCausalLM(_config("sdpa")).cuda().float()
        candidate = Gemma4ForCausalLM(_config(name)).cuda().float()
        candidate.load_state_dict(reference.state_dict())
        _assert_model_matches(reference, candidate, lengths)
    assert len(calls) == 2, "the two sliding layers must run FlexAttention on the one packed row"


def test_compiled_flex_kernel_matches_fp64_reference():
    """The compiled kernel the layers run, against an fp64 masked softmax over a packed sliding mask.
    Its fp32 dots run in TF32, so the bound is TF32's, well below any masking error (which is O(1))."""
    torch.manual_seed(0)
    lengths, heads, kv_heads, dim = [100, 7, 45], 4, 2, 64
    seq = sum(lengths)
    dense = _sliding_mask(lengths)
    q = torch.randn(1, heads, seq, dim, device="cuda", requires_grad=True)
    k = torch.randn(1, kv_heads, seq, dim, device="cuda", requires_grad=True)
    v = torch.randn(1, kv_heads, seq, dim, device="cuda", requires_grad=True)
    grad = torch.randn(1, seq, heads, dim, device="cuda")
    out, _ = flex_sliding_attention.flex_sliding_attention(
        _causal_module(), q, k, v, dense[None, None], scaling=1.0, sliding_window=WINDOW
    )
    out.backward(grad)

    q64, k64, v64 = (t.detach().double().requires_grad_(True) for t in (q, k, v))
    scores = q64 @ k64.repeat_interleave(heads // kv_heads, 1).transpose(-1, -2)
    expected = torch.softmax(scores.masked_fill(~dense, float("-inf")), -1) @ v64.repeat_interleave(
        heads // kv_heads, 1
    )
    expected.transpose(1, 2).backward(grad.double())
    assert fro_rel_err(out, expected.transpose(1, 2)) < 2e-3
    for got, want in ((q.grad, q64.grad), (k.grad, k64.grad), (v.grad, v64.grad)):
        assert fro_rel_err(got, want) < 5e-3


def test_tuned_tiles_match_sdpa_at_gemma4_shapes():
    """The production sliding shape (16 query / 8 KV heads, head_dim 256, window 1,024, bf16), where
    SM100+ runs the tuned tiles, over two rows of different packings: output and input gradients must
    match mem-efficient SDPA on the dense mask to bf16 accuracy."""
    torch.manual_seed(0)
    seq, window = 2048, 1024
    q = torch.randn(2, 16, seq, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(2, 8, seq, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn(2, 8, seq, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    grad = torch.randn_like(q)
    dense = torch.stack([_sliding_mask([seq], window), _sliding_mask([1500, 548], window)])[:, None]
    out, _ = flex_sliding_attention.flex_sliding_attention(
        _causal_module(), q, k, v, dense, scaling=1.0, sliding_window=window
    )
    out.backward(grad.transpose(1, 2))
    q2, k2, v2 = (t.detach().clone().requires_grad_(True) for t in (q, k, v))
    expected = torch.nn.functional.scaled_dot_product_attention(
        q2, k2.repeat_interleave(2, 1), v2.repeat_interleave(2, 1), attn_mask=dense, scale=1.0
    )
    expected.backward(grad)
    assert fro_rel_err(out, expected.transpose(1, 2)) < 2e-2
    for got, want in ((q.grad, q2.grad), (k.grad, k2.grad), (v.grad, v2.grad)):
        assert fro_rel_err(got, want) < 2e-2


def test_bf16_model_error_is_within_the_sdpa_noise_floor():
    """The production dtype end to end, compiled kernel included. bf16 gradients of two correct
    implementations already differ, so the bound is relative: against an fp32 SDPA reference, the flex
    model's bf16 error must stay within twice SDPA's own bf16 error, parameter by parameter."""
    patch_sdpa_for_wide_heads()
    name = register_flex_sliding_attention()
    torch.manual_seed(0)
    reference = Gemma4ForCausalLM(_config("sdpa")).cuda().float()
    ids, position_ids = _packed_batch([100, 7, 45])
    ref_loss, ref_grads = _loss_and_grads(reference, ids, position_ids)
    errors, calls = {}, []
    for attn in ("sdpa", name):
        model = Gemma4ForCausalLM(_config(attn)).cuda()
        model.load_state_dict(reference.state_dict())
        with patch.object(
            flex_sliding_attention, "_compiled_flex", _counting(flex_sliding_attention._compiled_flex, calls)
        ):
            loss, grads = _loss_and_grads(model.to(torch.bfloat16), ids, position_ids)
        assert len(calls) == (2 if attn == name else 0), f"{attn}: {len(calls)} FlexAttention calls"
        errors[attn] = (
            abs(loss.item() - ref_loss.item()),
            {k: fro_rel_err(grads[k], ref_grads[k]) for k in ref_grads},
        )
    sdpa_loss_err, sdpa_errs = errors["sdpa"]
    flex_loss_err, flex_errs = errors[name]
    assert flex_loss_err <= 2 * sdpa_loss_err + 1e-3
    for key, sdpa_err in sdpa_errs.items():
        assert flex_errs[key] <= 2 * sdpa_err + 1e-3, (key, flex_errs[key], sdpa_err)


def test_global_layers_fall_back_to_sdpa_past_the_budget():
    """Past the score-memory budget the global layers must take SDPA, and the result must not change."""
    patch_sdpa_for_wide_heads()
    name = register_flex_sliding_attention()
    torch.manual_seed(0)
    reference = Gemma4ForCausalLM(_config("sdpa")).cuda().float()
    candidate = Gemma4ForCausalLM(_config(name)).cuda().float()
    candidate.load_state_dict(reference.state_dict())
    ids, position_ids = _packed_batch([100, 7, 45])
    calls = []
    real = flex_sliding_attention._matmul_global_attention
    with (
        patch.object(flex_sliding_attention, "_compiled_flex", flex_attention),
        patch.object(flex_sliding_attention, "_matmul_global_attention", lambda *a: calls.append(1) or real(*a)),
    ):
        with patch.object(flex_sliding_attention, "EAGER_GLOBAL_BUDGET_BYTES", 0):
            loss, _ = _loss_and_grads(candidate, ids, position_ids)
        assert calls == []
        torch.testing.assert_close(loss, _loss_and_grads(reference, ids, position_ids)[0], rtol=1e-5, atol=1e-6)
        _loss_and_grads(candidate, ids, position_ids)
    assert len(calls) == 2  # the two global layers of the tiny model


def test_global_logits_stay_fp32_at_gemma4_scale():
    """Gemma 4's global layers attend at ``scaling`` 1.0 over 512-wide RMS-normed heads, so their logits
    reach the tens, where rounding a logit to bf16 moves its attention weight by percents. At that shape
    (16 query / 2 KV heads, bf16, causal), the matmul path's output and input gradients must be as close
    to an fp64 reference as mem-efficient SDPA's, the kernel it replaces, which keeps its logits in fp32."""
    seq, heads, kv_heads, dim = 1024, 16, 2, 512
    group = heads // kv_heads
    generator = torch.Generator(device="cuda").manual_seed(0)

    def rms_normed(*shape):
        x = torch.randn(*shape, device="cuda", generator=generator)
        return (x * x.pow(2).mean(-1, keepdim=True).rsqrt()).bfloat16().requires_grad_()

    q, k, v = rms_normed(1, heads, seq, dim), rms_normed(1, kv_heads, seq, dim), rms_normed(1, kv_heads, seq, dim)
    grad = torch.randn(1, seq, heads, dim, device="cuda", generator=generator).bfloat16()
    q64, k64, v64 = (t.detach().double().requires_grad_() for t in (q, k, v))
    causal = torch.ones(seq, seq, dtype=torch.bool, device="cuda").tril()
    scores = q64 @ k64.repeat_interleave(group, 1).transpose(-1, -2)
    expected = torch.softmax(scores.masked_fill(~causal, float("-inf")), -1) @ v64.repeat_interleave(group, 1)
    expected.transpose(1, 2).backward(grad.double())

    def errors(out):
        out.backward(grad)
        errs = [fro_rel_err(out, expected.transpose(1, 2))]
        errs += [fro_rel_err(t.grad, t64.grad) for t, t64 in ((q, q64), (k, k64), (v, v64))]
        for t in (q, k, v):
            t.grad = None
        return errs

    calls = []
    matmul_path = _counting(flex_sliding_attention._matmul_global_attention, calls)
    with patch.object(flex_sliding_attention, "_matmul_global_attention", matmul_path):
        matmul = errors(
            flex_sliding_attention.flex_sliding_attention(_causal_module(group), q, k, v, None, scaling=1.0)[0]
        )
    assert calls == [1], "the call must take the matmul path"
    with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION]):
        sdpa = errors(
            torch.nn.functional.scaled_dot_product_attention(
                q, k.repeat_interleave(group, 1), v.repeat_interleave(group, 1), is_causal=True, scale=1.0
            ).transpose(1, 2)
        )
    for name, got, floor in zip(("output", "query grad", "key grad", "value grad"), matmul, sdpa, strict=True):
        assert got <= 1.5 * floor, f"{name}: matmul error {got:.2e} against mem-efficient SDPA's {floor:.2e}"


def test_global_matmul_saves_the_budgeted_bytes():
    """``EAGER_GLOBAL_BUDGET_BYTES`` is spent at ``_SAVED_BYTES_PER_SCORE`` per score (the fp32 softmax output
    and the 16-bit probabilities): any other score-sized save, such as the fp32 logits, would let a layer the
    budget admits hold more than it."""
    seq, heads, kv_heads, dim = 2048, 16, 2, 512
    q = torch.randn(1, heads, seq, dim, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(1, kv_heads, seq, dim, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn(1, kv_heads, seq, dim, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    saved = {}

    def pack(tensor):
        saved[tensor.untyped_storage().data_ptr()] = tensor.untyped_storage().nbytes()
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        flex_sliding_attention._matmul_global_attention(q, k, v, None, 1.0, True)
    scores = heads * seq * seq
    score_sized = sum(nbytes for nbytes in saved.values() if nbytes >= scores)  # q, k, v save under a byte/score
    assert score_sized == flex_sliding_attention._SAVED_BYTES_PER_SCORE * scores, score_sized / scores


def test_global_layers_without_a_mask_attend_as_sdpa():
    """With no mask, the implementation registered as ``sdpa`` attends causally from the first key (it keeps
    the first ``q_len`` keys under PyTorch's top-left ``is_causal``) unless the call has a single query or an
    explicit ``is_causal=False``, which attend every key. The matmul path must match it, output and input
    gradients, on each: a decode step (1 query, 40 keys), a StaticCache prefill (8 queries, 40 keys), a
    square call, fewer keys than queries, and both explicit ``is_causal`` values. In fp64, so any difference
    is a masking one; the default ``head_dim ** -0.5`` scale runs the scaled branch."""
    heads, kv_heads, dim = 4, 2, 512
    module = _causal_module(heads // kv_heads)
    calls = []
    cases = [
        (1, 40, None),
        (8, 40, None),
        (8, 40, True),
        (8, 40, False),
        (40, 40, None),
        (40, 40, False),
        (40, 8, None),
    ]
    for q_len, kv_len, is_causal in cases:
        generator = torch.Generator(device="cuda").manual_seed(q_len * kv_len)
        shape = {"device": "cuda", "dtype": torch.float64, "generator": generator}
        q = torch.randn(1, heads, q_len, dim, **shape).requires_grad_()
        k = torch.randn(1, kv_heads, kv_len, dim, **shape).requires_grad_()
        v = torch.randn(1, kv_heads, kv_len, dim, **shape).requires_grad_()
        grad = torch.randn(1, q_len, heads, dim, **shape)
        q2, k2, v2 = (t.detach().clone().requires_grad_() for t in (q, k, v))
        matmul_path = _counting(flex_sliding_attention._matmul_global_attention, calls)
        with patch.object(flex_sliding_attention, "_matmul_global_attention", matmul_path):
            out, _ = flex_sliding_attention.flex_sliding_attention(module, q, k, v, None, is_causal=is_causal)
        out.backward(grad)
        with sdpa_kernel([SDPBackend.MATH]):
            expected, _ = flex_sliding_attention.ALL_ATTENTION_FUNCTIONS["sdpa"](
                module, q2, k2, v2, None, is_causal=is_causal
            )
        expected.backward(grad)
        case = f"q_len={q_len} kv_len={kv_len} is_causal={is_causal}"
        torch.testing.assert_close(out, expected, msg=case)
        for got, want in ((q.grad, q2.grad), (k.grad, k2.grad), (v.grad, v2.grad)):
            torch.testing.assert_close(got, want, msg=f"{case}: gradient")
    assert len(calls) == len(cases), "every call must take the matmul path"


def test_global_score_gemm_follows_autocast():
    """The fp32-output score GEMM is an overload autocast does not cast, so under autocast the path casts its
    inputs itself, as autocast casts a matmul and SDPA: fp32 queries and keys under bf16 autocast (a PEFT
    run's ``peft_bf16_autocast``) must reach the GEMM in bf16, not run an fp32 GEMM at ``highest``
    precision. Outside autocast the inputs keep their dtype."""
    q = torch.randn(1, 4, 64, 512, device="cuda")
    k = torch.randn(1, 2, 64, 512, device="cuda")
    seen = []
    real = flex_sliding_attention._Fp32ScoreMatmul.apply

    def spy(*inputs):
        seen.append({t.dtype for t in inputs})
        return real(*inputs)

    with patch.object(flex_sliding_attention._Fp32ScoreMatmul, "apply", spy):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            flex_sliding_attention.flex_sliding_attention(_causal_module(), q, k, k, None, scaling=1.0)
        flex_sliding_attention.flex_sliding_attention(_causal_module(), q, k, k, None, scaling=1.0)
    assert seen == [{torch.bfloat16}, {torch.float32}], seen


def test_bidirectional_modules_keep_sdpa():
    """The vision/audio towers attend bidirectionally; only causal layers may take the causal matmul path."""
    q = torch.randn(1, 4, 16, 32, device="cuda")
    k = torch.randn(1, 4, 16, 32, device="cuda")
    tower = torch.nn.Module()
    tower.is_causal = False
    out, _ = flex_sliding_attention.flex_sliding_attention(tower, q, k, k, None, scaling=1.0)
    expected = torch.nn.functional.scaled_dot_product_attention(q, k, k, scale=1.0).transpose(1, 2)
    torch.testing.assert_close(out, expected)


def test_block_masks_live_and_die_with_their_mask():
    """Every sliding layer of a forward gets the same mask tensor and must share its block masks; a new
    mask of the same shape (a later step's, which may reuse the freed storage) gets its own, built from
    its own entries; and a mask's block masks, with their padded copies of it, are freed with the mask
    rather than held until the next forward."""
    first = _sliding_mask([200])[None, None]
    a = flex_sliding_attention.sliding_block_masks(first)
    assert flex_sliding_attention.sliding_block_masks(first) is a
    a_alive = weakref.ref(a[0])
    del first, a
    assert a_alive() is None, "the block masks outlived their mask"
    # split at the 128-token tile boundary, so the second tile's key blocks differ from the first mask's
    second = _sliding_mask([128, 72])[None, None]
    b = flex_sliding_attention.sliding_block_masks(second)
    expected = flex_sliding_attention.interval_block_mask(*flex_sliding_attention.key_intervals(second[0, 0]), 200)
    stale = flex_sliding_attention.interval_block_mask(
        *flex_sliding_attention.key_intervals(_sliding_mask([200])), 200
    )
    assert not torch.equal(stale.kv_num_blocks, expected.kv_num_blocks)  # premise: a stale mask would show
    assert torch.equal(b[0].kv_num_blocks, expected.kv_num_blocks)
    assert torch.equal(b[0].full_kv_num_blocks, expected.full_kv_num_blocks)


def test_building_block_masks_holds_no_copy_of_the_mask():
    """The block masks reduce each query row to its key run, a chunk of rows at a time: building them for a
    16k-token packed row adds a small fraction of the mask, where evaluating a mask function over the full
    grid holds index tensors several times its size."""
    seq = 16_000
    dense = _sliding_mask([9_000, 7_000], window=1024)[None, None]
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    assert flex_sliding_attention.sliding_block_masks(dense) is not None
    torch.cuda.synchronize()
    added = torch.cuda.max_memory_allocated() - base
    assert added < 0.25 * seq * seq, f"building block masks took {added / 2**20:.0f} MiB"


def test_a_mask_with_a_non_contiguous_row_keeps_sdpa():
    """A query allowed two separate runs of keys has no single interval, so the call must reach SDPA with
    the mask as given, and the result must match SDPA's."""
    seq = 200
    dense = _sliding_mask([seq])
    dense[150, 130:135] = False  # query 150 (window 119..150) keeps two runs: 119..129 and 135..150
    q = torch.randn(1, 4, seq, 32, device="cuda")
    k = torch.randn(1, 2, seq, 32, device="cuda")
    calls = []
    with patch.object(flex_sliding_attention, "_compiled_flex", _counting(flex_attention, calls)):
        out, _ = flex_sliding_attention.flex_sliding_attention(
            _causal_module(), q, k, k, dense[None, None], scaling=1.0, sliding_window=WINDOW
        )
    assert calls == []
    expected = torch.nn.functional.scaled_dot_product_attention(
        q, k.repeat_interleave(2, 1), k.repeat_interleave(2, 1), attn_mask=dense, scale=1.0
    ).transpose(1, 2)
    torch.testing.assert_close(out, expected)


def test_a_non_gemma_sliding_model_matches_sdpa(lengths):
    """Nothing in the path is Gemma's: a Mistral model (one window on every layer, flash SDPA left enabled)
    built with the flex-sliding implementation reproduces the SDPA model's loss and every gradient with
    uncompiled FlexAttention. The SDPA backend flags are process-global, so the ones it turns on are
    restored for the checks after it."""
    flash, math = torch.backends.cuda.flash_sdp_enabled(), torch.backends.cuda.math_sdp_enabled()
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_math_sdp(True)
    try:
        name = register_flex_sliding_attention()
        torch.manual_seed(0)
        reference = MistralForCausalLM(_mistral_config("sdpa")).cuda().float()
        candidate = MistralForCausalLM(_mistral_config(name)).cuda().float()
        candidate.load_state_dict(reference.state_dict())
        calls = []
        with patch.object(
            flex_sliding_attention, "_compiled_flex", lambda *a, **k: calls.append(1) or flex_attention(*a, **k)
        ):
            _assert_model_matches(reference, candidate, lengths)
    finally:
        torch.backends.cuda.enable_flash_sdp(flash)
        torch.backends.cuda.enable_math_sdp(math)
    assert len(calls) == 2  # one call per sliding layer on the one packed row


def test_a_call_carrying_attention_sinks_keeps_sdpa():
    """FlexAttention is not given sinks here, so a sink-carrying call (GptOss's ``s_aux``) must reach the
    registered SDPA implementation with the sinks intact."""
    seen = {}

    def spy(module, query, key, value, attention_mask, **kwargs):
        seen.update(kwargs)
        return query.transpose(1, 2), None

    q = torch.randn(1, 4, 200, 32, device="cuda")  # long enough for FlexAttention but for the sinks
    sinks = torch.zeros(4, device="cuda")
    with patch.dict(flex_sliding_attention.ALL_ATTENTION_FUNCTIONS._global_mapping, {"sdpa": spy}):
        flex_sliding_attention.flex_sliding_attention(
            _causal_module(), q, q, q, _sliding_mask([200])[None, None], sliding_window=WINDOW, s_aux=sinks
        )
    assert seen.get("s_aux") is sinks


def test_the_resolver_picks_wide_head_models():
    """Heads wider than flash runs select the flex-sliding implementation on an SDPA run; a sliding model
    with narrow heads, a sinks model, a non-SDPA run and the opt-out keep what the run resolved."""
    wide = _config("sdpa", global_head_dim=512)
    assert resolve_flex_sliding_attn_implementation(wide, "sdpa") == FLEX_SLIDING
    assert resolve_flex_sliding_attn_implementation(_config("sdpa"), "sdpa") == "sdpa"
    assert resolve_flex_sliding_attn_implementation(_mistral_config("sdpa"), "sdpa") == "sdpa"
    wide_sinks = GptOssConfig(head_dim=512)
    assert head_dim_exceeds_flash(wide_sinks)  # premise: only the sinks keep it off
    assert resolve_flex_sliding_attn_implementation(wide_sinks, "sdpa") == "sdpa"
    assert resolve_flex_sliding_attn_implementation(wide, "flash_attention_4") == "flash_attention_4"
    with patch.dict(os.environ, {"HALO_FLEX_SLIDING": "0"}):
        assert resolve_flex_sliding_attn_implementation(wide, "sdpa") == "sdpa"


def test_one_warmup_serves_every_later_call():
    """After the load-time warm-up no call compiles: not a new length (a multiple of the tile or not, a
    single tile, which SDPA takes), batch size, packing or padding, nor a forward-only pass (``no_grad``, or
    frozen inputs under grad mode), nor a graph where only the query is trainable (a LoRA on ``q_proj``
    alone), nor autocast. Dynamo raises on any recompile here. A mid-run compile stalls the other ranks in
    the next collective, and past Dynamo's recompile limit (8) FlexAttention runs unfused with the full
    score matrix. The calls use the model's own ``scaling`` (1.0), as the model's layers do."""
    name = register_flex_sliding_attention()
    model = Gemma4ForCausalLM(_config(name))
    torch._dynamo.reset()
    counters = torch._dynamo.utils.counters
    counters.clear()
    warmup_flex_sliding_kernels(model, dtype=torch.bfloat16)
    heads, kv_heads, dim = 4, 2, 64
    # (packed documents per row, rows, trainable inputs: "qkv", "q", "" under grad mode or None under no_grad,
    # autocast on)
    cases = [
        ([100], 1, "qkv", False),
        ([64, 64], 1, "qkv", False),
        ([200], 1, "qkv", False),
        ([256], 1, "qkv", False),
        ([257], 2, "qkv", False),
        ([40, 20, 70], 3, "qkv", False),
        ([300], 1, "q", False),
        ([300], 2, "qkv", True),
        ([90], 1, None, False),
        ([384, 129], 2, None, False),
        ([500], 1, "", False),
    ]
    with patch.object(torch._dynamo.config, "error_on_recompile", True):
        for lengths, batch, trainable, autocast in cases:
            seq = sum(lengths)
            generator = torch.Generator(device="cuda").manual_seed(seq)
            shape = {"device": "cuda", "dtype": torch.bfloat16, "generator": generator}
            q = torch.randn(batch, heads, seq, dim, **shape).requires_grad_("q" in (trainable or ""))
            k = torch.randn(batch, kv_heads, seq, dim, **shape).requires_grad_("k" in (trainable or ""))
            v = torch.randn(batch, kv_heads, seq, dim, **shape).requires_grad_("v" in (trainable or ""))
            dense = _sliding_mask(lengths).expand(batch, 1, seq, seq).clone()
            dense[-1, :, :, : seq // 3] = False  # the last row pads its first third
            with (
                torch.set_grad_enabled(trainable is not None),
                torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast),
            ):
                out, _ = flex_sliding_attention.flex_sliding_attention(
                    _causal_module(), q, k, v, dense, scaling=1.0, sliding_window=WINDOW
                )
            if trainable:
                out.sum().backward()
            live = dense.any(-1)[:, 0]  # query rows with a key to attend
            ref = torch.nn.functional.scaled_dot_product_attention(
                q.detach(),
                k.detach().repeat_interleave(2, 1),
                v.detach().repeat_interleave(2, 1),
                attn_mask=dense,
                scale=1.0,
            ).transpose(1, 2)
            assert fro_rel_err(out.detach()[live], ref[live]) < 2e-2, f"{lengths} x{batch}: differs from SDPA"
    graphs = counters["stats"]["unique_graphs"]
    assert graphs == 2, f"{graphs} compiled graphs; the warm-up compiles a training and a forward-only one"


def test_a_warmed_gemma4_model_runs_without_compiling():
    """The warm-up compiles what the model's sliding layers call with, so a Gemma 4 forward and backward on
    packed rows past the window, and a forward-only pass, compile nothing. The warm-up reads each call off
    the attention modules; one built from the config alone guesses the scale (``head_dim ** -0.5`` rather
    than Gemma 4's 1.0), and its graphs would never be used."""
    patch_sdpa_for_wide_heads()
    name = register_flex_sliding_attention()
    torch.manual_seed(0)
    model = Gemma4ForCausalLM(_config(name)).cuda().to(torch.bfloat16)
    torch._dynamo.reset()
    warmup_flex_sliding_kernels(model, dtype=torch.bfloat16)
    ids, position_ids = _packed_batch([300, 150, 40])
    calls = []
    compiled = _counting(flex_sliding_attention._compiled_flex, calls)
    with (
        patch.object(torch._dynamo.config, "error_on_recompile", True),
        patch.object(flex_sliding_attention, "_compiled_flex", compiled),
    ):
        _loss_and_grads(model, ids, position_ids)
        with torch.no_grad():
            model(input_ids=ids, position_ids=position_ids, use_cache=False)
    assert len(calls) == 4, f"{len(calls)} compiled flex calls; want 2 sliding layers x 2 forwards"


def test_fp32_parameters_under_autocast_run_the_warmed_graphs():
    """Under bf16 autocast a Gemma 4 with fp32 parameters hands its sliding layers fp32 queries and keys (the
    fp32 rotary embedding promotes the bf16 projections) beside bf16 values. The sliding path must cast them to
    bf16, as SDPA's inputs are cast, so a forward and backward and a forward-only pass run the bf16 graphs the
    warm-up compiled: no compile, and every compiled call in bf16."""
    patch_sdpa_for_wide_heads()
    name = register_flex_sliding_attention()
    torch.manual_seed(0)
    model = Gemma4ForCausalLM(_config(name)).cuda().float()
    torch._dynamo.reset()
    warmup_flex_sliding_kernels(model, dtype=torch.bfloat16)
    ids, position_ids = _packed_batch([300, 150, 40])
    arrived, compiled_dtypes = [], []
    sliding, compiled = flex_sliding_attention._sliding_flex_attention, flex_sliding_attention._compiled_flex

    def sliding_spy(query, key, value, *args):
        arrived.append({query.dtype, key.dtype, value.dtype})
        return sliding(query, key, value, *args)

    def compiled_spy(query, key, value, **kwargs):
        compiled_dtypes.append({query.dtype, key.dtype, value.dtype})
        return compiled(query, key, value, **kwargs)

    with (
        patch.object(torch._dynamo.config, "error_on_recompile", True),
        patch.object(flex_sliding_attention, "_sliding_flex_attention", sliding_spy),
        patch.object(flex_sliding_attention, "_compiled_flex", compiled_spy),
        torch.autocast("cuda", dtype=torch.bfloat16),
    ):
        _loss_and_grads(model, ids, position_ids)
        with torch.no_grad():
            model(input_ids=ids, position_ids=position_ids, use_cache=False)
    assert arrived and all(dtypes == {torch.float32, torch.bfloat16} for dtypes in arrived), arrived  # premise
    assert compiled_dtypes == [{torch.bfloat16}] * 4, compiled_dtypes


def test_full_determinism_after_the_warmup_is_refused():
    """Deterministic-algorithms mode is a guard of the warmed graphs: switched on after the warm-up (HF's
    ``Trainer.__init__`` under ``full_determinism``), the next sliding call compiles again, which on a
    multi-rank run happens rank by rank mid-forward. The warm-up must record the mode it compiled under, so
    the trainer gate refuses ``full_determinism`` and keeps a run that does not set it."""
    name = register_flex_sliding_attention()
    model = Gemma4ForCausalLM(_config(name))
    torch._dynamo.reset()
    warmup_flex_sliding_kernels(model, dtype=torch.bfloat16)
    reject_full_determinism_after_warmup(False)
    try:
        reject_full_determinism_after_warmup(True)
    except ValueError as error:
        assert "HALO_FLEX_SLIDING=0" in str(error), error
    else:
        raise AssertionError("full_determinism after a non-deterministic warm-up was not refused")

    q = torch.zeros(1, 4, 200, 64, device="cuda", dtype=torch.bfloat16)  # the warmed shape family, forward-only
    k = torch.zeros(1, 2, 200, 64, device="cuda", dtype=torch.bfloat16)

    def sliding_call():
        with torch.no_grad(), patch.object(torch._dynamo.config, "error_on_recompile", True):
            flex_sliding_attention.flex_sliding_attention(
                _causal_module(), q, k, k, _sliding_mask([200])[None, None], scaling=1.0, sliding_window=WINDOW
            )

    sliding_call()  # control: in the warmed mode the call compiles nothing
    torch.use_deterministic_algorithms(True)
    try:
        sliding_call()
    except torch._dynamo.exc.RecompileError:
        pass
    else:
        raise AssertionError("premise: the warmed graph no longer recompiles under deterministic mode")
    finally:
        torch.use_deterministic_algorithms(False)


def test_rows_past_46k_tokens_match_sdpa():
    """Past about 46k tokens per padded row the flattened mask index ``q * S + kv`` no longer fits int32. A
    single document and a packed row of ~47.5k tokens at Gemma 4's window must still match SDPA on the
    dense mask, output and query gradient."""
    window, heads, kv_heads, dim = 1024, 2, 1, 64
    for lengths in ([47_500], [30_000, 17_500]):
        seq = sum(lengths)
        generator = torch.Generator(device="cuda").manual_seed(seq)
        shape = {"device": "cuda", "dtype": torch.bfloat16, "generator": generator}
        q = torch.randn(1, heads, seq, dim, **shape).requires_grad_()
        k = torch.randn(1, kv_heads, seq, dim, **shape).requires_grad_()
        v = torch.randn(1, kv_heads, seq, dim, **shape).requires_grad_()
        dense = _sliding_mask(lengths, window)[None, None]
        grad = torch.randn(1, seq, heads, dim, **shape)
        out, _ = flex_sliding_attention.flex_sliding_attention(
            _causal_module(heads // kv_heads), q, k, v, dense, scaling=1.0, sliding_window=window
        )
        out.backward(grad)
        q2, k2, v2 = (t.detach().clone().requires_grad_() for t in (q, k, v))
        ref = torch.nn.functional.scaled_dot_product_attention(
            q2, k2.repeat_interleave(2, 1), v2.repeat_interleave(2, 1), attn_mask=dense, scale=1.0
        ).transpose(1, 2)
        ref.backward(grad)
        assert fro_rel_err(out, ref) < 2e-2, f"{lengths}: output differs from SDPA"
        assert fro_rel_err(q.grad, q2.grad) < 2e-2, f"{lengths}: query gradient differs from SDPA"
        # the last rows are the ones an int32 overflow would misread
        assert fro_rel_err(out[:, -window:], ref[:, -window:]) < 2e-2, f"{lengths}: the last rows differ"
        del dense, q, k, v, q2, k2, v2, out, ref, grad
        torch.cuda.empty_cache()


@gpu_test_main(exact_world_size=1, prefix="flex_sliding")
def run(ctx) -> dict:
    checks: dict[str, bool] = {}
    for lengths in PACKINGS:
        record_check(
            checks,
            f"flex_sliding_wiring_matches_sdpa[{lengths}]",
            lambda: test_flex_sliding_wiring_matches_sdpa(lengths),
        )
    record_check(
        checks, "compiled_flex_kernel_matches_fp64_reference", test_compiled_flex_kernel_matches_fp64_reference
    )
    record_check(checks, "tuned_tiles_match_sdpa_at_gemma4_shapes", test_tuned_tiles_match_sdpa_at_gemma4_shapes)
    record_check(
        checks, "bf16_model_error_is_within_the_sdpa_noise_floor", test_bf16_model_error_is_within_the_sdpa_noise_floor
    )
    record_check(
        checks, "global_layers_fall_back_to_sdpa_past_the_budget", test_global_layers_fall_back_to_sdpa_past_the_budget
    )
    record_check(checks, "global_logits_stay_fp32_at_gemma4_scale", test_global_logits_stay_fp32_at_gemma4_scale)
    record_check(checks, "global_matmul_saves_the_budgeted_bytes", test_global_matmul_saves_the_budgeted_bytes)
    record_check(
        checks, "global_layers_without_a_mask_attend_as_sdpa", test_global_layers_without_a_mask_attend_as_sdpa
    )
    record_check(checks, "global_score_gemm_follows_autocast", test_global_score_gemm_follows_autocast)
    record_check(checks, "bidirectional_modules_keep_sdpa", test_bidirectional_modules_keep_sdpa)
    record_check(checks, "block_masks_live_and_die_with_their_mask", test_block_masks_live_and_die_with_their_mask)
    record_check(
        checks, "building_block_masks_holds_no_copy_of_the_mask", test_building_block_masks_holds_no_copy_of_the_mask
    )
    record_check(
        checks, "a_mask_with_a_non_contiguous_row_keeps_sdpa", test_a_mask_with_a_non_contiguous_row_keeps_sdpa
    )
    for lengths in PACKINGS:
        record_check(
            checks,
            f"a_non_gemma_sliding_model_matches_sdpa[{lengths}]",
            lambda: test_a_non_gemma_sliding_model_matches_sdpa(lengths),
        )
    record_check(checks, "a_call_carrying_attention_sinks_keeps_sdpa", test_a_call_carrying_attention_sinks_keeps_sdpa)
    record_check(checks, "the_resolver_picks_wide_head_models", test_the_resolver_picks_wide_head_models)
    record_check(checks, "one_warmup_serves_every_later_call", test_one_warmup_serves_every_later_call)
    record_check(
        checks, "a_warmed_gemma4_model_runs_without_compiling", test_a_warmed_gemma4_model_runs_without_compiling
    )
    record_check(
        checks,
        "fp32_parameters_under_autocast_run_the_warmed_graphs",
        test_fp32_parameters_under_autocast_run_the_warmed_graphs,
    )
    record_check(
        checks, "full_determinism_after_the_warmup_is_refused", test_full_determinism_after_the_warmup_is_refused
    )
    record_check(checks, "rows_past_46k_tokens_match_sdpa", test_rows_past_46k_tokens_match_sdpa)
    return {"checks": checks}


if __name__ == "__main__":
    run()
