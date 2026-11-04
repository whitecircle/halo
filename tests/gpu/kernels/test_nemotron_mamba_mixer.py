#!/usr/bin/env python
"""Nemotron-H's Mamba block at the release's shape on the CUDA conv kernel, whose packed path and recompute only run
here: documents equal themselves alone, each GC mode equals no GC, the grouped body equals upstream's, and the recompute
stays under its per-token memory bar."""

import functools
import statistics
import time

import torch
from torch.utils.checkpoint import checkpoint
from transformers import NemotronHConfig
from transformers.models.nemotron_h import modeling_nemotron_h as nemotron_modeling

from halo.models import nemotron_h
from halo.models.nemotron_h import apply_nemotron_h_patches
from tests.common.harness import gpu_test_main, record_check
from tests.common.models import NEMOTRON_H_RELEASE_TEXT_CONFIG
from tests.common.numerics import fro_rel_err, max_abs_rel_err
from tests.common.rank_log import log

_LAYER = 0
_SEED = 1234
_DOCUMENTS = {
    "even": (512,) * 16,
    "skewed": (4096, 2048, 1024, 512, 256, 128, 64, 64),
    "short": (128,) * 64,
}
_ROW_TOKENS = 8192
_LONG_ROW_TOKENS = 32768
# Stock init hides a crossed boundary below rounding.
_IN_PROJ_SCALE = 50.0
# bf16 at the block's output, a document against itself alone.
_ISOLATED = 2e-2
_CROSSED = 1e-1
_FP32_SCAN = 1e-5
_GC_GRAD_REL = 5e-3
# Peak net of the block's gradients, reentrant recompute, with the per-group scan.
_RECOMPUTE_MB_PER_TOKEN = 0.85
_TIMED_PASSES = 5


def _block(device: torch.device) -> nemotron_modeling.NemotronHBlock:
    torch.manual_seed(_SEED)
    config = NemotronHConfig(**NEMOTRON_H_RELEASE_TEXT_CONFIG)
    config._attn_implementation = "sdpa"
    block = nemotron_modeling.NemotronHBlock(config, layer_idx=_LAYER)
    assert isinstance(block.mixer, nemotron_modeling.NemotronHMamba2Mixer), "the release's layer 0 is Mamba"
    with torch.no_grad():
        block.mixer.in_proj.weight.mul_(_IN_PROJ_SCALE)
    return block.to(device=device, dtype=torch.bfloat16).train()


def _hidden(tokens: int, device: torch.device) -> torch.Tensor:
    generator = torch.Generator(device=device).manual_seed(_SEED)
    hidden_size = NEMOTRON_H_RELEASE_TEXT_CONFIG["hidden_size"]
    return torch.randn(1, tokens, hidden_size, generator=generator, device=device, dtype=torch.bfloat16)


def _seq_idx(lengths: tuple[int, ...], device: torch.device) -> torch.Tensor:
    return (
        torch.cat([torch.full((length,), index, device=device) for index, length in enumerate(lengths)])
        .int()
        .unsqueeze(0)
    )


def _spans(lengths: tuple[int, ...]) -> list[tuple[int, int]]:
    edges = torch.tensor([0, *lengths]).cumsum(0).tolist()
    return list(zip(edges[:-1], edges[1:], strict=True))


def _graph_names(output: torch.Tensor) -> set[str]:
    seen, stack, names = set(), [output.grad_fn], set()
    while stack:
        node = stack.pop()
        if node is None or node in seen:
            continue
        seen.add(node)
        names.add(type(node).__name__)
        stack.extend(next_node for next_node, _ in node.next_functions)
    return names


def test_the_conv_kernel_runs(device: torch.device) -> None:
    """The stack's proof of what it measured: the graph holds the kernel's backward, not a cuDNN convolution."""
    block = _block(device)
    names = _graph_names(block(_hidden(2048, device)))
    assert any("CausalConv1d" in name for name in names), sorted(names)
    assert not any(name.startswith("Convolution") for name in names), sorted(names)


def test_a_document_equals_itself_alone(device: torch.device, lengths: tuple[int, ...]) -> None:
    block = _block(device)
    hidden = _hidden(sum(lengths), device)
    with torch.no_grad():
        row = block(hidden, seq_idx=_seq_idx(lengths, device)).float()
        whole = block(hidden).float()
        alone = torch.cat([block(hidden[:, start:stop]).float() for start, stop in _spans(lengths)], dim=1)
    isolated, crossed = max_abs_rel_err(row, alone), max_abs_rel_err(whole, alone)
    log(f"  {len(lengths)} documents: isolated {isolated:.2e}, whole-row control {crossed:.2e}")
    assert isolated <= _ISOLATED, f"a document moved by {isolated}"
    assert crossed >= _CROSSED, f"the whole row moved only {crossed}, so the row tests nothing"


def test_upstreams_scan_refuses_the_marker(device: torch.device, stock_scan, stock_fused) -> None:
    """Upstream's dispatch raises on ``seq_idx`` without ``mamba-ssm``, so a packed row needs both replacements."""
    nemotron_modeling.mamba2_chunk_scan, patched_scan = stock_scan, nemotron_modeling.mamba2_chunk_scan
    nemotron_modeling.mamba2_split_conv1d_scan_combined, patched_fused = (
        stock_fused,
        nemotron_modeling.mamba2_split_conv1d_scan_combined,
    )
    lengths = _DOCUMENTS["even"]
    try:
        block = _block(device)
        try:
            with torch.no_grad():
                block(_hidden(sum(lengths), device), seq_idx=_seq_idx(lengths, device))
        except ValueError as error:
            assert "keep packed sequences apart" in str(error), error
        else:
            raise AssertionError("upstream took a packed row, so its dispatch no longer refuses and the pin is moot")
    finally:
        nemotron_modeling.mamba2_chunk_scan = patched_scan
        nemotron_modeling.mamba2_split_conv1d_scan_combined = patched_fused


def test_the_grouped_body_equals_upstreams(device: torch.device, stock_scan) -> None:
    """On the kernel's fp32 path, where the CPU suite cannot reach a GEMM: output and gradients."""
    block = _block(device)
    captured: dict[str, torch.Tensor] = {}
    scan = nemotron_modeling.mamba2_chunk_scan

    def capture(hidden_states, dt, A, B, C, **kwargs):
        captured.update(hidden_states=hidden_states.detach(), dt=dt.detach(), B=B.detach(), C=C.detach())
        captured.update(arguments=(A.detach(), kwargs))
        return scan(hidden_states, dt, A, B, C, **kwargs)

    nemotron_modeling.mamba2_chunk_scan = capture
    try:
        with torch.no_grad():
            block(_hidden(_ROW_TOKENS, device))
    finally:
        nemotron_modeling.mamba2_chunk_scan = scan
    A, kwargs = captured.pop("arguments")
    results = []
    for function in (stock_scan, nemotron_h._document_chunk_scan):
        # At fp32, where a leaf's stored gradient keeps what the body computed.
        inputs = {key: value.float().requires_grad_(True) for key, value in captured.items()}
        output = function(**inputs, A=A, **kwargs)
        output.float().pow(2).sum().backward()
        results.append((output.detach(), {key: value.grad for key, value in inputs.items()}))
    (upstream, upstream_grads), (ours, grads) = results
    assert max_abs_rel_err(ours, upstream) <= _FP32_SCAN
    for key, upstream_grad in upstream_grads.items():
        assert fro_rel_err(grads[key], upstream_grad) <= _FP32_SCAN, key


def _checkpointed(block, use_reentrant: bool | None):
    """The layer's own checkpoint path, as a trainer enables it."""
    block.gradient_checkpointing = use_reentrant is not None
    if use_reentrant is not None:
        block._gradient_checkpointing_func = functools.partial(checkpoint, use_reentrant=use_reentrant)
    return block


def _loss_and_grads(
    block, hidden, use_reentrant: bool | None, **kwargs
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    block.zero_grad(set_to_none=True)
    loss = _checkpointed(block, use_reentrant)(hidden, **kwargs).float().pow(2).mean()
    loss.backward()
    return loss.detach(), {name: param.grad.clone() for name, param in block.named_parameters()}


def test_every_gc_mode_equals_no_gc(
    device: torch.device, use_reentrant: bool, lengths: tuple[int, ...] | None
) -> None:
    block = _block(device)
    tokens = _ROW_TOKENS if lengths is None else sum(lengths)
    hidden = _hidden(tokens, device).requires_grad_(True)
    kwargs = {} if lengths is None else {"seq_idx": _seq_idx(lengths, device)}
    plain_loss, plain_grads = _loss_and_grads(block, hidden, None, **kwargs)
    loss, grads = _loss_and_grads(block, hidden, use_reentrant, **kwargs)
    assert loss.isfinite() and max_abs_rel_err(loss, plain_loss) <= _GC_GRAD_REL
    for name, plain in plain_grads.items():
        assert fro_rel_err(grads[name], plain) <= _GC_GRAD_REL, name


def test_the_recompute_stays_under_its_bar(device: torch.device, tokens: int) -> None:
    """Fails the fit table this row's bar underwrites: the recompute is the family's peak layer per token."""
    block = _block(device)
    hidden = _hidden(tokens, device).requires_grad_(True)
    grads_bytes = sum(parameter.numel() * parameter.element_size() for parameter in block.parameters())
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated(device)
    torch.cuda.reset_peak_memory_stats(device)
    _loss_and_grads(block, hidden, True)
    peak = torch.cuda.max_memory_allocated(device) - base - grads_bytes
    per_token = peak / tokens / 2**20
    log(f"  recompute peak at S={tokens}: {per_token:.3f} MB/token")
    assert per_token <= _RECOMPUTE_MB_PER_TOKEN, f"{per_token} MB/token"


def _layer_milliseconds(device: torch.device, lengths: tuple[int, ...] | None) -> float:
    """One layer's reentrant forward and backward, median over a few passes after the first."""
    block = _block(device)
    tokens = _ROW_TOKENS if lengths is None else sum(lengths)
    hidden = _hidden(tokens, device).requires_grad_(True)
    kwargs = {} if lengths is None else {"seq_idx": _seq_idx(lengths, device)}
    times = []
    for _ in range(_TIMED_PASSES + 1):
        torch.cuda.synchronize()
        start = time.perf_counter()
        _loss_and_grads(block, hidden, True, **kwargs)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1e3)
    return statistics.median(times[1:])


@gpu_test_main(exact_world_size=1, prefix="nemotron_mamba_mixer")
def run(ctx):
    stock_scan = nemotron_modeling.mamba2_chunk_scan
    stock_fused = nemotron_modeling.mamba2_split_conv1d_scan_combined
    apply_nemotron_h_patches()
    device = ctx.device
    checks: dict[str, bool] = {}
    record_check(checks, "the_conv_kernel_runs", lambda: test_the_conv_kernel_runs(device))
    for name, lengths in _DOCUMENTS.items():
        record_check(
            checks,
            f"a_document_equals_itself_alone[{name}]",
            functools.partial(test_a_document_equals_itself_alone, device, lengths),
        )
    record_check(
        checks,
        "upstreams_scan_refuses_the_marker",
        lambda: test_upstreams_scan_refuses_the_marker(device, stock_scan, stock_fused),
    )
    record_check(
        checks, "the_grouped_body_equals_upstreams", lambda: test_the_grouped_body_equals_upstreams(device, stock_scan)
    )
    for use_reentrant in (True, False):
        for name, lengths in (("whole_row", None), ("documents", _DOCUMENTS["even"])):
            record_check(
                checks,
                f"every_gc_mode_equals_no_gc[reentrant={use_reentrant}-{name}]",
                functools.partial(test_every_gc_mode_equals_no_gc, device, use_reentrant, lengths),
            )
    for tokens in (_ROW_TOKENS, _LONG_ROW_TOKENS):
        record_check(
            checks,
            f"the_recompute_stays_under_its_bar[{tokens}]",
            functools.partial(test_the_recompute_stays_under_its_bar, device, tokens),
        )
    metrics = {"layer_ms[whole_row]": _layer_milliseconds(device, None)}
    metrics.update({f"layer_ms[{name}]": _layer_milliseconds(device, lengths) for name, lengths in _DOCUMENTS.items()})
    return {"checks": checks, "metrics": metrics}


if __name__ == "__main__":
    run()
