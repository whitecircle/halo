#!/usr/bin/env python
"""Nemotron-H's two attention calls through their real modules at the release's geometry, forward and backward, against
eager at fp32 on the same weights and inputs. The text tower: 32 query heads over 2 KV heads at head_dim 128, causal, on
one document, across packed documents and over left-padded rows. The vision tower: 16 heads at head_dim 80, non-causal,
on one image and across two.

FA4 must stay within its FA2 error, on the kernel the form names; the same call with its causality flipped must read far
outside the band, or the row compares nothing. Below the module, ``kernels/test_fa4_gqa_ratio.py`` holds the 16:1 kernel
against fp64.
"""

import copy
import functools
import time
from dataclasses import dataclass

import torch
from transformers import NemotronHConfig
from transformers.models.nemotron_h.modeling_nemotron_h import NemotronHAttention
from transformers.models.radio.configuration_radio import RadioConfig
from transformers.models.radio.modeling_radio import RadioAttention

from halo.model_loading.model_preparation import apply_family_patches
from tests.common.harness import gpu_test_main, skip_unless_blackwell
from tests.common.models import NEMOTRON_H_RELEASE_TEXT_CONFIG, NEMOTRON_H_RELEASE_VISION_CONFIG
from tests.common.numerics import fro_rel_err, max_abs_rel_err
from tests.common.rank_log import log

FA4, FA2 = "flash_attention_4", "flash_attention_2"
_SEED = 1234
_SEQ = 2048
_DOCUMENTS = (700, 900, 448)
_LEFT_PADS = (0, 512)
_IMAGES = (1024, 576)
# Two bf16 kernels, different accumulation order; a misread is O(1).
_FA2_ERROR_RATIO = 1.5
# bf16 round-off at the module's output and the gradients it takes, far below a misread.
_ROUND_OFF_REL = 2e-2
_MISREAD_REL_FLOOR = 0.3

skip_unless_blackwell("Nemotron-H's attention kernel row")


@dataclass(frozen=True)
class Form:
    name: str
    tower: str
    varlen: bool


FORMS = (
    Form("text_one_document", "text", varlen=False),
    Form("text_packed_documents", "text", varlen=True),
    Form("text_left_padded", "text", varlen=True),
    Form("vision_one_image", "vision", varlen=False),
    Form("vision_two_images", "vision", varlen=True),
)


def _spied(calls: dict[str, int], name: str, kernel):
    @functools.wraps(kernel)
    def spy(*args, **kwargs):
        calls[name] = calls.get(name, 0) + 1
        return kernel(*args, **kwargs)

    return spy


def _install_spies(calls: dict[str, int]) -> None:
    """Ahead of the first flash forward, as transformers imports each kernel by name when a backend first runs."""
    import flash_attn  # noqa: PLC0415 - optional dependency (FA2, the fallback)
    import flash_attn.cute as cute  # noqa: PLC0415 - optional Blackwell-only dependency

    for module, backend in ((cute, FA4), (flash_attn, FA2)):
        for name in ("flash_attn_func", "flash_attn_varlen_func"):
            setattr(module, name, _spied(calls, f"{backend}/{name}", getattr(module, name)))


def _module(form: Form, device) -> torch.nn.Module:
    torch.manual_seed(_SEED)
    if form.tower == "text":
        config = NemotronHConfig(**NEMOTRON_H_RELEASE_TEXT_CONFIG)
        module = NemotronHAttention(config, layer_idx=0)
    else:
        module = RadioAttention(RadioConfig(**NEMOTRON_H_RELEASE_VISION_CONFIG))
    return module.to(device=device, dtype=torch.bfloat16).train()


def _lengths(form: Form) -> tuple[int, ...]:
    if form.name == "text_packed_documents":
        return _DOCUMENTS
    if form.name == "vision_two_images":
        return _IMAGES
    return (_SEQ,)


def _inputs(form: Form, module, device) -> torch.Tensor:
    generator = torch.Generator(device=device).manual_seed(_SEED)
    batch = len(_LEFT_PADS) if form.name == "text_left_padded" else 1
    hidden = module.config.hidden_size
    return torch.randn(batch, sum(_lengths(form)), hidden, generator=generator, device=device, dtype=torch.bfloat16)


def _allowed(form: Form, device) -> torch.Tensor:
    """``[row, query, key]``: the keys each query attends in this form."""
    lengths = _lengths(form)
    segments = torch.repeat_interleave(torch.arange(len(lengths), device=device), torch.tensor(lengths, device=device))
    allowed = segments[:, None] == segments[None, :]
    if form.tower == "text":
        allowed = allowed.tril()
    allowed = allowed[None]
    if form.name == "text_left_padded":
        real = torch.stack([torch.arange(_SEQ, device=device) >= pad for pad in _LEFT_PADS])
        allowed = allowed & real[:, :, None] & real[:, None, :]
    return allowed


def _attention_arguments(form: Form, device, *, eager: bool) -> dict:
    if form.name == "text_packed_documents":
        positions = torch.cat([torch.arange(length, device=device) for length in _DOCUMENTS]).unsqueeze(0)
        arguments = {} if eager else {"position_ids": positions}
    elif form.name == "text_left_padded":
        real = torch.stack([torch.arange(_SEQ, device=device) >= pad for pad in _LEFT_PADS]).long()
        arguments = {} if eager else {"attention_mask": real}
    elif form.name == "vision_two_images":
        return {"cu_seqlens": torch.tensor([0, _IMAGES[0], sum(_IMAGES)], device=device, dtype=torch.int32)}
    else:
        arguments = {}
    if eager and form.tower == "text":
        blocked = torch.zeros(_allowed(form, device).shape, device=device).masked_fill(
            ~_allowed(form, device), torch.finfo(torch.float32).min
        )
        arguments["attention_mask"] = blocked.unsqueeze(1)
    return arguments


def _real_tokens(form: Form, device) -> torch.Tensor:
    """Padding has no defined output: every kernel leaves it at its own value."""
    if form.name != "text_left_padded":
        return torch.ones(1, sum(_lengths(form)), dtype=torch.bool, device=device)
    return torch.stack([torch.arange(_SEQ, device=device) >= pad for pad in _LEFT_PADS])


def _run(module, form: Form, hidden, tangent, *, eager: bool) -> dict[str, torch.Tensor]:
    leaf = hidden.detach().clone().requires_grad_(True)
    output, _ = module(leaf, **_attention_arguments(form, hidden.device, eager=eager))
    real = _real_tokens(form, hidden.device)
    (output * real[..., None]).backward(tangent.to(output.dtype))
    results = {
        "output": output.detach() * real[..., None],
        "d_input": leaf.grad.detach() * real[..., None],
        "d_k_proj": module.k_proj.weight.grad.detach().clone(),
        "d_q_proj": module.q_proj.weight.grad.detach().clone(),
    }
    module.zero_grad()
    return results


def _errors(results: dict[str, torch.Tensor], reference: dict[str, torch.Tensor]) -> dict[str, float]:
    return {key: fro_rel_err(results[key], reference[key]) for key in reference}


def _on_backend(module, backend: str, *, flipped: bool = False) -> torch.nn.Module:
    apply_family_patches(module.config, backend)
    module = copy.deepcopy(module)
    module.config._attn_implementation = backend
    if flipped:
        module.is_causal = not module.is_causal
    return module


@gpu_test_main(exact_world_size=1, partial_state=False, prefix="nemotron_attention_kernels")
def run(ctx):
    checks, metrics, calls = {}, {}, {}
    _install_spies(calls)
    for form in FORMS:
        module = _module(form, ctx.device)
        hidden = _inputs(form, module, ctx.device)
        generator = torch.Generator(device=ctx.device).manual_seed(_SEED + 1)
        tangent = torch.randn(hidden.shape, generator=generator, device=ctx.device, dtype=torch.float32)

        reference_module = copy.deepcopy(module).float()
        reference_module.config._attn_implementation = "eager"
        reference = _run(reference_module, form, hidden.float(), tangent, eager=True)

        errors = {}
        for backend in (FA2, FA4):
            kernel = f"{backend}/{'flash_attn_varlen_func' if form.varlen else 'flash_attn_func'}"
            before = calls.get(kernel, 0)
            start = time.perf_counter()
            results = _run(_on_backend(module, backend), form, hidden, tangent, eager=False)
            metrics[f"{form.name}_{backend}_first_call_s"] = time.perf_counter() - start
            checks[f"{form.name}_{backend}_ran_its_kernel"] = calls.get(kernel, 0) == before + 1
            errors[backend] = _errors(results, reference)
            log(f"  [{form.name}] {backend} {errors[backend]}")
            for name, error in errors[backend].items():
                metrics[f"{form.name}_{backend}_{name}"] = error
                checks[f"{form.name}_{backend}_{name}_within_round_off"] = error <= _ROUND_OFF_REL
        for name, error in errors[FA4].items():
            checks[f"{form.name}_fa4_{name}_within_fa2_error"] = error <= _FA2_ERROR_RATIO * errors[FA2][name]

        flipped = _run(_on_backend(module, FA4, flipped=True), form, hidden, tangent, eager=False)
        misread = max_abs_rel_err(flipped["output"], reference["output"])
        metrics[f"{form.name}_flipped_causality"] = misread
        checks[f"{form.name}_flipped_causality_shows"] = misread > _MISREAD_REL_FLOOR
    return {"checks": checks, "metrics": metrics}


if __name__ == "__main__":
    run()
