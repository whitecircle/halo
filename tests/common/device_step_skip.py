"""The device skip flag's contract for an optimizer, on any device: a False flag writes as unflagged, a True one
changes nothing, state included."""

import copy
from collections.abc import Callable, Sequence

import torch
import torch.nn as nn

from halo.optimizers.adamw_bf16 import AdamWBF16
from halo.optimizers.muon import build_muon_optimizer

# On the non-finite steps the first parameter's gradient holds a NaN and the first fp32 one's an inf; the rest stay
# finite, as a rank's do when another rank overflowed.
STEPS = 5
NONFINITE_STEPS = (1, 3)
# (shape, dtype, weight decay): bf16 under, near and over one SR block of 1024 elements, and an fp32 master with and
# without decay.
ADAMW_BF16_PARAMS = (
    ((40, 24), torch.bfloat16, 0.1),
    ((24,), torch.bfloat16, 0.0),
    ((48, 64), torch.bfloat16, 0.1),
    ((8, 24), torch.float32, 0.1),
    ((8,), torch.float32, 0.0),
)

Build = Callable[[], tuple[list[nn.Parameter], torch.optim.Optimizer]]


def adamw_bf16_builder(
    device, *, optimizer_cls: type[AdamWBF16] = AdamWBF16, parameter_cls: type[nn.Parameter] = nn.Parameter
) -> Build:
    """A builder of :data:`ADAMW_BF16_PARAMS` on ``device`` under ``optimizer_cls``, two weight-decay groups, the
    undecayed on its own betas."""

    def build():
        generator = torch.Generator().manual_seed(0)
        params = [
            parameter_cls(torch.randn(shape, generator=generator).to(device=device, dtype=dtype))
            for shape, dtype, _ in ADAMW_BF16_PARAMS
        ]
        groups = [
            {
                "params": [p for p, (_, _, wd) in zip(params, ADAMW_BF16_PARAMS, strict=True) if wd == decay],
                "param_names": [f"p{i}" for i, (_, _, wd) in enumerate(ADAMW_BF16_PARAMS) if wd == decay],
                "weight_decay": decay,
                "betas": betas,
            }
            for decay, betas in ((0.1, (0.9, 0.999)), (0.0, (0.8, 0.99)))
        ]
        return params, optimizer_cls(groups, lr=1e-2)

    return build


def muon_builder(device) -> Build:
    """Muon over two bf16 matrices (Newton-Schulz) and a norm's vectors (its AdamWBF16 leg)."""

    def build():
        torch.manual_seed(0)
        model = nn.Sequential(nn.Linear(16, 32), nn.LayerNorm(32), nn.Linear(32, 24)).to(device, torch.bfloat16)
        return list(model.parameters()), build_muon_optimizer(model, lr=1e-2, weight_decay=0.1)

    return build


def _same(a, b) -> bool:
    return torch.equal(a, b) if torch.is_tensor(a) and torch.is_tensor(b) else a == b


def _state_differences(expected: dict, got: dict, what: str) -> list[str]:
    """Every per-parameter entry of two ``state_dict()`` payloads that differs: params, keys, scalars, tensor bits."""
    if expected["state"].keys() != got["state"].keys():
        return [f"{what}: state covers params {sorted(got['state'])}, expected {sorted(expected['state'])}"]
    found = []
    for index, entry in expected["state"].items():
        other = got["state"][index]
        if entry.keys() != other.keys():
            found.append(f"{what}: param {index} state keys {sorted(other)}, expected {sorted(entry)}")
            continue
        found += [f"{what}: param {index} {key!r} differs" for key in entry if not _same(entry[key], other[key])]
    return found


def _written_state_tensors(before: dict, after: dict, what: str) -> list[str]:
    return [
        f"{what}: param {index} {key!r} written"
        for index, entry in before["state"].items()
        for key, value in entry.items()
        if torch.is_tensor(value) and not torch.equal(value, after["state"][index][key])
    ]


def _param_differences(expected: Sequence[torch.Tensor], params: Sequence[torch.Tensor], what: str) -> list[str]:
    return [
        f"{what}: param {i} differs"
        for i, (value, p) in enumerate(zip(expected, params, strict=True))
        if not torch.equal(value, p)
    ]


def device_skip_mismatches(build_reference: Build, build: Build) -> list[str]:
    """Run :data:`STEPS` steps under the device flag beside ``build_reference()``'s unflagged finite steps, naming each
    departure. Both builders return equal parameters on one device and an optimizer with no state yet."""
    reference_params, reference = build_reference()
    params, optimizer = build()
    device = params[0].device
    generator = torch.Generator().manual_seed(1)
    found = _param_differences(reference_params, params, "premise: the two builds")
    for step in range(1, STEPS + 1):
        nonfinite = step in NONFINITE_STEPS
        grads = [torch.randn(p.shape, generator=generator).to(device=device, dtype=p.dtype) for p in params]
        if nonfinite:
            grads[0].view(-1)[0] = float("nan")
            fp32_grads = [grad for grad in grads if grad.dtype == torch.float32]
            if fp32_grads:
                fp32_grads[0].view(-1)[-1] = float("inf")
            weights = [p.detach().clone() for p in params]
            state = copy.deepcopy(optimizer.state_dict())
        else:
            for p, grad in zip(reference_params, grads, strict=True):
                p.grad = grad.clone()
            reference.step()
        for p, grad in zip(params, grads, strict=True):
            p.grad = grad
        optimizer.skip_next_step_if(torch.tensor(nonfinite, device=device))
        optimizer.step()
        if nonfinite:
            found += _param_differences(weights, params, f"skipped step {step}")
            found += _written_state_tensors(state, optimizer.state_dict(), f"skipped step {step}")
            optimizer.settle_step(skipped=True)
            found += _state_differences(state, optimizer.state_dict(), f"skipped step {step}, settled")
        else:
            optimizer.settle_step(skipped=False)
            found += _param_differences(reference_params, params, f"finite step {step}")
            found += _state_differences(reference.state_dict(), optimizer.state_dict(), f"finite step {step}")
    if not optimizer.state_dict()["state"]:
        found.append("premise: the finite steps built no optimizer state")
    return found
