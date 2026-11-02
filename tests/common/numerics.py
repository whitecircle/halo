"""Numerical comparisons the correctness tests bound: tensor error metrics, NaN-carrying extremes, equality
across float dtypes, Muon's orthogonality on a controlled spectrum, the autograd graph a loss builds, and the
deterministic weight perturbation a measured noise floor starts from."""

import math
from collections.abc import Callable, Iterable

import torch
from torch.distributed.tensor import DTensor

from tests.common.tolerances import TOL

# Spreads ``sin(value * phase)`` over its whole range across weights of similar magnitude.
_PERTURBATION_PHASE = 12345.678


def cos_sim(a: torch.Tensor, b: torch.Tensor, *, label: str) -> float:
    """Cosine similarity of two tensors, compared as flat fp64 vectors.

    The direction half of a correctness comparison: a norm-preserving corruption (a wrong expert
    bank, a permuted dispatch) reorients the vector while an absolute-difference bound on bf16
    tensors has to be loose enough to absorb accumulation noise. fp64 because the cosine of two
    bf16 vectors rounds its own accumulation, and a finite tensor near the bf16 range limit would
    overflow an fp32 norm.

    A zero-norm or non-finite operand raises, naming ``label``. A zero vector has no direction, and
    any number returned for it passes one side of a threshold: 1.0 matches a dead gradient against a
    live one, 0.0 lets a negative control whose gradient vanished read as decorrelated, and NaN slips
    past every ``<`` and ``min`` tracker. A pair that is legitimately all-zero is the caller's to
    handle explicitly.
    """
    if not (torch.isfinite(a).all() and torch.isfinite(b).all()):
        raise ValueError(f"{label}: cosine of a non-finite tensor")
    a, b = a.double().flatten(), b.double().flatten()
    norm_a, norm_b = a.norm().item(), b.norm().item()
    if norm_a == 0.0 or norm_b == 0.0:
        raise ValueError(f"{label}: cosine of a zero-norm tensor has no direction (norms {norm_a}, {norm_b})")
    return torch.dot(a, b).item() / (norm_a * norm_b)


def max_abs_rel_err(actual: torch.Tensor, reference: torch.Tensor) -> float:
    """Largest elementwise error, relative to the reference's largest magnitude (fp32).

    Scaled by the peak rather than elementwise, so near-zero reference entries do not dominate. The
    floor only keeps an all-zero reference finite: any nonzero ``actual`` then reads as a huge error.
    NaN in either operand propagates, so a ``<`` bound fails on it.
    """
    diff = (actual.float() - reference.float()).abs().max()
    return (diff / reference.float().abs().max().clamp(min=1e-9)).item()


def fro_rel_err(actual: torch.Tensor, reference: torch.Tensor) -> float:
    """Frobenius norm of the error relative to the reference's (fp64).

    A zero-norm reference raises: a relative error against it is undefined, and a fixture that
    produces one tests nothing.
    """
    reference = reference.double()
    norm = reference.norm().item()
    if norm == 0.0:
        raise ValueError("relative error against a zero-norm reference is undefined")
    return (actual.double() - reference).norm().item() / norm


def relative_l2(actual: dict[str, torch.Tensor], reference: dict[str, torch.Tensor]) -> float:
    """``||actual - reference|| / ||reference||`` over two tensor dicts taken as one vector (fp32).

    ``inf`` when ``reference`` is empty or all zero, or the two hold different keys, so a comparison
    of nothing fails a bound instead of passing it; NaN in either operand propagates.
    """
    if not reference or set(actual) != set(reference):
        return math.inf
    num = sum(float((actual[key].float() - value.float()).pow(2).sum()) for key, value in reference.items())
    den = sum(float(value.float().pow(2).sum()) for value in reference.values())
    return math.sqrt(num / den) if den else math.inf


def _extreme_or_nan(pick: Callable, values: Iterable[float], default: float | None) -> float:
    values = list(values)
    if any(math.isnan(value) for value in values):
        return math.nan
    return pick(values) if default is None else pick(values, default=default)


def max_or_nan(values: Iterable[float], *, default: float | None = None) -> float:
    """Largest of ``values``, or NaN when any of them is NaN.

    The builtin ``max`` compares with ``>``, which is false against NaN, so a NaN survives only in first
    position and is silently dropped anywhere else; a bound checked on the result must fail on it
    instead. ``default`` is returned for an empty ``values``; without one, an empty input raises as
    ``max`` does.
    """
    return _extreme_or_nan(max, values, default)


def min_or_nan(values: Iterable[float], *, default: float | None = None) -> float:
    """Smallest of ``values``, or NaN when any of them is NaN: :func:`max_or_nan` for the builtin ``min``,
    which drops a NaN the same way."""
    return _extreme_or_nan(min, values, default)


def tensors_equal_at_narrower_dtype(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Bitwise equality of two tensors holding the same stored value at possibly different float dtypes.

    The lazy loader casts float parameters to the run's dtype where ``from_pretrained`` honors a
    family's fp32 pins, and keeps a float buffer at the checkpoint's dtype where ``from_pretrained``
    casts it; either side rounds the one stored value, so they agree at the narrower dtype. Shapes
    must match exactly.
    """
    if a.shape != b.shape:
        return False
    if a.is_floating_point() and b.is_floating_point() and a.dtype != b.dtype:
        narrow = min(a.dtype, b.dtype, key=lambda dt: torch.finfo(dt).bits)
        a, b = a.to(narrow), b.to(narrow)
    return torch.equal(a, b.to(a.device))


def params_off_dtype(model: torch.nn.Module, dtype: torch.dtype) -> list[str]:
    """Names of the floating parameters of ``model`` not stored in ``dtype``."""
    return [name for name, param in model.named_parameters() if param.is_floating_point() and param.dtype != dtype]


def log_spectrum_matrix(rows: int, cols: int, generator: torch.Generator, decades: float = 1.0) -> torch.Tensor:
    """A random fp64 ``rows x cols`` matrix whose singular values are log-spaced from 1 down ``decades``.

    Random orthonormal factors around a prescribed spectrum, so a test controls the conditioning an
    iterative method sees instead of inheriting a Gaussian matrix's near-zero tail. At one decade and
    a few hundred singular values the smallest stays above 1e-2 of the Frobenius norm, inside the Muon
    band's domain on either Newton-Schulz path (``TOL.muon_band_domain_*``).
    """
    rank = min(rows, cols)
    spectrum = torch.logspace(0, -decades, rank, dtype=torch.float64)
    left, _ = torch.linalg.qr(torch.randn(rows, rank, generator=generator, dtype=torch.float64))
    right, _ = torch.linalg.qr(torch.randn(cols, rank, generator=generator, dtype=torch.float64))
    return (left * spectrum) @ right.T


def assert_orthogonalized(update: torch.Tensor, source: torch.Tensor, label: str) -> None:
    """Assert ``update`` is Muon's orthogonalization of the matrix ``source``.

    Two independent properties: every singular value sits in the Newton-Schulz band, and the update
    points along ``source``'s own polar factor ``U V^T``. The band alone passes an update
    orthogonalized from the wrong matrix, which is the failure a batched step's restack produces.
    A ``source`` whose spectrum leaves the band's domain for its path (the standard iteration for a
    square matrix, the Gram iteration otherwise, under the default ``ns_algorithm``) is a broken
    fixture and fails as such.
    """
    rows, cols = source.shape[-2:]
    domain = TOL.muon_band_domain_square if rows == cols else TOL.muon_band_domain_rectangular
    u, source_singular_values, vh = torch.linalg.svd(source.double(), full_matrices=False)
    relative_min = (source_singular_values.min() / source_singular_values.norm()).item()
    assert relative_min >= domain, (
        f"{label}: premise: the source's smallest singular value is {relative_min:.2e} of its Frobenius "
        f"norm, below the {domain:.1e} the Newton-Schulz band holds from on a {rows}x{cols} matrix"
    )
    assert torch.isfinite(update).all(), f"{label}: the update is not finite"
    singular_values = torch.linalg.svdvals(update.double())
    low, high = singular_values.min().item(), singular_values.max().item()
    assert TOL.muon_orthogonal_sv_min <= low and high <= TOL.muon_orthogonal_sv_max, (
        f"{label}: singular values span [{low:.4f}, {high:.4f}], outside the Newton-Schulz band "
        f"[{TOL.muon_orthogonal_sv_min}, {TOL.muon_orthogonal_sv_max}]"
    )
    cosine = cos_sim(update, u @ vh, label=label)
    assert cosine >= TOL.muon_polar_cosine_min(), (
        f"{label}: cosine to the source's polar factor is {cosine:.4f} (min {TOL.muon_polar_cosine_min():.4f})"
    )


def autograd_nodes(tensor: torch.Tensor) -> list[torch.autograd.graph.Node]:
    """Every autograd node reachable from ``tensor``'s ``grad_fn``, each once: the backward a loss or
    kernel test reads its graph's shape off (which ops, how many, on what inputs)."""
    nodes, seen, pending = [], set(), [tensor.grad_fn]
    while pending:
        node = pending.pop()
        if node is None or node in seen:
            continue
        seen.add(node)
        nodes.append(node)
        pending.extend(parent for parent, _ in node.next_functions)
    return nodes


def perturb_trainable(model: torch.nn.Module, relative: float) -> int:
    """Scale every trainable element of ``model`` in place by ``1 + relative * sin(value * phase)``, computed in fp32
    and stored back once at the parameter's dtype, and return how many stored elements changed.

    The factor is a function of each element's own value, so ranks holding copies of one tensor (DP replicas of an
    EP bank, a weight replicated across a TP or ETP split) perturb it alike without a collective; a DTensor changes
    its local shard. A ``relative`` under half a ULP of the dtype rounds every element back (``5e-4`` in bf16), so
    the count is the caller's proof that the perturbation took.
    """
    changed = 0
    with torch.no_grad():
        for param in model.parameters():
            if not param.requires_grad:
                continue
            local = param.to_local() if isinstance(param, DTensor) else param
            wide = local.float()
            perturbed = (wide * (1 + relative * torch.sin(wide * _PERTURBATION_PHASE))).to(local.dtype)
            changed += int((perturbed != local).sum())
            local.copy_(perturbed)
    return changed
