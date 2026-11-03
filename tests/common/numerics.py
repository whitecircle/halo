"""Numerical comparisons the correctness tests bound: error metrics, NaN-carrying extremes, Muon orthogonality,
autograd graphs."""

import math
from collections.abc import Callable, Iterable

import torch
from torch.distributed.tensor import DTensor

from tests.common.tolerances import TOL

# Spreads ``sin(value * phase)`` over its whole range across weights of similar magnitude.
_PERTURBATION_PHASE = 12345.678


def cos_sim(a: torch.Tensor, b: torch.Tensor, *, label: str) -> float:
    """Cosine similarity of two tensors as flat fp64 vectors, since bf16 accumulation rounds and an fp32 norm can
    overflow near the bf16 limit. A zero-norm or non-finite operand raises, naming ``label``: any number returned for
    it passes one side of some threshold."""
    if not (torch.isfinite(a).all() and torch.isfinite(b).all()):
        raise ValueError(f"{label}: cosine of a non-finite tensor")
    a, b = a.double().flatten(), b.double().flatten()
    norm_a, norm_b = a.norm().item(), b.norm().item()
    if norm_a == 0.0 or norm_b == 0.0:
        raise ValueError(f"{label}: cosine of a zero-norm tensor has no direction (norms {norm_a}, {norm_b})")
    return torch.dot(a, b).item() / (norm_a * norm_b)


def max_abs_rel_err(actual: torch.Tensor, reference: torch.Tensor) -> float:
    """Largest elementwise error relative to the reference's peak (fp32); a zero reference reads as a huge error, and
    NaN propagates."""
    diff = (actual.float() - reference.float()).abs().max()
    return (diff / reference.float().abs().max().clamp(min=1e-9)).item()


def fro_rel_err(actual: torch.Tensor, reference: torch.Tensor) -> float:
    """Frobenius norm of the error relative to the reference's (fp64); a zero-norm reference raises, since such a
    fixture tests nothing."""
    reference = reference.double()
    norm = reference.norm().item()
    if norm == 0.0:
        raise ValueError("relative error against a zero-norm reference is undefined")
    return (actual.double() - reference).norm().item() / norm


def relative_l2(actual: dict[str, torch.Tensor], reference: dict[str, torch.Tensor]) -> float:
    """``||actual - reference|| / ||reference||`` over two dicts as one vector (fp32); ``inf`` when empty, all zero or
    the keys differ."""
    if not reference or set(actual) != set(reference):
        return math.inf
    num = sum(float((actual[key].float() - value.float()).pow(2).sum()) for key, value in reference.items())
    den = sum(float(value.float().pow(2).sum()) for value in reference.values())
    return math.sqrt(num / den) if den else math.inf


def worst_relative_l2(actual: dict[str, torch.Tensor], reference: dict[str, torch.Tensor]) -> tuple[float, str | None]:
    """The largest per-tensor relative L2 over two dicts (fp64) and its key, so a small tensor's error does not
    dissolve into a large one's. ``(inf, None)`` when empty or the keys differ. A zero-norm reference counts any
    difference as ``inf``. NaN returns NaN with its key."""
    if not reference or set(actual) != set(reference):
        return math.inf, None
    worst, worst_key = -math.inf, None
    for key, ref in reference.items():
        ref = ref.double()
        error, norm = (actual[key].double() - ref).norm().item(), ref.norm().item()
        relative = error / norm if norm else (0.0 if error == 0.0 else math.inf)
        if math.isnan(relative):
            return math.nan, key
        if relative > worst:
            worst, worst_key = relative, key
    return worst, worst_key


def _extreme_or_nan(pick: Callable, values: Iterable[float], default: float | None) -> float:
    values = list(values)
    if any(math.isnan(value) for value in values):
        return math.nan
    return pick(values) if default is None else pick(values, default=default)


def max_or_nan(values: Iterable[float], *, default: float | None = None) -> float:
    """Largest of ``values``, or NaN when any is NaN; the builtin ``max`` silently drops a NaN anywhere but first
    position."""
    return _extreme_or_nan(max, values, default)


def min_or_nan(values: Iterable[float], *, default: float | None = None) -> float:
    """Smallest of ``values``, or NaN when any is NaN: :func:`max_or_nan` for the builtin ``min``."""
    return _extreme_or_nan(min, values, default)


def tensors_equal_at_narrower_dtype(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Bitwise equality at the narrower of two float dtypes; loaders round one stored value differently per dtype.
    Shapes must match."""
    if a.shape != b.shape:
        return False
    if a.is_floating_point() and b.is_floating_point() and a.dtype != b.dtype:
        narrow = min(a.dtype, b.dtype, key=lambda dt: torch.finfo(dt).bits)
        a, b = a.to(narrow), b.to(narrow)
    return torch.equal(a, b.to(a.device))


def params_off_dtype(model: torch.nn.Module, dtype: torch.dtype) -> list[str]:
    return [name for name, param in model.named_parameters() if param.is_floating_point() and param.dtype != dtype]


def log_spectrum_matrix(rows: int, cols: int, generator: torch.Generator, decades: float = 1.0) -> torch.Tensor:
    """A random fp64 ``rows x cols`` matrix with singular values log-spaced from 1 down ``decades``. At one decade and
    a few hundred SVs the smallest stays above 1e-2 of the Frobenius norm, inside the Muon band's domain on either
    path."""
    rank = min(rows, cols)
    spectrum = torch.logspace(0, -decades, rank, dtype=torch.float64)
    left, _ = torch.linalg.qr(torch.randn(rows, rank, generator=generator, dtype=torch.float64))
    right, _ = torch.linalg.qr(torch.randn(cols, rank, generator=generator, dtype=torch.float64))
    return (left * spectrum) @ right.T


def assert_orthogonalized(update: torch.Tensor, source: torch.Tensor, label: str) -> None:
    """Assert ``update`` is Muon's orthogonalization of ``source``: SVs in the band, and aligned with ``source``'s
    polar factor. The band alone passes an update from the wrong matrix, which is what a batched step's restack
    produces. A ``source`` outside the band's domain is a broken fixture."""
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
    """Every autograd node reachable from ``tensor``'s ``grad_fn``, each once."""
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
    """Scale every trainable element by ``1 + relative * sin(value * phase)`` in place; returns how many stored
    elements changed. A function of each element's own value, so replica ranks perturb alike with no collective. Under
    half a ULP rounds back (``5e-4`` in bf16), hence the count."""
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
