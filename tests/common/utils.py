"""Shared utilities for the test suites.

Rank-aware logging, memory cleanup, repo-root resolution, by-path script import, out-of-process
probes, and the shared state-snapshot and assertion helpers.
"""

import ast
import gc
import importlib.util
import os
import pathlib
import subprocess
import sys
from types import ModuleType

import torch
import torch.distributed as dist
from safetensors import safe_open
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_optimizer_state_dict

from src.distributed.fsdp import reshard_fsdp2_modules
from src.models.structure import unwrap_model
from tests.common.tolerances import TOL

# Repository root for tests that read source files (drift pins, AST sweeps). Self-locating rather
# than cwd-derived: the training images bake a repo copy at /workspace, so a relative path can
# resolve against the wrong tree.
REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]


def load_script_module(relative_path: str, name: str | None = None, *, register: bool = False) -> ModuleType:
    """Import a ``scripts/`` entry point by path, since ``scripts/`` is not a package.

    ``relative_path`` is relative to :data:`REPO_ROOT`; ``name`` defaults to the file stem.
    ``register`` also binds the module in ``sys.modules`` so ``unittest.mock.patch("<name>.x")``
    and pickling resolve it; leave it off for a plain by-path import.
    """
    path = REPO_ROOT / relative_path
    if not path.is_file():
        raise FileNotFoundError(f"no module to import at {path}")
    module_name = name or path.stem
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"no loadable import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    if register:
        # Before exec: a module that resolves string annotations or pickles its own classes reads
        # itself back out of sys.modules while its body is still running.
        sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def probe_findings(script: str, marker: str) -> list[str]:
    """Run ``script`` in a fresh interpreter at the repo root; return the findings it reported.

    A probe whose subject rebinds process-global state (a Liger applier swapping upstream module
    classes, a funnel wrapper installed on ``transformers.dynamic_module_utils``) would leak into
    every later test, so it runs out-of-process and reports through one line,
    ``print(marker + "|".join(findings))``, where an empty list means nothing was wrong. A probe that
    dies, or prints no marker line, raises here rather than reading as a clean run.
    """
    proc = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, cwd=REPO_ROOT, check=False)
    assert proc.returncode == 0, f"probe subprocess died:\n{proc.stdout}\n{proc.stderr}"

    line = next((line for line in proc.stdout.splitlines() if line.startswith(marker)), None)
    assert line is not None, f"probe printed no {marker!r} line:\n{proc.stdout}\n{proc.stderr}"
    return [finding for finding in line.removeprefix(marker).split("|") if finding]


def imports_name(path: pathlib.Path, name: str) -> bool:
    """Whether the module at ``path`` imports ``name``.

    The binding is the check rather than a text match: CLAUDE.md bans function-local imports, so a
    module-level import is the only way a call site reaches a symbol, while a mention in a docstring
    or a comment is not a use. ``asname`` counts, since an aliased import binds the name too.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return any(
        isinstance(node, ast.ImportFrom | ast.Import)
        and any(alias.name == name or alias.asname == name for alias in node.names)
        for node in ast.walk(tree)
    )


def safetensors_state_dict(checkpoint_dir: str | pathlib.Path) -> dict[str, torch.Tensor]:
    """Every tensor across a checkpoint directory's ``*.safetensors`` parts, keyed exactly as written.

    Reads the parts directly, since the files are the artifact under test and nothing may re-spell or
    re-cast them on the way in as ``from_pretrained`` does.
    """
    tensors: dict[str, torch.Tensor] = {}
    for part in sorted(pathlib.Path(checkpoint_dir).glob("*.safetensors")):
        with safe_open(str(part), framework="pt") as reader:
            for key in reader.keys():  # noqa: SIM118 - safe_open has .keys() but is not a mapping
                tensors[key] = reader.get_tensor(key)
    return tensors


def _current_rank() -> int:
    """This process's rank, surviving group teardown.

    After ``destroy_process_group`` (tests that tear down before a rank-0-only phase) ``dist`` can no
    longer answer but torchrun's ``RANK`` still can. Falling back to 0 there would make every rank
    print ``[Rank 0]``, so a non-zero rank's verdict could not be told from rank 0's.
    """
    if dist.is_initialized():
        return dist.get_rank()
    return int(os.environ.get("RANK", "0"))


def log(msg: str, rank: int | None = 0):
    """Log message from specified rank only."""
    current_rank = _current_rank()
    if current_rank == rank:
        print(f"[Rank {current_rank}] {msg}", flush=True)


def log_all(msg: str):
    """Log message from all ranks."""
    print(f"[Rank {_current_rank()}] {msg}", flush=True)


def cleanup_memory():
    """Free GPU memory with synchronization."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def gpu_mem_gb(device=None) -> float:
    """Current GPU memory usage in GB."""
    if device is None:
        device = torch.cuda.current_device()
    return torch.cuda.memory_allocated(device) / 1e9


def gpu_peak_mem_gb(device=None) -> float:
    """Peak GPU memory usage in GB."""
    if device is None:
        device = torch.cuda.current_device()
    return torch.cuda.max_memory_allocated(device) / 1e9


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


def local_optimizer_state(model, optimizer) -> dict:
    """This rank's optimizer state as an :func:`assert_optimizer_state_bit_exact` payload.

    The per-rank ``optimizer_shard_XXXXX.pt`` files hold this view (``get_optimizer_state_dict`` at
    ``full_state_dict=False``, keyed by param FQN), so a snapshot taken here is comparable to what a
    resume restores. DTensor entries are reduced to their local shard, keeping the comparison
    rank-local with no collective, so a per-rank divergence is not averaged away.

    ``reshard_fsdp2_modules`` first, for the same reason the save path does it: after an eval-only
    forward the FSDP2 modules still carry their transient unsharded params, and
    ``get_optimizer_state_dict`` then maps no optimizer param to an FQN (``KeyError: 0``, or an empty
    state).
    """
    reshard_fsdp2_modules(model)
    osd = get_optimizer_state_dict(
        unwrap_model(model),
        optimizer,
        options=StateDictOptions(full_state_dict=False, cpu_offload=True),
    )

    def _local(value):
        if not torch.is_tensor(value):
            return value
        local = value.to_local() if hasattr(value, "to_local") else value
        return local.detach().clone().cpu()

    return {
        "state": {
            fqn: {key: _local(value) for key, value in entry.items()} for fqn, entry in osd.get("state", {}).items()
        }
    }


def assert_optimizer_state_bit_exact(saved: dict, restored: dict) -> None:
    """Assert two ``optimizer.state_dict()`` payloads carry identical per-param state.

    Fails when a ``load_state_dict`` drops, casts, re-quantizes or resets any state tensor (or
    scalar, such as ``step``). ``saved`` must be non-empty, since comparing two empty states would
    check nothing.
    """
    assert len(saved["state"]) > 0, "premise: the saved optimizer state is empty"
    assert saved["state"].keys() == restored["state"].keys(), (
        f"restored state covers params {sorted(restored['state'])}, saved {sorted(saved['state'])}"
    )
    for idx, saved_entry in saved["state"].items():
        restored_entry = restored["state"][idx]
        assert saved_entry.keys() == restored_entry.keys(), f"param {idx}: state keys differ"
        for key, val in saved_entry.items():
            if torch.is_tensor(val):
                res = restored_entry[key]
                assert torch.is_tensor(res) and res.dtype == val.dtype and torch.equal(res.to(val.device), val), (
                    f"param {idx} state '{key}' not bit-exact after load"
                )
            else:
                assert restored_entry[key] == val, f"param {idx} state '{key}' changed after load"


def optimizer_state_matches(saved: dict, restored: dict) -> tuple[bool, str]:
    """:func:`assert_optimizer_state_bit_exact` as an ``(ok, why)`` pair instead of a raise.

    Every rank has to reach the next collective: a bare assertion on one rank leaves its peers
    blocked, so a distributed test records the verdict and agrees it rather than aborting the body.
    """
    try:
        assert_optimizer_state_bit_exact(saved, restored)
    except AssertionError as e:
        return False, str(e)
    return True, ""


def step_losses(trainer) -> list[float]:
    """Per-step training losses from a trainer's log history, eval entries and the run summary excluded."""
    return [e["loss"] for e in trainer.state.log_history if "loss" in e and "eval_loss" not in e]


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
