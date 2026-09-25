#!/usr/bin/env python
"""Tests for scripts/after_training/merge_models.py.

Four layers: (1) the pre-I/O input gates — method↔knob, knob ranges, finite weights; (2) the per-tensor
merge math (linear / slerp / task_arithmetic / ties) on known values; (3) the RAM preflight — each
method's declared float32 working set against the allocator's peak, and the estimate the preflight
builds from it; (4) an end-to-end merge of two tiny **Qwen3.5 MoE** checkpoints through the streaming
pipeline — write → reload with the real model class → verify the merged weights match the expected
interpolation and the model forwards.

Run: ``python tests/cpu/checkpoint/test_merge_models.py`` (or ``pytest -m cpu``).
"""

from __future__ import annotations

import json
import re
import tempfile
import warnings
import weakref
from collections.abc import Callable
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file
from torch.profiler import ProfilerActivity, profile
from transformers import CONFIG_MAPPING
from transformers.models.qwen3_5_moe import Qwen3_5MoeForCausalLM, Qwen3_5MoeTextConfig

from tests.common.utils import load_script_module

mm = load_script_module("scripts/after_training/merge_models.py")

_WORKING_SET_NUMEL = 1 << 20
# The ops also allocate a few 0-d results (norms, the slerp dot); they are not tensor-sized copies.
_SCALAR_SLACK_BYTES = 4096


def _write_tiny_checkpoint(path: Path, tensors: dict[str, torch.Tensor]) -> Path:
    """A one-shard checkpoint beside a real config (the aux copy round-trips it through AutoConfig)."""
    path.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(path / "model.safetensors"))
    CONFIG_MAPPING["qwen3"]().save_pretrained(str(path))
    return path


def test_parse_model_spec():
    # None (not 1.0) when no weight was given: the method↔knob gate must see explicitness.
    assert mm._parse_model_spec("/a/b") == ("/a/b", None)
    assert mm._parse_model_spec("/a/b:0.25") == ("/a/b", 0.25)
    # A trailing non-float colon segment is part of the path (e.g. an HF id), not a weight.
    assert mm._parse_model_spec("org/model") == ("org/model", None)


def test_a_knob_the_method_ignores_is_refused():
    """Ungated, `--models a:0.1 b:0.9` under slerp is silently IGNORED (slerp reads only --t), and
    --t/--density/--lambda are dead under the methods that never consume them — the merge then
    produces something other than what was asked, with no error. The gate raises before any I/O, so
    the model paths here deliberately do not exist."""
    with pytest.raises(ValueError, match="does not use models:weight"):
        mm.merge_models(["/nope/a:0.1", "/nope/b:0.9"], "/nope/out", method="slerp")
    with pytest.raises(ValueError, match="does not use t"):
        mm.merge_models(["/nope/a", "/nope/b"], "/nope/out", method="linear", knobs={"t": 0.5})
    with pytest.raises(ValueError, match="does not use density, lambda"):
        mm.merge_models(["/nope/a", "/nope/b"], "/nope/out", method="linear", knobs={"density": 0.6, "lambda": 1.0})
    with pytest.raises(ValueError, match="does not use base_model"):
        mm.merge_models(["/nope/a", "/nope/b"], "/nope/out", method="slerp", base_model="/nope/base")


# The full method↔knob contract, spelled out independently of the table the code reads.
_METHOD_KNOBS = {
    "linear": {"models:weight"},
    "slerp": {"t"},
    "task_arithmetic": {"models:weight", "base_model"},
    "ties": {"models:weight", "base_model", "density", "lambda"},
}
_ALL_KNOBS = sorted(set().union(*_METHOD_KNOBS.values()))


@pytest.mark.parametrize("method", sorted(_METHOD_KNOBS))
@pytest.mark.parametrize("knob", _ALL_KNOBS)
def test_every_method_knob_pair_accepts_or_refuses_as_declared(method, knob):
    """Every (method, knob) pair, accept and refuse alike.

    The gate is the only thing between a knob the method never reads and a merge that is silently
    not the one asked for — and over-rejecting a knob the method *does* read is just as wrong. The
    matrix is written out here so a change to the method table has to be a deliberate one.
    """
    if knob in _METHOD_KNOBS[method]:
        mm._check_method_knobs(method, {knob})
    else:
        with pytest.raises(ValueError, match=re.escape(f"--method {method} does not use {knob}")):
            mm._check_method_knobs(method, {knob})


def test_the_method_table_declares_every_method_and_default():
    """The knob defaults have one home, and every method the CLI offers is in it."""
    assert set(mm._METHODS) == set(_METHOD_KNOBS)
    assert mm._KNOB_DEFAULTS == {"t": 0.5, "density": 0.6, "lambda": 1.0}
    for method, knobs in _METHOD_KNOBS.items():
        assert mm._method_knobs(method) == frozenset(knobs)
        mm._check_method_knobs(method, set())  # nothing explicit is always fine (defaults apply)


@pytest.mark.parametrize(
    ("method", "knob", "value"),
    [
        ("slerp", "t", -0.1),
        ("slerp", "t", 1.5),
        ("slerp", "t", float("nan")),
        ("ties", "density", 0.0),
        ("ties", "density", -0.5),
        ("ties", "density", 1.5),
        ("ties", "density", float("nan")),
        ("ties", "lambda", float("nan")),
        ("ties", "lambda", float("inf")),
    ],
)
def test_a_knob_outside_its_domain_is_refused_before_any_io(method, knob, value):
    """None of these raise inside the ops: a density outside (0, 1] skips the trim and keeps every
    delta, a t outside [0, 1] extrapolates past both models, and a non-finite value writes non-finite
    weights — each a merge other than the one asked for. The paths do not exist, so the refusal must
    come before any read."""
    base = "/nope/base" if method == "ties" else None
    with pytest.raises(ValueError, match=re.escape(f"--{knob} must be")):
        mm.merge_models(["/nope/a", "/nope/b"], "/nope/out", method=method, base_model=base, knobs={knob: value})


@pytest.mark.parametrize("weight", ["nan", "inf", "-inf"])
def test_a_non_finite_model_weight_is_refused_before_any_io(weight):
    """``path:nan`` parses as a weight, and every merge that reads weights would write it into the
    checkpoint as non-finite tensors."""
    with pytest.raises(ValueError, match="--models weights must be finite"):
        mm.merge_models([f"/nope/a:{weight}", "/nope/b"], "/nope/out", method="linear")


def test_the_domain_edges_and_defaults_are_accepted():
    """Over-refusing is as wrong as under-refusing: both slerp endpoints, a full-density TIES (no
    trim), a zero or negative lambda (the base, or the task vector negated) and every default merge."""
    mm._check_knob_domains("slerp", {"t": 0.0})
    mm._check_knob_domains("slerp", {"t": 1.0})
    mm._check_knob_domains("ties", {"density": 1.0, "lambda": 0.0})
    mm._check_knob_domains("ties", {"density": 1e-6, "lambda": -1.0})
    for method in mm._METHODS:
        mm._check_knob_domains(method, mm._KNOB_DEFAULTS)
    assert set(mm._KNOB_DOMAINS) == set(mm._KNOB_DEFAULTS), "every knob needs a domain, and nothing else"


def test_an_unknown_knob_is_refused():
    """A knob no merge op declares is a typo, not a no-op: silently dropping it would merge with the
    method's default instead of the value the caller asked for."""
    with pytest.raises(ValueError, match=r"unknown merge knob\(s\) \['lam'\]"):
        mm.merge_models(["/nope/a", "/nope/b"], "/nope/out", method="ties", knobs={"lam": 1.0})


def test_dispatch_passes_each_op_the_knobs_its_signature_names():
    """The registry dispatch must reach the op with the caller's knob, not the default: a ties merge
    at ``lambda=0`` is exactly the base model, and at ``lambda=1`` it is the task vector on top."""
    with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
        base = _write_tiny_checkpoint(Path(tmp) / "base", {"w": torch.zeros(4)})
        model = _write_tiny_checkpoint(Path(tmp) / "m", {"w": torch.ones(4)})

        for lam, expected in ((0.0, 0.0), (1.0, 1.0)):
            out = Path(tmp) / f"out-{lam}"
            mm.merge_models(
                [str(model), str(model)],
                str(out),
                method="ties",
                base_model=str(base),
                knobs={"lambda": lam, "density": 1.0},
                dtype="float32",
                allow_missing_tokenizer=True,
                verbose=False,
            )
            merged = load_file(str(out / "model.safetensors"))["w"]
            assert torch.allclose(merged, torch.full((4,), expected)), f"lambda={lam} did not reach _merge_ties"


def test_linear_weighted_average():
    a = torch.ones(4)
    b = torch.zeros(4)
    # Always normalized by the weight sum: weights 1 and 3 → 0.25*a + 0.75*b = 0.25
    out = mm._merge_linear([a, b], [1.0, 3.0])
    assert torch.allclose(out, torch.full((4,), 0.25))
    # Default (equal) weights → a plain average, not a scale-doubling sum.
    out = mm._merge_linear([a, b], [1.0, 1.0])
    assert torch.allclose(out, torch.full((4,), 0.5))


def test_slerp_endpoints_and_midpoint():
    a = torch.tensor([1.0, 0.0, 0.0, 0.0])
    b = torch.tensor([0.0, 1.0, 0.0, 0.0])  # orthogonal unit vectors
    assert torch.allclose(mm._merge_slerp(a, b, 0.0), a, atol=1e-5)
    assert torch.allclose(mm._merge_slerp(a, b, 1.0), b, atol=1e-5)
    mid = mm._merge_slerp(a, b, 0.5)
    # Halfway on the unit circle between two orthonormal vectors: both coords = cos(45°).
    assert torch.allclose(mid, torch.tensor([0.70710677, 0.70710677, 0.0, 0.0]), atol=1e-5)
    assert torch.allclose(mid.norm(), torch.tensor(1.0), atol=1e-5)


def test_slerp_colinear_falls_back_to_lerp():
    a = torch.tensor([1.0, 2.0, 3.0])
    b = 2.0 * a  # colinear → lerp path
    out = mm._merge_slerp(a, b, 0.5)
    assert torch.allclose(out, torch.lerp(a, b, 0.5), atol=1e-5)


def test_slerp_of_neutralized_sinks_stays_finite():
    """Two gpt-oss checkpoints trained under ``reset_sinks`` (the SFT default) carry every sink at
    ``bfloat16.min``, whose square overflows fp32: a raw norm read ``inf``, the cosine of two
    identical vectors read 0, and the "orthogonal" arc summed to ``-inf`` — a non-finite sink in a
    checkpoint whose inputs were finite. Identical inputs must slerp to themselves, at any scale."""
    sinks = torch.full((64,), torch.finfo(torch.bfloat16).min, dtype=torch.bfloat16)
    out = mm._merge_slerp(sinks, sinks.clone(), 0.5)
    assert torch.isfinite(out).all(), "slerp of finite inputs produced a non-finite tensor"
    assert torch.equal(out, sinks.float())
    # The magnitude normalization must not leak into the result: an ordinary non-colinear pair still
    # lands on the great-circle arc at its own scale.
    a = torch.tensor([3.0, 0.0, 0.0, 0.0])
    b = torch.tensor([0.0, 3.0, 0.0, 0.0])
    mid = mm._merge_slerp(a, b, 0.5)
    assert torch.allclose(mid, torch.tensor([3 * 0.70710677, 3 * 0.70710677, 0.0, 0.0]), atol=1e-5)


def test_task_arithmetic_adds_task_vectors():
    base = torch.zeros(3)
    m1 = torch.tensor([1.0, 0.0, 0.0])
    m2 = torch.tensor([0.0, 2.0, 0.0])
    # base + 1.0*(m1-base) + 0.5*(m2-base) = [1, 1, 0]
    out = mm._merge_task_arithmetic(base, [m1, m2], [1.0, 0.5])
    assert torch.allclose(out, torch.tensor([1.0, 1.0, 0.0]))


def test_ties_sign_election_and_disjoint_merge():
    base = torch.zeros(3)
    m1 = torch.tensor([2.0, 0.1, 0.0])
    m2 = torch.tensor([-1.0, 0.3, 0.0])
    out = mm._merge_ties(base, [m1, m2], [1.0, 1.0], density=1.0, lambda_=1.0)
    # elem 0: signs disagree, elected sign + → only m1 counts (2.0); elem 1: both + → mean 0.2.
    assert torch.allclose(out, torch.tensor([2.0, 0.2, 0.0]), atol=1e-6)


def test_ties_density_trims_small_deltas():
    base = torch.zeros(4)
    delta = torch.tensor([0.05, 0.1, 5.0, 10.0])  # top-50% magnitudes are the last two
    out = mm._merge_ties(base, [delta], [1.0], density=0.5, lambda_=1.0)
    assert torch.allclose(out, torch.tensor([0.0, 0.0, 5.0, 10.0]), atol=1e-6)


def test_reference_keys_uses_base_for_task_methods():
    """task_arithmetic/ties merge over the BASE key set (vectors are base-relative), so a key in base
    but absent from model[0] is included — iterating model[0] instead would silently drop it.
    linear/slerp have no base and use model[0]."""

    class _StubReader:
        def __init__(self, ks):
            self._ks = set(ks)

        def keys(self):
            return self._ks

    m0 = _StubReader({"x"})
    m1 = _StubReader({"x", "y"})
    base = _StubReader({"x", "y"})  # 'y' is absent from model[0]
    # base-relative methods cover 'y' (would be dropped if iterating m0)
    assert mm._reference_keys("task_arithmetic", [m0, m1], base) == ["x", "y"]
    assert mm._reference_keys("ties", [m0, m1], base) == ["x", "y"]
    assert mm._reference_keys("linear", [m0, m1], None) == ["x"]
    assert mm._reference_keys("slerp", [m0, m1], None) == ["x"]


_TINY_QWEN35 = {
    "vocab_size": 256,
    "hidden_size": 64,
    "intermediate_size": 128,
    "moe_intermediate_size": 32,
    "shared_expert_intermediate_size": 64,
    "num_experts": 4,
    "num_experts_per_tok": 2,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 16,
    "hidden_act": "silu",
    "max_position_embeddings": 128,
    "tie_word_embeddings": False,
    "rope_parameters": {"rope_type": "default", "rope_theta": 10000.0},
}


def _build_tiny_qwen35(out_dir: Path, seed: int) -> None:
    torch.manual_seed(seed)
    config = Qwen3_5MoeTextConfig(**_TINY_QWEN35, layer_types=["full_attention"] * _TINY_QWEN35["num_hidden_layers"])
    Qwen3_5MoeForCausalLM(config).to(torch.bfloat16).save_pretrained(out_dir, safe_serialization=True)


def test_end_to_end_linear_merge_qwen3_5():
    """Merge two tiny Qwen3.5 checkpoints (linear 0.5/0.5), reload, verify average + a forward pass.
    Also pins that resume sidecars planted beside the tokenizer source do NOT ship: they describe
    one input run's state, and the merged artifact has no such run."""
    with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
        a, b, out = Path(tmp) / "a", Path(tmp) / "b", Path(tmp) / "merged"
        _build_tiny_qwen35(a, seed=0)
        _build_tiny_qwen35(b, seed=1)
        for sidecar in ("scheduler.pt", "rng_state_0.pth", "router_balancing_biases.pt"):
            (a / sidecar).write_bytes(b"x")

        mm.merge_models(
            model_specs=[str(a), str(b)],
            output_dir=str(out),
            method="linear",
            dtype="bfloat16",
            tokenizer_source=str(a),
            allow_missing_tokenizer=True,
            verbose=False,
        )

        for sidecar in ("scheduler.pt", "rng_state_0.pth", "router_balancing_biases.pt"):
            assert not (out / sidecar).exists(), f"{sidecar} is one input run's resume state, not the merge's"

        # Reload as the real Qwen3.5 class — proves config/shards/index are valid.
        merged = Qwen3_5MoeForCausalLM.from_pretrained(out, dtype=torch.bfloat16)
        ma = Qwen3_5MoeForCausalLM.from_pretrained(a, dtype=torch.bfloat16)
        mb = Qwen3_5MoeForCausalLM.from_pretrained(b, dtype=torch.bfloat16)

        msd, asd, bsd = merged.state_dict(), ma.state_dict(), mb.state_dict()
        assert set(msd) == set(asd), "merged checkpoint dropped/added keys"
        checked = 0
        for k, v in msd.items():
            if not v.is_floating_point():
                continue
            expected = 0.5 * asd[k].float() + 0.5 * bsd[k].float()
            assert torch.allclose(v.float(), expected, atol=2e-2), f"{k} != 0.5*(a+b)"
            checked += 1
        assert checked > 0

        # A weight-space average is not a logit-space average (the net is nonlinear), so parity is
        # checked against an INDEPENDENTLY averaged reference — proving the merge pipeline carries
        # the averaged weights into compute, not just onto disk.
        ids = torch.randint(0, _TINY_QWEN35["vocab_size"], (1, 8))
        reference = Qwen3_5MoeForCausalLM.from_pretrained(a, dtype=torch.bfloat16)
        independent_avg = {
            k: (0.5 * asd[k].float() + 0.5 * bsd[k].float()).to(v.dtype) if v.is_floating_point() else v
            for k, v in reference.state_dict().items()
        }
        reference.load_state_dict(independent_avg)
        with torch.no_grad():
            merged_logits = merged(input_ids=ids, use_cache=False).logits
            reference_logits = reference(input_ids=ids, use_cache=False).logits
        # bf16 round-trip through the merge pipeline vs a direct average: allow a tiny tolerance.
        assert torch.allclose(merged_logits, reference_logits, atol=1e-2), (
            "reloaded merged model's forward diverges from an independently-averaged reference"
        )


def test_balancing_biases_keep_their_trained_dtype():
    """`--dtype bfloat16` must not quantize router balancing tensors: the ALF sign-update biases are
    trained fp32, near-tied top-k picks flip under a bf16 round-trip, and every other merge path
    (merge_ep_shards, the direct gathered save) already keeps them at trained dtype."""
    with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
        a, b, out = Path(tmp) / "a", Path(tmp) / "b", Path(tmp) / "merged"
        for path, seed in ((a, 0), (b, 1)):
            torch.manual_seed(seed)
            _write_tiny_checkpoint(
                path,
                {
                    "model.layers.0.mlp.gate.e_score_correction_bias": torch.randn(4, dtype=torch.float32),
                    "model.layers.0.mlp.down.weight": torch.randn(4, 4, dtype=torch.float32),
                },
            )

        mm.merge_models(
            [str(a), str(b)], str(out), method="linear", dtype="bfloat16", allow_missing_tokenizer=True, verbose=False
        )

        merged = load_file(str(out / "model.safetensors"))
        assert merged["model.layers.0.mlp.down.weight"].dtype == torch.bfloat16, "--dtype must still apply to weights"
        bias = merged["model.layers.0.mlp.gate.e_score_correction_bias"]
        assert bias.dtype == torch.float32, "balancing biases must export at trained dtype, not --dtype"
        expected = 0.5 * load_file(str(a / "model.safetensors"))["model.layers.0.mlp.gate.e_score_correction_bias"]
        expected += 0.5 * load_file(str(b / "model.safetensors"))["model.layers.0.mlp.gate.e_score_correction_bias"]
        assert torch.allclose(bias, expected)


def test_a_base_only_key_names_the_model_that_lacks_it():
    """``task_arithmetic``/``ties`` merge over the BASE key set, and the RAM preflight sizes those
    keys against ``model[0]`` — ahead of the merge loop's own coverage check. A base key no
    fine-tune carries (the realistic case: a base saved untied, the runs saved tied) therefore
    surfaced as a bare ``KeyError: 'lm_head.weight'`` naming neither the model that lacks it nor
    why it was wanted."""
    with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
        base, a, b, out = Path(tmp) / "base", Path(tmp) / "a", Path(tmp) / "b", Path(tmp) / "merged"
        shared = {"model.layers.0.mlp.down.weight": torch.zeros(4, 4)}
        _write_tiny_checkpoint(base, {**shared, "lm_head.weight": torch.zeros(4, 4)})
        for path in (a, b):
            _write_tiny_checkpoint(path, dict(shared))

        with pytest.raises(KeyError, match=r"lm_head\.weight.* missing from"):
            mm.merge_models(
                [str(a), str(b)],
                str(out),
                method="task_arithmetic",
                base_model=str(base),
                allow_missing_tokenizer=True,
                verbose=False,
            )


def _peak_allocated_bytes(fn: Callable[[], object], trace_path: Path) -> int:
    """Peak bytes the CPU allocator holds live while ``fn`` runs, over what was live before it.

    Read from the ``[memory]`` events of the profiler's exported trace: each carries the allocator's
    running ``Total Allocated``, kernel-internal buffers (``kthvalue``'s copy and indices) included.
    The figure is exact and independent of thread count, host load and process RSS, whose high-water
    mark the kernel samples from per-CPU counters and undercounts once faults spread across CPUs.
    """
    with warnings.catch_warnings():
        # Emitted at profiler start about multi-cycle schedules; this is one un-scheduled cycle.
        warnings.filterwarnings("ignore", message="Warning: Profiler clears events", category=UserWarning)
        with profile(activities=[ProfilerActivity.CPU], profile_memory=True) as prof:
            result = fn()
            del result
    prof.export_chrome_trace(str(trace_path))
    events = json.loads(trace_path.read_text(encoding="utf-8"))["traceEvents"]
    allocator = sorted(
        (event["ts"], event["args"]["Total Allocated"], event["args"]["Bytes"])
        for event in events
        if event.get("name") == "[memory]" and event["args"].get("Device Type") == 0
    )
    if not allocator:
        return 0
    _, first_total, first_bytes = allocator[0]
    return max(total for _, total, _ in allocator) - (first_total - first_bytes)


@pytest.mark.parametrize(
    ("method", "n_models"),
    [("linear", 2), ("linear", 4), ("slerp", 2), ("task_arithmetic", 2), ("task_arithmetic", 4)]
    + [("ties", n_models) for n_models in (1, 2, 4)],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "fp32"])
def test_each_method_working_set_bound_covers_its_peak(method, n_models, dtype, tmp_path):
    """The RAM preflight sizes a merge by each method's declared float32 working set, which grows with
    the model count under TIES: every model's delta, their stacked copy, the sign-masked copy and the
    int64 cast of the agreement mask coexist. A bound below the peak lets a merge OOM without the
    warning; one above it warns on merges that fit — so bf16 inputs, the worst case, must also land
    within 10% of the bound."""
    spec = mm._METHODS[method]
    tensors = [torch.randn(_WORKING_SET_NUMEL).to(dtype) for _ in range(n_models)]
    per_key = {
        "tensors": tensors,
        "t0": tensors[0],
        "t1": tensors[-1],
        "weights": [1.0] * n_models,
        "base": torch.randn(_WORKING_SET_NUMEL).to(dtype),
    }
    knobs = {mm._knob_dest(knob): value for knob, value in spec.knobs.items()}

    op_args = {name: per_key[name] for name in spec.tensor_args}
    peak = _peak_allocated_bytes(lambda: spec.op(**op_args, **knobs), tmp_path / "trace.json")

    copy_bytes = torch.float32.itemsize * _WORKING_SET_NUMEL
    bound = spec.fp32_copies(n_models) * copy_bytes
    assert peak > copy_bytes, f"measured {peak} bytes, under one fp32 copy — the probe measured nothing"
    assert peak <= bound + _SCALAR_SLACK_BYTES, f"peak {peak / copy_bytes:.3f} fp32 copies exceeds the bound"
    if dtype == torch.bfloat16:
        assert bound <= 1.1 * peak, f"bound {bound / copy_bytes} is far above the peak {peak / copy_bytes:.3f}"


def test_the_ram_preflight_sizes_inputs_as_stored_plus_the_method_working_set(tmp_path, monkeypatch):
    """The RAM estimate is the costliest key — every contributor's copy as stored (a fp32 base beside
    bf16 fine-tunes here) plus the method's working set for the model count (not the contributor count)
    over it — and the writer's pending shard. The working set is float32 whatever the stored dtype, so
    key ``b``, with more elements, outweighs ``a``, which stores more bytes (fp32)."""
    captured = {}
    monkeypatch.setattr(mm, "preflight_resource_warning", lambda *_, ram_bytes, **__: captured.update(ram=ram_bytes))
    wide_bytes, large = torch.randn(600), torch.randn(64, 16)
    base = _write_tiny_checkpoint(tmp_path / "base", {"a": wide_bytes, "b": large})
    models = [_write_tiny_checkpoint(tmp_path / f"m{i}", {"a": wide_bytes, "b": large.bfloat16()}) for i in range(3)]

    mm.merge_models(
        [str(model) for model in models],
        str(tmp_path / "out"),
        method="ties",
        base_model=str(base),
        max_shard_size="1MB",
        allow_missing_tokenizer=True,
        verbose=False,
    )

    numel = large.numel()
    stored = 3 * numel * 2 + numel * 4
    working = mm._METHODS["ties"].fp32_copies(3) * 4 * numel
    shard = mm.StageShardWriter(str(tmp_path), "probe", "1MB", enabled=False).max_bytes
    assert captured["ram"] == stored + working + shard


def test_the_ram_preflight_counts_the_costliest_key_when_sizes_tie(tmp_path, monkeypatch):
    """Keys of equal element count can store different bytes: here only ``b``'s base copy is fp32. The
    estimate is the largest per-key total, not the first key with the most elements."""
    captured = {}
    monkeypatch.setattr(mm, "preflight_resource_warning", lambda *_, ram_bytes, **__: captured.update(ram=ram_bytes))
    tensor = torch.randn(32, 32)
    base = _write_tiny_checkpoint(tmp_path / "base", {"a": tensor.bfloat16(), "b": tensor})
    models = [
        _write_tiny_checkpoint(tmp_path / f"m{i}", {"a": tensor.bfloat16(), "b": tensor.bfloat16()}) for i in range(2)
    ]

    mm.merge_models(
        [str(model) for model in models],
        str(tmp_path / "out"),
        method="task_arithmetic",
        base_model=str(base),
        max_shard_size="1MB",
        allow_missing_tokenizer=True,
        verbose=False,
    )

    numel = tensor.numel()
    stored = 2 * numel * 2 + numel * 4
    working = mm._METHODS["task_arithmetic"].fp32_copies(2) * 4 * numel
    shard = mm.StageShardWriter(str(tmp_path), "probe", "1MB", enabled=False).max_bytes
    assert captured["ram"] == stored + working + shard


def test_the_merge_loop_releases_each_keys_inputs_before_reading_the_next(tmp_path, monkeypatch):
    """The RAM preflight counts one key's inputs: a binding that outlives its iteration (``per_key``
    holds every input, and an integer key is passed through on its own branch) keeps the previous
    key's tensors alive beside the next key's reads. The one copy the writer stages is its output."""
    keys = ["i0", *(f"w{i}" for i in range(3))]
    models = [
        _write_tiny_checkpoint(
            tmp_path / name,
            {"i0": torch.arange(4), **{key: torch.full((4,), value) for key in keys[1:]}},
        )
        for name, value in (("a", 0.0), ("b", 2.0))
    ]
    reads: list[tuple[str, weakref.ref]] = []
    staged: list[weakref.ref] = []
    held_over: list[tuple[str, str]] = []
    read_tensor = mm._TensorReader.get
    stage_tensor = mm.StageShardWriter.add

    def tracked_get(self, key):
        held_over.extend(
            (key, prior)
            for prior, ref in reads
            if prior != key and ref() is not None and not any(ref() is out() for out in staged)
        )
        tensor = read_tensor(self, key)
        reads.append((key, weakref.ref(tensor)))
        return tensor

    def tracked_add(self, key, tensor):
        staged.append(weakref.ref(tensor))
        return stage_tensor(self, key, tensor)

    monkeypatch.setattr(mm._TensorReader, "get", tracked_get)
    monkeypatch.setattr(mm.StageShardWriter, "add", tracked_add)
    mm.merge_models(
        [str(model) for model in models],
        str(tmp_path / "out"),
        method="linear",
        dtype="float32",
        allow_missing_tokenizer=True,
        verbose=False,
    )
    assert [key for key, _ref in reads] == [key for key in keys for _model in models], "premise: every key read"
    assert not held_over, f"(key read, earlier key still alive): {held_over}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
