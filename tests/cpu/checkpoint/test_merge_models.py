#!/usr/bin/env python
"""Tests for scripts/after_training/merge_models.py.

Two layers: (1) the per-tensor merge math (linear / slerp / task_arithmetic / ties) on known values,
and (2) an end-to-end merge of two tiny **Qwen3.5 MoE** checkpoints (the user-requested target)
through the streaming pipeline — write → reload with the real model class → verify the merged weights
match the expected interpolation and the model forwards.

Run: ``python tests/cpu/checkpoint/test_merge_models.py`` (or ``pytest -m cpu``).
"""

from __future__ import annotations

import re
import tempfile
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file
from transformers import CONFIG_MAPPING

from tests.common.utils import load_script_module

mm = load_script_module("scripts/after_training/merge_models.py")


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
    but absent from model[0] is included — not silently dropped as it was when iterating model[0].
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
    from transformers.models.qwen3_5_moe import Qwen3_5MoeForCausalLM, Qwen3_5MoeTextConfig

    torch.manual_seed(seed)
    config = Qwen3_5MoeTextConfig(**_TINY_QWEN35, layer_types=["full_attention"] * _TINY_QWEN35["num_hidden_layers"])
    Qwen3_5MoeForCausalLM(config).to(torch.bfloat16).save_pretrained(out_dir, safe_serialization=True)


def test_end_to_end_linear_merge_qwen3_5():
    """Merge two tiny Qwen3.5 checkpoints (linear 0.5/0.5), reload, verify average + a forward pass.
    Also pins that resume sidecars planted beside the tokenizer source do NOT ship: they describe
    one input run's state, and the merged artifact has no such run."""
    from transformers.models.qwen3_5_moe import Qwen3_5MoeForCausalLM

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


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
