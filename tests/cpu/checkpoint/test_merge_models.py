#!/usr/bin/env python
"""scripts/after_training/merge_models.py: input gates, merge math, key-set gate, RAM preflight, end-to-end merge."""

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
from transformers import CONFIG_MAPPING, Gemma3Config, Qwen3Config, Qwen3ForCausalLM
from transformers.models.qwen3_5_moe import Qwen3_5MoeForCausalLM, Qwen3_5MoeTextConfig
from transformers.trainer import SCHEDULER_NAME, TRAINER_STATE_NAME

from halo.checkpoint.format import (
    ADAPTER_CONFIG_FILE,
    ADAPTER_SAFETENSORS_FILE,
    REFERENCE_LOGPS_FILE,
    RESUME_ADAPTER_DIR,
    RESUME_ADAPTER_MARKER_FILE,
    ROUTER_BALANCING_BIASES_FILE,
    prefetch_pending_filename,
)
from halo.checkpoint.reading import iter_checkpoint_shard_entries
from halo.checkpoint.tools import model_merge as mm
from halo.checkpoint.writing import resolve_checkpoint_write_threads
from tests.common.checkpoint_io import mark_resume_adapter
from tests.common.models import TINY_QWEN3_CONFIG

_WORKING_SET_NUMEL = 1 << 20
# The ops also allocate a few 0-d results (norms, the slerp dot).
_SCALAR_SLACK_BYTES = 4096


def _write_tiny_checkpoint(path: Path, tensors: dict[str, torch.Tensor], config=None) -> Path:
    """The aux copy round-trips the config through AutoConfig, so it has to be a real one."""
    path.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(path / "model.safetensors"))
    (config or CONFIG_MAPPING["qwen3"]()).save_pretrained(str(path))
    return path


def test_parse_model_spec():
    # None when no weight was given: the method↔knob gate must see explicitness.
    assert mm._parse_model_spec("/a/b") == ("/a/b", None)
    assert mm._parse_model_spec("/a/b:0.25") == ("/a/b", 0.25)
    assert mm._parse_model_spec("org/model") == ("org/model", None)


def test_a_knob_the_method_ignores_is_refused():
    """Ungated, `--models a:0.1 b:0.9` under slerp is silently ignored. The model paths deliberately do not exist,
    so the gate raises before any I/O."""
    with pytest.raises(ValueError, match="does not use models:weight"):
        mm.merge_models(["/nope/a:0.1", "/nope/b:0.9"], "/nope/out", method="slerp")
    with pytest.raises(ValueError, match="does not use t"):
        mm.merge_models(["/nope/a", "/nope/b"], "/nope/out", method="linear", knobs={"t": 0.5})
    with pytest.raises(ValueError, match="does not use density, lambda"):
        mm.merge_models(["/nope/a", "/nope/b"], "/nope/out", method="linear", knobs={"density": 0.6, "lambda": 1.0})
    with pytest.raises(ValueError, match="does not use base_model"):
        mm.merge_models(["/nope/a", "/nope/b"], "/nope/out", method="slerp", base_model="/nope/base")


# Spelled out independently of the table the code reads, so a table change has to be deliberate.
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
    if knob in _METHOD_KNOBS[method]:
        mm._check_method_knobs(method, {knob})
    else:
        with pytest.raises(ValueError, match=re.escape(f"--method {method} does not use {knob}")):
            mm._check_method_knobs(method, {knob})


def test_the_method_table_declares_every_method_and_default():
    assert set(mm.MERGE_METHODS) == set(_METHOD_KNOBS)
    assert mm.KNOB_DEFAULTS == {"t": 0.5, "density": 0.6, "lambda": 1.0}
    for method, knobs in _METHOD_KNOBS.items():
        assert mm._method_knobs(method) == frozenset(knobs)
        mm._check_method_knobs(method, set())


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
    """None raise inside the ops: each merges something other than what was asked. The paths do not exist, so the
    refusal precedes any read."""
    base = "/nope/base" if method == "ties" else None
    with pytest.raises(ValueError, match=re.escape(f"--{knob} must be")):
        mm.merge_models(["/nope/a", "/nope/b"], "/nope/out", method=method, base_model=base, knobs={knob: value})


@pytest.mark.parametrize("weight", ["nan", "inf", "-inf"])
def test_a_non_finite_model_weight_is_refused_before_any_io(weight):
    """``path:nan`` parses as a weight and would write non-finite tensors."""
    with pytest.raises(ValueError, match="--models weights must be finite"):
        mm.merge_models([f"/nope/a:{weight}", "/nope/b"], "/nope/out", method="linear")


def test_the_domain_edges_and_defaults_are_accepted():
    """Over-refusing is as wrong as under-refusing."""
    mm._check_knob_domains("slerp", {"t": 0.0})
    mm._check_knob_domains("slerp", {"t": 1.0})
    mm._check_knob_domains("ties", {"density": 1.0, "lambda": 0.0})
    mm._check_knob_domains("ties", {"density": 1e-6, "lambda": -1.0})
    for method in mm.MERGE_METHODS:
        mm._check_knob_domains(method, mm.KNOB_DEFAULTS)
    assert set(mm.KNOB_DOMAINS) == set(mm.KNOB_DEFAULTS), "every knob needs a domain, and nothing else"


def test_an_unknown_knob_is_refused():
    """Silently dropping a typo would merge with the method's default."""
    with pytest.raises(ValueError, match=r"unknown merge knob\(s\) \['lam'\]"):
        mm.merge_models(["/nope/a", "/nope/b"], "/nope/out", method="ties", knobs={"lam": 1.0})


def test_dispatch_passes_each_op_the_knobs_its_signature_names():
    """A ties merge at ``lambda=0`` is exactly the base model, and at ``lambda=1`` the task vector on top."""
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
    out = mm._merge_linear([a, b], [1.0, 1.0])
    assert torch.allclose(out, torch.full((4,), 0.5))


def test_slerp_endpoints_and_midpoint():
    a = torch.tensor([1.0, 0.0, 0.0, 0.0])
    b = torch.tensor([0.0, 1.0, 0.0, 0.0])
    assert torch.allclose(mm._merge_slerp(a, b, 0.0), a, atol=1e-5)
    assert torch.allclose(mm._merge_slerp(a, b, 1.0), b, atol=1e-5)
    mid = mm._merge_slerp(a, b, 0.5)
    assert torch.allclose(mid, torch.tensor([0.70710677, 0.70710677, 0.0, 0.0]), atol=1e-5)
    assert torch.allclose(mid.norm(), torch.tensor(1.0), atol=1e-5)


def test_slerp_colinear_falls_back_to_lerp():
    a = torch.tensor([1.0, 2.0, 3.0])
    b = 2.0 * a
    out = mm._merge_slerp(a, b, 0.5)
    assert torch.allclose(out, torch.lerp(a, b, 0.5), atol=1e-5)


def test_slerp_of_neutralized_sinks_stays_finite():
    """Sinks at ``bfloat16.min`` square past fp32's range, so a raw norm reads ``inf`` and the arc sums to ``-inf``;
    identical inputs must slerp to themselves."""
    sinks = torch.full((64,), torch.finfo(torch.bfloat16).min, dtype=torch.bfloat16)
    out = mm._merge_slerp(sinks, sinks.clone(), 0.5)
    assert torch.isfinite(out).all(), "slerp of finite inputs produced a non-finite tensor"
    assert torch.equal(out, sinks.float())
    # The magnitude normalization must not leak into the result.
    a = torch.tensor([3.0, 0.0, 0.0, 0.0])
    b = torch.tensor([0.0, 3.0, 0.0, 0.0])
    mid = mm._merge_slerp(a, b, 0.5)
    assert torch.allclose(mid, torch.tensor([3 * 0.70710677, 3 * 0.70710677, 0.0, 0.0]), atol=1e-5)


def test_task_arithmetic_adds_task_vectors():
    base = torch.zeros(3)
    m1 = torch.tensor([1.0, 0.0, 0.0])
    m2 = torch.tensor([0.0, 2.0, 0.0])
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
    delta = torch.tensor([0.05, 0.1, 5.0, 10.0])
    out = mm._merge_ties(base, [delta], [1.0], density=0.5, lambda_=1.0)
    assert torch.allclose(out, torch.tensor([0.0, 0.0, 5.0, 10.0]), atol=1e-6)


def test_reference_keys_uses_base_for_task_methods():
    """Task vectors are base-relative, so iterating model[0] would silently drop a base-only key."""

    class _StubReader:
        def __init__(self, ks):
            self._ks = set(ks)

        def keys(self):
            return self._ks

    m0 = _StubReader({"x"})
    m1 = _StubReader({"x", "y"})
    base = _StubReader({"x", "y"})
    assert mm._reference_keys([m0, m1], base) == ["x", "y"]
    assert mm._reference_keys([m0, m1], None) == ["x"]


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
    """Resume state beside the tokenizer source must not ship: it describes one input run, and a resume adapter's
    marker would resume from one input."""
    with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
        a, b, out = Path(tmp) / "a", Path(tmp) / "b", Path(tmp) / "merged"
        _build_tiny_qwen35(a, seed=0)
        _build_tiny_qwen35(b, seed=1)
        sidecars = (
            TRAINER_STATE_NAME,
            SCHEDULER_NAME,
            "rng_state_0.pth",
            ROUTER_BALANCING_BIASES_FILE,
            REFERENCE_LOGPS_FILE,
            prefetch_pending_filename(0, 1),
        )
        for sidecar in sidecars:
            (a / sidecar).write_bytes(b"x")
        (a / RESUME_ADAPTER_DIR).mkdir()
        (a / RESUME_ADAPTER_DIR / ADAPTER_CONFIG_FILE).write_text("{}")
        save_file({"lora_A": torch.ones(2, 2)}, str(a / RESUME_ADAPTER_DIR / ADAPTER_SAFETENSORS_FILE))
        mark_resume_adapter(str(a))

        mm.merge_models(
            model_specs=[str(a), str(b)],
            output_dir=str(out),
            method="linear",
            dtype="bfloat16",
            tokenizer_source=str(a),
            allow_missing_tokenizer=True,
            verbose=False,
        )

        for entry in (*sidecars, RESUME_ADAPTER_MARKER_FILE, RESUME_ADAPTER_DIR):
            assert not (out / entry).exists(), f"{entry} is one input run's resume state, not the merge's"

        # Proves config, shards and index are valid.
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

        # A weight-space average is not a logit-space average, so parity is checked against an independently averaged
        # reference.
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
        # bf16 round-trip through the merge pipeline vs a direct average.
        assert torch.allclose(merged_logits, reference_logits, atol=1e-2), (
            "reloaded merged model's forward diverges from an independently-averaged reference"
        )


def test_balancing_biases_keep_their_trained_dtype():
    """ALF biases are trained fp32 and near-tied top-k picks flip under a bf16 round-trip."""
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
    """Refused up front with no output directory left behind."""
    with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
        base, a, b, out = Path(tmp) / "base", Path(tmp) / "a", Path(tmp) / "b", Path(tmp) / "merged"
        shared = {"model.embed_tokens.weight": torch.zeros(4, 4)}
        _write_tiny_checkpoint(base, {**shared, "lm_head.weight": torch.zeros(4, 4)})
        for path in (a, b):
            _write_tiny_checkpoint(path, dict(shared))
        assert not CONFIG_MAPPING["qwen3"]().tie_word_embeddings, "premise: the shipped config does not tie"

        with pytest.raises(ValueError, match=rf"{re.escape(str(a))} does not carry .* lacks 1 tensor\(s\) \['lm_head"):
            mm.merge_models(
                [str(a), str(b)],
                str(out),
                method="task_arithmetic",
                base_model=str(base),
                allow_missing_tokenizer=True,
                verbose=False,
            )
        assert not out.exists()


_BASE_METHODS = sorted(name for name, spec in mm.MERGE_METHODS.items() if "base" in spec.tensor_args)


def _tied_qwen3(path: Path, seed: int, *, store_head: bool, tie: bool = True) -> dict[str, torch.Tensor]:
    """A tied head is left out unless ``store_head`` writes it back, as some Hub checkpoints ship it."""
    torch.manual_seed(seed)
    model = Qwen3ForCausalLM(Qwen3Config(**{**TINY_QWEN3_CONFIG, "tie_word_embeddings": tie}))
    model.save_pretrained(path)
    tensors = load_file(str(path / "model.safetensors"))
    if store_head:
        tensors["lm_head.weight"] = tensors["model.embed_tokens.weight"].clone()
        save_file(tensors, str(path / "model.safetensors"), metadata={"format": "pt"})
    return tensors


@pytest.mark.parametrize("method", _BASE_METHODS)
def test_a_tied_hub_base_storing_its_head_merges_into_a_tied_checkpoint(method, tmp_path):
    """Under the tying config that head is the merged embedding, so the merge leaves it out and the reload ties it."""
    base, a, b, out = (tmp_path / name for name in ("base", "a", "b", "merged"))
    base_tensors = _tied_qwen3(base, seed=0, store_head=True)
    tuned = [_tied_qwen3(path, seed, store_head=False) for path, seed in ((a, 1), (b, 2))]
    assert "lm_head.weight" in base_tensors and all("lm_head.weight" not in t for t in tuned), "premise"

    mm.merge_models(
        [str(a), str(b)],
        str(out),
        method=method,
        base_model=str(base),
        dtype="float32",
        allow_missing_tokenizer=True,
        verbose=False,
    )

    merged_tensors = load_file(str(out / "model.safetensors"))
    assert "lm_head.weight" not in merged_tensors
    spec = mm.MERGE_METHODS[method]
    embed = "model.embed_tokens.weight"
    per_key = {"base": base_tensors[embed], "tensors": [t[embed] for t in tuned], "weights": [1.0, 1.0]}
    knobs = {mm.knob_dest(knob): default for knob, default in spec.knobs.items()}
    expected = spec.op(**{name: per_key[name] for name in spec.tensor_args}, **knobs)
    torch.testing.assert_close(merged_tensors[embed], expected)

    merged, info = Qwen3ForCausalLM.from_pretrained(out, output_loading_info=True)
    assert not info["missing_keys"] and not info["unexpected_keys"], info
    assert merged.lm_head.weight is merged.model.embed_tokens.weight
    torch.testing.assert_close(merged.lm_head.weight.detach(), expected)


@pytest.mark.parametrize(
    ("tie", "lacking_embedding"),
    [(False, ()), (True, ("b",)), (True, ("a", "b"))],
    ids=["untied-config", "a-fine-tune-lacks-the-embedding", "no-fine-tune-stores-either-half"],
)
def test_a_stored_base_head_is_missing_unless_the_tie_rebuilds_it(tie, lacking_embedding, tmp_path):
    """Without a tying config the head loads random-initialized, and without the embedding in every fine-tune the
    tie has nothing to rebuild from."""
    base, a, b, out = (tmp_path / name for name in ("base", "a", "b", "merged"))
    _tied_qwen3(base, seed=0, store_head=True, tie=tie)
    for path, seed in ((a, 1), (b, 2)):
        tensors = _tied_qwen3(path, seed, store_head=False)
        if path.name in lacking_embedding:
            tensors.pop("model.embed_tokens.weight")
        tensors.pop("lm_head.weight", None)
        save_file(tensors, str(path / "model.safetensors"), metadata={"format": "pt"})

    with pytest.raises(ValueError, match=r"does not carry the merge's reference key set"):
        mm.merge_models(
            [str(a), str(b)],
            str(out),
            method="ties",
            base_model=str(base),
            allow_missing_tokenizer=True,
            verbose=False,
        )
    assert not out.exists()


@pytest.mark.parametrize(
    ("wrapper_ties", "text_ties"),
    [(True, True), (False, True), (True, False)],
    ids=["both", "text-only", "wrapper-only"],
)
def test_a_stored_base_head_is_excused_only_where_both_config_levels_tie(wrapper_ties, text_ties, tmp_path):
    """transformers ties a wrapper's head on the wrapper's flag and a text-only class's on the text sub-config's,
    and Gemma 3's config lets the two disagree."""
    config = Gemma3Config(text_config={"tie_word_embeddings": text_ties}, tie_word_embeddings=wrapper_ties)
    assert (config.tie_word_embeddings, config.get_text_config().tie_word_embeddings) == (wrapper_ties, text_ties)
    embedding = {"model.language_model.embed_tokens.weight": torch.randn(8, 4)}
    base = _write_tiny_checkpoint(tmp_path / "base", {**embedding, "lm_head.weight": torch.randn(8, 4)}, config)
    tuned = [_write_tiny_checkpoint(tmp_path / name, {**embedding}, config) for name in ("a", "b")]
    out = tmp_path / "merged"

    def merge():
        mm.merge_models(
            [str(path) for path in tuned],
            str(out),
            method="task_arithmetic",
            base_model=str(base),
            allow_missing_tokenizer=True,
            verbose=False,
        )

    if wrapper_ties and text_ties:
        merge()
        assert set(load_file(str(out / "model.safetensors"))) == set(embedding)
        return
    with pytest.raises(ValueError, match=r"lacks 1 tensor\(s\) \['lm_head.weight'\]"):
        merge()
    assert not out.exists()


# A key each family's Hub base stores and its class drops at load (``_keys_to_ignore_on_load_unexpected``), so no
# fine-tune carries it.
_LOAD_DROPPED_BASE_KEYS = {
    "qwen3_5_moe": "mtp.layers.0.mlp.experts.gate_up_proj",
    "qwen3_5_moe_text": "mtp.fc.weight",
    "glm4_moe_lite": "model.layers.47.eh_proj.weight",
}


@pytest.mark.parametrize("model_type", sorted(_LOAD_DROPPED_BASE_KEYS))
@pytest.mark.parametrize("method", _BASE_METHODS)
def test_base_keys_the_shipped_class_drops_at_load_are_left_out(method, model_type, tmp_path):
    """A base-only key the class does not drop stays a refusal."""
    config = CONFIG_MAPPING[model_type]()
    dropped = _LOAD_DROPPED_BASE_KEYS[model_type]
    shared = {"model.embed_tokens.weight": torch.zeros(8, 4), "model.layers.0.mlp.down_proj.weight": torch.zeros(4, 4)}
    tuned = [
        _write_tiny_checkpoint(tmp_path / name, {key: value + seed for key, value in shared.items()}, config)
        for seed, name in enumerate(("a", "b"), start=1)
    ]

    def merge(base_tensors, out):
        base = _write_tiny_checkpoint(out.with_name(f"{out.name}-base"), base_tensors, config)
        mm.merge_models(
            [str(path) for path in tuned],
            str(out),
            method=method,
            base_model=str(base),
            dtype="float32",
            allow_missing_tokenizer=True,
            verbose=False,
        )

    merge({**shared, dropped: torch.ones(2, 2)}, tmp_path / "merged")
    assert set(load_file(str(tmp_path / "merged" / "model.safetensors"))) == set(shared)

    stray = tmp_path / "stray"
    with pytest.raises(ValueError, match=r"lacks 1 tensor\(s\) \['model.stray.weight'\]"):
        merge({**shared, dropped: torch.ones(2, 2), "model.stray.weight": torch.ones(2)}, stray)
    assert not stray.exists()


def test_a_qwen3_5_base_storing_its_mtp_head_merges_into_a_checkpoint_that_loads_whole(tmp_path):
    base, a, b, out = (tmp_path / name for name in ("base", "a", "b", "merged"))
    for path, seed in ((base, 0), (a, 1), (b, 2)):
        _build_tiny_qwen35(path, seed=seed)
    base_tensors = load_file(str(base / "model.safetensors"))
    mtp = {"mtp.fc.weight": torch.randn(64, 128), "mtp.norm.weight": torch.ones(64)}
    save_file({**base_tensors, **mtp}, str(base / "model.safetensors"), metadata={"format": "pt"})

    mm.merge_models(
        [str(a), str(b)], str(out), method="ties", base_model=str(base), allow_missing_tokenizer=True, verbose=False
    )

    merged_keys = {key for _shard, _reader, key in iter_checkpoint_shard_entries(str(out))}
    assert merged_keys == set(base_tensors), "the merge dropped a weight or carried the base's MTP head"
    _, info = Qwen3_5MoeForCausalLM.from_pretrained(out, output_loading_info=True)
    assert not info["missing_keys"] and not info["unexpected_keys"], info


@pytest.mark.parametrize("mismatched", ["fine-tune", "base"])
def test_a_shape_mismatch_is_refused_before_any_output(mismatched, tmp_path):
    """The base is held to it too: its task vector would otherwise broadcast into a tensor of neither shape."""
    good, odd = {"w": torch.zeros(4, 4)}, {"w": torch.zeros(4)}
    base = _write_tiny_checkpoint(tmp_path / "base", odd if mismatched == "base" else good)
    a = _write_tiny_checkpoint(tmp_path / "a", good)
    b = _write_tiny_checkpoint(tmp_path / "b", odd if mismatched == "fine-tune" else good)
    out = tmp_path / "merged"

    with pytest.raises(ValueError, match=r"shape mismatch for 'w'"):
        mm.merge_models(
            [str(a), str(b)],
            str(out),
            method="task_arithmetic",
            base_model=str(base),
            allow_missing_tokenizer=True,
            verbose=False,
        )
    assert not out.exists()


def _peak_allocated_bytes(fn: Callable[[], object], trace_path: Path) -> int:
    """Read from the profiler's ``[memory]`` events: exact and independent of threads and host load, whereas RSS
    undercounts once faults spread across CPUs."""
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
    """A bound below the peak lets a merge OOM without the warning; above it warns on merges that fit, so bf16, the
    worst case, lands within 10%."""
    spec = mm.MERGE_METHODS[method]
    tensors = [torch.randn(_WORKING_SET_NUMEL).to(dtype) for _ in range(n_models)]
    per_key = {
        "tensors": tensors,
        "t0": tensors[0],
        "t1": tensors[-1],
        "weights": [1.0] * n_models,
        "base": torch.randn(_WORKING_SET_NUMEL).to(dtype),
    }
    knobs = {mm.knob_dest(knob): value for knob, value in spec.knobs.items()}

    op_args = {name: per_key[name] for name in spec.tensor_args}
    peak = _peak_allocated_bytes(lambda: spec.op(**op_args, **knobs), tmp_path / "trace.json")

    copy_bytes = torch.float32.itemsize * _WORKING_SET_NUMEL
    bound = spec.fp32_copies(n_models) * copy_bytes
    assert peak > copy_bytes, f"measured {peak} bytes, under one fp32 copy — the probe measured nothing"
    assert peak <= bound + _SCALAR_SLACK_BYTES, f"peak {peak / copy_bytes:.3f} fp32 copies exceeds the bound"
    if dtype == torch.bfloat16:
        assert bound <= 1.1 * peak, f"bound {bound / copy_bytes} is far above the peak {peak / copy_bytes:.3f}"


def test_the_ram_preflight_sizes_inputs_as_stored_plus_the_method_working_set(tmp_path, monkeypatch):
    """The working set is float32 whatever the stored dtype, so key ``b``, with more elements, outweighs ``a``,
    which stores more bytes."""
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
    working = mm.MERGE_METHODS["ties"].fp32_copies(3) * 4 * numel
    shard = (1 + resolve_checkpoint_write_threads()) * mm.StreamingShardWriter(
        str(tmp_path), "1MB", enabled=False
    ).max_bytes
    assert captured["ram"] == stored + working + shard


def test_the_ram_preflight_counts_the_costliest_key_when_sizes_tie(tmp_path, monkeypatch):
    """The estimate is the largest per-key total, not the first key with the most elements."""
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
    working = mm.MERGE_METHODS["task_arithmetic"].fp32_copies(2) * 4 * numel
    shard = (1 + resolve_checkpoint_write_threads()) * mm.StreamingShardWriter(
        str(tmp_path), "1MB", enabled=False
    ).max_bytes
    assert captured["ram"] == stored + working + shard


def test_the_merge_loop_releases_each_keys_inputs_before_reading_the_next(tmp_path, monkeypatch):
    """A binding that outlives its iteration keeps the previous key's tensors alive beside the next key's reads."""
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
    stage_tensor = mm.StreamingShardWriter.add

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
    monkeypatch.setattr(mm.StreamingShardWriter, "add", tracked_add)
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
