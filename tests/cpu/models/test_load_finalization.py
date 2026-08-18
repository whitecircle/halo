#!/usr/bin/env python
"""``finalize_loaded_model`` contract + load-path coverage sweep — CPU-only.

transformers 5 builds every model on meta and re-materializes each non-persistent buffer as
``torch.empty_like`` (uninitialized); the only upstream repair is ``_init_weights``' RotaryEmbedding
branch, which a remote-code family overriding ``_init_weights`` without ``super()`` (Bailing/Ling V3)
never runs — a zero/garbage ``inv_freq`` degenerates RoPE to NoPE and trains with a plausible loss.
``finalize_loaded_model`` (src/models/patches/buffer_fixes.py) is the single post-load seam:
buffer recompute + re-tie, run by EVERY load path once weights sit on their device.

These tests pin (1) the seam's three legs — a seam that drops one silently breaks every path that
relies on it; (2) that every load path reaches the seam, swept from the dispatcher's own
``_load_*`` roots so a NEW loader that forgets it fails here rather than in a silently-NoPE run; and
(3) the trainer backstop: a buffer still meta at device placement is uninitialized memory and must
raise, not be allocated as ``torch.empty``; and (4) the ``scripts/`` tool surface, where a loader
that hands its model to a forward must finalize it and the save-only ones are pinned as such.

Run: python tests/cpu/models/test_load_finalization.py  (or pytest)
"""

import ast
import functools
import pathlib
import sys
import types

import pytest
import torch
import torch.nn as nn
from transformers import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding

from src.models.patches.buffer_fixes import finalize_loaded_model
from src.trainers.mixins.base import DistributedTrainerMixin

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]

SEAM = "finalize_loaded_model"

DISPATCHER = "src/distributed/loading/model_loading.py"

# The dispatcher's own load surface. Discovered by the sweep below; pinned here as an exact set so a
# renamed, removed or added ``_load_*`` is a deliberate edit rather than a silently shrunk sweep.
DISPATCHED_LOADERS = frozenset(
    {
        "_load_pp_stage_model",
        "_load_ep_tp_model",
        "_load_tp_model",
        "_load_tp_moe_model",
        "_load_ep_cp_model",
        "_load_ep_model",
        "_load_cp_model",
        "_load_undistributed_model",
    }
)

# The functions that materialize weights themselves (everything else delegates to one of these).
# Each must call the seam DIRECTLY: the reachability sweep is branch-insensitive, so a terminal that
# lost its call could still "reach" the seam through a sibling branch. Closed-world by assumption —
# a new terminal loader belongs here, and the reachability sweep is what catches one that is missing.
TERMINAL_LOADERS = (
    (DISPATCHER, "_sequential_load_to_cuda"),
    (DISPATCHER, "_from_pretrained_on_local_gpu"),
    (DISPATCHER, "_load_tp_model"),
    ("src/distributed/loading/frozen_models.py", "load_frozen_auxiliary_model"),
    ("src/distributed/loading/model_loading.py", "load_model_from_pretrained"),
    ("src/distributed/expert_parallel/loading.py", "_load_ep_model_huggingface"),
    ("src/distributed/expert_parallel/lazy_loader.py", "load_ep_model_lazy"),
    ("src/distributed/pipeline_parallel/lazy_loader.py", "load_pp_stage_model"),
    ("src/distributed/context_parallel/loading.py", "load_model_for_cp"),
)

# The two eager load entry points a ``scripts/`` tool materializes a model through.
TOOL_LOAD_CALLS = frozenset({"from_pretrained_verified", "auto_load_model"})

# Tool loaders whose model is handed to a FORWARD — reward scores, dedup embeddings, the
# ``--check_inference`` generation. An uninitialized buffer here becomes numbers the operator acts
# on, with no other symptom, so each must call the seam directly.
INFERENCE_TOOL_LOADERS = (
    ("scripts/inference/reward_model/_common.py", "load_reward_model"),
    ("scripts/inference/generation/dataset_deduplication.py", "compute_embeddings"),
    ("scripts/after_training/convert_to_bf16.py", "_load_verified"),
)

# Load → cast/patch/merge → save. No forward exists in these files, and ``state_dict`` omits
# non-persistent buffers, so an unrepaired one reaches neither a number nor the written checkpoint.
CONVERSION_TOOL_LOADERS = (
    ("scripts/after_training/merge_peft_adapters.py", "_load_base_model"),
    ("scripts/after_training/reset_sinks.py", "_reset_sinks_from_pretrained"),
    ("scripts/before_training/patch_vocab.py", "main"),
    ("scripts/before_training/convert_deepseek_v4_bf16.py", "main"),
)

# How a file betrays that it runs the model it loaded, for the save-only half of the split above.
_FORWARD_MARKERS = (".generate(", ".logits", "last_hidden_state")


def _rotary_with_reference(theta: float = 1_000_000.0) -> tuple[Qwen3RotaryEmbedding, torch.Tensor]:
    cfg = Qwen3Config(rope_theta=theta, hidden_size=64, num_attention_heads=4, head_dim=16)
    rotary = Qwen3RotaryEmbedding(cfg)
    return rotary, rotary.inv_freq.clone()


class _SlopeAttention(nn.Module):
    """The shape ``fix_non_persistent_buffers`` keys on (Bailing Lightning-Attention slopes)."""

    def __init__(self):
        super().__init__()
        self.config = types.SimpleNamespace(num_hidden_layers=4)
        self.num_heads = 4
        self.layer_idx = 1
        self.register_buffer("slope", torch.empty(4, device="meta"), persistent=False)


class _LoadedShell(nn.Module):
    """A loaded model as the seam sees one: rotary + slope buffers, and a tie to re-run."""

    def __init__(self):
        super().__init__()
        self.rotary_emb, self.reference_inv_freq = _rotary_with_reference()
        self.attn = _SlopeAttention()
        self.tied = 0

    def tie_weights(self):
        self.tied += 1


def test_seam_recomputes_a_meta_inv_freq():
    """The transformers-5 shape: the buffer never materialized, so values cannot be 'repaired' —
    they must be recomputed from the config, on a real device, in fp32."""
    shell = _LoadedShell()
    shell.rotary_emb.register_buffer(
        "inv_freq", torch.empty_like(shell.reference_inv_freq, device="meta"), persistent=False
    )

    finalize_loaded_model(shell)

    # .cpu(): a meta buffer is recomputed onto current_device(), which is cuda on a GPU host.
    inv_freq = shell.rotary_emb.inv_freq.cpu()
    assert shell.rotary_emb.inv_freq.device.type != "meta"
    assert inv_freq.dtype == torch.float32
    assert torch.allclose(inv_freq, shell.reference_inv_freq), (
        f"inv_freq not recomputed from the config: got {inv_freq[1].item():.6g}, "
        f"expected {shell.reference_inv_freq[1].item():.6g} (theta=1e6)"
    )


def test_seam_materializes_non_rotary_buffer_families():
    """Dropping the ``fix_non_persistent_buffers`` leg would strand Bailing slopes / Gemma4
    embed_scale on meta for every path that relies on the seam alone."""
    shell = _LoadedShell()

    finalize_loaded_model(shell)

    slope = shell.attn.slope
    assert slope.device.type != "meta"
    assert slope.shape == (4,)
    assert (slope < 0).all(), "ALiBi slopes are strictly negative by construction"


def test_seam_reties_shared_weights():
    """The lazy/low-cpu-mem loads leave the shadow tied lm_head on meta until ``tie_weights``; the
    EP/PP loaders delegated that step to the seam, so a seam without it crashes their first backward."""
    shell = _LoadedShell()

    finalize_loaded_model(shell)

    assert shell.tied == 1


def _graph_modules() -> list[str]:
    """Every module the sweep must see, DERIVED: the loading package plus every ``src`` module that
    names the seam (the EP/PP/CP loaders live in their own packages).

    A new loader module that finalizes joins the graph on its own. One that does NOT is absent from
    it, so the dispatcher path delegating to it stops reaching the seam and the sweep below fails —
    the failure direction this test exists for.
    """
    package = (REPO_ROOT / "src/models/loading").rglob("*.py")
    callers = (path for path in (REPO_ROOT / "src").rglob("*.py") if SEAM in path.read_text(encoding="utf-8"))
    return sorted({path.relative_to(REPO_ROOT).as_posix() for path in (*package, *callers)})


@functools.cache
def _call_graph() -> dict[tuple[str, str], set[str]]:
    """(module, name) -> names it calls, over every top-level function of the graph's modules.

    Keyed by module too: two modules defining the same function name are different functions, and
    merging their callees lets one module's finalizing twin vouch for the other's forgetful one.
    """
    graph: dict[tuple[str, str], set[str]] = {}
    for rel in _graph_modules():
        tree = ast.parse((REPO_ROOT / rel).read_text(encoding="utf-8"))
        for node in tree.body:
            if not isinstance(node, ast.FunctionDef):
                continue
            graph[(rel, node.name)] = {
                sub.func.id if isinstance(sub.func, ast.Name) else sub.func.attr
                for sub in ast.walk(node)
                if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name | ast.Attribute)
            }
    return graph


def _reaches_seam(root: tuple[str, str], graph: dict[tuple[str, str], set[str]]) -> bool:
    """Whether ``root`` calls the seam, directly or through a callee. A callee name resolves inside
    its own module first, then to any module defining it — the way an import does."""
    seen: set[tuple[str, str]] = set()
    frontier = [root]
    while frontier:
        node = frontier.pop()
        if node in seen:
            continue
        seen.add(node)
        callees = graph.get(node, set())
        if SEAM in callees:
            return True
        module, _ = node
        for callee in callees:
            local = (module, callee)
            frontier.extend([local] if local in graph else [key for key in graph if key[1] == callee])
    return False


def test_every_dispatched_load_path_reaches_the_seam():
    """Roots are DISCOVERED (every ``_load_*`` in model_loading.py — the dispatcher's own naming),
    not enumerated: a missing repair is precisely the omission a hand-list ships.
    A new per-mode loader that never finalizes fails here."""
    graph = _call_graph()
    roots = {name for (rel, name) in graph if rel == DISPATCHER and name.startswith("_load_")}
    assert roots == DISPATCHED_LOADERS, (
        f"the dispatcher's load surface changed: {sorted(roots ^ DISPATCHED_LOADERS)}. Update "
        f"DISPATCHED_LOADERS deliberately — a shrinking sweep proves nothing."
    )

    unfinalized = sorted(name for name in roots if not _reaches_seam((DISPATCHER, name), graph))
    assert not unfinalized, (
        f"{unfinalized} never reach {SEAM}: models on these load paths keep transformers-5's "
        f"uninitialized non-persistent buffers (zero inv_freq = NoPE with a plausible loss). Call "
        f"{SEAM} once the weights sit on their final device."
    )


@pytest.mark.parametrize("rel,function", TERMINAL_LOADERS, ids=[fn for _, fn in TERMINAL_LOADERS])
def test_every_terminal_loader_calls_the_seam_directly(rel, function):
    graph = _call_graph()
    assert (rel, function) in graph, f"{function} no longer exists in {rel} — update TERMINAL_LOADERS"
    assert SEAM in graph[(rel, function)], (
        f"{rel}:{function} materializes weights but does not call {SEAM} itself. Every terminal "
        f"loader finalizes directly — reachability through a sibling branch is not coverage."
    )


@functools.cache
def _tool_loaders() -> dict[tuple[str, str], set[str]]:
    """(script, function) -> the names it calls, for every ``scripts/`` function whose model comes
    from :data:`TOOL_LOAD_CALLS`.

    Walked over the whole tree rather than a file list, so a new tool is discovered by the load it
    makes. A tool whose model comes from a third-party loader instead (``SentenceTransformer``)
    reaches neither entry point and is outside this sweep.
    """
    loaders: dict[tuple[str, str], set[str]] = {}
    for path in sorted((REPO_ROOT / "scripts").rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                continue
            called = {
                call.func.id
                for call in ast.walk(node)
                if isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
            }
            if called & TOOL_LOAD_CALLS:
                loaders[(path.relative_to(REPO_ROOT).as_posix(), node.name)] = called
    return loaders


def test_tool_load_surface_is_pinned():
    """The split is a judgement about what each tool DOES with the model, so a new tool loader must
    be classified deliberately — discovering one that is in neither set fails here."""
    discovered = set(_tool_loaders())
    pinned = set(INFERENCE_TOOL_LOADERS) | set(CONVERSION_TOOL_LOADERS)
    assert discovered == pinned, (
        f"the scripts/ load surface changed: {sorted(discovered ^ pinned)}. Classify each new loader "
        f"— INFERENCE_TOOL_LOADERS if anything runs the model, CONVERSION_TOOL_LOADERS if it is only "
        f"cast/merged/saved."
    )


@pytest.mark.parametrize("rel,function", INFERENCE_TOOL_LOADERS, ids=[fn for _, fn in INFERENCE_TOOL_LOADERS])
def test_every_inference_tool_loader_finalizes(rel, function):
    """A tool that scores, embeds or generates with a bare load returns garbage from transformers-5's
    uninitialized non-persistent buffers — a dead RoPE on zeroed pages, NaN on reused ones."""
    loaders = _tool_loaders()
    assert (rel, function) in loaders, f"{function} no longer loads a model in {rel} — update the pinned sets"
    assert SEAM in loaders[(rel, function)], (
        f"{rel}:{function} hands a bare-loaded model to a forward without calling {SEAM}: its rotary "
        f"inv_freq and any family decay table are uninitialized memory."
    )


def test_conversion_tools_still_run_no_forward():
    """The save-only half of the split is a claim about the file, so it is checked against the file:
    a conversion tool that starts generating or scoring must be reclassified, not left on a bare load."""
    forwarding = sorted(
        rel
        for rel, _ in CONVERSION_TOOL_LOADERS
        if any(marker in (REPO_ROOT / rel).read_text(encoding="utf-8") for marker in _FORWARD_MARKERS)
    )
    assert not forwarding, (
        f"{forwarding} now run the model they load: move them to INFERENCE_TOOL_LOADERS and call "
        f"{SEAM} on the load, once the weights sit on their final device."
    )


def test_device_placement_rejects_a_meta_buffer():
    """The trainer backstop is a hard stop, mirroring the meta-parameter branch: a buffer still on
    meta holds no values, and ``torch.empty`` would run on per-rank-different uninitialized memory."""
    model = nn.Linear(4, 4)
    model.register_buffer("rope_inv_freq", torch.empty(4, device="meta"), persistent=False)

    with pytest.raises(RuntimeError, match=r"BUFFER\(s\).*rope_inv_freq"):
        DistributedTrainerMixin._move_model_to_device(None, model, "cpu")


def test_device_placement_accepts_a_materialized_model():
    """The raise must key on meta, not on the mere presence of non-persistent buffers."""
    model = nn.Linear(4, 4)
    model.register_buffer("mask", torch.zeros(4), persistent=False)

    DistributedTrainerMixin._move_model_to_device(None, model, "cpu")

    assert model.mask.device.type == "cpu"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
