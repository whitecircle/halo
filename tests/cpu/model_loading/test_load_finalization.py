#!/usr/bin/env python
"""``finalize_loaded_model`` and the sweep that every load path reaches it. transformers 5 re-materializes
non-persistent buffers uninitialized, and a garbage ``inv_freq`` degrades RoPE to NoPE."""

import ast
import functools
import re
import types

import pytest
import torch
import torch.nn as nn
from transformers import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding

from halo.model_loading.finalization import finalize_loaded_model
from halo.trainers.distributed_trainer import DistributedTrainerMixin
from tests.common.repo import REPO_ROOT
from tests.common.source_sweep import (
    TOOLKIT_LOAD_CALLS,
    builds_a_model,
    call_name,
    called_names,
    functions_calling,
    functions_in,
    imported_module_files,
)

SEAM = finalize_loaded_model.__name__

DISPATCHER = "halo/model_loading/parallel/loader.py"
LOADING_PACKAGE = "halo/model_loading"

# Pinned as an exact set so a renamed or added loader is a deliberate edit rather than a silently shrunk sweep.
DISPATCHED_LOADERS = frozenset(
    {
        "load_pp_stage_model",
        "load_ep_tp_model",
        "load_tp_model",
        "load_tp_moe_model",
        "load_ep_cp_model",
        "load_ep_model",
        "load_cp_model",
        "_load_undistributed_model",
    }
)

# Each must call the seam directly: the reachability sweep is branch-insensitive, so a terminal could still reach it
# through a sibling branch.
TERMINAL_LOADERS = (
    (DISPATCHER, "_build_random_init"),
    ("halo/model_loading/parallel/tp.py", "load_tp_model"),
    ("halo/model_loading/parallel/frozen_models.py", "load_frozen_auxiliary_model"),
    ("halo/model_loading/parallel/whole_model.py", "resolve_trainer_model"),
    ("halo/model_loading/parallel/whole_model.py", "load_whole_model"),
    ("halo/model_loading/parallel/ep.py", "load_ep_model_lazy"),
    ("halo/model_loading/parallel/pp.py", "_load_stage_lazy"),
)

# Includes every halo module a CLI outside scripts/training imports, so a driver moved to another module stays swept.
TOOL_DRIVER_ROOTS = (
    "scripts",
    *imported_module_files(
        sorted(
            path.relative_to(REPO_ROOT).as_posix()
            for path in (REPO_ROOT / "scripts").rglob("*.py")
            if not path.is_relative_to(REPO_ROOT / "scripts" / "training")
        ),
        "halo",
    ),
)

# An uninitialized buffer here becomes numbers the operator acts on, with no other symptom.
INFERENCE_TOOL_LOADERS = (
    ("halo/entrypoints/reward_scoring.py", "load_reward_model"),
    ("halo/data/deduplication.py", "compute_embeddings"),
    ("halo/checkpoint/tools/bf16_conversion.py", "_load_verified"),
)

# No forward exists in these files, and ``state_dict`` omits non-persistent buffers, so an unrepaired one reaches
# neither a number nor the checkpoint.
CONVERSION_TOOL_LOADERS = (
    ("halo/checkpoint/tools/adapter_merge.py", "load_task_base_model"),
    ("halo/checkpoint/tools/vocab_patch.py", "patch_vocabulary"),
    ("halo/checkpoint/tools/deepseek_v4_dequant.py", "convert_deepseek_v4_checkpoint"),
)

_FORWARD_MARKERS = (".generate(", ".logits", "last_hidden_state")

# A load passing a family's own ``trust_remote_code`` is roster-generic, so it takes the families whose
# ``_init_weights`` repairs no buffer.
TEST_ROOT = "tests"
# Loads every Ulysses family, Bailing among them, by its own flag.
CP_RESUME_ORACLE = ("tests/gpu/trainers/grpo/test_offline_grpo_cp_resume.py", "_eager_oracle")


def _rotary_with_reference(theta: float = 1_000_000.0) -> tuple[Qwen3RotaryEmbedding, torch.Tensor]:
    cfg = Qwen3Config(rope_theta=theta, hidden_size=64, num_attention_heads=4, head_dim=16)
    rotary = Qwen3RotaryEmbedding(cfg)
    return rotary, rotary.inv_freq.clone()


class _SlopeAttention(nn.Module):
    """The shape a ``_NON_PERSISTENT_FIXERS`` fixer keys on (Bailing Lightning-Attention slopes)."""

    def __init__(self):
        super().__init__()
        self.config = types.SimpleNamespace(num_hidden_layers=4)
        self.num_heads = 4
        self.layer_idx = 1
        self.register_buffer("slope", torch.empty(4, device="meta"), persistent=False)


class _LoadedShell(nn.Module):
    def __init__(self):
        super().__init__()
        self.rotary_emb, self.reference_inv_freq = _rotary_with_reference()
        self.attn = _SlopeAttention()
        self.tied = 0

    def tie_weights(self):
        self.tied += 1


def test_seam_recomputes_a_meta_inv_freq():
    """The buffer never materialized, so values must be recomputed from the config, on a real device, in fp32."""
    shell = _LoadedShell()
    shell.rotary_emb.register_buffer(
        "inv_freq", torch.empty_like(shell.reference_inv_freq, device="meta"), persistent=False
    )

    finalize_loaded_model(shell)

    # A meta buffer is recomputed onto current_device(), which is cuda on a GPU host.
    inv_freq = shell.rotary_emb.inv_freq.cpu()
    assert shell.rotary_emb.inv_freq.device.type != "meta"
    assert inv_freq.dtype == torch.float32
    assert torch.allclose(inv_freq, shell.reference_inv_freq), (
        f"inv_freq not recomputed from the config: got {inv_freq[1].item():.6g}, "
        f"expected {shell.reference_inv_freq[1].item():.6g} (theta=1e6)"
    )


def test_seam_materializes_non_rotary_buffer_families():
    """Dropping the ``_NON_PERSISTENT_FIXERS`` leg would strand Bailing slopes and Gemma4 embed_scale on meta."""
    shell = _LoadedShell()

    finalize_loaded_model(shell)

    slope = shell.attn.slope
    assert slope.device.type != "meta"
    assert slope.shape == (4,)
    assert (slope < 0).all(), "ALiBi slopes are strictly negative by construction"


def test_seam_reties_shared_weights():
    """The EP/PP loaders delegate re-tying to the seam; without it the meta lm_head crashes their first backward."""
    shell = _LoadedShell()

    finalize_loaded_model(shell)

    assert shell.tied == 1


def _graph_modules() -> list[str]:
    """A loader that never finalizes is in the graph without reaching the seam, so the sweep fails."""
    roots = (REPO_ROOT / LOADING_PACKAGE, REPO_ROOT / "halo/distributed")
    assert all(root.is_dir() for root in roots), f"the sweep lost a root: {roots}"
    package = (REPO_ROOT / LOADING_PACKAGE).rglob("*.py")
    distributed = (REPO_ROOT / "halo/distributed").rglob("*.py")
    callers = (path for path in (REPO_ROOT / "halo").rglob("*.py") if SEAM in path.read_text(encoding="utf-8"))
    return sorted({path.relative_to(REPO_ROOT).as_posix() for path in (*package, *distributed, *callers)})


@functools.cache
def _call_graph() -> dict[tuple[str, str], frozenset[str]]:
    """Keyed by module too, or one module's finalizing twin vouches for the other's forgetful one."""
    return {(rel, node.name): called_names(node) for rel, node in functions_in(_graph_modules(), top_level=True)}


def _reaches_seam(root: tuple[str, str], graph: dict[tuple[str, str], frozenset[str]]) -> bool:
    """A callee name resolves inside its own module first, then to any module defining it, the way an import does."""
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


def test_the_sweep_sees_the_loading_package():
    """Anti-vacuity: a moved package globs to nothing."""
    swept = [path for path in _graph_modules() if path.startswith(f"{LOADING_PACKAGE}/")]
    assert len(swept) > 5, f"the sweep sees {len(swept)} files under {LOADING_PACKAGE}/"


def test_every_dispatched_load_path_reaches_the_seam():
    """Roots are discovered: a missing repair is the omission a hand-list ships."""
    graph = _call_graph()
    called = graph[(DISPATCHER, "_dispatch_model_loading")]
    roots = {name for name in called if re.fullmatch(r"_?load_\w+_model", name)}
    assert roots == DISPATCHED_LOADERS, (
        f"the dispatcher's load surface changed: {sorted(roots ^ DISPATCHED_LOADERS)}. Update "
        f"DISPATCHED_LOADERS deliberately — a shrinking sweep proves nothing."
    )

    homes = {name: rel for (rel, name) in graph if name in roots}
    unfinalized = sorted(name for name in roots if not _reaches_seam((homes[name], name), graph))
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
def _tool_loaders() -> dict[tuple[str, str], frozenset[str]]:
    """Walked over the whole tree, so a new tool is discovered by the load it makes; a third-party loader
    (``SentenceTransformer``) is outside this sweep."""
    loaders = functions_calling(TOOL_DRIVER_ROOTS, lambda call: call_name(call) in TOOLKIT_LOAD_CALLS)
    return {key: names for key, names in loaders.items() if key[1] not in TOOLKIT_LOAD_CALLS}


def test_tool_load_surface_is_pinned():
    """The split judges what each tool does with the model, so a new loader must be classified deliberately."""
    discovered = set(_tool_loaders())
    pinned = set(INFERENCE_TOOL_LOADERS) | set(CONVERSION_TOOL_LOADERS)
    assert discovered == pinned, (
        f"the scripts/ load surface changed: {sorted(discovered ^ pinned)}. Classify each new loader "
        f"— INFERENCE_TOOL_LOADERS if anything runs the model, CONVERSION_TOOL_LOADERS if it is only "
        f"cast/merged/saved."
    )


def test_the_tool_roots_hold_every_halo_module_they_import():
    """Moving a driver one import deeper must not drop it from the sweep."""
    halo_roots = [root for root in TOOL_DRIVER_ROOTS if root.startswith("halo/")]
    missing = sorted(set(imported_module_files(halo_roots, "halo")) - set(halo_roots))
    assert not missing, f"the tool roots import {missing} without sweeping them"


@pytest.mark.parametrize("rel,function", INFERENCE_TOOL_LOADERS, ids=[fn for _, fn in INFERENCE_TOOL_LOADERS])
def test_every_inference_tool_loader_finalizes(rel, function):
    """A bare load returns garbage: a dead RoPE on zeroed pages, NaN on reused ones."""
    loaders = _tool_loaders()
    assert (rel, function) in loaders, f"{function} no longer loads a model in {rel} — update the pinned sets"
    assert SEAM in loaders[(rel, function)], (
        f"{rel}:{function} hands a bare-loaded model to a forward without calling {SEAM}: its rotary "
        f"inv_freq and any family decay table are uninitialized memory."
    )


def _runs_a_forward(rel: str) -> bool:
    return any(marker in (REPO_ROOT / rel).read_text(encoding="utf-8") for marker in _FORWARD_MARKERS)


def test_conversion_tools_still_run_no_forward():
    """A conversion tool that starts generating or scoring must be reclassified."""
    forwarding = sorted(rel for rel, _ in CONVERSION_TOOL_LOADERS if _runs_a_forward(rel))
    assert not forwarding, (
        f"{forwarding} now run the model they load: move them to INFERENCE_TOOL_LOADERS and call "
        f"{SEAM} on the load, once the weights sit on their final device."
    )


def _loads_by_family_flag(call: ast.Call) -> bool:
    return builds_a_model(call) and any(
        keyword.arg == "trust_remote_code" and not isinstance(keyword.value, ast.Constant) for keyword in call.keywords
    )


@functools.cache
def _test_oracle_loaders() -> dict[tuple[str, str], frozenset[str]]:
    loaders = functions_calling((TEST_ROOT,), _loads_by_family_flag)
    return {key: names for key, names in loaders.items() if _runs_a_forward(key[0])}


def test_the_test_oracle_sweep_sees_the_roster_oracle():
    """Anti-vacuity: a predicate or forward marker that stopped matching would pass every oracle."""
    assert CP_RESUME_ORACLE in _test_oracle_loaders(), f"the sweep found {sorted(_test_oracle_loaders())}"


def test_every_test_side_oracle_finalizes():
    """A bare oracle holds a remote-code rotary table as uninitialized memory, so the suite reads noise or NaN as
    the parallel run's error."""
    unfinalized = sorted(f"{rel}:{name}" for (rel, name), names in _test_oracle_loaders().items() if SEAM not in names)
    assert not unfinalized, (
        f"{unfinalized} load a model by its family's trust_remote_code and hand it to a forward without calling "
        f"{SEAM}. Call it once the weights sit on their final device, as every production loader does."
    )


def test_no_trainer_loads_a_model_itself():
    """A load inside ``halo/trainers`` skips the buffer repair, the sinks policy and the family attention patches."""
    offenders = sorted(f"{rel}:{name}" for rel, name in functions_calling(("halo/trainers",), builds_a_model))
    assert not offenders, (
        f"{offenders} load a model inside a trainer. Load it in the entry script through the loader "
        f"that owns its kind ({SEAM} runs there) and pass it to the trainer."
    )


def test_device_placement_rejects_a_meta_buffer():
    """``torch.empty`` would run on per-rank-different uninitialized memory."""
    model = nn.Linear(4, 4)
    model.register_buffer("rope_inv_freq", torch.empty(4, device="meta"), persistent=False)

    with pytest.raises(RuntimeError, match=r"BUFFER\(s\).*rope_inv_freq"):
        DistributedTrainerMixin._move_model_to_device(None, model, "cpu")


def test_device_placement_accepts_a_materialized_model():
    model = nn.Linear(4, 4)
    model.register_buffer("mask", torch.zeros(4), persistent=False)

    DistributedTrainerMixin._move_model_to_device(None, model, "cpu")

    assert model.mask.device.type == "cpu"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
