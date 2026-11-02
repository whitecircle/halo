#!/usr/bin/env python
"""Every shipped ``examples/`` YAML must be *runnable*, not merely parseable.

Its sibling ``test_examples_parse.py`` pins the parse layer (unknown key, stale knob, bad
``Literal``). This module pins the contracts that only bite AFTER the parse — the ones that abort a
run minutes in, on a booked multi-GPU host, once the model is already resident:

* the parallelism shape, built through ``parallelism_config_from_args`` at the topology the file's
  own launch comment declares, with the capability flags its entry script passes;
* the rollout timeouts, validated against the NCCL watchdog the launch comment documents;
* the allocator setting the launch lines carry: expandable segments on the dense SFT, DPO and SMPO
  examples, never on an EP one;
* the ``batch_rebalance`` train sampler, kept to the base loader and to a processing class whose first
  input a text row carries;
* against the model's real config (skipped when it is not in the local HF cache — this test never
  reaches the network): the router-balancing export contract, the context-window budget, the
  expert/head divisibility, and the GptOss sink rules.

Every check runs through the production seam rather than a restatement of it, so a rule change lands
here as a failure instead of drifting.

Run: pytest tests/cpu/config/test_examples_roster.py
"""

import ast
import re
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from datasets import Dataset
from transformers import Trainer
from transformers.trainer_pt_utils import BatchRebalanceSampler

from halo.configs.parallelism_arguments import parallelism_config_from_args
from halo.distributed.expert_parallel.families.roster import ep_layer_classes_for_config
from halo.environments.rollout_config import resolve_reasoning_end_ids
from halo.model_loading.parallel.load_rules import reject_unsupported_model
from halo.model_loading.parallel.peft_setup import split_expert_lora_targets
from halo.model_loading.tokenizer_setup import load_processing_class, resolve_tokenizer
from halo.models.attention.gpt_oss_sinks import SinksPolicy, uses_gpt_oss_sinks
from halo.models.attention.selection import flash_kernel, is_flash_attn_implementation, validate_attn_implementation
from halo.models.config_access import text_config
from halo.models.context_window import is_bounded_length
from halo.models.moe.balancing import BIAS_UPDATE_MODES
from halo.models.moe.config_fields import has_routed_experts
from halo.trainers.dataloaders import uses_dp_sharded_loader
from tests.common.hub_seed import is_hub_repo
from tests.common.parallelism import make_parallelism_config
from tests.common.repo import REPO_ROOT
from tests.common.tiny_models import TINY_MOE_FAMILIES, build_tiny_family_checkpoint, word_level_tokenizer
from tests.common.tokenizers import load_cached_config, load_cached_tokenizer, try_cached_config
from tests.cpu.config.test_examples_parse import (
    _EXAMPLES,
    ENV_GRPO_EXAMPLES,
    ENV_GRPO_SCRIPT,
    EXAMPLES_ROOT,
    TRAINING_ROOT,
    _bf16_capable,  # noqa: F401 — autouse fixture, imported so both modules validate as a training host does
    _script_module,
    parsed_field,
    parser_for,
    script_for,
    uses_expert_parallelism,
)

# The repository's standard node, assumed for a config whose comments state no launch width. Every
# rank-math rule (NVLink domain, EP group count, DP width) is relative to it.
DEFAULT_GPUS_PER_NODE = 8

_NPROC = re.compile(r"--nproc[-_]per[-_]node[= ](\d+)")
_NNODES = re.compile(r"--nnodes[= ](\d+)")
_NCCL_MINUTES = re.compile(r"DIST_NCCL_TIMEOUT_MINUTES=(\d+)")
_EXPANDABLE_SEGMENTS = re.compile(r"PYTORCH_(?:CUDA_)?ALLOC_CONF=\S*expandable_segments:True")
# The setting as the launcher's env prefix (other assignments may sit between them); placed after the
# launcher it would be an argument, and the run would not see it.
_EXPANDABLE_SEGMENTS_PREFIX = re.compile(
    r"PYTORCH_(?:CUDA_)?ALLOC_CONF=\S*expandable_segments:True\s+(?:\w+=\S*\s+)*(?:python|torchrun|accelerate|halo)\s"
)
_HALO_LAUNCH = re.compile(r"\bhalo launch\s")
_COMMENT_CONTINUATION = re.compile(r"\\\n\s*#\s*")
_EXPORT_LINE = re.compile(r"\s*#\s*export\s")
_ALLOCATOR_DOC = "agent-docs/optimization/throughput-benchmarks.md#allocator-expandable-segments"
# The methods whose dense runs the allocator setting was measured on (_ALLOCATOR_DOC).
_EXPANDABLE_SEGMENTS_SCRIPTS = frozenset({"sft.py", "preference/dpo.py", "preference/smpo.py"})

# Parametrization ids: the example's path under examples/, so a failure names the file to open.
_ID = {"ids": lambda config: config.relative_to(EXAMPLES_ROOT).as_posix()}


@pytest.fixture(autouse=True)
def _hub_offline(monkeypatch):
    """No network from this tier, on any path.

    Several production helpers resolve the model's own ``config.json`` themselves (the expert-LoRA
    peel among them). Offline turns a cache miss into an immediate error the caller skips on,
    instead of a hub round-trip that makes the CPU suite depend on connectivity.
    """
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")


def raw_config(config: Path) -> dict:
    return yaml.safe_load(config.read_text(encoding="utf-8")) or {}


def comment_block(config: Path) -> str:
    """The example's comment lines — where its launch line and its serving contract live."""
    return "\n".join(line for line in config.read_text(encoding="utf-8").splitlines() if line.lstrip().startswith("#"))


def launch_commands(config: Path) -> list[str]:
    """The example's launch commands: comment lines naming an entry script or ``halo launch``, ``\\``
    continuations folded in.

    Keyed on the entry script rather than the launcher, so prose that names ``torchrun`` is not a command and
    a wrapped command whose script sits on its continuation line still is.
    """
    folded = _COMMENT_CONTINUATION.sub(" ", comment_block(config))
    return [line for line in folded.splitlines() if "scripts/training/" in line or _HALO_LAUNCH.search(line)]


# Only the rollout methods carry an episode budget to check against the watchdog.
_ROLLOUT_EXAMPLES = [config for config in _EXAMPLES if "episode_timeout" in raw_config(config)]

# The two methods that push trained weights into a live rollout engine. Their launch comments are the
# only place the server-side half of the contract is written down, so it is checked as part of the file.
_WEIGHT_SYNC_EXAMPLES = [config for config in _EXAMPLES if script_for(config) in ("grpo/online.py", ENV_GRPO_SCRIPT)]
_BATCH_REBALANCE_EXAMPLES = [
    config for config in _EXAMPLES if raw_config(config).get("train_sampling_strategy") == "batch_rebalance"
]

# The tiny family each host-local placeholder checkpoint holds. Where the placeholder is absent, a check that
# reads only the processing class runs on that family's tiny checkpoint with an offline tokenizer: the
# release's own file set (config and tokenizer, no processor). A release that ships a processor has no such
# stand-in.
_PLACEHOLDER_FAMILIES = {"/mnt/models/DeepSeek-V4-Flash-BF16": "deepseek_v4"}


def launch_topology(config: Path) -> tuple[int, int]:
    """``(world_size, gpus_per_node)`` the example's own launch comment declares.

    Read off the comment rather than guessed: ``ep_size=4`` is one legal dispatch group under
    ``--nproc_per_node=4`` and two racy ones on a full node, so the launch width is what decides
    whether the shape below is accepted. Falls back to a single standard node.
    """
    comments = comment_block(config)
    gpus_per_node = int(match.group(1)) if (match := _NPROC.search(comments)) else DEFAULT_GPUS_PER_NODE
    nodes = int(match.group(1)) if (match := _NNODES.search(comments)) else 1
    return gpus_per_node * nodes, gpus_per_node


def declared_nccl_timeout_minutes(config: Path) -> str | None:
    """``DIST_NCCL_TIMEOUT_MINUTES`` as the example's own instructions set it, if they do."""
    match = _NCCL_MINUTES.search(config.read_text(encoding="utf-8"))
    return match.group(1) if match else None


def script_call_kwargs(script: str, func: str, names: tuple[str, ...]) -> dict:
    """Constant keyword arguments the entry script passes to ``func``.

    The capability flags a script passes (``supports_init_from_scratch``) live in the
    script's own call, so reading them there keeps this test honest when a script gains
    or loses a mode — a hand-kept table here would keep asserting the retired one.
    """
    tree = ast.parse((TRAINING_ROOT / script).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == func:
            return {
                keyword.arg: keyword.value.value
                for keyword in node.keywords
                if keyword.arg in names and isinstance(keyword.value, ast.Constant)
            }
    return {}


def script_trainer_classes(script: str) -> tuple[type, ...]:
    """Every trainer class the entry script can hand ``init_training_script``: the classes named in
    its ``trainer_cls=`` argument, or in the assignment that argument reads (``--use_sdpg`` picks one
    of two). The CP/PP gates are those classes' own ``SUPPORTS_CP`` / ``PP_REFUSAL``."""
    module = _script_module(script)
    tree = ast.parse((TRAINING_ROOT / script).read_text(encoding="utf-8"))
    call = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "init_training_script"
    )
    value = next(keyword.value for keyword in call.keywords if keyword.arg == "trainer_cls")
    if isinstance(value, ast.Name) and not hasattr(module, value.id):
        value = next(
            node.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign) and any(getattr(target, "id", None) == value.id for target in node.targets)
        )
    classes = tuple(
        getattr(module, node.id)
        for node in ast.walk(value)
        if isinstance(node, ast.Name) and isinstance(getattr(module, node.id, None), type)
    )
    assert classes, f"{script}: no trainer class resolved from its init_training_script(trainer_cls=...)"
    return classes


def build_parallelism_config(config: Path):
    """The ``ParallelismConfig`` this example would build on its declared launch topology.

    Goes through ``parallelism_config_from_args`` — the factory the entry scripts call — so the
    trainer-capability rejections (CP/PP on a script that supports neither) and every
    ``__post_init__`` rank rule are the ones under test. The world-size seams are the three
    ``halo.runtime.ranks`` readers the module imported; a CPU tier has no launcher to report them.
    """
    script = script_for(config)
    parsed = parser_for(script).parse_yaml_file(str(config))
    dist_args = next(obj for obj in parsed if type(obj).__name__ == "ParallelismArguments")
    model_config = next(obj for obj in parsed if type(obj).__name__ == "ModelArguments")
    flags = script_call_kwargs(script, "init_training_script", ("supports_init_from_scratch",))
    # The peel reads the model's own config; where that is unreachable the expert-LoRA rejections
    # (PP and Expert-TP both refuse it) go unchecked, exactly like the model-dependent cases below.
    expert_lora = (
        split_expert_lora_targets(model_config)
        if try_cached_config(model_config.model_name_or_path, trust_remote_code=True) is not None
        else None
    )
    world_size, gpus_per_node = launch_topology(config)
    with pytest.MonkeyPatch.context() as patcher:
        for name, value in (
            ("get_global_world_size", world_size),
            ("get_local_world_size", gpus_per_node),
            ("get_global_rank", 0),
        ):
            patcher.setattr(f"halo.distributed.parallelism_config.{name}", lambda _value=value: _value)
        built = [
            parallelism_config_from_args(dist_args, trainer_cls=trainer_cls, expert_lora=expert_lora, **flags)
            for trainer_cls in script_trainer_classes(script)
        ]
        return parsed, built[0]


def model_config_for(config: Path, parsed: tuple):
    """The example's model config from the local HF cache; a skip when this host cannot resolve it.

    ``local_files_only`` keeps this tier offline: a runner without the cache skips the
    config-dependent checks instead of hanging on the hub, and a host-local checkpoint path
    (``/mnt/...``) that does not exist here skips the same way.
    """
    return load_cached_config(parsed_field(parsed, "model_name_or_path"), trust_remote_code=True)


def processing_class_source(reference: str, stand_in_dir: Path) -> str:
    """Where the checkpoint ``reference`` loads its processing class from on this host: itself when it is a
    Hub id or a directory here, else the tiny stand-in :data:`_PLACEHOLDER_FAMILIES` names for it, written to
    ``stand_in_dir``."""
    if is_hub_repo(reference) or Path(reference).exists():
        return reference
    family = _PLACEHOLDER_FAMILIES.get(reference)
    assert family is not None, (
        f"{reference} is a local checkpoint this host does not have. Name its tiny family in "
        f"_PLACEHOLDER_FAMILIES only if it holds no processor, since the stand-in is tokenizer-only; one that "
        f"holds a processor is checked only on a host that has it"
    )
    build_tiny_family_checkpoint(TINY_MOE_FAMILIES[family], str(stand_in_dir), word_level_tokenizer())
    return str(stand_in_dir)


def test_examples_declare_their_launch_width():
    """A multi-node example must state its launch line, or the topology assumed here is fiction."""
    undeclared = [
        config.relative_to(REPO_ROOT).as_posix()
        for config in _EXAMPLES
        if raw_config(config).get("ep_scope") == "global" and not _NPROC.search(config.read_text(encoding="utf-8"))
    ]
    assert not undeclared, (
        "these configs declare cross-domain EP but no --nproc_per_node/--nnodes launch line, so "
        "their rank math cannot be checked:\n  " + "\n  ".join(undeclared)
    )


def test_an_expandable_segments_example_exists():
    """The EP refusal below reads launch commands through this pattern; one that matches none would refuse nothing."""
    assert any(_EXPANDABLE_SEGMENTS.search(line) for config in _EXAMPLES for line in launch_commands(config)), (
        "no example's launch command sets PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True, so the pattern has "
        "drifted from the spelling the dense examples use"
    )


@pytest.mark.parametrize("config", _EXAMPLES, **_ID)
def test_example_allocator_setting_follows_expert_parallelism(config):
    """Dense SFT, DPO and SMPO examples launch with expandable segments; no EP example does.

    On those dense runs the setting lowers peak reserved memory at unchanged throughput, so each of their launch
    commands carries it as the launcher's env prefix (a reader copies one); under EP it costs throughput, so no EP
    launch command or ``export`` line sets it, while prose may still name it as the fit-only fix. Other methods and
    a MoE run without EP are unmeasured, so neither rule binds them. Measurements: ``_ALLOCATOR_DOC``.
    """
    commands = launch_commands(config)
    if uses_expert_parallelism(raw_config(config)):
        exports = [line for line in comment_block(config).splitlines() if _EXPORT_LINE.match(line)]
        assert not any(_EXPANDABLE_SEGMENTS.search(line) for line in commands + exports), (
            f"this EP example launches with expandable_segments, which costs throughput under EP ({_ALLOCATOR_DOC}); "
            "drop it"
        )
        return
    script = script_for(config)
    if script not in _EXPANDABLE_SEGMENTS_SCRIPTS:
        return
    parsed = parser_for(script).parse_yaml_file(str(config))
    if has_routed_experts(model_config_for(config, parsed)):
        return
    bare = [line for line in commands if not _EXPANDABLE_SEGMENTS_PREFIX.search(line)]
    assert commands and not bare, (
        "this dense example's launch commands must start with PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "
        f"in front of the launcher ({_ALLOCATOR_DOC}); without it: {bare or 'no launch command found'}"
    )


@pytest.mark.parametrize("config", _EXAMPLES, **_ID)
def test_example_parallelism_shape_is_accepted(config):
    """The declared axes must survive ``ParallelismConfig`` at the example's own launch width.

    This is where a racy single-domain multi-group EP topology, an unlisted axis combination, or a
    CP/PP request on a trainer that supports neither is rejected — all of them config-time raises
    that would otherwise burn a booked multi-GPU host at step 0.
    """
    build_parallelism_config(config)


@pytest.mark.parametrize("config", _EXAMPLES, **_ID)
def test_example_thinking_budget_has_a_server_side_parser(config):
    """A thinking budget the serving engine cannot enforce is a 400 on EVERY rollout, not a no-op.

    ``rollout_max_thinking_tokens`` rides on vLLM's ``thinking_token_budget``, which the server only
    accepts with a reasoning parser loaded — 0.32.0 registers ``qwen3`` and ``gemma4`` among others,
    while gpt-oss takes the baked ``openai_gptoss`` plugin (the image comments its built-in entry out).
    Sending it anyway errors every episode (``episode/error_rate`` 1) with nothing raised at config
    time, so the file's own launch instructions have to name the parser flag that makes the knob real.
    """
    if raw_config(config).get("rollout_max_thinking_tokens") is None:
        return
    assert "--reasoning-parser" in comment_block(config), (
        "this config sends a thinking budget but its launch instructions name no --reasoning-parser; "
        "vLLM refuses thinking_token_budget with a 400 on every rollout without one"
    )


@pytest.mark.parametrize("config", ENV_GRPO_EXAMPLES, **_ID)
def test_vllm_example_names_a_reasoning_close_its_tokenizer_writes(config):
    """The loss finds the reasoning closes vLLM's thinking budget forces by ``rollout_reasoning_end_token``.

    A marker the model's tokenizer does not write — the ``</think>`` default on Gemma 4 (``<channel|>``) or
    gpt-oss (its six-token move into the final channel) — only warns at startup, and every forced close then
    trains with the episode's advantage. SGLang enforces no budget, so forces no close to find.
    """
    parsed = parser_for(script_for(config)).parse_yaml_file(str(config))
    if parsed_field(parsed, "rollout_backend") == "sglang":
        return
    tokenizer = load_cached_tokenizer(parsed_field(parsed, "model_name_or_path"), trust_remote_code=True)
    resolve_reasoning_end_ids(tokenizer, parsed_field(parsed, "rollout_reasoning_end_token"))


def test_a_weight_sync_example_exists():
    """The three server-contract checks below are parametrized over this list."""
    assert _WEIGHT_SYNC_EXAMPLES and ENV_GRPO_EXAMPLES, "no shipped example drives a rollout engine"


@pytest.mark.parametrize("config", ENV_GRPO_EXAMPLES, **_ID)
def test_sampled_token_example_names_the_server_flag(config):
    """Sampled-token training without ``--return-tokens-as-token-ids`` trains on ids nobody sampled.

    The knob defaults ON, and a server that returns no ids is not an error: every turn falls back to
    re-tokenizing a chat-template re-render — one warning, then the whole run — and the vLLM
    importance-sampling correction, which needs the sampling log-probs that ride along with the ids,
    goes silently with it. Read through the parser, since the file usually leaves the knob at its
    default. Only vLLM takes a flag; SGLang returns the ids per request.
    """
    parsed = parser_for(script_for(config)).parse_yaml_file(str(config))
    if not parsed_field(parsed, "train_on_sampled_tokens") or parsed_field(parsed, "rollout_backend") == "sglang":
        return
    assert "--return-tokens-as-token-ids" in comment_block(config), (
        "this config trains on the engine's sampled tokens but its launch instructions name no "
        "--return-tokens-as-token-ids; without it every turn silently falls back to a re-tokenized "
        "re-render and the IS correction is disabled"
    )


@pytest.mark.parametrize("config", _WEIGHT_SYNC_EXAMPLES, **_ID)
def test_custom_chat_template_example_hands_it_to_the_server(config):
    """A template forced on the trainer alone leaves the ENGINE on the checkpoint's own.

    The rollout prompt is rendered server-side, so the divergence lands in the context the policy is
    sampled under, not merely in the trainer's fallback render: the shipped harmony template guards
    ``tool_call.arguments`` against double-encoding, and the gpt-oss checkpoint's built-in does not,
    so every tool-carrying turn is served quoted-and-escaped. Nothing raises on either side.
    """
    if not raw_config(config).get("chat_template"):
        return
    comments = comment_block(config)
    assert "--chat-template" in comments or "CHAT_TEMPLATE" in comments, (
        "this config forces its own chat template on the trainer but its launch instructions never "
        "pass the same file to the server, which then serves the checkpoint's built-in template"
    )


@pytest.mark.parametrize("config", _WEIGHT_SYNC_EXAMPLES, **_ID)
def test_moe_weight_sync_example_names_the_triton_moe_backend(config):
    """An auto-selected MoE backend repacks the expert weights the sync writes.

    vLLM's Blackwell defaults (FLASHINFER_TRTLLM / CUTLASS) and SGLang's flashinfer runner hold the
    experts in a repacked layout, while the push writes the bf16 unpacked one. The engine accepts
    the update either way — nothing raises; the served policy just stops being the trained one.
    """
    parsed = parser_for(script_for(config)).parse_yaml_file(str(config))
    if not ep_layer_classes_for_config(model_config_for(config, parsed)):
        return  # dense: no expert weights to repack
    comments = comment_block(config)
    assert "--moe-backend triton" in comments or "--moe-runner-backend triton" in comments, (
        "this config syncs weights into an engine serving a MoE model but its launch instructions "
        "name no triton MoE backend; the auto-selected ones repack the expert weights and every "
        "update after the first lands corrupted"
    )


@pytest.mark.parametrize("config", _EXAMPLES, **_ID)
def test_moe_example_declares_the_reentrant_checkpointing_it_runs(config):
    """A MoE (EP or not) and CP FORCE ``use_reentrant: True``; a config saying ``false`` describes fiction.

    CP's and EP's collectives do not survive non-reentrant recompute, and a MoE routes in its
    recompute, so the mixin overwrites the kwarg before gradient checkpointing is enabled — warning
    when it is explicitly ``false``. Both spellings therefore run identically, so a tree carrying both
    reads as two contradicting comments. Pipeline parallelism is the opposite rule (non-reentrant
    REQUIRED, reentrant raises), so it is excluded here rather than folded in. A MoE this host cannot
    resolve from its cache is read off ``expert_parallel_size`` alone.
    """
    raw = raw_config(config)
    if (raw.get("pipeline_parallel_size") or 1) > 1:
        return
    parsed = parser_for(script_for(config)).parse_yaml_file(str(config))
    model_config = try_cached_config(parsed_field(parsed, "model_name_or_path"), trust_remote_code=True)
    if not uses_expert_parallelism(raw) and not has_routed_experts(model_config):
        return
    declared = (raw.get("gradient_checkpointing_kwargs") or {}).get("use_reentrant")
    assert declared is not False, (
        "a MoE force-promotes use_reentrant to True (halo/trainers/distributed_trainer.py), so this config's "
        "use_reentrant: false is silently overridden — write true, or drop the kwarg"
    )


def test_a_racy_ep_topology_is_still_rejected():
    """Negative control: without this, the sweep above would pass on an unvalidated rank rule."""
    with pytest.raises(ValueError, match="dispatch groups"):
        make_parallelism_config(world_size=8, gpus_per_node=8, ep_size=4)


def test_a_rollout_example_exists():
    """The timeout check is parametrized over this list — an empty one would assert nothing."""
    assert _ROLLOUT_EXAMPLES, "no shipped example declares episode_timeout"


@pytest.mark.parametrize("config", _ROLLOUT_EXAMPLES, **_ID)
def test_rollout_timeouts_fit_the_documented_nccl_watchdog(config, monkeypatch):
    """An example's episode budget must fit the watchdog ITS OWN instructions install.

    ``episode_timeout`` at or above the NCCL collective timeout lets a straggler's peers abort
    before it can be cancelled, so the config refuses it — but only at training start, after every
    rank has loaded the model. The watchdog comes from ``DIST_NCCL_TIMEOUT_MINUTES``, which an
    example raises by documenting it in its launch line; a budget that needs a raised watchdog and
    never says so ships a run that dies late, on the default.
    """
    declared = declared_nccl_timeout_minutes(config)
    if declared is None:
        monkeypatch.delenv("DIST_NCCL_TIMEOUT_MINUTES", raising=False)
    else:
        monkeypatch.setenv("DIST_NCCL_TIMEOUT_MINUTES", declared)
    async_config = next(
        obj
        for obj in parser_for(script_for(config)).parse_yaml_file(str(config))
        if type(obj).__name__ == "AsyncRolloutConfig"
    )
    async_config.get_rollout_config()


def test_some_example_model_is_cached():
    """Guards the config-dependent cases below: all-skipped would look green while asserting nothing."""
    cached = [
        config
        for config in _EXAMPLES
        if try_cached_config(raw_config(config)["model_name_or_path"], trust_remote_code=True) is not None
    ]
    assert cached, (
        "no example's model is in the local HF cache, so every config-dependent check below skipped. "
        "Populate HF_HOME (configs only) before trusting this file's result."
    )


@pytest.mark.parametrize("config", _EXAMPLES, **_ID)
def test_example_balancing_mode_matches_the_family_export_contract(config):
    """``bias_update`` only where the family's bias EXPORTS; ``bias_update_transient`` only where it does not.

    Both directions raise at run start. The verdict is the family spec's ``balancing_slot``, resolved
    off the checkpoint's own ``model_type`` — never a family name.
    The remaining run-start raise, for a model whose routers carry no bias at all, needs the loaded
    modules and stays with the callback.
    """
    parsed = parser_for(script_for(config)).parse_yaml_file(str(config))
    mode = parsed_field(parsed, "moe_balancing")
    if mode not in BIAS_UPDATE_MODES:
        return
    layer_classes = ep_layer_classes_for_config(model_config_for(config, parsed))
    transient = sorted(
        cls.__name__ for cls in layer_classes if cls.SUPPORTS_BIAS_BALANCING and not cls.FAMILY.balancing_slot
    )
    if mode == "bias_update":
        assert not transient, (
            f"moe_balancing=bias_update trains a routing bias no export carries on {', '.join(transient)}: "
            f"the served model would route on raw gate scores and flip near-tied top-k picks. Use "
            f"moe_balancing=bias_update_transient to accept trainer-only balancing, or none."
        )
    else:
        assert transient, (
            f"moe_balancing=bias_update_transient, but every EP layer here ({', '.join(c.__name__ for c in layer_classes)}) "
            f"carries its bias in checkpoint-exported state — nothing is transient. Use moe_balancing=bias_update."
        )


@pytest.mark.parametrize("config", _EXAMPLES, **_ID)
def test_example_max_length_fits_the_model_context_window(config):
    """A declared sequence budget above the model's own position budget truncates or indexes past RoPE.

    The window is read off the text sub-config, the level a composite (VLM) config keeps it on.
    """
    parsed = parser_for(script_for(config)).parse_yaml_file(str(config))
    declared = [
        (name, parsed_field(parsed, name))
        for name in ("max_length", "max_prompt_length", "max_completion_length")
        if any(hasattr(obj, name) for obj in parsed)
    ]
    if not any(is_bounded_length(value) for _, value in declared):
        return
    decoder = text_config(model_config_for(config, parsed))
    window = next(
        (
            int(getattr(decoder, attr))
            for attr in ("max_position_embeddings", "max_seq_length", "n_positions")
            if is_bounded_length(getattr(decoder, attr, None))
        ),
        None,
    )
    if window is None:
        pytest.skip("model config states no position budget; the tokenizer resolves it at load")
    for name, value in declared:
        if is_bounded_length(value):
            assert value <= window, f"{name}={value} exceeds the model's context window ({window})"


@pytest.mark.parametrize("config", _EXAMPLES, **_ID)
def test_example_parallelism_fits_the_model_shape(config):
    """Expert counts must divide the expert-distribution axes, and TP must divide the head counts.

    Model-dependent, so the production path defers it to the load — where a booked multi-GPU job has already
    been booked and every rank has begun materializing weights.
    """
    parsed, parallelism = build_parallelism_config(config)
    reject_unsupported_model(
        parallelism, model_config_for(config, parsed), sinks_policy=SinksPolicy.NEUTRALIZED, wraps_experts=False
    )


# Knobs the VLM data path refuses outright (sft_rules.reject_unsupported_vlm_sft_options, prepare_vlm_dataset).
_VLM_REFUSED_KNOBS = ("packing", "padding_free", "train_on_last_assistant_only", "interleaved_thinking")


@pytest.mark.parametrize("config", _EXAMPLES, **_ID)
def test_image_declaring_example_asks_for_nothing_the_vlm_path_refuses(config):
    """An example that declares image data takes the VLM data path, which rejects these outright.

    Modality follows the RUN, not the checkpoint (``is_vlm_run``), so a multimodal checkpoint on
    text-only rows is free to pack — that is what the Gemma 4 / Qwen3.5 text recipes rely on. The
    file that cannot run is the other one: an images column plus a knob the VLM path refuses, which
    today only says so once the dataset has been loaded on a booked multi-GPU host.
    """
    raw = raw_config(config)
    if not raw.get("images_field"):
        pytest.skip("example declares no image column")
    refused = [knob for knob in _VLM_REFUSED_KNOBS if raw.get(knob)]
    assert not refused, f"{refused} are refused on the VLM data path, which images_field selects"


@pytest.mark.parametrize("config", _EXAMPLES, **_ID)
def test_example_attention_carries_the_sinks_it_keeps_live(config):
    """A sinks model (GptOss) must not pair live sinks with a sink-dropping attention kernel.

    ``reset_sinks: false`` keeps the pretrained sinks in the softmax; sdpa and the flash builds
    whose kernel exposes no sink argument silently drop them, shifting every training log-prob by
    nats against the served policy. The validator owns the kernel-capability matrix — it is read
    from the installed kernel's signature, so a flash upgrade moves this test with it.
    """
    parsed = parser_for(script_for(config)).parse_yaml_file(str(config))
    attn_implementation = parsed_field(parsed, "attn_implementation")
    if attn_implementation is None:
        return  # unset: the resolver picks a sink-carrying candidate itself
    model_config = model_config_for(config, parsed)
    if not uses_gpt_oss_sinks(model_config):
        return
    if is_flash_attn_implementation(attn_implementation):
        try:
            flash_kernel(attn_implementation)
        except ImportError:
            pytest.skip(f"{attn_implementation} is not built into this image (the recipe targets another GPU)")
    validate_attn_implementation(
        model_config, attn_implementation, SinksPolicy.from_flags(reset_sinks=parsed_field(parsed, "reset_sinks"))
    )


@pytest.mark.parametrize("config", _EXAMPLES, **_ID)
def test_weight_sync_example_keeps_a_pushable_sink_parameter(config):
    """On-policy RL cannot use the flash_attention_2 sink reset: it DELETES the parameter.

    That reset is the one branch of ``_reset_gpt_oss_sinks`` that assigns ``sinks = None`` (every
    other kernel fills with ``dtype.min`` and keeps the slot). The weight sync ships
    ``named_parameters`` only, so nothing is ever pushed for the removed slots and the rollout
    engine keeps generating with the pretrained sinks against a sink-free trainer — permanently
    off-policy, which ``reject_unsyncable_model`` refuses at construction.
    """
    script = script_for(config)
    if not script_call_kwargs(script, "configure_balancing_and_build_callbacks", ("syncs_to_external_generator",)).get(
        "syncs_to_external_generator"
    ):
        return
    parsed = parser_for(script).parse_yaml_file(str(config))
    if parsed_field(parsed, "attn_implementation") != "flash_attention_2" or not parsed_field(parsed, "reset_sinks"):
        return
    assert not uses_gpt_oss_sinks(model_config_for(config, parsed)), (
        "weight-sync RL with the flash_attention_2 reset_sinks reset removes the sinks parameter, so "
        "the sync never pushes it and the generator stays on the pretrained sinks. Set reset_sinks: "
        "false with a sink-carrying implementation (flash_attention_4 on Blackwell; eager or "
        "flex_attention on Hopper)."
    )


def test_a_batch_rebalance_example_exists():
    """The two checks below are parametrized over this list — an empty one would assert nothing."""
    assert _BATCH_REBALANCE_EXAMPLES, "no shipped example sets train_sampling_strategy: batch_rebalance"


@pytest.mark.parametrize("config", _BATCH_REBALANCE_EXAMPLES, **_ID)
def test_batch_rebalance_example_keeps_the_base_loader(config):
    """HF's rebalance sampler shards over every process, so the toolkit refuses it on its DP-sharded loader
    (TP/CP/ETP/PP) when the trainer builds the train loader, after every rank has loaded the model."""
    _, parallelism = build_parallelism_config(config)
    assert not uses_dp_sharded_loader(parallelism, dataset_presharded=False), (
        "this config asks for batch_rebalance on a shape that shards batches over data-parallel replicas, where "
        "the trainer refuses it; drop the line or run pure EP / DDP"
    )


@pytest.mark.parametrize("config", _BATCH_REBALANCE_EXAMPLES, **_ID)
def test_batch_rebalance_example_reads_lengths_its_rows_carry(config, tmp_path):
    """HF builds the sampler from each row's ``processing_class.model_input_names[0]``. A multimodal
    checkpoint's processor lists an image or audio input first, which a text SFT row lacks, so the sampler
    raises at loader build. Built here through HF's own ``_get_train_sampler`` with the processing class the
    entry script loads, on rows carrying the tokenizer's keys; a placeholder checkpoint absent on this host
    is checked through its tiny stand-in."""
    parsed = parser_for(script_for(config)).parse_yaml_file(str(config))
    checkpoint = processing_class_source(parsed_field(parsed, "model_name_or_path"), tmp_path)
    load_cached_config(checkpoint, trust_remote_code=True)  # skips a Hub checkpoint not cached here
    processing_class = load_processing_class(
        checkpoint, trust_remote_code=bool(parsed_field(parsed, "trust_remote_code"))
    )
    tokenizer = resolve_tokenizer(processing_class)
    rows = Dataset.from_list([dict(tokenizer(text)) for text in ("a short row", "a row that runs a little longer")])
    args = SimpleNamespace(
        train_sampling_strategy="batch_rebalance",
        length_column_name="length",
        world_size=1,
        process_index=0,
        gradient_accumulation_steps=1,
        train_batch_size=1,
        dataloader_drop_last=False,
    )
    host = SimpleNamespace(args=args, processing_class=processing_class, train_dataset=rows)
    assert isinstance(Trainer._get_train_sampler(host), BatchRebalanceSampler)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
