#!/usr/bin/env python
"""Every shipped ``examples/`` YAML must be runnable: its parallelism shape at the declared launch width, rollout
timeouts against the documented watchdog, the allocator setting, and against the model's cached config the
export, context-window and shape contracts. Checks run through the production seam."""

import ast
import math
import re
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from datasets import Dataset
from transformers import Trainer
from transformers.trainer_pt_utils import BatchRebalanceSampler

from halo.configs.parallelism_arguments import parallelism_config_from_args
from halo.data.collators.collator_rules import evaluation_batch_size
from halo.distributed.expert_parallel.deepep.capacity import (
    ep_dispatch_capacity,
    padded_wire_hidden,
    reject_oversized_dispatch,
)
from halo.distributed.expert_parallel.families.roster import ep_layer_classes_for_config
from halo.environments.rollout_config import resolve_reasoning_end_ids
from halo.model_loading.parallel.load_rules import reject_unsupported_model
from halo.model_loading.parallel.peft_setup import split_expert_lora_targets
from halo.model_loading.tokenizer_setup import load_processing_class, resolve_tokenizer
from halo.models.attention.gpt_oss_sinks import SinksPolicy, uses_gpt_oss_sinks
from halo.models.attention.selection import flash_kernel, is_flash_attn_implementation, validate_attn_implementation
from halo.models.config_access import text_config
from halo.models.context_window import bounded_row_length, is_bounded_length
from halo.models.moe.balancing import BIAS_UPDATE_MODES
from halo.models.moe.config_fields import has_routed_experts, resolve_router_topk
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
_VISIBLE_DEVICES = re.compile(r"CUDA_VISIBLE_DEVICES=([\d,]+)")
_ACCELERATE_CONFIG = re.compile(r"--config_file\s+(\S+)")
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
    """No network from this tier: offline turns a cache miss into an immediate error the caller skips on."""
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")


def raw_config(config: Path) -> dict:
    return yaml.safe_load(config.read_text(encoding="utf-8")) or {}


def comment_block(config: Path) -> str:
    return "\n".join(line for line in config.read_text(encoding="utf-8").splitlines() if line.lstrip().startswith("#"))


def launch_commands(config: Path) -> list[str]:
    """Comment lines naming an entry script or ``halo launch``, line continuations folded in. Keyed on the script,
    so prose naming ``torchrun`` is no command."""
    folded = _COMMENT_CONTINUATION.sub(" ", comment_block(config))
    return [line for line in folded.splitlines() if "scripts/training/" in line or _HALO_LAUNCH.search(line)]


_ROLLOUT_EXAMPLES = [config for config in _EXAMPLES if "episode_timeout" in raw_config(config)]

# Their launch comments are the only place the server-side half of the contract is written down, so it is checked as
# part of the file.
_WEIGHT_SYNC_EXAMPLES = [config for config in _EXAMPLES if script_for(config) in ("grpo/online.py", ENV_GRPO_SCRIPT)]
_BATCH_REBALANCE_EXAMPLES = [
    config for config in _EXAMPLES if raw_config(config).get("train_sampling_strategy") == "batch_rebalance"
]

# The tiny family each host-local placeholder checkpoint holds. Where the placeholder is absent, a check reading only
# the processing class runs on that family's tiny checkpoint with an offline tokenizer; a release shipping a processor
# has no such stand-in.
_PLACEHOLDER_FAMILIES = {"/mnt/models/DeepSeek-V4-Flash-BF16": "deepseek_v4"}


def launch_topology(config: Path) -> tuple[int, int]:
    """``(world_size, gpus_per_node)`` the launch comment declares, falling back to a single standard node.
    ``ep_size=4`` is one legal group under ``--nproc_per_node=4`` and two racy ones on a full node, so the width
    decides."""
    comments = comment_block(config)
    gpus_per_node = int(match.group(1)) if (match := _NPROC.search(comments)) else DEFAULT_GPUS_PER_NODE
    nodes = int(match.group(1)) if (match := _NNODES.search(comments)) else 1
    return gpus_per_node * nodes, gpus_per_node


def declared_nccl_timeout_minutes(config: Path) -> str | None:
    """``DIST_NCCL_TIMEOUT_MINUTES`` as the example's own instructions set it, if they do."""
    match = _NCCL_MINUTES.search(config.read_text(encoding="utf-8"))
    return match.group(1) if match else None


def script_call_kwargs(script: str, func: str, names: tuple[str, ...]) -> dict:
    """Constant keyword arguments the entry script passes to ``func``, read from its call so a hand-kept table
    cannot go stale."""
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
    """Every trainer class the entry script can hand ``init_training_script``: the classes named in ``trainer_cls=``
    or in the assignment it reads."""
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


def training_arguments(parsed: tuple):
    return next(obj for obj in parsed if hasattr(obj, "per_device_train_batch_size"))


def build_parallelism_config(config: Path, *, train_batch: int | None = None):
    """The ``ParallelismConfig`` this example builds on its declared topology, through
    ``parallelism_config_from_args``, its training arguments declaring the token budget. The world-size seams are the
    three ``halo.runtime.ranks`` readers, since a CPU tier has no launcher. ``train_batch`` overrides the batch."""
    script = script_for(config)
    parsed = parser_for(script).parse_yaml_file(str(config))
    if train_batch is not None:
        training_arguments(parsed).per_device_train_batch_size = train_batch
    dist_args = next(obj for obj in parsed if type(obj).__name__ == "ParallelismArguments")
    model_config = next(obj for obj in parsed if type(obj).__name__ == "ModelArguments")
    flags = script_call_kwargs(script, "init_training_script", ("supports_init_from_scratch",))
    # Where the model's config is unreachable, the expert-LoRA rejections go unchecked, like the model-dependent cases
    # below.
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
            parallelism_config_from_args(
                dist_args,
                trainer_cls=trainer_cls,
                training_config=training_arguments(parsed),
                expert_lora=expert_lora,
                **flags,
            )
            for trainer_cls in script_trainer_classes(script)
        ]
        return parsed, built[0]


def model_config_for(config: Path, parsed: tuple):
    """The example's model config from the local HF cache, or a skip when this host cannot resolve it."""
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
    """On dense runs the setting lowers peak reserved memory at unchanged throughput; under EP it costs throughput.
    Other methods and a MoE run without EP are unmeasured, so neither rule binds them. Measurements:
    ``_ALLOCATOR_DOC``."""
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
    """A racy single-domain multi-group EP topology, an unlisted axis combination or an unsupported CP/PP request
    would otherwise burn a booked host at step 0."""
    build_parallelism_config(config)


@pytest.mark.parametrize("config", _EXAMPLES, **_ID)
def test_example_thinking_budget_has_a_server_side_parser(config):
    """``rollout_max_thinking_tokens`` rides on vLLM's ``thinking_token_budget``, which needs a reasoning parser
    loaded, so the launch instructions must name its flag. Without it every episode errors with a 400 and nothing
    raises at config time."""
    if raw_config(config).get("rollout_max_thinking_tokens") is None:
        return
    assert "--reasoning-parser" in comment_block(config), (
        "this config sends a thinking budget but its launch instructions name no --reasoning-parser; "
        "vLLM refuses thinking_token_budget with a 400 on every rollout without one"
    )


@pytest.mark.parametrize("config", ENV_GRPO_EXAMPLES, **_ID)
def test_vllm_example_names_a_reasoning_close_its_tokenizer_writes(config):
    """A marker the tokenizer does not write (``</think>`` on Gemma 4 or gpt-oss) only warns at startup, and every
    forced close then trains with the episode's advantage. SGLang enforces no budget, so forces no close to find."""
    parsed = parser_for(script_for(config)).parse_yaml_file(str(config))
    if parsed_field(parsed, "rollout_backend") == "sglang":
        return
    tokenizer = load_cached_tokenizer(parsed_field(parsed, "model_name_or_path"), trust_remote_code=True)
    resolve_reasoning_end_ids(tokenizer, parsed_field(parsed, "rollout_reasoning_end_token"))


def test_a_weight_sync_example_exists():
    assert _WEIGHT_SYNC_EXAMPLES and ENV_GRPO_EXAMPLES, "no shipped example drives a rollout engine"


@pytest.mark.parametrize("config", ENV_GRPO_EXAMPLES, **_ID)
def test_sampled_token_example_names_the_server_flag(config):
    """A server that returns no ids is no error: every turn re-tokenizes a chat-template re-render, and the vLLM
    importance-sampling correction goes silently with it. Read through the parser, since the file usually leaves
    the knob at its default. Only vLLM takes a flag."""
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
    """The rollout prompt renders server-side, so a template forced on the trainer alone diverges in the context the
    policy is sampled under: the gpt-oss built-in serves every tool-carrying turn quoted-and-escaped, and nothing
    raises."""
    if not raw_config(config).get("chat_template"):
        return
    comments = comment_block(config)
    assert "--chat-template" in comments or "CHAT_TEMPLATE" in comments, (
        "this config forces its own chat template on the trainer but its launch instructions never "
        "pass the same file to the server, which then serves the checkpoint's built-in template"
    )


@pytest.mark.parametrize("config", _WEIGHT_SYNC_EXAMPLES, **_ID)
def test_moe_weight_sync_example_names_the_triton_moe_backend(config):
    """An auto-selected backend holds the experts in a repacked layout while the push writes the bf16 unpacked one.
    The engine accepts the update and the served policy stops being the trained one."""
    parsed = parser_for(script_for(config)).parse_yaml_file(str(config))
    if not ep_layer_classes_for_config(model_config_for(config, parsed)):
        return
    comments = comment_block(config)
    assert "--moe-backend triton" in comments or "--moe-runner-backend triton" in comments, (
        "this config syncs weights into an engine serving a MoE model but its launch instructions "
        "name no triton MoE backend; the auto-selected ones repack the expert weights and every "
        "update after the first lands corrupted"
    )


@pytest.mark.parametrize("config", _EXAMPLES, **_ID)
def test_moe_example_declares_the_reentrant_checkpointing_it_runs(config):
    """CP's and EP's collectives do not survive non-reentrant recompute and a MoE routes in its recompute, so the
    mixin forces ``use_reentrant: True``; a config saying ``false`` contradicts what runs. Pipeline parallelism
    is the opposite rule and is excluded."""
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


@pytest.mark.parametrize("config", _EXAMPLES, **_ID)
def test_example_launch_lines_fit_the_config(config):
    """The header is the run's only launch instruction, so each line names this config's script, under a launcher and
    at a width its axes take. Accelerate's FSDP2 wrap cannot flatten a 4-bit base."""
    raw = raw_config(config)
    commands = launch_commands(config)
    assert commands, "no comment line launches this config"
    for line in commands:
        assert f"scripts/training/{script_for(config)}" in line, f"names another entry script: {line}"
        nproc = _NPROC.search(line)
        if uses_expert_parallelism(raw):
            assert nproc, f"expert parallelism needs torchrun's width: {line}"
        devices = _VISIBLE_DEVICES.search(line)
        if devices and nproc:
            assert len(devices[1].split(",")) == int(nproc[1]), f"exposes another GPU count than it launches: {line}"
        if accelerate_config := _ACCELERATE_CONFIG.search(line):
            distributed_type = yaml.safe_load((REPO_ROOT / accelerate_config[1]).read_text())["distributed_type"]
            assert not (raw.get("load_in_4bit") and distributed_type == "FSDP"), line


def eval_forward_rows(parsed: tuple, parallelism, trainer_cls: type) -> int:
    """Rows one evaluation forward presents to a rank: the eval batch of ``trainer_cls``'s rows, one microbatch's under
    pipeline parallelism."""
    training = training_arguments(parsed)
    rows = trainer_cls.FORWARD_ROWS_PER_EXAMPLE * evaluation_batch_size(training)
    if parallelism.pp_size > 1:
        rows /= parallelism.pp_microbatch_count(training.gradient_accumulation_steps)
    return math.ceil(rows)


@pytest.mark.parametrize("config", _EXAMPLES, **_ID)
def test_example_eval_batch_fits_the_dispatch_budget(config):
    """The load gate sizes a forward off the training batch, and an evaluation forward dispatches through the same
    wire index and, across NVLink domains, the same Gin ceiling. An eval batch past either fails at the first eval."""
    parsed, parallelism = build_parallelism_config(config)
    row_length = bounded_row_length(*parsed)
    if parallelism.ep_size <= 1 or row_length is None:
        pytest.skip("no expert dispatch, or a run that declares no row length")
    decoder = text_config(model_config_for(config, parsed))
    for trainer_cls in script_trainer_classes(script_for(config)):
        tokens = eval_forward_rows(parsed, parallelism, trainer_cls) * row_length // parallelism.cp_size
        reject_oversized_dispatch(
            ep_dispatch_capacity(tokens),
            num_topk=resolve_router_topk(decoder),
            padded_hidden=padded_wire_hidden(decoder.hidden_size),
            spans_domains=parallelism.requires_rdma,
        )


def test_a_train_batch_past_the_dispatch_budget_is_refused():
    """Against the load gate the shape checks reach: without a declared budget it would pass any batch."""
    config = EXAMPLES_ROOT / "sft/gemma4/gemma4-26b-a4b-ultrachat-ep.yaml"
    parsed, parallelism = build_parallelism_config(config, train_batch=4)
    with pytest.raises(ValueError, match="32-bit wire-index"):
        reject_unsupported_model(
            parallelism, model_config_for(config, parsed), sinks_policy=SinksPolicy.NEUTRALIZED, wraps_experts=False
        )


def test_an_eval_batch_counts_one_microbatch_under_pipeline_parallelism():
    config = EXAMPLES_ROOT / "sft/gptoss/gptoss-20b-multinode-pp2-ep8.yaml"
    parsed, parallelism = build_parallelism_config(config)
    training = training_arguments(parsed)
    assert parallelism.pp_size == 2 and training.per_device_eval_batch_size == 8
    assert eval_forward_rows(parsed, parallelism, script_trainer_classes(script_for(config))[0]) == 1


def test_a_racy_ep_topology_is_still_rejected():
    with pytest.raises(ValueError, match="dispatch groups"):
        make_parallelism_config(world_size=8, gpus_per_node=8, ep_size=4)


def test_a_rollout_example_exists():
    assert _ROLLOUT_EXAMPLES, "no shipped example declares episode_timeout"


@pytest.mark.parametrize("config", _ROLLOUT_EXAMPLES, **_ID)
def test_rollout_timeouts_fit_the_documented_nccl_watchdog(config, monkeypatch):
    """``episode_timeout`` at or above the NCCL timeout is refused only at training start, after every rank loaded
    the model. An example raising it documents ``DIST_NCCL_TIMEOUT_MINUTES`` in its launch line."""
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
    """Both directions raise at run start. The verdict is the family spec's ``balancing_slot``, off the checkpoint's
    own ``model_type``."""
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
    """Above the model's position budget a sequence truncates or indexes past RoPE. The window is read off the text
    sub-config, where a composite config keeps it."""
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
    """Model-dependent, so the production path defers it to the load, after a multi-GPU job is booked and every rank
    has begun materializing weights."""
    parsed, parallelism = build_parallelism_config(config)
    reject_unsupported_model(
        parallelism, model_config_for(config, parsed), sinks_policy=SinksPolicy.NEUTRALIZED, wraps_experts=False
    )


# Knobs the VLM data path refuses outright (sft_rules.reject_unsupported_vlm_sft_options, prepare_vlm_dataset).
_VLM_REFUSED_KNOBS = ("packing", "padding_free", "train_on_last_assistant_only", "interleaved_thinking")


@pytest.mark.parametrize("config", _EXAMPLES, **_ID)
def test_image_declaring_example_asks_for_nothing_the_vlm_path_refuses(config):
    """Modality follows the run (``is_vlm_run``), so a multimodal checkpoint on text-only rows may pack; an images
    column plus a knob the VLM path refuses cannot run."""
    raw = raw_config(config)
    if not raw.get("images_field"):
        pytest.skip("example declares no image column")
    refused = [knob for knob in _VLM_REFUSED_KNOBS if raw.get(knob)]
    assert not refused, f"{refused} are refused on the VLM data path, which images_field selects"


@pytest.mark.parametrize("config", _EXAMPLES, **_ID)
def test_example_attention_carries_the_sinks_it_keeps_live(config):
    """``reset_sinks: false`` keeps the sinks in the softmax, and kernels exposing no sink argument silently drop
    them, shifting every log-prob against the served policy. The validator reads the kernel's signature, so a
    flash upgrade moves this test with it."""
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
    """The flash_attention_2 reset assigns ``sinks = None`` and the sync ships ``named_parameters`` only, so the
    engine keeps the pretrained sinks against a sink-free trainer, permanently off-policy.
    ``reject_unsyncable_model`` refuses that at construction."""
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
    assert _BATCH_REBALANCE_EXAMPLES, "no shipped example sets train_sampling_strategy: batch_rebalance"


@pytest.mark.parametrize("config", _BATCH_REBALANCE_EXAMPLES, **_ID)
def test_batch_rebalance_example_keeps_the_base_loader(config):
    """HF's rebalance sampler shards over every process, so the toolkit refuses it on its DP-sharded loader
    (TP/CP/ETP/PP), after every rank has loaded the model."""
    _, parallelism = build_parallelism_config(config)
    assert not uses_dp_sharded_loader(parallelism, dataset_presharded=False), (
        "this config asks for batch_rebalance on a shape that shards batches over data-parallel replicas, where "
        "the trainer refuses it; drop the line or run pure EP / DDP"
    )


@pytest.mark.parametrize("config", _BATCH_REBALANCE_EXAMPLES, **_ID)
def test_batch_rebalance_example_reads_lengths_its_rows_carry(config, tmp_path):
    """HF builds the sampler from ``processing_class.model_input_names[0]``; a multimodal processor lists an image
    or audio input first, which a text row lacks, so the sampler raises at loader build."""
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
