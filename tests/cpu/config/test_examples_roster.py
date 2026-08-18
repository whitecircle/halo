#!/usr/bin/env python
"""Every shipped ``examples/`` YAML must be *runnable*, not merely parseable.

Its sibling ``test_examples_parse.py`` pins the parse layer (unknown key, stale knob, bad
``Literal``). This module pins the contracts that only bite AFTER the parse — the ones that abort a
run minutes in, on a booked multi-GPU host, once the model is already resident:

* the parallelism shape, built through ``parallelism_config_from_args`` at the topology the file's
  own launch comment declares, with the capability flags its entry script passes;
* the rollout timeouts, validated against the NCCL watchdog the launch comment documents;
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

import pytest
import yaml
from transformers import AutoConfig

from src.distributed.expert_parallel.expert_weights import ep_layer_classes_for_config
from src.distributed.loading.peft_setup import split_expert_lora_targets
from src.models.loading.config_levels import text_config
from src.models.loading.tokenizer_setup import is_bounded_length
from src.models.moe_balancing import BIAS_UPDATE_MODES, native_balancing_bias_attrs
from src.models.patches.attention import model_has_sinks, validate_attn_implementation
from src.training.parallelism_args import parallelism_config_from_args
from tests.common.parallelism import make_parallelism_config
from tests.cpu.config.test_examples_parse import (
    _EXAMPLES,
    EXAMPLES_ROOT,
    PROJECT_ROOT,
    TRAINING_ROOT,
    _bf16_capable,  # noqa: F401 — autouse fixture, imported so both modules validate as a training host does
    parsed_field,
    parser_for,
    script_for,
)

# The repository's standard node, assumed for a config whose comments state no launch width. Every
# rank-math rule (NVLink domain, EP group count, DP width) is relative to it.
DEFAULT_GPUS_PER_NODE = 8

_NPROC = re.compile(r"--nproc[-_]per[-_]node[= ](\d+)")
_NNODES = re.compile(r"--nnodes[= ](\d+)")
_NCCL_MINUTES = re.compile(r"DIST_NCCL_TIMEOUT_MINUTES=(\d+)")

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


# Only the rollout methods carry an episode budget to check against the watchdog.
_ROLLOUT_EXAMPLES = [config for config in _EXAMPLES if "episode_timeout" in raw_config(config)]


def launch_topology(config: Path) -> tuple[int, int]:
    """``(world_size, gpus_per_node)`` the example's own launch comment declares.

    Read off the comment rather than guessed: ``ep_size=4`` is one legal dispatch group under
    ``--nproc_per_node=4`` and two racy ones on a full node, so the launch width is what decides
    whether the shape below is accepted. Falls back to a single standard node.
    """
    comments = "\n".join(
        line for line in config.read_text(encoding="utf-8").splitlines() if line.lstrip().startswith("#")
    )
    gpus_per_node = int(match.group(1)) if (match := _NPROC.search(comments)) else DEFAULT_GPUS_PER_NODE
    nodes = int(match.group(1)) if (match := _NNODES.search(comments)) else 1
    return gpus_per_node * nodes, gpus_per_node


def declared_nccl_timeout_minutes(config: Path) -> str | None:
    """``DIST_NCCL_TIMEOUT_MINUTES`` as the example's own instructions set it, if they do."""
    match = _NCCL_MINUTES.search(config.read_text(encoding="utf-8"))
    return match.group(1) if match else None


def script_call_kwargs(script: str, func: str, names: tuple[str, ...]) -> dict:
    """Constant keyword arguments the entry script passes to ``func``.

    The capability flags (``supports_cp``, ``supports_pp``, ``syncs_to_external_generator``, ...)
    live in the script's own call, so reading them there keeps this test honest when a script gains
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


def build_parallelism_config(config: Path):
    """The ``ParallelismConfig`` this example would build on its declared launch topology.

    Goes through ``parallelism_config_from_args`` — the factory the entry scripts call — so the
    trainer-capability rejections (CP/PP on a script that supports neither) and every
    ``__post_init__`` rank rule are the ones under test. The world-size seams are the three
    ``src.distributed.runtime`` readers the module imported; a CPU tier has no launcher to report them.
    """
    script = script_for(config)
    parsed = parser_for(script).parse_yaml_file(str(config))
    dist_args = next(obj for obj in parsed if type(obj).__name__ == "DistributedArguments")
    model_config = next(obj for obj in parsed if type(obj).__name__ == "ModelConfig")
    flags = script_call_kwargs(
        script,
        "init_training_script",
        ("supports_cp", "supports_pp", "allow_low_precision", "supports_init_from_scratch"),
    )
    # The peel reads the model's own config; where that is unreachable the expert-LoRA rejections
    # (PP and Expert-TP both refuse it) go unchecked, exactly like the model-dependent cases below.
    expert_lora = (
        split_expert_lora_targets(model_config)
        if cached_model_config(model_config.model_name_or_path) is not None
        else None
    )
    world_size, gpus_per_node = launch_topology(config)
    with pytest.MonkeyPatch.context() as patcher:
        for name, value in (
            ("get_global_world_size", world_size),
            ("get_local_world_size", gpus_per_node),
            ("get_global_rank", 0),
        ):
            patcher.setattr(f"src.distributed.parallelism_config.{name}", lambda _value=value: _value)
        return parsed, parallelism_config_from_args(
            dist_args,
            expert_lora=expert_lora,
            supports_cp=flags.get("supports_cp", True),
            supports_pp=flags.get("supports_pp", True),
            **{k: v for k, v in flags.items() if k in ("allow_low_precision", "supports_init_from_scratch")},
        )


def cached_model_config(reference: str):
    """The model's real config from the local HF cache, or ``None`` when it is not there.

    ``local_files_only`` keeps this tier offline: a CI runner without the weights cache skips the
    config-dependent checks instead of hanging on the hub, and a host-local checkpoint path
    (``/mnt/...``) that does not exist here skips the same way.
    """
    try:
        return AutoConfig.from_pretrained(reference, local_files_only=True, trust_remote_code=True)
    except Exception:
        return None


def model_config_for(config: Path, parsed: tuple):
    """The example's resolved model config, or a skip when this host cannot resolve it."""
    reference = parsed_field(parsed, "model_name_or_path")
    resolved = cached_model_config(reference)
    if resolved is None:
        pytest.skip(f"model {reference!r} is not in the local HF cache; config-dependent checks need it")
    return resolved


def test_examples_declare_their_launch_width():
    """A multi-node example must state its launch line, or the topology assumed here is fiction."""
    undeclared = [
        config.relative_to(PROJECT_ROOT).as_posix()
        for config in _EXAMPLES
        if raw_config(config).get("ep_scope") == "global" and not _NPROC.search(config.read_text(encoding="utf-8"))
    ]
    assert not undeclared, (
        "these configs declare cross-domain EP but no --nproc_per_node/--nnodes launch line, so "
        "their rank math cannot be checked:\n  " + "\n  ".join(undeclared)
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
def test_example_does_not_claim_prefetch_it_cannot_get(config):
    """``enable_prefetch`` on a single-server config is a knob the trainer turns straight back off.

    Prefetch overlaps rollout collection with training, which needs a SECOND engine to serve while
    one syncs weights — so the trainer disables it (with a warning) whenever no
    ``rollout_server_configs`` list is present. A config shipping ``enable_prefetch: true`` beside a
    lone ``rollout_server_url`` therefore describes a run nobody gets, and the reader has to know the
    trainer's override to tell what it actually does. Its DEFAULT is ``true``, so honesty here means
    writing ``false``, not deleting the line.
    """
    raw = raw_config(config)
    if "enable_prefetch" not in raw or raw.get("rollout_server_configs"):
        return
    assert raw["enable_prefetch"] is False, (
        "this config declares one rollout_server_url, where the trainer force-disables prefetch and "
        "warns; set enable_prefetch: false, or give it a rollout_server_configs list of 2+ servers"
    )


@pytest.mark.parametrize("config", _EXAMPLES, **_ID)
def test_example_thinking_budget_has_a_server_side_parser(config):
    """A thinking budget the serving engine cannot enforce is a 400 on EVERY rollout, not a no-op.

    ``rollout_max_thinking_tokens`` rides on vLLM's ``thinking_token_budget``, which the server only
    accepts with a reasoning parser loaded — 0.26.0 registers ``qwen3`` and ``gemma4`` among others,
    while gpt-oss takes the baked ``openai_gptoss`` plugin (its built-in entry is commented out) —
    and only with Model Runner V2 off. Sending it anyway errors every episode
    (``episode/error_rate`` 1) with nothing raised at config time, so the file's own launch
    instructions have to name the parser flag that makes the knob real.
    """
    if raw_config(config).get("rollout_max_thinking_tokens") is None:
        return
    comments = "\n".join(
        line for line in config.read_text(encoding="utf-8").splitlines() if line.lstrip().startswith("#")
    )
    assert "--reasoning-parser" in comments, (
        "this config sends a thinking budget but its launch instructions name no --reasoning-parser; "
        "vLLM refuses thinking_token_budget with a 400 on every rollout without one"
    )


@pytest.mark.parametrize("config", _EXAMPLES, **_ID)
def test_ep_example_declares_the_reentrant_checkpointing_it_runs(config):
    """Under EP the trainer FORCES ``use_reentrant: True``; a config saying ``false`` describes fiction.

    CP's and EP's collectives do not survive non-reentrant recompute, so the mixin overwrites the
    kwarg before gradient checkpointing is enabled — warning when it is explicitly ``false``. Both
    spellings therefore run identically, so a tree carrying both reads as two contradicting
    comments. Pipeline parallelism is the opposite rule (non-reentrant REQUIRED, reentrant raises),
    so it is excluded here rather than folded in.
    """
    raw = raw_config(config)
    if (raw.get("expert_parallel_size") or 1) <= 1 or (raw.get("pipeline_parallel_size") or 1) > 1:
        return
    declared = (raw.get("gradient_checkpointing_kwargs") or {}).get("use_reentrant")
    assert declared is not False, (
        "expert parallelism force-promotes use_reentrant to True (src/trainers/mixins/base.py), so this "
        "config's use_reentrant: false is silently overridden — write true, or drop the kwarg"
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
        if type(obj).__name__ == "AsyncTrainingConfig"
    )
    async_config.get_rollout_config()


def test_some_example_model_is_cached():
    """Guards the config-dependent cases below: all-skipped would look green while asserting nothing."""
    cached = [
        config
        for config in _EXAMPLES
        if cached_model_config(raw_config(config).get("model_name_or_path", "")) is not None
    ]
    assert cached, (
        "no example's model is in the local HF cache, so every config-dependent check below skipped. "
        "Populate HF_HOME (configs only) before trusting this file's result."
    )


@pytest.mark.parametrize("config", _EXAMPLES, **_ID)
def test_example_balancing_mode_matches_the_family_export_contract(config):
    """``bias_update`` only where the family's bias EXPORTS; ``bias_update_transient`` only where it does not.

    Both directions raise at run start. The verdict is the EP layer class's declared native slot
    (``_NATIVE_BALANCING_BIAS_ATTR``, read through ``native_balancing_bias_attrs`` so a hub
    respelling counts), resolved off the checkpoint's own ``model_type`` — never a family name.
    The remaining run-start raise, for a model whose routers carry no bias at all, needs the loaded
    modules and stays with the callback.
    """
    parsed = parser_for(script_for(config)).parse_yaml_file(str(config))
    mode = parsed_field(parsed, "moe_balancing")
    if mode not in BIAS_UPDATE_MODES:
        return
    layer_classes = ep_layer_classes_for_config(model_config_for(config, parsed))
    transient = sorted(
        cls.__name__ for cls in layer_classes if cls._supports_bias_balancing and not native_balancing_bias_attrs(cls)
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

    Model-dependent, so the production path defers it to the load — where an 8×B300 job has already
    been booked and every rank has begun materializing weights.
    """
    parsed, parallelism = build_parallelism_config(config)
    parallelism.validate_against_model_config(model_config_for(config, parsed))


# Knobs the VLM data path refuses outright (sft.py::_prepare_vlm_data, prepare_vlm_dataset).
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
    if not model_has_sinks(model_config):
        return
    validate_attn_implementation(model_config, attn_implementation, sinks_reset=parsed_field(parsed, "reset_sinks"))


@pytest.mark.parametrize("config", _EXAMPLES, **_ID)
def test_weight_sync_example_keeps_a_pushable_sink_parameter(config):
    """On-policy RL cannot use the flash_attention_2 sink reset: it DELETES the parameter.

    That reset is the one branch of ``_reset_gpt_oss_sinks`` that assigns ``sinks = None`` (every
    other kernel fills with ``dtype.min`` and keeps the slot). The weight sync ships
    ``named_parameters`` only, so nothing is ever pushed for the removed slots and the rollout
    engine keeps generating with the pretrained sinks against a sink-free trainer — permanently
    off-policy, which ``validate_weight_sync_support`` refuses at construction.
    """
    script = script_for(config)
    if not script_call_kwargs(script, "build_training_callbacks", ("syncs_to_external_generator",)).get(
        "syncs_to_external_generator"
    ):
        return
    parsed = parser_for(script).parse_yaml_file(str(config))
    if parsed_field(parsed, "attn_implementation") != "flash_attention_2" or not parsed_field(parsed, "reset_sinks"):
        return
    assert not model_has_sinks(model_config_for(config, parsed)), (
        "weight-sync RL with the flash_attention_2 reset_sinks reset removes the sinks parameter, so "
        "the sync never pushes it and the generator stays on the pretrained sinks. Set reset_sinks: "
        "false with a sink-carrying implementation (flash_attention_4 on Blackwell; eager or "
        "flex_attention on Hopper)."
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
