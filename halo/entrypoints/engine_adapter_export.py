"""A saved LoRA adapter as the adapter directory a rollout engine loads by path: the names, expert layout, scaling and
``adapter_config.json`` that adapter sync publishes for a live trainer, read off the adapter's base built without
weights."""

import logging
import os
from pathlib import Path
from typing import NamedTuple

from halo.checkpoint.adapters import (
    ExpertAdapterShape,
    find_adapter_weight_file,
    find_expert_lora_adapter_shape,
    is_expert_lora_key,
    read_adapter_config,
    read_adapter_file,
)
from halo.checkpoint.format import ADAPTER_CONFIG_FILE, ADAPTER_WEIGHT_NAMES
from halo.checkpoint.reading import reject_partial_checkpoint
from halo.checkpoint.tools.adapter_merge import adapter_base_trust, load_base_for_adapter
from halo.checkpoint.tools.adapter_stream_merge import resolve_expert_lora_spec
from halo.checkpoint.tools.offline_export import resolve_adapter_base_revision
from halo.model_loading.lazy_safetensors.meta_shell import instantiate_on_meta
from halo.model_loading.model_preparation import (
    resolve_auto_model_class,
    resolve_build_config,
    resolve_text_only_config,
)
from halo.trainers.grpo.rollout.adapter_sync import (
    build_adapter_serving,
    build_engine_adapter,
    engine_adapter_refusal,
    unservable_adapter_reasons,
)
from halo.trainers.grpo.rollout.weight_push import reject_unsyncable_config
from halo.weight_sync.adapter_directory import publish_adapter_directory
from halo.weight_sync.registry import resolve_weight_sync_client, rollout_backends

logger = logging.getLogger(__name__)


class _SavedAdapter(NamedTuple):
    """What the engine checks read off a saved adapter."""

    config: dict
    shape: ExpertAdapterShape | None
    # The modules the PEFT half adapts, as its keys spell them.
    module_names: list[str]


def _engine_reasons(backend: str, base, adapter: _SavedAdapter) -> list[str]:
    """Why ``backend``'s engine cannot serve ``adapter`` on ``base``: a family it serves no adapter of, and what the
    adapters declare that it refuses."""
    family_refusal = engine_adapter_refusal(backend, base.config)
    reasons = [family_refusal] if family_refusal is not None else []
    expert = resolve_expert_lora_spec(adapter.config) if adapter.shape is not None else None
    client_cls = resolve_weight_sync_client(backend)
    reasons.extend(unservable_adapter_reasons(adapter.config, adapter.module_names, base, client_cls, expert))
    return reasons


def _is_served_by(backend: str, base, adapter: _SavedAdapter) -> bool:
    """Whether ``backend``'s engine serves ``adapter`` on ``base``: the family gate of a run under
    ``rollout_adapter_sync`` passes, and no check of :func:`_engine_reasons` refuses it."""
    try:
        reject_unsyncable_config(base.config, backend, adapter_sync=True)
    except ValueError:
        return False
    return not _engine_reasons(backend, base, adapter)


def _remedy(rollout_backend: str, base, adapter: _SavedAdapter, engine_reasons: list[str], held: list[str]) -> str:
    """The path that remains for ``adapter``, which ``rollout_backend``'s engine refuses for ``engine_reasons`` and
    every engine for ``held``: the attention target where only the experts' being alone stops this engine; else the
    engines that serve it as it is, and a merge."""
    if adapter.shape == "expert" and not engine_reasons and len(held) == 1:
        return (
            "Train with an attention target beside the experts (lora_target_modules: [o_proj, gate_up_proj, "
            "down_proj]): this engine serves the expert adapters of this base beside it."
        )
    merged = "merge_peft_adapters.py folds it into its base"
    if adapter.shape is not None:
        merged += (
            " where its streaming merge finds each expert layer stored under its module path as floats, and "
            "merge_expert_lora_on_save: true folds it at the run's save where it does not; nothing folds a delta into "
            "the packed banks frozen_quantized_experts trains over"
        )
    served_by = (
        []
        if held
        else [
            backend
            for backend in rollout_backends()
            if backend != rollout_backend and _is_served_by(backend, base, adapter)
        ]
    )
    if served_by:
        return (
            f"Export it with --rollout_backend {' or '.join(served_by)}, whose engine serves it, or serve it merged: "
            f"{merged}."
        )
    return f"Serve it merged instead: {merged}."


def export_engine_adapter(
    adapter_dir: str,
    output_dir: str,
    *,
    rollout_backend: str,
    base_revision: str | None = None,
    trust_remote_code: bool | None = None,
) -> Path:
    """Writes ``output_dir`` as the adapter directory adapter sync would publish to ``rollout_backend`` for the
    adapters saved at ``adapter_dir``, and returns it. A Hub base is read, config only, at the commit the adapter
    records or at ``base_revision``; an unset ``trust_remote_code`` follows the base. Raises before any write: for an
    existing ``output_dir``, for a base whose family a run under ``rollout_adapter_sync`` is refused for on that
    engine, and for an adapter the engine cannot serve, with every reason and the path that remains."""
    reject_partial_checkpoint(adapter_dir)
    if os.path.lexists(output_dir):
        raise FileExistsError(
            f"{output_dir} exists, and the engine directory is written whole into a new path: name an output_dir "
            f"that does not exist."
        )
    config = read_adapter_config(adapter_dir)
    base_path = config.get("base_model_name_or_path")
    if not base_path:
        raise ValueError(
            f"{os.path.join(adapter_dir, ADAPTER_CONFIG_FILE)} names no base_model_name_or_path: set it to the "
            f"checkpoint the adapter was trained on, a local directory or a Hub id."
        )
    trust_remote_code = adapter_base_trust(adapter_dir, trust_remote_code)
    revision = resolve_adapter_base_revision(base_path, config.get("revision"), base_revision)

    def build_base(path: str, *, excuse_task_head: bool, text_only: bool = False):
        build_config = resolve_build_config(path, revision=revision, trust_remote_code=trust_remote_code)
        return instantiate_on_meta(
            path,
            resolve_auto_model_class(build_config, text_only=text_only),
            resolve_text_only_config(build_config) if text_only else build_config,
            trust_remote_code=trust_remote_code,
            config_only=True,
            revision=revision,
        )

    base = load_base_for_adapter(adapter_dir, base_path, build_base, excuse_task_head=True, log=logger.info).model
    reject_unsyncable_config(base.config, rollout_backend, adapter_sync=True)
    weight_file = find_adapter_weight_file(adapter_dir)
    if weight_file is None:
        raise FileNotFoundError(
            f"{adapter_dir} holds none of the adapter weight files {', '.join(ADAPTER_WEIGHT_NAMES)}."
        )
    tensors = read_adapter_file(weight_file)
    experts = {key: tensor for key, tensor in tensors.items() if is_expert_lora_key(key)}
    state = {key: tensor for key, tensor in tensors.items() if key not in experts}
    adapter = _SavedAdapter(
        config,
        find_expert_lora_adapter_shape(adapter_dir),
        sorted({key.rpartition(".lora_")[0] for key in state if ".lora_" in key}),
    )
    whole = sorted(key for key in state if ".lora_" not in key)

    engine_reasons = _engine_reasons(rollout_backend, base, adapter)
    # What no engine serves, whatever it serves of the family.
    held = []
    if adapter.shape == "expert":
        held.append(
            "expert adapters alone: adapter sync serves native expert LoRA only beside attention adapters, whose "
            "config carries the rank and scaling the engine reads for both"
        )
    if whole:
        held.append(
            f"whole tensors beside the adapters ({', '.join(whole[:3])}): a grown vocabulary's embedding rows or a "
            f"fully trained module, which the engine has no route for"
        )
    if engine_reasons or held:
        raise ValueError(
            f"{adapter_dir} cannot be served as an adapter by the {rollout_backend} engine: "
            f"{'; '.join([*engine_reasons, *held])}. {_remedy(rollout_backend, base, adapter, engine_reasons, held)}"
        )
    wire, engine_config = build_engine_adapter(
        base, build_adapter_serving(config, adapter.module_names), state, experts
    )
    output = Path(output_dir).absolute()
    published = publish_adapter_directory(output.parent, output.name, wire, engine_config)
    logger.info(f"Wrote {len(wire)} adapter tensors for the {rollout_backend} engine to {published}")
    return published
