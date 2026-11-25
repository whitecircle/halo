"""A saved LoRA adapter as the adapter directory a rollout engine loads by path: the names, expert layout, scaling and
``adapter_config.json`` that adapter sync publishes for a live trainer, read off the adapter's base built without
weights."""

import logging
import os
from pathlib import Path

from halo.checkpoint.adapters import (
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
from halo.checkpoint.tools.offline_export import reject_in_place_conversion, resolve_adapter_base_revision
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
from halo.weight_sync.registry import resolve_weight_sync_client

logger = logging.getLogger(__name__)


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
    records or at ``base_revision``; an unset ``trust_remote_code`` follows the base. Raises before any write: for a
    base whose family a run under ``rollout_adapter_sync`` is refused for on that engine, for an adapter the engine
    cannot serve, with every reason, and for an existing ``output_dir``."""
    reject_partial_checkpoint(adapter_dir)
    reject_in_place_conversion(adapter_dir, output_dir)
    if os.path.lexists(output_dir):
        raise FileExistsError(f"{output_dir} exists: the engine directory is written whole into a new path.")
    config = read_adapter_config(adapter_dir)
    shape = find_expert_lora_adapter_shape(adapter_dir)
    if shape == "expert":
        raise ValueError(
            f"{adapter_dir} holds expert adapters alone, and adapter sync serves native expert LoRA only beside "
            f"attention adapters, whose config carries the rank and scaling the engine reads for both. Train with an "
            f"attention target beside the experts (lora_target_modules: [o_proj, gate_up_proj, down_proj])."
        )
    base_path = config.get("base_model_name_or_path")
    if not base_path:
        raise ValueError(f"{os.path.join(adapter_dir, ADAPTER_CONFIG_FILE)} names no base_model_name_or_path.")
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
    module_names = sorted({key.rpartition(".lora_")[0] for key in state if ".lora_" in key})
    whole = sorted(key for key in state if ".lora_" not in key)

    client_cls = resolve_weight_sync_client(rollout_backend)
    family_refusal = engine_adapter_refusal(rollout_backend, base.config)
    reasons = [family_refusal] if family_refusal is not None else []
    if whole:
        reasons.append(
            f"whole tensors beside the adapters ({', '.join(whole[:3])}): a grown vocabulary's embedding rows or a "
            f"fully trained module, which the engine has no route for"
        )
    expert = resolve_expert_lora_spec(config) if shape == "mixed" else None
    reasons.extend(unservable_adapter_reasons(config, module_names, base, client_cls, expert))
    if reasons:
        raise ValueError(
            f"{adapter_dir} cannot be served as an adapter by the {rollout_backend} engine: {'; '.join(reasons)}."
        )
    wire, engine_config = build_engine_adapter(base, build_adapter_serving(config, module_names), state, experts)
    output = Path(output_dir).absolute()
    published = publish_adapter_directory(output.parent, output.name, wire, engine_config)
    logger.info(f"Wrote {len(wire)} adapter tensors for the {rollout_backend} engine to {published}")
    return published
