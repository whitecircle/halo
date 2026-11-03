"""Re-attach a multimodal base's vision tower (and MTP tail) to a text-only export: the trained text weights are
re-prefixed onto the wrapper layout, the base's other tensors stream back in, and the composite config is regrafted."""

from __future__ import annotations

import json
import logging
import os
import shutil

from transformers import AutoConfig

from halo.checkpoint.config_export import finalize_exported_config
from halo.checkpoint.format import DEFAULT_MAX_SHARD_SIZE, PROCESSOR_FILES
from halo.checkpoint.reading import checkpoint_shard_files, iter_checkpoint_shard_entries, iter_checkpoint_tensors
from halo.checkpoint.tools.offline_export import (
    copy_checkpoint_aux_files,
    preflight_resource_warning,
    reject_in_place_conversion,
    resolve_checkpoint_source,
)
from halo.checkpoint.writing import StreamingShardWriter

logger = logging.getLogger(__name__)

TEXT_PREFIX = "model."
WRAPPER_TEXT_PREFIX = "model.language_model."
HEAD_PREFIX = "lm_head."


def _require_text_only_export(input_dir: str, trust_remote_code: bool) -> AutoConfig:
    config = AutoConfig.from_pretrained(input_dir, trust_remote_code=trust_remote_code)
    if getattr(config, "vision_config", None) is not None or not str(config.model_type).endswith("_text"):
        raise ValueError(
            f"{input_dir} is not a text-only export (model_type={config.model_type!r}): it already "
            f"carries the multimodal layout and needs no re-attachment."
        )
    return config


def _require_text_tower_keys(input_dir: str, model_type: str, base_text_children: set[str]) -> None:
    """Refuse an export whose keys are not the text-only layout its config declares; a key naming no module of the
    base's text tower lands on no wrapper slot and reloads random-initialized. Header-only."""
    misplaced = [
        key
        for _shard, _reader, key in iter_checkpoint_shard_entries(input_dir)
        if not key.startswith(HEAD_PREFIX)
        and not (key.startswith(TEXT_PREFIX) and key[len(TEXT_PREFIX) :].split(".", 1)[0] in base_text_children)
    ]
    if not misplaced:
        return
    if any(key.startswith(WRAPPER_TEXT_PREFIX) for key in misplaced):
        raise ValueError(
            f"{input_dir} declares the text-only model_type {model_type!r} but stores its text tower under "
            f"{WRAPPER_TEXT_PREFIX}* (e.g. {misplaced[0]!r}), the multimodal wrapper's layout, so its keys "
            f"do not match its config and re-prefixing them would nest the wrapper prefix. Re-export it "
            f"in the text-only layout: scripts/after_training/convert_to_bf16.py for a full checkpoint, "
            f"merge_peft_adapters.py for an adapter."
        )
    raise ValueError(
        f"{len(misplaced)} key(s) of {input_dir} address no module of the base's text tower "
        f"({WRAPPER_TEXT_PREFIX}{{{', '.join(sorted(base_text_children))}}}; e.g. {misplaced[0]!r}): it is not "
        f"a text-only export of this base."
    )


def _grafted_composite_config(base_dir: str, text_config: AutoConfig, trust_remote_code: bool):
    """The base's composite config with its text sub-config replaced by the trained one."""
    composite = AutoConfig.from_pretrained(base_dir, trust_remote_code=trust_remote_code)
    if getattr(composite, "vision_config", None) is None:
        raise ValueError(
            f"the base checkpoint's config ({base_dir}) declares no vision_config — it is not the "
            f"multimodal checkpoint this export trained from."
        )
    # A text tower grafted into another family's wrapper yields a config that loads but serves incorrect output.
    base_text_type = composite.get_text_config().model_type
    if base_text_type != text_config.model_type:
        raise ValueError(
            f"the base checkpoint ({base_dir}) wraps a {base_text_type!r} text tower, but the export is "
            f"{text_config.model_type!r} — not the multimodal checkpoint this export trained from."
        )
    composite.text_config = text_config
    return composite


def reattach_vision_tower(
    input_dir: str,
    model_id: str,
    output_dir: str,
    *,
    revision: str | None = None,
    trust_remote_code: bool = True,
    max_shard_size: str = DEFAULT_MAX_SHARD_SIZE,
) -> None:
    reject_in_place_conversion(input_dir, output_dir)
    text_config = _require_text_only_export(input_dir, trust_remote_code)
    base_dir = resolve_checkpoint_source(model_id, revision)
    # The writer's close removes every model*.safetensors it did not write, so an output at the base would replace it.
    reject_in_place_conversion(base_dir, output_dir)
    composite = _grafted_composite_config(base_dir, text_config, trust_remote_code)

    input_shards = checkpoint_shard_files(input_dir)
    base_shards = checkpoint_shard_files(base_dir)

    # Header-only, ahead of the output directory. The export supersedes the base's text tower only under the wrapper
    # prefix; a base keeping a vendor namespace would be carried whole beside the trained tower, two towers colliding
    # on load.
    base_keys = [key for _shard, _reader, key in iter_checkpoint_shard_entries(base_dir)]
    superseded = {key for key in base_keys if key.startswith((WRAPPER_TEXT_PREFIX, HEAD_PREFIX))}
    base_text_children = {
        key[len(WRAPPER_TEXT_PREFIX) :].split(".", 1)[0] for key in superseded if key.startswith(WRAPPER_TEXT_PREFIX)
    }
    if not base_text_children:
        raise ValueError(
            f"the base checkpoint ({base_dir}) stores no text tower under {WRAPPER_TEXT_PREFIX}*, so the "
            f"trained export cannot supersede it — its text keys would be carried over beside the "
            f"trained ones. Re-attach onto a base saved in the wrapper layout."
        )
    if len(superseded) == len(base_keys):
        raise ValueError(
            f"the base checkpoint ({base_dir}) contributes no non-text tensors — nothing to "
            f"re-attach. Its weight layout does not match the expected multimodal wrapper."
        )
    _require_text_tower_keys(input_dir, text_config.model_type, base_text_children)

    with StreamingShardWriter(output_dir, max_shard_size, enabled=True) as writer:
        preflight_resource_warning(
            "reattach_vision_tower",
            output_dir,
            disk_bytes=sum(os.path.getsize(s) for s in input_shards + base_shards),
            ram_bytes=writer.max_resident_bytes,
        )
        os.makedirs(output_dir, exist_ok=True)

        trained = 0
        for key, tensor in iter_checkpoint_tensors(input_dir):
            if key.startswith(TEXT_PREFIX):
                key = WRAPPER_TEXT_PREFIX + key[len(TEXT_PREFIX) :]
            writer.add(key, tensor)  # lm_head.* passes through verbatim
            trained += 1

        carried = 0
        for key, tensor in iter_checkpoint_tensors(base_dir, predicate=lambda key: key not in superseded):
            writer.add(key, tensor)
            carried += 1
        writer.close_as_hf_checkpoint()

    # Tokenizer and templates follow the trained export; processor files come from the base, their only source.
    copy_checkpoint_aux_files(input_dir, output_dir)
    for name in PROCESSOR_FILES:
        src = os.path.join(base_dir, name)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(output_dir, name))
    # Written last so the artifact's config matches the wrapper-layout keys on disk.
    composite.save_pretrained(output_dir)
    finalize_exported_config(composite, output_dir, source=base_dir)
    logger.info(
        f"✓ Re-attached vision tower: {trained} trained tensor(s) re-prefixed, {carried} base "
        f"tensor(s) carried over -> {output_dir} ({json.dumps(composite.architectures)})"
    )
