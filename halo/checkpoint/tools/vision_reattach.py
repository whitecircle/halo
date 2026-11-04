"""Re-attach a multimodal base's vision tower to a text-only export: the trained text weights and any MTP head the
export trained are re-prefixed onto the wrapper layout, the base's other tensors stream back in (its MTP tail, untrained
against this trunk, when the export has none), and the composite config is regrafted."""

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
from halo.model_loading.model_preparation import declared_load_ignores
from halo.models.moe.family_specs import CompositeTextLayout, find_moe_family_spec
from halo.models.mtp.spec import find_mtp_head_spec, stored_tail_keys

logger = logging.getLogger(__name__)


def _declared_text_layout(model_type: str | None) -> CompositeTextLayout | None:
    spec = find_moe_family_spec(model_type)
    return None if spec is None else spec.text_only_layout


def _require_text_only_export(input_dir: str, trust_remote_code: bool) -> AutoConfig:
    config = AutoConfig.from_pretrained(input_dir, trust_remote_code=trust_remote_code)
    text_only_type = str(config.model_type).endswith("_text") or _declared_text_layout(config.model_type) is not None
    if getattr(config, "vision_config", None) is not None or not text_only_type:
        raise ValueError(
            f"{input_dir} is not a text-only export (model_type={config.model_type!r}): it already "
            f"carries the multimodal layout and needs no re-attachment."
        )
    return config


def _require_text_tower_keys(
    input_dir: str, model_type: str, base_text_children: set[str], layout: CompositeTextLayout, trained_tail: set[str]
) -> None:
    """Refuse an export whose keys are not the text-only layout its config declares; a key naming no module of the
    base's text tower lands on no wrapper slot and reloads random-initialized. Header-only."""
    misplaced = [
        key
        for _shard, _reader, key in iter_checkpoint_shard_entries(input_dir)
        if key not in trained_tail
        and not key.startswith(layout.head)
        and not (key.startswith(layout.tower) and key[len(layout.tower) :].split(".", 1)[0] in base_text_children)
    ]
    if not misplaced:
        return
    if any(key.startswith(layout.composite_tower) for key in misplaced):
        raise ValueError(
            f"{input_dir} declares the text-only model_type {model_type!r} but stores its text tower under "
            f"{layout.composite_tower}* (e.g. {misplaced[0]!r}), the multimodal wrapper's layout, so its keys "
            f"do not match its config and re-prefixing them would nest the wrapper prefix. Re-export it "
            f"in the text-only layout: scripts/after_training/convert_to_bf16.py for a full checkpoint, "
            f"merge_peft_adapters.py for an adapter."
        )
    raise ValueError(
        f"{len(misplaced)} key(s) of {input_dir} address no module of the base's text tower "
        f"({layout.composite_tower}{{{', '.join(sorted(base_text_children))}}}; e.g. {misplaced[0]!r}): it is not "
        f"a text-only export of this base."
    )


def _require_the_bases_tower_shape(
    input_dir: str, base_dir: str, base_shapes: dict[str, list[int]], vocab_size: int, layout: CompositeTextLayout
) -> None:
    """Refuse a base of another size in the same family: what streams back from it was built against its own tower, so
    its vision projector or MTP tail would land beside a text tower of another width. A grown vocabulary moves only the
    rows of its embedding and head. Header-only."""
    for _shard, reader, key in iter_checkpoint_shard_entries(input_dir):
        shape, base_shape = reader.get_slice(key).get_shape(), base_shapes.get(layout.to_composite_key(key))
        grown = base_shape is not None and shape[1:] == base_shape[1:] and shape[0] == vocab_size
        if base_shape is not None and shape != base_shape and not grown:
            raise ValueError(
                f"{key!r} of {input_dir} has shape {shape}, and the base checkpoint ({base_dir}) stores it as "
                f"{base_shape}: the base is another model of this family, not the one this export trained from."
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
    # The attribute holding it, as a release's remote config keeps it under its own name and reads ``text_config`` off it.
    held = composite.get_text_config()
    # What loading the export stamped on it, which no run changed and the artifact would carry as a local path.
    text_config._name_or_path, text_config.architectures = held._name_or_path, held.architectures
    setattr(composite, next(name for name, value in vars(composite).items() if value is held), text_config)
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
    layout = _declared_text_layout(composite.model_type) or CompositeTextLayout()

    input_shards = checkpoint_shard_files(input_dir)
    base_shards = checkpoint_shard_files(base_dir)

    # Header-only, ahead of the output directory. The export supersedes the base's text tower only under the wrapper
    # prefix; a base keeping a vendor namespace would be carried whole beside the trained tower, two towers colliding
    # on load.
    base_shapes = {
        key: reader.get_slice(key).get_shape() for _shard, reader, key in iter_checkpoint_shard_entries(base_dir)
    }
    # Both classes: Nemotron-H's composite names no tail where its text class does.
    load_ignores = (*declared_load_ignores(composite), *declared_load_ignores(text_config))
    trained_tail = set(
        stored_tail_keys((key for _shard, _reader, key in iter_checkpoint_shard_entries(input_dir)), load_ignores)
    )
    base_tail = stored_tail_keys(base_shapes, load_ignores)
    superseded = {key for key in base_shapes if key.startswith((layout.composite_tower, layout.composite_head))}
    if trained_tail:
        superseded.update(base_tail)
    base_text_children = {
        key[len(layout.composite_tower) :].split(".", 1)[0]
        for key in superseded
        if key.startswith(layout.composite_tower)
    }
    if not base_text_children:
        raise ValueError(
            f"the base checkpoint ({base_dir}) stores no text tower under {layout.composite_tower}*, so the "
            f"trained export cannot supersede it — its text keys would be carried over beside the "
            f"trained ones. Re-attach onto a base saved in the wrapper layout."
        )
    if len(superseded) == len(base_shapes):
        raise ValueError(
            f"the base checkpoint ({base_dir}) contributes no non-text tensors — nothing to "
            f"re-attach. Its weight layout does not match the expected multimodal wrapper."
        )
    _require_text_tower_keys(input_dir, text_config.model_type, base_text_children, layout, trained_tail)
    _require_the_bases_tower_shape(input_dir, base_dir, base_shapes, text_config.vocab_size, layout)

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
            writer.add(layout.to_composite_key(key), tensor)
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
    stale_tail = [key for key in base_tail if key not in superseded]
    if stale_tail:
        refit = f" Train one with mtp_train_scope: head_only on {output_dir} to match the trunk."
        logger.warning(
            f"The MTP head in {output_dir} is {base_dir}'s {len(stale_tail)} tensor(s), carried as they are: they never "
            f"saw this trunk, so a server drafting with them accepts fewer tokens, with the output distribution "
            f"unchanged.{refit if find_mtp_head_spec(composite) is not None else ''}"
        )
    logger.info(
        f"✓ Re-attached vision tower: {trained} trained tensor(s) re-prefixed, {carried} base "
        f"tensor(s) carried over -> {output_dir} ({json.dumps(composite.architectures)})"
    )
