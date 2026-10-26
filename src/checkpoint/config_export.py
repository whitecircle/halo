"""What an exported ``config.json`` must contain, and the one finalizer every save path ends with.

transformers' serialization is not the artifact contract: a vendor class may drop ``model_type``, a pinned
rollout server may need flat legacy keys, and a remote-code family may need the source repo's own schema.
"""

from __future__ import annotations

import copy
import json
import os
import shutil
import tempfile
from collections.abc import Mapping

from transformers.dynamic_module_utils import custom_object_save, get_relative_imports
from transformers.utils import CONFIG_NAME, cached_file

from src.checkpoint.model_card import tag_exported_model_card
from src.checkpoint.reading import read_checkpoint_config
from src.checkpoint.weights_source import config_schema_source
from src.log import info_logger
from src.models.config_access import config_export_ready
from src.models.model_classes import remote_code_model_class, transformers_model_class
from src.models.moe.family_specs import find_moe_family_spec

logger = info_logger(__name__)

# A key the serialized config does not spell, distinct from one spelled as JSON ``null``.
_CONFIG_KEY_ABSENT = object()


def write_config_json(output_dir: str, payload: dict) -> None:
    """Write ``output_dir``'s ``config.json`` in the one form every toolkit rewrite emits, so a re-export
    diffs only where the content changed."""
    with open(os.path.join(output_dir, CONFIG_NAME), "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def hf_architecture_name(model) -> str | None:
    """The class name ``config.architectures`` should carry; ``None`` for a non-HF carrier.

    Re-derived because a task-head swap invalidates the hub's class, and every ``architectures[0]``-keyed
    consumer would serve the wrong head.
    """
    model_class = transformers_model_class(model)
    return None if model_class is None else model_class.__name__


def restore_model_type(config, output_dir: str) -> None:
    """Put the live ``model_type`` back into a ``config.json`` that lost it: ``to_dict`` reads it off the
    class, which some vendor config classes (Bailing/Ling) leave undeclared."""
    live = getattr(config, "model_type", "")
    if not live:
        logger.warning(
            f"restore_model_type: the live config carries no model_type — "
            f"{os.path.join(output_dir, CONFIG_NAME)} stays family-less and every model-type-keyed reader "
            f"(shard merge, hub renames) will see no family. Re-derive the checkpoint from its hub source."
        )
        return
    payload = read_checkpoint_config(output_dir, missing_ok=True)
    if payload is None or payload.get("model_type"):
        return
    payload["model_type"] = live
    write_config_json(output_dir, payload)


def flatten_per_layer_config(section: dict, legacy_keys: Mapping[str, tuple[str, str]]) -> bool:
    """Rewrite one serialized section's ``per_layer_config`` into the family's flat legacy keys
    (``MoEFamilySpec.served_flat_config_keys``), in place; return whether it changed.

    Each flat key takes its field's value on the layers of its layer type. ``per_layer_config`` is then
    dropped, since a 5.14-line reader refuses it and a 5.16 reader rebuilds it; a config the flat keys
    cannot express is refused.
    """
    overrides = section.get("per_layer_config")
    if overrides is None:
        return False
    layer_types = section["layer_types"]
    overrides = {int(index): fields for index, fields in overrides.items()}
    for legacy_key, (layer_type, field) in legacy_keys.items():
        values = {
            overrides.get(index, {}).get(field, section[field])
            for index, this_type in enumerate(layer_types)
            if this_type == layer_type
        }
        if len(values) != 1:
            raise ValueError(
                f"config field {field!r} takes {sorted(values, key=str)} across the {layer_type!r} layers; "
                f"the flat {legacy_key!r} key the rollout server reads holds one value, so this "
                f"per_layer_config cannot be exported losslessly."
            )
        section[legacy_key] = values.pop()
    expressible = set(legacy_keys.values())
    for index, fields in overrides.items():
        for field, value in fields.items():
            if (layer_types[index], field) not in expressible and value != section.get(field):
                raise ValueError(
                    f"per_layer_config overrides {field!r} on layer {index} ({layer_types[index]!r}) to "
                    f"{value!r}, which no flat legacy key of this family expresses; dropping the "
                    f"override would silently change the served model."
                )
    del section["per_layer_config"]
    return True


def export_legacy_per_layer_config(output_dir: str) -> None:
    """Flatten ``per_layer_config`` in ``output_dir``'s ``config.json`` for every family declaring
    ``served_flat_config_keys``, per section, so a composite config's text tower rewrites too.

    transformers 5.16 dropped the flat fields the pinned rollout servers (transformers 5.14 line) read.
    """
    payload = read_checkpoint_config(output_dir, missing_ok=True)
    if payload is None:
        return
    sections = [payload, *(v for v in payload.values() if isinstance(v, dict) and "model_type" in v)]
    changed = False
    for section in sections:
        spec = find_moe_family_spec(section.get("model_type"))
        if spec and spec.served_flat_config_keys:
            changed |= flatten_per_layer_config(section, spec.served_flat_config_keys)
    if changed:
        write_config_json(output_dir, payload)


def _source_config_payload(source: str) -> dict | None:
    """The source checkpoint's own ``config.json`` when it is a servable schema, else ``None`` with a
    warning: an unreadable source, or one without ``auto_map``, has nothing to carry."""
    try:
        config_file = cached_file(source, CONFIG_NAME)
        with open(str(config_file), encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, ValueError) as error:  # unreachable or unparseable; anything else is a bug
        logger.warning(
            f"export_source_config_schema: could not read {CONFIG_NAME} from source {source!r} "
            f"({type(error).__name__}: {error}) — the export keeps transformers' native schema, "
            f"which the pinned rollout servers cannot parse for this family."
        )
        return None
    if not payload.get("auto_map"):
        logger.warning(
            f"export_source_config_schema: source {source!r} declares no auto_map, so it carries no "
            f"servable schema to hand on — the export keeps transformers' native one, which the "
            f"pinned rollout servers cannot parse for this family. Re-derive the source from the release."
        )
        return None
    return payload


def _auto_map_references(entry: str | list | tuple) -> tuple[str, ...]:
    """The class references one ``auto_map`` entry holds (a tokenizer entry is a ``[slow, fast]`` pair)."""
    return tuple(value for value in ((entry,) if isinstance(entry, str) else entry) if value)


def _copy_auto_map_modules(source: str, auto_map: dict, output_dir: str) -> None:
    """Copy the remote-code modules ``auto_map`` names from ``source``, with the siblings they import
    (Step-3.7's ``vision_encoder``)."""
    pending = {reference.split(".")[0] for entry in auto_map.values() for reference in _auto_map_references(entry)}
    copied: set[str] = set()
    while pending:
        module = pending.pop()
        if module in copied:
            continue
        copied.add(module)
        module_file = cached_file(source, f"{module}.py")
        if module_file is None:
            raise FileNotFoundError(
                f"export_source_config_schema: source {source!r} names remote-code module "
                f"{module}.py but does not carry it — the export would ship a config no consumer "
                f"can load. Re-derive the source from a complete checkpoint."
            )
        shutil.copyfile(module_file, os.path.join(output_dir, f"{module}.py"))
        pending.update(get_relative_imports(module_file))


def _serialized_config(config, output_dir: str) -> dict:
    """``config`` through every rewrite the writers run before the source-schema carry, as a comparable
    dict: a skipped rewrite would read as a run-made change."""
    config.save_pretrained(output_dir)
    restore_model_type(config, output_dir)
    export_legacy_per_layer_config(output_dir)
    payload = read_checkpoint_config(output_dir)
    payload.pop("transformers_version", None)
    return payload


def _config_payload_diff(expected: dict, actual: dict, path: str = "") -> dict[str, tuple]:
    """``{dotted key: (expected, actual)}`` over two serialized configs, recursing into sub-configs."""
    diff: dict[str, tuple] = {}
    for key in sorted(set(expected) | set(actual)):
        want, got = expected.get(key, _CONFIG_KEY_ABSENT), actual.get(key, _CONFIG_KEY_ABSENT)
        if isinstance(want, dict) and isinstance(got, dict) and "model_type" in want:
            diff.update(_config_payload_diff(want, got, f"{path}{key}."))
        elif want != got:
            diff[f"{path}{key}"] = tuple("<absent>" if v is _CONFIG_KEY_ABSENT else v for v in (want, got))
    return diff


def _apply_config_changes(source_section: dict, before: dict, after: dict, config, path: str = "") -> None:
    """Write this run's config changes (``after`` minus ``before``) into one source-schema section, in place.

    Each changed key goes in under the spelling that section already uses (the key or its
    ``attribute_map`` alias), else under its own name. The caller re-parses the result and compares.
    """
    aliases: dict[str, list[str]] = {}
    for alias, native in getattr(type(config), "attribute_map", {}).items():
        aliases.setdefault(native, []).append(alias)
    for key in sorted(set(before) | set(after)):
        old, new = before.get(key, _CONFIG_KEY_ABSENT), after.get(key, _CONFIG_KEY_ABSENT)
        if isinstance(old, dict) and isinstance(new, dict) and "model_type" in new:
            sub_section = source_section.get(key)
            if not isinstance(sub_section, dict):
                raise ValueError(
                    f"export_source_config_schema: the source config has no {path}{key!r} section to "
                    f"carry this model's sub-config into — the source and the live model are not the "
                    f"same family."
                )
            _apply_config_changes(sub_section, old, new, getattr(config, key, None), f"{path}{key}.")
            continue
        if old == new:
            continue
        spellings = [name for name in (key, *aliases.get(key, ())) if name in source_section]
        if new is _CONFIG_KEY_ABSENT:
            # The run left the key at its class default; the source's value must go with it.
            for name in spellings:
                del source_section[name]
            continue
        source_section[spellings[0] if spellings else key] = new


def export_source_config_schema(config, output_dir: str, *, source: str | None) -> None:
    """Rewrite ``output_dir``'s ``config.json`` into the source checkpoint's own schema, for every family
    declaring ``exports_source_config_schema``; a no-op elsewhere.

    transformers 5.16 absorbs the vendor spellings at load and never re-emits them, so the source
    config is carried over with this run's changes applied, plus the modules its ``auto_map`` names.
    """
    spec = find_moe_family_spec(getattr(config, "model_type", None))
    if not (spec and spec.exports_source_config_schema):
        return
    exported = read_checkpoint_config(output_dir, missing_ok=True)
    if exported is None:
        return
    if not source:
        logger.warning(
            "export_source_config_schema: this model records no source checkpoint, so its "
            "config.json stays in transformers' native schema — which the pinned rollout servers "
            "cannot parse for this family. Export from a checkpoint loaded off the release."
        )
        return
    payload = _source_config_payload(source)
    if payload is None:
        return
    exported.pop("transformers_version", None)
    # ``auto_map`` is carried by construction, and the live config holds it only when loaded off the source.
    exported.pop("auto_map", None)
    with tempfile.TemporaryDirectory() as staging:
        source_values = _serialized_config(type(config).from_dict(copy.deepcopy(payload)), staging)
        source_values.pop("auto_map", None)
        _apply_config_changes(payload, source_values, exported, config)
        carried = _serialized_config(type(config).from_dict(copy.deepcopy(payload)), staging)
    carried.pop("auto_map", None)
    if carried != exported:
        raise ValueError(
            f"export_source_config_schema: the source schema cannot express this run's config. "
            f"Re-parsing the carried config.json yields {_config_payload_diff(exported, carried)} — "
            f"the source spells those fields through legacy keys this export did not update, so the "
            f"served model would run a geometry the trainer never had. Apply the change to the "
            f"source checkpoint's own config.json and re-export."
        )
    write_config_json(output_dir, payload)
    _copy_auto_map_modules(source, payload["auto_map"], output_dir)


def drop_unshipped_auto_map(output_dir: str) -> None:
    """Drop the ``auto_map`` entries of ``output_dir``'s ``config.json`` whose modules it does not carry.

    transformers serializes the loaded ``auto_map`` but copies remote code only for an auto-registered
    class, so an in-library load of a remote-code hub checkpoint (Laguna) exports a config naming a
    module it never wrote. A ``repo--module.Class`` reference resolves from its own repo and stays.
    """
    payload = read_checkpoint_config(output_dir, missing_ok=True)
    auto_map = (payload or {}).get("auto_map")
    if not auto_map:
        return
    unshipped = {
        auto_class
        for auto_class, entry in auto_map.items()
        if any(
            "--" not in reference and not os.path.isfile(os.path.join(output_dir, f"{reference.split('.')[0]}.py"))
            for reference in _auto_map_references(entry)
        )
    }
    if not unshipped:
        return
    logger.info(
        f"drop_unshipped_auto_map: {os.path.join(output_dir, CONFIG_NAME)} names remote code the export "
        f"does not carry for {sorted(unshipped)}; dropped those auto_map entries, so a reload uses the "
        f"in-library class."
    )
    kept = {auto_class: entry for auto_class, entry in auto_map.items() if auto_class not in unshipped}
    if kept:
        payload["auto_map"] = kept
    else:
        del payload["auto_map"]
    write_config_json(output_dir, payload)


def finalize_exported_config(config, output_dir: str, *, source: str | None) -> None:
    """Bring a just-written ``config.json`` up to the artifact contract and tag the model card; every
    writer's last step.

    Each rewrite is a no-op for a family that needs none, so writers run the whole set: a checkpoint
    missing one trains normally and fails only in the merge tools or the rollout server.
    """
    restore_model_type(config, output_dir)
    export_legacy_per_layer_config(output_dir)
    export_source_config_schema(config, output_dir, source=source)
    drop_unshipped_auto_map(output_dir)
    tag_exported_model_card(output_dir)


def save_model_config(model, output_dir: str) -> None:
    """Write what ``model.save_pretrained`` would beside a parallel save's hand-gathered weights:
    ``config.json`` (finalized), ``generation_config.json`` and the remote-code modeling module."""
    if not hasattr(model, "config"):
        return
    config = model.config
    architecture = hf_architecture_name(model)
    if architecture is not None:
        config.architectures = [architecture]
    # Run-scoped router mutations (zeroed aux coef, forced output_router_logits, toolkit stamps)
    # configure this run rather than the artifact: serialize with the pre-mutation values restored.
    with config_export_ready(config):
        config.save_pretrained(output_dir)
    remote_code_cls = getattr(model, "_remote_code_model_class", None) or remote_code_model_class(model)
    if remote_code_cls is not None:
        custom_object_save(remote_code_cls, output_dir)
    finalize_exported_config(config, output_dir, source=config_schema_source(model))
    generation_config = getattr(model, "generation_config", None)
    if generation_config is not None and getattr(model, "can_generate", lambda: False)():
        generation_config.save_pretrained(output_dir)
