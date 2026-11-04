"""What an exported ``config.json`` must contain beyond transformers' serialization, and the finalizer every save path
ends with; a vendor class may drop ``model_type``, and a remote-code family may need the source repo's own schema."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from collections.abc import Iterator

from transformers.dynamic_module_utils import custom_object_save, get_relative_imports
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
from transformers.tokenization_utils_base import TOKENIZER_CONFIG_FILE
from transformers.utils import CONFIG_NAME, cached_file

from halo.checkpoint.format import PROCESSOR_FILES
from halo.checkpoint.model_card import tag_exported_model_card
from halo.checkpoint.reading import read_checkpoint_config
from halo.checkpoint.weights_source import config_schema_source
from halo.log import info_logger
from halo.models.config_access import config_export_ready
from halo.models.model_classes import remote_code_model_class, transformers_model_class
from halo.models.moe.family_specs import find_moe_family_spec

logger = info_logger(__name__)

# A key the serialized config does not spell, distinct from one spelled as JSON ``null``.
_CONFIG_KEY_ABSENT = object()
# Copied raw from a release by the tools, so they keep an ``auto_map`` naming modules config.json does not.
_AUX_CONFIG_FILES = (*PROCESSOR_FILES, TOKENIZER_CONFIG_FILE)


def write_config_json(output_dir: str, payload: dict) -> None:
    """Write ``config.json`` in the one form every toolkit rewrite emits, so a re-export diffs only what changed."""
    with open(os.path.join(output_dir, CONFIG_NAME), "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def hf_architecture_name(model) -> str | None:
    """The class name ``config.architectures`` should carry, or None; a task-head swap invalidates the hub's class."""
    model_class = transformers_model_class(model)
    return None if model_class is None else model_class.__name__


def restore_model_type(config, output_dir: str) -> None:
    """Restore a lost ``model_type``; ``to_dict`` reads the class, which vendors (Bailing/Ling) leave unset."""
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


def _read_without_remote_code(config, payload: dict) -> bool:
    """Whether a source naming no modules still has a schema to hand on: one of transformers' own causal classes, which
    stock ``from_pretrained`` and vLLM read through that class and SGLang through its own. A composite reaches the
    engines only through the modules it names."""
    return payload.get("model_type") == config.model_type and config.model_type in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES


def _source_config_payload(source: str, config) -> dict | None:
    """The source's ``config.json`` if servable, else None with a warning (unreadable, or an ``auto_map`` its class
    needs and lacks)."""
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
    if not payload.get("auto_map") and not _read_without_remote_code(config, payload):
        logger.warning(
            f"export_source_config_schema: source {source!r} declares no auto_map, so it carries no "
            f"servable schema to hand on — the export keeps transformers' native one, which the "
            f"pinned rollout servers cannot parse for this family. Re-derive the source from the release."
        )
        return None
    return payload


def _auto_map_references(entry: str | list | tuple | dict) -> tuple[str, ...]:
    """The class references one ``auto_map`` entry holds: one, a tokenizer's ``[slow, fast]`` pair, or an image
    processor's mapping of backend to class."""
    values = entry.values() if isinstance(entry, dict) else (entry,) if isinstance(entry, str) else entry
    return tuple(value for value in values if value)


def _aux_auto_maps(payload: dict) -> Iterator[dict]:
    """Every ``auto_map`` one processor or tokenizer config holds, as a load reads them: at its top, where a bare list is
    a tokenizer's legacy pair, or in a sub-processor's section of a ``processor_config.json``."""
    for section in (payload, *(value for value in payload.values() if isinstance(value, dict))):
        auto_map = section.get("auto_map")
        if auto_map:
            yield auto_map if isinstance(auto_map, dict) else {"AutoTokenizer": auto_map}


def _copy_auto_map_modules(source: str, auto_map: dict, output_dir: str) -> None:
    """Copy the remote-code modules ``auto_map`` names, with their sibling imports (Step-3.7's ``vision_encoder``)."""
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
    """``config`` through every rewrite before the schema carry; a skipped one reads as a run-made change."""
    config.save_pretrained(output_dir)
    restore_model_type(config, output_dir)
    payload = read_checkpoint_config(output_dir)
    payload.pop("transformers_version", None)
    return payload


def _reparsed_source_values(config, payload: dict, staging: str) -> dict:
    """``payload`` through ``config``'s class and back, as a run would write it. The class reads floats JSON cannot
    spell (``Infinity``) decoded, and raw they reach its validators as the ``{"__float__": ...}`` tag."""
    values = _serialized_config(type(config).from_dict(type(config)._decode_special_floats(payload)), staging)
    # Carried by construction, and the live config holds it only when loaded off the source.
    values.pop("auto_map", None)
    return values


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
    """Write this run's config changes into one source-schema section in place, under the spelling it already uses (the
    key or its ``attribute_map`` or ``_legacy_field_names`` alias), else the key's own name. The caller re-parses the
    result and compares."""
    aliases: dict[str, list[str]] = {}
    for table in ("attribute_map", "_legacy_field_names"):
        for alias, native in getattr(type(config), table, {}).items():
            aliases.setdefault(native, []).append(alias)
    for key in sorted(set(before) | set(after)):
        old, new = before.get(key, _CONFIG_KEY_ABSENT), after.get(key, _CONFIG_KEY_ABSENT)
        spellings = [name for name in (key, *aliases.get(key, ())) if name in source_section]
        if isinstance(old, dict) and isinstance(new, dict) and "model_type" in new:
            sub_section = source_section[spellings[0]] if spellings else None
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
        if new is _CONFIG_KEY_ABSENT:
            # The run left the key at its class default; the source's value must go with it.
            for name in spellings:
                del source_section[name]
            continue
        source_section[spellings[0] if spellings else key] = new


def export_source_config_schema(config, output_dir: str, *, source: str | None) -> None:
    """Rewrite ``config.json`` into the source checkpoint's schema for families with ``exports_source_config_schema``.
    transformers 5.20 absorbs the vendor spellings at load and never re-emits them, so the source config is carried
    over with this run's changes applied, plus the modules its ``auto_map`` names."""
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
    payload = _source_config_payload(source, config)
    if payload is None:
        return
    source_type = payload.get("model_type")
    if source_type and source_type != config.model_type:
        logger.info(
            f"export_source_config_schema: this {config.model_type!r} export reads a {source_type!r} source, a "
            f"text-only load of a composite, so its config.json keeps transformers' schema for {config.model_type!r}."
        )
        return
    exported.pop("transformers_version", None)
    exported.pop("auto_map", None)
    with tempfile.TemporaryDirectory() as staging:
        source_values = _reparsed_source_values(config, payload, staging)
        # A trainer's KV-cache write on a composite's root, which the class does not hold: run state, not geometry.
        if "use_cache" not in source_values:
            exported.pop("use_cache", None)
        _apply_config_changes(payload, source_values, exported, config)
        carried = _reparsed_source_values(config, payload, staging)
    if carried != exported:
        raise ValueError(
            f"export_source_config_schema: the source schema cannot express this run's config. "
            f"Re-parsing the carried config.json yields {_config_payload_diff(exported, carried)} — "
            f"the source spells those fields through legacy keys this export did not update, so the "
            f"served model would run a geometry the trainer never had. Apply the change to the "
            f"source checkpoint's own config.json and re-export."
        )
    write_config_json(output_dir, payload)
    _copy_auto_map_modules(source, payload.get("auto_map", {}), output_dir)


def drop_unshipped_auto_map(output_dir: str) -> None:
    """Drop ``auto_map`` entries whose modules the directory lacks; a ``repo--module.Class`` reference resolves itself.
    transformers copies remote code only for an auto-registered class, so an in-library load (Laguna) names unwritten
    modules."""
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


def carry_aux_auto_map_modules(output_dir: str, *, source: str | None) -> None:
    """Copy the remote-code modules a processor or tokenizer config in ``output_dir`` names, as a trusted load of the
    export reads them there. Warns on what ``source`` lacks, since a trainer's save ends here and the served load
    raises by itself."""
    for name in _AUX_CONFIG_FILES:
        path = os.path.join(output_dir, name)
        if not os.path.isfile(path):
            continue
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
        local: dict[str, list[str]] = {}
        for auto_map in _aux_auto_maps(payload):
            for auto_class, entry in auto_map.items():
                references = (reference for reference in _auto_map_references(entry) if "--" not in reference)
                local.setdefault(auto_class, []).extend(references)
        absent = {
            auto_class: references
            for auto_class, references in local.items()
            if any(not os.path.isfile(os.path.join(output_dir, f"{r.split('.')[0]}.py")) for r in references)
        }
        if not absent:
            continue
        try:
            if not source:
                raise FileNotFoundError("this export records no source checkpoint")
            _copy_auto_map_modules(source, absent, output_dir)
        except OSError as error:
            logger.warning(
                f"carry_aux_auto_map_modules: {path} names remote code this export does not carry for "
                f"{sorted(absent)}, so a trust_remote_code load of it will fail: {error}"
            )


def finalize_exported_config(config, output_dir: str, *, source: str | None) -> None:
    """Bring a just-written ``config.json`` up to the artifact contract and tag the model card; every writer's last
    step. A checkpoint missing a rewrite trains normally and fails only in the merge tools or the rollout server."""
    restore_model_type(config, output_dir)
    export_source_config_schema(config, output_dir, source=source)
    drop_unshipped_auto_map(output_dir)
    carry_aux_auto_map_modules(output_dir, source=source)
    tag_exported_model_card(output_dir)


def save_model_config(model, output_dir: str) -> None:
    """Write what ``save_pretrained`` would beside gathered weights: config, generation config, remote code."""
    if not hasattr(model, "config"):
        return
    config = model.config
    architecture = hf_architecture_name(model)
    if architecture is not None:
        config.architectures = [architecture]
    # Run-scoped router mutations (zeroed aux coef, forced output_router_logits, stamps) stay out of the artifact.
    with config_export_ready(config):
        config.save_pretrained(output_dir)
    if (remote_code_cls := remote_code_model_class(model)) is not None:
        custom_object_save(remote_code_cls, output_dir)
    finalize_exported_config(config, output_dir, source=config_schema_source(model))
    generation_config = getattr(model, "generation_config", None)
    if generation_config is not None and getattr(model, "can_generate", lambda: False)():
        generation_config.save_pretrained(output_dir)
