"""What an exported ``config.json`` must contain, and the one writer every save path finalizes through.

transformers' serialization is not the artifact contract: a vendor class may drop ``model_type``, a
pinned rollout server may need flat legacy keys the current schema folds away, and a remote-code
family may need the source repo's own schema plus the modules its ``auto_map`` names. Each rewrite is
a no-op for a family that needs none, so writers call the whole set rather than selecting.
"""

from __future__ import annotations

import copy
import json
import logging
import os
import shutil
import tempfile

from transformers.dynamic_module_utils import custom_object_save, get_relative_imports
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import CONFIG_NAME, cached_file

from src.models.loading.config_levels import config_export_ready
from src.models.moe_balancing import (
    ep_roster_registered,
    exports_source_config_schema,
    legacy_per_layer_config_keys,
)

logger = logging.getLogger(__name__)

# Stamped by load_distributed_model on the loaded model: where its weights were actually read (None
# under init_from_scratch). One spelling for the writer and the resume-coverage reader.
LOADED_WEIGHTS_FROM_ATTR = "_loaded_weights_from"

# Marks a key the serialized config does not spell, in the source-schema diff. Distinct from a key
# spelled as JSON ``null``, which is a value the run may legitimately have set.
_CONFIG_KEY_ABSENT = object()


def hf_architecture_name(model) -> str | None:
    """The class name a checkpoint's ``config.architectures`` should carry, or None.

    First ``PreTrainedModel`` subclass in the MRO that is not torch's FSDP2 in-place class swap
    (``FSDP<Name>``, module ``torch.*``); None for a non-HF carrier (a sentence-transformers shell,
    a PP stage), whose config came from elsewhere and whose caller decides.

    Re-derived because the load-time value is the hub's class, which a task-head swap invalidates: a
    reward run would ship ``*ForCausalLM`` and every ``architectures[0]``-keyed consumer (vLLM, TGI,
    ``Auto*``) would serve the wrong head.
    """
    return next(
        (
            cls.__name__
            for cls in type(model).__mro__
            if isinstance(cls, type)
            and issubclass(cls, PreTrainedModel)
            and cls is not PreTrainedModel
            and not cls.__module__.startswith("torch")
        ),
        None,
    )


def restore_model_type(config, output_dir: str) -> None:
    """Put the live ``model_type`` back into a ``config.json`` that lost it.

    ``to_dict`` reads ``model_type`` off the class, and some vendor config classes (Bailing/Ling)
    declare none, so the serialized config carries no family while the live object does; the
    sharded-EP merge then refuses the checkpoint it just produced and hub renames resolve to none.
    """
    live = getattr(config, "model_type", "")
    config_file = os.path.join(output_dir, CONFIG_NAME)
    if not live:
        # The live config declares no family either, so there is nothing correct to write.
        logger.warning(
            f"restore_model_type: the live config carries no model_type — {config_file} stays "
            f"family-less and every model-type-keyed reader (shard merge, hub renames) will see "
            f"no family. Re-derive the checkpoint from its hub source."
        )
        return
    if not os.path.isfile(config_file):
        return
    with open(config_file, encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("model_type"):
        return
    payload["model_type"] = live
    with open(config_file, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def flatten_per_layer_config(section: dict, legacy_keys: dict[str, tuple[str, str]]) -> bool:
    """Rewrite one serialized config's ``per_layer_config`` into the family's flat legacy keys, in place.

    Each flat key of ``legacy_keys`` (the family's :attr:`~EPMoELayerBase._LEGACY_PER_LAYER_CONFIG_KEYS`)
    takes ``field``'s value on the layers of ``layer_type``: the override where one is declared, the
    section's global value otherwise. ``per_layer_config`` is then dropped, since a 5.14-line reader
    refuses a config carrying it at all and a 5.16 reader rebuilds it. A config the flat keys cannot
    express is refused rather than flattened. Returns whether the section changed.
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
    :attr:`~EPMoELayerBase._LEGACY_PER_LAYER_CONFIG_KEYS`; a no-op for every other checkpoint.

    Runs after every config write the toolkit performs, because transformers 5.16's ``to_dict()`` has
    no flat form to offer: the family's config class dropped the legacy fields the pinned rollout
    server (vLLM 0.26.0, transformers 5.14 line) reads. Applied per serialized section (the root and
    every sub-config with a ``model_type``), so a composite VLM's text tower rewrites too.
    """
    config_file = os.path.join(output_dir, CONFIG_NAME)
    if not os.path.isfile(config_file):
        return
    with open(config_file, encoding="utf-8") as handle:
        payload = json.load(handle)
    sections = [payload, *(value for value in payload.values() if isinstance(value, dict) and "model_type" in value)]
    changed = False
    for section in sections:
        legacy_keys = legacy_per_layer_config_keys(section.get("model_type") or "")
        if legacy_keys:
            changed |= flatten_per_layer_config(section, legacy_keys)
    if not changed:
        return
    with open(config_file, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def checkpoint_source_ref(model) -> str | None:
    """The checkpoint this model's schema came from: a local directory or a Hub repo id, or None.

    ``load_distributed_model`` stamps :data:`LOADED_WEIGHTS_FROM_ATTR`; every other loader leaves
    transformers' ``config._name_or_path``, the same string. An ``init_from_scratch`` model took its
    geometry from that config, so the fallback is the right schema source there too.
    """
    return getattr(model, LOADED_WEIGHTS_FROM_ATTR, None) or getattr(
        getattr(model, "config", None), "_name_or_path", None
    )


def _source_config_payload(source: str) -> dict | None:
    """The source checkpoint's own ``config.json`` when it is a schema an engine can be served from.

    Returns None, with a warning naming the consequence, when the source is unreadable or declares
    no ``auto_map``: such a source was never servable by the engines that read this family through
    remote code, so there is nothing to carry. Not a raise, since the artifact is still trainable.
    """
    try:
        config_file = cached_file(source, CONFIG_NAME)
        with open(str(config_file), encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception as error:
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


def _copy_auto_map_modules(source: str, auto_map: dict, output_dir: str) -> None:
    """Copy the remote-code modules ``auto_map`` names from ``source`` into ``output_dir``.

    Copied transitively, through each module's relative imports: a modeling module that imports a
    sibling (Step-3.7's ``vision_encoder``) loads nowhere without it, and only the module the
    ``auto_map`` entry spells is named there.
    """
    pending = {
        value.split(".")[0]
        for entry in auto_map.values()
        for value in ((entry,) if isinstance(entry, str) else entry)
        if value
    }
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
    """``config`` through the same pipeline a toolkit config write applies, as a comparable dict.

    Every step the writers run before the source-schema carry, so the payloads being differenced were
    produced identically; one of them skipping a rewrite would read as a run-made change.
    """
    config.save_pretrained(output_dir)
    restore_model_type(config, output_dir)
    export_legacy_per_layer_config(output_dir)
    with open(os.path.join(output_dir, CONFIG_NAME), encoding="utf-8") as handle:
        payload = json.load(handle)
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
    """Write this run's config changes into one section of the source-schema payload, in place.

    ``before``/``after`` are the same config serialized from the source's values and from the live
    ones, so their difference is exactly what the run changed. Each changed key goes in under the
    spelling that section already uses (the key, or the source's alias for it read backwards out of
    ``attribute_map``), and under its own name where the source spells neither, never both.

    Nothing here can prove the result parses back to the live values, so the caller re-parses and
    compares.
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
    """Rewrite ``output_dir``'s ``config.json`` into the source checkpoint's own config schema, for
    every family declaring :attr:`~EPMoELayerBase._EXPORTS_SOURCE_CONFIG_SCHEMA`; a no-op elsewhere.

    Runs after every config write the toolkit performs, one step past
    :func:`export_legacy_per_layer_config`: for these families transformers 5.16's serialization is a
    schema the server has no class for. The vendor spellings are absorbed at load and never
    re-emitted, so the source config is carried over with this run's changed values applied under its
    spellings, plus the modules its ``auto_map`` names. A 5.16 reload is unaffected.
    """
    if not exports_source_config_schema(getattr(config, "model_type", "") or ""):
        return
    config_file = os.path.join(output_dir, CONFIG_NAME)
    if not os.path.isfile(config_file):
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
    with open(config_file, encoding="utf-8") as handle:
        exported = json.load(handle)
    exported.pop("transformers_version", None)
    # ``auto_map`` names the source's modules and is carried across by construction; the live config
    # carries it only when it was loaded from that source, so it is excluded from the comparison.
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
    with open(config_file, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    _copy_auto_map_modules(source, payload["auto_map"], output_dir)


def finalize_exported_config(config, output_dir: str, *, source: str | None) -> None:
    """Bring a just-written ``config.json`` up to the artifact contract. Every writer's last step.

    Three rewrites transformers' serialization does not do, in the order they compose: the live
    ``model_type`` restored for vendor classes declaring none, the flat legacy per-layer keys the
    pinned servers read, and the source repo's own schema where the family declares it. Each is a
    no-op for a family that needs none, so the writers call the whole set rather than selecting.

    One function because the parallel writers and the single-GPU save must produce the same
    directory: a checkpoint missing any of these trains normally and fails only in the merge tools or
    the rollout server.
    """
    if not ep_roster_registered():
        raise RuntimeError(
            "finalize_exported_config reached with no EP family registered in this process: the "
            "legacy-key and source-schema rosters are filled by importing "
            "src.distributed.expert_parallel.layers.roster, and an empty roster would silently export a config its servers cannot load."
        )
    restore_model_type(config, output_dir)
    export_legacy_per_layer_config(output_dir)
    export_source_config_schema(config, output_dir, source=source)


def save_model_config(model, output_dir: str) -> None:
    """Write ``config.json``, ``generation_config.json`` and any remote-code modules for a parallel save.

    A parallel path cannot call ``model.save_pretrained`` (it would rewrite the weights just gathered
    by hand), so what that would have emitted is reproduced here:

    * ``PretrainedConfig`` does not carry the generation config, so custom ``eos_token_id`` sets,
      stop strings and sampling defaults would be lost, and vLLM reads that file directly.
    * ``config.save_pretrained`` copies a remote-code model's configuration module but not the
      modeling module its ``auto_map`` names, and the directory then loads nowhere.
    * ``config.architectures`` still holds the hub's class after a task-head swap.

    :func:`finalize_exported_config` is called here too, so a parallel save and a
    ``save_pretrained`` export produce the same directory.
    """
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
    # Ship the modeling modules the config's ``auto_map`` names; the write above copies only the
    # configuration module. Read from the MRO rather than ``type(model)``: FSDP2 rewrites
    # ``model.__class__`` in place, so a live-class check would skip every remote-code model saved
    # from a sharded run.
    remote_code_cls = getattr(model, "_remote_code_model_class", None) or next(
        (cls for cls in type(model).__mro__ if cls.__module__.startswith("transformers_modules")), None
    )
    if remote_code_cls is not None:
        custom_object_save(remote_code_cls, output_dir)
    finalize_exported_config(config, output_dir, source=checkpoint_source_ref(model))
    generation_config = getattr(model, "generation_config", None)
    if generation_config is not None and getattr(model, "can_generate", lambda: False)():
        generation_config.save_pretrained(output_dir)
