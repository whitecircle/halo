"""The ``metadata.json`` contract of a preprocessed dataset: the recorded config, the stamp, and the checks of an
artifact against the run."""

import json
import logging
import os
from dataclasses import MISSING, dataclass, field, is_dataclass
from dataclasses import fields as dataclass_fields
from datetime import UTC, datetime
from typing import Any, Literal, get_args

from huggingface_hub import hf_hub_download, is_offline_mode
from huggingface_hub.errors import EntryNotFoundError, LocalEntryNotFoundError, RepositoryNotFoundError

from halo.data.preprocessed.shard_index import (
    PREPROCESSING_VERSION,
    IncompatiblePreprocessedDataset,
    reject_incompatible_stamp,
)
from halo.data.sources.paths import (
    METADATA_FILE,
    hub_repo_id,
    parse_dataset_source,
    reject_ambiguous_hub_source,
    s3_object_key,
)
from halo.data.sources.s3_dataset_store import read_control_json_with_cache
from halo.model_loading.tokenizer_backend import TokenizerBackend
from halo.model_loading.tokenizer_setup import load_chat_template

logger = logging.getLogger(__name__)

# The mode that reads each mode-specific knob; set in the other mode, the knob is refused.
_KNOB_MODES = {
    **dict.fromkeys(
        (
            "conversation_field",
            "system_prompt",
            "model_supports_system_role",
            "tools_field",
            "interleaved_thinking",
            "images_field",
            "train_on_completions_only",
            "assistant_message_template",
        ),
        "chat",
    ),
    "text_field": "text",
    "append_eos": "text",
}

# Fields the render-knob check skips (a new one is compared): checked on their own, text-only, baked or layout-only.
_UNCOMPARED_FIELDS = frozenset(
    {
        "model_name_or_path",
        "max_length",
        "mode",
        "text_field",
        "append_eos",
        "train_on_completions_only",
        "pack_sequences",
        "packing_strategy",
        "num_shards",
        "num_proc",
        "tokenizer_backend",
        "trust_remote_code",
        "is_vlm",
        "min_pixels",
        "max_pixels",
    }
)

_PreprocessingMode = Literal["chat", "text"]
_PackingStrategy = Literal["bfd", "bfd_split", "wrapped"]
_DEFAULT_PACKING_STRATEGY = "bfd"


@dataclass
class PreprocessingConfig:
    """How a corpus is baked; ``prepare_dataset.py``'s flags are these fields. ``__post_init__`` refuses an unknown
    mode, a knob the mode does not read and a combination the bake cannot honor."""

    model_name_or_path: str = field(
        metadata={"help": "Model whose tokenizer bakes the rows and whose config.json lists the turn terminators."}
    )
    max_length: int = field(
        default=8192,
        metadata={"help": "Token budget: a longer row is dropped, not cut. Training must use the same value."},
    )
    conversation_field: str = field(default="prompt", metadata={"help": "Column holding the conversation."})
    system_prompt: str | None = field(default=None, metadata={"help": "System prompt for every conversation."})
    model_supports_system_role: bool = field(
        default=True, metadata={"help": "False folds the system prompt into the first user turn."}
    )
    tools_field: str | None = field(
        default=None, metadata={"help": "Column holding a tool list for the chat template, as in training."}
    )
    interleaved_thinking: bool = field(
        default=False, metadata={"help": "Keep thinking in non-final assistant turns (clear_thinking=False)."}
    )
    images_field: str | None = field(
        default=None,
        metadata={"help": "(is_vlm) Column holding the images of a dataset that keeps them outside the messages."},
    )
    mode: _PreprocessingMode = field(
        default="chat",
        metadata={"help": "'chat' renders conversation_field (SFT); 'text' tokenizes text_field (pretraining)."},
    )
    text_field: str = field(default="text", metadata={"help": "(mode=text) Column holding the raw text."})
    append_eos: bool = field(
        default=True, metadata={"help": "(mode=text) End each document with EOS, keeping its boundary when packed."}
    )
    train_on_completions_only: bool = field(
        default=False,
        metadata={
            "help": "Bake completion-only labels; needs assistant_message_template. prepare_dataset.py "
            "turns it on in chat mode when unset, as training does."
        },
    )
    assistant_message_template: str | None = field(
        default=None, metadata={"help": "The rendered start of an assistant turn, e.g. '<|im_start|>assistant\\n'."}
    )
    pack_sequences: bool = field(default=False, metadata={"help": "Pack rows to max_length (text only)."})
    packing_strategy: _PackingStrategy = field(
        default=_DEFAULT_PACKING_STRATEGY,
        metadata={
            "help": "'bfd' drops a document's overflow, 'bfd_split' carries it into later rows, 'wrapped' "
            "chunks across documents."
        },
    )
    # Tokenizer overrides are recorded: they change the baked ids and nothing else in the artifact shows them.
    pad_token: str | None = field(default=None, metadata={"help": "Override the pad token."})
    eos_token: str | None = field(default=None, metadata={"help": "Override the EOS token."})
    bos_token: str | None = field(default=None, metadata={"help": "Override the BOS token."})
    chat_template: str | None = field(
        default=None,
        metadata={"help": "Override the chat template: Jinja text or a .jinja/.jinja2/.j2 file, recorded resolved."},
    )
    num_shards: int = field(
        default=1,
        metadata={
            "help": "Shards per split. 1 writes an unsharded dataset that trains at any data-parallel size; "
            "more need at least the data-parallel size and a test split."
        },
    )
    num_proc: int | None = field(
        default=None,
        metadata={"help": "Map worker processes; unset means HALO_DATASET_NUM_PROC, else every usable CPU but two."},
    )
    tokenizer_backend: TokenizerBackend = field(
        default="hf", metadata={"help": "Text-to-ids backend; 'gigatoken' is verified to emit the same ids."}
    )
    trust_remote_code: bool = field(
        default=False, metadata={"help": "Run the model's own code to read its config for the label bake."}
    )
    is_vlm: bool = field(
        default=False,
        metadata={
            "help": "Bake through the model's processor, storing pixel_values and image_grid_thw. An image row whose "
            "processor emits another vision key (mm_token_type_ids) is refused: train those through the runtime VLM path."
        },
    )
    min_pixels: int | None = field(default=None, metadata={"help": "(is_vlm) Minimum pixels per image."})
    max_pixels: int | None = field(default=None, metadata={"help": "(is_vlm) Maximum pixels per image."})

    def __post_init__(self) -> None:
        modes, strategies = get_args(_PreprocessingMode), get_args(_PackingStrategy)
        if self.mode not in modes:
            raise ValueError(f"Invalid preprocessing mode '{self.mode}'. Expected one of {list(modes)}.")
        if self.pack_sequences and self.packing_strategy not in strategies:
            raise ValueError(
                f"Invalid packing_strategy '{self.packing_strategy}'. TRL pack_dataset only accepts {list(strategies)}."
            )
        if self.is_vlm and self.pack_sequences:
            raise ValueError(
                "Packing is not supported for VLM datasets. Set pack_sequences=False when using is_vlm=True."
            )

        inapplicable = sorted(
            f.name
            for f in dataclass_fields(self)
            if _KNOB_MODES.get(f.name, self.mode) != self.mode and getattr(self, f.name) != f.default
        )
        if inapplicable:
            raise ValueError(
                f"{inapplicable} {'is' if len(inapplicable) == 1 else 'are'} not applicable to "
                f"mode='{self.mode}' and would be silently ignored: a mode='text' run renders no chat "
                f"template, and a mode='chat' run tokenizes no raw-text field. Drop the setting, or "
                f"switch mode."
            )

        unread = [name for name in ("min_pixels", "max_pixels") if getattr(self, name) is not None and not self.is_vlm]
        if unread:
            raise ValueError(
                f"{unread} set the image processor's bounds, but is_vlm=False: a text bake loads no image processor. "
                f"Set is_vlm=True, or drop them."
            )
        if not self.pack_sequences and self.packing_strategy != _DEFAULT_PACKING_STRATEGY:
            raise ValueError(
                f"packing_strategy='{self.packing_strategy}' is set, but pack_sequences=False: no pack reads it. Set "
                f"pack_sequences=True, or drop it."
            )

        if self.images_field and not self.is_vlm:
            raise ValueError(
                f"images_field='{self.images_field}' names an image column, but is_vlm=False: only the "
                f"VLM tokenization merges that column into the conversation, so the text path would "
                f"drop it and bake a text-only dataset. Set is_vlm=True, or drop images_field."
            )

        if self.train_on_completions_only and not self.assistant_message_template:
            raise ValueError(
                "train_on_completions_only=True requires assistant_message_template (the response "
                "marker, e.g. '<|im_start|>assistant\\n') so completion-only labels can be built. "
                "Pass the marker your model's chat template renders, or set "
                "train_on_completions_only=False to train on the full sequence."
            )


_RENDER_CHECKED_FIELDS = tuple(
    f.name for f in dataclass_fields(PreprocessingConfig) if f.name not in _UNCOMPARED_FIELDS
)


@dataclass
class PreprocessedDatasetMetadata:
    version: str = PREPROCESSING_VERSION
    preprocessed: bool = True

    model_name: str = ""
    tokenizer_vocab_size: int = 0
    # The template the chat rows were rendered with, overrides applied; None for a text artifact.
    rendered_chat_template: str | dict[str, str] | None = None

    max_length: int = 0
    packed: bool = False
    packing_strategy: str | None = None
    train_on_completions_only: bool = False

    is_vlm: bool = False
    has_pixel_values: bool = False

    num_shards: int = 1
    total_train_examples: int = 0
    total_test_examples: int = 0

    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    config: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def claims_payload(cls, data: Any) -> bool:
        """A hub dataset may ship its own file of that name, which this schema must not judge."""
        return isinstance(data, dict) and "preprocessed" in data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PreprocessedDatasetMetadata":
        """Refuses a stamp this build cannot read rather than re-tokenizing pre-tokenized rows as raw; retired
        spellings are not migrated, since a stamp records what was baked."""
        reject_incompatible_stamp(cls, data, "Preprocessed dataset metadata")
        return cls(**data)


def _read_metadata_payload(path: str, *, s3_outage_reads_absent: bool) -> dict[str, Any] | None:
    """The raw payload, None only on a confirmed absence; any other failure raises, so an unreachable source never
    reads as raw. ``s3_outage_reads_absent`` (the probe's) reads an unmirrored S3 outage as absent with a warning,
    since a warm-cache raw dataset loads without S3."""
    source_type, bucket, key = parse_dataset_source(path)
    if source_type == "s3":
        # Mirrored: an outage serves the local mirror, so a warm-cache run still reads as preprocessed.
        try:
            return read_control_json_with_cache(bucket, s3_object_key(key, METADATA_FILE))
        except FileNotFoundError:
            return None
        except ValueError:
            # The live read parses the stamp itself: a stamp that does not parse is no outage.
            raise
        except Exception as e:
            if not s3_outage_reads_absent:
                raise
            logger.warning(
                f"S3 is unreachable for {path}/{METADATA_FILE} and no local mirror of it exists "
                f"({type(e).__name__}: {e}); treating the dataset as raw."
            )
            return None

    if source_type == "hf_hub":
        reject_ambiguous_hub_source(path)
        try:
            local_path = hf_hub_download(hub_repo_id(path), METADATA_FILE, repo_type="dataset")
        except LocalEntryNotFoundError as e:
            if is_offline_mode():
                return None
            raise ConnectionError(
                f"The Hub could not be reached for {path}/{METADATA_FILE}, so whether the dataset is "
                f"preprocessed is unknown. Retry, or set HF_HUB_OFFLINE=1 to run from the local cache."
            ) from e
        except (EntryNotFoundError, RepositoryNotFoundError):
            return None
        with open(local_path) as f:
            return json.load(f)

    metadata_path = os.path.join(key, METADATA_FILE)
    if not os.path.exists(metadata_path):
        return None
    with open(metadata_path) as f:
        return json.load(f)


def is_preprocessed_dataset(path: str) -> bool:
    """An absent or foreign file reads as raw, and an incompatible toolkit stamp as preprocessed, so
    :func:`load_preprocessed_metadata` refuses it instead of the run re-tokenizing its rows."""
    payload = _read_metadata_payload(path, s3_outage_reads_absent=True)
    if payload is None:
        return False
    if not PreprocessedDatasetMetadata.claims_payload(payload):
        logger.warning(
            f"{path} carries a {METADATA_FILE} that is not a toolkit preprocessing stamp "
            f"(no 'preprocessed' field); treating the dataset as raw."
        )
        return False
    try:
        return PreprocessedDatasetMetadata.from_dict(payload).preprocessed
    except IncompatiblePreprocessedDataset:
        return True


def load_preprocessed_metadata(path: str) -> PreprocessedDatasetMetadata:
    payload = _read_metadata_payload(path, s3_outage_reads_absent=False)
    if payload is None:
        raise FileNotFoundError(f"No {METADATA_FILE} at {path}: not a preprocessed dataset.")
    if not PreprocessedDatasetMetadata.claims_payload(payload):
        raise FileNotFoundError(
            f"{METADATA_FILE} at {path} is not a toolkit preprocessing stamp (no 'preprocessed' "
            f"field): the dataset is raw and carries someone else's metadata file."
        )
    return PreprocessedDatasetMetadata.from_dict(payload)


def _stated_render_knobs(render_args: Any) -> frozenset[str] | None:
    """Knobs held off their default, or None (every knob stated) for non-dataclass args. A default makes no claim on
    the artifact, and the two sides' defaults move apart."""
    if not is_dataclass(render_args) or isinstance(render_args, type):
        return None
    stated: set[str] = set()
    for f in dataclass_fields(type(render_args)):
        default = f.default_factory() if f.default_factory is not MISSING else f.default
        if default is MISSING or getattr(render_args, f.name, default) != default:
            stated.add(f.name)
    return frozenset(stated)


def _render_knob_differences(
    metadata: PreprocessedDatasetMetadata, render_args: Any, recorded: dict[str, Any]
) -> tuple[dict[str, tuple[Any, Any]], dict[str, tuple[Any, Any]]]:
    """``(stated, unstated)`` differing knobs as ``name -> (prepared, run)``."""
    stated = _stated_render_knobs(render_args)
    mismatched: dict[str, tuple[Any, Any]] = {}
    unstated: dict[str, tuple[Any, Any]] = {}
    for name in _RENDER_CHECKED_FIELDS:
        if name not in recorded or not hasattr(render_args, name):
            continue
        if name == "assistant_message_template" and not metadata.train_on_completions_only:
            # Without completion masking the template touches nothing baked.
            continue
        run_value = getattr(render_args, name)
        if name == "chat_template":
            # The config records the RESOLVED text; the run may spell the same template as a path.
            run_value = load_chat_template(run_value) if run_value else None
        if recorded[name] != run_value:
            target = mismatched if stated is None or name in stated else unstated
            target[name] = (recorded[name], run_value)
    return mismatched, unstated


def _verify_render_knobs(metadata: PreprocessedDatasetMetadata, render_args: Any) -> None:
    """Raise when a knob the run STATES differs from the baked value; knobs the metadata predates, ``render_args``
    lacks or the run leaves at default are reported instead, since the prepared value is what trained either way."""
    recorded = metadata.config or {}
    if recorded.get("mode") == "text":
        # Raw-text artifacts render no chat template, so no knob was baked.
        return

    unrecorded = [name for name in _RENDER_CHECKED_FIELDS if name not in recorded and hasattr(render_args, name)]
    if unrecorded:
        logger.warning(
            f"Preprocessed dataset metadata predates recording of {unrecorded}; cannot verify these "
            f"render settings against the training config."
        )

    mismatched, unstated = _render_knob_differences(metadata, render_args, recorded)
    if unstated:
        detail = "; ".join(
            f"{name}: prepared={prep!r} vs this run's default={run!r}" for name, (prep, run) in unstated.items()
        )
        remedy = "set these explicitly in the YAML to silence this"
        if "chat_template" in unstated:
            # A stated chat_template replaces a shipped one only with force_chat_template: the exported template
            # is the run tokenizer's.
            remedy += ", chat_template together with force_chat_template: true over a template the checkpoint ships"
        logger.warning(
            f"Preprocessed dataset was prepared with render settings the training config does not "
            f"state ({detail}). Rendering is baked, so the PREPARED value is what trained; {remedy}."
        )
    if mismatched:
        detail = "; ".join(f"{name}: prepared={prep!r} vs run={run!r}" for name, (prep, run) in mismatched.items())
        raise ValueError(
            f"Preprocessed dataset render settings do not match the training config ({detail}). "
            f"Rendering and tokenization are baked at preparation time, so the runtime values cannot "
            f"take effect — re-preprocess with the run's settings "
            f"(scripts/before_training/prepare_dataset.py), or set the training YAML to the prepared "
            f"values."
        )


def _rows_chat_template(metadata: PreprocessedDatasetMetadata) -> tuple[Any, str] | None:
    """``(template, origin)`` of the template the chat rows were rendered with, or None where the stamp does not say: a
    text artifact renders none, and one without ``rendered_chat_template`` names at most a ``--chat_template``
    override, which a VLM stamp does not show its processor used."""
    recorded = metadata.config or {}
    if recorded.get("mode") == "text":
        return None
    override = recorded.get("chat_template")
    template = metadata.rendered_chat_template
    if template is None and not metadata.is_vlm:
        template = override
    if template is None:
        return None
    origin = (
        "the --chat_template override metadata.json records"
        if override
        else f"the template {metadata.model_name} ships"
    )
    return template, origin


def _verify_exported_chat_template(
    metadata: PreprocessedDatasetMetadata, exported: Any, run_model: str | None
) -> None:
    """Raise when the template the run exports is not the one the rows were rendered with: the model would train on one
    and be served with the other."""
    rows = _rows_chat_template(metadata)
    if rows is None or rows[0] == exported:
        return
    run_side = "no chat template" if exported is None else f"the template {run_model or 'its checkpoint'} ships"
    raise ValueError(
        f"The preprocessed rows were rendered with {rows[1]}, but this run ends with {run_side}. The saved "
        f"checkpoint exports the run's template, so the model would train on one template and be served with "
        f"another. Rendering is baked: train with the rows' template (chat_template set to it, plus "
        f"force_chat_template: true), or re-preprocess with the template this run exports "
        f"(scripts/before_training/prepare_dataset.py)."
    )


def verify_preprocessed_dataset(
    metadata: PreprocessedDatasetMetadata,
    required_max_length: int,
    required_model: str | None = None,
    required_train_on_completions_only: bool | None = None,
    render_args: Any | None = None,
    required_packing: bool | None = None,
    run_processing_class: Any | None = None,
) -> None:
    """Raise on a max_length, baked-label or render-knob mismatch, and on an exported chat template other than the rows
    were rendered with; warn on a model mismatch and on ``required_packing`` against an unpacked artifact."""
    if metadata.max_length < required_max_length:
        raise ValueError(
            f"Preprocessed dataset max_length ({metadata.max_length}) is less than "
            f"required max_length ({required_max_length}). Re-preprocess with larger max_length."
        )
    if metadata.max_length > required_max_length:
        # Rows baked at metadata.max_length are never re-truncated, so the run would exceed its budget.
        raise ValueError(
            f"Preprocessed dataset max_length ({metadata.max_length}) exceeds the configured "
            f"max_length ({required_max_length}) and preprocessed rows are not re-truncated at "
            f"runtime. Re-preprocess at {required_max_length}, or raise max_length to match."
        )

    if required_train_on_completions_only is not None:
        # Labels are baked, so a mismatch would train the opposite of the YAML.
        if "train_on_completions_only" not in (metadata.config or {}):
            logger.warning(
                "Preprocessed dataset metadata predates the train_on_completions_only field; cannot "
                "verify that the baked labels match the runtime flag "
                f"(train_on_completions_only={required_train_on_completions_only})."
            )
        elif bool(metadata.train_on_completions_only) != bool(required_train_on_completions_only):
            raise ValueError(
                f"Preprocessed dataset was prepared with train_on_completions_only="
                f"{metadata.train_on_completions_only} but the training config sets "
                f"train_on_completions_only={required_train_on_completions_only}. Labels are baked at "
                f"preparation time, so the runtime flag cannot change them. Either re-preprocess the "
                f"dataset with the desired masking (scripts/before_training/prepare_dataset.py) or set "
                f"train_on_completions_only={metadata.train_on_completions_only} in the training YAML."
            )

    if required_model and metadata.model_name != required_model:
        logger.warning(
            f"Preprocessed dataset was created with model '{metadata.model_name}' "
            f"but training with '{required_model}'. Tokenization may differ."
        )

    if render_args is not None:
        _verify_render_knobs(metadata, render_args)
    if run_processing_class is not None:
        _verify_exported_chat_template(metadata, run_processing_class.chat_template, required_model)

    if required_packing and not metadata.packed:
        logger.warning(
            "The training config sets packing=True but the preprocessed dataset was prepared "
            "UNPACKED (metadata.packed=False). Preprocessed rows are never packed at runtime, so "
            "this run trains on individually padded rows without packing's throughput — "
            "re-preprocess with --pack_sequences to actually pack."
        )
