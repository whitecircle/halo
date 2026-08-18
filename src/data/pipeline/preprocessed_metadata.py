"""The ``metadata.json`` contract of a preprocessed dataset: the recorded
:class:`PreprocessingConfig`, the stamp written beside the rows, and the compatibility verdicts the
training path reads. Split from the bake (:mod:`src.data.pipeline.preprocessing`) so a run that only
reads the stamp does not import the tokenizers, collators and VLM machinery that wrote the rows.
"""

import json
import logging
import os
from dataclasses import MISSING, asdict, dataclass, field, is_dataclass
from dataclasses import fields as dataclass_fields
from datetime import UTC, datetime
from typing import Any

from huggingface_hub import hf_hub_download
from huggingface_hub.errors import EntryNotFoundError, RepositoryNotFoundError

from src.data.shard_index import (
    PREPROCESSING_VERSION,
    IncompatiblePreprocessedDataset,
    reject_incompatible_stamp,
    write_stamped_sidecar,
)
from src.data.sources.paths import METADATA_FILE, hub_repo_id, parse_dataset_source
from src.data.sources.s3_client import read_control_json_with_cache
from src.models.loading.tokenizer_setup import load_chat_template

logger = logging.getLogger(__name__)

_PREPROCESSING_MODES = ("chat", "text")

# Per-field metadata rather than hand-kept lists, so a new knob states its own applicability where it
# is defined. ``modes``: a knob set outside the mode that consumes it does not do what it says, and
# ``__post_init__`` refuses it. ``render_check=False`` exempts a field from
# :func:`_validate_render_compatibility`; every other field is compared, because preprocessed rows are
# baked and a differing runtime render knob would silently train something other than the YAML.
_CHAT_ONLY = {"modes": ("chat",)}
_TEXT_ONLY = {"modes": ("text",)}
_NO_RENDER_CHECK = {"render_check": False}


@dataclass
class PreprocessingConfig:
    """Configuration for dataset preprocessing.

    ``__post_init__`` rejects an unknown ``mode`` and any knob set outside the mode that consumes it
    (fields declare their own applicability via the ``modes`` field metadata).
    """

    # Compared separately in validate_preprocessing_compatibility: the model warns (a different
    # checkpoint can share the tokenizer), max_length raises with direction-specific messages.
    model_name_or_path: str = field(metadata=_NO_RENDER_CHECK)

    max_length: int = field(default=8192, metadata=_NO_RENDER_CHECK)
    conversation_field: str = field(default="prompt", metadata=_CHAT_ONLY)
    system_prompt: str | None = field(default=None, metadata=_CHAT_ONLY)
    model_supports_system_role: bool = field(default=True, metadata=_CHAT_ONLY)
    tools_field: str | None = field(default=None, metadata=_CHAT_ONLY)
    interleaved_thinking: bool = field(default=False, metadata=_CHAT_ONLY)
    # Top-level image column merged into the conversation, exactly as the runtime VLM row processor
    # does: hub VLM datasets keep images outside the messages, so without this the column is dropped
    # and the artifact bakes text.
    images_field: str | None = field(default=None, metadata=_CHAT_ONLY)

    # "chat" = chat-template conversation_field (SFT); "text" = raw text_field documents (pretraining).
    # No render check on the three: the training scripts expose no mode / text_field / append_eos, so
    # a text-mode artifact's tokenization is entirely baked.
    mode: str = field(default="chat", metadata=_NO_RENDER_CHECK)
    text_field: str = field(default="text", metadata=_TEXT_ONLY | _NO_RENDER_CHECK)
    # text mode only: document boundary marker
    append_eos: bool = field(default=True, metadata=_TEXT_ONLY | _NO_RENDER_CHECK)

    # Default False, unlike the training side: it reads as "not set" for the mode check above, and a
    # True here is inert without assistant_message_template (prepare_dataset's CLI defaults it ON and
    # demands the marker). No render check — the baked-label mismatch raises instead.
    train_on_completions_only: bool = field(default=False, metadata=_CHAT_ONLY | _NO_RENDER_CHECK)
    assistant_message_template: str | None = field(default=None, metadata=_CHAT_ONLY)

    # No render check: covered by validate_preprocessing_compatibility's packed warning; the strategy
    # is baked into the rows and the collator follows metadata.packed, never the runtime flag.
    pack_sequences: bool = field(default=False, metadata=_NO_RENDER_CHECK)
    # TRL pack_dataset: "bfd", "bfd_split" or "wrapped"
    packing_strategy: str = field(default="bfd", metadata=_NO_RENDER_CHECK)

    # Tokenizer mutations, recorded because they change the produced ids: an EOS override moves the
    # completion-mask boundaries and a template override re-renders every turn, yet the artifact is
    # otherwise byte-indistinguishable from an unmutated one.
    pad_token: str | None = None
    eos_token: str | None = None
    bos_token: str | None = None
    chat_template: str | None = None  # RESOLVED template text, never the file path it may have come from

    # Prep-only execution knobs with no effect on the produced tokens, hence no render check:
    # sharding/process counts change layout only, and a non-hf backend emits identical ids.
    num_shards: int = field(default=1, metadata=_NO_RENDER_CHECK)  # 1 = no sharding
    num_proc: int | None = field(default=None, metadata=_NO_RENDER_CHECK)  # None = the toolkit default
    tokenizer_backend: str = field(default="hf", metadata=_NO_RENDER_CHECK)  # "hf" or "gigatoken"

    # No render check: the consuming VLM branch already raises on a non-VLM artifact, and the pixel
    # budget is baked into the stored pixel_values.
    is_vlm: bool = field(default=False, metadata=_NO_RENDER_CHECK)
    min_pixels: int | None = field(default=None, metadata=_NO_RENDER_CHECK)
    max_pixels: int | None = field(default=None, metadata=_NO_RENDER_CHECK)

    def __post_init__(self) -> None:
        if self.mode not in _PREPROCESSING_MODES:
            raise ValueError(
                f"Invalid preprocessing mode '{self.mode}'. Expected one of {list(_PREPROCESSING_MODES)}."
            )

        inapplicable = sorted(
            f.name
            for f in dataclass_fields(self)
            if self.mode not in f.metadata.get("modes", _PREPROCESSING_MODES) and getattr(self, f.name) != f.default
        )
        if inapplicable:
            raise ValueError(
                f"{inapplicable} {'is' if len(inapplicable) == 1 else 'are'} not applicable to "
                f"mode='{self.mode}' and would be silently ignored: a mode='text' run renders no chat "
                f"template, and a mode='chat' run tokenizes no raw-text field. Drop the setting, or "
                f"switch mode."
            )

        if self.images_field and not self.is_vlm:
            raise ValueError(
                f"images_field='{self.images_field}' names an image column, but is_vlm=False: only the "
                f"VLM tokenization merges that column into the conversation, so the text path would "
                f"drop it and bake a text-only dataset. Set is_vlm=True (prepare_dataset's --vlm), or "
                f"drop images_field."
            )

        if self.train_on_completions_only and not self.assistant_message_template:
            raise ValueError(
                "train_on_completions_only=True requires assistant_message_template (the response "
                "marker, e.g. '<|im_start|>assistant\\n') so completion-only labels can be built. "
                "Pass the marker your model's chat template renders, or set "
                "train_on_completions_only=False to train on the full sequence."
            )


# The render knobs a run's args are compared against (:func:`_validate_render_compatibility`) —
# every field that did not declare itself exempt, so a newly added knob is compared by default.
_RENDER_CHECKED_FIELDS = tuple(
    f.name for f in dataclass_fields(PreprocessingConfig) if f.metadata.get("render_check", True)
)


@dataclass
class PreprocessedDatasetMetadata:
    """Metadata for a preprocessed dataset."""

    version: str = PREPROCESSING_VERSION
    preprocessed: bool = True

    model_name: str = ""
    tokenizer_vocab_size: int = 0

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

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def claims_payload(cls, data: Any) -> bool:
        """Whether a ``metadata.json`` payload is a toolkit preprocessing stamp at all.

        ``metadata.json`` is a common file name: a hub dataset may ship its own (a card, a license
        blob, a producer's provenance) beside raw rows. Judging such a payload against this build's
        schema turns an ordinary raw dataset into a hard startup failure, so identity comes first —
        keyed on the field every stamp this toolkit ever wrote carries, which is also the one the
        detection verdict reads.
        """
        return isinstance(data, dict) and "preprocessed" in data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PreprocessedDatasetMetadata":
        """Create from a ``metadata.json`` payload, refusing a stamp this build cannot read.

        An unreadable stamp — a bumped version, a retired or unknown key — raises rather than
        downgrading to the raw path, which would silently re-tokenize pre-tokenized rows. Retired
        spellings are not migrated: a stamp is the record of what was baked, and re-spelling one
        would claim a knob was recorded when nothing verified it. Only payloads this toolkit wrote
        (:meth:`claims_payload`) reach here; a foreign one is not this schema's business.
        """
        reject_incompatible_stamp(cls, data, "Preprocessed dataset metadata")
        return cls(**data)

    def save(self, path: str) -> None:
        """Save metadata to JSON file."""
        write_stamped_sidecar(path, self.to_dict())


def _read_metadata_payload(path: str, *, best_effort: bool) -> dict[str, Any] | None:
    """Raw ``metadata.json`` payload for a dataset path (local, S3 or Hub), or None if there is none.

    One reader behind both the detection probe and the metadata load, so the three sources cannot
    drift apart. ``best_effort`` (the probe's contract) turns a transient failure into None after a
    warning — an ABSENT metadata.json is the normal raw-dataset case and stays silent.
    """
    source_type, bucket, key = parse_dataset_source(path)
    try:
        if source_type == "s3":
            # Mirrored read: an S3/SSO outage serves the local mirror, so a warm-cache run keeps
            # classifying as preprocessed instead of degrading to "raw" and dying on the stripped
            # source columns. A live absence still raises FileNotFoundError, the raw-dataset case.
            try:
                return read_control_json_with_cache(bucket, f"{key.rstrip('/')}/{METADATA_FILE}")
            except FileNotFoundError:
                return None

        if source_type == "hf_hub":
            local_path = hf_hub_download(hub_repo_id(path), METADATA_FILE, repo_type="dataset")
            with open(local_path) as f:
                return json.load(f)

        metadata_path = os.path.join(path, METADATA_FILE)
        if not os.path.exists(metadata_path):
            return None
        with open(metadata_path) as f:
            return json.load(f)

    except (EntryNotFoundError, RepositoryNotFoundError):
        # A Hub dataset without metadata.json is simply a raw dataset — the common case, not an error.
        return None
    except Exception as e:  # transient creds/throttle/torn read must not kill one rank
        if not best_effort:
            raise
        logger.warning(f"Metadata probe for {path} failed ({type(e).__name__}: {e}); treating as raw.")
        return None


def is_preprocessed_dataset(path: str) -> bool:
    """True if ``path`` holds a toolkit ``metadata.json`` with ``preprocessed=True``.

    Absent, unreadable, or someone else's ``metadata.json`` errs toward "raw"; the caller reconciles
    the verdict across ranks. This probe never raises — it runs per rank right before a consensus
    all-reduce, so a rank-local raise would block its peers there. An INCOMPATIBLE toolkit stamp
    therefore reports True: every rank takes the same branch and
    :func:`load_preprocessed_metadata` raises the version error on all of them, rather than the run
    silently re-tokenizing pre-tokenized rows.
    """
    payload = _read_metadata_payload(path, best_effort=True)
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
    """Load metadata from a preprocessed dataset (local, S3 or HF Hub)."""
    payload = _read_metadata_payload(path, best_effort=False)
    if payload is None:
        raise FileNotFoundError(f"No {METADATA_FILE} at {path}: not a preprocessed dataset.")
    if not PreprocessedDatasetMetadata.claims_payload(payload):
        raise FileNotFoundError(
            f"{METADATA_FILE} at {path} is not a toolkit preprocessing stamp (no 'preprocessed' "
            f"field): the dataset is raw and carries someone else's metadata file."
        )
    return PreprocessedDatasetMetadata.from_dict(payload)


def _stated_render_knobs(render_args: Any) -> frozenset[str] | None:
    """Knob names the run's args actually STATE — those holding something other than their default.

    ``None`` when the defaults cannot be read off ``render_args`` (not a dataclass) — treat every
    knob as stated. A value the YAML never set makes no claim about the baked artifact, and the two
    sides' defaults move independently, so comparing default against default would turn every
    artifact prepared before a default moved into a hard startup failure. An explicitly set knob
    still raises: that YAML says something the run cannot deliver.
    """
    if not is_dataclass(render_args) or isinstance(render_args, type):
        return None
    stated: set[str] = set()
    for f in dataclass_fields(type(render_args)):
        default = f.default_factory() if f.default_factory is not MISSING else f.default
        if default is MISSING or getattr(render_args, f.name, default) != default:
            stated.add(f.name)
    return frozenset(stated)


def _validate_render_compatibility(metadata: PreprocessedDatasetMetadata, render_args: Any) -> None:
    """Raise when a render knob the run STATES differs from the value baked into the artifact.

    The checked set is :data:`_RENDER_CHECKED_FIELDS`, derived from the fields' own ``render_check``
    metadata, so a newly added render knob is compared by default instead of silently skipped. Knobs the metadata
    predates, that ``render_args`` does not carry, or that the run leaves at its own default
    (:func:`_stated_render_knobs`) are reported rather than raised — the prepared value is what
    trained either way.
    """
    recorded = metadata.config or {}
    if recorded.get("mode") == "text":
        # Raw-text pretraining artifacts render no chat template: every knob below was inert at
        # preparation time, so there is nothing baked for the run's values to disagree with.
        return
    checked = _RENDER_CHECKED_FIELDS
    stated = _stated_render_knobs(render_args)

    unrecorded = [name for name in checked if name not in recorded and hasattr(render_args, name)]
    if unrecorded:
        logger.warning(
            f"Preprocessed dataset metadata predates recording of {unrecorded}; cannot verify these "
            f"render settings against the training config."
        )

    mismatched: dict[str, tuple[Any, Any]] = {}
    unstated: dict[str, tuple[Any, Any]] = {}
    for name in checked:
        if name not in recorded or not hasattr(render_args, name):
            continue
        if name == "assistant_message_template" and not metadata.train_on_completions_only:
            # Without completion masking the template touches nothing baked — a differing value
            # changes neither the prepared labels nor the run (the earlier baked-label check
            # already forces the masking flags to agree).
            continue
        run_value = getattr(render_args, name)
        if name == "chat_template":
            # Both sides may spell the same template as a path or as the text itself; the config
            # records the RESOLVED text, so resolve the run's value through the same helper rather
            # than reporting a path-vs-text difference as a template change.
            run_value = load_chat_template(run_value) if run_value else None
        if recorded[name] != run_value:
            target = mismatched if stated is None or name in stated else unstated
            target[name] = (recorded[name], run_value)

    if unstated:
        detail = "; ".join(
            f"{name}: prepared={prep!r} vs this run's default={run!r}" for name, (prep, run) in unstated.items()
        )
        logger.warning(
            f"Preprocessed dataset was prepared with render settings the training config does not "
            f"state ({detail}). Rendering is baked, so the PREPARED value is what trained; set these "
            f"explicitly in the YAML to silence this."
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


def validate_preprocessing_compatibility(
    metadata: PreprocessedDatasetMetadata,
    required_max_length: int,
    required_model: str | None = None,
    required_train_on_completions_only: bool | None = None,
    render_args: Any | None = None,
    required_packing: bool | None = None,
) -> None:
    """Raise on a max_length, baked-label-masking, or render-knob mismatch; warn on model mismatch
    and on ``packing`` requested against an unpacked artifact.

    ``render_args`` is the run's script-args object; every :class:`PreprocessingConfig` render knob
    it carries (minus the separately-checked/exempt fields) is compared against the recorded
    metadata. ``required_packing`` is the run's ``packing`` flag.
    """
    if metadata.max_length < required_max_length:
        raise ValueError(
            f"Preprocessed dataset max_length ({metadata.max_length}) is less than "
            f"required max_length ({required_max_length}). Re-preprocess with larger max_length."
        )
    if metadata.max_length > required_max_length:
        # The dangerous direction: rows are already baked at metadata.max_length and never re-truncated
        # at runtime, so training would silently exceed the configured activation budget.
        raise ValueError(
            f"Preprocessed dataset max_length ({metadata.max_length}) exceeds the configured "
            f"max_length ({required_max_length}) and preprocessed rows are not re-truncated at "
            f"runtime. Re-preprocess at {required_max_length}, or raise max_length to match."
        )

    if required_train_on_completions_only is not None:
        # Labels are baked at preparation time, so a mismatch silently trains the opposite of the YAML.
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
        _validate_render_compatibility(metadata, render_args)

    if required_packing and not metadata.packed:
        logger.warning(
            "The training config sets packing=True but the preprocessed dataset was prepared "
            "UNPACKED (metadata.packed=False). Preprocessed rows are never packed at runtime, so "
            "this run trains on individually padded rows without packing's throughput — "
            "re-preprocess with --pack-sequences to actually pack."
        )
