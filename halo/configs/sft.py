"""SFT script arguments, the conversation-rendering fields both distillation scripts share, tree packing, MTP head
training and datatrove token sources."""

from collections.abc import Mapping
from dataclasses import MISSING, dataclass, field, fields
from typing import Any, ClassVar, get_args

from halo.config_validation import (
    ValidatedConfig,
    differs_from_default,
    present,
    reject_non_default_args,
    require_finite,
    require_text,
)
from halo.configs.script_arguments import (
    CommonScriptArguments,
    CrossDocumentLinearAttentionArguments,
    GenerationEvalArguments,
    VisionTrainingArguments,
)
from halo.data.conversation import TEMPLATE_COLUMN_KNOBS
from halo.data.sources.token_mix import TokenSource
from halo.models.attention.tree_plan import FullAttentionForm
from halo.models.mtp.spec import MtpHeadRequest, MtpLossType, MtpObjective, MtpScope


@dataclass
class ConversationRenderArguments(ValidatedConfig):
    conversation_field: str = field(
        default="prompt", metadata={"help": "Dataset column holding the conversation, a list of role/content dicts."}
    )
    images_field: str | None = field(
        default=None,
        metadata={
            "help": "VLM only: dataset column holding the row's image(s) (HF Image feature, single or list). "
            "Injected into the first user turn ahead of its text, so hub datasets that store images "
            "outside the conversation (e.g. FineVision/the_cauldron/Docmatix) train without preprocessing."
        },
    )
    system_prompt: str | None = field(
        default=None, metadata={"help": "System prompt added to a conversation that has none; unset adds nothing."}
    )
    train_on_completions_only: bool = field(
        default=True, metadata={"help": "Train only on assistant completions, masking the prompt tokens."}
    )
    assistant_message_template: str | None = field(
        default=None,
        metadata={
            "help": "The rendered assistant-turn prefix of the model's chat template, like '<|im_start|>assistant\\n'. "
            "Required when train_on_completions_only is on, since no default fits every template, and refused when "
            "it is off."
        },
    )
    model_supports_system_role: bool = field(
        default=True,
        metadata={
            "help": "Whether the chat template accepts a system role; false prepends the system prompt to the first "
            "user message."
        },
    )
    interleaved_thinking: bool = field(
        default=False,
        metadata={
            "help": "Pass clear_thinking=False to apply_chat_template so earlier assistant turns keep their "
            "reasoning; only templates with that switch (the GLM family) read it. Text-only: the VLM path rejects "
            "it."
        },
    )

    def _validate_fields(self) -> None:
        super()._validate_fields()
        owner = type(self).__name__
        # The rows are rendered from this column, and a null would reach the loader as no column to check.
        require_text(owner, conversation_field=self.conversation_field)
        if self.assistant_message_template is not None and not self.train_on_completions_only:
            raise ValueError(
                f"{owner}: assistant_message_template marks the completions train_on_completions_only trains on, and "
                f"that is off, so nothing would read it. Remove it, or set train_on_completions_only: true."
            )


@dataclass
class _TreePackingArguments:
    """``tree_packing`` and the ``tree_*`` options it gates (``agent-docs/optimization/tree-packing.md``)."""

    tree_packing: bool = field(
        default=False,
        metadata={
            "help": "Pack rows that share a prefix as a tree: the shared part is computed once and each row's "
            "remainder attends it and itself only. The same examples as flat packing, at fewer tokens. "
            "Requires packing and tree_share_markers."
        },
    )
    tree_share_markers: list[str] = field(
        default_factory=list,
        metadata={
            "help": "Tree packing: rows whose message before the final turn is identical up to and including one of "
            "these markers are examined together for a shared prefix."
        },
    )
    tree_share_requires: str | None = field(
        default=None,
        metadata={
            "help": "Tree packing: a row is shared only if this text occurs before its marker, e.g. the heading of "
            "the long part, so rows that put it after the marker are left alone instead of sharing a short header."
        },
    )
    tree_full_attention: FullAttentionForm | None = field(
        default=None,
        metadata={
            "help": "Tree packing, full-attention layers: 'split' adds a rectangular ancestor pass to the stock "
            "packed call (flash_attention_4, except its head_dim-256 kernel), 'gather' builds one key sequence per "
            "tree node (flash_attention_2 as well; memory grows with fan-out x prefix length). Unset takes 'split' "
            "where the model's backend runs it, else 'gather'. Refused on GLM-5 Next, whose layers attend in their "
            "own forward."
        },
    )


# What a run that packs no trees must find at its default: set there, these options do nothing.
TREE_PACKING_FIELDS = tuple(f.name for f in fields(_TreePackingArguments))


@dataclass
class MtpArguments(ValidatedConfig):
    """Train the family's released MTP head (``agent-docs/optimization/mtp-heads.md``)."""

    mtp_loss_weight: float = field(
        default=0.0,
        metadata={
            "help": "Weight of the MTP head's cross-entropy beside the next-token loss. Above 0 builds the family's "
            "released head and trains it; 0 builds nothing and adds no hook."
        },
    )
    mtp_loss_type: MtpLossType = field(
        default="hard",
        metadata={
            "help": "hard: the head fits the gold token two ahead. soft: the same loss value, with the head's gradient "
            "routed to the trunk's own next-token distribution."
        },
    )
    mtp_train_scope: MtpScope = field(
        default="joint",
        metadata={
            "help": "joint: the head's term reaches the head, trunk, embedding and output embedding. detached: the "
            "head alone, while the trunk trains on the next-token loss. head_only: detached over a frozen trunk."
        },
    )
    mtp_head_source: str | None = field(
        default=None,
        metadata={
            "help": "Checkpoint to read the head from when the model's own is stale or absent; a hub id or a local "
            "directory. A resume from a checkpoint keeps its trained head and ignores it."
        },
    )
    mtp_min_acceptance_gain: float | None = field(
        default=0.0,
        metadata={
            "help": "head_only: the least gain over the first held-out acceptance a finished fit must show, or the "
            "run exits non-zero, naming the checkpoint that holds the rejected head. null turns the gate off."
        },
    )

    def _validate_fields(self) -> None:
        super()._validate_fields()
        owner = type(self).__name__
        require_finite(owner, ge=0, mtp_loss_weight=self.mtp_loss_weight)
        require_finite(owner, **present(mtp_min_acceptance_gain=self.mtp_min_acceptance_gain))
        if self.mtp_loss_type not in get_args(MtpLossType):
            raise ValueError(
                f"{owner}: mtp_loss_type must be one of {get_args(MtpLossType)}, got {self.mtp_loss_type!r}."
            )
        if self.mtp_train_scope not in get_args(MtpScope):
            raise ValueError(
                f"{owner}: mtp_train_scope must be one of {get_args(MtpScope)}, got {self.mtp_train_scope!r}."
            )
        if self.mtp_loss_weight == 0:
            # Every one of them would be silently inert with no head built.
            reject_non_default_args(
                "Halo SFT with mtp_loss_weight: 0", self, *(name for name in MTP_FIELDS if name != "mtp_loss_weight")
            )
        elif self.mtp_train_scope != "head_only" and differs_from_default(self, "mtp_min_acceptance_gain"):
            raise ValueError(
                f"{owner}: mtp_min_acceptance_gain gates a head_only fit, and mtp_train_scope is "
                f"{self.mtp_train_scope!r}, where the trunk moves under the head. Remove it, or set "
                "mtp_train_scope: head_only."
            )

    @property
    def mtp_head_request(self) -> MtpHeadRequest | None:
        """``None`` at weight 0, which leaves the load exactly as a run with no head takes it."""
        if self.mtp_loss_weight == 0:
            return None
        objective = MtpObjective(self.mtp_loss_weight, self.mtp_loss_type, self.mtp_train_scope)
        return MtpHeadRequest(objective, self.mtp_head_source)

    @property
    def gates_mtp_acceptance(self) -> bool:
        return (
            self.mtp_loss_weight > 0
            and self.mtp_train_scope == "head_only"
            and self.mtp_min_acceptance_gain is not None
        )


# What a run with no head must find at its default: set there, these options do nothing.
MTP_FIELDS = tuple(f.name for f in fields(MtpArguments))

_TOKEN_SOURCE_KEYS = frozenset(f.name for f in fields(TokenSource))
_TOKEN_SOURCE_REQUIRED = sorted(f.name for f in fields(TokenSource) if f.default is MISSING)


def _token_sources(owner: str, knob: str, entries: list[dict[str, Any]] | None) -> list[TokenSource] | None:
    """``entries`` as sources, refusing a list the mix cannot honor as written: anything but a non-empty list of
    mappings of known keys, a blank name or path, a repeated name, a weight that is not a positive finite number, or a
    weight on some sources only."""
    if entries is None:
        return None
    if not isinstance(entries, list) or not entries:
        raise ValueError(
            f"{owner}: {knob} must be a non-empty list of mappings with 'name' and 'path', got {entries!r}"
        )
    sources = []
    for index, entry in enumerate(entries):
        where = f"{knob}[{index}]"
        if not isinstance(entry, Mapping):
            raise ValueError(f"{owner}: {where} must be a mapping with 'name' and 'path', got {entry!r}")
        unknown = sorted(map(str, set(entry) - _TOKEN_SOURCE_KEYS))
        if unknown:
            raise ValueError(f"{owner}: {where} has unknown keys {unknown}; allowed: {sorted(_TOKEN_SOURCE_KEYS)}")
        missing = [key for key in _TOKEN_SOURCE_REQUIRED if key not in entry]
        if missing:
            raise ValueError(f"{owner}: {where} lacks {missing}; every source needs {_TOKEN_SOURCE_REQUIRED}.")
        source = TokenSource(**entry)
        require_text(owner, **{f"{where}.name": source.name, f"{where}.path": source.path})
        require_finite(owner, gt=0, **present(**{f"{where}.weight": source.weight}))
        sources.append(source)
    names = [source.name for source in sources]
    repeated = sorted({name for name in names if names.count(name) > 1})
    if repeated:
        raise ValueError(
            f"{owner}: {knob} repeats the names {repeated}; a name keys its source in the checkpoint's mix."
        )
    weighted = [source.name for source in sources if source.weight is not None]
    if weighted and len(weighted) != len(sources):
        raise ValueError(
            f"{owner}: {knob} weighs {weighted} and leaves the others unweighted, so their shares are undefined. Weigh "
            f"every source, or none for each source's share of the windows."
        )
    return sources


@dataclass
class TokenSourceArguments(ValidatedConfig):
    """datatrove DocumentTokenizer folders trained as a weighted mix (``agent-docs/training-methods/pretraining.md``)."""

    token_sources: list[dict[str, Any]] | None = field(
        default=None,
        metadata={
            "help": "Train on datatrove DocumentTokenizer output folders instead of dataset: each a mapping with "
            "'name', 'path' (local or s3://) and an optional relative 'weight' (on every source or none), mixed into "
            "windows of max_length tokens. "
            "See agent-docs/training-methods/pretraining.md#training-from-datatrove-token-folders."
        },
    )
    eval_token_sources: list[dict[str, Any]] | None = field(
        default=None,
        metadata={
            "help": "Held-out datatrove folders of a token_sources run, each a mapping with 'name' and 'path', read "
            "whole at every evaluation and reported as eval_<name>_loss."
        },
    )

    def _validate_fields(self) -> None:
        super()._validate_fields()
        owner = type(self).__name__
        # Parsing the lists refuses a malformed one.
        mixed, held_out = self.token_source_specs, self.eval_token_source_specs
        if held_out is None:
            return
        if mixed is None:
            raise ValueError(
                f"{owner}: eval_token_sources holds out folders of a token_sources run; set token_sources."
            )
        weighted = [source.name for source in held_out if source.weight is not None]
        if weighted:
            raise ValueError(
                f"{owner}: eval_token_sources {weighted} carry a weight, but a held-out folder is read whole at every "
                f"evaluation, unweighted. Remove the weights."
            )

    @property
    def token_source_specs(self) -> list[TokenSource] | None:
        return _token_sources(type(self).__name__, "token_sources", self.token_sources)

    @property
    def eval_token_source_specs(self) -> list[TokenSource] | None:
        return _token_sources(type(self).__name__, "eval_token_sources", self.eval_token_sources)


# What a token_sources run must find at its default: its rows are token windows, so no dataset loads and no
# conversation renders. train_on_completions_only is refused on its own, since its default is on.
_TOKEN_RUN_INERT_FIELDS = (
    "dataset",
    "dataset_ratio",
    "test_size",
    "train_on_last_assistant_only",
    "generate_eval_examples",
    *TEMPLATE_COLUMN_KNOBS,
    *(f.name for f in fields(ConversationRenderArguments) if f.name != "train_on_completions_only"),
    *(f.name for f in fields(VisionTrainingArguments)),
    *TREE_PACKING_FIELDS,
)


@dataclass
class SFTScriptArguments(
    TokenSourceArguments,
    MtpArguments,
    _TreePackingArguments,
    CrossDocumentLinearAttentionArguments,
    VisionTrainingArguments,
    ConversationRenderArguments,
    GenerationEvalArguments,
    CommonScriptArguments,
):
    PROJECT_NAME: ClassVar[str] = "sft-tuning"

    # SFT does not generate during eval by default (the preference/GRPO trainers do).
    generate_eval_examples: bool = field(default=False, metadata={"help": "Generate text examples during evaluation."})
    train_on_last_assistant_only: bool = field(
        default=False,
        metadata={
            "help": "If True, train only on the LAST assistant message instead of all. "
            "Requires train_on_completions_only=True."
        },
    )

    def _validate_fields(self) -> None:
        super()._validate_fields()
        if self.token_sources is None:
            return
        if self.train_on_completions_only:
            raise ValueError(
                f"{type(self).__name__}: token_sources rows carry no completion marks, so every token trains. Set "
                f"train_on_completions_only: false."
            )
        reject_non_default_args(
            "Halo SFT on token_sources (its rows are token windows: no dataset loads and no conversation renders)",
            self,
            *_TOKEN_RUN_INERT_FIELDS,
        )
