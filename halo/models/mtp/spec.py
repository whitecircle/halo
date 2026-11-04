"""Where each family's released MTP head sits and how its chain is spelled, why a tail Halo cannot train as it ships is
refused, and which checkpoint keys a class's load patterns name as a tail."""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

from halo.models.config_access import config_sources, text_config
from halo.models.moe.family_specs import BAILING, DEEPSEEK_V4, GLM5_NEXT, INKLING, NEMOTRON_H, QWEN4_EXP, STEP3P7

MTP_LAYER_COUNT_FIELDS: tuple[str, ...] = ("num_nextn_predict_layers", "mtp_num_hidden_layers", "num_mtp_layers")
# Every spec's release stores one draft layer; a config declaring no count (Qwen3-Next) reads as the engines read it.
_RELEASED_TAIL_DEPTH = 1
# A load-ignore pattern naming a tail: an ``mtp`` module or a fixed decoder index past the trunk. A class's other
# entries (the vision tower a text-only class drops) do not match, and ``layers\.\d+`` names no index.
_TAIL_PATTERN = re.compile(r"mtp|layers\\?\.\d")
# A stored key under an ``mtp`` module, and a decoder list's entry by its index.
_MTP_MODULE_KEY = re.compile(r"(?:^|\.)mtp\.")
_DECODER_ENTRY_KEY = re.compile(r"(?:^|\.)layers\.(\d+)\.")
# The attention a release's draft layer runs, which decides the trunk layer it takes its class and call from.
_DRAFT_LAYER_TYPE = "full_attention"
# The attribute a ``root`` head hangs off the model under, as the release's keys spell it.
ROOT_HEAD_ATTR = "mtp"
# The chain's decoder layer inside a ``root`` head.
ROOT_HEAD_LAYERS_ATTR = "layers"

UNSURVEYED_TAIL = (
    "this family is outside the supported roster, so no release of it has had its tail checked key by key against "
    "its decoder layer"
)


MtpScope = Literal["joint", "detached", "head_only"]
MtpLossType = Literal["hard", "soft"]


@dataclass(frozen=True)
class MtpObjective:
    """``scope`` is what the head's term may reach: ``joint`` head, trunk, embedding and output embedding; ``detached``
    the head alone; ``head_only`` the head alone over a frozen trunk. ``soft`` routes the head's gradient toward the
    next-token distribution, a KL at the same step whose target is the trunk's own."""

    weight: float
    loss_type: MtpLossType = "hard"
    scope: MtpScope = "joint"


@dataclass(frozen=True)
class MtpHeadRequest:
    """What a run asks of the head: its objective, and the checkpoint to read a tail from over the model's own."""

    objective: MtpObjective
    source: str | None = None


@dataclass(frozen=True)
class MtpHeadSpec:
    """Layer class, index, config and norm class are read off the trunk; only what a release spells its own way is
    declared. ``placement`` is ``root`` for ``mtp.*`` beside the backbone and ``list`` for the decoder list's entry at
    ``num_hidden_layers``. The module paths are relative to the head."""

    model_types: tuple[str, ...]
    placement: Literal["root", "list"]
    projection: str
    embedding_norm: str
    hidden_norm: str
    output_norm: str
    # (tail key under the head, the trunk embedding it copies): stored by the release, shared with the trunk in training.
    trunk_copies: tuple[tuple[str, Literal["input", "output"]], ...] = ()
    # The release's drafter reads zeros for the embedding at position 0, which holds no draft context.
    zero_first_embedding: bool = False


MTP_HEAD_SPECS: tuple[MtpHeadSpec, ...] = (
    # Qwen3.6 and Qwen3.8-27B ship under these same types.
    MtpHeadSpec(
        model_types=("qwen3_5", "qwen3_5_text", "qwen3_5_moe", "qwen3_5_moe_text", "qwen3_next"),
        placement="root",
        projection="fc",
        embedding_norm="pre_fc_norm_embedding",
        hidden_norm="pre_fc_norm_hidden",
        output_norm="norm",
    ),
    MtpHeadSpec(
        model_types=("glm4_moe_lite",),
        placement="list",
        projection="eh_proj",
        embedding_norm="enorm",
        hidden_norm="hnorm",
        output_norm="shared_head.norm",
        trunk_copies=(("embed_tokens.weight", "input"), ("shared_head.head.weight", "output")),
        zero_first_embedding=True,
    ),
)

MTP_REFUSALS: dict[str, str] = {
    **dict.fromkeys(
        BAILING.model_types,
        "its remote-code forward runs its own head; Ling-3.0-flash trains that head through model_init_kwargs: "
        "{mtp_loss_scaling_factor: <weight>} at one row per batch",
    ),
    **dict.fromkeys(
        DEEPSEEK_V4.model_types,
        "its tail stores vendor names and quantized experts, and its BF16 converter drops the tail",
    ),
    **dict.fromkeys(
        GLM5_NEXT.model_types,
        "its decoder class builds attn_hc and ffn_hc layers the release's tail lacks, and the release stores the tail "
        "in fp8",
    ),
    **dict.fromkeys(
        INKLING.model_types,
        "its release stores eight draft layers under names that need renames",
    ),
    **dict.fromkeys(
        NEMOTRON_H.model_types,
        "its config declares mtp_layers_block_type [attention, moe], a draft chain of two blocks of different kinds, "
        "which one decoder layer does not build",
    ),
    **dict.fromkeys(
        QWEN4_EXP.model_types,
        "its draft layer reads hyper-connection streams the trunk's layers carry between them, which an embedding and "
        "a hidden state do not supply",
    ),
    **dict.fromkeys(
        STEP3P7.model_types,
        "each of its draft layers owns its own head, so the soft objective's cancellation does not hold, and its "
        "attention widths differ from the trunk's layers",
    ),
    **dict.fromkeys(
        (
            "deepseek_v3",
            "deepseek_v32",
            "ernie4_5_moe",
            "exaone4_5",
            "exaone_moe",
            "glm4_moe",
            "glm4v_moe",
            "glm_moe_dsa",
            "glm_ocr",
            "hy_v3",
            "hy_v4",
            "longcat_flash",
            "mellum",
            "mimo_v2_flash",
            "minimax_m3_vl",
            "minimax_m3_vl_text",
            "olmo_hybrid",
        ),
        UNSURVEYED_TAIL,
    ),
}


def _index_by_model_type(specs: tuple[MtpHeadSpec, ...]) -> dict[str, MtpHeadSpec]:
    by_model_type: dict[str, MtpHeadSpec] = {}
    for spec in specs:
        for model_type in spec.model_types:
            if by_model_type.setdefault(model_type, spec) is not spec:
                raise ValueError(f"model_type {model_type!r} is claimed by two MTP head specs")
            if model_type in MTP_REFUSALS:
                raise ValueError(f"model_type {model_type!r} has both an MTP head spec and a refusal")
    return by_model_type


_SPECS_BY_MODEL_TYPE = _index_by_model_type(MTP_HEAD_SPECS)


def mtp_tail_depth(config) -> int:
    text = text_config(config)
    return next(
        (depth for field in MTP_LAYER_COUNT_FIELDS if (depth := getattr(text, field, None)) is not None),
        _RELEASED_TAIL_DEPTH,
    )


def declared_tail_fields(config) -> dict[str, int]:
    """The count fields ``config`` declares draft layers under; a missing field and an explicit 0 declare none."""
    text = text_config(config)
    return {field: depth for field in MTP_LAYER_COUNT_FIELDS if (depth := getattr(text, field, None))}


def mtp_draft_layer_index(text) -> int:
    """The last trunk layer running the draft layer's attention, whose class and call arguments the head reuses."""
    kinds = getattr(text, "layer_types", None)
    if not kinds:
        return text.num_hidden_layers - 1
    running = [index for index, kind in enumerate(kinds[: text.num_hidden_layers]) if kind == _DRAFT_LAYER_TYPE]
    if not running:
        raise ValueError(
            f"mtp_loss_weight builds the head's draft layer from the trunk's last {_DRAFT_LAYER_TYPE} layer, and this "
            f"config's layer_types hold none: {sorted(set(kinds))}. Train from a checkpoint whose config keeps one."
        )
    return running[-1]


def require_mtp_head_spec(config) -> MtpHeadSpec:
    """The spec of ``config``'s family, the wrapper's ``model_type`` first; raises naming why it has none."""
    levels = [getattr(level, "model_type", None) for level in config_sources(config)]
    for model_type in levels:
        if model_type in MTP_REFUSALS:
            raise ValueError(f"mtp_loss_weight cannot train {model_type!r}: {MTP_REFUSALS[model_type]}.")
        spec = _SPECS_BY_MODEL_TYPE.get(model_type)
        if spec is None:
            continue
        depth = mtp_tail_depth(config)
        if depth != _RELEASED_TAIL_DEPTH:
            raise ValueError(
                f"mtp_loss_weight trains the {_RELEASED_TAIL_DEPTH}-layer head {model_type!r} releases ship, and this "
                f"config declares {depth}. Train from a checkpoint whose config declares the release's depth."
            )
        mtp_draft_layer_index(text_config(config))
        return spec
    raise ValueError(
        f"mtp_loss_weight has no head to train on {levels[0]!r}: no release of this family ships a tail Halo maps. "
        "Remove mtp_loss_weight."
    )


def find_mtp_head_spec(config) -> MtpHeadSpec | None:
    """The spec of ``config``'s family, the wrapper's ``model_type`` first, or ``None`` where it has none."""
    levels = (getattr(level, "model_type", None) for level in config_sources(config))
    return next((_SPECS_BY_MODEL_TYPE[level] for level in levels if level in _SPECS_BY_MODEL_TYPE), None)


def tail_load_patterns(load_ignores: Iterable[str]) -> tuple[str, ...]:
    """Those of a class's ``_keys_to_ignore_on_load_unexpected`` entries that name an MTP tail."""
    return tuple(pattern for pattern in load_ignores if _TAIL_PATTERN.search(pattern))


def stored_tail_keys(keys: Iterable[str], load_ignores: Iterable[str]) -> list[str]:
    """Those of ``keys`` a class drops at load as its MTP tail, matched as transformers matches: anywhere in the key."""
    patterns = [re.compile(pattern) for pattern in tail_load_patterns(load_ignores)]
    return [key for key in keys if any(pattern.search(key) for pattern in patterns)]


def unbuilt_tail_keys(keys: Iterable[str], config, built: Iterable[str]) -> list[str]:
    """Those of ``keys`` that read as the tail ``config`` declares and ``built`` (the model's own keys) does not hold:
    under an ``mtp`` module, or at a decoder index past the trunk. A load drops them whether or not the class names
    them, and Nemotron-H's and Step-3.7's composites name none."""
    depth = max(declared_tail_fields(config).values(), default=0)
    if not depth:
        return []
    first = text_config(config).num_hidden_layers
    held = set(built)

    def reads_as_tail(key: str) -> bool:
        entry = _DECODER_ENTRY_KEY.search(key)
        return bool(_MTP_MODULE_KEY.search(key) or (entry and first <= int(entry[1]) < first + depth))

    return [key for key in keys if key not in held and reads_as_tail(key)]
