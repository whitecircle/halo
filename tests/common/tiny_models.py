"""Builders a tiny random-init model needs: a synthetic checkpoint, a filled routing table and one model per EP MoE
family. Kept apart from :mod:`tests.common.models`, the torch-free catalogue the CPU tier and image build scripts
import."""

import shutil
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    Cohere2Config,
    Cohere2ForCausalLM,
    Cohere2MoeConfig,
    Cohere2MoeForCausalLM,
    Cohere2VisionConfig,
    Cohere2VisionForConditionalGeneration,
    DeepseekV4Config,
    DeepseekV4ForCausalLM,
    Gemma2Config,
    Gemma2ForCausalLM,
    Gemma3ForCausalLM,
    Gemma3TextConfig,
    Gemma4Config,
    Gemma4ForCausalLM,
    Gemma4ForConditionalGeneration,
    Gemma4TextConfig,
    Glm4MoeLiteConfig,
    Glm4MoeLiteForCausalLM,
    Glm5NextConfig,
    Glm5NextForConditionalGeneration,
    GptOssConfig,
    GptOssForCausalLM,
    InklingConfig,
    InklingForCausalLM,
    InklingForConditionalGeneration,
    InklingTextConfig,
    LagunaConfig,
    LagunaForCausalLM,
    Lfm2MoeConfig,
    Lfm2MoeForCausalLM,
    Lfm2VlConfig,
    Lfm2VlForConditionalGeneration,
    LlamaConfig,
    LlamaForCausalLM,
    Mistral3Config,
    Mistral3ForConditionalGeneration,
    MistralConfig,
    MistralForCausalLM,
    MixtralConfig,
    MixtralForCausalLM,
    Olmo3Config,
    Olmo3ForCausalLM,
    PretrainedConfig,
    PreTrainedModel,
    PreTrainedTokenizerFast,
    Qwen2Config,
    Qwen2ForCausalLM,
    Qwen3_5ForCausalLM,
    Qwen3_5MoeConfig,
    Qwen3_5MoeForCausalLM,
    Qwen3_5MoeForConditionalGeneration,
    Qwen3_5MoeTextConfig,
    Qwen3_5TextConfig,
    Qwen3Config,
    Qwen3ForCausalLM,
    Qwen3MoeConfig,
    Qwen3MoeForCausalLM,
    Qwen3NextConfig,
    Qwen3NextForCausalLM,
    Qwen3VLConfig,
    Qwen3VLForConditionalGeneration,
    Qwen4ExpConfig,
    Qwen4ExpForCausalLM,
    Qwen4ExpForConditionalGeneration,
    Qwen4ExpTextConfig,
    Step3p7Config,
    Step3p7ForConditionalGeneration,
    ZayaConfig,
    ZayaForCausalLM,
)
from transformers.models.mistral4 import Mistral4Config, Mistral4ForCausalLM
from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextPLELayer
from transformers.models.zaya.modeling_zaya import ZayaQKNorm

from halo.model_loading.remote_code import apply_remote_code_compat_shims
from halo.models.attention.sdpa_flex_sliding import FLEX_MIN_HEAD_DIM
from halo.models.param_roles import fp32_pinned_names
from halo.models.seq_cls_heads import Gemma4TextForSequenceClassification
from tests.common.distributed import shared_scratch_dir
from tests.common.models import (
    BAILING_LING_3_TINY,
    BAILING_MOE_LING_MINI,
    BAILING_MOE_RING_MINI,
    MISTRAL3_119B_MOE,
    TINY_ALTERNATING_SLIDING_CONFIG,
    TINY_BAILING_LING3_MLA_CONFIG,
    TINY_BAILING_MOE_CONFIG,
    TINY_BAILING_RING_CONFIG,
    TINY_COHERE2_MOE_CONFIG,
    TINY_DSV4_CONFIG,
    TINY_GEMMA4_MOE_CONFIG,
    TINY_GEMMA4_VISION_CONFIG,
    TINY_GEMMA4_WIDE_HEAD_CONFIG,
    TINY_GLM4_MOE_LITE_CONFIG,
    TINY_GLM5_CONFIG,
    TINY_GLM5_VISION_CONFIG,
    TINY_GPTOSS_CONFIG,
    TINY_INKLING_CONFIG,
    TINY_INKLING_PP_OVERRIDES,
    TINY_INKLING_VISION_CONFIG,
    TINY_LAGUNA_CONFIG,
    TINY_LAGUNA_PP_OVERRIDES,
    TINY_LFM2_MOE_CONFIG,
    TINY_MISTRAL4_CONFIG,
    TINY_MISTRAL_SLIDING_CONFIG,
    TINY_MIXTRAL_CONFIG,
    TINY_PIXTRAL_VISION_CONFIG,
    TINY_QWEN3_CONFIG,
    TINY_QWEN3_MOE_CONFIG,
    TINY_QWEN3_NEXT_CONFIG,
    TINY_QWEN3_VL_TEXT_CONFIG,
    TINY_QWEN3_VL_VISION_CONFIG,
    TINY_QWEN4_EXP_CONFIG,
    TINY_QWEN4_EXP_VISION_CONFIG,
    TINY_QWEN35_CONFIG,
    TINY_QWEN35_MOE_CONFIG,
    TINY_QWEN35_VISION_CONFIG,
    TINY_SIGLIP2_VISION_CONFIG,
    TINY_SIGLIP_VISION_CONFIG,
    TINY_STEP3P7_CONFIG,
    TINY_STEP3P7_VISION_CONFIG,
    TINY_TIED_QWEN3_CONFIG,
    TINY_TIED_QWEN3_MOE_FIELDS,
    TINY_ZAYA_CONFIG,
)

# Multiple a tiny family checkpoint pads the tokenizer's vocab to, and the shard size it is saved at.
VOCAB_PAD_MULTIPLE = 128
TINY_SHARD_SIZE = "4MB"
DSV4_TID2EID_SEED = 1234
# Families whose transformers class pins parameters (not only buffers) in fp32 via ``_keep_in_fp32_modules_strict``.
PINNED_FP32_FAMILIES = ("deepseek_v4", "glm5_next", "inkling_text")
# Copied from a release so ``AutoTokenizer`` loads a synthetic checkpoint offline.
TOKENIZER_FILE_PREFIXES = ("tokenizer", "special_tokens", "chat_template")
CONTENT_ONLY_CHAT_TEMPLATE = "{% for message in messages %}{{ message['content'] }} {% endfor %}"
# Shorter than the documents a packed test row carries, so it masks keys inside one.
TINY_SLIDING_WINDOW = 3


def padded_vocab_size(tokenizer) -> int:
    """``tokenizer``'s vocab rounded up to :data:`VOCAB_PAD_MULTIPLE`, as a release pads it so TP-sharded embedding and
    head divide it."""
    return -(-len(tokenizer) // VOCAB_PAD_MULTIPLE) * VOCAB_PAD_MULTIPLE


class _WeightLeaf(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))


def module_with_weight_keys(paths: list[str]) -> torch.nn.Module:
    """A module tree whose state dict is exactly ``paths``, each ending in ``.weight``, with no model behind it."""
    root = torch.nn.Module()
    for path in paths:
        parent = root
        parts = path.split(".")
        for name in parts[:-2]:
            child = parent._modules.get(name)
            if child is None:
                child = torch.nn.Module()
                parent.add_module(name, child)
            parent = child
        parent.add_module(parts[-2], _WeightLeaf())
    return root


def config_only_module(config) -> torch.nn.Module:
    """A bare module carrying ``config`` and nothing else: what a gate reading ``model.config`` and walking the tree
    sees."""
    module = torch.nn.Module()
    module.config = config
    return module


def randomize_tid2eid(model, seed: int = DSV4_TID2EID_SEED) -> None:
    """Fill every DeepSeek-V4 hash layer's ``tid2eid`` with distinct experts per token id; DeepEP dispatch and the EP
    init guard require it."""
    gen = torch.Generator().manual_seed(seed)
    num_experts = model.config.n_routed_experts
    for layer in model.model.layers:
        if layer.mlp.is_hash:
            table = layer.mlp.gate.tid2eid
            perm = torch.rand(table.shape[0], num_experts, generator=gen).argsort(dim=-1)
            table.copy_(perm[:, : table.shape[1]])


def copy_release_tokenizer(repo_id: str, out_dir: Path) -> None:
    tokenizer_dir = Path(snapshot_download(repo_id, allow_patterns=[f"{p}*" for p in TOKENIZER_FILE_PREFIXES]))
    out_dir.mkdir(parents=True, exist_ok=True)
    for src in tokenizer_dir.iterdir():
        if src.is_file() and src.name.startswith(TOKENIZER_FILE_PREFIXES):
            shutil.copy2(src, out_dir / src.name)


def word_level_tokenizer(words: Sequence[str] = ()) -> PreTrainedTokenizerFast:
    """An offline whitespace tokenizer over ``<unk>``, ``<eos>`` (also its pad) and ``words``, for a checkpoint with no
    release tokenizer."""
    vocab = {"<unk>": 0, "<eos>": 1, **{word: index for index, word in enumerate(words, start=2)}}
    backend = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
    backend.pre_tokenizer = Whitespace()
    return PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="<unk>", eos_token="<eos>", pad_token="<eos>")


def build_wide_head_gemma4_reward_checkpoint(path: str, words: Sequence[str], *, seed: int = 0) -> None:
    """Save a random-init single-score Gemma 4 text classifier with wide global heads, as reward scorers load one. The
    shape whose ``sdpa`` takes the wide-head SDPA pin and, on CUDA, the flex-sliding variant."""
    tokenizer = word_level_tokenizer(words)
    tokenizer.chat_template = CONTENT_ONLY_CHAT_TEMPLATE
    torch.manual_seed(seed)
    config = Gemma4TextConfig(
        **TINY_GEMMA4_WIDE_HEAD_CONFIG,
        layer_types=["sliding_attention", "full_attention"],
        num_labels=1,
        pad_token_id=tokenizer.pad_token_id,
    )
    Gemma4TextForSequenceClassification(config).save_pretrained(path)
    tokenizer.save_pretrained(path)


class UnknownHybridConfig(PretrainedConfig):
    """A family transformers does not know and no toolkit list names, so a gate must read linear-attention layers off
    its fields."""

    model_type = "halo_unknown_hybrid"


def unknown_hybrid_config() -> PretrainedConfig:
    return UnknownHybridConfig(layer_types=["linear_attention", "full_attention"])


def remote_config(repo: str, **overrides) -> PretrainedConfig:
    """Release ``repo``'s config, built by its remote code once the shims restore names transformers 5 dropped."""
    apply_remote_code_compat_shims()
    return AutoConfig.from_pretrained(repo, trust_remote_code=True, **overrides)


def build_tied_qwen3_checkpoint(target_dir: str, tokenizer, *, moe: bool, seed: int) -> None:
    """Save a random-init :data:`TINY_TIED_QWEN3_CONFIG` model at ``tokenizer``'s padded vocab, with that tokenizer.
    Rank 0 only."""
    torch.manual_seed(seed)
    fields = {**TINY_TIED_QWEN3_CONFIG, "vocab_size": padded_vocab_size(tokenizer)}
    if moe:
        model = Qwen3MoeForCausalLM(Qwen3MoeConfig(**fields, **TINY_TIED_QWEN3_MOE_FIELDS))
    else:
        model = Qwen3ForCausalLM(Qwen3Config(**fields))
    model.to(torch.bfloat16).save_pretrained(target_dir)
    tokenizer.save_pretrained(target_dir)


def build_tiny_mistral4_checkpoint(out_dir: Path, seed: int = 0) -> Path:
    """Write a random-init :data:`TINY_MISTRAL4_CONFIG` model with the release's tokenizer, so loaders run end to end
    with no 119B download."""
    copy_release_tokenizer(MISTRAL3_119B_MOE, out_dir)
    torch.manual_seed(seed)
    model = Mistral4ForCausalLM(Mistral4Config(**TINY_MISTRAL4_CONFIG)).to(torch.bfloat16)
    model.save_pretrained(out_dir, safe_serialization=True)
    return out_dir


@dataclass(frozen=True)
class TinyFamily:
    """How to build one family's tiny model at a tokenizer's vocab, and how a server reloads it. ``ulysses_cp`` is
    False where Ulysses CP refuses the family at load. ``attention_targets`` names LoRA's projections where the
    checkpoint index cannot."""

    build: Callable[[dict], PreTrainedModel]
    load_class: type = AutoModelForCausalLM
    trust_remote_code: bool = False
    text_overrides: dict = field(default_factory=dict)
    ulysses_cp: bool = True
    attention_targets: tuple[str, ...] = ()


def _causal(config_cls: type, model_cls: type, tiny: dict) -> Callable[[dict], PreTrainedModel]:
    return lambda overrides: model_cls(config_cls(**{**tiny, **overrides}))


def _composite(
    config_cls: type, model_cls: type, text: dict, vision: dict, **wrapper
) -> Callable[[dict], PreTrainedModel]:
    return lambda overrides: model_cls(
        config_cls(text_config={**text, **overrides}, vision_config=dict(vision), **wrapper)
    )


def _tiny_deepseek_v4(overrides: dict) -> PreTrainedModel:
    model = DeepseekV4ForCausalLM(DeepseekV4Config(**{**TINY_DSV4_CONFIG, **overrides}))
    randomize_tid2eid(model)
    return model


def _tiny_gemma4(overrides: dict) -> PreTrainedModel:
    """Gemma 4's per-layer input table embeds the same ids as the main one, pad included."""
    config = {**TINY_GEMMA4_MOE_CONFIG, **overrides}
    config["vocab_size_per_layer_input"] = config["vocab_size"]
    return Gemma4ForCausalLM(Gemma4TextConfig(**config))


def _tiny_zaya(overrides: dict) -> PreTrainedModel:
    """Random init zeroes ``ZayaQKNorm.temp``, which zeroes every key and every gradient into q and k; a release
    carries it learned."""
    model = ZayaForCausalLM(ZayaConfig(**{**TINY_ZAYA_CONFIG, **overrides}))
    for module in model.modules():
        if isinstance(module, ZayaQKNorm):
            torch.nn.init.ones_(module.temp)
    return model


def _with_live_ple_convs(model: PreTrainedModel) -> PreTrainedModel:
    """Random init zeroes every PLE depthwise conv, and every gradient into it; a release carries it learned."""
    for module in model.modules():
        if isinstance(module, Qwen4ExpTextPLELayer):
            torch.nn.init.normal_(module.conv1d.weight, std=model.config.get_text_config().initializer_range)
    return model


def _tiny_bailing(repo: str, tiny: dict) -> Callable[[dict], PreTrainedModel]:
    """Bailing is remote code: its classes come from release ``repo``'s module, which needs the toolkit's shims to
    import."""

    def build(overrides: dict) -> PreTrainedModel:
        config = remote_config(repo)
        for key, value in {**tiny, **overrides}.items():
            setattr(config, key, value)
        return AutoModelForCausalLM.from_config(config, trust_remote_code=True)

    return build


# One entry per EP MoE family, keyed by ``model_type``. ``test_tiny_family_roster.py`` holds the keys to the EP
# registry.
TINY_MOE_FAMILIES: dict[str, TinyFamily] = {
    "bailing_moe": TinyFamily(
        _tiny_bailing(BAILING_MOE_LING_MINI, TINY_BAILING_MOE_CONFIG),
        trust_remote_code=True,
        attention_targets=("query_key_value", "dense"),
    ),
    "cohere2_moe": TinyFamily(_causal(Cohere2MoeConfig, Cohere2MoeForCausalLM, TINY_COHERE2_MOE_CONFIG)),
    # The CSA/HCA compressor and indexer pool token windows along the sequence.
    "deepseek_v4": TinyFamily(_tiny_deepseek_v4, ulysses_cp=False, attention_targets=("q_a_proj", "o_b_proj")),
    # Not an attention the Ulysses wrapper implements.
    "gemma4_text": TinyFamily(_tiny_gemma4, text_overrides={"max_position_embeddings": 512}, ulysses_cp=False),
    "glm4_moe_lite": TinyFamily(_causal(Glm4MoeLiteConfig, Glm4MoeLiteForCausalLM, TINY_GLM4_MOE_LITE_CONFIG)),
    # KDA linear attention: a causal conv and a recurrent scan along the sequence.
    "glm5_next": TinyFamily(
        _composite(Glm5NextConfig, Glm5NextForConditionalGeneration, TINY_GLM5_CONFIG, TINY_GLM5_VISION_CONFIG),
        load_class=AutoModelForImageTextToText,
        ulysses_cp=False,
    ),
    "gpt_oss": TinyFamily(_causal(GptOssConfig, GptOssForCausalLM, TINY_GPTOSS_CONFIG)),
    # A depthwise causal conv inside attention and around each sublayer.
    "inkling_text": TinyFamily(
        _causal(InklingTextConfig, InklingForCausalLM, TINY_INKLING_CONFIG),
        text_overrides={"max_position_embeddings": 512},
        ulysses_cp=False,
    ),
    # Neither attention is one the Ulysses wrapper implements.
    "laguna": TinyFamily(_causal(LagunaConfig, LagunaForCausalLM, TINY_LAGUNA_CONFIG), ulysses_cp=False),
    "lfm2_moe": TinyFamily(_causal(Lfm2MoeConfig, Lfm2MoeForCausalLM, TINY_LFM2_MOE_CONFIG), ulysses_cp=False),
    # No AutoModelForCausalLM entry for mistral4 in transformers 5.20.
    "mistral4": TinyFamily(
        _causal(Mistral4Config, Mistral4ForCausalLM, TINY_MISTRAL4_CONFIG), load_class=Mistral4ForCausalLM
    ),
    # Gated DeltaNet layers: a conv and a recurrent scan along the sequence.
    "qwen3_5_moe_text": TinyFamily(
        _causal(Qwen3_5MoeTextConfig, Qwen3_5MoeForCausalLM, TINY_QWEN35_MOE_CONFIG), ulysses_cp=False
    ),
    "qwen3_moe": TinyFamily(_causal(Qwen3MoeConfig, Qwen3MoeForCausalLM, TINY_QWEN3_MOE_CONFIG)),
    "qwen3_next": TinyFamily(_causal(Qwen3NextConfig, Qwen3NextForCausalLM, TINY_QWEN3_NEXT_CONFIG), ulysses_cp=False),
    # Gated DeltaNet layers and the PLE conv along the sequence.
    "qwen4_exp_text": TinyFamily(
        lambda overrides: _with_live_ple_convs(
            _causal(Qwen4ExpTextConfig, Qwen4ExpForCausalLM, TINY_QWEN4_EXP_CONFIG)(overrides)
        ),
        ulysses_cp=False,
    ),
    # Not an attention the Ulysses wrapper implements.
    "step3p7": TinyFamily(
        _composite(Step3p7Config, Step3p7ForConditionalGeneration, TINY_STEP3P7_CONFIG, TINY_STEP3P7_VISION_CONFIG),
        load_class=AutoModelForImageTextToText,
        ulysses_cp=False,
    ),
    # The CCA projection convolves and shifts along the sequence and yields q, k and v, so ``o_proj`` is the one plain
    # linear.
    "zaya": TinyFamily(
        _tiny_zaya,
        text_overrides={"max_position_embeddings": 512},
        ulysses_cp=False,
        attention_targets=("o_proj",),
    ),
}
# The multimodal wrappers a text tower ships under, keyed by the wrapper's ``model_type``; GLM-5 Next and Step-3.7
# above are already wrappers. ``test_ep_hub_namespace_export.py`` holds both rosters to the EP registry.
TINY_MOE_VLM_FAMILIES: dict[str, TinyFamily] = {
    "cohere2_vision": TinyFamily(
        _composite(
            Cohere2VisionConfig,
            Cohere2VisionForConditionalGeneration,
            {**TINY_COHERE2_MOE_CONFIG, "model_type": "cohere2_moe"},
            TINY_SIGLIP_VISION_CONFIG,
            downsample_factor=2,
            alignment_intermediate_size=64,
        ),
        load_class=AutoModelForImageTextToText,
    ),
    "gemma4": TinyFamily(
        _composite(
            Gemma4Config,
            Gemma4ForConditionalGeneration,
            TINY_GEMMA4_MOE_CONFIG,
            TINY_GEMMA4_VISION_CONFIG,
            audio_config=None,
        ),
        load_class=AutoModelForImageTextToText,
        ulysses_cp=False,
    ),
    "inkling_mm_model": TinyFamily(
        _composite(InklingConfig, InklingForConditionalGeneration, TINY_INKLING_CONFIG, TINY_INKLING_VISION_CONFIG),
        load_class=AutoModelForImageTextToText,
        ulysses_cp=False,
    ),
    "lfm2_vl": TinyFamily(
        _composite(
            Lfm2VlConfig,
            Lfm2VlForConditionalGeneration,
            {**TINY_LFM2_MOE_CONFIG, "model_type": "lfm2_moe"},
            TINY_SIGLIP2_VISION_CONFIG,
            projector_hidden_size=64,
        ),
        load_class=AutoModelForImageTextToText,
        ulysses_cp=False,
    ),
    "mistral3": TinyFamily(
        _composite(
            Mistral3Config,
            Mistral3ForConditionalGeneration,
            {**TINY_MISTRAL4_CONFIG, "model_type": "mistral4"},
            TINY_PIXTRAL_VISION_CONFIG,
            spatial_merge_size=2,
            tie_word_embeddings=TINY_MISTRAL4_CONFIG["tie_word_embeddings"],
        ),
        load_class=AutoModelForImageTextToText,
    ),
    "qwen3_5_moe": TinyFamily(
        _composite(
            Qwen3_5MoeConfig, Qwen3_5MoeForConditionalGeneration, TINY_QWEN35_MOE_CONFIG, TINY_QWEN35_VISION_CONFIG
        ),
        load_class=AutoModelForImageTextToText,
        ulysses_cp=False,
    ),
    "qwen4_exp": TinyFamily(
        lambda overrides: _with_live_ple_convs(
            _composite(
                Qwen4ExpConfig, Qwen4ExpForConditionalGeneration, TINY_QWEN4_EXP_CONFIG, TINY_QWEN4_EXP_VISION_CONFIG
            )(overrides)
        ),
        load_class=AutoModelForImageTextToText,
        ulysses_cp=False,
    ),
}
# Overrides under which a tiny model splits into two pipeline stages on a layer_types period with a MoE layer on each.
TINY_PP_SPLIT_OVERRIDES = {
    "glm4_moe_lite": {"num_hidden_layers": 4},
    "inkling_mm_model": TINY_INKLING_PP_OVERRIDES,
    "inkling_text": TINY_INKLING_PP_OVERRIDES,
    "laguna": TINY_LAGUNA_PP_OVERRIDES,
    "qwen3_5_moe_text": {"num_hidden_layers": 8},
}


def _flex_wide_heads(family: TinyFamily) -> TinyFamily:
    return replace(family, text_overrides={**family.text_overrides, "head_dim": FLEX_MIN_HEAD_DIM})


# Sliding heads as wide as FlexAttention runs (the roster's 8 keeps SDPA): a CUDA load builds Gemma 4 on ``sdpa`` as
# flex-sliding.
TINY_GEMMA4_TEXT_FLEX_FAMILY = _flex_wide_heads(TINY_MOE_FAMILIES["gemma4_text"])
TINY_GEMMA4_FLEX_FAMILY = _flex_wide_heads(TINY_MOE_VLM_FAMILIES["gemma4"])
# A window on the config alone (Mistral), NoPE full layers (Cohere2), q/k norm over the whole projection (OLMo 3).
TINY_SLIDING_DENSE_FAMILIES: dict[str, TinyFamily] = {
    "cohere2": TinyFamily(_causal(Cohere2Config, Cohere2ForCausalLM, TINY_ALTERNATING_SLIDING_CONFIG)),
    "mistral": TinyFamily(_causal(MistralConfig, MistralForCausalLM, TINY_MISTRAL_SLIDING_CONFIG)),
    "olmo3": TinyFamily(_causal(Olmo3Config, Olmo3ForCausalLM, TINY_ALTERNATING_SLIDING_CONFIG)),
}
TINY_DENSE_FAMILY = TinyFamily(_causal(Qwen3Config, Qwen3ForCausalLM, TINY_QWEN3_CONFIG))
# Gated DeltaNet and attention layers, which the MoE roster does not carry.
TINY_QWEN35_DENSE_FAMILY = TinyFamily(_causal(Qwen3_5TextConfig, Qwen3_5ForCausalLM, TINY_QWEN35_CONFIG))
TINY_DENSE_FAMILIES: dict[str, TinyFamily] = {"qwen3": TINY_DENSE_FAMILY, "qwen3_5": TINY_QWEN35_DENSE_FAMILY}
# Ring-mini-linear-2.0: the Bailing family's other attention layout (Lightning Attention-2 beside softmax).
TINY_BAILING_RING_FAMILY = TinyFamily(
    _tiny_bailing(BAILING_MOE_RING_MINI, TINY_BAILING_RING_CONFIG),
    trust_remote_code=True,
    ulysses_cp=False,
    attention_targets=("query_key_value", "dense"),
)
# Ling-3.0's MLA attention without its KDA layers, at the same shape.
TINY_BAILING_LING3_MLA_FAMILY = TinyFamily(
    _tiny_bailing(BAILING_LING_3_TINY, TINY_BAILING_LING3_MLA_CONFIG), trust_remote_code=True, ulysses_cp=False
)
TINY_UNWRAPPED_MOE_FAMILY = TinyFamily(_causal(MixtralConfig, MixtralForCausalLM, TINY_MIXTRAL_CONFIG))
# Its own text attention class behind a multimodal backbone.
TINY_QWEN3_VL_FAMILY = TinyFamily(
    _composite(Qwen3VLConfig, Qwen3VLForConditionalGeneration, TINY_QWEN3_VL_TEXT_CONFIG, TINY_QWEN3_VL_VISION_CONFIG),
    load_class=AutoModelForImageTextToText,
)

# Past the dense roster, at its geometry: plain GQA (Llama, Qwen2), a config-level window (Mistral), Gemma 2's softcaps
# over alternating sliding layers, and Gemma 3's layer types at the 256-wide head FA4 runs on its dedicated kernel.
TINY_DENSE_ATTENTION_FAMILIES: dict[str, TinyFamily] = {
    "llama": TinyFamily(_causal(LlamaConfig, LlamaForCausalLM, TINY_QWEN3_CONFIG)),
    "mistral": TinyFamily(
        _causal(MistralConfig, MistralForCausalLM, {**TINY_QWEN3_CONFIG, "sliding_window": TINY_SLIDING_WINDOW})
    ),
    "qwen2": TinyFamily(_causal(Qwen2Config, Qwen2ForCausalLM, TINY_QWEN3_CONFIG)),
    "gemma2": TinyFamily(
        _causal(
            Gemma2Config,
            Gemma2ForCausalLM,
            {
                **TINY_QWEN3_CONFIG,
                "attn_logit_softcapping": 50.0,
                "final_logit_softcapping": 30.0,
                "sliding_window": TINY_SLIDING_WINDOW,
            },
        )
    ),
    "gemma3_text": TinyFamily(
        _causal(
            Gemma3TextConfig,
            Gemma3ForCausalLM,
            {
                **TINY_QWEN3_CONFIG,
                "head_dim": 256,
                "layer_types": ["sliding_attention", "full_attention"] * (TINY_QWEN3_CONFIG["num_hidden_layers"] // 2),
                "sliding_window": TINY_SLIDING_WINDOW,
            },
        )
    ),
}


def tiny_family_model(family: TinyFamily, tokenizer=None, *, overrides: dict | None = None) -> PreTrainedModel:
    """``family``'s random-init tiny model (fp32, seeded by the caller), at ``tokenizer``'s vocab when given."""
    overrides = {**family.text_overrides, **(overrides or {})}
    if tokenizer is not None:
        overrides |= {
            "vocab_size": padded_vocab_size(tokenizer),
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
    return family.build(overrides)


def build_tiny_family_checkpoint(
    family: TinyFamily,
    target_dir: str,
    tokenizer=None,
    seed: int = 0,
    *,
    fp32_pins: bool = False,
    edit: Callable[[PreTrainedModel], None] | None = None,
    overrides: dict | None = None,
) -> None:
    """Save ``family``'s seeded tiny model at ``tokenizer``'s vocab as a checkpoint the production loaders read. Rank 0
    only. Sharded, so it carries the index attention names are read from. ``fp32_pins`` stores pinned parameters at
    full fp32, as a release does."""
    torch.manual_seed(seed)
    model = tiny_family_model(family, tokenizer, overrides=overrides)
    if edit is not None:
        edit(model)
    pinned = fp32_pinned_names(model) if fp32_pins else frozenset()
    stored = {name: param.detach().clone() for name, param in model.named_parameters() if name in pinned}
    model.to(torch.bfloat16)
    for name, param in model.named_parameters():
        if name in stored:
            param.data = stored[name]
    model.save_pretrained(target_dir, max_shard_size=TINY_SHARD_SIZE)
    if tokenizer is not None:
        tokenizer.save_pretrained(target_dir)


def shared_tiny_family_checkpoint(
    ctx,
    family: TinyFamily,
    name: str,
    tokenizer,
    seed: int,
    *,
    fp32_pins: bool = False,
    edit: Callable[[PreTrainedModel], None] | None = None,
    overrides: dict | None = None,
) -> str:
    """:func:`build_tiny_family_checkpoint` into the rank-shared scratch dir ``name``; rank 0 builds it and every rank
    waits. Collective."""
    path = shared_scratch_dir(name)
    if ctx.rank == 0:
        shutil.rmtree(path, ignore_errors=True)
        ctx.on_teardown(lambda: shutil.rmtree(path, ignore_errors=True))
        build_tiny_family_checkpoint(
            family, path, tokenizer, seed, fp32_pins=fp32_pins, edit=edit, overrides=overrides
        )
    ctx.barrier()
    return path
