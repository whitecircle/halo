"""Builders a tiny random-init model needs before it can run: a synthetic on-disk checkpoint, a filled
routing table, and one tiny model per EP MoE family.

Kept apart from :mod:`tests.common.models`, the torch-free catalogue of names and configs that the CPU
tier and the image build scripts import.
"""

import shutil
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    Cohere2MoeConfig,
    Cohere2MoeForCausalLM,
    DeepseekV4Config,
    DeepseekV4ForCausalLM,
    Gemma4ForCausalLM,
    Gemma4TextConfig,
    Glm4MoeLiteConfig,
    Glm4MoeLiteForCausalLM,
    Glm5NextConfig,
    Glm5NextForConditionalGeneration,
    GptOssConfig,
    GptOssForCausalLM,
    InklingForCausalLM,
    InklingTextConfig,
    LagunaConfig,
    LagunaForCausalLM,
    Lfm2MoeConfig,
    Lfm2MoeForCausalLM,
    PreTrainedModel,
    Qwen3_5MoeForCausalLM,
    Qwen3_5MoeTextConfig,
    Qwen3Config,
    Qwen3ForCausalLM,
    Qwen3MoeConfig,
    Qwen3MoeForCausalLM,
    Step3p7Config,
    Step3p7ForConditionalGeneration,
    ZayaConfig,
    ZayaForCausalLM,
)
from transformers.models.mistral4 import Mistral4Config, Mistral4ForCausalLM
from transformers.models.zaya.modeling_zaya import ZayaQKNorm

from src.models.patches.remote_code_compat import apply_remote_code_compat_shims
from tests.common.models import (
    BAILING_MOE_LING_MINI,
    MISTRAL3_119B_MOE,
    TINY_BAILING_MOE_CONFIG,
    TINY_COHERE2_MOE_CONFIG,
    TINY_DSV4_CONFIG,
    TINY_GEMMA4_MOE_CONFIG,
    TINY_GLM4_MOE_LITE_CONFIG,
    TINY_GLM5_CONFIG,
    TINY_GLM5_VISION_CONFIG,
    TINY_GPTOSS_CONFIG,
    TINY_INKLING_CONFIG,
    TINY_LAGUNA_CONFIG,
    TINY_LFM2_MOE_CONFIG,
    TINY_MISTRAL4_CONFIG,
    TINY_QWEN3_CONFIG,
    TINY_QWEN3_MOE_CONFIG,
    TINY_QWEN35_MOE_CONFIG,
    TINY_STEP3P7_CONFIG,
    TINY_STEP3P7_VISION_CONFIG,
    TINY_ZAYA_CONFIG,
)

# Multiple a tiny family checkpoint pads the tokenizer's vocab to, and the shard size it is saved at.
VOCAB_PAD_MULTIPLE = 128
TINY_SHARD_SIZE = "4MB"
# Seed ``randomize_tid2eid`` fills the hash table from unless a test pins its own.
DSV4_TID2EID_SEED = 1234

# Small enough that a tokenizer-sized vocab spills into several shards, so a pinned-family checkpoint
# carries the index :func:`tests.common.peft_helpers.attention_target_modules` reads projection names from.
PINNED_SHARD_SIZE = "20MB"


def randomize_tid2eid(model, seed: int = DSV4_TID2EID_SEED) -> None:
    """Fill every DeepSeek-V4 hash layer's ``tid2eid`` with DISTINCT experts per token id.

    Random init leaves the table all-zero, and DeepEP dispatch and the EP wrapper's init guard both
    require distinct top-k experts per token.
    """
    gen = torch.Generator().manual_seed(seed)
    num_experts = model.config.n_routed_experts
    for layer in model.model.layers:
        if layer.mlp.is_hash:
            table = layer.mlp.gate.tid2eid
            perm = torch.rand(table.shape[0], num_experts, generator=gen).argsort(dim=-1)
            table.copy_(perm[:, : table.shape[1]])


# The files a synthetic checkpoint copies from its release so ``AutoTokenizer`` loads it offline.
TOKENIZER_FILE_PREFIXES = ("tokenizer", "special_tokens", "chat_template")


def build_tiny_mistral4_checkpoint(out_dir: Path, seed: int = 0) -> Path:
    """Write a random-init :data:`TINY_MISTRAL4_CONFIG` model to ``out_dir`` and return the path.

    The layout ``save_pretrained`` writes (``model.safetensors`` + ``config.json``), plus the release's
    tokenizer files, so the lazy loader and ``load_distributed_model`` run end to end without the
    119B download.
    """
    tokenizer_dir = Path(
        snapshot_download(MISTRAL3_119B_MOE, allow_patterns=[f"{p}*" for p in TOKENIZER_FILE_PREFIXES])
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    for src in tokenizer_dir.iterdir():
        if src.is_file() and src.name.startswith(TOKENIZER_FILE_PREFIXES):
            shutil.copy2(src, out_dir / src.name)

    torch.manual_seed(seed)
    model = Mistral4ForCausalLM(Mistral4Config(**TINY_MISTRAL4_CONFIG)).to(torch.bfloat16)
    model.save_pretrained(out_dir, safe_serialization=True)
    return out_dir


@dataclass(frozen=True)
class TinyFamily:
    """How to build one family's tiny model at a tokenizer's vocab, and how a server reloads it.

    ``build`` takes the text-config overrides (vocab and special-token ids) and returns the random-init
    model; ``load_class`` is the stock class ``from_pretrained`` reads its checkpoint back with.
    ``ulysses_cp`` is False where Ulysses CP refuses the family at load (a sequence-axis mixer or
    compressor it cannot shard, or an attention with no flash kernel). ``attention_targets`` names the
    attention projections LoRA adapts where the checkpoint index cannot (hub key names that differ
    from the module tree, or projections not spelled ``*_proj``). ``cp_attn_implementation`` is the
    label a family loads with under CP where the auto-selected flash label is refused (remote code
    declaring only the v4 flash flag; the Ulysses wrapper calls flash through its own probe).
    """

    build: Callable[[dict], PreTrainedModel]
    load_class: type = AutoModelForCausalLM
    trust_remote_code: bool = False
    text_overrides: dict = field(default_factory=dict)
    ulysses_cp: bool = True
    attention_targets: tuple[str, ...] = ()
    cp_attn_implementation: str | None = None


def _causal(config_cls: type, model_cls: type, tiny: dict) -> Callable[[dict], PreTrainedModel]:
    return lambda overrides: model_cls(config_cls(**{**tiny, **overrides}))


def _composite(config_cls: type, model_cls: type, text: dict, vision: dict) -> Callable[[dict], PreTrainedModel]:
    return lambda overrides: model_cls(config_cls(text_config={**text, **overrides}, vision_config=dict(vision)))


def _tiny_deepseek_v4(overrides: dict) -> PreTrainedModel:
    model = DeepseekV4ForCausalLM(DeepseekV4Config(**{**TINY_DSV4_CONFIG, **overrides}))
    randomize_tid2eid(model)
    return model


def _tiny_gemma4(overrides: dict) -> PreTrainedModel:
    """Gemma 4's per-layer input table embeds the same ids as the main one, pad included."""
    config = {**TINY_GEMMA4_MOE_CONFIG, **overrides, "vocab_size_per_layer_input": overrides["vocab_size"]}
    return Gemma4ForCausalLM(Gemma4TextConfig(**config))


def _tiny_zaya(overrides: dict) -> PreTrainedModel:
    """Random init zeroes Zaya's per-KV-head key scale (``ZayaQKNorm.temp``), which zeroes every key and
    with it every gradient into the query and key projections; a released checkpoint carries it learned."""
    model = ZayaForCausalLM(ZayaConfig(**{**TINY_ZAYA_CONFIG, **overrides}))
    for module in model.modules():
        if isinstance(module, ZayaQKNorm):
            torch.nn.init.ones_(module.temp)
    return model


def _tiny_bailing(overrides: dict) -> PreTrainedModel:
    """Bailing is remote code: its config and model classes come from the module the release ships,
    which imports names transformers 5 dropped until the toolkit's shims restore them."""
    apply_remote_code_compat_shims()
    config = AutoConfig.from_pretrained(BAILING_MOE_LING_MINI, trust_remote_code=True)
    for key, value in {**TINY_BAILING_MOE_CONFIG, **overrides}.items():
        setattr(config, key, value)
    return AutoModelForCausalLM.from_config(config, trust_remote_code=True)


# One entry per EP MoE family (``src/distributed/expert_parallel/layers``), keyed by the ``model_type``
# its tiny model carries, which the GPU suites take as ``--family``.
# ``tests/cpu/conventions/test_tiny_family_roster.py`` holds the keys to the EP registry, so a new
# family without a tiny model fails there.
TINY_MOE_FAMILIES: dict[str, TinyFamily] = {
    "bailing_moe": TinyFamily(
        _tiny_bailing,
        trust_remote_code=True,
        attention_targets=("query_key_value", "dense"),
        cp_attn_implementation="sdpa",
    ),
    "cohere2_moe": TinyFamily(_causal(Cohere2MoeConfig, Cohere2MoeForCausalLM, TINY_COHERE2_MOE_CONFIG)),
    # The CSA/HCA compressor and indexer pool token windows along the sequence.
    "deepseek_v4": TinyFamily(_tiny_deepseek_v4, ulysses_cp=False, attention_targets=("q_a_proj", "o_b_proj")),
    # Its attention is not one the Ulysses wrapper implements.
    "gemma4_text": TinyFamily(_tiny_gemma4, text_overrides={"max_position_embeddings": 512}, ulysses_cp=False),
    # Its MLA head shapes have no flash kernel, so the auto-selected attention is sdpa.
    "glm4_moe_lite": TinyFamily(
        _causal(Glm4MoeLiteConfig, Glm4MoeLiteForCausalLM, TINY_GLM4_MOE_LITE_CONFIG), ulysses_cp=False
    ),
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
    # No AutoModelForCausalLM entry for mistral4 in transformers 5.16.
    "mistral4": TinyFamily(
        _causal(Mistral4Config, Mistral4ForCausalLM, TINY_MISTRAL4_CONFIG), load_class=Mistral4ForCausalLM
    ),
    # Gated DeltaNet layers: a conv and a recurrent scan along the sequence.
    "qwen3_5_moe_text": TinyFamily(
        _causal(Qwen3_5MoeTextConfig, Qwen3_5MoeForCausalLM, TINY_QWEN35_MOE_CONFIG), ulysses_cp=False
    ),
    "qwen3_moe": TinyFamily(_causal(Qwen3MoeConfig, Qwen3MoeForCausalLM, TINY_QWEN3_MOE_CONFIG)),
    # Its attention is not one the Ulysses wrapper implements.
    "step3p7": TinyFamily(
        _composite(Step3p7Config, Step3p7ForConditionalGeneration, TINY_STEP3P7_CONFIG, TINY_STEP3P7_VISION_CONFIG),
        load_class=AutoModelForImageTextToText,
        ulysses_cp=False,
    ),
    # The CCA projection convolves and shifts along the sequence; query, key and value all come out of
    # it, so ``o_proj`` is the attention's one plain linear.
    "zaya": TinyFamily(
        _tiny_zaya,
        text_overrides={"max_position_embeddings": 512},
        ulysses_cp=False,
        attention_targets=("o_proj",),
    ),
}
# The dense model the family sweeps pair with the MoE roster.
TINY_DENSE_FAMILY = TinyFamily(_causal(Qwen3Config, Qwen3ForCausalLM, TINY_QWEN3_CONFIG))


def build_tiny_family_checkpoint(family: TinyFamily, target_dir: str, tokenizer, seed: int) -> None:
    """Save ``family``'s seeded tiny model at ``tokenizer``'s vocab, with that tokenizer, as a checkpoint
    the production loaders read. Rank 0 only.

    Sharded, so the checkpoint carries the index a family's attention projection names are read from
    (:func:`tests.common.peft_helpers.attention_target_modules`).
    """
    overrides = {
        **family.text_overrides,
        # Padded as a release vocab is, so a TP-sharded head divides it.
        "vocab_size": -(-len(tokenizer) // VOCAB_PAD_MULTIPLE) * VOCAB_PAD_MULTIPLE,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    torch.manual_seed(seed)
    family.build(overrides).to(torch.bfloat16).save_pretrained(target_dir, max_shard_size=TINY_SHARD_SIZE)
    tokenizer.save_pretrained(target_dir)


def _tiny_glm5_next(text: dict) -> PreTrainedModel:
    # The family ships no text-only CausalLM; its special-token defaults index a 154k vocab.
    config = Glm5NextConfig(
        text_config={**TINY_GLM5_CONFIG, **text},
        vision_config=dict(TINY_GLM5_VISION_CONFIG),
        image_token_id=2000,
        video_token_id=2001,
        image_start_token_id=2002,
        image_end_token_id=2003,
        video_start_token_id=2004,
        video_end_token_id=2005,
    )
    return Glm5NextForConditionalGeneration(config)


def _tiny_inkling(text: dict) -> PreTrainedModel:
    return InklingForCausalLM(InklingTextConfig(**{**TINY_INKLING_CONFIG, **text}))


# The roster families whose transformers class pins parameters (not only buffers) in fp32 through
# ``_keep_in_fp32_modules_strict``, keyed by the builder of their tiny model.
PINNED_FP32_FAMILIES: dict[str, Callable[[dict], PreTrainedModel]] = {
    "deepseek_v4": _tiny_deepseek_v4,
    "glm5_next": _tiny_glm5_next,
    "inkling": _tiny_inkling,
}


def build_tiny_pinned_checkpoint(family: str, out_dir: str, *, tokenizer=None, seed: int = 0) -> str:
    """Save a seeded bf16 tiny model of a :data:`PINNED_FP32_FAMILIES` family to ``out_dir``; returns it.

    With ``tokenizer`` the vocab and pad id follow it and it is saved beside the weights, so the
    production loaders and trainers run end to end; without, the tiny config's own vocab is kept.
    """
    text = {} if tokenizer is None else {"vocab_size": len(tokenizer), "pad_token_id": tokenizer.pad_token_id}
    torch.manual_seed(seed)
    model = PINNED_FP32_FAMILIES[family](text).to(torch.bfloat16)
    model.save_pretrained(out_dir, max_shard_size=PINNED_SHARD_SIZE)
    if tokenizer is not None:
        tokenizer.save_pretrained(out_dir)
    return out_dir
