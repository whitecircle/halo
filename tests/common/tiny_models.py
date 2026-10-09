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
    Cohere2VisionConfig,
    Cohere2VisionForConditionalGeneration,
    DeepseekV4Config,
    DeepseekV4ForCausalLM,
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
    Mistral3Config,
    Mistral3ForConditionalGeneration,
    PreTrainedModel,
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
    Step3p7Config,
    Step3p7ForConditionalGeneration,
    ZayaConfig,
    ZayaForCausalLM,
)
from transformers.models.mistral4 import Mistral4Config, Mistral4ForCausalLM
from transformers.models.zaya.modeling_zaya import ZayaQKNorm

from src.models.patches.remote_code_compat import apply_remote_code_compat_shims
from src.models.structure import fp32_pinned_param_names
from tests.common.distributed import shared_scratch_dir
from tests.common.models import (
    BAILING_MOE_LING_MINI,
    MISTRAL3_119B_MOE,
    TINY_BAILING_MOE_CONFIG,
    TINY_COHERE2_MOE_CONFIG,
    TINY_DSV4_CONFIG,
    TINY_GEMMA4_MOE_CONFIG,
    TINY_GEMMA4_VISION_CONFIG,
    TINY_GLM4_MOE_LITE_CONFIG,
    TINY_GLM5_CONFIG,
    TINY_GLM5_VISION_CONFIG,
    TINY_GPTOSS_CONFIG,
    TINY_INKLING_CONFIG,
    TINY_INKLING_VISION_CONFIG,
    TINY_LAGUNA_CONFIG,
    TINY_LFM2_MOE_CONFIG,
    TINY_MISTRAL4_CONFIG,
    TINY_PIXTRAL_VISION_CONFIG,
    TINY_QWEN3_CONFIG,
    TINY_QWEN3_MOE_CONFIG,
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
# Seed ``randomize_tid2eid`` fills the hash table from unless a test pins its own.
DSV4_TID2EID_SEED = 1234
# The :data:`TINY_MOE_FAMILIES` whose transformers class pins parameters (not only buffers) in fp32
# through ``_keep_in_fp32_modules_strict``.
PINNED_FP32_FAMILIES = ("deepseek_v4", "glm5_next", "inkling_text")
# The files a synthetic checkpoint copies from its release so ``AutoTokenizer`` loads it offline.
TOKENIZER_FILE_PREFIXES = ("tokenizer", "special_tokens", "chat_template")


def padded_vocab_size(tokenizer) -> int:
    """``tokenizer``'s vocab rounded up to :data:`VOCAB_PAD_MULTIPLE`, as a release pads it so a TP-sharded
    embedding and head divide it."""
    return -(-len(tokenizer) // VOCAB_PAD_MULTIPLE) * VOCAB_PAD_MULTIPLE


class _WeightLeaf(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))


def module_with_weight_keys(paths: list[str]) -> torch.nn.Module:
    """A module tree whose state dict is exactly ``paths`` (each ending in ``.weight``): the key space a
    checkpoint-conversion or master-replay test maps disk keys onto, without a model behind it."""
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


def copy_release_tokenizer(repo_id: str, out_dir: Path) -> None:
    """Copy ``repo_id``'s tokenizer files into ``out_dir``, downloading only those, not the weights."""
    tokenizer_dir = Path(snapshot_download(repo_id, allow_patterns=[f"{p}*" for p in TOKENIZER_FILE_PREFIXES]))
    out_dir.mkdir(parents=True, exist_ok=True)
    for src in tokenizer_dir.iterdir():
        if src.is_file() and src.name.startswith(TOKENIZER_FILE_PREFIXES):
            shutil.copy2(src, out_dir / src.name)


def build_tied_qwen3_checkpoint(target_dir: str, tokenizer, *, moe: bool, seed: int) -> None:
    """Save a random-init :data:`TINY_TIED_QWEN3_CONFIG` model (Qwen3-MoE with ``moe``) at ``tokenizer``'s
    padded vocab, with that tokenizer, as a checkpoint the production loaders read. Rank 0 only."""
    torch.manual_seed(seed)
    fields = {**TINY_TIED_QWEN3_CONFIG, "vocab_size": padded_vocab_size(tokenizer)}
    if moe:
        model = Qwen3MoeForCausalLM(Qwen3MoeConfig(**fields, **TINY_TIED_QWEN3_MOE_FIELDS))
    else:
        model = Qwen3ForCausalLM(Qwen3Config(**fields))
    model.to(torch.bfloat16).save_pretrained(target_dir)
    tokenizer.save_pretrained(target_dir)


def build_tiny_mistral4_checkpoint(out_dir: Path, seed: int = 0) -> Path:
    """Write a random-init :data:`TINY_MISTRAL4_CONFIG` model to ``out_dir`` and return the path.

    The layout ``save_pretrained`` writes (``model.safetensors`` + ``config.json``), plus the release's
    tokenizer files, so the lazy loader and ``load_distributed_model`` run end to end without the
    119B download.
    """
    copy_release_tokenizer(MISTRAL3_119B_MOE, out_dir)
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
# The multimodal wrappers an EP family's text tower ships under, keyed by the wrapper's ``model_type``
# (GLM-5 Next and Step-3.7 ship no text-only class, so their roster model above is already the
# wrapper). A ``text_config`` names its ``model_type`` where the wrapper defaults to another family's
# tower. ``tests/cpu/checkpoint/test_ep_hub_namespace_export.py`` holds the two rosters to every
# multimodal ``model_type`` the EP registry claims.
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
}
# The dense model the family sweeps pair with the MoE roster.
TINY_DENSE_FAMILY = TinyFamily(_causal(Qwen3Config, Qwen3ForCausalLM, TINY_QWEN3_CONFIG))
# The dense Qwen3.5 text model (gated DeltaNet and attention layers), which the MoE roster does not carry.
TINY_QWEN35_DENSE_FAMILY = TinyFamily(_causal(Qwen3_5TextConfig, Qwen3_5ForCausalLM, TINY_QWEN35_CONFIG))
# The dense roster, keyed as the dense rows name it (``--family``).
TINY_DENSE_FAMILIES: dict[str, TinyFamily] = {"qwen3": TINY_DENSE_FAMILY, "qwen3_5": TINY_QWEN35_DENSE_FAMILY}


def tiny_family_model(family: TinyFamily, tokenizer=None, *, overrides: dict | None = None) -> PreTrainedModel:
    """``family``'s random-init tiny model (fp32, seeded by the caller), at ``tokenizer``'s vocab when one
    is given and at the tiny config's own otherwise. ``overrides`` sets further text-config fields."""
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
) -> None:
    """Save ``family``'s seeded tiny model at ``tokenizer``'s vocab, with that tokenizer, as a checkpoint
    the production loaders read. Rank 0 only.

    Sharded, so the checkpoint carries the index a family's attention projection names are read from
    (:func:`tests.common.peft_helpers.attention_target_modules`). Without ``tokenizer`` the tiny config's
    own vocab is kept and no tokenizer is saved. ``fp32_pins`` stores the family's fp32-pinned parameters
    at full fp32 precision, as a release does, rather than rounded through bf16. ``edit`` changes the
    random-init model before it is saved (a test pinning its routing, say).
    """
    torch.manual_seed(seed)
    model = tiny_family_model(family, tokenizer)
    if edit is not None:
        edit(model)
    pinned = fp32_pinned_param_names(model) if fp32_pins else frozenset()
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
) -> str:
    """:func:`build_tiny_family_checkpoint` into the rank-shared scratch dir ``name``, returned: rank 0
    builds it (removing it at teardown) and every rank waits for it. Collective."""
    path = shared_scratch_dir(name)
    if ctx.rank == 0:
        shutil.rmtree(path, ignore_errors=True)
        ctx.on_teardown(lambda: shutil.rmtree(path, ignore_errors=True))
        build_tiny_family_checkpoint(family, path, tokenizer, seed, fp32_pins=fp32_pins, edit=edit)
    ctx.barrier()
    return path
