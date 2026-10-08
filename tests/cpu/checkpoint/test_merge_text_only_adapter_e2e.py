#!/usr/bin/env python
"""An adapter trained through a multimodal checkpoint's text-only class must merge — or refuse loudly
— into a checkpoint whose keys are the layout its config declares.

``text_only_model: true`` trains a VLM checkpoint as its ``*ForCausalLM`` sibling, whose module tree
spells the decoder ``model.layers``; the merge tool resolves the widest class for the same
checkpoint, which spells it ``model.language_model.layers``. PEFT resolves no key across that gap,
warns, and merges nothing, so the tool would write the untouched base as a finished model. The merge
reads the adapter's keys, meets them through the class they address, and refuses a base the keys do
not belong to.

That load strips the checkpoint's ``language_model`` prefix, and a stock ``save_pretrained`` puts it
back: a text-only config over wrapper-prefixed keys, which transformers re-strips on reload (so a
reload through it proves nothing) while ``reattach_vision_tower.py`` nests the prefix a second time.
The chain test therefore pins the on-disk keys to what a fresh model of each saved config writes and
reloads every artifact through the toolkit loader.

    python tests/cpu/checkpoint/test_merge_text_only_adapter_e2e.py
"""

from __future__ import annotations

# Ahead of the Qwen3.5 modeling modules, which bind transformers' hub-kernel fallback at import: the
# chain test runs CPU forwards, which otherwise reach the CUDA-only causal_conv1d kernel.
import src.models.patches.kernel_dispatch  # noqa: F401  # isort: skip

import tempfile
from pathlib import Path
from typing import NamedTuple

import pytest
import torch
from accelerate import PartialState
from peft import LoraConfig, PeftModel, get_peft_model
from safetensors.torch import load_file, save_file
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import (
    AutoConfig,
    PreTrainedTokenizerBase,
    PreTrainedTokenizerFast,
    Qwen3_5Config,
    Qwen3_5ForCausalLM,
    Qwen3_5ForConditionalGeneration,
    Qwen3_5MoeConfig,
    Qwen3_5MoeForCausalLM,
    Qwen3_5MoeForConditionalGeneration,
)

from scripts.after_training.reattach_vision_tower import reattach_vision_tower
from src.checkpoint.format import ADAPTER_SAFETENSORS_FILE
from src.checkpoint.tool_io import PROCESSOR_FILES, iter_checkpoint_shard_entries
from src.models.loading.model_preparation import auto_load_model, resolve_auto_model_class
from src.models.loading.tokenizer_setup import load_processing_class
from tests.common.models import QWEN3_5_2B, TINY_QWEN35_CONFIG, TINY_QWEN35_MOE_CONFIG
from tests.common.tokenizers import load_cached_processor
from tests.common.utils import load_script_module

_TARGET, _CONTROL = "q_proj", "o_proj"
_WRAPPER_TEXT_PREFIX = "model.language_model."


class _Family(NamedTuple):
    config_cls: type
    wrapper_cls: type
    text_cls: type
    text_config: dict


# The families whose transformers text-only class loads the wrapper checkpoint by stripping its prefix.
_FAMILIES = {
    "qwen3_5": _Family(Qwen3_5Config, Qwen3_5ForConditionalGeneration, Qwen3_5ForCausalLM, TINY_QWEN35_CONFIG),
    "qwen3_5_moe": _Family(
        Qwen3_5MoeConfig, Qwen3_5MoeForConditionalGeneration, Qwen3_5MoeForCausalLM, TINY_QWEN35_MOE_CONFIG
    ),
}


def _tiny_tokenizer() -> PreTrainedTokenizerFast:
    backend = Tokenizer(models.WordLevel({"<unk>": 0, "<eos>": 1, "hello": 2, "world": 3}, unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    return PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="<unk>", eos_token="<eos>", pad_token="<eos>")


# What a run customizes on the tokenizer it saves beside its adapter, none of which the base carries.
_RUN_CHAT_TEMPLATE = "{% for message in messages %}<run>{{ message['content'] }}{% endfor %}"
_RUN_SPECIAL_TOKEN = "<run_special>"


def _run_tokenizer() -> PreTrainedTokenizerFast:
    tokenizer = _tiny_tokenizer()
    tokenizer.chat_template = _RUN_CHAT_TEMPLATE
    tokenizer.add_special_tokens({"additional_special_tokens": [_RUN_SPECIAL_TOKEN]})
    return tokenizer


def _assert_is_the_run_tokenizer(tokenizer) -> None:
    assert isinstance(tokenizer, PreTrainedTokenizerBase), type(tokenizer)
    assert tokenizer.chat_template == _RUN_CHAT_TEMPLATE, "the run's chat template did not ship"
    assert _RUN_SPECIAL_TOKEN in tokenizer.get_vocab() and len(tokenizer) == len(_run_tokenizer())


def _build_multimodal_base(base_dir: Path, family: _Family) -> None:
    torch.manual_seed(0)
    vision = {"depth": 1, "hidden_size": 16, "intermediate_size": 16, "num_heads": 2}
    config = family.config_cls(
        text_config={**family.text_config, "tie_word_embeddings": False},
        vision_config={**vision, "out_hidden_size": family.text_config["hidden_size"]},
    )
    family.wrapper_cls(config).save_pretrained(base_dir)
    _tiny_tokenizer().save_pretrained(base_dir)


def _train_text_only_adapter(
    base_dir: Path, adapter_dir: Path, family: _Family, *, model_cls: type | None = None, processing_class=None
) -> dict[str, torch.Tensor]:
    """The adapter a ``text_only_model`` run saves: keys spelled on the text-only class's tree (or on
    ``model_cls``'s, for a run on the multimodal class), beside the run's processing class — its own
    tokenizer unless ``processing_class`` is given."""
    model = (model_cls or family.text_cls).from_pretrained(base_dir, dtype=torch.float32)
    peft_model = get_peft_model(model, LoraConfig(r=4, lora_alpha=8, target_modules=[_TARGET], task_type="CAUSAL_LM"))
    deltas: dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for name, module in peft_model.named_modules():
            if name.endswith(_TARGET) and hasattr(module, "lora_B"):
                module.lora_B["default"].weight.normal_()
                delta = (module.lora_B["default"].weight @ module.lora_A["default"].weight) * module.scaling["default"]
                deltas[name.removeprefix("base_model.model.") + ".weight"] = delta.clone()
    peft_model.save_pretrained(adapter_dir)
    (processing_class or _run_tokenizer()).save_pretrained(adapter_dir)
    return deltas


def _merge_script():
    return load_script_module("scripts/after_training/merge_peft_adapters.py", "merge_peft_adapters_text_only")


def _on_disk_keys(checkpoint_dir: Path) -> set[str]:
    return {key for _shard, _reader, key in iter_checkpoint_shard_entries(str(checkpoint_dir))}


def _declared_layout(checkpoint_dir: Path, scratch: Path) -> set[str]:
    """The keys a fresh model of ``checkpoint_dir``'s config writes: the layout that config declares."""
    config = AutoConfig.from_pretrained(checkpoint_dir)
    resolve_auto_model_class(config).from_config(config).save_pretrained(scratch)
    return _on_disk_keys(scratch)


def test_a_text_only_adapter_merges_into_the_class_it_was_trained_through():
    PartialState()
    family = _FAMILIES["qwen3_5"]
    with tempfile.TemporaryDirectory() as tmp:
        base, adapter, out = Path(tmp) / "base", Path(tmp) / "adapter", Path(tmp) / "out"
        _build_multimodal_base(base, family)
        deltas = _train_text_only_adapter(base, adapter, family)

        _merge_script().merge_peft_adapter(
            adapter_dir=str(adapter), output_dir=str(out), dtype=torch.float32, verbose=False
        )

        merged = Qwen3_5ForCausalLM.from_pretrained(out, dtype=torch.float32).state_dict()
        original = Qwen3_5ForCausalLM.from_pretrained(base, dtype=torch.float32).state_dict()

    assert deltas and all(delta.abs().max() > 1e-3 for delta in deltas.values()), "premise: the adapter is a no-op"
    for key, delta in deltas.items():
        assert torch.allclose(merged[key], original[key] + delta, atol=1e-5), (
            f"{key} did not receive its delta: the merge met the adapter through the wrong class and PEFT merged nothing"
        )
    controls = [key for key in original if key.endswith(f"{_CONTROL}.weight")]
    assert controls and all(torch.equal(merged[key], original[key]) for key in controls)


@pytest.mark.parametrize("family_name", sorted(_FAMILIES))
def test_a_merged_text_only_adapter_reattaches_into_a_checkpoint_the_toolkit_loader_reads_back(family_name, tmp_path):
    """merge → reattach → reload: both artifacts carry the layout their configs declare, and both
    reload through the toolkit loader with every tensor consumed and the merged model's logits."""
    PartialState()
    family = _FAMILIES[family_name]
    base, adapter, merged, served = (tmp_path / name for name in ("base", "adapter", "merged", "served"))
    _build_multimodal_base(base, family)
    _train_text_only_adapter(base, adapter, family)
    input_ids = torch.randint(4, family.text_config["vocab_size"], (2, 12), generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        text_base = family.text_cls.from_pretrained(base, dtype=torch.float32)
        reference = PeftModel.from_pretrained(text_base, adapter).merge_and_unload().eval()(input_ids=input_ids).logits

    _merge_script().merge_peft_adapter(
        adapter_dir=str(adapter), output_dir=str(merged), dtype=torch.float32, verbose=False
    )
    merged_keys = _on_disk_keys(merged)
    assert not any(key.startswith(_WRAPPER_TEXT_PREFIX) for key in merged_keys), sorted(merged_keys)[:4]
    assert merged_keys == _declared_layout(merged, tmp_path / "fresh_merged")

    reattach_vision_tower(str(merged), str(base), str(served), trust_remote_code=False)
    assert _on_disk_keys(served) == _declared_layout(served, tmp_path / "fresh_served")

    for directory, expected_class in ((merged, family.text_cls), (served, family.wrapper_cls)):
        model = auto_load_model(str(directory), dtype=torch.float32).eval()
        assert type(model) is expected_class
        _, loading_info = expected_class.from_pretrained(directory, dtype=torch.float32, output_loading_info=True)
        assert not loading_info["missing_keys"] and not loading_info["unexpected_keys"], loading_info
        with torch.no_grad():
            logits = model(input_ids=input_ids).logits
        torch.testing.assert_close(logits, reference, rtol=1e-5, atol=1e-5)


def test_the_bf16_merge_writes_the_text_only_layout(tmp_path):
    """``convert_to_bf16 --peft --merge_adapter`` shares the merge's save, so it owes the same layout."""
    PartialState()
    family = _FAMILIES["qwen3_5"]
    base, adapter, out = tmp_path / "base", tmp_path / "adapter", tmp_path / "out"
    _build_multimodal_base(base, family)
    _train_text_only_adapter(base, adapter, family)

    convert = load_script_module("scripts/after_training/convert_to_bf16.py", "convert_to_bf16_text_only")
    convert.convert_to_bf16(str(adapter), str(out), "causal_lm", is_peft=True, merge_adapter=True)

    assert _on_disk_keys(out) == _declared_layout(out, tmp_path / "fresh")


def _build_multimodal_base_with_processor(base_dir: Path, family: _Family):
    """A multimodal base shipping the family's full processor, as a Hub VLM checkpoint does."""
    processor = load_cached_processor(QWEN3_5_2B)
    _build_multimodal_base(base_dir, family)
    for component in (processor, processor.image_processor, processor.video_processor):
        component.save_pretrained(base_dir)
    assert all((base_dir / name).is_file() for name in PROCESSOR_FILES), "premise: the base ships every processor file"
    return processor


@pytest.mark.parametrize("text_only", [True, False], ids=["text-only-class", "multimodal-class"])
def test_the_merge_ships_the_runs_processing_class_for_the_class_it_merged_through(text_only, tmp_path):
    """The merge ships the processing class the run saved beside its adapter, not the base's.

    A text-only merge is the export a ``text_only_model`` run writes: a ``*_text`` config with no
    vision path, so the run's tokenizer ships (its chat template and added tokens, which an embedding
    saved under ``save_embedding_layers`` is sized to) and none of the multimodal base's processor
    files, which would have ``AutoProcessor`` insert image tokens for a model with no tower.
    ``reattach_vision_tower.py`` takes the processor files from the base and the tokenizer from the
    export, so the served artifact carries both. A run on the multimodal class saves its processor,
    and the merge keeps it."""
    PartialState()
    family = _FAMILIES["qwen3_5"]
    base, adapter, merged, served = (tmp_path / name for name in ("base", "adapter", "merged", "served"))
    processor = _build_multimodal_base_with_processor(base, family)
    if text_only:
        _train_text_only_adapter(base, adapter, family)
    else:
        processor.tokenizer.add_special_tokens({"additional_special_tokens": [_RUN_SPECIAL_TOKEN]})
        _train_text_only_adapter(base, adapter, family, model_cls=family.wrapper_cls, processing_class=processor)

    _merge_script().merge_peft_adapter(
        adapter_dir=str(adapter), output_dir=str(merged), dtype=torch.float32, verbose=False
    )

    shipped = [name for name in PROCESSOR_FILES if (merged / name).is_file()]
    if not text_only:
        merged_processor = load_processing_class(str(merged))
        assert shipped and not isinstance(merged_processor, PreTrainedTokenizerBase)
        assert _RUN_SPECIAL_TOKEN in merged_processor.tokenizer.get_vocab(), "the run's processor did not ship"
        return
    assert not shipped, f"the text-only merge carries the multimodal base's processor files {shipped}"
    _assert_is_the_run_tokenizer(load_processing_class(str(merged)))
    reattach_vision_tower(str(merged), str(base), str(served), trust_remote_code=False)
    served_processor = load_processing_class(str(served))
    assert not isinstance(served_processor, PreTrainedTokenizerBase)
    _assert_is_the_run_tokenizer(served_processor.tokenizer)


def test_an_unmerged_bf16_adapter_save_ships_what_the_merge_ships(tmp_path):
    """``convert_to_bf16 --peft`` without ``--merge_adapter`` writes the processing class the merge
    would: for a text-only adapter, the run's tokenizer and none of the multimodal base's processor
    files, so merging the converted adapter later reaches the same artifact."""
    PartialState()
    family = _FAMILIES["qwen3_5"]
    base, adapter, out = tmp_path / "base", tmp_path / "adapter", tmp_path / "out"
    _build_multimodal_base_with_processor(base, family)
    _train_text_only_adapter(base, adapter, family)

    convert = load_script_module("scripts/after_training/convert_to_bf16.py", "convert_to_bf16_unmerged_text_only")
    convert.convert_to_bf16(str(adapter), str(out), "causal_lm", is_peft=True)

    shipped = [name for name in PROCESSOR_FILES if (out / name).is_file()]
    assert not shipped, f"the text-only adapter save carries the multimodal base's processor files {shipped}"
    _assert_is_the_run_tokenizer(load_processing_class(str(out)))


def test_an_adapter_whose_keys_address_no_module_of_the_base_is_refused():
    """A relabelled key set that fits neither class must not produce a merged checkpoint at all."""
    PartialState()
    family = _FAMILIES["qwen3_5"]
    with tempfile.TemporaryDirectory() as tmp:
        base, adapter, out = Path(tmp) / "base", Path(tmp) / "adapter", Path(tmp) / "out"
        _build_multimodal_base(base, family)
        _train_text_only_adapter(base, adapter, family)
        weights = adapter / ADAPTER_SAFETENSORS_FILE
        save_file({key.replace(".layers.", ".blocks."): tensor for key, tensor in load_file(weights).items()}, weights)

        with pytest.raises(ValueError, match="does not have"):
            _merge_script().merge_peft_adapter(
                adapter_dir=str(adapter), output_dir=str(out), dtype=torch.float32, verbose=False
            )
        assert not out.exists(), "a refused merge left an output directory behind"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
