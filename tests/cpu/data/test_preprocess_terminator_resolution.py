"""The terminator set a chat bake closes its rows and masks its completions against is resolved once per corpus,
before any tokenization pass, so an unreadable model config fails before the train split is tokenized, and it is the
set a run with the same eos_token reads."""

from types import SimpleNamespace

import pytest
from datasets import Dataset, DatasetDict

from halo.data.preprocessed import tokenization
from halo.data.preprocessed.metadata import PreprocessingConfig
from halo.data.preprocessed.preparation import preprocess_dataset
from halo.data.preprocessed.prepare_dataset_tool import apply_tokenizer_overrides
from halo.data.turn_markers import resolve_eos_token_ids
from halo.model_loading.tokenizer_setup import sync_special_token_id
from tests.common.models import GLM4_FLASH, QWEN3_0_6B
from tests.common.tokenizers import load_cached_config, load_cached_tokenizer
from tests.common.vlm_fakes import FakeVLMProcessorBase

_TURNS = [{"role": "user", "content": "What is two plus two?"}, {"role": "assistant", "content": "Four."}]


class _Tokenized(Exception):
    """Raised by the stubbed map: a tokenization pass started."""


def _corpus() -> DatasetDict:
    split = Dataset.from_dict({"prompt": [_TURNS, _TURNS]})
    return DatasetDict({"train": split, "test": split})


def _config() -> PreprocessingConfig:
    return PreprocessingConfig(
        model_name_or_path=QWEN3_0_6B,
        max_length=128,
        train_on_completions_only=True,
        assistant_message_template="<|im_start|>assistant\n",
        num_proc=1,
    )


def _make_the_model_config_unreadable(monkeypatch) -> None:
    """Call after any tokenizer load, which reads the config too. A tokenization pass that starts raises."""

    def _unreadable(*args, **kwargs):
        raise OSError("simulated unreadable model config")

    def _map(*args, **kwargs):
        raise _Tokenized

    monkeypatch.setattr(tokenization.AutoConfig, "from_pretrained", _unreadable)
    monkeypatch.setattr(tokenization, "coordinated_map", _map)


def test_an_unreadable_model_config_fails_before_the_first_tokenization_pass(monkeypatch):
    tokenizer = load_cached_tokenizer(QWEN3_0_6B)
    _make_the_model_config_unreadable(monkeypatch)
    with pytest.raises(OSError, match="simulated unreadable model config"):
        preprocess_dataset(_corpus(), tokenizer, _config())


@pytest.mark.parametrize("is_vlm", [False, True], ids=["text", "vlm"])
def test_an_unreadable_model_config_fails_a_full_sequence_chat_bake_too(monkeypatch, is_vlm):
    """Its rows close their last assistant turn against the terminators, so an unreadable config must not pass as a
    bake that closes nothing."""
    tokenizer_or_processor = FakeVLMProcessorBase() if is_vlm else load_cached_tokenizer(QWEN3_0_6B)
    config = PreprocessingConfig(model_name_or_path=QWEN3_0_6B, max_length=128, is_vlm=is_vlm, num_proc=1)
    _make_the_model_config_unreadable(monkeypatch)
    with pytest.raises(OSError, match="simulated unreadable model config"):
        preprocess_dataset(_corpus(), tokenizer_or_processor, config)


def test_a_raw_text_bake_reads_no_model_config(monkeypatch):
    """It renders no chat turn, so nothing in it waits on the terminators."""
    tokenizer = load_cached_tokenizer(QWEN3_0_6B)
    config = PreprocessingConfig(model_name_or_path=QWEN3_0_6B, max_length=128, mode="text", num_proc=1)
    _make_the_model_config_unreadable(monkeypatch)
    with pytest.raises(_Tokenized):
        preprocess_dataset(Dataset.from_dict({"text": ["Two plus two is four."]}), tokenizer, config)


def test_the_model_config_is_read_once_for_every_split(monkeypatch):
    model_config = load_cached_config(QWEN3_0_6B)
    tokenizer = load_cached_tokenizer(QWEN3_0_6B)  # before the patch: the tokenizer load reads the config too
    reads = []

    def _read(*args, **kwargs):
        reads.append(args)
        return model_config

    monkeypatch.setattr(tokenization.AutoConfig, "from_pretrained", _read)
    result = preprocess_dataset(_corpus(), tokenizer, _config())
    assert len(result["train"]) == len(result["test"]) == 2
    assert len(reads) == 1, f"the model config was read {len(reads)} times for two splits"


@pytest.mark.parametrize(
    ("model", "eos_token"),
    [
        (QWEN3_0_6B, "<|im_end|>"),
        # The config names <|im_end|> alone, so the run replaces it.
        (QWEN3_0_6B, "<|endoftext|>"),
        # The config lists it beside the role markers that end a turn, so the run keeps the list.
        (GLM4_FLASH, "<|endoftext|>"),
    ],
)
def test_the_bake_resolves_the_terminators_a_run_with_its_eos_token_reads(monkeypatch, model, eos_token):
    """Else a baked row's last turn closes, and its spans end, at an id the run's collator never ends one at."""
    config = PreprocessingConfig(model_name_or_path=model, max_length=128, eos_token=eos_token, num_proc=1)
    tokenizer = load_cached_tokenizer(model)
    apply_tokenizer_overrides(tokenizer, config)
    run_model = SimpleNamespace(config=load_cached_config(model), generation_config=None)
    sync_special_token_id(run_model, "eos_token_id", tokenizer.eos_token_id)
    checkpoint_config = load_cached_config(model)
    monkeypatch.setattr(tokenization.AutoConfig, "from_pretrained", lambda *args, **kwargs: checkpoint_config)

    assert tokenization.resolve_label_eos_token_ids(config, tokenizer) == resolve_eos_token_ids(
        tokenizer, run_model.config
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
