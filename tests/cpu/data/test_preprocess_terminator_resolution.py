"""The completion mask's terminator set is resolved once per corpus, before any tokenization pass.

It reads the model config, so an unreadable one (a remote-code config under
``--no-trust_remote_code``, a mistyped model id) must fail before the train split is tokenized, not
after, and the read must not repeat per split.
"""

import pytest
from accelerate import PartialState
from datasets import Dataset, DatasetDict

from src.data.pipeline import preprocessing
from src.data.pipeline.preprocessed_metadata import PreprocessingConfig
from src.data.pipeline.preprocessing import preprocess_dataset
from tests.common.models import QWEN3_0_6B
from tests.common.tokenizers import load_cached_config, load_cached_tokenizer

_TURNS = [{"role": "user", "content": "What is two plus two?"}, {"role": "assistant", "content": "Four."}]

PartialState()  # the data helpers log through accelerate's rank-aware logger


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


def test_an_unreadable_model_config_fails_before_the_first_tokenization_pass(monkeypatch):
    def _unreadable(*args, **kwargs):
        raise OSError("simulated unreadable model config")

    def _map(*args, **kwargs):
        raise _Tokenized

    tokenizer = load_cached_tokenizer(QWEN3_0_6B)
    monkeypatch.setattr(preprocessing.AutoConfig, "from_pretrained", _unreadable)
    monkeypatch.setattr(preprocessing, "coordinated_map", _map)
    with pytest.raises(OSError, match="simulated unreadable model config"):
        preprocess_dataset(_corpus(), tokenizer, _config())


def test_the_model_config_is_read_once_for_every_split(monkeypatch):
    model_config = load_cached_config(QWEN3_0_6B)
    tokenizer = load_cached_tokenizer(QWEN3_0_6B)  # before the patch: the tokenizer load reads the config too
    reads = []

    def _read(*args, **kwargs):
        reads.append(args)
        return model_config

    monkeypatch.setattr(preprocessing.AutoConfig, "from_pretrained", _read)
    result = preprocess_dataset(_corpus(), tokenizer, _config())
    assert len(result["train"]) == len(result["test"]) == 2
    assert len(reads) == 1, f"the model config was read {len(reads)} times for two splits"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
