#!/usr/bin/env python
"""The processing class a training run loads for a multimodal checkpoint.

An image run needs the checkpoint's processor and fails loud without one. A run without image data
keeps the processor where the checkpoint ships one, since the export of the multimodal class is
served through it, and otherwise takes the tokenizer (Step-3.7 Flash ships no native processor
config). Whether the checkpoint ships one is a confirmed answer, never an unreachable Hub.

Run: pytest tests/cpu/models/test_run_processing_class.py
"""

from types import SimpleNamespace

import httpx
import pytest
import torch
from accelerate import PartialState
from huggingface_hub import constants as hub_constants
from huggingface_hub.errors import LocalEntryNotFoundError, RemoteEntryNotFoundError
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    PreTrainedTokenizerBase,
    PreTrainedTokenizerFast,
    Step3p7Config,
    Step3p7ForConditionalGeneration,
    Step3p7ImageProcessor,
    Step3p7Processor,
)

from src.distributed.loading import vlm_setup
from src.models.loading.model_preparation import resolve_auto_model_class
from src.training.script_runner import install_resolved_tokenizer
from tests.common.models import (
    PINNED_REVISIONS,
    STEP3P7_FLASH,
    TINY_STEP3P7_CONFIG,
    TINY_STEP3P7_VISION_CONFIG,
)
from tests.common.tokenizers import load_cached_config, load_cached_tokenizer


@pytest.fixture(autouse=True)
def _hub_offline(monkeypatch):
    """The Hub stack reads the flag at call time; the environment variable was latched at import."""
    monkeypatch.setattr(hub_constants, "HF_HUB_OFFLINE", True)


@pytest.fixture(autouse=True)
def _no_weight_load(monkeypatch):
    """The processing-class rule runs ahead of the weight load, which is stubbed out here."""
    PartialState()  # the loader logs through accelerate
    monkeypatch.setattr(vlm_setup, "load_model_consuming_init_kwargs", lambda *args, **kwargs: ("model", None))


def _tokenizer() -> PreTrainedTokenizerFast:
    vocab = {"<unk>": 0, "<eos>": 1, "<im_patch>": 2, "hello": 3}
    backend = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    return PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="<unk>", eos_token="<eos>", pad_token="<eos>")


def _step3p7_checkpoint(path, *, with_processor: bool) -> str:
    """A tiny Step-3.7 composite config plus tokenizer, with or without the processor's own config."""
    Step3p7Config(text_config=TINY_STEP3P7_CONFIG, vision_config=TINY_STEP3P7_VISION_CONFIG).save_pretrained(path)
    tokenizer = _tokenizer()
    tokenizer.save_pretrained(path)
    if with_processor:
        Step3p7Processor(image_processor=Step3p7ImageProcessor(), tokenizer=tokenizer).save_pretrained(path)
    return str(path)


def _load(model_name_or_path: str, *, vlm_run: bool, revision: str | None = None):
    model_config = SimpleNamespace(
        model_name_or_path=model_name_or_path, model_revision=revision, trust_remote_code=False
    )
    return vlm_setup.load_model_for_training(model_config, SimpleNamespace(), SimpleNamespace(), vlm_run=vlm_run)


def test_a_text_run_on_a_checkpoint_without_a_processor_loads_its_tokenizer(tmp_path):
    path = _step3p7_checkpoint(tmp_path, with_processor=False)
    with pytest.raises(OSError, match="image processor"):
        AutoProcessor.from_pretrained(path)  # premise: requiring the processor refuses the run

    _model, processing_class, tokenizer, is_vlm = _load(path, vlm_run=False)

    assert isinstance(processing_class, PreTrainedTokenizerBase)
    assert processing_class is tokenizer
    assert is_vlm, "the checkpoint verdict pins the reference and teacher class; it stays multimodal"


def test_an_image_run_on_a_checkpoint_without_a_processor_is_refused(tmp_path):
    path = _step3p7_checkpoint(tmp_path, with_processor=False)
    with pytest.raises(OSError, match="image processor"):
        _load(path, vlm_run=True)


@pytest.mark.parametrize("vlm_run", [False, True], ids=["text-run", "image-run"])
def test_a_shipped_processor_is_the_processing_class_of_every_run(tmp_path, vlm_run):
    """A text run keeps it too: the export saves through it, and a served multimodal class builds
    its processor from the saved config at startup."""
    path = _step3p7_checkpoint(tmp_path, with_processor=True)

    _model, processing_class, tokenizer, _ = _load(path, vlm_run=vlm_run)

    assert isinstance(processing_class, Step3p7Processor)
    assert processing_class.tokenizer is tokenizer


def _hub_answers(monkeypatch, error: Exception):
    def download(repo_id, filename, **_kwargs):
        raise error

    monkeypatch.setattr(vlm_setup, "hf_hub_download", download)


_HUB_MODEL = SimpleNamespace(model_name_or_path="org/multimodal", model_revision=None, trust_remote_code=False)


def test_a_hub_404_reads_as_no_processor_config(monkeypatch):
    request = httpx.Request("HEAD", "https://huggingface.co/org/multimodal/resolve/main/processor_config.json")
    _hub_answers(monkeypatch, RemoteEntryNotFoundError("404", response=httpx.Response(404, request=request)))
    monkeypatch.setattr(hub_constants, "HF_HUB_OFFLINE", False)

    assert vlm_setup.load_vlm_processor(_HUB_MODEL, required=False) is None


def test_an_unreachable_hub_is_not_read_as_no_processor_config(monkeypatch):
    """Online, reading an unanswered probe as "none shipped" would silently drop a shipped processor
    from every export of the run. Offline, the cache stands for the checkpoint."""
    _hub_answers(monkeypatch, LocalEntryNotFoundError("unreachable, not cached"))
    monkeypatch.setattr(hub_constants, "HF_HUB_OFFLINE", False)
    with pytest.raises(LocalEntryNotFoundError):
        vlm_setup.load_vlm_processor(_HUB_MODEL, required=False)

    monkeypatch.setattr(hub_constants, "HF_HUB_OFFLINE", True)
    assert vlm_setup.load_vlm_processor(_HUB_MODEL, required=False) is None


def test_the_resolved_tokenizer_lands_on_whichever_class_the_run_loaded(tmp_path):
    """Read off the object: a multimodal checkpoint's text run can hold a bare tokenizer."""
    loaded, resolved = _tokenizer(), _tokenizer()
    assert install_resolved_tokenizer(loaded, resolved) is resolved

    processor = Step3p7Processor(image_processor=Step3p7ImageProcessor(), tokenizer=loaded)
    assert install_resolved_tokenizer(processor, resolved) is processor
    assert processor.tokenizer is resolved


def test_a_text_run_resolves_the_step3p7_hub_release_natively():
    """The shipped recipe's own checkpoint, config files only: tokenizer, native config and native
    composite class, built on the meta device."""
    revision = PINNED_REVISIONS[STEP3P7_FLASH]
    config = load_cached_config(STEP3P7_FLASH, revision=revision)
    load_cached_tokenizer(STEP3P7_FLASH, revision=revision)

    _model, processing_class, _tokenizer, is_vlm = _load(STEP3P7_FLASH, vlm_run=False, revision=revision)

    assert is_vlm
    assert isinstance(processing_class, PreTrainedTokenizerBase)
    assert processing_class.chat_template
    assert resolve_auto_model_class(config) is AutoModelForImageTextToText
    with torch.device("meta"):
        model = AutoModelForImageTextToText.from_config(config)
    assert isinstance(model, Step3p7ForConditionalGeneration)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
