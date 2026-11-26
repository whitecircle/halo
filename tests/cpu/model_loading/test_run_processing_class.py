#!/usr/bin/env python
"""The processing class a run loads for a multimodal checkpoint. A text run keeps a shipped processor since the export
is served through it; Step-3.7 Flash ships none, so it takes the tokenizer."""

from types import SimpleNamespace

import httpx
import pytest
import torch
from huggingface_hub import constants as hub_constants
from huggingface_hub.errors import LocalEntryNotFoundError, RemoteEntryNotFoundError
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    PreTrainedTokenizerBase,
    Qwen3Config,
    Step3p7Config,
    Step3p7ForConditionalGeneration,
    Step3p7ImageProcessor,
    Step3p7Processor,
)

from halo.checkpoint.parallel.save import save_checkpoint
from halo.configs.script_arguments import CommonScriptArguments
from halo.distributed.tensor_parallel.hf_plan import is_input_embedding_tp_sharded
from halo.entrypoints import model_setup
from halo.entrypoints.model_setup import install_resolved_tokenizer
from halo.model_loading.model_preparation import resolve_auto_model_class
from halo.model_loading.tokenizer_setup import configure_run_tokenizer
from halo.trainers.loop import checkpointing
from halo.trainers.loop.checkpointing import CheckpointingMixin
from tests.common.checkpoint_io import save_context
from tests.common.models import (
    PINNED_REVISIONS,
    QWEN3_5_9B,
    STEP3P7_FLASH,
    TINY_QWEN3_CONFIG,
    TINY_STEP3P7_CONFIG,
    TINY_STEP3P7_VISION_CONFIG,
)
from tests.common.tiny_models import word_level_tokenizer
from tests.common.tokenizers import load_cached_config, load_cached_processor, load_cached_tokenizer

# Step3p7Processor reads its image placeholder's id off the tokenizer.
_TOKENIZER_WORDS = ("<im_patch>", "hello")


@pytest.fixture(autouse=True)
def _hub_offline(monkeypatch):
    """The Hub stack reads the flag at call time; the environment variable was latched at import."""
    monkeypatch.setattr(hub_constants, "HF_HUB_OFFLINE", True)


@pytest.fixture(autouse=True)
def _no_weight_load(monkeypatch):
    """The rule runs ahead of the weight load."""
    monkeypatch.setattr(model_setup, "load_model_and_tokenizer", lambda *args, **kwargs: ("model", None))


def _step3p7_checkpoint(path, *, with_processor: bool) -> str:
    Step3p7Config(text_config=TINY_STEP3P7_CONFIG, vision_config=TINY_STEP3P7_VISION_CONFIG).save_pretrained(path)
    tokenizer = word_level_tokenizer(_TOKENIZER_WORDS)
    tokenizer.save_pretrained(path)
    if with_processor:
        Step3p7Processor(image_processor=Step3p7ImageProcessor(), tokenizer=tokenizer).save_pretrained(path)
    return str(path)


def _load(model_name_or_path: str, *, vlm_run: bool, revision: str | None = None):
    model_config = SimpleNamespace(
        model_name_or_path=model_name_or_path, model_revision=revision, trust_remote_code=False
    )
    dist_args = SimpleNamespace(text_only_model=False, init_from_scratch=False)
    return model_setup.load_policy_model(
        SimpleNamespace(), SimpleNamespace(), model_config, dist_args, vlm_run=vlm_run
    )


def test_a_text_run_on_a_checkpoint_without_a_processor_loads_its_tokenizer(tmp_path):
    path = _step3p7_checkpoint(tmp_path, with_processor=False)
    with pytest.raises(OSError, match="image processor"):
        AutoProcessor.from_pretrained(path)  # Premise: requiring the processor refuses the run.

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
    """A served multimodal class builds its processor from the saved config at startup."""
    path = _step3p7_checkpoint(tmp_path, with_processor=True)

    _model, processing_class, tokenizer, _ = _load(path, vlm_run=vlm_run)

    assert isinstance(processing_class, Step3p7Processor)
    assert processing_class.tokenizer is tokenizer


def _hub_answers(monkeypatch, error: Exception):
    def download(repo_id, filename, **_kwargs):
        raise error

    monkeypatch.setattr(model_setup, "hf_hub_download", download)


_HUB_MODEL = SimpleNamespace(model_name_or_path="org/multimodal", model_revision=None, trust_remote_code=False)


def test_a_hub_404_reads_as_no_processor_config(monkeypatch):
    request = httpx.Request("HEAD", "https://huggingface.co/org/multimodal/resolve/main/processor_config.json")
    _hub_answers(monkeypatch, RemoteEntryNotFoundError("404", response=httpx.Response(404, request=request)))
    monkeypatch.setattr(hub_constants, "HF_HUB_OFFLINE", False)

    assert model_setup.load_vlm_processor(_HUB_MODEL, required=False) is None


def test_an_unreachable_hub_is_not_read_as_no_processor_config(monkeypatch):
    """Online, an unanswered probe read as none shipped would drop the processor from every export. Offline, the
    cache stands for the checkpoint."""
    _hub_answers(monkeypatch, LocalEntryNotFoundError("unreachable, not cached"))
    monkeypatch.setattr(hub_constants, "HF_HUB_OFFLINE", False)
    with pytest.raises(LocalEntryNotFoundError):
        model_setup.load_vlm_processor(_HUB_MODEL, required=False)

    monkeypatch.setattr(hub_constants, "HF_HUB_OFFLINE", True)
    assert model_setup.load_vlm_processor(_HUB_MODEL, required=False) is None


def test_a_video_processor_config_alone_counts_as_a_shipped_processor(tmp_path):
    """The probe reads the processor files the merge tools copy."""
    assert not model_setup._ships_processor_config(str(tmp_path), None, local_files_only=False)
    (tmp_path / "video_preprocessor_config.json").write_text("{}")
    assert model_setup._ships_processor_config(str(tmp_path), None, local_files_only=False)


def test_a_follower_reads_a_cache_miss_as_no_processor_config(monkeypatch):
    """A follower reads only its main rank's cache, and asks the Hub nothing."""
    asked = []

    def download(repo_id, filename, **kwargs):
        asked.append(kwargs["local_files_only"])
        raise LocalEntryNotFoundError("not cached")

    monkeypatch.setattr(model_setup, "hf_hub_download", download)
    monkeypatch.setattr(hub_constants, "HF_HUB_OFFLINE", False)

    assert not model_setup._ships_processor_config("org/multimodal", None, local_files_only=True)
    assert asked and all(asked), f"a follower probed the Hub: {asked}"


def test_the_resolved_tokenizer_lands_on_whichever_class_the_run_loaded(tmp_path):
    """A multimodal checkpoint's text run can hold a bare tokenizer."""
    loaded, resolved = word_level_tokenizer(_TOKENIZER_WORDS), word_level_tokenizer(_TOKENIZER_WORDS)
    assert install_resolved_tokenizer(loaded, resolved) is resolved

    processor = Step3p7Processor(image_processor=Step3p7ImageProcessor(), tokenizer=loaded)
    assert install_resolved_tokenizer(processor, resolved) is processor
    assert processor.tokenizer is resolved


def _export_args(path: str, *, text_only_model: bool = False):
    model_config = SimpleNamespace(model_name_or_path=path, model_revision=None, trust_remote_code=False)
    return model_config, SimpleNamespace(text_only_model=text_only_model)


def test_a_composite_export_carries_the_processor_beside_the_runs_tokenizer(tmp_path):
    path = _step3p7_checkpoint(tmp_path, with_processor=True)
    tokenizer = word_level_tokenizer(_TOKENIZER_WORDS)
    processor = model_setup.load_export_processor(*_export_args(path), tokenizer)
    assert isinstance(processor, Step3p7Processor)
    assert processor.tokenizer is tokenizer, "the window pins live on the run's tokenizer"


@pytest.mark.parametrize(
    ("with_processor", "text_only_model"), [(False, False), (True, True)], ids=["ships_none", "text_only_model"]
)
def test_an_export_with_no_processor_to_carry_stays_on_the_tokenizer(tmp_path, with_processor, text_only_model):
    path = _step3p7_checkpoint(tmp_path, with_processor=with_processor)
    model_config, args = _export_args(path, text_only_model=text_only_model)
    assert model_setup.load_export_processor(model_config, args, word_level_tokenizer(_TOKENIZER_WORDS)) is None


def test_a_text_checkpoint_export_never_builds_a_processor(tmp_path, monkeypatch):
    Qwen3Config(**TINY_QWEN3_CONFIG).save_pretrained(tmp_path)
    monkeypatch.setattr(model_setup, "load_vlm_processor", lambda *a, **k: pytest.fail("a processor was loaded"))
    model_config, args = _export_args(str(tmp_path))
    assert model_setup.load_export_processor(model_config, args, word_level_tokenizer(_TOKENIZER_WORDS)) is None


class _BaseSave:
    def __init__(self):
        self.saved_by_base = False

    def save_model(self, output_dir, _internal_call=False):
        self.saved_by_base = True


class _Exporter(CheckpointingMixin, _BaseSave):
    """The two seams of the mixin an export runs through, on a trainer with nothing else."""

    parallelism_config = None
    args = SimpleNamespace(should_save=True)


def _exporter(processing_class, export_processing_class) -> _Exporter:
    exporter = _Exporter()
    exporter.processing_class = processing_class
    exporter.export_processing_class = export_processing_class
    return exporter


def test_the_export_prefers_what_the_run_carries_for_it():
    assert _exporter("tokenizer", "processor")._exported_processing_class() == "processor"
    assert _exporter("tokenizer", None)._exported_processing_class() == "tokenizer"


@pytest.mark.parametrize("carried", [True, False], ids=["carried", "tokenizer_only"])
def test_the_base_save_writes_the_processor_a_grpo_run_carries(tmp_path, monkeypatch, carried):
    """The path of a run with no toolkit saver: one GPU, or DDP."""
    saved = []
    processor = SimpleNamespace(save_pretrained=saved.append) if carried else None
    exporter = _exporter("tokenizer", processor)
    monkeypatch.setattr(checkpointing, "save_checkpoint", lambda *a: False)
    monkeypatch.setattr(checkpointing, "finalize_exported_config", lambda *a, **k: None)
    monkeypatch.setattr(checkpointing, "config_schema_source", lambda *a: None)
    monkeypatch.setattr(checkpointing, "apply_umask_mode", lambda *a: None)
    context = SimpleNamespace(model=torch.nn.Linear(1, 1))
    context.model.config = None
    exporter._write_model_payload(context, str(tmp_path), False)
    assert exporter.saved_by_base
    assert saved == ([str(tmp_path)] if carried else [])


@pytest.mark.parametrize("carried", [True, False], ids=["carried", "tokenizer_only"])
def test_a_hub_push_uploads_the_processor_the_base_push_leaves_out(tmp_path, monkeypatch, carried):
    """The base push copies the weights into ``output_dir``, resaves ``processing_class`` and uploads the directory, none
    of it with the run's window as the served context."""
    model_context, run_window = 8192, 512
    tokenizer = word_level_tokenizer(_TOKENIZER_WORDS)
    tokenizer.model_max_length = model_context
    tokenizer = configure_run_tokenizer(
        CommonScriptArguments(), None, tokenizer, run_window, embeddings_sharded=is_input_embedding_tp_sharded
    )
    order = []
    processor = None
    if carried:
        processor = SimpleNamespace(
            tokenizer=tokenizer,
            save_pretrained=lambda path: order.append(("processor", path, tokenizer.model_max_length)),
        )
    exporter = _exporter(tokenizer, processor)
    exporter.args = SimpleNamespace(
        should_save=True, push_to_hub=True, output_dir=str(tmp_path), save_total_limit=None
    )
    monkeypatch.setattr(checkpointing, "commit_trainer_state", lambda *a: None)
    monkeypatch.setattr(checkpointing, "rotate_checkpoints", lambda **k: None)
    exporter._get_output_dir = lambda trial=None: str(tmp_path)
    exporter.state = SimpleNamespace(best_model_checkpoint=None)
    exporter._push_from_checkpoint = lambda folder: order.append(("push", folder, tokenizer.model_max_length))
    exporter._publish_checkpoint("checkpoint-1", None)
    expected = [("processor", str(tmp_path), model_context)] if carried else []
    assert order == [*expected, ("push", "checkpoint-1", model_context)]
    assert tokenizer.model_max_length == run_window, "the run must keep truncating at its window after the push"


def test_the_toolkit_saver_writes_files_an_engine_builds_a_processor_from(tmp_path):
    processor = load_cached_processor(QWEN3_5_9B)
    save_checkpoint(save_context(torch.nn.Linear(4, 4), toolkit_fsdp2=True, tokenizer=processor), str(tmp_path))
    assert any((tmp_path / name).is_file() for name in model_setup.PROCESSOR_FILES), sorted(
        path.name for path in tmp_path.iterdir()
    )


def test_a_text_run_resolves_the_step3p7_hub_release_natively():
    """The shipped recipe's own checkpoint, config files only, built on the meta device."""
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
