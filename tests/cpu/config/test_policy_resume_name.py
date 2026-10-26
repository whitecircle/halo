"""A Path-B resume reads the policy's weights from the checkpoint but keeps the run's base model as its
``_name_or_path``, which every exported config.json and TRL's model card (``base_model``) name, on every
policy load entry; a policy loaded from the base keeps the loader's name."""

from types import SimpleNamespace

import pytest
from accelerate import PartialState

from src.configs.script_arguments import ModelArguments
from src.entrypoints import model_setup
from src.entrypoints.script import ScriptContext

_BASE = "org/base-model"
_CHECKPOINT = "/runs/out/checkpoint-10"


def _loaded_name(source: str) -> str:
    return f"loaded-from:{source}"


@pytest.fixture(autouse=True)
def _stub_loads(monkeypatch):
    PartialState()  # the VLM load logs through accelerate's logger

    def load_distributed_model(*, model_name_or_path, **_):
        return SimpleNamespace(config=SimpleNamespace(_name_or_path=_loaded_name(model_name_or_path))), "tokenizer"

    monkeypatch.setattr(model_setup, "load_distributed_model", load_distributed_model)
    monkeypatch.setattr(model_setup, "load_vlm_processor", lambda *_, **__: SimpleNamespace(tokenizer="tokenizer"))


def _load(entry: str, *, resumed: bool, monkeypatch) -> str:
    """The ``_name_or_path`` the policy ends with, loaded through ``entry`` from the checkpoint (``resumed``)
    or from the base."""
    context = ScriptContext(
        parallelism_config=None,
        local_rank=0,
        resume_checkpoint=_CHECKPOINT if resumed else None,
        model_source=_CHECKPOINT if resumed else _BASE,
        trainer_cls=object,
    )
    assert context.policy_from_checkpoint is resumed
    training_config = SimpleNamespace(use_liger_kernel=False, liger_kernel_config=None, bf16=True)
    args = SimpleNamespace(reset_sinks=True, train_sinks=False, text_only_model=False)
    model_config = ModelArguments(model_name_or_path=_BASE)
    if entry == "load_model_and_tokenizer":
        model, _ = model_setup.load_model_and_tokenizer(context, training_config, model_config, args)
    else:
        monkeypatch.setattr(
            model_setup, "probe_checkpoint", lambda *_, **__: (None, entry == "load_policy_model[vlm]")
        )
        model = model_setup.load_policy_model(context, training_config, model_config, args, vlm_run=True).model
    return model.config._name_or_path


@pytest.mark.parametrize("entry", ["load_policy_model[text]", "load_policy_model[vlm]", "load_model_and_tokenizer"])
def test_a_policy_read_from_the_resume_checkpoint_keeps_the_base_name(entry, monkeypatch):
    assert _load(entry, resumed=True, monkeypatch=monkeypatch) == _BASE


@pytest.mark.parametrize("entry", ["load_policy_model[text]", "load_policy_model[vlm]", "load_model_and_tokenizer"])
def test_a_policy_read_from_the_base_keeps_the_loader_name(entry, monkeypatch):
    assert _load(entry, resumed=False, monkeypatch=monkeypatch) == _loaded_name(_BASE)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
