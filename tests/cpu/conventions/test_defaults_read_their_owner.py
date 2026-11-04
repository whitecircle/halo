"""A default lives with the config that owns it: a callable below the config classes takes the value from its
caller, and two constants never spell one default twice."""

import argparse
import inspect

import pytest

from halo.callbacks.generate_examples import GenerateExamplesCallback
from halo.configs.grpo_environmental import AsyncRolloutConfig
from halo.data.collators.factory import select_data_collator
from halo.data.collators.self_distill import SelfDistillTextCollator, SelfDistillVLMDataCollator
from halo.data.collators.vlm import PreprocessedVLMDataCollator, VLMDataCollator
from halo.data.pipeline.text_rows import build_chat_row_processor, render_row_conversation
from halo.data.pipeline.vlm_rows import build_vlm_row_processor
from halo.entrypoints.eval_contract import rollout_config_from_args
from halo.inference.endpoints import DEFAULT_LOCAL_BASE_URL, DEFAULT_LOCAL_SERVER_URL
from halo.inference.openai_client import REQUEST_TIMEOUT_SECONDS, generate_openai_response
from halo.trainers.distillation.sdpg import DistributedSDPGTrainer

# (callable, the parameters a config owns): every caller passes them, so none carries a default of its own.
_CONFIG_OWNED_PARAMETERS = [
    (SelfDistillTextCollator, ("answer_field", "solution_field", "confidence_power")),
    (
        SelfDistillVLMDataCollator,
        ("answer_field", "solution_field", "confidence_power", "max_length", "train_on_completions_only"),
    ),
    (VLMDataCollator, ("max_length", "train_on_completions_only")),
    (PreprocessedVLMDataCollator, ("max_length",)),
    (render_row_conversation, ("conversation_field",)),
    (build_chat_row_processor, ("conversation_field",)),
    (build_vlm_row_processor, ("conversation_field",)),
    (GenerateExamplesCallback, ("num_examples",)),
    (select_data_collator, ("train_on_completions_only",)),
    (DistributedSDPGTrainer, ("sdpg_answer_field", "opd_positive_advantage_only")),
]


def _parameters(owner) -> dict[str, inspect.Parameter]:
    """Every parameter ``owner`` takes, a subclass's ``**kwargs`` resolved through its bases' ``__init__``."""
    found = {}
    for cls in inspect.getmro(owner) if inspect.isclass(owner) else (owner,):
        init = cls.__init__ if inspect.isclass(owner) else cls
        for name, parameter in inspect.signature(init).parameters.items():
            if parameter.kind not in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD):
                found.setdefault(name, parameter)
    return found


@pytest.mark.parametrize(
    ("owner", "names"), _CONFIG_OWNED_PARAMETERS, ids=[owner.__name__ for owner, _ in _CONFIG_OWNED_PARAMETERS]
)
def test_a_config_owned_parameter_has_no_default_of_its_own(owner, names):
    parameters = _parameters(owner)
    restated = {
        name: parameters[name].default for name in names if parameters[name].default is not inspect.Parameter.empty
    }
    assert not restated, f"{owner.__name__} restates {restated}; take the value from the caller's config"


def test_one_request_timeout_serves_every_unconfigured_request():
    """The batched generation and an eval without --training_config time a request out alike."""
    batched = inspect.signature(generate_openai_response).parameters["request_timeout"].default
    args = argparse.Namespace(model="m", temperature=None, top_p=None, max_tokens=None, request_timeout=None)
    evaluated = rollout_config_from_args(args, None, None, default_temperature=0.0, default_max_tokens=8)
    assert batched == evaluated.request_timeout == REQUEST_TIMEOUT_SECONDS


def test_the_rollout_server_default_is_the_local_endpoint_the_clients_dial():
    assert AsyncRolloutConfig().rollout_server_url == DEFAULT_LOCAL_SERVER_URL
    assert f"{DEFAULT_LOCAL_SERVER_URL}/v1" == DEFAULT_LOCAL_BASE_URL


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
