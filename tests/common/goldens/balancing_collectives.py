"""Balancing collective-sequence golden: what configuring a run's MoE balancing and building its callback
roster issue on rank 0 of a two-rank job, for every requested ``moe_balancing`` mode, loss kind and
weight-sync flag, over tiny dense and MoE stand-ins.

A case's trace holds the collectives in call order with the toolkit's log records between them, what the
build raised, the roster it returned (the balancing callbacks with their settings) and the router config
fields it left.
"""

from __future__ import annotations

import contextlib
import itertools
import json
import logging
from collections.abc import Callable, Iterator
from pathlib import Path
from types import SimpleNamespace

import torch
from transformers import TrainerCallback

import src.models.moe.balancing as moe_balancing
from src.callbacks.balancing import configure_moe_balancing
from src.callbacks.factory import build_run_callbacks
from src.callbacks.moe_metrics import MoELoadMetricsCallback
from src.models.moe.family_specs import MoEFamilySpec, NativeBalancingSlot
from src.models.moe.router_logits import router_logits_forced_off
from tests.common.collective_recorder import CollectiveTrace, recording_rank
from tests.common.goldens import regen_main
from tests.common.parallelism import make_parallelism_config

GOLDEN_PATH = Path(__file__).with_suffix(".json")
WORLD_SIZE = 2
MODES = ("auto", "aux_loss", "bias_update", "bias_update_transient", "none")
# The logging period MoE load callbacks receive, so the trace shows it reaching them.
LOGGING_STEPS = 5
# The balancing callbacks' settings a trace records.
_CALLBACK_SETTINGS = ("topk", "enable_router_logits", "update_rate", "exclude_last_slot", "log_every_n_steps")


class _Config:
    def __init__(self, router_aux_loss_coef=0.001, **fields):
        self.output_router_logits = False
        self.router_aux_loss_coef = router_aux_loss_coef
        self.model_type = "golden_moe"
        for name, value in fields.items():
            setattr(self, name, value)


class _Model(torch.nn.Module):
    """A config and the submodules ``modules()`` yields; ``forward`` declares ``output_router_logits``, the
    parameter that says the class honours the config flag."""

    def __init__(self, config: _Config, submodules=()):
        super().__init__()
        self.config = config
        self._submodules = list(submodules)

    def forward(self, input_ids=None, output_router_logits=None, **kwargs):
        raise NotImplementedError("stand-in: only the signature is read")

    def modules(self):
        return iter([self, *self._submodules])

    def named_modules(self, *args, **kwargs):
        return iter([("", self), *((f"sub{i}", module) for i, module in enumerate(self._submodules))])

    def register_forward_hook(self, hook):
        return SimpleNamespace(remove=lambda: None)


class _NativeAdoptingLayer(torch.nn.Module):
    """An EP layer whose bias adopts a checkpoint-exported slot (GPT-OSS, GLM-4)."""

    FAMILY = MoEFamilySpec(model_types=(), balancing_slot=NativeBalancingSlot(("balancing_biases",)))
    SUPPORTS_BIAS_BALANCING = True

    def can_adopt_native_balancing(self):
        return True

    def enable_bias_balancing(self):
        self._native_balancing_bias_adopted = True
        self.register_buffer("balancing_biases", torch.zeros(4))
        self.expert_load_counter = None
        return True


class _TransientLayer:
    """An EP layer whose bias lives in a transient side-buffer (Qwen3, Mistral4)."""

    SUPPORTS_BIAS_BALANCING = True

    def enable_bias_balancing(self):
        self.balancing_biases = torch.zeros(4)
        self.expert_load_counter = None
        return True


class _NativeRouter(torch.nn.Module):
    """A router shipping its own balancing buffer and a trailing discard slot (Zaya)."""

    _has_discard_expert_slot = True

    def __init__(self):
        super().__init__()
        self.register_buffer("balancing_biases", torch.zeros(5))
        self.expert_load_counter = None


def _moe(**fields) -> _Config:
    return _Config(num_experts=4, num_experts_per_tok=2, **fields)


MODELS: dict[str, Callable[[], _Model]] = {
    "dense": lambda: _Model(_Config()),
    "moe without wrappers": lambda: _Model(_moe()),
    "moe native slot": lambda: _Model(_moe(), [_NativeAdoptingLayer()]),
    "moe transient": lambda: _Model(_moe(), [_TransientLayer()]),
    "moe native router": lambda: _Model(_moe(router_aux_loss_coef=0.0), [_NativeRouter()]),
}
CASES = tuple(
    (model, mode, policy_gradient_loss, syncs)
    for model, mode, policy_gradient_loss, syncs in itertools.product(MODELS, MODES, (False, True), (False, True))
)


class _GenerateStandIn(TrainerCallback):
    """The generation-eval callback a script hands over."""


class _TracedRecords(logging.Handler):
    def __init__(self, trace: CollectiveTrace):
        super().__init__(logging.INFO)
        self.trace = trace

    def emit(self, record: logging.LogRecord) -> None:
        if record.name.startswith("src."):
            self.trace.record("log", record.levelname, record.getMessage())


@contextlib.contextmanager
def _traced_records(trace: CollectiveTrace) -> Iterator[None]:
    """Every INFO-and-above toolkit record into ``trace``, in place of the root handlers and whatever
    level :func:`regen_main` disables."""
    root = logging.getLogger()
    handlers, level, disabled = root.handlers[:], root.level, root.manager.disable
    root.handlers, root.level = [_TracedRecords(trace)], logging.INFO
    logging.disable(logging.NOTSET)
    try:
        yield
    finally:
        root.handlers, root.level = handlers, level
        logging.disable(disabled)


def _script_args(mode: str) -> SimpleNamespace:
    return SimpleNamespace(
        moe_balancing=mode,
        enable_efficiency_metrics=True,
        enable_moe_metrics=True,
        enable_torch_profiler=False,
        router_balancing_rate=1e-3,
        num_full_model_params=None,
        report_mfu_diagnostics=False,
    )


def _roster_entry(callback) -> str:
    if isinstance(callback, type):
        return callback.__name__
    if not isinstance(callback, MoELoadMetricsCallback):
        return type(callback).__name__
    settings = {name: getattr(callback, name) for name in _CALLBACK_SETTINGS if hasattr(callback, name)}
    return f"{type(callback).__name__} {settings}"


def case_key(model: str, mode: str, policy_gradient_loss: bool, syncs: bool) -> str:
    return f"{model} / {mode} / policy_gradient_loss={policy_gradient_loss} / syncs_to_external_generator={syncs}"


def trace_case(model_name: str, mode: str, policy_gradient_loss: bool, syncs: bool) -> list[str]:
    """What configuring balancing and building the roster issue for one case, as rank 0."""
    moe_balancing._warned_transient_only_auto.clear()
    moe_balancing._warned_unservable_auto.clear()
    parallelism_config = make_parallelism_config(world_size=WORLD_SIZE, gpus_per_node=WORLD_SIZE)
    model = MODELS[model_name]()
    training_config = SimpleNamespace(logging_steps=LOGGING_STEPS, gradient_accumulation_steps=1)
    with recording_rank(0, WORLD_SIZE, WORLD_SIZE) as trace, _traced_records(trace):
        try:
            args = _script_args(mode)
            balancing_mode = configure_moe_balancing(
                args,
                model,
                parallelism_config,
                policy_gradient_loss=policy_gradient_loss,
                syncs_to_external_generator=syncs,
            )
            callbacks = build_run_callbacks(
                args,
                training_config,
                model,
                parallelism_config,
                balancing_mode=balancing_mode,
                generate_callback=_GenerateStandIn(),
            )
        except Exception as error:
            trace.record("raised", type(error).__name__, str(error))
        else:
            trace.record("roster", [_roster_entry(callback) for callback in callbacks])
    config = model.config
    trace.events.append(
        f"config output_router_logits={config.output_router_logits} "
        f"router_aux_loss_coef={config.router_aux_loss_coef} forced_off={router_logits_forced_off(config)}"
    )
    return trace.events


def render_golden() -> str:
    """The golden file: every case's trace, one event a line."""
    return json.dumps({case_key(*case): trace_case(*case) for case in CASES}, indent=2) + "\n"


if __name__ == "__main__":
    regen_main(GOLDEN_PATH, render_golden)
