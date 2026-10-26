#!/usr/bin/env python
"""Tests for GenerateExamplesCallback. Run: python tests/cpu/callbacks/test_generate_examples.py"""

import itertools
import logging
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import pytest
import torch
from datasets import Dataset

import src.callbacks.generate_examples as cb_module
from src.callbacks.generate_examples import (
    _MAX_PRINTED_PROMPT_CHARS,
    GenerateExamplesCallback,
    _pretty_print_dataframe,
)
from tests.common.ep_stubs import StubEPLayerBase

_SKIP_PP = (
    "[GenerateExamplesCallback] Skipping: a pipeline-parallel stage holds only part of the model "
    "and cannot generate. Generate from a saved checkpoint instead."
)
_SKIP_CP = "[GenerateExamplesCallback] Skipping: Context Parallelism is incompatible with autoregressive generation."
_SKIP_TP = (
    "[GenerateExamplesCallback] Skipping: Tensor Parallelism (DTensor) is incompatible with autoregressive generation."
)


class _Tokenizer:
    def decode(self, token_ids, skip_special_tokens=False):
        return "decoded:" + ",".join(str(t) for t in token_ids.tolist())


class _FakeDTensor:
    def __init__(self, mesh_dim_names):
        self.device_mesh = SimpleNamespace(mesh_dim_names=mesh_dim_names)


class _FakeCPWrapper(torch.nn.Module):
    pass


class _FakePipelineStage(torch.nn.Module):
    pass


class _GCLayer(torch.nn.Module):
    def __init__(self, gradient_checkpointing: bool):
        super().__init__()
        self.gradient_checkpointing = gradient_checkpointing


class _Generator(torch.nn.Module):
    """A ``generate()``-capable model recording, per call, the prompt, greedy flag and GC flags it saw."""

    def __init__(self, *children: torch.nn.Module, fail: bool = False):
        super().__init__()
        self.proj = torch.nn.Linear(2, 2)
        self.blocks = torch.nn.ModuleList(children)
        self.fail = fail
        self.calls: list[tuple[list[int], bool, list[bool]]] = []

    @property
    def device(self):
        return torch.device("cpu")

    def generate(self, input_ids, attention_mask=None, **kwargs):
        gc_flags = [m.gradient_checkpointing for m in self.modules() if isinstance(m, _GCLayer)]
        self.calls.append((input_ids[0].tolist(), kwargs.get("do_sample") is False, gc_flags))
        if self.fail:
            raise RuntimeError("boom")
        return torch.cat([input_ids, torch.tensor([[100]])], dim=1)


class _DTensorGenerator(_Generator):
    """A generator whose parameters are DTensors on a mesh named ``mesh_dim_names``."""

    def __init__(self, *children: torch.nn.Module, mesh_dim_names, fail: bool = False):
        super().__init__(*children, fail=fail)
        self.mesh_dim_names = mesh_dim_names

    def parameters(self, recurse: bool = True):
        return iter([SimpleNamespace(data=_FakeDTensor(self.mesh_dim_names))])


def _fsdp_stamped(model: torch.nn.Module) -> torch.nn.Module:
    model._is_fsdp_managed_module = True
    return model


def _peft_over(inner: torch.nn.Module) -> torch.nn.Module:
    """PEFT's nesting: the wrapped model sits at ``base_model.model``."""
    peft = torch.nn.Module()
    peft.base_model = torch.nn.Module()
    peft.base_model.model = inner
    return peft


def _generation_dataset(num_rows: int = 4) -> Dataset:
    return Dataset.from_list([{"input_ids": [1, 10 + i], "attention_mask": [1, 1]} for i in range(num_rows)])


def _callback(dataset: Dataset, num_examples: int) -> GenerateExamplesCallback:
    return GenerateExamplesCallback(dataset, _Tokenizer(), num_examples=num_examples, logger_backend="none")


def _evaluate(model, *, world_size: int, rank: int = 0, num_examples: int = 2, num_rows: int = 4):
    """Run ``on_evaluate`` as ``rank`` of ``world_size``; return its collectives, warnings and INFO lines."""
    callback = _callback(_generation_dataset(num_rows), num_examples)
    collectives: list[str] = []

    def _all_gather_object(out, obj, group=None):
        collectives.append("all_gather_object")
        out[:] = [obj] * len(out)

    warnings: list[str] = []
    info_lines: list[str] = []
    with (
        patch.object(cb_module, "DTensor", _FakeDTensor),
        patch.object(cb_module, "UlyssesModel", _FakeCPWrapper),
        patch.object(cb_module, "PipelineStageModule", _FakePipelineStage),
        patch.object(cb_module, "get_global_world_size", return_value=world_size),
        patch.object(cb_module, "get_global_rank", return_value=rank),
        patch.object(cb_module, "is_global_main_process", return_value=rank == 0),
        patch.object(cb_module, "barrier", side_effect=lambda *a, **k: collectives.append("barrier")),
        patch.object(torch.distributed, "all_gather_object", side_effect=_all_gather_object),
        patch.object(cb_module.logger, "warning", side_effect=lambda msg, *a, **k: warnings.append(str(msg))),
        patch.object(cb_module.logger, "info", side_effect=lambda msg, *a, **k: info_lines.append(str(msg))),
    ):
        callback.on_evaluate(None, SimpleNamespace(global_step=7), None, model=model)
    return collectives, warnings, info_lines


def _dispatch(build, world_size: int):
    """``("skip", message)`` or ``(path, greedy, failure)``, the last read from a run whose generate raises."""
    model, generator = build(fail=False)
    collectives, warnings, _ = _evaluate(model, world_size=world_size)
    skips = [w for w in warnings if w.startswith("[GenerateExamplesCallback] Skipping:")]
    if skips:
        assert generator is None or generator.calls == [], "a skipped layout still generated"
        assert collectives == [], "a skipped layout still issued a collective"
        return ("skip", *skips)
    path = "all_ranks" if "barrier" in collectives else "split" if "all_gather_object" in collectives else "none"
    assert generator.calls, "a generating layout produced no sample"
    greedy = all(greedy for _, greedy, _ in generator.calls)

    model, generator = build(fail=True)
    try:
        _, warnings, info_lines = _evaluate(model, world_size=world_size)
    except RuntimeError:
        failure = "raises"
    else:
        if any("Generation failed at step 7: boom. Training will continue." in w for w in warnings):
            failure = "warns"
        elif any("<generation failed: boom>" in line for line in info_lines):
            failure = "recorded"
        else:
            failure = "silent"
    return (path, greedy, failure)


def _plain(fail):
    g = _Generator(fail=fail)
    return g, g


def _ddp(fail):
    g = _Generator(fail=fail)
    return torch.nn.DataParallel(g), g


def _fsdp1(fail):
    g = _fsdp_stamped(_Generator(fail=fail))
    return g, g


def _fsdp_named_without_stamp(fail):
    g = type("FSDPGenerator", (_Generator,), {})(fail=fail)
    return g, g


def _fsdp2(mesh_dim_names):
    def build(fail):
        g = _DTensorGenerator(mesh_dim_names=mesh_dim_names, fail=fail)
        return g, g

    return build


def _ep(fail):
    g = _Generator(torch.nn.Sequential(StubEPLayerBase()), fail=fail)
    return g, g


def _ep_compiled(fail):
    g = _Generator(torch.nn.Sequential(StubEPLayerBase()), fail=fail)
    return torch.compile(g), g


def _tp(fail):
    g = _DTensorGenerator(mesh_dim_names=("dp", "tp"), fail=fail)
    return g, g


def _tp_with_ep(fail):
    g = _DTensorGenerator(StubEPLayerBase(), mesh_dim_names=("tp",), fail=fail)
    return g, g


def _cp_under_peft(fail):
    return _peft_over(_FakeCPWrapper()), None


def _cp_compiled(fail):
    return torch.compile(_FakeCPWrapper()), None


def _cp_with_tp(fail):
    return _DTensorGenerator(_FakeCPWrapper(), mesh_dim_names=("tp",), fail=fail), None


def _pp(fail):
    return _FakePipelineStage(), None


def _pp_compiled(fail):
    return torch.compile(_FakePipelineStage()), None


def _pp_holding_cp(fail):
    stage = _FakePipelineStage()
    stage.inner = _FakeCPWrapper()
    return stage, None


# (id, layout builder, world size, expected dispatch)
_DISPATCH_GOLDEN = [
    ("single_process", _plain, 1, ("all_ranks", True, "warns")),
    ("ddp", _ddp, 2, ("split", False, "recorded")),
    ("fsdp1_stamped", _fsdp1, 2, ("all_ranks", True, "raises")),
    ("fsdp1_stamped_single_process", _fsdp1, 1, ("all_ranks", True, "raises")),
    ("fsdp_named_without_stamp", _fsdp_named_without_stamp, 2, ("split", False, "recorded")),
    ("fsdp2_dp_mesh", _fsdp2(("dp",)), 2, ("all_ranks", True, "raises")),
    ("fsdp2_unnamed_mesh", _fsdp2(()), 2, ("all_ranks", True, "raises")),
    ("fsdp2_none_mesh", _fsdp2(None), 2, ("all_ranks", True, "raises")),
    ("ep", _ep, 2, ("all_ranks", True, "raises")),
    ("ep_compiled", _ep_compiled, 2, ("all_ranks", True, "raises")),
    ("tp", _tp, 2, ("skip", _SKIP_TP)),
    ("tp_with_ep", _tp_with_ep, 2, ("skip", _SKIP_TP)),
    ("cp_under_peft", _cp_under_peft, 2, ("skip", _SKIP_CP)),
    ("cp_compiled", _cp_compiled, 2, ("skip", _SKIP_CP)),
    ("cp_with_tp", _cp_with_tp, 2, ("skip", _SKIP_CP)),
    ("pp", _pp, 2, ("skip", _SKIP_PP)),
    ("pp_compiled", _pp_compiled, 2, ("skip", _SKIP_PP)),
    ("pp_holding_cp", _pp_holding_cp, 2, ("skip", _SKIP_PP)),
]


@pytest.mark.parametrize(
    ("build", "world_size", "expected"),
    [case[1:] for case in _DISPATCH_GOLDEN],
    ids=[case[0] for case in _DISPATCH_GOLDEN],
)
def test_dispatch_matches_golden(build, world_size, expected):
    """Skip with the layout's message, generate collectively (greedy, errors raise) or split per rank."""
    assert _dispatch(build, world_size) == expected


def test_skip_warns_on_the_main_rank_only():
    model, _ = _pp(fail=False)
    collectives, warnings, _ = _evaluate(model, world_size=2, rank=1)
    assert (collectives, warnings) == ([], [])


def test_split_ranks_generate_every_world_size_th_sample():
    """Rank 1 of 3 generates sample positions 1 and 4 of the seeded draw (rows 4, 0, 1, 1, 5 of seven)."""
    generator = _Generator()
    collectives, _, _ = _evaluate(generator, world_size=3, rank=1, num_examples=5, num_rows=7)
    assert collectives == ["all_gather_object"]
    assert [prompt for prompt, _, _ in generator.calls] == [[1, 10], [1, 15]]


@pytest.mark.parametrize("flags", list(itertools.product([True, False], repeat=3)))
@pytest.mark.parametrize("fail", [False, True])
def test_gradient_checkpointing_is_off_during_generation_and_restored_exactly(flags, fail):
    """Every module generates with GC off, and afterwards exactly the modules that had it on get it back.

    Checked over every on/off pattern: switching it back on everywhere would change memory and step
    time on layers a partially checkpointed model left off. A collective failure must restore too.
    """
    generator = _fsdp_stamped(_Generator(*(_GCLayer(flag) for flag in flags), fail=fail))
    if fail:
        with pytest.raises(RuntimeError, match="boom"):
            _evaluate(generator, world_size=1, num_examples=1)
    else:
        _evaluate(generator, world_size=1, num_examples=1)
    assert [gc_flags for _, _, gc_flags in generator.calls] == [[False] * 3]
    assert [m.gradient_checkpointing for m in generator.blocks] == list(flags)


def test_sample_draw_is_seeded_and_clamped_to_the_dataset():
    """Every rank draws the same samples, so collective generation and the gather line up."""
    dataset = _generation_dataset(20)
    first, second = _callback(dataset, 5), _callback(dataset, 5)
    assert first.samples["input_ids"] == second.samples["input_ids"]
    assert len(first.samples) == 5
    assert _callback(_generation_dataset(3), 10).num_examples == 3


def test_generate_single_strips_the_prompt_and_keeps_preference_replies():
    """The completion is the decoded suffix past the prompt; chosen/rejected ride along when present."""
    example = {
        "input_ids": [1, 2, 3],
        "attention_mask": [1, 1, 1],
        "chosen": [{"content": "good answer"}],
        "rejected": [{"content": "bad answer"}],
    }
    record = _callback(_generation_dataset(1), 1)._generate_single(_Generator(), example)
    assert record == {
        "Prompt": "decoded:1,2,3",
        "Completion": ",100",
        "Chosen": "good answer",
        "Rejected": "bad answer",
    }


def test_build_record_carries_preference_replies_only_when_present():
    assert GenerateExamplesCallback._build_record("P", "C", {"input_ids": [1]}) == {"Prompt": "P", "Completion": "C"}


def test_pretty_print_dataframe_truncates_long_prompt(caplog):
    """Long prompts are tail-truncated to the module's character budget with a leading ellipsis, so
    the completion stays readable in the log table; the caller's frame is left untouched."""
    long_prompt = "y" * (_MAX_PRINTED_PROMPT_CHARS + 1) + "x" * _MAX_PRINTED_PROMPT_CHARS
    df = pd.DataFrame([{"Prompt": long_prompt, "Completion": "c"}])

    with caplog.at_level(logging.INFO, logger="src.callbacks.generate_examples"):
        _pretty_print_dataframe(df)

    table = caplog.records[-1].getMessage()
    assert "..." + "x" * _MAX_PRINTED_PROMPT_CHARS in table, table
    assert "y" not in table, "the truncated head must not reach the table"
    assert df.loc[0, "Prompt"] == long_prompt  # original untouched (works on a copy)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
