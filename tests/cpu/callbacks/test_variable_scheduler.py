#!/usr/bin/env python

import pytest
import torch

from halo.callbacks.variable_scheduler import VariableSchedulerCallback


class MockModel:
    pass


class MockModuleWrappedModel:
    def __init__(self, inner_model):
        self.module = inner_model


class MockTrainerState:
    def __init__(self, global_step=0, max_steps=100):
        self.global_step = global_step
        self.max_steps = max_steps


class MockArgs:
    pass


class MockControl:
    pass


def test_linear_schedule():
    """Holds at final past max_steps rather than extrapolating."""
    model = MockModel()
    cb = VariableSchedulerCallback(
        attribute_name="test_attr",
        initial_value=2.0,
        final_value=10.0,
    )
    state = MockTrainerState(global_step=0, max_steps=100)
    cb.on_train_begin(MockArgs(), state, MockControl(), model=model)

    scheduled = []
    for step in (0, 50, 100, 150):
        state.global_step = step
        cb.on_step_begin(MockArgs(), state, MockControl(), model=model)
        scheduled.append(model.test_attr)
    assert scheduled == pytest.approx([2.0, 6.0, 10.0, 10.0])


def test_error_no_training_steps():
    """A run with no steps fails at train begin, not on a division."""
    cb = VariableSchedulerCallback(
        attribute_name="test_attr",
        initial_value=1.0,
        final_value=0.0,
    )

    state = MockTrainerState(global_step=0, max_steps=0)
    model = MockModel()

    with pytest.raises(ValueError, match="(?i)positive"):
        cb.on_train_begin(MockArgs(), state, MockControl(), model=model)


def test_missing_model_in_kwargs_raises():
    cb = VariableSchedulerCallback(
        attribute_name="x",
        initial_value=1.0,
        final_value=0.0,
    )
    state = MockTrainerState(global_step=0, max_steps=100)
    with pytest.raises(ValueError, match="(?i)model"):
        cb.on_train_begin(MockArgs(), state, MockControl())


def test_module_wrapped_model_unwrapped():
    inner = MockModel()
    wrapped = MockModuleWrappedModel(inner)
    cb = VariableSchedulerCallback(
        attribute_name="my_attr",
        initial_value=1.0,
        final_value=0.0,
    )
    state = MockTrainerState(global_step=0, max_steps=100)
    cb.on_train_begin(MockArgs(), state, MockControl(), model=wrapped)

    assert hasattr(inner, "my_attr")
    assert not hasattr(wrapped, "my_attr")

    state.global_step = 50
    cb.on_step_begin(MockArgs(), state, MockControl(), model=wrapped)
    assert abs(inner.my_attr - 0.5) < 1e-6


def test_existing_attribute_not_overwritten_on_train_begin():
    model = MockModel()
    model.my_attr = 0.123
    cb = VariableSchedulerCallback(
        attribute_name="my_attr",
        initial_value=1.0,
        final_value=0.0,
    )
    state = MockTrainerState(global_step=0, max_steps=100)
    cb.on_train_begin(MockArgs(), state, MockControl(), model=model)
    assert abs(model.my_attr - 0.123) < 1e-6


def test_model_attribute_is_initialized():
    model = MockModel()
    cb = VariableSchedulerCallback(
        attribute_name="my_attr",
        initial_value=1.0,
        final_value=0.0,
    )

    state = MockTrainerState(global_step=0, max_steps=100)
    cb.on_train_begin(MockArgs(), state, MockControl(), model=model)

    assert hasattr(model, "my_attr"), "Expected attribute 'my_attr' set on model"
    assert abs(model.my_attr - 1.0) < 1e-6, f"Expected initial value 1.0, got {model.my_attr}"


def test_step_leaves_the_module_tree_unchanged():
    """A nested child exposing ``.module`` (an FSDP1 unit) stays in place; a recursive unwrap rewrites the model."""

    class Unit(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.module = torch.nn.Linear(2, 2)

    class Inner(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.block = Unit()

    class Wrapper(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.module = inner

    inner = Inner()
    wrapped = Wrapper(inner)
    tree = [(name, id(module)) for name, module in wrapped.named_modules()]
    cb = VariableSchedulerCallback(attribute_name="my_attr", initial_value=1.0, final_value=0.0)
    state = MockTrainerState(global_step=0, max_steps=10)

    cb.on_train_begin(MockArgs(), state, MockControl(), model=wrapped)
    state.global_step = 5
    cb.on_step_begin(MockArgs(), state, MockControl(), model=wrapped)

    assert [(name, id(module)) for name, module in wrapped.named_modules()] == tree
    assert inner.my_attr == pytest.approx(0.5)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
