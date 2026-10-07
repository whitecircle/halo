#!/usr/bin/env python
"""Tests for VariableSchedulerCallback. Run: python tests/cpu/callbacks/test_variable_scheduler.py"""

import pytest

from src.callbacks.variable_scheduler import VariableSchedulerCallback

# Mock helpers


class MockModel:
    """Simple mock model to receive scheduled attributes."""

    pass


class MockModuleWrappedModel:
    """Simulates DDP/FSDP wrapped model with .module attribute."""

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


# Tests


def test_linear_schedule():
    cb = VariableSchedulerCallback(
        attribute_name="test_attr",
        initial_value=2.0,
        final_value=10.0,
    )
    cb.total_steps = 100

    # step=0 -> initial
    val = cb._calculate_value(0)
    assert abs(val - 2.0) < 1e-6, f"step=0: expected 2.0, got {val}"

    # step=max -> final
    val = cb._calculate_value(100)
    assert abs(val - 10.0) < 1e-6, f"step=100: expected 10.0, got {val}"

    # midpoint: linear at 0.5 progress -> 2.0 + (10.0-2.0)*0.5 = 6.0
    val = cb._calculate_value(50)
    assert abs(val - 6.0) < 1e-6, f"step=50: expected 6.0, got {val}"

    # past max_steps the value holds at final rather than extrapolating
    val = cb._calculate_value(150)
    assert abs(val - 10.0) < 1e-6, f"step=150: expected 10.0, got {val}"


def test_error_no_training_steps():
    """A run with no steps has nothing to schedule over; it fails at train begin, not on a division."""
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
    """No model in the callback kwargs is a hard error — the attribute has nowhere to land."""
    cb = VariableSchedulerCallback(
        attribute_name="x",
        initial_value=1.0,
        final_value=0.0,
    )
    state = MockTrainerState(global_step=0, max_steps=100)
    with pytest.raises(ValueError, match="(?i)model"):
        cb.on_train_begin(MockArgs(), state, MockControl())  # no model=...


def test_module_wrapped_model_unwrapped():
    """A DDP/FSDP-style wrapper (.module) has the attribute set on the inner model."""
    inner = MockModel()
    wrapped = MockModuleWrappedModel(inner)
    cb = VariableSchedulerCallback(
        attribute_name="my_attr",
        initial_value=1.0,
        final_value=0.0,
    )
    state = MockTrainerState(global_step=0, max_steps=100)
    cb.on_train_begin(MockArgs(), state, MockControl(), model=wrapped)

    # The attribute lands on the inner module, not the wrapper.
    assert hasattr(inner, "my_attr")
    assert not hasattr(wrapped, "my_attr")

    state.global_step = 50
    cb.on_step_begin(MockArgs(), state, MockControl(), model=wrapped)
    assert abs(inner.my_attr - 0.5) < 1e-6


def test_existing_attribute_not_overwritten_on_train_begin():
    """If the model already defines the attribute, on_train_begin leaves it intact."""
    model = MockModel()
    model.my_attr = 0.123  # pre-existing value
    cb = VariableSchedulerCallback(
        attribute_name="my_attr",
        initial_value=1.0,
        final_value=0.0,
    )
    state = MockTrainerState(global_step=0, max_steps=100)
    cb.on_train_begin(MockArgs(), state, MockControl(), model=model)
    # on_train_begin only sets the attr when absent — pre-existing stays.
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


def test_on_step_begin():
    model = MockModel()
    cb = VariableSchedulerCallback(
        attribute_name="my_attr",
        initial_value=1.0,
        final_value=0.0,
    )

    state = MockTrainerState(global_step=0, max_steps=100)
    cb.on_train_begin(MockArgs(), state, MockControl(), model=model)

    # Step to halfway
    state.global_step = 50
    cb.on_step_begin(MockArgs(), state, MockControl(), model=model)

    expected = 0.5  # linear: 1.0 + (0.0 - 1.0) * 0.5 = 0.5
    assert abs(model.my_attr - expected) < 1e-6, f"Expected {expected}, got {model.my_attr}"

    # Step to end
    state.global_step = 100
    cb.on_step_begin(MockArgs(), state, MockControl(), model=model)

    assert abs(model.my_attr - 0.0) < 1e-6, f"Expected 0.0, got {model.my_attr}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
