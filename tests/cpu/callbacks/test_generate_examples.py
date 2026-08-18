#!/usr/bin/env python
"""Tests for GenerateExamplesCallback. Run: python tests/cpu/callbacks/test_generate_examples.py"""

import sys

import pytest
import torch

# Mock helpers


class MockModule:
    """A mock nn.Module-like object."""

    def __init__(self, has_ep_config=False, has_dispatcher=False):
        if has_ep_config:
            self.ep_config = {}
        if has_dispatcher:
            self.dispatcher = object()

    def modules(self):
        return iter([self])

    def parameters(self):
        return iter([torch.randn(10)])


class MockTokenizer:
    """Mock tokenizer with encode/decode."""

    def __init__(self):
        self._vocab = {}

    def decode(self, token_ids, skip_special_tokens=False):
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        return "decoded:" + ",".join(str(t) for t in token_ids)

    def encode(self, text, return_tensors=None):
        return [1, 2, 3]


class MockGenerateModel:
    """Mock model with generate() method."""

    def __init__(self):
        self.device = torch.device("cpu")
        self._generate_output = None

    def generate(self, input_ids, attention_mask=None, max_new_tokens=256):
        # Return input_ids + some generated tokens
        new_tokens = torch.tensor([[100, 101, 102]])
        return torch.cat([input_ids, new_tokens], dim=1)

    def eval(self):
        pass

    def modules(self):
        return iter([self])

    def parameters(self):
        return iter([torch.randn(10)])


class MockDataset:
    """Mock HuggingFace Dataset for testing."""

    def __init__(self, data):
        self._data = data

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        if isinstance(idx, int):
            return self._data[idx]
        return self._data[idx]

    def select(self, indices):
        selected_data = [self._data[i] for i in indices]
        return MockDataset(selected_data)

    def __iter__(self):
        return iter(self._data)


# Tests


def test_is_distributed_parallel_model():
    """Test _is_distributed_parallel_model with mock objects."""
    from src.callbacks.generate_examples import _is_distributed_parallel_model

    # Model without EP/TP
    plain_model = MockModule(has_ep_config=False)
    assert _is_distributed_parallel_model(plain_model) is False

    # Model with ep_config attribute (EP mode)
    ep_model = MockModule(has_ep_config=True)
    assert _is_distributed_parallel_model(ep_model) is True

    # Model with dispatcher attribute (EP dispatcher)
    disp_model = MockModule(has_dispatcher=True)
    assert _is_distributed_parallel_model(disp_model) is True


def test_callback_init():
    """Test GenerateExamplesCallback deterministic sample selection with seed=42."""
    from src.callbacks.generate_examples import GenerateExamplesCallback

    # Create a dataset with 20 examples
    data = []
    for i in range(20):
        data.append(
            {
                "input_ids": [1, 2, 3, i],
                "attention_mask": [1, 1, 1, 1],
            }
        )
    dataset = MockDataset(data)
    tokenizer = MockTokenizer()

    cb1 = GenerateExamplesCallback(
        preprocessed_dataset=dataset,
        tokenizer=tokenizer,
        num_examples=5,
        max_new_tokens=128,
    )

    cb2 = GenerateExamplesCallback(
        preprocessed_dataset=dataset,
        tokenizer=tokenizer,
        num_examples=5,
        max_new_tokens=128,
    )

    # Both callbacks should select the same samples (same seed=42)
    assert len(cb1.samples) == 5, f"Expected 5 samples, got {len(cb1.samples)}"
    assert len(cb2.samples) == 5, f"Expected 5 samples, got {len(cb2.samples)}"

    for i in range(5):
        s1 = cb1.samples[i]
        s2 = cb2.samples[i]
        assert s1["input_ids"] == s2["input_ids"], f"Sample {i} differs: {s1['input_ids']} vs {s2['input_ids']}"

    # num_examples should be clamped to dataset length
    cb_small = GenerateExamplesCallback(
        preprocessed_dataset=MockDataset(data[:3]),
        tokenizer=tokenizer,
        num_examples=10,  # more than dataset size (3)
    )
    assert cb_small.num_examples == 3, f"Expected clamped to 3, got {cb_small.num_examples}"


def test_has_tp_dtensor_params():
    """Test _has_tp_dtensor_params distinguishes TP DTensors from FSDP2 DTensors."""
    from unittest.mock import patch

    import src.callbacks.generate_examples as cb_module

    # Plain model (no DTensors) — both should return False
    plain_model = MockModule()
    assert cb_module._has_tp_dtensor_params(plain_model) is False
    assert cb_module._has_any_dtensor_params(plain_model) is False

    # Use a base class that our mocks can be instances of
    class FakeDTensorBase:
        pass

    class MockMesh:
        def __init__(self, dim_names):
            self.mesh_dim_names = dim_names

    class FakeDTensor(FakeDTensorBase):
        def __init__(self, mesh):
            self.device_mesh = mesh

    class MockModelWithDTensors:
        def __init__(self, mesh_names):
            mesh = MockMesh(mesh_names)
            self._params = [type("P", (), {"data": FakeDTensor(mesh)})()]

        def parameters(self):
            return iter(self._params)

        def modules(self):
            return iter([self])

    # Patch the DTensor reference in the callback module so isinstance checks work
    with patch.object(cb_module, "DTensor", FakeDTensorBase):
        # TP DTensor (mesh with 'tp' dimension)
        tp_model = MockModelWithDTensors(("dp", "tp"))
        assert cb_module._has_tp_dtensor_params(tp_model) is True
        assert cb_module._has_any_dtensor_params(tp_model) is True

        # FSDP2 DTensor (mesh without 'tp' dimension)
        fsdp2_model = MockModelWithDTensors(("dp",))
        assert cb_module._has_tp_dtensor_params(fsdp2_model) is False
        assert cb_module._has_any_dtensor_params(fsdp2_model) is True

        # FSDP2 with no named dimensions
        fsdp2_unnamed = MockModelWithDTensors(())
        assert cb_module._has_tp_dtensor_params(fsdp2_unnamed) is False
        assert cb_module._has_any_dtensor_params(fsdp2_unnamed) is True

        # FSDP2 with None dim names
        fsdp2_none = MockModelWithDTensors(None)
        assert cb_module._has_tp_dtensor_params(fsdp2_none) is False
        assert cb_module._has_any_dtensor_params(fsdp2_none) is True


def test_fsdp2_routes_to_all_ranks():
    """Test that FSDP2 DTensor models route to all-ranks generation (not split)."""
    from unittest.mock import patch

    import src.callbacks.generate_examples as cb_module

    class FakeDTensorBase:
        pass

    class MockMesh:
        def __init__(self, dim_names):
            self.mesh_dim_names = dim_names

    class FakeDTensor(FakeDTensorBase):
        def __init__(self, mesh):
            self.device_mesh = mesh

    class MockFSDP2Model:
        def __init__(self):
            mesh = MockMesh(("dp",))
            self._params = [type("P", (), {"data": FakeDTensor(mesh)})()]

        def parameters(self):
            return iter(self._params)

        def modules(self):
            return iter([self])

    # Patch DTensor so isinstance checks work with our mocks
    with patch.object(cb_module, "DTensor", FakeDTensorBase):
        fsdp2_model = MockFSDP2Model()
        assert cb_module._is_distributed_parallel_model(fsdp2_model) is True, (
            "FSDP2 DTensor model should be detected as distributed parallel (requires all ranks in forward)"
        )


def test_generate_single_with_chosen_rejected():
    """Test _generate_single includes chosen/rejected when present."""
    from src.callbacks.generate_examples import GenerateExamplesCallback

    data = [
        {
            "input_ids": [1, 2, 3],
            "attention_mask": [1, 1, 1],
            "chosen": [{"content": "good answer"}],
            "rejected": [{"content": "bad answer"}],
        }
    ]
    dataset = MockDataset(data)
    tokenizer = MockTokenizer()

    cb = GenerateExamplesCallback(
        preprocessed_dataset=dataset,
        tokenizer=tokenizer,
        num_examples=1,
    )

    model = MockGenerateModel()

    # MockDataset items need a .keys() method for the "chosen"/"rejected" check
    class DictLikeExample(dict):
        pass

    example = DictLikeExample(data[0])
    record = cb._generate_single(model, example)

    assert "Chosen" in record, "Record should have 'Chosen' key"
    assert record["Chosen"] == "good answer", f"Expected 'good answer', got {record['Chosen']!r}"
    assert "Rejected" in record, "Record should have 'Rejected' key"
    assert record["Rejected"] == "bad answer", f"Expected 'bad answer', got {record['Rejected']!r}"


def test_is_cp_wrapped_sees_a_wrapper_nested_under_peft():
    """PEFT keeps the CP wrapper at ``base_model.model`` and delegates ``generate`` down to it, so a
    root-only isinstance misses it and the callback would generate on sequence-sharded ranks."""
    from unittest.mock import patch

    import src.callbacks.generate_examples as cb_module

    class FakeCPWrapper(torch.nn.Module):
        pass

    class FakePeftModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.base_model = torch.nn.Module()
            self.base_model.model = FakeCPWrapper()

    with patch.object(cb_module, "UlyssesCPModelWrapper", FakeCPWrapper):
        assert cb_module._is_cp_wrapped(FakePeftModel()) is True
        assert cb_module._is_cp_wrapped(FakeCPWrapper()) is True
        assert cb_module._is_cp_wrapped(torch.nn.Linear(2, 2)) is False


def test_is_fsdp2_model():
    """FSDP2 is detected by the ``_is_fsdp_managed_module`` flag ``fully_shard`` stamps on every
    module it manages — not by the ``FSDP<Name>`` class rename, which covers only the roots and
    would misread any unrelated class whose name happens to start with FSDP."""
    import src.callbacks.generate_examples as cb_module

    class Plain:
        pass

    assert cb_module._is_fsdp2_model(Plain()) is False

    class Flagged:
        _is_fsdp_managed_module = True

    assert cb_module._is_fsdp2_model(Flagged()) is True

    assert cb_module._is_fsdp2_model(type("FSDPLookalike", (), {})()) is False


def test_gradient_checkpointing_disable_restore_is_an_exact_inverse():
    """Disable-then-restore must leave every module's GC flag exactly as it was found.

    Generation is a side trip inside training: a restore that switches GC on for layers that had it
    off silently changes memory and step time for the remainder of the run. Checked over every
    on/off pattern rather than the uniform all-on case, which cannot distinguish the two.
    """
    import itertools

    import src.callbacks.generate_examples as cb_module

    class Layer:
        def __init__(self, gc):
            self.gradient_checkpointing = gc

    class Model:
        def __init__(self, flags):
            self._mods = [Layer(flag) for flag in flags]

        def modules(self):
            return iter(self._mods)

    for flags in itertools.product([True, False], repeat=3):
        m = Model(flags)
        touched = cb_module._disable_gradient_checkpointing(m)
        assert [mod.gradient_checkpointing for mod in m._mods] == [False] * 3, flags
        assert len(touched) == sum(flags), f"{flags}: reported {len(touched)} enabled modules"

        cb_module._restore_gradient_checkpointing(touched)
        assert [mod.gradient_checkpointing for mod in m._mods] == list(flags), (
            f"restore is not the inverse of disable for pattern {flags}"
        )


def test_gradient_checkpointing_restored_when_generation_raises():
    """The finally-block contract: a failed generation must not leave GC off for the rest of training."""
    from src.callbacks.generate_examples import GenerateExamplesCallback

    class RaisingModel(MockGenerateModel):
        def __init__(self):
            super().__init__()
            self.gradient_checkpointing = True

        def generate(self, input_ids, **kwargs):
            raise RuntimeError("boom")

    dataset = MockDataset([{"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]}])
    cb = GenerateExamplesCallback(preprocessed_dataset=dataset, tokenizer=MockTokenizer(), num_examples=1)
    model = RaisingModel()

    with pytest.raises(RuntimeError, match="boom"):
        cb._generate_standard(model, type("S", (), {"global_step": 1})())
    assert model.gradient_checkpointing is True, "generation failure left gradient checkpointing off"


def test_build_record_plain_and_preference():
    """_build_record carries chosen/rejected content only when those keys exist."""
    from src.callbacks.generate_examples import GenerateExamplesCallback

    plain = GenerateExamplesCallback._build_record("P", "C", {"input_ids": [1]})
    assert plain == {"Prompt": "P", "Completion": "C"}

    pref = GenerateExamplesCallback._build_record(
        "P",
        "C",
        {"chosen": [{"content": "good"}], "rejected": [{"content": "bad"}]},
    )
    assert pref["Chosen"] == "good"
    assert pref["Rejected"] == "bad"


def test_generate_single_strips_prompt_prefix():
    """The completion is the generated suffix with the prompt prefix removed.

    The mock tokenizer decodes ``decoded:<ids>``; the prompt decode is a strict
    prefix of the full-output decode, so the completion is exactly the new ids.
    """
    from src.callbacks.generate_examples import GenerateExamplesCallback

    dataset = MockDataset([{"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]}])
    cb = GenerateExamplesCallback(preprocessed_dataset=dataset, tokenizer=MockTokenizer(), num_examples=1)
    record = cb._generate_single(MockGenerateModel(), dataset[0])

    # MockGenerateModel appends tokens [100, 101, 102]; the prompt prefix is
    # "decoded:1,2,3" and the full output is "decoded:1,2,3,100,101,102",
    # so the completion is the trailing ",100,101,102".
    assert record["Prompt"] == "decoded:1,2,3"
    assert record["Completion"] == ",100,101,102"


def test_pretty_print_dataframe_truncates_long_prompt(caplog):
    """Long prompts are tail-truncated to the module's character budget with a leading ellipsis, so
    the completion stays readable in the log table; the caller's frame is left untouched."""
    import logging

    import pandas as pd

    from src.callbacks.generate_examples import _MAX_PRINTED_PROMPT_CHARS, _pretty_print_dataframe

    long_prompt = "y" * (_MAX_PRINTED_PROMPT_CHARS + 1) + "x" * _MAX_PRINTED_PROMPT_CHARS
    df = pd.DataFrame([{"Prompt": long_prompt, "Completion": "c"}])

    with caplog.at_level(logging.INFO, logger="src.callbacks.generate_examples"):
        _pretty_print_dataframe(df)

    table = caplog.records[-1].getMessage()
    assert "..." + "x" * _MAX_PRINTED_PROMPT_CHARS in table, table
    assert "y" not in table, "the truncated head must not reach the table"
    assert df.loc[0, "Prompt"] == long_prompt  # original untouched (works on a copy)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
