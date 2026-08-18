"""Packed batches require right padding, on both the loader and the collator.

``DataCollatorWithPacking`` lays ``position_ids`` out from index 0 forward using ``seq_lengths``,
then drops the dense ``attention_mask`` so the backend derives document boundaries from those
positions. That contract only holds for TRAILING pads. With a left-padding tokenizer the leading
pads receive positions 0..n and become an attended document, while the last real document's
positions collapse to zeros and its labels are fully masked — it disappears from the loss.

The trigger is not hypothetical: GLM-4.7-Flash and both Gemma-4 tokenizers ship
``padding_side: "left"``, and the shipped GLM-4 SFT configs run ``packing: true`` at batch size 2.
Eleven collators force their side per call, and the packing collator must too: inheriting the
tokenizer's side is invisible to any collator test that builds its own right-padded fixture.

    python tests/cpu/data/test_packing_requires_right_padding.py
"""

import inspect
import sys
from unittest.mock import MagicMock, patch

import pytest

from src.data.collators.factory import select_data_collator
from src.data.collators.packing import DataCollatorWithPacking

# Constructor arguments the subclasses add on top of ``tokenizer``. Only values needed to reach the
# guard; a subclass introducing a new REQUIRED argument fails
# ``test_extra_required_args_are_declared`` rather than silently dropping out of the sweep below.
_EXTRA_REQUIRED_ARGS = {"response_prompt_template": "<|assistant|>"}


def _packing_collator_classes() -> list[type]:
    """``DataCollatorWithPacking`` and every subclass, from the class hierarchy — never a hand list."""
    classes = [DataCollatorWithPacking]
    for cls in classes:
        classes.extend(sub for sub in cls.__subclasses__() if sub not in classes)
    return classes


def _required_extra_params(cls: type) -> set[str]:
    return {
        name
        for name, param in inspect.signature(cls.__init__).parameters.items()
        if param.default is inspect.Parameter.empty and name not in ("self", "tokenizer", "args", "kwargs")
    }


class _Tok:
    """Minimal stand-in — only the attributes the collator's guard reads."""

    def __init__(self, padding_side):
        self.padding_side = padding_side
        self.pad_token_id = 0


def test_left_padding_tokenizer_is_rejected():
    with pytest.raises(ValueError, match="right-padding tokenizer"):
        DataCollatorWithPacking(tokenizer=_Tok("left"))


def test_right_padding_tokenizer_is_accepted():
    """Anti-vacuity: the guard must not reject the supported case."""
    DataCollatorWithPacking(tokenizer=_Tok("right"))


def test_error_names_the_consequence_not_just_the_rule():
    """A reader hitting this must learn that a document is silently dropped, not merely that a
    flag is wrong — otherwise the obvious 'fix' is to flip the guard off."""
    with pytest.raises(ValueError) as exc:
        DataCollatorWithPacking(tokenizer=_Tok("left"))
    message = str(exc.value)
    assert "position_ids" in message
    assert "loss" in message


def test_extra_required_args_are_declared():
    """Anti-vacuity for the hierarchy sweep: every subclass must be constructible from
    ``_EXTRA_REQUIRED_ARGS``, so a new packing collator cannot silently skip the guard check."""
    classes = _packing_collator_classes()
    assert len(classes) >= 2, f"the packing hierarchy collapsed to {classes} — the sweep tests nothing"
    undeclared = {name for cls in classes for name in _required_extra_params(cls)} - set(_EXTRA_REQUIRED_ARGS)
    assert not undeclared, f"packing collators require undeclared constructor args {sorted(undeclared)}"


@pytest.mark.parametrize("cls", _packing_collator_classes(), ids=lambda c: c.__name__)
def test_every_packing_collator_rejects_left_padding(cls):
    """The guard belongs to the whole hierarchy, not just the base. The class production actually
    reaches on ``packing: true`` + ``train_on_completions_only: true`` is the SUBCLASS — a subclass
    that stopped delegating to ``super().__init__`` would reopen the silent document drop."""
    kwargs = {name: _EXTRA_REQUIRED_ARGS[name] for name in _required_extra_params(cls)}
    with pytest.raises(ValueError, match="right-padding tokenizer"):
        cls(tokenizer=_Tok("left"), **kwargs)


@pytest.mark.parametrize("train_on_completions_only", [False, True])
def test_factory_propagates_the_guard(train_on_completions_only):
    """The user-facing seam: ``packing: true`` in a YAML with a left-padding tokenizer must fail at
    collator selection, both with and without completion-only masking."""
    tok = MagicMock()
    tok.padding_side = "left"
    tok.pad_token_id = 0
    tok.eos_token_id = 2
    tok.encode.return_value = [10, 11]
    with patch("src.data.collators.factory.PartialState") as mock_state:
        mock_state.return_value.is_main_process = False
        with pytest.raises(ValueError, match="right-padding tokenizer"):
            select_data_collator(
                tokenizer=tok,
                packing=True,
                train_on_completions_only=train_on_completions_only,
                assistant_message_template="<|assistant|>" if train_on_completions_only else None,
            )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
