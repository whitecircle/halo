"""What the loader's two string-to-object resolvers may return, and what they must not swallow.

Both decide something a run then EXPORTS, and both fail quietly when wrong:

  * ``load_processing_class`` returning ``None`` for EVERY tokenizer-load failure, unlogged, reads to
    callers as "nothing to save" — ``save_full_checkpoint(processing_class=None)`` skips the
    tokenizer write — so a tokenizer that exists but fails to import (a remote-code class, a missing
    backend package) lets a tool ship a resized model beside the directory's stale tokenizer and
    report success. Only "there is no tokenizer here" may yield ``None``.
  * ``resolve_model_dtype`` must not resolve names through ``getattr(torch, ...)`` alone: the
    short spellings ``DTYPE_BY_NAME`` defines — the ones every ``--dtype`` CLI advertises through
    ``choices=list(DTYPE_BY_NAME)`` — then raise when the same string arrives via
    ``model_init_kwargs``. It must also keep accepting the pre-v5 ``torch_dtype`` spelling a user
    config may still carry, while handing transformers only its own ``dtype`` key.

    python tests/cpu/models/test_model_preparation_resolvers.py
"""

from __future__ import annotations

import pytest
import torch
from accelerate import PartialState

from src.models.loading import dtype as dtype_mod
from src.models.loading import tokenizer_setup
from src.models.loading.dtype import DTYPE_BY_NAME

PartialState()  # the loader logs its warning through accelerate's logger


class _UnexpectedTokenizerError(RuntimeError):
    """An error that does NOT mean "this directory has no tokenizer" — e.g. a broken remote-code import."""


class _Loader:
    """Stand-in for an ``Auto*`` class: raises ``error``, or returns ``result`` when there is none."""

    def __init__(self, error: Exception | None = None, result: object = None):
        self.error = error
        self.result = result

    def from_pretrained(self, *args, **kwargs):
        if self.error is not None:
            raise self.error
        return self.result


def _stub_loaders(monkeypatch, *, processor: _Loader, tokenizer: _Loader) -> None:
    monkeypatch.setattr(tokenizer_setup, "AutoProcessor", processor)
    monkeypatch.setattr(tokenizer_setup, "AutoTokenizer", tokenizer)


@pytest.mark.parametrize("absent", [OSError("no such file"), ValueError("Unrecognized configuration class")])
def test_absent_tokenizer_yields_none_and_says_so(monkeypatch, caplog, absent):
    """The documented "raw weights dir" case returns ``None`` — with a warning, not in silence."""
    _stub_loaders(monkeypatch, processor=_Loader(OSError("no processor")), tokenizer=_Loader(absent))

    with caplog.at_level("WARNING"):
        assert tokenizer_setup.load_processing_class("/weights/only") is None

    warnings = [rec.getMessage() for rec in caplog.records if rec.levelname == "WARNING"]
    assert any("/weights/only" in msg and type(absent).__name__ in msg for msg in warnings), (
        f"the None is unexplained — nothing names the path or the cause: {warnings}"
    )


def test_unexpected_tokenizer_error_propagates(monkeypatch):
    """A tokenizer that EXISTS but fails to load must not degrade to ``None``.

    ``None`` is indistinguishable from "no tokenizer here" downstream, and it makes
    ``save_full_checkpoint`` skip the tokenizer save — shipping a model beside a stale one.
    """
    _stub_loaders(
        monkeypatch,
        processor=_Loader(OSError("no processor")),
        tokenizer=_Loader(_UnexpectedTokenizerError("remote code import blew up")),
    )

    with pytest.raises(_UnexpectedTokenizerError):
        tokenizer_setup.load_processing_class("/model/with/broken/tokenizer")


def test_tokenizer_fallback_still_returns_the_tokenizer(monkeypatch):
    """Narrowing must not cost the text-model path: no processor, a loadable tokenizer, tokenizer wins."""
    sentinel = object()
    _stub_loaders(monkeypatch, processor=_Loader(OSError("not multimodal")), tokenizer=_Loader(result=sentinel))

    assert tokenizer_setup.load_processing_class("/text/model") is sentinel


@pytest.mark.parametrize(("name", "expected"), sorted(DTYPE_BY_NAME.items()))
def test_every_advertised_dtype_spelling_resolves(name, expected):
    """Each ``DTYPE_BY_NAME`` key is a value the CLIs offer; all must resolve from model_init_kwargs too."""
    kwargs = {"dtype": name}

    dtype_mod.resolve_model_dtype(kwargs)

    assert kwargs["dtype"] is expected


def test_torch_attribute_names_outside_the_map_still_resolve():
    """The getattr fallback survives: a valid torch.dtype name the map does not list still works."""
    kwargs = {"dtype": "float64"}

    dtype_mod.resolve_model_dtype(kwargs)

    assert kwargs["dtype"] is torch.float64


def test_legacy_torch_dtype_spelling_is_folded_into_dtype():
    """A user config written against transformers 4 must keep working, and must not reach
    ``from_pretrained`` under the deprecated alias."""
    kwargs = {"torch_dtype": "bf16"}

    dtype_mod.resolve_model_dtype(kwargs)

    assert kwargs == {"dtype": torch.bfloat16}


def test_explicit_dtype_wins_over_the_legacy_spelling():
    """Mirrors transformers' own precedence when a config carries both."""
    kwargs = {"dtype": "float32", "torch_dtype": "bfloat16"}

    dtype_mod.resolve_model_dtype(kwargs)

    assert kwargs == {"dtype": torch.float32}


def test_pass_through_values_are_untouched():
    """ "auto", ``None`` and real dtypes are not names to resolve."""
    for value in ("auto", None, torch.bfloat16):
        kwargs = {"dtype": value}
        assert dtype_mod.resolve_model_dtype(kwargs)["dtype"] is value


def test_invalid_dtype_name_still_raises():
    """A typo must not silently become a dtype — the widened lookup keeps the refusal."""
    with pytest.raises(ValueError, match="Invalid dtype"):
        dtype_mod.resolve_model_dtype({"dtype": "bfloat8"})


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
