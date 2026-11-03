"""What the loader's two string-to-object resolvers may return and must not swallow. A ``None`` tokenizer skips the
save, shipping a resized model beside a stale tokenizer; every advertised ``--dtype`` spelling must resolve."""

from __future__ import annotations

import pytest
import torch

from halo.model_loading import dtype as dtype_mod
from halo.model_loading import tokenizer_setup
from halo.model_loading.dtype import DTYPE_BY_NAME


class _UnexpectedTokenizerError(RuntimeError):
    """A broken remote-code import, unlike a directory with no tokenizer."""


class _Loader:
    """Raises ``error``, or returns ``result`` when there is none."""

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
    """The raw-weights-directory case warns."""
    _stub_loaders(monkeypatch, processor=_Loader(OSError("no processor")), tokenizer=_Loader(absent))

    with caplog.at_level("WARNING"):
        assert tokenizer_setup.load_processing_class("/weights/only") is None

    warnings = [rec.getMessage() for rec in caplog.records if rec.levelname == "WARNING"]
    assert any("/weights/only" in msg and type(absent).__name__ in msg for msg in warnings), (
        f"the None is unexplained — nothing names the path or the cause: {warnings}"
    )


def test_unexpected_tokenizer_error_propagates(monkeypatch):
    """``None`` makes ``save_full_checkpoint`` skip the tokenizer save, shipping a model beside a stale one."""
    _stub_loaders(
        monkeypatch,
        processor=_Loader(OSError("no processor")),
        tokenizer=_Loader(_UnexpectedTokenizerError("remote code import blew up")),
    )

    with pytest.raises(_UnexpectedTokenizerError):
        tokenizer_setup.load_processing_class("/model/with/broken/tokenizer")


def test_tokenizer_fallback_still_returns_the_tokenizer(monkeypatch):
    sentinel = object()
    _stub_loaders(monkeypatch, processor=_Loader(OSError("not multimodal")), tokenizer=_Loader(result=sentinel))

    assert tokenizer_setup.load_processing_class("/text/model") is sentinel


@pytest.mark.parametrize(("name", "expected"), sorted(DTYPE_BY_NAME.items()))
def test_every_advertised_dtype_spelling_resolves(name, expected):
    """Each key is a value the CLIs offer."""
    kwargs = {"dtype": name}

    dtype_mod.normalize_model_init_dtype(kwargs)

    assert kwargs["dtype"] is expected


def test_torch_attribute_names_outside_the_map_still_resolve():
    kwargs = {"dtype": "float64"}

    dtype_mod.normalize_model_init_dtype(kwargs)

    assert kwargs["dtype"] is torch.float64


def test_legacy_torch_dtype_spelling_is_folded_into_dtype():
    """It must not reach ``from_pretrained`` under the deprecated alias."""
    kwargs = {"torch_dtype": "bf16"}

    dtype_mod.normalize_model_init_dtype(kwargs)

    assert kwargs == {"dtype": torch.bfloat16}


def test_explicit_dtype_wins_over_the_legacy_spelling():
    """Mirrors transformers' own precedence."""
    kwargs = {"dtype": "float32", "torch_dtype": "bfloat16"}

    dtype_mod.normalize_model_init_dtype(kwargs)

    assert kwargs == {"dtype": torch.float32}


def test_pass_through_values_are_untouched():
    for value in ("auto", None, torch.bfloat16):
        kwargs = {"dtype": value}
        assert dtype_mod.normalize_model_init_dtype(kwargs)["dtype"] is value


def test_invalid_dtype_name_still_raises():
    with pytest.raises(ValueError, match="Invalid dtype"):
        dtype_mod.normalize_model_init_dtype({"dtype": "bfloat8"})


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
