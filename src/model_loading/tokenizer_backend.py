"""Pluggable text→ids backend for dataset tokenization.

"hf" (default) encodes with the HuggingFace tokenizer itself; "gigatoken" (optional extra
`halo[gigatoken]`) wraps it in a proxy that encodes through gigatoken's Rust encoder, after
verifying on a probe set that both produce identical ids. Resolved centrally in
``setup_model_and_tokenizer``, so every training method honors ``tokenizer_backend``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Literal, get_args

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer

try:
    import gigatoken
except ImportError:
    gigatoken = None

logger = logging.getLogger(__name__)

TokenizerBackend = Literal["hf", "gigatoken"]
TOKENIZER_BACKENDS: tuple[str, ...] = get_args(TokenizerBackend)

_PROBE_TEXTS = (
    "The quick brown fox jumps over the lazy dog. 12345",
    "Ünïcodé — em—dashes… “quotes” and emoji 🚀🔥",
    "def f(x):\n\treturn x * 2  # comment\n\n\n   trailing   spaces   ",
)
# Cap the special-token join probe (a tokenizer can declare hundreds of specials).
_MAX_SPECIAL_TOKEN_PROBES = 8
# Shorter than every probe's token count, so the truncated comparison actually truncates. Production
# paths encode with truncation=True (e.g. tokenize_rendered), so parity must hold there too.
_TRUNCATION_PROBE_LENGTH = 4


def _rebuild_proxy(tokenizer: Any) -> GigatokenTokenizerProxy:
    """Unpickle hook: recreate the proxy with the Rust backend dropped (rebuilt lazily)."""
    proxy = GigatokenTokenizerProxy.__new__(GigatokenTokenizerProxy)
    object.__setattr__(proxy, "_wrapped", tokenizer)
    object.__setattr__(proxy, "_compat", None)
    return proxy


class GigatokenTokenizerProxy:
    """Transparent stand-in for an HF tokenizer that encodes via gigatoken.

    Only ``__call__`` reaches the Rust backend; attribute reads and writes pass through, and
    ``__class__`` reports the wrapped type so trainer/collator ``isinstance`` checks behave (``type()``
    still reveals the proxy). The backend is dropped on pickling and rebuilt lazily in map workers.
    """

    def __init__(self, tokenizer: PreTrainedTokenizer) -> None:
        if gigatoken is None:
            raise ImportError(
                "tokenizer_backend='gigatoken' requires the gigatoken package. "
                "Install the extra: uv pip install 'halo[gigatoken]'."
            )
        object.__setattr__(self, "_wrapped", tokenizer)
        object.__setattr__(self, "_compat", gigatoken.Tokenizer(tokenizer).as_hf())

    def __call__(self, text: Any = None, **kwargs: Any) -> Any:
        if self._compat is None:
            object.__setattr__(self, "_compat", gigatoken.Tokenizer(self._wrapped).as_hf())
        return self._compat(text, **kwargs)

    @property  # type: ignore[misc]
    def __class__(self) -> type:
        return type(self._wrapped)

    def __getattr__(self, name: str) -> Any:
        if name in ("_wrapped", "_compat"):
            raise AttributeError(name)
        return getattr(self._wrapped, name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name in ("_wrapped", "_compat"):
            object.__setattr__(self, name, value)
        else:
            setattr(self._wrapped, name, value)

    def __len__(self) -> int:
        return len(self._wrapped)

    def __reduce__(self):
        return (_rebuild_proxy, (self._wrapped,))


def _equivalence_probes(tokenizer: PreTrainedTokenizer) -> list[str]:
    probes = list(_PROBE_TEXTS)
    special_tokens = [tok for tok in tokenizer.all_special_tokens if isinstance(tok, str)]
    if special_tokens:
        probes.append(" text between ".join(special_tokens[:_MAX_SPECIAL_TOKEN_PROBES]))
    if getattr(tokenizer, "chat_template", None):
        try:
            probes.append(
                tokenizer.apply_chat_template(
                    [
                        {"role": "user", "content": "backend probe?"},
                        {"role": "assistant", "content": "backend probe."},
                    ],
                    tokenize=False,
                )
            )
        except Exception as exc:  # a template needing extra kwargs is not a backend defect
            logger.debug("Skipping chat-template equivalence probe (render failed: %s)", exc)
    return probes


def _verify_backend_equivalence(tokenizer: PreTrainedTokenizer, proxy: GigatokenTokenizerProxy) -> None:
    """Raise if the proxy and the HF tokenizer disagree on any probe's token IDs,
    untruncated or truncated."""
    for text in _equivalence_probes(tokenizer):
        for add_special_tokens in (False, True):
            for extra in ({}, {"truncation": True, "max_length": _TRUNCATION_PROBE_LENGTH}):
                expected = tokenizer(text, add_special_tokens=add_special_tokens, **extra)["input_ids"]
                actual = list(proxy(text, add_special_tokens=add_special_tokens, **extra)["input_ids"])
                if expected != actual:
                    raise ValueError(
                        f"gigatoken backend diverges from the HF tokenizer "
                        f"({tokenizer.name_or_path}) on probe {text[:60]!r} "
                        f"(add_special_tokens={add_special_tokens}, {extra or 'untruncated'}): "
                        f"{expected[:10]}... != {actual[:10]}... "
                        f"This tokenizer is not gigatoken-compatible; use tokenizer_backend='hf'."
                    )


def resolve_tokenizer_backend(tokenizer: PreTrainedTokenizer, backend: str) -> PreTrainedTokenizer:
    """Return the tokenizer to encode with under ``backend`` ("hf" passes through)."""
    if backend == "hf":
        return tokenizer
    if backend != "gigatoken":
        raise ValueError(f"Unknown tokenizer_backend {backend!r}; expected one of {TOKENIZER_BACKENDS}.")
    # type(), not isinstance(): the proxy spoofs __class__ to the wrapped type.
    if type(tokenizer) is GigatokenTokenizerProxy:
        return tokenizer
    proxy = GigatokenTokenizerProxy(tokenizer)
    _verify_backend_equivalence(tokenizer, proxy)
    logger.info("Tokenizing with the gigatoken backend (IDs verified against %s).", tokenizer.name_or_path)
    return proxy


def resolve_processor_backend(processor: Any, backend: str) -> Any:
    """Install the ``backend`` tokenizer on a VLM processor ("hf" passes through).

    VLM processors tokenize internally, so the proxy is set as ``processor.tokenizer``;
    image processing is untouched.
    """
    if backend == "hf":
        return processor
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        raise ValueError(f"tokenizer_backend={backend!r} requires a processor with a .tokenizer attribute.")
    processor.tokenizer = resolve_tokenizer_backend(tokenizer, backend)
    return processor
