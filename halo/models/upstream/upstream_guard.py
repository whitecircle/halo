"""The guard on a toolkit forward written against one transformers release, so a release that moved it fails at load."""

import functools
import inspect

_LEGACY_CACHE_NAMES = ("past_key_values", "cache_params")


def takes_legacy_cache_names(forward):
    """``forward`` with ``past_key_values=`` and ``cache_params=`` taken as ``cache=``, the names transformers 5.24 keeps
    on the forwards it ships. A replacement naming ``cache`` would otherwise take a legacy cache into ``**kwargs`` and
    run, or refuse, as if it had none. One wrapper, so a test reaching past the replacement unwraps it once."""

    @functools.wraps(forward)
    def with_legacy_cache_names(*args, **kwargs):
        for name in _LEGACY_CACHE_NAMES:
            if name in kwargs:
                if "cache" in kwargs:
                    raise ValueError(f"You cannot specify both `cache` and `{name}`. Use `cache` only.")
                kwargs["cache"] = kwargs.pop(name)
        return forward(*args, **kwargs)

    return with_legacy_cache_names


def verify_upstream_parameters(function, expected: tuple[str, ...], patch: str) -> None:
    """Raise when ``function`` no longer takes ``expected``, in order; ``patch`` names the replacement written for it."""
    names = tuple(inspect.signature(function).parameters)
    if names != expected:
        raise RuntimeError(
            f"{function.__qualname__} takes {names} in this transformers release, but the toolkit's replacement "
            f"({patch}) is written for {expected}: re-derive it on this release's forward."
        )
