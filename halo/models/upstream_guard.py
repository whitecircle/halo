"""The guard on a toolkit forward written against one transformers release, so a release that moved it fails at load."""

import inspect


def verify_upstream_parameters(function, expected: tuple[str, ...], patch: str) -> None:
    """Raise when ``function`` no longer takes ``expected``, in order; ``patch`` names the replacement written for it."""
    names = tuple(inspect.signature(function).parameters)
    if names != expected:
        raise RuntimeError(
            f"{function.__qualname__} takes {names} in this transformers release, but the toolkit's replacement "
            f"({patch}) is written for {expected}: re-derive it on this release's forward."
        )
