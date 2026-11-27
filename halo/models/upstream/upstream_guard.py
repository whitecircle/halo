"""The guard on a toolkit replacement written against one upstream release (transformers, PEFT), so a release that
moved what it replaces fails at load."""

import inspect


def verify_upstream_parameters(function, expected: tuple[str, ...], patch: str) -> None:
    """Raise when ``function`` no longer takes ``expected``, in order; ``patch`` names the replacement written for it."""
    names = tuple(inspect.signature(function).parameters)
    if names != expected:
        library = function.__module__.partition(".")[0]
        raise RuntimeError(
            f"{function.__qualname__} takes {names} in this {library} release, but the toolkit's replacement "
            f"({patch}) is written for {expected}: re-derive it on this release's code."
        )
