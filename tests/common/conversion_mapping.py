"""Scoping a conversion-mapping registration to a test: transformers registers process-wide and takes nothing back."""

from collections.abc import Iterator
from contextlib import contextmanager

from transformers import conversion_mapping
from transformers.conversion_mapping import get_checkpoint_conversion_mapping


@contextmanager
def restored_conversion_mapping() -> Iterator[None]:
    get_checkpoint_conversion_mapping("legacy")
    cache = dict(conversion_mapping._checkpoint_conversion_mapping_cache)
    registered = set(conversion_mapping.USER_REGISTERED_MAPPINGS)
    try:
        yield
    finally:
        conversion_mapping._checkpoint_conversion_mapping_cache.clear()
        conversion_mapping._checkpoint_conversion_mapping_cache.update(cache)
        conversion_mapping.USER_REGISTERED_MAPPINGS.clear()
        conversion_mapping.USER_REGISTERED_MAPPINGS.update(registered)
