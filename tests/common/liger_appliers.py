"""The Liger applier a model type resolves to, looked up the way the toolkit's dispatch resolves one."""

from halo.model_loading.liger.dispatch import _resolve_applier


def liger_applier(model_type: str):
    """The Liger applier claiming ``model_type``, toolkit first, or ``None``."""
    return _resolve_applier(model_type, None)[1]
