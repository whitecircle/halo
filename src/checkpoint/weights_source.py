"""Which checkpoint a live model's weights and config schema came from."""

# Stamped by load_distributed_model on the loaded model: where its weights were actually read (None
# under init_from_scratch). One spelling for the writer and the resume-coverage reader.
LOADED_WEIGHTS_FROM_ATTR = "_loaded_weights_from"


def config_schema_source(model) -> str | None:
    """The checkpoint this model's schema came from: a local directory or a Hub repo id, or None.

    ``load_distributed_model`` stamps :data:`LOADED_WEIGHTS_FROM_ATTR`; every other loader leaves
    transformers' ``config._name_or_path``, the same string. An ``init_from_scratch`` model took its
    geometry from that config, so the fallback is the right schema source there too.
    """
    return getattr(model, LOADED_WEIGHTS_FROM_ATTR, None) or getattr(
        getattr(model, "config", None), "_name_or_path", None
    )
