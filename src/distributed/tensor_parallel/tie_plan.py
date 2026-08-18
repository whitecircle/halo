"""Keeping a tied embedding/head pair consistent across an HF-native TP load.

transformers shards module by module and ties afterwards, so the two ends of a tied pair must land
on the same side of the TP plan or the first forward meets mixed plain/DTensor operands. The load-time
context manager below arranges that; the post-load check fails loud when it did not hold.
"""

from __future__ import annotations

from contextlib import contextmanager

from torch.distributed.tensor import DTensor

from src.distributed.tensor_parallel.state_dict import tp_plan_shards_params
from src.models.loading.config_levels import text_config


def config_ties_word_embeddings(model_config) -> bool:
    """Whether this checkpoint declares a tied embedding/head pair.

    Read through the shared composite-config accessor: the flag lives on the text sub-config of a
    multimodal wrapper, and a config class defining no ``get_text_config`` (remote code) resolves to
    the config itself instead of raising.
    """
    return bool(getattr(text_config(model_config), "tie_word_embeddings", False))


def _concrete_model_class(model_class, model_config):
    """The class an Auto class dispatches ``model_config`` to — where ``_tp_plan`` and the tie
    declaration live. The Auto class itself for a remote-code config outside the mapping."""
    mapping = getattr(model_class, "_model_mapping", None)
    if mapping is None:
        return model_class
    try:
        return mapping[type(model_config)]
    except KeyError:
        return model_class


def _tied_module_names(concrete) -> tuple[set[str], set[str]]:
    """``(head modules, embedding modules)`` the class's tie declaration names.

    Empty when the class declares no ``_tied_weights_keys`` — nothing can be derived from it, and
    the post-load check is what fails loud there.
    """
    raw = getattr(concrete, "_tied_weights_keys", None) or {}
    pairs = list(raw.items()) if isinstance(raw, dict) else [(key, key) for key in raw]
    heads = {target.rsplit(".", 1)[0] for target, _source in pairs}
    embeddings = {source.rsplit(".", 1)[0] for _target, source in pairs}
    return heads, embeddings


def _plan_keys_naming(plan: dict, modules: set[str]) -> set[str]:
    """Keys of ``plan`` naming one of ``modules``.

    A class ``_tp_plan`` is model-absolute (``lm_head``) while a config ``base_model_tp_plan`` is
    backbone-relative (``embed_tokens`` for ``model.embed_tokens``), so either spelling matches.
    """
    return {
        key
        for key in plan
        for module in modules
        if module == key or module.endswith("." + key) or key.endswith("." + module)
    }


@contextmanager
def consistent_tied_tp_plan(model_class, model_config):
    """Keep both ends of a tied embedding/head pair on the same side of the TP plan.

    transformers shards module by module (``apply_tensor_parallelism``) and ties afterwards, so a
    tied pair ends as ONE parameter: a tied config injects ``embed_tokens: embedding_rowwise`` into
    the backbone plan and a ForCausalLM's ``lm_head: colwise_gather_output`` agrees with it on
    ``Shard(0)`` of the vocab dim — the vocab-parallel pair TP wants, and the reason the embedding
    and head shard at all. Where only ONE end carries a plan entry (a multimodal wrapper class
    declaring no ``lm_head`` entry; a backbone shipping no ``base_model_tp_plan``) the tie hands the
    unplanned end a weight of the other kind, and the first forward dies inside ``F.linear`` on mixed
    plain/DTensor operands. Drop that lone entry for the load so the pair stays replicated instead —
    correct, just not free.
    """
    concrete = _concrete_model_class(model_class, model_config)
    heads, embeddings = _tied_module_names(concrete)
    if not config_ties_word_embeddings(model_config) or not heads or not embeddings:
        yield
        return
    class_plan = getattr(concrete, "_tp_plan", None)
    # The embedding entry lives on the BACKBONE's plan, whose config is the text one: a composite
    # wrapper's own base_model_tp_plan is None (Qwen3.5), so the outer config would report no entry.
    plan_config = text_config(model_config)
    base_plan = getattr(plan_config, "base_model_tp_plan", None)
    head_keys = _plan_keys_naming(class_plan or {}, heads)
    embedding_keys = _plan_keys_naming(class_plan or {}, embeddings) | _plan_keys_naming(base_plan or {}, embeddings)
    if bool(head_keys) == bool(embedding_keys):
        yield
        return

    drop = head_keys or embedding_keys
    original_class_plan, original_base_plan = class_plan, base_plan
    if class_plan is not None:
        concrete._tp_plan = {key: value for key, value in class_plan.items() if key not in drop}
    if base_plan is not None:
        plan_config.base_model_tp_plan = {key: value for key, value in base_plan.items() if key not in drop}
    try:
        yield
    finally:
        if original_class_plan is not None:
            concrete._tp_plan = original_class_plan
        if original_base_plan is not None:
            plan_config.base_model_tp_plan = original_base_plan


def validate_tied_pair_consistent(model, applied_plan: dict) -> None:
    """Fail loud when a tied embedding/head pair survived the load in an unusable shape.

    Two silent breakages, both fatal at the first forward or the first step:

    * two distinct parameters — transformers refused the tie (a checkpoint carrying both keys with
      different values), so each end would train on half the tied gradient;
    * a pair whose sharding disagrees with the plan — a sharded weight in an un-transformed head, or
      a replicated weight in a head the plan gave TP's input/output transforms. Either dies inside
      ``F.linear`` on mixed plain/DTensor operands.
    """
    in_emb = model.get_input_embeddings()
    out_emb = model.get_output_embeddings()
    if in_emb is None or out_emb is None:
        return
    if out_emb.weight is not in_emb.weight:
        raise RuntimeError(
            "tie_word_embeddings=True but the TP load produced two independent parameters for the "
            "tied embedding/head — each would train on half the tied gradient. transformers ties "
            "AFTER sharding, so this is a refused tie: the checkpoint carries both keys with "
            "different values. Re-export it with tie_word_embeddings=false."
        )
    # By module, not by parameter name: the tie makes both ends one object, which ``named_parameters``
    # emits only under the embedding's name.
    head_name = next((name for name, module in model.named_modules() if module is out_emb), None)
    if head_name is None:
        return
    head_sharded = tp_plan_shards_params(f"{head_name}.weight", applied_plan)
    if head_sharded != isinstance(in_emb.weight, DTensor):
        raise RuntimeError(
            f"The tied embedding/head pair and the applied TP plan disagree: {head_name!r} is "
            f"{'covered' if head_sharded else 'not covered'} by a sharding style while the tied "
            f"weight loaded {'sharded' if not head_sharded else 'replicated'}. The forward would die "
            f"on mixed plain/DTensor operands. Train this architecture without tensor parallelism."
        )
