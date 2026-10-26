"""Positional constructor slots of a TRL/HF trainer base, read off its installed signature."""

import inspect
from typing import Any


def resolve_ctor_positions(trainer_cls: type, *names: str) -> dict[str, int]:
    """``name → *args slot`` for ``trainer_cls``'s named ctor parameters, read off its signature so a TRL release
    that inserts or reorders a parameter moves the slot with it. A name the signature lacks raises: guessing a
    slot would gate on the wrong argument."""
    params = [
        name
        for name, param in inspect.signature(trainer_cls.__init__).parameters.items()
        if name != "self" and param.kind not in (param.VAR_POSITIONAL, param.VAR_KEYWORD)
    ]
    missing = [name for name in names if name not in params]
    if missing:
        raise ValueError(
            f"{trainer_cls.__name__}.__init__ declares no parameter named {missing}; the installed "
            f"TRL signature no longer matches this trainer's ctor gate."
        )
    return {name: params.index(name) for name in names}


def get_ctor_arg(ctor_args: tuple, kwargs: dict, name: str, positions: dict[str, int]):
    """A trainer ctor parameter by keyword, else by its slot in ``positions`` (:func:`resolve_ctor_positions`)."""
    if name in kwargs:
        return kwargs[name]
    position = positions[name]
    return ctor_args[position] if len(ctor_args) > position else None


def get_ctor_config(ctor_args: tuple, kwargs: dict, positions: dict[str, int]):
    """The training config a trainer ctor was handed. Truthiness, not membership: an explicit ``args=None``
    keyword falls through to the positional slot."""
    return kwargs.get("args") or get_ctor_arg(ctor_args, {}, "args", positions)


def get_ctor_model_and_config(ctor_args: tuple, kwargs: dict, positions: dict[str, int] | None) -> tuple[Any, Any]:
    """``(model, args)`` a trainer ctor was handed. Without a ``positions`` table only keywords are read and
    positionals raise: a positional model or config would otherwise miss every setup step that reads it."""
    if positions is None:
        if ctor_args:
            raise ValueError(
                "Positional trainer arguments reached _init_distributed_config without a ctor_positions "
                "table, so a positional `model` or `args` cannot be read and the setup that needs them "
                "would be skipped. Pass ctor_positions=resolve_ctor_positions(<forwarded base>, 'model', "
                "'args'), or construct the trainer with keywords."
            )
        return kwargs.get("model"), kwargs.get("args")
    return get_ctor_arg(ctor_args, kwargs, "model", positions), get_ctor_config(ctor_args, kwargs, positions)
