"""CPU CP-wrapper fixtures without CUDA attention patching."""

from types import SimpleNamespace

from torch import nn

from src.distributed.context_parallel.wrapper import UlyssesCPModelWrapper


def unpatched_cp_wrapper(model, *, cp_size=1, cp_rank=0, cp_config=None, attention_layers=()):
    """Exercise the production wrapper while supplying CPU-compatible attention or hidden states."""
    config = (
        cp_config if cp_config is not None else SimpleNamespace(cp_size=cp_size, cp_rank=cp_rank, process_group=None)
    )
    wrapper = UlyssesCPModelWrapper.__new__(UlyssesCPModelWrapper)
    nn.Module.__init__(wrapper)
    wrapper.model = model
    wrapper.cp_config = config
    wrapper.cp_size, wrapper.cp_rank = config.cp_size, config.cp_rank
    wrapper.cp_group = config.process_group
    wrapper._attention_layers = list(attention_layers)
    return wrapper
