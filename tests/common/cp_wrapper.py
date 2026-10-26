"""CPU CP-wrapper fixtures without CUDA attention patching."""

from torch import nn

from src.distributed.context_parallel.config import CPConfig
from src.distributed.context_parallel.ulysses_model import UlyssesModel


def unpatched_cp_wrapper(model, *, cp_size=1, cp_rank=0, cp_config=None):
    """A production :class:`UlyssesModel` over ``model`` with its attention left unpatched, so a CPU test can
    supply its own attention or hidden states; without ``cp_config``, one placing this rank at ``cp_rank``."""
    if cp_config is None:
        cp_config = CPConfig.__new__(CPConfig)
        cp_config.cp_size, cp_config.cp_rank, cp_config.process_group = cp_size, cp_rank, None
    wrapper = UlyssesModel.__new__(UlyssesModel)
    nn.Module.__init__(wrapper)
    wrapper.model = model
    wrapper.cp_config = cp_config
    return wrapper
