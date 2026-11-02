"""CP-wrapper fixtures: a wrapper without CUDA attention patching, and a probe of the output embedding's inputs."""

import contextlib
from collections.abc import Iterator

from torch import nn

from halo.distributed.context_parallel.config import CPConfig
from halo.distributed.context_parallel.ulysses_model import UlyssesModel


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


@contextlib.contextmanager
def head_input_lengths(model: nn.Module) -> Iterator[list[int]]:
    """The sequence length of each input ``model``'s output embedding projects inside the block: how many positions'
    logits the forward builds."""
    lengths: list[int] = []
    handle = model.get_output_embeddings().register_forward_hook(
        lambda _module, args, _output: lengths.append(args[0].shape[-2])
    )
    try:
        yield lengths
    finally:
        handle.remove()
