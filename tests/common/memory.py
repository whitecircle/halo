"""Device and host memory for the test suites: release between phases, current and peak GPU use, the
process's peak resident set size, and the peak bytes a CPU region's ops hold at once."""

import gc
import pathlib
import re
import weakref

import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_leaves

# Written to /proc/self/clear_refs, restarts the process's peak resident set size (VmHWM) from its current size.
_RESET_PEAK_RSS = "5"


def cleanup_memory():
    """Free GPU memory with synchronization."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def finish_phase(trainer) -> None:
    """Release a finished phase's trainer before the next phase builds on the same GPUs: its DeepEP
    buffers (``cleanup_ep``, collective, a no-op without EP layers) and the cached memory. A trainer a
    teardown hook still references stays alive, so dropping the name alone would free neither."""
    trainer.cleanup_ep()
    cleanup_memory()


def gpu_mem_gb() -> float:
    """Current GPU memory usage of the current device, in GB."""
    return torch.cuda.memory_allocated() / 1e9


def gpu_peak_mem_gb() -> float:
    """Peak GPU memory usage of the current device, in GB."""
    return torch.cuda.max_memory_allocated() / 1e9


def reset_host_peak_rss() -> None:
    """Restart this process's peak resident set size (Linux ``VmHWM``) from its current size."""
    pathlib.Path("/proc/self/clear_refs").write_text(_RESET_PEAK_RSS)


def host_peak_rss_gib() -> float:
    """This process's peak resident set size since it started or since :func:`reset_host_peak_rss`, in GiB."""
    peak = re.search(r"^VmHWM:\s+(\d+) kB$", pathlib.Path("/proc/self/status").read_text(), re.MULTILINE)
    if peak is None:
        raise RuntimeError("/proc/self/status carries no VmHWM line to read a peak resident set size from")
    return int(peak.group(1)) / 1024**2


class LiveTensorBytes(TorchDispatchMode):
    """The most bytes the tensors allocated inside the region hold at once (``peak``), and what they still hold
    (``live``). An op's output counts when its storage is new (not an input's, not a view); it stops counting
    when the tensor is freed, which a live view of it postpones. Tensors allocated before the region never count.
    """

    def __init__(self):
        super().__init__()
        self.live = 0
        self.peak = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        output = func(*args, **kwargs)
        inputs = {t.untyped_storage().data_ptr() for t in tree_leaves((args, kwargs)) if isinstance(t, torch.Tensor)}
        for tensor in tree_leaves(output):
            if not isinstance(tensor, torch.Tensor) or tensor._base is not None:
                continue
            storage = tensor.untyped_storage()
            if storage.data_ptr() in inputs or storage.nbytes() == 0:
                continue
            self.live += storage.nbytes()
            self.peak = max(self.peak, self.live)
            weakref.finalize(tensor, self._release, storage.nbytes())
        return output

    def _release(self, nbytes: int) -> None:
        self.live -= nbytes
