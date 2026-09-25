"""Host-address classification shared by the weight-sync clients and the rollout actors.

Standard library only: :mod:`src.environments.ray_actors` imports it, and the weight-sync client
module it lives beside pulls in torch, DTensor and the NCCL transport.
"""

_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "0.0.0.0", "::1", "localhost"})


def is_loopback(host: str) -> bool:
    """Whether ``host`` names this machine only: a loopback address, ``localhost`` or the wildcard."""
    return host in _LOOPBACK_HOSTS or host.startswith("127.")
