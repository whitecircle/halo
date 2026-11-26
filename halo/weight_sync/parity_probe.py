"""The ``HALO_WEIGHT_SYNC_PARITY_PROBE`` mode, resolved once for the trainer's probe and the rank-uniform check."""

from typing import Literal, get_args

from halo.env import env_str

ParityProbeMode = Literal["warn", "raise", "off"]


def resolve_parity_probe_mode() -> ParityProbeMode:
    """The probe mode from the environment; raises ``ValueError`` on a value outside ``ParityProbeMode``."""
    mode = env_str("HALO_WEIGHT_SYNC_PARITY_PROBE", "warn").strip().lower()
    if mode not in get_args(ParityProbeMode):
        raise ValueError(f"HALO_WEIGHT_SYNC_PARITY_PROBE={mode!r} is not one of {get_args(ParityProbeMode)}.")
    return mode
