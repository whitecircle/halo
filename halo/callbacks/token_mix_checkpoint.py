"""The token mix's fingerprint in every checkpoint of a ``token_sources`` run, carried in the ``trainer_state.json`` a
save publishes last, and read back for a resume to hold the live mix to."""

import json
import os

from transformers import TrainerCallback
from transformers.trainer_callback import ExportableState

from halo.checkpoint.format import TRAINER_STATE_FILE


class TokenMixCheckpointCallback(TrainerCallback, ExportableState):
    """Exports ``fingerprint`` (``TokenMix.fingerprint``) as its state, which the base save writes into each
    checkpoint's ``trainer_state.json`` under ``stateful_callbacks``: a checkpoint complete by that file holds it."""

    def __init__(self, fingerprint: dict):
        self.fingerprint = fingerprint

    def state(self) -> dict:
        return {"args": {"fingerprint": self.fingerprint}, "attributes": {}}


def find_saved_token_mix(checkpoint: str) -> dict | None:
    """The fingerprint ``checkpoint``'s ``trainer_state.json`` carries; ``None`` when it carries none or is absent."""
    path = os.path.join(checkpoint, TRAINER_STATE_FILE)
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as handle:
        exported = json.load(handle).get("stateful_callbacks", {}).get(TokenMixCheckpointCallback.__name__)
    return None if exported is None else exported["args"]["fingerprint"]
