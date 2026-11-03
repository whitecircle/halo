"""Optimizer state as a resume compares it: this rank's sharded view, and the bit-exact comparison of saved vs
restored."""

import torch
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_optimizer_state_dict

from halo.distributed.fsdp import reshard_fsdp2_modules
from halo.models.unwrapping import unwrap_toolkit_wrappers


def local_optimizer_state(model, optimizer) -> dict:
    """This rank's optimizer state as an :func:`assert_optimizer_state_bit_exact` payload, the view the per-rank shards
    hold; no collective. Reshards first: after an eval-only forward ``get_optimizer_state_dict`` maps no param to an
    FQN (``KeyError: 0``). A stateless optimizer reads as empty, since that call would fill it with a zero-LR step
    running the trainer's step-pre-hooks outside the restore's mute."""
    reshard_fsdp2_modules(model)
    if not optimizer.state:
        return {"state": {}}
    osd = get_optimizer_state_dict(
        unwrap_toolkit_wrappers(model),
        optimizer,
        options=StateDictOptions(full_state_dict=False, cpu_offload=True),
    )

    def _local(value):
        if not torch.is_tensor(value):
            return value
        local = value.to_local() if hasattr(value, "to_local") else value
        return local.detach().clone().cpu()

    return {
        "state": {
            fqn: {key: _local(value) for key, value in entry.items()} for fqn, entry in osd.get("state", {}).items()
        }
    }


def assert_optimizer_state_bit_exact(saved: dict, restored: dict) -> None:
    """Assert two ``optimizer.state_dict()`` payloads carry identical per-param state; ``saved`` must be non-empty."""
    assert len(saved["state"]) > 0, "premise: the saved optimizer state is empty"
    assert saved["state"].keys() == restored["state"].keys(), (
        f"restored state covers params {sorted(restored['state'])}, saved {sorted(saved['state'])}"
    )
    for idx, saved_entry in saved["state"].items():
        restored_entry = restored["state"][idx]
        assert saved_entry.keys() == restored_entry.keys(), f"param {idx}: state keys differ"
        for key, val in saved_entry.items():
            if torch.is_tensor(val):
                res = restored_entry[key]
                assert torch.is_tensor(res) and res.dtype == val.dtype and torch.equal(res.to(val.device), val), (
                    f"param {idx} state '{key}' not bit-exact after load"
                )
            else:
                assert restored_entry[key] == val, f"param {idx} state '{key}' changed after load"


def optimizer_state_matches(saved: dict, restored: dict) -> tuple[bool, str]:
    """:func:`assert_optimizer_state_bit_exact` as ``(ok, why)``; a bare assert on one rank leaves its peers blocked at
    the next collective."""
    try:
        assert_optimizer_state_bit_exact(saved, restored)
    except AssertionError as e:
        return False, str(e)
    return True, ""
