"""The re-init suite restores the served policy even when an earlier row left trained weights served. Its restore
re-pushes only the tensor its cycles perturb, so its baseline is taken after pushing the loaded checkpoint whole."""

import types

import pytest
import torch
from torch import nn

from tests.common import weight_sync_reinit
from tests.common.weight_sync_reinit import PERTURBED_PARAM, run_reinit_cycles


class _Checkpoint(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.ParameterDict(
            {
                PERTURBED_PARAM.replace(".", "_"): nn.Parameter(torch.ones(4)),
                "layer": nn.Parameter(torch.full((4,), 2.0)),
            }
        )

    def named_parameters(self, *args, **kwargs):
        names = {PERTURBED_PARAM.replace(".", "_"): PERTURBED_PARAM, "layer": "model.layers.0.weight"}
        for key, param in self.weights.items():
            yield names[key], param

    def to(self, device):
        return self


@pytest.fixture
def served(monkeypatch):
    """A server whose every tensor holds a trained value the checkpoint does not."""
    state = {PERTURBED_PARAM: torch.full((4,), 1.5), "model.layers.0.weight": torch.full((4,), 2.5)}

    def sync(client_class, server_url, named_params, device, group_port):
        state.update({name: tensor.detach().clone() for name, tensor in named_params})

    def probe(server_url, model_name):
        return {name: round(float(tensor.sum()), 6) for name, tensor in sorted(state.items())}

    def baseline(server_url, model_name, checks):
        checks["server_serves_the_trained_model"] = True
        return probe(server_url, model_name)

    monkeypatch.setattr(weight_sync_reinit, "_sync", sync)
    monkeypatch.setattr(weight_sync_reinit, "probe_top_logprobs", probe)
    monkeypatch.setattr(weight_sync_reinit, "record_served_baseline", baseline)
    monkeypatch.setattr(weight_sync_reinit, "_server_gpu_used_mib", lambda server_gpu: 0)
    monkeypatch.setattr(weight_sync_reinit, "_trainer_gpu_used_mib", lambda device: 0)
    monkeypatch.setattr(weight_sync_reinit, "weight_transfer_port", lambda knob: 29999)
    monkeypatch.setattr(
        weight_sync_reinit,
        "AutoModelForCausalLM",
        types.SimpleNamespace(from_pretrained=lambda name, dtype: _Checkpoint()),
    )
    return state


def _restored(cycles: int = 3) -> bool:
    ctx = types.SimpleNamespace(device=torch.device("cpu"), broadcast_checks=lambda checks: checks)
    client_class = types.SimpleNamespace(BACKEND_NAME="fake")
    result = run_reinit_cycles(
        ctx, client_class, server_url="fake", model_name="fake", server_gpu=None, cycles=cycles, group_port_knob="K"
    )
    return result["checks"]["served_policy_restored_by_a_final_sync"]


def test_restore_reaches_the_baseline_over_a_trained_server(served):
    assert _restored()


def test_without_the_checkpoint_push_the_restore_misses(served, monkeypatch):
    """The baseline taken over the trained weights is out of the restore's reach."""
    monkeypatch.setattr(weight_sync_reinit, "serve_the_checkpoint", lambda *args, **kwargs: None)
    assert not _restored()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
