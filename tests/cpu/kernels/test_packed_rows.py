"""The packed-row gate helpers (tests/common/packed_rows.py) on a tiny CPU model, so a helper that backpropagates
under ``torch.no_grad`` fails here and not in the first GPU launch."""

import pytest
import torch

from tests.common import packed_rows as gate
from tests.common.tiny_models import TINY_DENSE_ATTENTION_FAMILIES, TINY_DENSE_FAMILIES, tiny_family_model


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return tiny_family_model(TINY_DENSE_ATTENTION_FAMILIES["llama"], overrides={"num_hidden_layers": 2}).eval()


def test_isolation_check_passes_when_documents_are_isolated(model):
    checks = {}
    gate.check_isolation(model, "sdpa", "position_ids", (12, 9, 7), checks)
    assert checks["sdpa_a_neighbour_does_not_move_the_documents_around_it"] is True


def test_isolation_check_detects_a_leak_across_documents(model, monkeypatch):
    """A row that hands the model no document boundary is one causal sequence: the changed second document moves the
    third, and the check must say so."""
    monkeypatch.setattr(gate, "emission_kwargs", lambda emission, position_ids, config: {})
    checks = {}
    gate.check_isolation(model, "sdpa", "position_ids", (12, 9, 7), checks)
    assert checks["sdpa_a_neighbour_does_not_move_the_documents_around_it"] is False


def test_a_group_scalar_that_flips_does_not_stand_for_its_group():
    reference = {"layers.0.linear_attn.A_log": torch.tensor([1.0, -1.0]), "layers.0.mlp.experts.w": torch.ones(64)}
    noisy = {"layers.0.linear_attn.A_log": torch.tensor([-1.0, 1.0]), "layers.0.mlp.experts.w": torch.ones(64) * 1.001}
    errors = gate.group_grad_errors(noisy, reference)
    assert errors["gdn_scalars"] == pytest.approx(2.0)
    assert errors["experts"] == pytest.approx(1e-3, rel=1e-3)
    floors = gate.grad_noise_floor(noisy, reference, reference)
    assert floors["gdn_scalars"] > 1.0 and floors["experts"] < 1e-2


def test_a_leak_into_a_projection_exceeds_the_noise_floor():
    reference = {"layers.0.self_attn.q_proj.weight": torch.ones(64)}
    leaked = {"layers.0.self_attn.q_proj.weight": torch.ones(64) * 1.5}
    floors = gate.grad_noise_floor(reference, reference, reference, reference)
    assert gate.group_grad_errors(leaked, reference)["projections"] > 2 * floors["projections"]


def test_a_doubled_gdn_scalar_gradient_is_past_the_floor_of_a_bf16_model_in_another_order():
    """The floor comes from a bf16 model's own reruns; a packed backward that doubled a head's A_log gradient stays
    past the gate's ratio while the reruns themselves stay under it."""
    torch.manual_seed(0)
    gdn = tiny_family_model(
        TINY_DENSE_FAMILIES["qwen3_5"],
        overrides={"num_hidden_layers": 2, "layer_types": ["linear_attention", "full_attention"]},
    ).to(torch.bfloat16)
    documents = gate.build_documents((12, 9, 7), gdn.config.get_text_config().vocab_size, "cpu")
    _, per_row = gate.row_step(gdn, documents, None)
    _, reordered = gate.row_step(gdn, documents[::-1], None)
    split = gate.split_row_step(gdn, documents)
    floors = gate.grad_noise_floor(per_row, per_row, reordered, split)
    assert any("A_log" in name for name in per_row)
    doubled = {name: grad * 2 if "A_log" in name else grad for name, grad in per_row.items()}
    assert gate.group_grad_errors(doubled, per_row)["gdn_scalars"] > 2.0 * floors["gdn_scalars"]
    assert gate.group_grad_errors(split, per_row)["gdn_scalars"] <= floors["gdn_scalars"]


def test_a_noise_dominated_scalar_group_is_judged_by_its_norm():
    """A_log whose bf16 gradient is far from fp32 passes with a flipped direction and equal norm, and fails at 2x
    and 0.5x; a resolvable projection group keeps the direction rule."""
    reference = {"l.linear_attn.A_log": torch.tensor([1.0, -1.0, 0.5]), "l.self_attn.q_proj.weight": torch.ones(64)}
    per_row = {"l.linear_attn.A_log": torch.tensor([0.9, 0.2, -0.4]), "l.self_attn.q_proj.weight": torch.ones(64)}
    floors = gate.grad_noise_floor(per_row, reference)

    def verdict(packed):
        return gate.judge_grad_groups(packed, per_row, reference, floors, 2.0)

    same = {**per_row, "l.linear_attn.A_log": torch.tensor([-0.4, 0.9, 0.2])}
    verdicts, by_norm = verdict(same)
    assert by_norm == ["gdn_scalars"] and all(verdicts.values())
    for scale in (2.0, 0.5):
        scaled = {**per_row, "l.linear_attn.A_log": per_row["l.linear_attn.A_log"] * scale}
        assert verdict(scaled)[0]["gdn_scalars"] is False
    wrong = {**per_row, "l.self_attn.q_proj.weight": torch.ones(64) * 1.5}
    assert verdict(wrong)[0]["projections"] is False


def test_the_split_step_sums_to_the_joint_step(model):
    documents = gate.build_documents((12, 9), model.config.vocab_size, "cpu")
    _, joint = gate.row_step(model, documents, None)
    split = gate.split_row_step(model, documents)
    assert split.keys() == joint.keys()
    assert all(torch.allclose(split[n], joint[n], rtol=1e-3, atol=1e-5) for n in joint)


def test_step_returns_gradients_and_logits_carry_the_graph(model):
    documents = gate.build_documents((12, 9), model.config.vocab_size, "cpu")
    logits = gate.row_logits(model, documents, None)
    assert logits.requires_grad
    step_logits, grads = gate.row_step(model, documents, None)
    assert grads and torch.equal(step_logits, logits.detach().float())


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
