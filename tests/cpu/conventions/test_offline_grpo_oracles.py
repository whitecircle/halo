#!/usr/bin/env python
"""The offline-GRPO GPU suites' shared oracles (:mod:`tests.common.offline_grpo`) grade the real trainer, on CPU:
they agree with it, reject a wrong objective, see each row in the sweep's batches, leave the policy as found, compare
only rows whose routing leads bf16 noise with headroom on the policy compared, and read a resumed export's move at the
export's dtype."""

import copy
import json
import random
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from accelerate import PartialState
from safetensors.torch import save_file
from torch import nn
from transformers import PretrainedConfig, Qwen3Config, Qwen3ForCausalLM, Qwen3MoeConfig, Qwen3MoeForCausalLM
from transformers.utils.output_capturing import OutputRecorder

from halo.data.collators.offline_grpo import REF_PER_TOKEN_LOGPS_COLUMN, OfflineGRPODataCollatorWithPadding
from halo.distributed.parallelism_config import ParallelismConfig
from tests.common.gloo import run_gloo_ranks
from tests.common.offline_grpo import (
    OFFLINE_TEXT_WORDS,
    OFFLINE_VOCAB,
    ROUTING_HEADROOM,
    agree_kl_rows,
    doubled_head_kl_verdict,
    full_logits_kl,
    kl_oracle_training_set,
    offline_grpo_config,
    offline_grpo_dataset,
    offline_grpo_trainer,
    pure_kl_batch,
    reference_oracle_rows,
    resumed_export_verdict,
    routing_clear_rows,
    sweep_batches,
    swept_reference_error,
)
from tests.common.tolerances import TOL

PartialState()

BETA = 0.2
# The table routers' bf16 perturbation wherever their choice holds, so the scan's noise is exactly this.
NOISE = 0.01
# How far a flip upstream moves every later router logit: far past NOISE, so counting it would clear no row.
FLIP_SHIFT = 0.5
# Table-router tokens: fp32 logits, what the bf16 oracle adds, and the router's own pick where it is not its top-k.
WIDE, AMPLE, ROOMY, CLEAR, TIGHT, FLIP, GROUPED, LEADING_BLOCK, NEAR_BLOCK, RIVAL_BLOCK = range(1, 11)
TOP1_TOKENS = {
    WIDE: ([1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, NOISE], None),
    # Past the headroom by more than ROOMY, short of twice it.
    AMPLE: ([1.0, 1.0 - (ROUTING_HEADROOM + 1.5) * NOISE, 0.0, 0.0], [0.0] * 4, None),
    # Just past the headroom, by far less than WIDE.
    ROOMY: ([1.0, 1.0 - (ROUTING_HEADROOM + 0.5) * NOISE, 0.0, 0.0], [0.0] * 4, None),
    # Past twice NOISE, short of the headroom.
    CLEAR: ([1.0, 1.0 - (ROUTING_HEADROOM + 2) / 2 * NOISE, 0.0, 0.0], [0.0] * 4, None),
    TIGHT: ([1.0, 1.0 - 1.5 * NOISE, 0.0, 0.0], [0.0] * 4, None),
    # A margin past twice NOISE, which a larger bf16 perturbation still flips.
    FLIP: (
        [1.0, 1.0 - (ROUTING_HEADROOM + 1) * NOISE, 0.0, 0.0],
        [-(ROUTING_HEADROOM + 2) / 2 * NOISE, (ROUTING_HEADROOM + 2) / 2 * NOISE, 0.0, 0.0],
        None,
    ),
    # Picks expert 1 under a lead of expert 0, as a group-limited router may.
    GROUPED: ([1.0, 0.9, 0.0, 0.0], [0.0] * 4, [1]),
}
# A tied tiny MoE and its random-word rows, of which both picked on the policy as saved stop clearing once the head
# doubles.
TIED_SEED = 4
TIED_ROWS = 24
# Two blocks of two experts, one picked whole.
ONE_OF_TWO_BLOCKS = {
    WIDE: ([1.0, 0.9, 0.0, 0.0], [0.0, 0.0, 0.0, NOISE], [0, 1]),
    # Each picked logit leads its rank in the other block, though expert 0 outscores expert 3.
    LEADING_BLOCK: ([0.8, 0.1, 0.9, 0.5], [0.0] * 4, [2, 3]),
    # The picked block's first logit leads the other's by less than twice NOISE.
    NEAR_BLOCK: ([0.8, 0.1, 0.8 + NOISE / 2, 0.5], [0.0] * 4, [2, 3]),
}
# Four blocks of two, two picked: expert 2 races the selection from a picked block that holds none of it.
TWO_OF_FOUR_BLOCKS = {
    WIDE: ([1.0, 0.9, *[0.0] * 6], [*[0.0] * 7, NOISE], [0, 1]),
    RIVAL_BLOCK: ([1.0, 0.95, 0.95 - NOISE, *[0.0] * 5], [0.0] * 8, [0, 1]),
}


def _tiny_policy(seed, *, moe=False, tied=False):
    torch.manual_seed(seed)
    fields = {
        "vocab_size": len(OFFLINE_VOCAB),
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_hidden_layers": 2 if moe else 1,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 8,
        "pad_token_id": OFFLINE_VOCAB["<pad>"],
        "eos_token_id": OFFLINE_VOCAB["<eos>"],
        "tie_word_embeddings": tied,
    }
    if moe:
        fields |= {
            "num_experts": 8,
            "num_experts_per_tok": 2,
            "moe_intermediate_size": 32,
            "router_aux_loss_coef": 0.0,
        }
        return Qwen3MoeForCausalLM(Qwen3MoeConfig(**fields))
    return Qwen3ForCausalLM(Qwen3Config(**fields))


def _cpu_trainer(tmp_path, name, *, beta, finalizers, moe, **overrides):
    args = offline_grpo_config(
        str(tmp_path / name),
        steps=1,
        save_steps=1,
        seed=0,
        kl_beta=beta,
        sequence_length=16,
        evaluate=False,
        save=False,
        use_cpu=True,
        bf16=False,
        gradient_checkpointing=False,
        dataset_num_proc=1,
        **overrides,
    )
    ctx = SimpleNamespace(on_teardown=finalizers.append)
    train = kl_oracle_training_set(2, 0) if moe else offline_grpo_dataset(4)
    return offline_grpo_trainer(ctx, _tiny_policy(7, moe=moe), ParallelismConfig(), args, train)


@pytest.mark.parametrize("moe", [False, True], ids=["dense", "moe"])
def test_the_shared_oracles_agree_with_the_trainer_and_reject_a_wrong_objective(tmp_path, moe):
    finalizers = []
    expected = reference_oracle_rows(_cpu_trainer(tmp_path, "oracle", beta=0.0, finalizers=finalizers, moe=moe))
    trainer = _cpu_trainer(tmp_path, "train", beta=BETA, finalizers=finalizers, moe=moe)
    assert finalizers[-1] == trainer.cleanup_ep, "teardown must release the trainer's DeepEP buffers"
    assert swept_reference_error(trainer.train_dataset[REF_PER_TOKEN_LOGPS_COLUMN], expected) < TOL.logprob_atol

    head = trainer.model.get_output_embeddings().weight
    before = head.detach().clone()
    trainer.model.train()
    rows = routing_clear_rows(lambda dtype: copy.deepcopy(trainer.model).to(dtype).eval(), trainer.train_dataset)
    assert rows.clear, f"too few rows of the tiny {'MoE' if moe else 'dense'} policy clear the routing noise: {rows}"
    every = tuple(range(len(trainer.train_dataset)))
    if moe:
        assert rows.rows != (0, 1), "the scan chose the first rows, which a batch ignoring its rows also collates"
    else:
        assert rows.rows == (0, 1) and rows.cleared == rows.headroom == every, (
            "a dense model has no routing to clear: every row clears with headroom and the first make the batch"
        )
    batch = pure_kl_batch(trainer, rows.rows)
    collated = [
        (prompt[prompt_mask.bool()].tolist(), completion[completion_mask.bool()].tolist())
        for prompt, prompt_mask, completion, completion_mask in zip(
            batch["prompt_input_ids"],
            batch["prompt_attention_mask"],
            batch["completion_input_ids"],
            batch["completion_attention_mask"],
            strict=True,
        )
    ]
    chosen = [trainer.train_dataset[index] for index in rows.rows]
    assert collated == [(row["prompt_input_ids"], row["completion_input_ids"]) for row in chosen], (
        "the KL batch is not the rows the scan chose"
    )
    assert not batch["advantage"].any()
    nonzero, matches = doubled_head_kl_verdict(trainer, batch, lambda model: full_logits_kl(model, batch, BETA), "cpu")
    assert nonzero and matches
    _, wrong_beta_matches = doubled_head_kl_verdict(
        trainer, batch, lambda model: full_logits_kl(model, batch, BETA / 2), "cpu control"
    )
    assert not wrong_beta_matches, "the oracle comparison cannot tell a halved KL coefficient apart"
    assert trainer.model.get_output_embeddings().weight is head and torch.equal(head.detach(), before)
    assert trainer.model.training


@pytest.mark.parametrize("width", [2, 5], ids=["whole_batches", "short_last_batch"])
def test_the_reference_oracle_collates_the_sweeps_batches(tmp_path, width):
    """A row padded otherwise on the two sides can resolve a routing near-tie otherwise."""
    batches = []
    collate = OfflineGRPODataCollatorWithPadding.__call__

    def recorded(collator, features):
        batches.append([(tuple(row["prompt_input_ids"]), tuple(row["completion_input_ids"])) for row in features])
        return collate(collator, features)

    finalizers = []
    with patch.object(OfflineGRPODataCollatorWithPadding, "__call__", recorded):
        trainer = _cpu_trainer(
            tmp_path, "train", beta=BETA, finalizers=finalizers, moe=True, per_device_train_batch_size=width
        )
        swept = list(batches)
        batches.clear()
        oracle = _cpu_trainer(
            tmp_path, "oracle", beta=0.0, finalizers=finalizers, moe=True, per_device_train_batch_size=width
        )
        expected = reference_oracle_rows(oracle)
    assert (len(trainer.train_dataset) % width > 0) == (width == 5)
    assert len(swept) > 1 and batches == swept
    assert len(expected) == len(trainer.train_dataset)
    assert swept_reference_error(trainer.train_dataset[REF_PER_TOKEN_LOGPS_COLUMN], expected) < TOL.logprob_atol


class _TableRouter(nn.Module):
    """Logits a per-token table gives, moved by ``noise_scale`` times the table's noise, all moved by ``shift``;
    selects the table's pick where it names one, else its top-k."""

    def __init__(self, tokens, top_k, noise_scale):
        super().__init__()
        logits = torch.zeros(max(tokens) + 1, len(next(iter(tokens.values()))[0]))
        picks = torch.full((max(tokens) + 1, top_k), -1)
        for token, (row, noise, pick) in tokens.items():
            logits[token] = torch.tensor(row) + noise_scale * torch.tensor(noise)
            if pick is not None:
                picks[token] = torch.tensor(pick)
        self.register_buffer("logits", logits)
        self.register_buffer("picks", picks)

    def forward(self, input_ids, shift):
        logits, picks = self.logits[input_ids] + shift, self.picks[input_ids]
        ids = torch.where(picks >= 0, picks, logits.topk(picks.size(-1), dim=-1).indices)
        return logits, logits.gather(-1, ids), ids


class _TableMoE(nn.Module):
    """Two table routers, the bf16 oracle's moved by ``noise_scales`` times the table's noise; the second's logits move
    by ``FLIP_SHIFT`` wherever the first picked expert 1. Without ``declared``, its routers are read off the forward's
    ``router_logits``, as a remote-code family's are."""

    _can_record_outputs = {"router_logits": OutputRecorder(_TableRouter, index=0)}

    def __init__(
        self, dtype, tokens=TOP1_TOKENS, *, top_k=1, declared=True, returns_logits=True, noise_scales=(1, 1), **groups
    ):
        super().__init__()
        if not declared:
            self._can_record_outputs = {}
        self.returns_logits = returns_logits
        experts = len(next(iter(tokens.values()))[0])
        self.config = PretrainedConfig(num_experts=experts, num_experts_per_tok=top_k, **groups)
        self.device = torch.device("cpu")
        self.routers = nn.ModuleList(
            _TableRouter(tokens, top_k, scale if dtype == torch.bfloat16 else 0) for scale in noise_scales
        )
        # An untied head, which no router reads.
        self.head = nn.Linear(1, 1, bias=False)

    def get_output_embeddings(self):
        return self.head

    def forward(self, input_ids, attention_mask, use_cache, output_router_logits=False):
        first = self.routers[0](input_ids, 0.0)
        second = self.routers[1](input_ids, FLIP_SHIFT * (first[2] == 1).any(-1, keepdim=True).float())
        outputs = (first[0::2], second[0::2]) if output_router_logits and self.returns_logits else None
        return SimpleNamespace(router_logits=outputs)


def _table_rows(*rows):
    return [{"prompt_input_ids": row[:1], "completion_input_ids": row[1:]} for row in rows]


@pytest.mark.parametrize("declared", [True, False], ids=["declared_routers", "forward_router_logits"])
def test_kl_rows_clear_twice_the_bf16_noise_at_every_scored_position(declared):
    rows = _table_rows(
        [WIDE, FLIP, WIDE],  # the bf16 oracle flips at a scored position
        [WIDE, TIGHT, WIDE],  # a margin past the noise but within twice it
        [WIDE, GROUPED, WIDE],  # the router's own pick trails an unpicked expert
        [WIDE, CLEAR, WIDE],
        [WIDE, WIDE, TIGHT],  # the near-tie sits on the last position, which no scored log-prob reads
        [WIDE, CLEAR, FLIP],
    )
    selected = routing_clear_rows(lambda dtype: _TableMoE(dtype, declared=declared), rows)
    assert selected.noise == pytest.approx(NOISE), "a flipped row's later routers counted as kernel noise"
    assert selected.cleared == (3, 4, 5) and selected.headroom == (4,), "CLEAR clears without headroom"
    # Row 4 leads by WIDE's whole margin; rows 3 and 5 clear by CLEAR's alike.
    assert not selected.clear and selected.rows == (4, 3), "with one row of headroom, the widest that clear fill in"


def test_kl_rows_take_twice_the_headroom_then_the_most_completion_tokens_then_the_wider_margin():
    rows = [
        *_table_rows([WIDE, ROOMY, WIDE], [WIDE, WIDE, WIDE]),
        # Four tokens, of which a KL check scores one.
        {"prompt_input_ids": [WIDE, WIDE, WIDE], "completion_input_ids": [WIDE]},
        *_table_rows([WIDE, ROOMY, ROOMY, WIDE], [WIDE, CLEAR, WIDE, WIDE]),
    ]
    selected = routing_clear_rows(_TableMoE, rows)
    assert selected.cleared == (0, 1, 2, 3, 4) and selected.headroom == (0, 1, 2, 3) and selected.clear
    assert selected.rows == (1, 2), "rows with twice the headroom come before a longer row at the floor, longest first"
    floor = routing_clear_rows(
        _TableMoE, _table_rows([WIDE, ROOMY, WIDE], [WIDE, AMPLE, WIDE], [WIDE, ROOMY, ROOMY, WIDE])
    )
    assert floor.rows == (2, 1), (
        "short of twice the headroom the longest row leads, then the wider margin among equals"
    )


@pytest.mark.parametrize(
    ("tokens", "picked_groups", "rows", "cleared", "chosen"),
    [
        (
            ONE_OF_TWO_BLOCKS,
            1,
            ([WIDE, LEADING_BLOCK, WIDE], [WIDE, NEAR_BLOCK, WIDE], [WIDE, WIDE, WIDE]),
            (0, 2),
            (2, 0),
        ),
        (TWO_OF_FOUR_BLOCKS, 2, ([WIDE, RIVAL_BLOCK, WIDE], [WIDE, WIDE, WIDE], [WIDE, WIDE, WIDE]), (1, 2), (1, 2)),
    ],
    ids=["one_of_two_blocks", "two_of_four_blocks"],
)
def test_kl_rows_of_a_group_limited_router_hold_its_blocks_and_their_experts(
    tokens, picked_groups, rows, cleared, chosen
):
    groups = len(next(iter(tokens.values()))[0]) // 2
    selected = routing_clear_rows(
        partial(_TableMoE, tokens=tokens, top_k=2, n_group=groups, topk_group=picked_groups), _table_rows(*rows)
    )
    assert selected.cleared == selected.headroom == cleared and selected.rows == chosen and selected.clear


@pytest.mark.parametrize(
    ("first", "cleared", "chosen"), [(GROUPED, (), (1, 2)), (ROOMY, (0,), (0, 1))], ids=["none_clears", "one_clears"]
)
def test_kl_rows_too_few_clear_fails_the_premise_and_fills_by_margin(first, cleared, chosen):
    rows = _table_rows([WIDE, first, WIDE], [WIDE, FLIP, WIDE], [WIDE, TIGHT, WIDE])
    selected = routing_clear_rows(_TableMoE, rows)
    assert not selected.clear and selected.cleared == selected.headroom == cleared
    assert selected.rows == chosen, "the clearing rows, then the widest-margin rest, fill the batch"


@pytest.mark.parametrize("noise_scales", [(1, 2), (2, 1)], ids=["louder_second", "louder_first"])
def test_the_routing_noise_is_the_largest_over_routers(noise_scales):
    selected = routing_clear_rows(
        partial(_TableMoE, noise_scales=noise_scales), _table_rows([WIDE, CLEAR, WIDE], [WIDE, WIDE, WIDE])
    )
    assert selected.noise == pytest.approx(2 * NOISE)
    assert selected.cleared == (1,), "CLEAR's margin is within twice the louder router's noise"


def _unloadable(dtype):
    raise AssertionError("only rank 0 loads the routing oracles")


def _agree_on_rank(rank: int, root: str) -> None:
    """Rank 1 holds no oracle and a dataset of its own: every rank must collate rank 0's choice."""
    rows = _table_rows([WIDE, TIGHT, WIDE], [WIDE, ROOMY, WIDE], [WIDE, WIDE, WIDE])
    selected = agree_kl_rows(SimpleNamespace(rank=rank), _TableMoE if rank == 0 else _unloadable, rows[rank:])
    (Path(root) / f"rank_{rank}.json").write_text(json.dumps(selected._asdict()))


def test_every_rank_collates_rank_zeros_kl_rows(tmp_path):
    run_gloo_ranks(_agree_on_rank, 2, str(tmp_path))
    first, second = (json.loads((tmp_path / f"rank_{rank}.json").read_text()) for rank in range(2))
    assert first == second and first["rows"] == [2, 1] and first["clear"]


def _tied_scans_on_rank(rank: int, root: str) -> None:
    """A tied tiny MoE scanned as saved, with its head doubled here, and through :func:`agree_kl_rows`."""
    policy = _tiny_policy(TIED_SEED, moe=True, tied=True)
    rng = random.Random(TIED_SEED)
    words = [OFFLINE_VOCAB[word] for word in OFFLINE_TEXT_WORDS]
    rows = [
        {
            "prompt_input_ids": rng.choices(words, k=rng.randint(1, 3)),
            "completion_input_ids": rng.choices(words, k=rng.randint(1, 2)),
        }
        for _ in range(TIED_ROWS)
    ]

    def load(dtype):
        return copy.deepcopy(policy).to(dtype).eval()

    def load_doubled(dtype):
        model = load(dtype)
        with torch.no_grad():
            model.get_output_embeddings().weight.mul_(2)
        return model

    scans = {
        "as_saved": routing_clear_rows(load, rows),
        "doubled": routing_clear_rows(load_doubled, rows),
        "compared": agree_kl_rows(SimpleNamespace(rank=rank), load, rows),
    }
    (Path(root) / "scans.json").write_text(json.dumps({name: scan._asdict() for name, scan in scans.items()}))


def test_kl_rows_of_a_tied_moe_clear_on_the_doubled_head_the_check_compares(tmp_path):
    """Doubling a tied head doubles the input embeddings, which moves the routers."""
    run_gloo_ranks(_tied_scans_on_rank, 1, str(tmp_path))
    scans = json.loads((tmp_path / "scans.json").read_text())
    moved = set(scans["as_saved"]["rows"]) - set(scans["doubled"]["headroom"])
    assert moved, "every row the scan picks on the policy as saved keeps its headroom under the doubled head"
    assert scans["compared"] == scans["doubled"], "agree_kl_rows did not scan the policy the KL check compares"


def test_the_sweep_batches_each_shard_from_its_own_start():
    # Shards of 3, 4 and 4 rows: the first one's last batch is short.
    assert sweep_batches(11, 2, 3) == [range(0, 2), range(2, 3), range(3, 5), range(5, 7), range(7, 9), range(9, 11)]
    assert sweep_batches(5, 2, 1) == [range(0, 2), range(2, 4), range(4, 5)]


def test_kl_rows_of_a_routed_model_without_router_logits_raise():
    with pytest.raises(ValueError, match="router_logits"):
        routing_clear_rows(
            lambda dtype: _TableMoE(dtype, declared=False, returns_logits=False), _table_rows([WIDE, CLEAR, WIDE])
        )


def _export(path, state):
    path.mkdir()
    model = _tiny_policy(7).to(torch.bfloat16)
    model.load_state_dict(state)
    model.save_pretrained(path)
    return path


def test_the_resumed_export_verdict_reads_a_move_at_the_export_dtype(tmp_path):
    state = {name: tensor.detach().clone() for name, tensor in _tiny_policy(7).to(torch.bfloat16).state_dict().items()}
    continuous = _export(tmp_path / "continuous", state)
    resumed = _export(tmp_path / "resumed", state)
    # The checkpoint stores fp32 masters a bf16 round trip loses; at the export dtype nothing moved.
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    masters = {name: tensor.float() * (1 + 2**-10) for name, tensor in state.items()}
    assert all(torch.equal(masters[name].bfloat16(), tensor) for name, tensor in state.items())
    assert any(not torch.equal(master, master.bfloat16().float()) for master in masters.values())
    save_file(masters, str(checkpoint / "model.safetensors"))
    assert resumed_export_verdict(continuous, resumed, checkpoint, "cpu") == (True, False, True)

    moved = dict(state)
    moved["lm_head.weight"] = state["lm_head.weight"] * 2
    diverged = _export(tmp_path / "diverged", moved)
    assert resumed_export_verdict(continuous, diverged, checkpoint, "cpu") == (False, True, True)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
