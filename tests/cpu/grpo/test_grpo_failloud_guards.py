#!/usr/bin/env python
"""Fail-loud guards in the GRPO trainers — each one covers a silent degradation.

- ``_context_limit`` (env-GRPO): when neither the tokenizer nor the model config carries a real
  context window, the trainer must RAISE instead of silently resolving to 10**9, which disables
  the trajectory-overflow check entirely.
- ``tokenize_offline_grpo_rows``: a row whose completions and rewards differ in length must
  raise with row context instead of ``zip`` silently truncating the pairing (and recording the
  pre-truncation ``group_size``).
- ``_resolve_rollout_stop_token_ids``: stop tokens that resolve to NOTHING must raise, not degrade
  to "the user configured no stop tokens" (which inverts the gpt-oss rollout shape).
- ``_extract_prompts_and_contexts``: a conversation with no user turn must record a batch error
  instead of shipping a Python ``repr`` of the message list to the environment as the task.
- ``OfflineGRPOTrainer``: an unknown ``loss_type`` / ``policy_gradient_formulation`` must be
  rejected at construction on EVERY path, not only under pipeline parallelism.

    python tests/cpu/grpo/test_grpo_failloud_guards.py
"""

import sys
import types

import pytest
from accelerate import PartialState

from src.trainers.grpo.environmental import DistributedAsyncEnvironmentalGRPOTrainer, batch_reward_std
from src.trainers.grpo.offline import OfflineGRPOTrainer, tokenize_offline_grpo_rows

PartialState()  # both trainers log through accelerate, which refuses to log without it

# --- _context_limit fail-loud (F3-F) ---

_UNSET_SENTINEL = int(1e30)  # HF tokenizers' "no limit" model_max_length


def _limit_host(model_max_length, config: types.SimpleNamespace):
    host = types.SimpleNamespace(
        _tokenizer=types.SimpleNamespace(model_max_length=model_max_length),
        model=types.SimpleNamespace(config=config),
    )
    return DistributedAsyncEnvironmentalGRPOTrainer._context_limit.__get__(host)


def test_context_limit_from_tokenizer():
    assert _limit_host(8192, types.SimpleNamespace())() == 8192


def test_context_limit_falls_back_to_model_config():
    limit = _limit_host(_UNSET_SENTINEL, types.SimpleNamespace(max_position_embeddings=4096))
    assert limit() == 4096


def test_context_limit_uses_text_config():
    text = types.SimpleNamespace(max_position_embeddings=2048)
    config = types.SimpleNamespace(get_text_config=lambda: text)
    assert _limit_host(None, config)() == 2048


def test_context_limit_raises_when_underivable():
    # Resolving to 10**9 instead would disable the trajectory-overflow check entirely.
    with pytest.raises(ValueError, match="context window"):
        _limit_host(_UNSET_SENTINEL, types.SimpleNamespace())()


# --- Offline GRPO completions/rewards pairing (F3-G) ---


class _OneTokenTokenizer:
    """Minimal stand-in: the pairing guard runs before any real tokenization matters."""

    bos_token_id = None
    eos_token_id = None

    def __call__(self, text, **kwargs):
        return {"input_ids": [1]}


_TOKENIZE_KWARGS = {
    "processing_class": _OneTokenTokenizer(),
    "max_prompt_length": None,
    "max_completion_length": None,
    "advantage_method": "z_norm",
    "best_completion_emphasis": 0.0,
    "is_encoder_decoder": False,
}


def test_mispaired_completions_and_rewards_raise_with_row_context():
    batch = {
        "prompt": ["q"],
        "completions": [["a", "b", "c"]],
        "rewards": [[1.0, 0.0]],  # 2 rewards for 3 completions: a bare zip trains 2 and reports 3
    }
    with pytest.raises(ValueError, match=r"row 7.*3 completions but[\s\S]*2 rewards"):
        tokenize_offline_grpo_rows(batch, [7], **_TOKENIZE_KWARGS)


def test_paired_row_tokenizes_with_true_group_size():
    batch = {"prompt": ["q"], "completions": [["a", "b"]], "rewards": [[1.0, 0.0]]}
    out = tokenize_offline_grpo_rows(batch, [0], **_TOKENIZE_KWARGS)
    assert len(out["advantage"]) == 2
    assert out["group_size"] == [2, 2]


# --- rollout stop tokens must resolve or raise (B-3) ---


class _StopTokenTokenizer:
    """Resolves only the names in ``known``; everything else comes back as ``unk_token_id``."""

    name_or_path = "stub/tokenizer"
    unk_token_id = 3

    def __init__(self, known: dict[str, int]):
        self._known = known

    def convert_tokens_to_ids(self, name):
        return self._known.get(name, self.unk_token_id)


def _stop_token_host(names, tokenizer):
    host = types.SimpleNamespace(
        async_config=types.SimpleNamespace(rollout_stop_tokens=names),
        _tokenizer=tokenizer,
    )
    return DistributedAsyncEnvironmentalGRPOTrainer._resolve_rollout_stop_token_ids.__get__(host)


def test_stop_tokens_that_all_fail_to_resolve_raise():
    # Returning None here is indistinguishable from "no stop tokens configured": a gpt-oss episode
    # would then play out in a single generation instead of stopping at <|call|>.
    resolve = _stop_token_host(["<|call|>", "<|nope|>"], _StopTokenTokenizer({}))
    with pytest.raises(ValueError, match=r"<\|call\|>"):
        resolve()


def test_partially_resolved_stop_tokens_keep_the_ones_that_worked():
    resolve = _stop_token_host(["<|call|>", "<|nope|>"], _StopTokenTokenizer({"<|call|>": 17}))
    assert resolve() == [17]


def test_no_stop_tokens_configured_stays_none():
    assert _stop_token_host([], _StopTokenTokenizer({}))() is None


# --- a conversation with no user turn is a batch error, not a repr (B-2) ---


def _prompt_host():
    return types.SimpleNamespace(_batch_build_error=None, _group_random_effort=False)


def test_conversation_without_a_user_turn_records_a_batch_error():
    host = _prompt_host()
    convo = [{"role": "system", "content": "you are a helpful assistant"}]
    prompts, _ = DistributedAsyncEnvironmentalGRPOTrainer._extract_prompts_and_contexts(host, [{"prompt": convo}])

    # A per-rank raise would strand DP peers in the next collective, so it is RECORDED.
    assert host._batch_build_error is not None
    assert "system" in host._batch_build_error
    # And the environment must never be handed a Python repr of the message list as its task.
    assert "role" not in prompts[0] and "'content'" not in prompts[0]


def test_a_prompt_with_no_user_turn_raises_before_any_rollout_is_submitted():
    """The recorded error must be raised BEFORE the environment is handed the batch.

    Raised only after the rollout, the empty task burns a full generation round, and an environment
    that rejects an empty task raises inside the Ray actor first — replacing a clean config error with
    an actor traceback.
    """
    import torch

    class _NoRollouts:
        def collect_rollouts(self, *args, **kwargs):
            raise AssertionError("the rollout must not be submitted for an unusable prompt batch")

    # A real (un-inited) trainer so the method resolution under test is the real one.
    host = object.__new__(DistributedAsyncEnvironmentalGRPOTrainer)
    host._batch_build_error = None
    host._group_random_effort = False
    host.accelerator = types.SimpleNamespace(device=torch.device("cpu"))
    host.model = types.SimpleNamespace(training=True)
    host._prefetch_enabled = False
    host._rollout_manager = _NoRollouts()
    convo = [{"role": "system", "content": "sys only"}]

    with pytest.raises(ValueError, match="no 'user' message"):
        host._generate_and_score_completions_base([{"prompt": convo}])


def test_last_user_turn_is_the_task_text():
    host = _prompt_host()
    convo = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "ack"},
        {"role": "user", "content": "second"},
    ]
    prompts, contexts = DistributedAsyncEnvironmentalGRPOTrainer._extract_prompts_and_contexts(
        host, [{"prompt": convo, "answer": "42"}]
    )
    assert prompts == ["second"]
    assert contexts == [{"answer": "42"}]
    assert host._batch_build_error is None


# --- reward_std on a 1-element gathered batch (B-18) ---


def test_reward_std_of_a_single_rollout_is_finite():
    import math

    import torch

    # torch's .std() is correction=1, so an unguarded call on one reward is NaN and, once appended,
    # NaNs the mean of the whole logging window.
    assert batch_reward_std(torch.tensor([1.5])) == 0.0
    assert math.isfinite(batch_reward_std(torch.tensor([1.5])))
    assert batch_reward_std(torch.tensor([0.0, 2.0])) == pytest.approx(math.sqrt(2.0))


# --- offline loss-type / PG-formulation validated off-PP too (B-10) ---


def _offline_args(**overrides):
    args = types.SimpleNamespace(
        padding_value=0,
        max_prompt_length=64,
        max_completion_length=64,
        advantage_method="z_norm",
        best_completion_emphasis=0.0,
        min_log_prob=None,
        loss_type="grpo",
        policy_gradient_formulation="prob_weighted",
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def _construct_offline(**overrides):
    """Drive ``__init__`` far enough to reach the dispatch-string gate (it precedes model loading)."""
    host = object.__new__(OfflineGRPOTrainer)
    OfflineGRPOTrainer.__init__(
        host,
        model=None,
        args=_offline_args(**overrides),
        processing_class=types.SimpleNamespace(pad_token_id=0),
        parallelism_config=None,
    )


def test_unknown_loss_type_is_rejected_at_construction_without_pp():
    # Catching this inside compute_loss instead would spend the model/dataset/optimizer build first.
    with pytest.raises(ValueError, match="Unknown loss type"):
        _construct_offline(loss_type="nonsense")


def test_unknown_pg_formulation_is_rejected_at_construction_without_pp():
    with pytest.raises(ValueError, match="policy_gradient_formulation"):
        _construct_offline(policy_gradient_formulation="nonsense")


def test_pp_normalizer_refuses_an_unknown_loss_type():
    # Without a terminal branch an unknown loss type silently takes the dr_grpo denominator.
    import torch

    host = types.SimpleNamespace(loss_type="nonsense", max_completion_length=64)
    normalizer = OfflineGRPOTrainer._pp_normalizer.__get__(host)
    with pytest.raises(ValueError, match="Unknown loss type"):
        normalizer({"group_size": torch.tensor([2.0, 2.0]), "labels": torch.zeros(2, 4, dtype=torch.long)})


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
