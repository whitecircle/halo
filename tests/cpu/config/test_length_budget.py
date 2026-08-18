#!/usr/bin/env python
"""Tests for the toolkit's sequence-length contract (``agent-docs/reference/configuration-reference.md``).

The properties pinned here:

  * ``get_model_context_window`` reads the window off ``config.get_text_config()``, so a composite
    (VLM) config — which carries ``max_position_embeddings`` on its nested ``text_config`` only —
    resolves instead of falling through to the tokenizer's unset sentinel and raising, and it is the
    single resolver (env-GRPO's ``_context_limit`` delegates to it rather than keeping a copy);
  * ``SmoothMarginPOConfig.resolve_length_budget`` derives the prompt/completion shares from
    ``max_length`` so they scale with a context-resolved budget, and rejects a split that overflows
    it (prompt and completion truncate independently, so an over-budget split silently exceeds
    ``max_length``);
  * ``resolve_length_to_context`` treats null AND non-positive alike as "use the model's own limit";
  * ``apply_max_length`` (one ``max_length``) and ``apply_prompt_completion_window`` (the GRPO
    family's two budgets) are the only two writers of ``tokenizer.model_max_length``, and the latter
    refuses to pin a partial sum — which HF would turn into a silent cap on the unbounded half.

Run: pytest tests/cpu/config/test_length_budget.py
"""

import inspect
from types import SimpleNamespace

import pytest
from accelerate import PartialState
from transformers import PretrainedConfig

from src.configs.offline_grpo_config import OfflineGRPOConfig
from src.configs.smpo_config import SmoothMarginPOConfig
from src.models.loading.tokenizer_setup import get_model_context_window, resolve_length_to_context
from src.trainers.grpo.environmental import DistributedAsyncEnvironmentalGRPOTrainer
from src.trainers.grpo.offline import OfflineGRPOTrainer
from src.training.parser import H4ArgumentParser
from src.training.script_runner import apply_max_length, apply_prompt_completion_window

# resolve_length_to_context logs through accelerate's rank-aware logger, which raises until the
# state exists. Production always has it (init_training_script runs first); mirror that here.
PartialState()

_CPU_OK = {"output_dir": "/tmp/test_length_budget", "bf16": False, "use_cpu": True}

# Well past HF's ~1e9 "unset" sentinel guard, and distinct from every window asserted below.
_UNSET_SENTINEL = int(1e30)


class _TextConfig(PretrainedConfig):
    model_type = "halo_test_text"

    def __init__(self, max_position_embeddings: int = 4096, **kwargs):
        self.max_position_embeddings = max_position_embeddings
        super().__init__(**kwargs)


class _CompositeConfig(PretrainedConfig):
    """A VLM-shaped config: the position budget lives on ``text_config``, never at the top level."""

    model_type = "halo_test_composite"
    sub_configs = {"text_config": _TextConfig}

    def __init__(self, text_config=None, **kwargs):
        self.text_config = text_config if isinstance(text_config, _TextConfig) else _TextConfig(**(text_config or {}))
        super().__init__(**kwargs)


def _model(config) -> SimpleNamespace:
    return SimpleNamespace(config=config)


def _tokenizer(model_max_length=_UNSET_SENTINEL) -> SimpleNamespace:
    return SimpleNamespace(model_max_length=model_max_length)


def test_context_window_from_plain_config():
    assert get_model_context_window(_model(_TextConfig(max_position_embeddings=8192)), _tokenizer()) == 8192


def test_context_window_from_composite_vlm_config():
    """The regression this guards: reading ``max_position_embeddings`` off the TOP level of a
    composite config finds nothing, falls through to the tokenizer's unset sentinel, and raises —
    so ``max_length: null`` was unusable on every VLM."""
    config = _CompositeConfig(text_config={"max_position_embeddings": 131072})
    assert not hasattr(config, "max_position_embeddings"), "fixture must not carry a top-level window"
    assert get_model_context_window(_model(config), _tokenizer()) == 131072


def test_context_window_rejects_tokenizer_unset_sentinel():
    """No window anywhere is a hard error, never a ~1e9 sentinel silently used as the cap."""

    class _NoWindow(PretrainedConfig):
        model_type = "halo_test_no_window"

    with pytest.raises(ValueError, match="context window"):
        get_model_context_window(_model(_NoWindow()), _tokenizer())


def test_context_window_falls_back_to_tokenizer():
    class _NoWindow(PretrainedConfig):
        model_type = "halo_test_no_window"

    assert get_model_context_window(_model(_NoWindow()), _tokenizer(model_max_length=2048)) == 2048


@pytest.mark.parametrize("unset", [None, 0, -1])
def test_resolve_length_to_context_treats_non_positive_as_unset(unset):
    """`null` and any non-positive value both mean "use the model's own limit" — a 0 slipping
    through as a literal cap would tokenize every row to nothing."""
    resolved = resolve_length_to_context(unset, _model(_TextConfig(max_position_embeddings=8192)), _tokenizer())
    assert resolved == 8192


def test_resolve_length_to_context_keeps_explicit_value():
    assert resolve_length_to_context(512, _model(_TextConfig(max_position_embeddings=8192)), _tokenizer()) == 512


def _setup_tokenizer() -> SimpleNamespace:
    """A tokenizer whose special tokens are already settled, so ``setup_model_and_tokenizer``
    changes nothing but ``model_max_length``.

    ``pad_token_id`` accompanies ``pad_token`` because no real tokenizer can carry one without the
    other — ``PreTrainedTokenizerBase.__getattr__`` resolves every ``<special>_token_id`` — and the
    setup seam records that id on the model whatever settled it.
    """
    return SimpleNamespace(
        model_max_length=_UNSET_SENTINEL,
        eos_token=None,
        eos_token_id=None,
        bos_token=None,
        pad_token="<pad>",
        pad_token_id=0,
        chat_template="{}",
    )


def _setup_args() -> SimpleNamespace:
    return SimpleNamespace(
        eos_token=None,
        bos_token=None,
        pad_token=None,
        chat_template=None,
        force_chat_template=False,
        added_special_tokens=None,
        tokenizer_backend="hf",
    )


def test_apply_max_length_writes_back_and_pins_the_tokenizer():
    """The seam every offline script uses: the resolved cap must land on BOTH the config (what the
    collators / length filters / packers read) and ``tokenizer.model_max_length`` — a write to only
    one of them leaves the two disagreeing about the run's budget."""
    config = SimpleNamespace(max_length=None)
    tokenizer = _setup_tokenizer()

    returned = apply_max_length(config, _setup_args(), _model(_TextConfig(max_position_embeddings=8192)), tokenizer)

    assert returned is tokenizer, "apply_max_length must return the backend-resolved tokenizer"
    assert config.max_length == 8192, "resolved cap was not written back to the config"
    assert tokenizer.model_max_length == 8192, "resolved cap was not pinned on the tokenizer"


def test_apply_max_length_keeps_an_explicit_cap():
    config = SimpleNamespace(max_length=4096)
    tokenizer = _setup_tokenizer()

    apply_max_length(config, _setup_args(), _model(_TextConfig(max_position_embeddings=131072)), tokenizer)

    assert config.max_length == 4096
    assert tokenizer.model_max_length == 4096


def _apply_window(max_prompt_length, max_completion_length, **kwargs):
    """Returns ``(window, tokenizer)`` — the tokenizer being the one the seam HANDS BACK, which is
    what callers must use. Unpacking the pair is itself the return-contract gate: dropping the
    tokenizer from the seam makes ``tokenizer_backend`` a silent no-op on every GRPO script."""
    tokenizer, window = apply_prompt_completion_window(
        _setup_args(),
        _model(_TextConfig(max_position_embeddings=131072)),
        _setup_tokenizer(),
        max_prompt_length=max_prompt_length,
        max_completion_length=max_completion_length,
        **kwargs,
    )
    return window, tokenizer


def test_prompt_completion_window_pins_the_sum_of_both_budgets():
    """The GRPO-family seam: no single ``max_length`` exists, so the window is prompt + completion."""
    window, tokenizer = _apply_window(512, 256)
    assert window == 768
    assert tokenizer.model_max_length == 768


@pytest.mark.parametrize(("max_prompt_length", "max_completion_length"), [(None, 256), (512, None), (None, None)])
def test_prompt_completion_window_unset_when_either_half_is_unbounded(max_prompt_length, max_completion_length):
    """A partial sum is not a window. HF resolves every ``truncation=True, max_length=None`` call
    against ``model_max_length``, so pinning one half's budget silently caps the *other*, unbounded
    half there — offline GRPO's shipped defaults (prompt 512, completion null) would truncate every
    stored completion at 512 while documenting "no cap". An unbounded half must leave the tokenizer
    alone."""
    window, tokenizer = _apply_window(max_prompt_length, max_completion_length)
    assert window is None
    assert tokenizer.model_max_length == _UNSET_SENTINEL, "a partial budget was pinned as the window"


@pytest.mark.parametrize("unset", [0, -1])
def test_prompt_completion_window_treats_non_positive_shares_as_unset(unset):
    """One boundedness predicate across the whole contract: a ``0`` prompt budget is not a budget
    (``ids[-0:]`` is the whole list), so it must not be summed into a window as if it were."""
    window, tokenizer = _apply_window(unset, 256)
    assert window is None
    assert tokenizer.model_max_length == _UNSET_SENTINEL


def test_prompt_completion_window_requires_an_online_generation_budget():
    """A policy that must stop generating has no unbounded budget. Without this raise, a null
    ``max_completion_length`` slips through the tokenizer pin and surfaces phases later as a bare
    ``TypeError`` inside the vLLM context-window check."""
    with pytest.raises(ValueError, match="generation budget"):
        _apply_window(512, None, completion_budget_required=True)


def test_prompt_completion_window_allows_an_unbounded_offline_completion_cap():
    """The same seam serves offline GRPO, where the completion knob is a truncation cap and `null`
    legitimately means "no cap" — the raise above must not fire there."""
    window, _ = _apply_window(512, None)
    assert window is None


def test_prompt_completion_window_resolves_the_tokenizer_backend():
    """This seam is the GRPO family's ONLY ``resolve_tokenizer_backend`` call, so it must make one.
    A seam that pinned the window but skipped resolution would make ``--tokenizer_backend`` a silent
    no-op on offline GRPO and RLVR — invisible except as lost tokenization throughput.
    An unknown backend is the cheap probe for "resolution ran": it needs no optional package."""
    args = _setup_args()
    args.tokenizer_backend = "not-a-backend"

    with pytest.raises(ValueError, match="tokenizer_backend"):
        apply_prompt_completion_window(
            args,
            _model(_TextConfig(max_position_embeddings=131072)),
            _setup_tokenizer(),
            max_prompt_length=512,
            max_completion_length=256,
        )


def test_env_grpo_context_limit_uses_the_shared_resolver():
    """One context-window resolver, not two. env-GRPO reads the tokenizer first (the script pins the
    *served* window there) and delegates every other case, so it inherits the composite-config walk
    and the ``max_seq_length`` / ``n_positions`` spellings instead of a narrower private copy."""

    class _NPositionsConfig(PretrainedConfig):
        model_type = "halo_test_n_positions"

        def __init__(self, **kwargs):
            self.n_positions = 3072
            super().__init__(**kwargs)

    host = SimpleNamespace(_tokenizer=_tokenizer(), model=_model(_NPositionsConfig()))
    limit = DistributedAsyncEnvironmentalGRPOTrainer._context_limit.__get__(host)

    assert limit() == get_model_context_window(host.model, host._tokenizer) == 3072


def test_smpo_budget_defaults_split_max_length_in_half():
    """Shipped defaults are unchanged by the null-share derivation (1024 → 512 / 512)."""
    assert SmoothMarginPOConfig(**_CPU_OK).resolve_length_budget() == (1024, 512, 512)


def test_smpo_budget_prompt_share_scales_with_max_length():
    """The footgun this closes: a fixed 512 prompt default silently truncated prompts to 512 on a
    run whose max_length was resolved to a long context window."""
    cfg = SmoothMarginPOConfig(max_length=131072, **_CPU_OK)
    assert cfg.resolve_length_budget() == (131072, 65536, 65536)


def test_smpo_budget_explicit_shares_are_honoured():
    cfg = SmoothMarginPOConfig(max_length=4096, max_prompt_length=1024, max_completion_length=512, **_CPU_OK)
    assert cfg.resolve_length_budget() == (4096, 1024, 512)


def test_smpo_budget_completion_takes_the_remainder():
    cfg = SmoothMarginPOConfig(max_length=4096, max_prompt_length=3072, **_CPU_OK)
    assert cfg.resolve_length_budget() == (4096, 3072, 1024)


def test_smpo_budget_rejects_overflowing_split():
    """Prompt and completion truncate independently, so shares summing past max_length would build
    sequences longer than the stated budget without either cut firing."""
    cfg = SmoothMarginPOConfig(max_length=4096, max_prompt_length=2048, max_completion_length=4096, **_CPU_OK)
    with pytest.raises(ValueError, match="overflows max_length"):
        cfg.resolve_length_budget()


def test_smpo_budget_rejects_prompt_share_exceeding_max_length():
    cfg = SmoothMarginPOConfig(max_length=1024, max_prompt_length=1024, **_CPU_OK)
    with pytest.raises(ValueError, match="no room"):
        cfg.resolve_length_budget()


@pytest.mark.parametrize("unresolved", [None, 0, -1])
def test_smpo_budget_requires_a_resolved_max_length(unresolved):
    """A null max_length must be resolved to the context window by the script BEFORE the trainer;
    reaching resolve_length_budget() unresolved is a construction error, not a silent default."""
    cfg = SmoothMarginPOConfig(max_length=unresolved, **_CPU_OK)
    with pytest.raises(ValueError, match="must be a positive int"):
        cfg.resolve_length_budget()


def _offline_config(**kwargs) -> OfflineGRPOConfig:
    return OfflineGRPOConfig(output_dir="/tmp/test_length_budget", bf16=False, use_cpu=True, **kwargs)


@pytest.mark.parametrize("parallelism_config", [None, SimpleNamespace(is_pp_mode=False)])
def test_offline_grpo_max_length_is_refused_off_pipeline_parallelism(parallelism_config):
    """Only the PP path reads ``max_length`` (the fixed shape P2P buffers freeze on). Off PP the
    budget is the two caps, so a set ``max_length`` reached nothing and the run trained uncapped."""
    with pytest.raises(ValueError, match="pipeline-parallel-only"):
        OfflineGRPOTrainer._reject_inert_max_length(_offline_config(max_length=4096), parallelism_config)


def test_offline_grpo_max_length_is_accepted_under_pipeline_parallelism():
    """Under PP it IS the knob — the guard must not fire on the one mode that honors it."""
    OfflineGRPOTrainer._reject_inert_max_length(_offline_config(max_length=4096), SimpleNamespace(is_pp_mode=True))


def test_offline_grpo_unset_max_length_is_the_normal_case():
    OfflineGRPOTrainer._reject_inert_max_length(_offline_config(), SimpleNamespace(is_pp_mode=False))


def test_offline_grpo_construction_runs_the_inert_max_length_guard():
    """The guard is only worth anything if ``__init__`` calls it — before every dataset map, so the
    rejection costs no tokenization."""
    assert "_reject_inert_max_length" in inspect.getsource(OfflineGRPOTrainer.__init__)


def test_legacy_max_seq_length_yaml_is_refused_by_the_parser(tmp_path):
    """A legacy YAML asking for a sequence cap under TRL's retired ``max_seq_length`` must be named
    and refused, not migrated onto this inert field where it would silently stop capping."""
    path = tmp_path / "offline_grpo.yaml"
    path.write_text("output_dir: /tmp/test_length_budget\nbf16: false\nuse_cpu: true\nmax_seq_length: 4096\n")
    with pytest.raises(ValueError, match="max_seq_length"):
        H4ArgumentParser((OfflineGRPOConfig,)).parse_yaml_file(str(path))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
