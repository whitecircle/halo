#!/usr/bin/env python
"""The toolkit's sequence-length contract (``agent-docs/reference/configuration-reference.md``): the context-window
resolver, SMPO's budget shares, and the writers of ``tokenizer.model_max_length``."""

from types import SimpleNamespace

import pytest
from transformers import PretrainedConfig

from halo.configs.grpo_offline import OfflineGRPOConfig
from halo.configs.parser import HaloArgumentParser
from halo.configs.preference import SMPOConfig
from halo.entrypoints.model_setup import apply_max_length, apply_prompt_completion_window, apply_tokenizer_window
from halo.models.context_window import resolve_length_to_context, resolve_model_context_window
from halo.trainers.grpo.environmental.trainer import DistributedAsyncEnvironmentalGRPOTrainer
from halo.trainers.grpo.offline.offline_rules import resolve_pp_max_length

_CPU_OK = {"output_dir": "/tmp/test_length_budget", "bf16": False, "use_cpu": True}

# Well past HF's ~1e9 unset-sentinel guard, and distinct from every window asserted below.
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
    assert resolve_model_context_window(_model(_TextConfig(max_position_embeddings=8192)), _tokenizer()) == 8192


def test_context_window_from_composite_vlm_config():
    """Reading the top level of a composite config finds nothing and raises, leaving ``max_length: null`` unusable
    on every VLM."""
    config = _CompositeConfig(text_config={"max_position_embeddings": 131072})
    assert not hasattr(config, "max_position_embeddings"), "fixture must not carry a top-level window"
    assert resolve_model_context_window(_model(config), _tokenizer()) == 131072


def test_context_window_rejects_tokenizer_unset_sentinel():
    """No window anywhere is a hard error, never a ~1e9 sentinel silently used as the cap."""

    class _NoWindow(PretrainedConfig):
        model_type = "halo_test_no_window"

    with pytest.raises(ValueError, match="context window"):
        resolve_model_context_window(_model(_NoWindow()), _tokenizer())


def test_context_window_falls_back_to_tokenizer():
    class _NoWindow(PretrainedConfig):
        model_type = "halo_test_no_window"

    assert resolve_model_context_window(_model(_NoWindow()), _tokenizer(model_max_length=2048)) == 2048


@pytest.mark.parametrize("unset", [None, 0, -1])
def test_resolve_length_to_context_treats_non_positive_as_unset(unset):
    """A 0 slipping through as a literal cap would tokenize every row to nothing."""
    resolved = resolve_length_to_context(unset, _model(_TextConfig(max_position_embeddings=8192)), _tokenizer())
    assert resolved == 8192


def test_resolve_length_to_context_keeps_explicit_value():
    assert resolve_length_to_context(512, _model(_TextConfig(max_position_embeddings=8192)), _tokenizer()) == 512


def _setup_tokenizer() -> SimpleNamespace:
    """A tokenizer whose special tokens are settled, so ``configure_run_tokenizer`` changes only
    ``model_max_length``. ``pad_token_id`` comes with ``pad_token``, as on any real tokenizer."""
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
    """The cap must land on both the config and ``tokenizer.model_max_length``, or the two disagree about the run's
    budget."""
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
    """Returns ``(window, tokenizer)``, the tokenizer being the one the seam hands back; dropping it makes
    ``tokenizer_backend`` a silent no-op on every GRPO script."""
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
    window, tokenizer = _apply_window(512, 256)
    assert window == 768
    assert tokenizer.model_max_length == 768


@pytest.mark.parametrize(("max_prompt_length", "max_completion_length"), [(None, 256), (512, None), (None, None)])
def test_prompt_completion_window_unset_when_either_half_is_unbounded(max_prompt_length, max_completion_length):
    """HF resolves every ``truncation=True, max_length=None`` call against ``model_max_length``, so pinning one half
    silently caps the unbounded half: offline GRPO's defaults (prompt 512, completion null) would truncate every
    stored completion at 512."""
    window, tokenizer = _apply_window(max_prompt_length, max_completion_length)
    assert window is None
    assert tokenizer.model_max_length == _UNSET_SENTINEL, "a partial budget was pinned as the window"


@pytest.mark.parametrize("unset", [0, -1])
def test_prompt_completion_window_treats_non_positive_shares_as_unset(unset):
    """A ``0`` prompt budget is no budget (``ids[-0:]`` is the whole list), so it is not summed into a window."""
    window, tokenizer = _apply_window(unset, 256)
    assert window is None
    assert tokenizer.model_max_length == _UNSET_SENTINEL


def test_prompt_completion_window_requires_an_online_generation_budget():
    """Without this raise, a null ``max_completion_length`` surfaces phases later as a bare ``TypeError`` inside the
    vLLM context-window check."""
    with pytest.raises(ValueError, match="generation budget"):
        _apply_window(512, None, completion_budget_required=True)


def test_prompt_completion_window_allows_an_unbounded_offline_completion_cap():
    """Offline GRPO's completion knob is a truncation cap and null means no cap, so the raise above must not fire
    there."""
    window, _ = _apply_window(512, None)
    assert window is None


def test_prompt_completion_window_resolves_the_tokenizer_backend():
    """This seam is the GRPO family's only ``resolve_tokenizer_backend`` call. An unknown backend is the cheap probe
    that resolution ran: it needs no optional package."""
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


def test_env_grpo_pins_the_models_resolved_context_window():
    """The window env GRPO's trainer fails a trajectory against, so it must be the model's own, read through the
    composite-config walk."""
    model = _model(_CompositeConfig(text_config={"max_position_embeddings": 131072}))
    tokenizer = _setup_tokenizer()

    returned = apply_tokenizer_window(_setup_args(), model, tokenizer, resolve_model_context_window(model, tokenizer))

    assert returned is tokenizer, "apply_tokenizer_window must return the backend-resolved tokenizer"
    assert tokenizer.model_max_length == 131072


def test_env_grpo_context_limit_uses_the_shared_resolver():
    """env-GRPO reads the tokenizer first and delegates every other case, so it inherits the composite-config walk
    and the other spellings."""

    class _NPositionsConfig(PretrainedConfig):
        model_type = "halo_test_n_positions"

        def __init__(self, **kwargs):
            self.n_positions = 3072
            super().__init__(**kwargs)

    host = SimpleNamespace(_tokenizer=_tokenizer(), model=_model(_NPositionsConfig()))
    limit = DistributedAsyncEnvironmentalGRPOTrainer._context_limit.__get__(host)

    assert limit() == resolve_model_context_window(host.model, host._tokenizer) == 3072


def test_smpo_budget_defaults_split_max_length_in_half():
    assert SMPOConfig(**_CPU_OK).resolve_length_budget() == (1024, 512, 512)


def test_smpo_budget_prompt_share_scales_with_max_length():
    """A fixed 512 prompt default would silently truncate prompts on a run whose max_length resolves to a long
    context window."""
    cfg = SMPOConfig(max_length=131072, **_CPU_OK)
    assert cfg.resolve_length_budget() == (131072, 65536, 65536)


def test_smpo_budget_explicit_shares_are_honoured():
    cfg = SMPOConfig(max_length=4096, max_prompt_length=1024, max_completion_length=512, **_CPU_OK)
    assert cfg.resolve_length_budget() == (4096, 1024, 512)


def test_smpo_budget_completion_takes_the_remainder():
    cfg = SMPOConfig(max_length=4096, max_prompt_length=3072, **_CPU_OK)
    assert cfg.resolve_length_budget() == (4096, 3072, 1024)


def test_smpo_budget_rejects_overflowing_split():
    """Prompt and completion truncate independently, so shares summing past max_length would exceed the budget
    without either cut firing."""
    cfg = SMPOConfig(max_length=4096, max_prompt_length=2048, max_completion_length=4096, **_CPU_OK)
    with pytest.raises(ValueError, match="overflows max_length"):
        cfg.resolve_length_budget()


def test_smpo_budget_rejects_prompt_share_exceeding_max_length():
    cfg = SMPOConfig(max_length=1024, max_prompt_length=1024, **_CPU_OK)
    with pytest.raises(ValueError, match="no room"):
        cfg.resolve_length_budget()


@pytest.mark.parametrize("unresolved", [None, 0, -1])
def test_smpo_budget_requires_a_resolved_max_length(unresolved):
    """A null max_length must be resolved by the script before the trainer; reaching resolve_length_budget()
    unresolved is a construction error."""
    cfg = SMPOConfig(max_length=unresolved, **_CPU_OK)
    with pytest.raises(ValueError, match="must be a positive int"):
        cfg.resolve_length_budget()


def test_offline_grpo_max_length_is_no_constructor_argument():
    with pytest.raises(TypeError, match="max_length"):
        OfflineGRPOConfig(output_dir="/tmp/test_length_budget", bf16=False, use_cpu=True, max_length=4096)


@pytest.mark.parametrize("parallelism_config", [None, SimpleNamespace(is_pp_mode=False)])
def test_offline_grpo_row_length_is_unset_off_pipeline_parallelism(parallelism_config):
    """The window off PP is the two caps the tokenization applies, so a second trainer on the same args must not read
    the first one's fixed shape."""
    args = SimpleNamespace(max_prompt_length=512, max_completion_length=1024, max_length=1536)
    assert resolve_pp_max_length(args, parallelism_config) is None


def test_legacy_max_seq_length_yaml_is_refused_by_the_parser(tmp_path):
    """TRL's retired ``max_seq_length`` is refused, since it would silently cap nothing here."""
    path = tmp_path / "offline_grpo.yaml"
    path.write_text("output_dir: /tmp/test_length_budget\nbf16: false\nuse_cpu: true\nmax_seq_length: 4096\n")
    with pytest.raises(ValueError, match="max_seq_length"):
        HaloArgumentParser((OfflineGRPOConfig,)).parse_yaml_file(str(path))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
