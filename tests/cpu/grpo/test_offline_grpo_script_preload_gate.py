"""The offline GRPO script refuses splits a run-start reference sweep cannot use before the model load.

The trainer refuses a pre-sharded dataset (and every other split its sweep cannot read) at
construction, which follows the policy load: minutes on a large model, on every rank. The script
loads its raw splits first and runs the same refusal before ``load_script_model``.

    python tests/cpu/grpo/test_offline_grpo_script_preload_gate.py
"""

from types import SimpleNamespace

import pytest
from accelerate import PartialState
from datasets import DatasetDict
from trl import ModelConfig

import scripts.training.offline_grpo as offline_script
from src.configs.offline_grpo_config import OfflineGRPOConfig
from src.distributed.parallelism_config import ParallelismConfig
from tests.common.offline_grpo import offline_grpo_dataset

PartialState(cpu=True)


class _ModelLoadReached(Exception):
    """Raised by the stand-in model load: the script got past every pre-load gate."""


def _run_script(monkeypatch, tmp_path, *, presharded: bool, kl_beta: float, use_peft: bool = False) -> None:
    config = OfflineGRPOConfig(
        output_dir=str(tmp_path), kl_beta=kl_beta, use_cpu=True, bf16=False, use_liger_kernel=False, report_to="none"
    )
    model_config = ModelConfig(model_name_or_path="local-policy", use_peft=use_peft)
    parsed = (SimpleNamespace(), config, model_config, SimpleNamespace(reset_sinks=True))
    splits = DatasetDict(train=offline_grpo_dataset(2), test=offline_grpo_dataset(2, offset=2))

    def model_load(*_args, **_kwargs):
        raise _ModelLoadReached

    monkeypatch.setattr(offline_script, "H4ArgumentParser", lambda *_: SimpleNamespace(parse=lambda: parsed))
    monkeypatch.setattr(
        offline_script, "init_training_script", lambda *a, **k: SimpleNamespace(parallelism_config=ParallelismConfig())
    )
    monkeypatch.setattr(offline_script, "load_script_datasets", lambda *a, **k: (splits, presharded))
    monkeypatch.setattr(offline_script, "padded_workload_attn_implementation", lambda *a, **k: "sdpa")
    monkeypatch.setattr(offline_script, "load_script_model", model_load)
    offline_script.main()


def test_a_presharded_run_start_sweep_is_refused_before_the_model_load(monkeypatch, tmp_path):
    with pytest.raises(ValueError, match="pre-sharded dataset"):
        _run_script(monkeypatch, tmp_path, presharded=True, kl_beta=0.2)


@pytest.mark.parametrize(
    ("kl_beta", "use_peft"), [(0.0, False), (0.2, True)], ids=["no_kl_reference", "adapters_off_reference"]
)
def test_runs_without_a_run_start_sweep_load_a_presharded_dataset(monkeypatch, tmp_path, kl_beta, use_peft):
    """Anti-over-refusal: no KL term, or a PEFT policy scored with its adapters off, sweeps nothing."""
    with pytest.raises(_ModelLoadReached):
        _run_script(monkeypatch, tmp_path, presharded=True, kl_beta=kl_beta, use_peft=use_peft)


def test_an_unsharded_run_start_sweep_reaches_the_model_load(monkeypatch, tmp_path):
    with pytest.raises(_ModelLoadReached):
        _run_script(monkeypatch, tmp_path, presharded=False, kl_beta=0.2)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
