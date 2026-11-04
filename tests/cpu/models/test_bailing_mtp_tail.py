#!/usr/bin/env python
"""Ling-3.0's remote code builds its MTP tail as live decoder layers: an idle one warns, a weighted one trains the
objective the docs state, and the documented override builds none and leaves the trunk's logits alone."""

import logging
import shutil

import pytest
import torch
import yaml
from safetensors.torch import load_file

from halo.checkpoint.tools.offline_export import save_full_checkpoint
from halo.model_loading import finalization
from halo.model_loading.finalization import finalize_loaded_model
from tests.common.models import BAILING_LING_3_TINY, QWEN3_0_6B, TINY_BAILING_LING3_MLA_CONFIG
from tests.common.mtp_heads import moved_keys, run_entry, saved_tensors, write_entry_corpus
from tests.common.tiny_models import TINY_BAILING_LING3_MLA_FAMILY, _tiny_remote, build_tiny_family_checkpoint
from tests.common.tokenizers import load_cached_tokenizer

_LAYERS = TINY_BAILING_LING3_MLA_CONFIG["num_hidden_layers"]


def _tiny_ling3(**overrides):
    torch.manual_seed(0)
    model = _tiny_remote(BAILING_LING_3_TINY, TINY_BAILING_LING3_MLA_CONFIG)(
        {"_attn_implementation": "sdpa", **overrides}
    ).eval()
    finalization._warned_idle_tails.discard(type(model).__name__)
    return model


def _idle_warnings(model, caplog) -> list[str]:
    with caplog.at_level(logging.WARNING, logger=finalization.__name__):
        finalize_loaded_model(model)
    return [record.getMessage() for record in caplog.records if "built its MTP tail" in record.getMessage()]


def _batch() -> dict[str, torch.Tensor]:
    ids = torch.randint(2, 200, (1, 16), generator=torch.Generator().manual_seed(1))
    return {"input_ids": ids, "labels": ids.clone()}


def test_an_idle_tail_warns_with_its_parameter_count_and_the_override(caplog):
    model = _tiny_ling3(num_nextn_predict_layers=1, mtp_loss_scaling_factor=0)
    warned = _idle_warnings(model, caplog)
    tail = sum(param.numel() for layer in model.model.layers[_LAYERS:] for param in layer.parameters())
    assert tail > 0, "the release's tail was not built, so nothing here ran"
    assert len(warned) == 1 and f"({tail} parameters)" in warned[0], warned
    assert "num_nextn_predict_layers: 0" in warned[0]


@pytest.mark.parametrize(
    "overrides",
    [
        {"num_nextn_predict_layers": 1, "mtp_loss_scaling_factor": 0.3},
        {"num_nextn_predict_layers": 0, "mtp_loss_scaling_factor": 0},
    ],
    ids=["weighted", "built_none"],
)
def test_a_weighted_or_absent_tail_stays_silent(overrides, caplog):
    assert not _idle_warnings(_tiny_ling3(**overrides), caplog)


def test_the_warning_fires_once_per_class(caplog):
    model = _tiny_ling3(num_nextn_predict_layers=1, mtp_loss_scaling_factor=0)
    assert len(_idle_warnings(model, caplog)) == 1
    caplog.clear()
    assert not _idle_warnings(model, caplog)


def test_only_the_main_rank_warns(caplog, monkeypatch):
    monkeypatch.setattr(finalization, "is_global_main_process", lambda: False)
    assert not _idle_warnings(_tiny_ling3(num_nextn_predict_layers=1, mtp_loss_scaling_factor=0), caplog)


def test_the_remote_loss_is_the_main_loss_plus_the_weighted_term_on_one_row():
    """``main + factor * mtp``: the route the docs give Ling-3.0-flash. Fails if the remote code's term drifts."""
    model = _tiny_ling3(num_nextn_predict_layers=1, mtp_loss_scaling_factor=0)
    finalize_loaded_model(model)
    batch = _batch()
    with torch.no_grad():
        off = model(**batch)
        model.mtp_loss_scaling_factor = 0.3
        weighted = model(**batch).loss
    assert off.mtp_loss[0] > 0, "the term reached nothing, so the sum would hold vacuously"
    assert torch.isclose(weighted, off.loss + 0.3 * off.mtp_loss[0], atol=1e-5)


def _ling_run(target, corpus: str, *, steps: int, **fields):
    """The entry script on a tiny Ling-3.0 checkpoint carrying its live tail, at the head's documented route."""
    base = target / "base"
    if not base.exists():
        build_tiny_family_checkpoint(
            TINY_BAILING_LING3_MLA_FAMILY,
            str(base),
            load_cached_tokenizer(QWEN3_0_6B),
            overrides={"num_nextn_predict_layers": 1, "mtp_loss_scaling_factor": 0},
        )
    config = target / "config.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "model_name_or_path": str(base),
                "trust_remote_code": True,
                "dataset": corpus,
                "output_dir": str(target / "run"),
                "attn_implementation": "sdpa",
                "assistant_message_template": "<|im_start|>assistant\n",
                "max_length": 64,
                "per_device_train_batch_size": 1,
                "learning_rate": 1e-3,
                "lr_scheduler_type": "constant",
                "logging_steps": 1,
                "report_to": "none",
                "max_steps": steps,
                "save_steps": steps,
                "bf16": False,
                "use_cpu": True,
                "use_liger_kernel": False,
                "gradient_checkpointing": False,
                "full_determinism": True,
                "seed": 42,
                "max_grad_norm": 0,
                # Bias balancing walks every gate and would move the tail's slot with no gradient behind it.
                "moe_balancing": "none",
                **fields,
            }
        )
    )
    return run_entry(config)


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    return write_entry_corpus(tmp_path_factory.mktemp("corpus") / "rows")


_TAIL_PREFIX = f"model.layers.{_LAYERS}."


def test_the_documented_route_trains_the_tail_through_the_entry_script_and_a_weight_of_zero_leaves_it(
    tmp_path, corpus
):
    """``model_init_kwargs`` reaches the remote forward, and every save carries the tail under its release keys."""
    on, off = tmp_path / "on", tmp_path / "off"
    on.mkdir()
    off.mkdir()
    weighted = _ling_run(on, corpus, steps=2, model_init_kwargs={"mtp_loss_scaling_factor": 0.3})
    assert weighted.returncode == 0, (weighted.stderr + weighted.stdout)[-3000:]
    saved, base = saved_tensors(on / "run" / "checkpoint-2"), saved_tensors(on / "base")
    tail = [key for key in saved if key.startswith(_TAIL_PREFIX)]
    assert tail and set(tail) == {key for key in base if key.startswith(_TAIL_PREFIX)}
    assert moved_keys({key: saved[key] for key in tail}, base), "a weighted tail took no step"

    idle = _ling_run(off, corpus, steps=2, model_init_kwargs={"mtp_loss_scaling_factor": 0})
    assert idle.returncode == 0, (idle.stderr + idle.stdout)[-3000:]
    saved, base = saved_tensors(off / "run" / "checkpoint-2"), saved_tensors(off / "base")
    assert not moved_keys({key: saved[key] for key in saved if key.startswith(_TAIL_PREFIX)}, base)


def test_the_documented_route_resumes_with_its_tail_and_its_weight(tmp_path, corpus):
    fields = {"model_init_kwargs": {"mtp_loss_scaling_factor": 0.3}, "save_steps": 2}
    done = _ling_run(tmp_path, corpus, steps=4, **fields)
    assert done.returncode == 0, (done.stderr + done.stdout)[-3000:]
    run = tmp_path / "run"
    straight = saved_tensors(run / "checkpoint-4")
    shutil.rmtree(run / "checkpoint-4")
    done = _ling_run(tmp_path, corpus, steps=4, resume_from_checkpoint=True, **fields)
    assert done.returncode == 0, (done.stderr + done.stdout)[-3000:]
    resumed = saved_tensors(run / "checkpoint-4")
    differing = [key for key in straight if not torch.equal(straight[key], resumed[key])]
    assert not differing, differing[:4]


def test_the_documented_route_refuses_rows_it_would_train_across(tmp_path, corpus):
    done = _ling_run(
        tmp_path,
        corpus,
        steps=1,
        per_device_train_batch_size=2,
        model_init_kwargs={"mtp_loss_scaling_factor": 0.3},
    )
    assert done.returncode != 0
    assert "flattens rows before it shifts" in done.stderr + done.stdout


@pytest.mark.parametrize(("eval_rows", "refused"), [(2, True), (1, False)])
def test_an_evaluation_past_one_row_is_refused_as_its_loss_would_cross_rows(tmp_path, corpus, eval_rows, refused):
    done = _ling_run(
        tmp_path,
        corpus,
        steps=1,
        eval_strategy="steps",
        eval_steps=1,
        per_device_eval_batch_size=eval_rows,
        model_init_kwargs={"mtp_loss_scaling_factor": 0.3},
    )
    assert (done.returncode != 0) == refused, (done.stderr + done.stdout)[-3000:]
    assert ("per_device_eval_batch_size: 1" in done.stderr + done.stdout) == refused


def test_the_override_keeps_the_main_logits_and_exports_no_tail(tmp_path):
    live = _tiny_ling3(num_nextn_predict_layers=1, mtp_loss_scaling_factor=0)
    finalize_loaded_model(live)
    off = _tiny_ling3(num_nextn_predict_layers=0, mtp_loss_scaling_factor=0)
    finalize_loaded_model(off)
    trunk = {key: value for key, value in live.state_dict().items() if f"layers.{_LAYERS}." not in key}
    assert not off.load_state_dict(trunk, strict=False).missing_keys
    ids = _batch()["input_ids"]
    with torch.no_grad():
        assert torch.equal(live(input_ids=ids).logits, off(input_ids=ids).logits)
    save_full_checkpoint(live, str(tmp_path / "live"))
    save_full_checkpoint(off, str(tmp_path / "off"))
    assert any(f"layers.{_LAYERS}." in key for key in load_file(tmp_path / "live" / "model.safetensors"))
    assert not any(f"layers.{_LAYERS}." in key for key in load_file(tmp_path / "off" / "model.safetensors"))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
