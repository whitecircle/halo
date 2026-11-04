#!/usr/bin/env python
"""The MTP GPU scripts' own ``run()`` on gloo ranks, with the CPU named for the GPU: where a row lays out its files, what
it hands the loader and whether its checks read anything. A row that cannot launch is otherwise found on a GPU slot."""

import contextlib
import http.server
import json
import threading
from pathlib import Path

import pytest
from datasets import DatasetDict
from huggingface_hub import snapshot_download
from huggingface_hub.errors import LocalEntryNotFoundError
from transformers import AutoConfig

from tests.common.models import QWEN3_0_6B
from tests.common.mtp_heads import run_gpu_row, tokenized_rows, write_checkpoint_with_tail
from tests.common.mtp_real_rows import ULTRACHAT_DATASET
from tests.common.reporting import parse_result
from tests.common.tokenizers import load_cached_tokenizer

_SCRIPTS = Path("tests/gpu/trainers/sft")
_HEAD = str(_SCRIPTS / "test_sft_mtp_head.py")
_CHECKPOINT = str(_SCRIPTS / "test_sft_mtp_head_checkpoint.py")
_REAL = str(_SCRIPTS / "test_sft_mtp_head_real.py")
_COST = str(_SCRIPTS / "test_sft_mtp_head_cost.py")
_LING = str(_SCRIPTS / "test_sft_ling_mtp_tail.py")
_SERVING = str(_SCRIPTS / "test_sft_mtp_head_serving.py")
_LENGTH = 32
_PROMPT = 4
_REAL_FAMILY = "qwen3_5_moe_text"
_PACKED_FAMILY = "glm4_moe_lite"


def _ultrachat_is_cached() -> bool:
    try:
        snapshot_download(ULTRACHAT_DATASET, repo_type="dataset", local_files_only=True)
    except LocalEntryNotFoundError:
        return False
    return True


def _passes(done) -> dict:
    result = parse_result(done.stdout + done.stderr)
    assert result is not None, (done.stdout + done.stderr)[-3000:]
    failed = [name for name, ok in result.get("checks", {}).items() if not ok]
    assert done.returncode == 0 and result["status"] == "pass", (result.get("error"), failed)
    return result


@pytest.mark.parametrize(
    "flags",
    [
        ("--family", "qwen3_5", "--mode", "fsdp", "--arms", "soft", "--kernel", "chunked"),
        ("--family", "qwen3_5_moe_text", "--mode", "fsdp", "--arms", "soft", "--kernel", "chunked", "--gc"),
        ("--family", "qwen3_5_moe_text", "--mode", "etp", "--arms", "soft", "--kernel", "chunked"),
        ("--family", "qwen3_5_moe_text", "--mode", "ep1", "--arms", "soft", "--kernel", "chunked", "--gc"),
        ("--family", "glm4_moe_lite", "--mode", "fsdp_zero3", "--arms", "soft", "--kernel", "chunked"),
        ("--family", "qwen3_5_moe_text", "--mode", "ep1_ignored", "--arms", "soft", "--kernel", "chunked"),
        ("--family", "glm4_moe_lite", "--mode", "ep1", "--arms", "soft", "--kernel", "chunked", "--packed"),
    ],
    ids=lambda flags: "-".join(flags[1:4:2]) + ("-packed" if "--packed" in flags else ""),
)
def test_a_tiny_row_launches_and_its_gradient_reference_holds(flags, tmp_path):
    result = _passes(run_gpu_row(_HEAD, *flags, work=tmp_path))
    assert result["checks"]["joint_soft_head_gradient_within_bar"] is True
    assert result["checks"]["joint_soft_every_head_param_moved"] is True


@pytest.mark.parametrize("family", ["qwen3_5_moe_text", "glm4_moe_lite"])
def test_the_ep1_row_holds_the_experts_to_the_reference_too(family, tmp_path):
    result = _passes(
        run_gpu_row(_HEAD, "--family", family, "--mode", "ep1", "--arms", "soft", "--kernel", "chunked", work=tmp_path)
    )
    assert result["checks"]["joint_soft_compared_the_experts"] is True
    assert result["checks"]["joint_soft_experts_took_gradient"] is True


@pytest.mark.parametrize("family", ["qwen3_5_moe_text", "glm4_moe_lite"])
def test_accumulated_forwards_reach_the_first_step_whole(family, tmp_path):
    """Each forward's head gradient waits in FSDP2's accumulator until the last reduces, on the recipe's own window."""
    result = _passes(
        run_gpu_row(
            _HEAD,
            "--family",
            family,
            "--mode",
            "ep1",
            "--arms",
            "trunk_trains",
            "--kernel",
            "chunked",
            "--accumulation",
            "2",
            work=tmp_path,
        )
    )
    for arm in ("joint_hard", "joint_soft", "detached_soft"):
        assert result["checks"][f"{arm}_head_gradient_within_bar"] is True


def test_the_resume_row_replays_and_refuses_a_wider_resume_at_the_trainers_bf16(tmp_path):
    result = _passes(
        run_gpu_row(_CHECKPOINT, "--family", "qwen3_5", "--mode", "fsdp", "--flip", "--gate", work=tmp_path)
    )
    assert result["checks"]["a_wider_resume_raises_on_every_rank"] is True
    assert result["checks"]["joint_soft_resume_replays_every_tensor"] is True


@pytest.mark.parametrize("flags", [(), ("--simulated_domain", "1")], ids=["one_domain", "simulated_domains"])
def test_the_two_machine_row_writes_each_nodes_own_copy_and_feeds_each_rank_its_own_row(flags, tmp_path):
    """Plain FSDP2 at two ranks, on per-node output storage as two machines hold it."""
    result = _passes(
        run_gpu_row(
            _CHECKPOINT,
            "--family",
            "qwen3_5_moe_text",
            "--mode",
            "multi_node",
            *flags,
            work=tmp_path,
            env={"DIST_SHARED_FILESYSTEM": "0"},
        )
    )
    assert result["checks"]["joint_soft_every_rank_trained_its_own_row"] is True
    assert result["checks"]["head_only_hard_resume_replays_every_tensor"] is True


def test_the_ignored_experts_row_holds_replica_experts_byte_equal(tmp_path):
    result = _passes(run_gpu_row(_CHECKPOINT, "--family", "qwen3_5_moe_text", "--mode", "ep1_ignored", work=tmp_path))
    assert result["checks"]["joint_soft_replica_experts_byte_equal"] is True
    assert result["checks"]["joint_soft_every_rank_trained_its_own_row"] is True


def test_the_ling_row_trains_its_tail_at_a_weight_and_leaves_it_at_zero(tmp_path):
    result = _passes(
        run_gpu_row(_LING, "--mode", "ep1", work=tmp_path, env={"HF_MODULES_CACHE": str(tmp_path / "modules")})
    )
    assert result["checks"]["weighted_tail_moved"] is True
    assert result["checks"]["idle_tail_stayed"] is True


def _stand_in(root: Path, family: str) -> dict:
    """A tiny family in the real rows' place: a trunk holding its own head, a stale head elsewhere, prepared rows."""
    write_checkpoint_with_tail(family, root / "stale", per_expert=True)
    write_checkpoint_with_tail(family, root / "own", per_expert=True, seed=321)
    load_cached_tokenizer(QWEN3_0_6B).save_pretrained(root / "own")
    vocab = AutoConfig.from_pretrained(root / "own").vocab_size
    DatasetDict(
        {
            "train": tokenized_rows(vocab, 16, seed=1, length=_LENGTH, prompt=_PROMPT),
            "test": tokenized_rows(vocab, 4, seed=2, length=_LENGTH, prompt=_PROMPT),
        }
    ).save_to_disk(root / "rows")
    return {
        "env": {"HALO_TEST_MTP_REAL_MODEL": str(root / "own"), "HALO_TEST_MTP_REAL_STALE_HEAD": str(root / "stale")},
        "rows": str(root / "rows"),
    }


@pytest.fixture(scope="module")
def real_stand_in(tmp_path_factory) -> dict:
    return _stand_in(tmp_path_factory.mktemp("real"), _REAL_FAMILY)


def _real_row(real_stand_in, row: str, tmp_path, *flags: str) -> dict:
    return _passes(
        run_gpu_row(
            _REAL,
            "--row",
            row,
            "--ep_size",
            "1",
            "--kernel",
            "chunked",
            "--stream_dir",
            real_stand_in["rows"],
            "--steps",
            "4",
            "--learning_rate",
            "0.1",
            *flags,
            work=tmp_path,
            env=real_stand_in["env"],
        )
    )


def test_the_real_fit_row_scores_at_both_ends_and_leaves_the_trunk_alone(real_stand_in, tmp_path):
    result = _real_row(real_stand_in, "stale_soft", tmp_path, "--export_dir", str(tmp_path / "export"))
    assert result["checks"]["every_head_tensor_moved"] is True
    assert result["checks"]["the_trunk_kept_its_bytes"] is True
    assert result["checks"]["the_export_holds_the_head"] is True
    assert [step for step, _tv, _greedy in result["metrics"]["acceptance"]][0] == 0
    assert (tmp_path / "export" / "checkpoint-4").is_dir()


def test_the_real_detached_row_holds_the_trunk_to_a_run_with_no_head(real_stand_in, tmp_path):
    result = _real_row(real_stand_in, "detached", tmp_path)
    for loss in ("soft", "hard"):
        assert result["checks"][f"{loss}_trunk_bytes_are_the_run_with_no_head"] is True
    assert result["checks"]["next_token_curve_is_the_same_under_either_head_loss"] is True
    assert result["checks"]["the_run_with_no_head_moved_its_trunk"] is True


_TINY_COST_FLAGS = (
    "--ep_size",
    "1",
    "--kernel",
    "chunked",
    "--attn",
    "sdpa",
    "--length",
    str(_LENGTH),
    "--documents",
    "2",
    "--steps",
    "3",
    "--drop",
    "0",
)


def test_the_cost_row_runs_only_the_arms_it_is_given(tmp_path):
    stand_in = _stand_in(tmp_path / "stand_in", _PACKED_FAMILY)
    result = _passes(
        run_gpu_row(_COST, *_TINY_COST_FLAGS, "--arms", "off,head_only_soft", work=tmp_path, env=stand_in["env"])
    )
    assert sorted(result["metrics"]) == ["head_only_soft", "off"]
    assert result["checks"]["head_only_soft_took_its_term"] is True
    assert result["checks"]["off_took_its_term"] is True


def test_the_cost_row_runs_every_arm_and_tells_a_head_from_none(tmp_path):
    """On GLM-4, as Qwen3.5's packed rows take a CUDA kernel."""
    stand_in = _stand_in(tmp_path / "stand_in", _PACKED_FAMILY)
    result = _passes(run_gpu_row(_COST, *_TINY_COST_FLAGS, work=tmp_path, env=stand_in["env"]))
    assert sorted(name for name in result["metrics"]) == sorted(
        (
            "off",
            "joint_hard",
            "joint_soft",
            "detached_hard",
            "detached_soft",
            "head_only_hard",
            "head_only_soft",
            "off_again",
        )
    )
    assert result["checks"]["joint_soft_took_its_term"] is True
    assert result["checks"]["off_took_its_term"] is True


class _FakeEngine(http.server.BaseHTTPRequestHandler):
    """The endpoints the serving row reads, drafting ``3`` of every ``4`` tokens on request. ``drafting`` off answers as
    a server with no speculative config."""

    drafting = True
    model = ""
    drafted = {"proposed": 0.0, "accepted": 0.0}
    greedy: dict | None = None

    def log_message(self, *_args):
        pass

    def _reply(self, body: str, content_type: str = "application/json") -> None:
        encoded = body.encode()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def do_GET(self):
        if self.path == "/v1/models":
            self._reply(json.dumps({"data": [{"id": type(self).model}]}))
            return
        drafted = type(self).drafted
        self._reply(
            f'vllm:spec_decode_num_drafts_total{{engine="0"}} {drafted["proposed"]}\n'
            f'vllm:spec_decode_num_accepted_tokens_per_pos_total{{engine="0",position="0"}} {drafted["accepted"]}\n'
            f'vllm:spec_decode_num_accepted_tokens_per_pos_total{{engine="0",position="1"}} 1000.0\n',
            "text/plain",
        )

    def do_POST(self):
        request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path == "/generate":
            self._reply(
                json.dumps(
                    {"text": "reply", "meta_info": {"spec_num_proposed_drafts": 4, "spec_num_correct_drafts": 3}}
                    if type(self).drafting
                    else {"text": "reply", "meta_info": {}}
                )
            )
            return
        if type(self).drafting:
            type(self).drafted["proposed"] += 4
            type(self).drafted["accepted"] += 3
        choice = {"text": "reply"}
        if "logprobs" in request:
            choice["logprobs"] = type(self).greedy
        self._reply(json.dumps({"choices": [choice]}))


@contextlib.contextmanager
def _running(handler):
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        thread.join()


@pytest.fixture
def fake_engine(tmp_path):
    if not _ultrachat_is_cached():
        pytest.skip(f"{ULTRACHAT_DATASET} is not in the local Hub cache")
    tokenizer_dir = tmp_path / "tokenizer"
    load_cached_tokenizer(QWEN3_0_6B).save_pretrained(tokenizer_dir)
    handler = type(
        "Engine", (_FakeEngine,), {"model": str(tokenizer_dir), "drafted": {"proposed": 0.0, "accepted": 0.0}}
    )
    with _running(handler) as url:
        yield handler, url


@pytest.mark.parametrize(("engine", "variable"), [("vllm", "VLLM_SERVER_URL"), ("sglang", "SGLANG_SERVER_URL")])
def test_the_serving_row_reads_the_engines_own_counts_and_holds_its_bars(engine, variable, fake_engine, tmp_path):
    handler, url = fake_engine
    row = (_SERVING, "--engine", engine, "--prompts", "4")
    passing = _passes(run_gpu_row(*row, "--at_least", "0.7", work=tmp_path, ranks=1, env={variable: url}))
    assert passing["metrics"]["acceptance"] == pytest.approx(0.75)
    failing = run_gpu_row(*row, "--at_least", "0.8", work=tmp_path, ranks=1, env={variable: url})
    assert parse_result(failing.stdout + failing.stderr)["checks"]["acceptance_reached_its_bar"] is False
    ceiling = run_gpu_row(*row, "--at_most", "0.05", work=tmp_path, ranks=1, env={variable: url})
    assert parse_result(ceiling.stdout + ceiling.stderr)["checks"]["acceptance_stayed_under_its_bar"] is False


_DRAFTED_GREEDY = {"tokens": ["a", "b"], "top_logprobs": [{"a": -0.1}, {"b": -0.5, "c": -0.75}]}


@pytest.mark.parametrize(
    ("plain", "holds"),
    [
        (_DRAFTED_GREEDY, True),
        ({"tokens": ["a", "c"], "top_logprobs": [{"a": -0.1}, {"c": -0.5, "b": -0.75}]}, True),
        ({"tokens": ["a", "c"], "top_logprobs": [{"a": -0.1}, {"c": -0.1, "b": -3.0}]}, False),
        ({"tokens": ["a", "c"], "top_logprobs": [{"a": -0.1}, {"c": -0.1}]}, False),
        ({"tokens": ["a"], "top_logprobs": [{"a": -0.1}]}, False),
    ],
    ids=("same", "near_tie", "wide_margin", "pick_outside_the_top", "stopped_early"),
)
def test_the_serving_row_lets_greedy_output_part_only_at_a_near_tie(plain, holds, fake_engine, tmp_path):
    handler, url = fake_engine
    handler.greedy = _DRAFTED_GREEDY
    with _running(type("Plain", (handler,), {"greedy": plain, "drafting": False})) as plain_url:
        done = run_gpu_row(
            _SERVING,
            "--engine",
            "vllm",
            "--prompts",
            "4",
            "--plain_url",
            plain_url,
            work=tmp_path,
            ranks=1,
            env={"VLLM_SERVER_URL": url},
        )
    assert parse_result(done.stdout + done.stderr)["checks"]["greedy_output_parts_only_at_a_near_tie"] is holds


def test_the_serving_row_skips_a_server_that_proposes_no_draft(fake_engine, tmp_path):
    handler, url = fake_engine
    handler.drafting = False
    done = run_gpu_row(
        _SERVING, "--engine", "vllm", "--prompts", "4", work=tmp_path, ranks=1, env={"VLLM_SERVER_URL": url}
    )
    assert done.returncode == 0, (done.stdout + done.stderr)[-2000:]
    assert "SKIP:" in done.stdout + done.stderr
    assert parse_result(done.stdout + done.stderr) is None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
