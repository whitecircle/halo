#!/usr/bin/env python
"""Inference-script defaults that decide where a run goes and what it computes in.

* Reward-model dtype: the RM scripts take the scoring dtype as a knob, defaulting to the toolkit's
  bf16 — a hardcoded fp16 would be the one fp16 in the repo. Reward logits are unbounded and
  out-of-distribution completions push them furthest, which is exactly where fp16's narrow range
  saturates.
* Gradio bind address: every app here holds a live OpenRouter/OpenAI/vLLM key, so ``--host``
  defaults to loopback and ``--share`` is off; a ``0.0.0.0`` default would publish a key-spending UI
  to every interface out of the box. Reaching it from off-box takes an explicit flag. The apps
  declare that block once (``add_gradio_server_args``), so the checks build each app's real parser
  rather than reading a literal out of its source.
* Throughput/resume defaults: ``--n_parallel`` (4/8/32) and ``--checkpoint_interval`` (50/100) read
  one home each rather than four and two independent literals across scripts driving the same
  endpoint.
* Environment-playground request plumbing: the app documents a keyless local vLLM, so a ``None``
  API key (which ``AsyncOpenAI`` refuses at construction), an empty ``"model"`` sent verbatim, and a
  scheme-less base URL each break exactly the invocation the docstring advertises.

Run: pytest tests/cpu/inference/test_script_defaults.py
"""

import ast
import functools
import json
import sys
import types
from pathlib import Path

import httpx
import pytest
import torch
from openai import AsyncOpenAI

from src.inference.openai_client import DEFAULT_LOCAL_BASE_URL
from tests.common.utils import load_script_module

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_INFERENCE_ROOT = _PROJECT_ROOT / "scripts" / "inference"
_GRADIO_APPS = sorted(_INFERENCE_ROOT.rglob("gradio_*.py"))
# Loopback spellings: an app reachable only from its own host. Anything else is published.
_LOOPBACK = {"127.0.0.1", "localhost", "::1"}
# Env keys whose presence in a source file means a live, spendable credential is resident.
_SECRET_ENV_KEYS = ("OPENROUTER_API_KEY", "OPENAI_API_KEY")


_ABSENT = object()  # distinguishes "the script declares no such flag" from "default=None"


@functools.cache
def _gradio_app(path: Path) -> types.ModuleType:
    """The imported module of one Gradio app, by path — ``scripts/`` is not a package.

    Cached: every app pulls in gradio, a seconds-long import.
    """
    return load_script_module(str(path.relative_to(_PROJECT_ROOT)))


def _argparse_default(source: str, flag: str):
    """The ``default=`` of the ``add_argument(flag, ...)`` call in ``source``, read off the AST.

    These parsers are built under ``if __name__ == "__main__"`` in some apps, so there is no
    importable ``parse_args`` to call — the declaration itself is the contract under test. A default
    given as a name (a shared constant) is resolved by the caller, not here.
    """
    for node in ast.walk(ast.parse(source)):
        if not (isinstance(node, ast.Call) and getattr(node.func, "attr", None) == "add_argument"):
            continue
        if not (node.args and isinstance(node.args[0], ast.Constant) and node.args[0].value == flag):
            continue
        for keyword in node.keywords:
            if keyword.arg == "default":
                if isinstance(keyword.value, ast.Constant):
                    return keyword.value.value
                if isinstance(keyword.value, ast.Name):
                    return keyword.value.id
                raise AssertionError(f"{flag} declares a default this reader cannot evaluate")
        raise AssertionError(f"{flag} declares no default")
    return _ABSENT


def _rm_args(*extra: str):
    from scripts.inference.reward_model._common import build_generation_parser

    parser = build_generation_parser("test", temperature_default=0.0)
    return parser.parse_args(
        ["--model_name", "gen", "--prompts_source", "p.jsonl", "--rm_model_path", "org/rm", *extra]
    )


def test_reward_model_dtype_defaults_to_bfloat16():
    assert _rm_args().rm_dtype == "bfloat16", "fp16 reward logits saturate exactly where the RM is least sure"


@pytest.mark.parametrize(("requested", "expected"), [(None, torch.bfloat16), ("fp16", torch.float16)])
def test_reward_model_loader_applies_the_requested_dtype(monkeypatch, requested, expected):
    """The knob has to reach the model — a parsed-and-ignored dtype is worse than no knob."""
    from scripts.inference.reward_model import _common

    captured = {}

    class _Model(torch.nn.Module):
        """A real module: the loader finalizes what it loads, and that walks the module tree."""

        def to(self, dtype=None, device=None):
            captured["dtype"] = dtype
            return self

        def eval(self):
            return self

    monkeypatch.setattr(_common, "reject_sharded_checkpoint", lambda path: None)
    monkeypatch.setattr(
        _common, "AutoTokenizer", types.SimpleNamespace(from_pretrained=lambda *a, **k: types.SimpleNamespace())
    )
    monkeypatch.setattr(_common, "from_pretrained_verified", lambda *a, **k: _Model())

    args = _rm_args(*(["--rm_dtype", requested] if requested else []))
    _common.load_reward_model(
        args.rm_model_path,
        args.rm_model_atten_impl,
        args.rm_max_seq_len,
        "cpu",
        args.rm_dtype,
        trust_remote_code=False,
    )

    assert captured["dtype"] is expected


# --- Gradio server block -------------------------------------------------------------------------


def test_the_gradio_apps_under_test_exist():
    """Guards the sweep below: an empty glob would assert nothing."""
    assert len(_GRADIO_APPS) >= 2, f"expected the shipped gradio apps, found {[p.name for p in _GRADIO_APPS]}"
    holders = [p.name for p in _GRADIO_APPS if any(k in p.read_text(encoding="utf-8") for k in _SECRET_ENV_KEYS)]
    assert holders, "no gradio app reads an API key from the environment — the rules below cover nothing"


@pytest.mark.parametrize("app", _GRADIO_APPS, ids=lambda p: p.name)
def test_a_gradio_app_publishes_nothing_by_default(app):
    """A Gradio app holding a live API key must not publish itself out of the box.

    ``--host`` is passed straight to ``demo.launch(server_name=...)``, so it is a BIND address:
    these apps resolve an OpenRouter/OpenAI/vLLM key from the environment, and a ``0.0.0.0`` default
    hands the UI — and with it that key's spend — to anything that can route to the box, with no
    auth in front, while a ``--share`` that defaults on hands it to the internet through Gradio's
    own tunnel. Publishing stays possible, on the explicit flag.

    Read off the app's real parser, since the block is declared once in
    ``scripts/inference/_common.py``: a source-level literal is no longer the contract.
    """
    parser = _gradio_app(app).build_parser()

    host = parser.get_default("host")
    assert host in _LOOPBACK, (
        f"{app.name} defaults --host to {host!r}; an omitted flag must keep the key-holding UI on loopback"
    )
    assert parser.get_default("share") is False, (
        f"{app.name} defaults --share on; a public Gradio tunnel out of the box exposes the UI's key spend"
    )
    assert "--port" in parser.format_usage(), (
        f"{app.name} does not declare --port; the apps share one spelling of the address block, so an "
        f"operator's pinned command line works against all of them"
    )


# --- Environment playground: the keyless-local-vLLM invocation its docstring documents ------------


def _playground():
    from scripts.inference.playground import gradio_environment_playground

    return gradio_environment_playground


def test_the_environment_playground_key_defaults_to_the_vllm_placeholder(monkeypatch):
    """``AsyncOpenAI`` raises on ``api_key=None``, so a ``None`` default made every run against the
    keyless local server the module's own usage block documents fail at client construction. The
    sibling gradio apps default to the served placeholder; this one must too."""
    mod = _playground()
    monkeypatch.delenv("VLLM_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    captured = {}

    def _create_demo(default_base_url, api_key):
        captured.update(base_url=default_base_url, api_key=api_key)
        return types.SimpleNamespace(queue=lambda: types.SimpleNamespace(launch=lambda **kwargs: None))

    monkeypatch.setattr(mod, "create_demo", _create_demo)
    monkeypatch.setattr(sys, "argv", ["gradio_environment_playground.py"])
    mod.main()

    assert captured["api_key"] == "EMPTY", (
        f"--api-key defaults to {captured['api_key']!r}; a keyless local vLLM needs the placeholder, "
        f"and None makes AsyncOpenAI raise before the first request"
    )


def _mock_playground_client(monkeypatch, seen, *, finish_reason="stop", content="hi"):
    """Point the playground's client factory at a MockTransport, recording each request."""
    mod = _playground()

    def _handler(request):
        seen.append((str(request.url), json.loads(request.content)))
        return httpx.Response(
            200,
            json={
                "id": "c",
                "object": "chat.completion",
                "created": 0,
                "model": "served",
                "choices": [
                    {"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": finish_reason}
                ],
            },
        )

    monkeypatch.setattr(
        mod,
        "create_openai_client",
        lambda base_url, api_key_override: AsyncOpenAI(
            base_url=base_url,
            api_key=api_key_override,
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(_handler)),
        ),
    )
    return mod


def test_the_environment_playground_sends_no_model_until_one_is_typed(monkeypatch):
    """The Model Name box ships EMPTY, and an empty ``"model"`` is sent verbatim (the SDK only drops
    ``NOT_GIVEN``) — a single-model server 404s it instead of answering with what it serves. The URL
    box is hand-edited, so a scheme-less ``localhost:8000/v1`` must still route."""
    seen = []
    mod = _mock_playground_client(monkeypatch, seen, content="Final Answer: 4")

    for model_name in ("", "my-model"):
        mod.run_playground_episode(
            "react_math", "2+2?", "4", "localhost:8000/v1", "EMPTY", model_name, 0.7, 16, 0.95, 1, 1.0
        )

    blank_url, blank_body = seen[0]
    _named_url, named_body = seen[1]
    assert blank_url.startswith("http://localhost:8000/v1"), (
        f"a scheme-less base URL must be normalized before the client sees it, got {blank_url!r}"
    )
    assert "model" not in blank_body, f"an unset Model Name must send no model field, got {blank_body.get('model')!r}"
    assert named_body["model"] == "my-model", "a typed Model Name must still reach the server"


def test_the_environment_playground_reports_a_length_cut_turn_as_one(monkeypatch):
    """The playground drives the shared eval episode driver, so ``finish_reason`` reaches the env.

    A local copy of the loop that forgets the stamp grades a mid-sentence fragment as the model's
    deliberate final answer — the episode reads as a clean natural termination in the UI, and the
    playground stops reproducing what training does with the same generation.
    """
    seen = []
    mod = _mock_playground_client(monkeypatch, seen, finish_reason="length", content="Thought: I was cut off mid-")

    _messages, summary = mod.run_playground_episode(
        "native_math", "2+2?", "4", "http://localhost:8000/v1", "EMPTY", "m", 0.7, 16, 0.95, 2, 1.0
    )

    assert "**Length-capped turns:** 2" in summary, summary
    assert len(seen) == 2, "a length-cut turn must be retried within max_turns, not finalized as an answer"


# --- Shared throughput / resume defaults ---------------------------------------------------------


def test_the_generation_clis_share_one_concurrency_and_checkpoint_default():
    """One home per knob, or the siblings drift.

    The CLIs drive the same local rollout server through the same async client, so a per-script
    literal — ``--n_parallel`` at 4, 8 or 32, ``--checkpoint_interval`` at 50 or 100 — silently
    throttles a sibling's throughput and widens what an interrupted run must regenerate, with
    nothing claiming the difference is deliberate.
    """
    from scripts.inference import _common

    expected = {"--n_parallel": "DEFAULT_N_PARALLEL", "--checkpoint_interval": "DEFAULT_CHECKPOINT_INTERVAL"}
    declared: dict[str, list] = {flag: [] for flag in expected}
    for script in sorted(_INFERENCE_ROOT.rglob("*.py")):
        source = script.read_text(encoding="utf-8")
        for flag in expected:
            default = _argparse_default(source, flag)
            if default is not _ABSENT:
                declared[flag].append((script.name, default))

    for flag, constant in expected.items():
        assert getattr(_common, constant), f"{constant} is not defined in scripts/inference/_common.py"
        assert declared[flag], f"no inference CLI declares {flag} — this check covers nothing"
        literal = sorted(entry for entry in declared[flag] if entry[1] != constant)
        assert not literal, (
            f"{flag} must default to the shared {constant} from scripts/inference/_common.py, not to a "
            f"per-script literal; found {literal}"
        )


def test_the_local_endpoint_default_has_one_home():
    """One spelling of the value that decides whether a run's conversations stay on this host.

    Every CLI that defaults an endpoint — the inference scripts and the environment eval scripts —
    reads ``src.inference.openai_client``'s constant; a re-declaration anywhere under ``scripts/`` is a
    second source of truth.
    """
    from scripts.environments import _common as env_common

    assert env_common.DEFAULT_LOCAL_BASE_URL is DEFAULT_LOCAL_BASE_URL
    for script in sorted((_PROJECT_ROOT / "scripts").rglob("*.py")):
        source = script.read_text(encoding="utf-8")
        assert "DEFAULT_LOCAL_BASE_URL = " not in source, f"{script.name} re-declares the endpoint constant"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
