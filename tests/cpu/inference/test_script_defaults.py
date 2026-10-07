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
* Gradio API: each app's real ``create_demo`` is built and served under the gradio the lock pins.
  A gradio major drops constructor arguments (``type=`` on ``ChatInterface``/``Chatbot``, the
  theme on ``Blocks``); the apps still import and their parsers still build, so only
  constructing the demo and launching it catches the break.
* Endpoint flags: every generation, eval and playground CLI takes ``--base_url``/``--api_key`` from
  the one helper in ``scripts/_common.py``, so a command line carries from one to the next.
* Shared flag blocks: every dtype flag under ``scripts/`` comes from ``add_dtype_arg`` and every
  prompt-row field flag of the generation CLIs from ``add_prompt_field_args``, so a re-typed copy
  cannot drift onto its own default or choices.
* Environment-playground request plumbing: the app documents a keyless local vLLM, so a ``None``
  API key (which ``AsyncOpenAI`` refuses at construction), an empty ``"model"`` sent verbatim, and a
  scheme-less base URL each break exactly the invocation the docstring advertises.

Run: pytest tests/cpu/inference/test_script_defaults.py
"""

import argparse
import ast
import functools
import json
import logging
import re
import sys
import types
import warnings
from pathlib import Path

import gradio as gr
import httpx
import pytest
import torch
from openai import AsyncOpenAI

from scripts._common import add_dtype_arg, add_openai_endpoint_args
from scripts.inference import _common as inference_common
from scripts.inference._common import add_prompt_field_args
from scripts.inference.playground import gradio_environment_playground
from scripts.inference.reward_model import _common as reward_model_common
from scripts.inference.reward_model._common import build_generation_parser
from src.inference.endpoints import DEFAULT_LOCAL_BASE_URL
from tests.common.ports import free_port
from tests.common.utils import load_script_module

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_INFERENCE_ROOT = _PROJECT_ROOT / "scripts" / "inference"
_GRADIO_APPS = sorted(_INFERENCE_ROOT.rglob("gradio_*.py"))
# Loopback spellings: an app reachable only from its own host. Anything else is published.
_LOOPBACK = {"127.0.0.1", "localhost", "::1"}
# A flag carrying an OpenAI-compatible endpoint's address or key, in any spelling.
_ENDPOINT_FLAG = re.compile(r"^--.*(url|api[-_]key)$")


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
    parser = build_generation_parser("test", temperature_default=0.0)
    return parser.parse_args(["--model", "gen", "--prompts_source", "p.jsonl", "--rm_model_path", "org/rm", *extra])


def test_reward_model_dtype_defaults_to_bfloat16():
    assert _rm_args().rm_dtype == "bfloat16", "fp16 reward logits saturate exactly where the RM is least sure"


@pytest.mark.parametrize(("requested", "expected"), [(None, torch.bfloat16), ("fp16", torch.float16)])
def test_reward_model_loader_applies_the_requested_dtype(monkeypatch, requested, expected):
    """The knob has to reach the model — a parsed-and-ignored dtype is worse than no knob."""
    captured = {}

    class _Model(torch.nn.Module):
        """A real module: the loader finalizes what it loads, and that walks the module tree."""

        def to(self, dtype=None, device=None):
            captured["dtype"] = dtype
            return self

        def eval(self):
            return self

    monkeypatch.setattr(reward_model_common, "reject_sharded_checkpoint", lambda path: None)
    monkeypatch.setattr(
        reward_model_common,
        "AutoTokenizer",
        types.SimpleNamespace(from_pretrained=lambda *a, **k: types.SimpleNamespace()),
    )
    monkeypatch.setattr(reward_model_common, "from_pretrained_verified", lambda *a, **k: _Model())

    args = _rm_args(*(["--rm_dtype", requested] if requested else []))
    reward_model_common.load_reward_model(
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
    holders = [p.name for p in _GRADIO_APPS if _gradio_app(p).build_parser().get_default("api_key") is not None]
    assert holders, "no gradio app holds an API key — the rules below cover nothing"


@pytest.mark.parametrize("app", _GRADIO_APPS, ids=lambda p: p.name)
def test_a_gradio_app_publishes_nothing_by_default(app):
    """A Gradio app holding a live API key must not publish itself out of the box.

    ``--host`` is passed straight to ``demo.launch(server_name=...)``, so it is a BIND address:
    these apps resolve an OpenRouter/OpenAI/vLLM key from the environment, and a ``0.0.0.0`` default
    hands the UI — and with it that key's spend — to anything that can route to the box, with no
    auth in front, while a ``--share`` that defaults on hands it to the internet through Gradio's
    own tunnel. Publishing stays possible, on the explicit flag.

    Read off the app's real parser, since the block is declared once in
    ``scripts/inference/_common.py``: a source-level literal is not the contract.
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


# --- Gradio API: the demos build and serve under the pinned gradio ------------------------------


def _chatbot_demo(mod):
    return mod.create_demo(AsyncOpenAI(base_url=DEFAULT_LOCAL_BASE_URL, api_key="EMPTY"), None, [])


def _playground_demo(mod):
    return mod.create_demo(api_key="EMPTY")


# One builder per shipped app, keyed by file name; the sweep below fails on an app with none.
_DEMO_BUILDERS = {
    "gradio_openai_chatbot.py": _chatbot_demo,
    "gradio_environment_playground.py": _playground_demo,
}


@pytest.mark.parametrize("app", _GRADIO_APPS, ids=lambda p: p.name)
def test_a_gradio_app_builds_and_serves_under_the_pinned_gradio(app):
    """The app's real ``create_demo``, then queue + launch with the theme, as ``launch_gradio`` does.

    Construction runs with gradio's warnings as errors: a theme passed on the ``Blocks`` is
    accepted with a warning and silently dropped, which is the same regression as a removed
    argument, only quieter.
    """
    build = _DEMO_BUILDERS.get(app.name)
    assert build is not None, f"{app.name} has no demo builder here, so its gradio API surface goes untested"

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        demo = build(_gradio_app(app))
    assert isinstance(demo, gr.Blocks), f"{app.name}.create_demo returned {type(demo).__name__}"

    port = free_port()
    try:
        _app, local_url, _share_url = demo.queue().launch(
            server_name="127.0.0.1", server_port=port, share=False, theme=gr.themes.Soft(), prevent_thread_lock=True
        )
        assert httpx.get(local_url, timeout=10).status_code == 200, f"{app.name} launched but does not serve"
    finally:
        demo.close()


# --- Environment playground: the keyless-local-vLLM invocation its docstring documents ------------


def _playground():
    return gradio_environment_playground


def test_the_environment_playground_key_defaults_to_the_vllm_placeholder(monkeypatch):
    """``AsyncOpenAI`` raises on ``api_key=None``, so a ``None`` default would fail every run against
    the keyless local server the module's own usage block documents at client construction. The
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
        f"--api_key defaults to {captured['api_key']!r}; a keyless local vLLM needs the placeholder, "
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
            "react_math", "2+2?", "4", "localhost:8000/v1", "EMPTY", model_name, 0.7, 16, 0.95, 1
        )

    blank_url, blank_body = seen[0]
    _named_url, named_body = seen[1]
    assert blank_url.startswith("http://localhost:8000/v1"), (
        f"a scheme-less base URL must be normalized before the client sees it, got {blank_url!r}"
    )
    assert "model" not in blank_body, f"an unset Model Name must send no model field, got {blank_body.get('model')!r}"
    assert named_body["model"] == "my-model", "a typed Model Name must still reach the server"


def test_the_environment_playground_refuses_an_answer_graded_env_without_an_answer(monkeypatch):
    """An environment that grades against the expected answer must be refused before any generation,
    not after a whole episode that then cannot be graded."""
    seen = []
    mod = _mock_playground_client(monkeypatch, seen, content="done")
    with pytest.raises(ValueError, match="Expected Answer"):
        mod.run_playground_episode("swe", "fix it", "", "http://localhost:8000/v1", "EMPTY", "m", 0.7, 16, 0.95, 2)
    assert seen == []


def test_the_environment_playground_closes_the_environment_of_every_click(monkeypatch):
    """Each click builds its own environment: one never closed keeps its sandbox sessions, server
    connections and scorer clients for the life of the app, and a refused click must close it too."""
    seen, closed = [], []
    mod = _mock_playground_client(monkeypatch, seen, content="Final Answer: 4")
    resolve = mod.resolve_environment

    def _resolve(env_type, config):
        env = resolve(env_type, config)
        close = env.close
        env.close = lambda: (closed.append(env_type), close())
        return env

    monkeypatch.setattr(mod, "resolve_environment", _resolve)
    mod.run_playground_episode("react_math", "2+2?", "4", "http://localhost:8000/v1", "EMPTY", "m", 0.7, 16, 0.95, 1)
    with pytest.raises(ValueError, match="Expected Answer"):
        mod.run_playground_episode("swe", "fix it", "", "http://localhost:8000/v1", "EMPTY", "m", 0.7, 16, 0.95, 2)
    assert closed == ["react_math", "swe"]


def test_the_environment_playground_logs_the_traceback_of_a_failed_click(monkeypatch, caplog):
    """The UI shows only the message; the traceback is what localizes the fault, so it goes to the log."""
    mod = _playground()

    def _boom(*args):
        raise RuntimeError("server exploded")

    monkeypatch.setattr(mod, "run_playground_episode", _boom)
    (on_run,) = (block.fn for block in mod.create_demo(api_key="EMPTY").fns.values())
    with caplog.at_level(logging.ERROR, logger=mod.__name__):
        messages, summary = on_run("react_math", "2+2?", "4", "http://x/v1", "", 0.7, 16, 0.95, 1)
    assert (messages, summary) == ([], "**Error:** server exploded")
    (logged,) = (r for r in caplog.records if r.levelno >= logging.ERROR)
    assert logged.exc_info[1].args == ("server exploded",)


def test_the_environment_playground_reports_a_length_cut_turn_as_one(monkeypatch):
    """The playground drives the shared eval episode driver, so ``finish_reason`` reaches the env.

    A local copy of the loop that forgets the stamp grades a mid-sentence fragment as the model's
    deliberate final answer — the episode reads as a clean natural termination in the UI, and the
    playground stops reproducing what training does with the same generation.
    """
    seen = []
    mod = _mock_playground_client(monkeypatch, seen, finish_reason="length", content="Thought: I was cut off mid-")

    _messages, summary = mod.run_playground_episode(
        "native_math", "2+2?", "4", "http://localhost:8000/v1", "EMPTY", "m", 0.7, 16, 0.95, 2
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
    expected = {"--n_parallel": "DEFAULT_N_PARALLEL", "--checkpoint_interval": "DEFAULT_CHECKPOINT_INTERVAL"}
    declared: dict[str, list] = {flag: [] for flag in expected}
    for script in sorted(_INFERENCE_ROOT.rglob("*.py")):
        source = script.read_text(encoding="utf-8")
        for flag in expected:
            default = _argparse_default(source, flag)
            if default is not _ABSENT:
                declared[flag].append((script.name, default))

    for flag, constant in expected.items():
        assert getattr(inference_common, constant), f"{constant} is not defined in scripts/inference/_common.py"
        assert declared[flag], f"no inference CLI declares {flag} — this check covers nothing"
        literal = sorted(entry for entry in declared[flag] if entry[1] != constant)
        assert not literal, (
            f"{flag} must default to the shared {constant} from scripts/inference/_common.py, not to a "
            f"per-script literal; found {literal}"
        )


def test_the_local_endpoint_default_has_one_home():
    """One spelling of the value that decides whether a run's conversations stay on this host.

    Every CLI that defaults an endpoint takes it from ``add_openai_endpoint_args``, which reads
    ``src.inference.endpoints``'s constant; a re-declaration anywhere under ``scripts/`` is a
    second source of truth.
    """
    assert add_openai_endpoint_args(argparse.ArgumentParser()).get_default("base_url") is DEFAULT_LOCAL_BASE_URL
    for script in sorted((_PROJECT_ROOT / "scripts").rglob("*.py")):
        source = script.read_text(encoding="utf-8")
        assert "DEFAULT_LOCAL_BASE_URL = " not in source, f"{script.name} re-declares the endpoint constant"


def _declared_flags(source: str) -> list[str]:
    """Every flag spelling an ``add_argument`` call in ``source`` declares."""
    return [
        arg.value
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call) and getattr(node.func, "attr", None) == "add_argument"
        for arg in node.args
        if isinstance(arg, ast.Constant) and isinstance(arg.value, str)
    ]


def test_every_endpoint_flag_comes_from_the_shared_helper():
    """One spelling of the endpoint across the generation, eval and playground CLIs.

    They drive the same served model, so a command line has to carry from one to the next; a CLI that
    declares its own URL or key flag is how one ends up ``--base_url`` and the next ``--model-url``,
    with the key's default and help copied beside each. ``scripts/profiling/`` is out of scope: its
    ``--server-url`` is the rollout server's root for the weight-sync group, not an OpenAI endpoint.
    """
    shared = add_openai_endpoint_args(argparse.ArgumentParser())
    shared_flags = {spelling for action in shared._actions for spelling in action.option_strings}
    assert {"--base_url", "--api_key"} <= {flag for flag in shared_flags if _ENDPOINT_FLAG.match(flag)}, (
        "the endpoint pattern no longer recognizes the shared spellings, so the sweep below covers nothing"
    )

    scripts_root = _PROJECT_ROOT / "scripts"
    redeclared = [
        f"{script.relative_to(_PROJECT_ROOT)}: {flag}"
        for script in sorted(scripts_root.rglob("*.py"))
        if script != scripts_root / "_common.py" and script.relative_to(scripts_root).parts[0] != "profiling"
        for flag in _declared_flags(script.read_text(encoding="utf-8"))
        if _ENDPOINT_FLAG.match(flag)
    ]
    assert not redeclared, (
        f"endpoint flags declared outside scripts/_common.py's add_openai_endpoint_args: {redeclared}"
    )


def _declared_action(parser: argparse.ArgumentParser, flag: str) -> argparse.Action:
    return next(action for action in parser._actions if flag in action.option_strings)


def test_every_dtype_flag_comes_from_the_shared_helper():
    """One default and one choice set for every dtype flag under ``scripts/``: the scorer's
    ``--rm_dtype`` names a different model's dtype, not a different set of spellings or a different
    default from the checkpoint tools that produced that model."""
    scripts_root = _PROJECT_ROOT / "scripts"
    redeclared = [
        f"{script.relative_to(_PROJECT_ROOT)}: {flag}"
        for script in sorted(scripts_root.rglob("*.py"))
        if script != scripts_root / "_common.py"
        for flag in _declared_flags(script.read_text(encoding="utf-8"))
        if "dtype" in flag
    ]
    assert not redeclared, f"dtype flags declared outside scripts/_common.py's add_dtype_arg: {redeclared}"

    scorer = _declared_action(build_generation_parser("test", temperature_default=0.0), "--rm_dtype")
    shared = _declared_action(add_dtype_arg(argparse.ArgumentParser()), "--dtype")
    assert (scorer.default, scorer.choices) == (shared.default, shared.choices)


def test_the_prompt_row_fields_come_from_the_shared_helper():
    """The S3 generation CLI and the reward-model scorers read the same row fields, so a prompt file
    keyed for one is keyed for the other; one block declares them."""
    shared = {
        spelling
        for action in add_prompt_field_args(argparse.ArgumentParser(add_help=False))._actions
        for spelling in action.option_strings
    }
    scorer = {
        spelling
        for action in build_generation_parser("test", temperature_default=0.0)._actions
        for spelling in action.option_strings
    }
    assert shared and shared <= scorer, f"the scorers do not carry the shared row fields {sorted(shared - scorer)}"

    redeclared = [
        f"{script.relative_to(_PROJECT_ROOT)}: {flag}"
        for script in sorted(_INFERENCE_ROOT.rglob("*.py"))
        if script != _INFERENCE_ROOT / "_common.py"
        for flag in _declared_flags(script.read_text(encoding="utf-8"))
        if flag in shared
    ]
    assert not redeclared, f"row-field flags declared outside add_prompt_field_args: {redeclared}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
