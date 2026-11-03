#!/usr/bin/env python
"""Inference-script defaults that decide where a run goes and what it computes in: the reward-model dtype, the
Gradio bind address, the shared throughput and endpoint flags, and the environment playground's request plumbing."""

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
from transformers import Qwen3Config

from halo.entrypoints import reward_scoring
from halo.inference.endpoints import DEFAULT_LOCAL_BASE_URL
from scripts.cli_arguments import add_dtype_arg, add_openai_endpoint_args
from scripts.inference import inference_arguments
from scripts.inference.inference_arguments import add_prompt_field_args
from scripts.inference.playground import gradio_environment_playground
from scripts.inference.reward_model.reward_model_arguments import build_generation_parser
from tests.common.ports import free_port
from tests.common.repo import REPO_ROOT, load_script_module

_INFERENCE_ROOT = REPO_ROOT / "scripts" / "inference"
_GRADIO_APPS = sorted(_INFERENCE_ROOT.rglob("gradio_*.py"))
# Loopback spellings: an app reachable only from its own host. Anything else is published.
_LOOPBACK = {"127.0.0.1", "localhost", "::1"}
_ENDPOINT_FLAG = re.compile(r"^--.*(url|api[-_]key)$")


_ABSENT = object()  # distinguishes "the script declares no such flag" from "default=None"


@functools.cache
def _gradio_app(path: Path) -> types.ModuleType:
    """The imported module of one Gradio app, by path. Cached, since every app pulls in gradio, a seconds-long
    import."""
    return load_script_module(str(path.relative_to(REPO_ROOT)))


def _argparse_default(source: str, flag: str):
    """The ``default=`` of ``add_argument(flag, ...)`` in ``source``, read off the AST, since some apps build their
    parser under ``__main__``. A default given as a name is resolved by the caller."""
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
    return parser.parse_args(["--model", "gen", "--dataset", "p.jsonl", "--rm_model_id", "org/rm", *extra])


def test_reward_model_dtype_defaults_to_bfloat16():
    assert _rm_args().rm_dtype == "bfloat16", "fp16 reward logits saturate exactly where the RM is least sure"


@pytest.mark.parametrize(("requested", "expected"), [(None, torch.bfloat16), ("fp16", torch.float16)])
def test_reward_model_loader_applies_the_requested_dtype(monkeypatch, requested, expected):
    captured = {}

    class _Model(torch.nn.Module):
        """A real module: the loader finalizes what it loads, and that walks the module tree."""

        def to(self, dtype=None, device=None):
            captured["dtype"] = dtype
            return self

        def eval(self):
            return self

    monkeypatch.setattr(reward_scoring, "reject_partial_checkpoint", lambda path: None)
    monkeypatch.setattr(
        reward_scoring,
        "AutoTokenizer",
        types.SimpleNamespace(from_pretrained=lambda *a, **k: types.SimpleNamespace()),
    )
    monkeypatch.setattr(
        reward_scoring, "AutoConfig", types.SimpleNamespace(from_pretrained=lambda *a, **k: Qwen3Config())
    )
    monkeypatch.setattr(reward_scoring, "from_pretrained_verified", lambda *a, **k: _Model())

    args = _rm_args(*(["--rm_dtype", requested] if requested else []))
    reward_scoring.load_reward_model(
        args.rm_model_id,
        args.rm_attn_implementation,
        args.rm_max_seq_len,
        "cpu",
        args.rm_dtype,
        trust_remote_code=False,
    )

    assert captured["dtype"] is expected


def test_the_gradio_apps_under_test_exist():
    assert len(_GRADIO_APPS) >= 2, f"expected the shipped gradio apps, found {[p.name for p in _GRADIO_APPS]}"
    holders = [p.name for p in _GRADIO_APPS if _gradio_app(p).build_parser().get_default("api_key") is not None]
    assert holders, "no gradio app holds an API key — the rules below cover nothing"


@pytest.mark.parametrize("app", _GRADIO_APPS, ids=lambda p: p.name)
def test_a_gradio_app_publishes_nothing_by_default(app):
    """``--host`` is a bind address and these apps hold a live API key: a ``0.0.0.0`` default hands that key's spend
    to anything that can route to the box, and ``--share`` to the internet. Read off the app's real parser."""
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
    """The app's real ``create_demo``, then queue + launch with the theme, as ``launch_gradio`` does. gradio's
    warnings are errors: a theme passed on ``Blocks`` is accepted with a warning and silently dropped."""
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


def _playground():
    return gradio_environment_playground


def test_the_environment_playground_key_defaults_to_the_vllm_placeholder(monkeypatch):
    """``AsyncOpenAI`` raises on ``api_key=None``, so a ``None`` default would fail every run against the documented
    keyless local server."""
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
        "build_openai_client",
        lambda base_url, api_key_override: AsyncOpenAI(
            base_url=base_url,
            api_key=api_key_override,
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(_handler)),
        ),
    )
    return mod


def test_the_environment_playground_sends_no_model_until_one_is_typed(monkeypatch):
    """The Model Name box ships empty and an empty ``"model"`` is sent verbatim, which a single-model server 404s.
    The URL box is hand-edited, so a scheme-less ``localhost:8000/v1`` must still route."""
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
    """Refused before any generation, not after a whole episode that cannot be graded."""
    seen = []
    mod = _mock_playground_client(monkeypatch, seen, content="done")
    with pytest.raises(ValueError, match="Expected Answer"):
        mod.run_playground_episode("swe", "fix it", "", "http://localhost:8000/v1", "EMPTY", "m", 0.7, 16, 0.95, 2)
    assert seen == []


def test_the_environment_playground_closes_the_environment_of_every_click(monkeypatch):
    """An unclosed environment keeps its sandbox sessions, server connections and scorer clients for the life of the
    app, and a refused click must close it too."""
    seen, closed = [], []
    mod = _mock_playground_client(monkeypatch, seen, content="Final Answer: 4")
    resolve = mod.build_environment

    def _resolve(env_type, config):
        env = resolve(env_type, config)
        close = env.close
        env.close = lambda: (closed.append(env_type), close())
        return env

    monkeypatch.setattr(mod, "build_environment", _resolve)
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
    """A local copy of the episode loop that forgets the stamp grades a mid-sentence fragment as the model's final
    answer."""
    seen = []
    mod = _mock_playground_client(monkeypatch, seen, finish_reason="length", content="Thought: I was cut off mid-")

    _messages, summary = mod.run_playground_episode(
        "native_math", "2+2?", "4", "http://localhost:8000/v1", "EMPTY", "m", 0.7, 16, 0.95, 2
    )

    assert "**Length-capped turns:** 2" in summary, summary
    assert len(seen) == 2, "a length-cut turn must be retried within max_turns, not finalized as an answer"


def test_the_generation_clis_share_one_concurrency_and_checkpoint_default():
    """The CLIs drive the same server through the same async client, so a per-script literal silently throttles a
    sibling and widens what an interrupted run must regenerate. dedup's ``--max_workers`` counts local threads."""
    expected = {"--max_workers": "DEFAULT_MAX_WORKERS", "--checkpoint_interval": "DEFAULT_CHECKPOINT_INTERVAL"}
    declared: dict[str, list] = {flag: [] for flag in expected}
    for script in sorted(_INFERENCE_ROOT.rglob("*.py")):
        source = script.read_text(encoding="utf-8")
        if script.name != "inference_arguments.py" and "add_openai_endpoint_args" not in source:
            continue
        for flag in expected:
            default = _argparse_default(source, flag)
            if default is not _ABSENT:
                declared[flag].append((script.name, default))

    for flag, constant in expected.items():
        assert getattr(inference_arguments, constant), (
            f"{constant} is not defined in scripts/inference/inference_arguments.py"
        )
        assert declared[flag], f"no inference CLI declares {flag} — this check covers nothing"
        literal = sorted(entry for entry in declared[flag] if entry[1] != constant)
        assert not literal, (
            f"{flag} must default to the shared {constant} from scripts/inference/inference_arguments.py, not to a "
            f"per-script literal; found {literal}"
        )


def test_the_local_endpoint_default_has_one_home():
    """Every CLI that defaults an endpoint takes it from ``add_openai_endpoint_args``, which reads
    ``halo.inference.endpoints``'s constant."""
    assert add_openai_endpoint_args(argparse.ArgumentParser()).get_default("base_url") is DEFAULT_LOCAL_BASE_URL
    for script in sorted((REPO_ROOT / "scripts").rglob("*.py")):
        source = script.read_text(encoding="utf-8")
        assert "DEFAULT_LOCAL_BASE_URL = " not in source, f"{script.name} re-declares the endpoint constant"


def _declared_flags(source: str) -> list[str]:
    return [
        arg.value
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call) and getattr(node.func, "attr", None) == "add_argument"
        for arg in node.args
        if isinstance(arg, ast.Constant) and isinstance(arg.value, str)
    ]


def test_every_endpoint_flag_comes_from_the_shared_helper():
    """A command line has to carry from one CLI to the next. ``scripts/profiling/`` is out of scope: its
    ``--server_url`` is the rollout server's root for the weight-sync group."""
    shared = add_openai_endpoint_args(argparse.ArgumentParser())
    shared_flags = {spelling for action in shared._actions for spelling in action.option_strings}
    assert {"--base_url", "--api_key"} <= {flag for flag in shared_flags if _ENDPOINT_FLAG.match(flag)}, (
        "the endpoint pattern no longer recognizes the shared spellings, so the sweep below covers nothing"
    )

    scripts_root = REPO_ROOT / "scripts"
    redeclared = [
        f"{script.relative_to(REPO_ROOT)}: {flag}"
        for script in sorted(scripts_root.rglob("*.py"))
        if script != scripts_root / "cli_arguments.py" and script.relative_to(scripts_root).parts[0] != "profiling"
        for flag in _declared_flags(script.read_text(encoding="utf-8"))
        if _ENDPOINT_FLAG.match(flag)
    ]
    assert not redeclared, (
        f"endpoint flags declared outside scripts/cli_arguments.py's add_openai_endpoint_args: {redeclared}"
    )


def _declared_action(parser: argparse.ArgumentParser, flag: str) -> argparse.Action:
    return next(action for action in parser._actions if flag in action.option_strings)


def test_every_dtype_flag_comes_from_the_shared_helper():
    """The scorer's ``--rm_dtype`` names a different model's dtype, not a different set of spellings or default."""
    scripts_root = REPO_ROOT / "scripts"
    redeclared = [
        f"{script.relative_to(REPO_ROOT)}: {flag}"
        for script in sorted(scripts_root.rglob("*.py"))
        if script != scripts_root / "cli_arguments.py"
        for flag in _declared_flags(script.read_text(encoding="utf-8"))
        if "dtype" in flag
    ]
    assert not redeclared, f"dtype flags declared outside scripts/cli_arguments.py's add_dtype_arg: {redeclared}"

    scorer = _declared_action(build_generation_parser("test", temperature_default=0.0), "--rm_dtype")
    shared = _declared_action(add_dtype_arg(argparse.ArgumentParser()), "--dtype")
    assert (scorer.default, scorer.choices) == (shared.default, shared.choices)


def test_the_prompt_row_fields_come_from_the_shared_helper():
    """The S3 generation CLI and the reward-model scorers read the same row fields, so one block declares them."""
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
        f"{script.relative_to(REPO_ROOT)}: {flag}"
        for script in sorted(_INFERENCE_ROOT.rglob("*.py"))
        if script != _INFERENCE_ROOT / "inference_arguments.py"
        for flag in _declared_flags(script.read_text(encoding="utf-8"))
        if flag in shared
    ]
    assert not redeclared, f"row-field flags declared outside add_prompt_field_args: {redeclared}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
