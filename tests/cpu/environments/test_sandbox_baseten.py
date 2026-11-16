"""BasetenSandbox against a local fake of the Baseten Sandboxes API (token, instances, filesystem, process). Each fake
sandbox runs commands for real under its own root directory, so these tests check the wire contract, per-session
isolation and the compile/run/reset behavior offline, without Baseten credentials."""

import contextlib
import json
import logging
import os
import shutil
import subprocess
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote

import pytest

from halo.environments.sandbox.baseten import BASETEN_SANDBOX_LIMIT, BasetenSandbox
from halo.environments.sandbox.factory import build_sandbox
from halo.environments.tasks.coding.code_contests import CodeContestsEnvironment

API_KEY = "bt-secret-key-0123456789"
TOKEN = "sbx-token-1"
PREFIX = "/tmp/halo"


class FakeBaseten:
    """Management API under /v1; sandbox ``sbx-N``'s runtime API under /sbx-N, its ``/tmp/halo`` mapped to a private
    host directory. ``faults`` maps a path prefix to statuses returned once each before the real handler runs."""

    def __init__(self):
        self.requests: list[dict] = []
        self.faults: dict[str, list[int]] = {}
        self.delay: dict[str, float] = {}
        self.roots: dict[str, str] = {}
        self.deleted: list[str] = []
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def _reply(self, status, payload=None, raw=None, ctype="application/json"):
                data = raw if raw is not None else json.dumps(payload or {}).encode()
                self.send_response(status)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(data)))
                if status == 429:
                    self.send_header("Retry-After", "0")
                self.end_headers()
                with contextlib.suppress(BrokenPipeError):
                    self.wfile.write(data)

            def _handle(self):
                length = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(length)) if length else None
                fake.requests.append(
                    {
                        "method": self.command,
                        "path": self.path,
                        "auth": self.headers.get("Authorization"),
                        "body": body,
                    }
                )
                for prefix, statuses in fake.faults.items():
                    if self.path.startswith(prefix) and statuses:
                        return self._reply(statuses.pop(0), {"error": "injected"})
                for prefix, seconds in fake.delay.items():
                    if self.path.startswith(prefix):
                        time.sleep(seconds)
                if self.path == "/v1/token":
                    if self.headers.get("Authorization") != f"Bearer {API_KEY}":
                        return self._reply(401, {"error": "bad key"})
                    return self._reply(200, {"token": TOKEN, "expires_at": "2099-01-01T00:00:00Z"})
                if self.headers.get("Authorization") != f"Bearer {TOKEN}":
                    return self._reply(401, {"error": "bad token"})
                if self.path == "/v1/sandboxes/instances":
                    name = f"sbx-{len(fake.roots) + 1}"
                    fake.roots[name] = tempfile.mkdtemp(prefix=f"fake-{name}-")
                    url = f"http://127.0.0.1:{fake.port}/{name}/"
                    return self._reply(201, {"name": name, "url": url, "status": "DEPLOYED"})
                if self.path.startswith("/v1/sandboxes/instances/") and self.command == "DELETE":
                    fake.deleted.append(self.path.rsplit("/", 1)[-1])
                    return self._reply(202, {"status": "DELETING"})
                name, _, rest = self.path.lstrip("/").partition("/")
                root = fake.roots.get(name)
                if root is None or name in fake.deleted:
                    return self._reply(404, {"error": "no sandbox"})

                def mapped(text: str) -> str:
                    return text.replace(PREFIX, root + PREFIX)

                if rest.startswith("filesystem/"):
                    path = mapped(unquote(rest[len("filesystem/") :]))
                    if self.command == "PUT":
                        if not os.path.isdir(os.path.dirname(path)):
                            return self._reply(422, {"error": "parent missing"})
                        with open(path, "w") as fh:
                            fh.write(body["content"])
                        return self._reply(200, {"message": "File created/updated successfully"})
                    if not os.path.isfile(path):
                        return self._reply(404, {"error": "not found"})
                    with open(path, "rb") as fh:
                        return self._reply(200, raw=fh.read(), ctype="application/octet-stream")
                if rest == "process":
                    workdir = mapped(body["workingDir"])
                    os.makedirs(root + PREFIX, exist_ok=True)
                    try:
                        proc = subprocess.run(
                            mapped(body["command"]), shell=True, cwd=workdir, capture_output=True, text=True,
                            timeout=body["timeout"],
                        )  # fmt: skip
                        out = {"status": "completed", "exitCode": proc.returncode}
                        out |= {"stdout": proc.stdout, "stderr": proc.stderr}
                    except subprocess.TimeoutExpired:
                        out = {"status": "killed", "exitCode": -1, "stdout": "", "stderr": ""}
                    return self._reply(200, out)
                return self._reply(404, {"error": "no route"})

            do_GET = do_POST = do_PUT = do_DELETE = _handle

            def log_message(self, *args):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = self.server.server_address[1]
        self.api_url = f"http://127.0.0.1:{self.port}/v1"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def calls(self, path_prefix: str, method: str | None = None) -> list[dict]:
        return [
            r for r in self.requests if r["path"].startswith(path_prefix) and (method is None or r["method"] == method)
        ]

    def program_runs(self) -> list[dict]:
        return [r["body"] for r in self.requests if r["path"].endswith("/process") and "exec " in r["body"]["command"]]


@pytest.fixture
def fake(monkeypatch):
    monkeypatch.setenv("BASETEN_API_KEY", API_KEY)
    monkeypatch.delenv("HALO_SANDBOX_URL", raising=False)
    server = FakeBaseten()
    yield server
    server.server.shutdown()
    server.server.server_close()
    for root in server.roots.values():
        shutil.rmtree(root, ignore_errors=True)


def _sandbox(fake, **kwargs):
    return BasetenSandbox(api_url=fake.api_url, backoff_factor=0, **kwargs)


def test_request_shape_and_auth(fake):
    result = _sandbox(fake, cpu=4, idle_ttl="10m").run("print('hi')", timeout=7)

    assert result.ok and result.stdout == "hi\n"
    (token_req,) = fake.calls("/v1/token")
    assert token_req["auth"] == f"Bearer {API_KEY}"
    assert token_req["body"] == {"scopes": ["sandboxes"]}
    (create,) = fake.calls("/v1/sandboxes/instances", "POST")
    assert create["body"]["cpu"] == 4
    assert create["body"]["image"] == "baseten/base-image:latest"
    assert create["body"]["lifecycle"]["expiration_policies"] == [
        {"type": "TTL_IDLE", "value": "10m", "action": "DELETE"}
    ]
    (run_call,) = fake.program_runs()
    assert run_call["waitForCompletion"] is True
    assert run_call["timeout"] == 7
    assert run_call["workingDir"] == "/tmp/halo/work"
    # The API key only ever mints the token; every sandbox call carries the token.
    assert all(r["auth"] == f"Bearer {TOKEN}" for r in fake.requests if r["path"] != "/v1/token")


def test_one_shot_run_deletes_its_sandbox(fake):
    _sandbox(fake).run("print(1)")

    assert fake.deleted == ["sbx-1"]


def test_each_session_gets_its_own_sandbox(fake):
    sb = _sandbox(fake)
    with sb.open_session() as first, sb.open_session() as second:
        first.write_file("secret.txt", "first episode")
        second.run("print(1)")
        peek = second.run("import os; print(os.path.exists('secret.txt'))")

        assert peek.stdout == "False\n"
        assert second.read_file("secret.txt") is None
        assert first.read_file("secret.txt") == "first episode"
    assert sorted(fake.deleted) == ["sbx-1", "sbx-2"]


def test_session_state_persists_and_stdin_is_fed(fake):
    with _sandbox(fake).open_session() as session:
        session.write_file("pkg/helper.py", "VALUE = 41\n")
        result = session.run("from pkg.helper import VALUE\nprint(VALUE + int(input()))", stdin="1\n")

        assert result.stdout == "42\n"
        assert {"main.py", "pkg/helper.py"} <= set(session.list_files())
        assert session.read_file("missing.txt") is None


def test_reset_to_prepared_drops_program_output_and_keeps_source(fake):
    code = "open('out.txt', 'a').write('x')\nprint(open('out.txt').read())"
    with _sandbox(fake).open_session() as session:
        assert session.run(code).stdout == "x\n"
        session.reset_to_prepared()
        assert session.read_file("out.txt") is None
        assert session.read_file("main.py") == code
        assert session.run(code).stdout == "x\n"


def test_unchanged_program_is_not_rewritten_or_rebuilt(fake):
    with _sandbox(fake).open_session() as session:
        session.run("print(1)")
        puts = len(fake.calls("/sbx-1/filesystem", "PUT"))
        session.run("print(1)", stdin="ignored")

        # Only the new stdin is written on the second run.
        assert len(fake.calls("/sbx-1/filesystem", "PUT")) == puts + 1


def test_nonzero_exit_is_a_program_verdict(fake):
    result = _sandbox(fake).run("import sys; sys.exit(3)")

    assert result.returncode == 3 and result.error is None and not result.ok


def test_retries_5xx_then_succeeds(fake):
    fake.faults["/v1/sandboxes/instances"] = [503, 502]

    result = _sandbox(fake).run("print('ok')")

    assert result.stdout == "ok\n"
    assert len(fake.calls("/v1/sandboxes/instances", "POST")) == 3


def test_retry_budget_on_429_is_bounded_and_reported(fake):
    fake.faults["/v1/sandboxes/instances"] = [429, 429, 429]

    result = _sandbox(fake, max_retries=1).run("print(1)")

    assert result.error is not None and "HTTP 429" in result.error
    assert len(fake.calls("/v1/sandboxes/instances", "POST")) == 2


def test_auth_failure_is_an_error_and_not_retried(fake, monkeypatch):
    monkeypatch.setenv("BASETEN_API_KEY", "wrong")

    result = _sandbox(fake).run("print(1)")

    assert result.error is not None and "HTTP 401" in result.error
    assert len(fake.calls("/v1/token")) == 1


def test_failed_create_releases_its_slot(fake):
    fake.faults["/v1/sandboxes/instances"] = [400]
    sb = _sandbox(fake, max_concurrency=1)

    assert sb.run("print(1)").error is not None
    assert sb.slots.acquire(timeout=1), "a failed create must give its slot back"


def test_program_time_limit_is_killed_server_side(fake):
    result = _sandbox(fake).run("import time; time.sleep(10)", timeout=1)

    assert result.timed_out and result.returncode is None and result.error is None


def test_service_not_answering_is_an_error_and_not_rerun(fake):
    sb = _sandbox(fake, request_overhead=0.2, max_retries=3)
    with sb.open_session() as session:
        session.write_file("warm.txt", "")
        fake.delay["/sbx-1/process"] = 2.0
        result = session.run("print(1)", timeout=0.5)
        fake.delay.clear()

    assert result.error is not None and "timed out" in result.error and not result.timed_out
    assert len(fake.program_runs()) == 1


@pytest.mark.skipif(shutil.which("gcc") is None, reason="needs gcc in the image")
def test_compile_error_is_a_compile_verdict(fake):
    result = _sandbox(fake).run("int main( {", language="c")

    assert result.compile_failed and result.error is None and result.stderr


def test_unsafe_path_refused(fake):
    with _sandbox(fake).open_session() as session, pytest.raises(ValueError):
        session.write_file("../escape.py", "x")


def test_missing_api_key_refused_at_construction(monkeypatch):
    monkeypatch.delenv("BASETEN_API_KEY", raising=False)

    with pytest.raises(ValueError, match="BASETEN_API_KEY"):
        BasetenSandbox()


def test_concurrency_past_the_org_limit_refused_at_construction(fake):
    with pytest.raises(ValueError, match="at most 10"):
        _sandbox(fake, max_concurrency=BASETEN_SANDBOX_LIMIT + 1)


def test_api_key_never_logged(fake, caplog):
    with caplog.at_level(logging.DEBUG):
        result = _sandbox(fake).run("print(1)")

    assert result.ok
    assert API_KEY not in caplog.text and TOKEN not in caplog.text


def test_factory_and_env_config_select_baseten(fake, monkeypatch):
    monkeypatch.setenv("HALO_SANDBOX_BACKEND", "baseten")
    assert isinstance(build_sandbox(), BasetenSandbox)

    env = CodeContestsEnvironment(sandbox_backend="baseten", sandbox_options={"cpu": 4, "api_url": fake.api_url})

    assert isinstance(env.sandbox, BasetenSandbox)
    assert env.sandbox.api_url == fake.api_url and env.sandbox.cpu == 4


def test_factory_refuses_a_sandbox_url_for_baseten(fake, monkeypatch):
    monkeypatch.setenv("HALO_SANDBOX_URL", "http://sandboxfusion:8080")

    with pytest.raises(ValueError, match="takes no url"):
        build_sandbox(backend="baseten")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
