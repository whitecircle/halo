"""HTTP stand-ins for the rollout engines' control planes, recording every request they receive.

Each fake keeps the engine state a client's protocol depends on and acts on a request before it
replies, as both engines' handlers do, so a scripted failure reply still leaves its state change.
"""

import json
import re
import threading
from collections import deque
from datetime import timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch
from torch.distributed import TCPStore

from src.weight_sync.clients.base import WeightSyncClientBase
from src.weight_sync.clients.sglang import SGLangWeightSyncClient
from src.weight_sync.clients.vllm import VLLMWeightSyncClient
from tests.common.repo import REPO_ROOT

# vLLM's default CUDA-graph capture ceiling on H100/H200 (1024 on B200/B300): prefills up to this many
# tokens replay a captured graph.
GRAPH_CAPTURE_TOKENS = 512
# Stamped on each request by a recorder that needs to tell a request's retries apart from new requests.
CALL_ID_HEADER = "X-Halo-Call-Id"


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"

    def log_message(self, *args):
        pass

    def do_GET(self):
        self.server.fake.handle(self, "GET", None)

    def do_POST(self):
        raw = self.rfile.read(int(self.headers.get("Content-Length") or 0))
        self.server.fake.handle(self, "POST", json.loads(raw) if raw else None)


class FakeEngineServer:
    """One engine's routes on a loopback port, with failures scripted per path.

    ``status[path]`` answers every request on ``path`` with that code, and ``replies[path]`` answers the
    next ones in order before ``status`` applies again. ``refuse`` answers 503 before the engine acts, a
    request that never reached it. ``hold`` keeps the reply until :meth:`release`, an engine blocked in
    the call. A route the engine does not serve answers 404.
    """

    # The client this fake answers.
    CLIENT_CLASS: type[WeightSyncClientBase]
    # What its ``/v1/completions`` logprobs account for: vLLM's raw default, its processed mode
    # (temperature and nucleus renormalization), or SGLang's default (temperature only).
    LOGPROBS_MODE: str

    def __init__(self):
        self.paused = False
        # Declared on the model card only when set, as a server without one reports no context window.
        self.max_model_len: int | None = None
        self.logprobs_mode = self.LOGPROBS_MODE
        # Added to every prefill log-prob of a request within the capture range: an engine whose prompt
        # log-probs depend on whether the prefill replayed a graph (the V1 runner's spec-decode overwrite).
        self.graph_prefill_shift = 0.0
        self.graph_capture_tokens = GRAPH_CAPTURE_TOKENS
        self.requests: list[str] = []
        self.receipts: list[str | None] = []
        self.bodies: dict[str, dict] = {}
        self.posted: list[tuple[str, dict]] = []
        self.queries: dict[str, str] = {}
        self.status: dict[str, int] = {}
        self.replies: dict[str, deque[int]] = {}
        self.refuse: set[str] = set()
        self.hold: set[str] = set()
        self._released = threading.Event()
        # Guards the records above and wakes a waiter on each arrival.
        self._lock = threading.Condition()
        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self._httpd.fake = self
        self.netloc = f"127.0.0.1:{self._httpd.server_address[1]}"
        self.url = f"http://{self.netloc}"
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()

    @staticmethod
    def prefill_logprob(token: int, shift: float) -> float:
        """The log-prob this fake scores ``token`` at in a prefill, a function of the id alone."""
        return -((token % 7) + 1) / 10 + shift

    def prefill_shift(self, num_tokens: int) -> float:
        return self.graph_prefill_shift if num_tokens <= self.graph_capture_tokens else 0.0

    def handle(self, request: BaseHTTPRequestHandler, method: str, body: dict | None) -> None:
        path, _, query = request.path.partition("?")
        with self._lock:
            self.requests.append(f"{method} {path}")
            self.receipts.append(request.headers.get(CALL_ID_HEADER))
            if method == "POST":
                self.queries[path] = query
                if body is not None:
                    self.bodies[path] = body
                    self.posted.append((path, body))
            if path in self.refuse:
                code, reply = 503, {}
            else:
                code, reply = self.answer(method, path, body) or (404, {})
                scripted = self.replies.get(path)
                code = scripted.popleft() if scripted else self.status.get(path, code)
            self._lock.notify_all()
        if path in self.hold:
            self._released.wait()
        payload = json.dumps(reply).encode()
        request.send_response(code)
        request.send_header("Content-Type", "application/json")
        request.send_header("Content-Length", str(len(payload)))
        request.end_headers()
        request.wfile.write(payload)

    def answer(self, method: str, path: str, body: dict | None) -> tuple[int, dict] | None:
        """Act on one request and return ``(status, reply)``, or ``None`` for a route the engine lacks.
        Called under the server's lock."""
        if (method, path) == ("GET", "/health"):
            return 200, {}
        if (method, path) == ("GET", "/v1/models"):
            card = {"id": "fake-model"}
            if self.max_model_len is not None:
                card["max_model_len"] = self.max_model_len
            return 200, {"data": [card]}
        if (method, path) == ("POST", "/v1/completions"):
            values = self._completion_logprobs(body)
            top = {f"tok{i}": value for i, value in enumerate(values)}
            choice = {"text": "tok0", "logprobs": {"token_logprobs": [values[0]], "top_logprobs": [top]}}
            return 200, {"choices": [choice]}
        return None

    def _completion_logprobs(self, body: dict) -> list[float]:
        """Top logprobs of a fixed 3-token prefill under the server's logprobs mode."""
        logits = torch.tensor([2.0, 0.0, -1.0])
        if self.logprobs_mode == "raw_logprobs":
            return torch.log_softmax(logits, dim=0).tolist()
        logprobs = torch.log_softmax(logits / float(body.get("temperature", 1.0)), dim=0)
        if self.logprobs_mode == "processed_logprobs" and float(body.get("top_p", 1.0)) < 1.0:
            probs = logprobs.exp()
            keep = probs.cumsum(0) - probs < float(body["top_p"])
            logprobs = torch.where(keep, torch.log(probs / probs[keep].sum()), torch.full_like(probs, -1e9))
        return logprobs.tolist()

    def count(self, entry: str) -> int:
        with self._lock:
            return self.requests.count(entry)

    def wait_for_request(self, entry: str, timeout: float = 30.0) -> None:
        """Block until a request ``entry`` (``"<METHOD> <path>"``) arrived, raising after ``timeout`` seconds."""
        with self._lock:
            if not self._lock.wait_for(lambda: entry in self.requests, timeout):
                raise TimeoutError(f"{entry} never reached {self.url}")

    def received(self, call_id: str) -> int:
        """How many requests carrying ``call_id`` in :data:`CALL_ID_HEADER` arrived: one, plus its retries."""
        with self._lock:
            return self.receipts.count(call_id)

    def release(self) -> None:
        """Answer every held request, and every later one at once."""
        self._released.set()

    def close(self) -> None:
        self.release()
        self._httpd.shutdown()
        self._httpd.server_close()


class FakeVLLMServer(FakeEngineServer):
    """vLLM's weight-transfer routes: tracks whether the engine is paused and its layerwise reload open.

    Its ``world_size`` workers join the trainer's rendezvous store on ``/init_weight_transfer_engine``,
    in the background as the engine's workers do; 0 leaves the trainer a group of one. It reports the
    release ``Dockerfile.vllm`` builds, and refuses an init or update request whose fields are not the
    ones that release's request dataclasses declare, as the engine does.
    """

    CLIENT_CLASS = VLLMWeightSyncClient
    LOGPROBS_MODE = "raw_logprobs"
    VERSION = re.search(r"^ARG VLLM_VERSION=(\S+)$", (REPO_ROOT / "Dockerfile.vllm").read_text(), re.M).group(1)
    INIT_INFO_FIELDS = frozenset(
        {
            "master_address",
            "master_port",
            "rank_offset",
            "world_size",
            "nccl_unique_id_b64",
            "packed",
            "packed_buffer_size_bytes",
            "packed_num_buffers",
        }
    )
    UPDATE_INFO_FIELDS = frozenset({"names", "dtype_names", "shapes"})

    def __init__(self):
        super().__init__()
        self.version = self.VERSION
        self.update_open = False
        self.world_size = 1
        # Held so the workers' store connections live until close(), as an engine's workers stay joined.
        self.joined_stores: list[TCPStore] = []

    def answer(self, method: str, path: str, body: dict | None) -> tuple[int, dict] | None:
        if (method, path) == ("GET", "/is_paused"):
            return 200, {"is_paused": self.paused}
        if (method, path) == ("GET", "/get_world_size"):
            return 200, {"world_size": self.world_size}
        if (method, path) == ("GET", "/version"):
            return 200, {} if self.version is None else {"version": self.version}
        if method == "POST" and path == "/v1/completions" and "prompt_logprobs" in (body or {}):
            # Prefill echo: one entry per prompt token (None for the first), keyed by token id.
            prompt = body["prompt"]
            shift = self.prefill_shift(len(prompt))
            echo = [None] + [{str(tok): {"logprob": self.prefill_logprob(tok, shift)}} for tok in prompt[1:]]
            return 200, {"choices": [{"text": "", "prompt_logprobs": echo}]}
        if method == "POST" and path == "/init_weight_transfer_engine":
            init = body["init_info"]
            if unknown := set(init) - self.INIT_INFO_FIELDS:
                return 500, {"error": f"init_info carries fields the engine does not declare: {sorted(unknown)}"}
            address = (init["master_address"], init["master_port"], init["world_size"])
            for _ in range(self.world_size):
                threading.Thread(target=self._join_store, args=address, daemon=True).start()
            return 200, {}
        if method == "POST" and path in ("/pause", "/resume"):
            self.paused = path == "/pause"
            return 200, {}
        if method == "POST" and path in ("/start_weight_update", "/finish_weight_update"):
            self.update_open = path == "/start_weight_update"
            return 200, {}
        if method == "POST" and path == "/update_weights":
            if unknown := set(body["update_info"]) - self.UPDATE_INFO_FIELDS:
                return 500, {"error": f"update_info carries fields the engine does not declare: {sorted(unknown)}"}
            return 200, {}
        return super().answer(method, path, body)

    def _join_store(self, host: str, port: int, world_size: int) -> None:
        store = TCPStore(
            host_name=host, port=port, world_size=world_size, is_master=False, timeout=timedelta(seconds=30)
        )
        self.joined_stores.append(store)

    def close(self) -> None:
        super().close()
        self.joined_stores.clear()


class FakeSGLangServer(FakeEngineServer):
    """SGLang's weight-update routes: tracks whether generation is paused, the group names it holds and
    whether a weight-update session is open.

    A join under a name it already holds is rejected, as the engine registers a group by name when
    asked to join. An update outside a session, a second begin and an end without a begin are refused,
    as the engine refuses them. The routes that parse a request object reject a request without a JSON
    body.
    """

    CLIENT_CLASS = SGLangWeightSyncClient
    LOGPROBS_MODE = "sglang_default"
    _REQUEST_OBJECT_ROUTES = frozenset(
        {
            "/init_weights_update_group",
            "/destroy_weights_update_group",
            "/update_weights_from_distributed",
            "/begin_weight_update",
            "/end_weight_update",
            "/pause_generation",
            "/continue_generation",
            "/generate",
        }
    )

    def __init__(self):
        super().__init__()
        self.tp_size = 1
        self.groups: set[str] = set()
        self.session_open = False

    def answer(self, method: str, path: str, body: dict | None) -> tuple[int, dict] | None:
        if (method, path) == ("GET", "/server_info"):
            return 200, {"tp_size": self.tp_size, "dp_size": 1}
        if method != "POST":
            return super().answer(method, path, body)
        if path in self._REQUEST_OBJECT_ROUTES and body is None:
            return 422, {"detail": "a request body is required"}
        if path == "/init_weights_update_group":
            if body["group_name"] in self.groups:
                return 500, {"message": f"group name {body['group_name']} has already been created"}
            self.groups.add(body["group_name"])
            return 200, {}
        if path == "/destroy_weights_update_group":
            self.groups.discard(body["group_name"])
            return 200, {}
        if path in ("/pause_generation", "/continue_generation"):
            self.paused = path == "/pause_generation"
            return 200, {}
        if path in ("/begin_weight_update", "/end_weight_update"):
            if self.session_open == (path == "/begin_weight_update"):
                return 400, {"message": "begin_weight_update() and end_weight_update() must alternate"}
            self.session_open = path == "/begin_weight_update"
            return 200, {}
        if path == "/update_weights_from_distributed":
            if not self.session_open:
                return 400, {"message": "must run between begin_weight_update() and end_weight_update()"}
            return 200, {}
        if path == "/flush_cache":
            return 200, {}
        if path == "/generate" and body.get("return_logprob"):
            # [logprob, token_id, text] per input token from logprob_start_len; the window's first
            # entry is its anchor and carries None.
            shift = self.prefill_shift(len(body["input_ids"]))
            ids = body["input_ids"][body["logprob_start_len"] :]
            echo = [[None, ids[0], None]] + [[self.prefill_logprob(tok, shift), tok, None] for tok in ids[1:]]
            return 200, {"meta_info": {"input_token_logprobs": echo}}
        return super().answer(method, path, body)


def fake_server_class(client_class: type[WeightSyncClientBase]) -> type[FakeEngineServer]:
    """The fake answering ``client_class``."""
    for cls in FakeEngineServer.__subclasses__():
        if cls.CLIENT_CLASS is client_class:
            return cls
    raise KeyError(f"no fake engine server answers {client_class.__qualname__}")
