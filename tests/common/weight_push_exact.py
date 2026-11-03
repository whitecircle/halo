"""Body of the push-exactness suite: a push of the unchanged checkpoint leaves a live server's log-probs where they
were. Scored on the prompt log-probs of fixed token rows through the clients' own re-score, against the bar the row
measures: twice the server's run-to-run noise, with a floor for a server that repeats itself exactly. A second push
with one tensor scaled must clear that bar, or the bar proves nothing; text rows never reach a vision tensor, so the
scaled one sits in a text tower. After one real optimizer step, a push must leave the server computing what the trainer
computes, within the gap the two already had before the step."""

import argparse
import math
import re
import sys

import requests
import torch
import torch.distributed as dist
import torch.nn.functional as F
from transformers import AutoTokenizer

from halo.distributed.expert_parallel.moe_layer.layer import named_ep_layers
from halo.distributed.fsdp import reshard_fsdp2_modules
from halo.env import env_flag, env_str
from halo.optimizers.adamw_bf16 import AdamWBF16
from halo.trainers.grpo.rollout.weight_push import reject_unsyncable_model, sync_weights_to_client
from halo.weight_sync.registry import resolve_weight_sync_client, rollout_backends
from tests.common.harness import Ctx
from tests.common.models import (
    BAILING_MOE_LING_MINI,
    GEMMA4_26B_A4B,
    GLM4_FLASH,
    GPT_OSS_20B,
    LAGUNA_XS_2_1,
    LFM2_24B_A2B,
    QWEN3_0_6B,
    QWEN3_5_MOE_35B,
    QWEN3_6_MOE_35B,
    QWEN3_30B_A3B,
)
from tests.common.on_policy_e2e import PolicyLayout, load_policy
from tests.common.rank_log import log
from tests.common.weight_sync import SGLANG_SERVER_URL, weight_transfer_port

WORLD_SIZE = 2
# Every serving family plus Qwen3.6, a second release of Qwen3.5's block. Step-3.7 serves the tiny checkpoint its own
# suite writes (tests/gpu/trainers/grpo/test_step3p7_vllm_weight_sync_e2e.py --write-checkpoint).
CHECKPOINTS: dict[str, str | None] = {
    "bailing_moe": BAILING_MOE_LING_MINI,
    "gemma4_text": GEMMA4_26B_A4B,
    "glm4_moe_lite": GLM4_FLASH,
    "gpt_oss": GPT_OSS_20B,
    "laguna": LAGUNA_XS_2_1,
    "lfm2_moe": LFM2_24B_A2B,
    "qwen3": QWEN3_0_6B,
    "qwen3_5_moe_text": QWEN3_5_MOE_35B,
    "qwen3_6_moe_text": QWEN3_6_MOE_35B,
    "qwen3_moe": QWEN3_30B_A3B,
    "step3p7": env_str("HALO_TEST_STEP3P7_MODEL"),
}
# That checkpoint's auto_map exists for the server's config parse; the trainer takes the native class, as its suite does.
_TRAINER_LOAD_OVERRIDES = {"step3p7": {"trust_remote_code": False, "attn_implementation": "sdpa"}}
# Prose, code and a second language, so the rows reach many experts.
_TEXT = (
    "The river bends twice before it reaches the old mill, where the water slows and turns green. Farmers along its "
    "banks measure the season by the height of the reeds, and the miller keeps a ledger of every flood since his "
    "grandfather built the wheel. In 1887 the stone bridge was finished, and the first train crossed the valley three "
    "years later, carrying coal, letters and a brass band. def merge(a, b):\n    out = []\n    while a and b:\n"
    "        out.append(a.pop(0) if a[0] < b[0] else b.pop(0))\n    return out + a + b\nThe sum of the first ten "
    "primes is 129, and their product has ten digits. What would change if the river ran north instead? Une rivière "
    "coule sous le pont, et les enfants comptent les bateaux. "
)
_TEXT_REPEATS = 40
_ROW_LENGTHS = (32, 64, 128, 256, 512, 1024)
_ROW_STRIDE = 7
_BASELINE_RUNS = 3
_NOISE_MARGIN = 2
_BAR_FLOOR = 1e-5
_CONTROL_SCALE = 1.01
# A text attention weight past the first layer, whichever layer type the family starts with.
_CONTROL_TENSOR = re.compile(r"layers\.\d*[1-9]\.(self_attn|attention)\.")
_NO_TEXT_ROW_REACHES = re.compile(r"vision|visual|audio")
# What a push without a family's declared buffers does: vLLM's reload rebuilds the module they sit on from
# uninitialized storage, while SGLang's loader copies name by name.
_LEAVING_THEM_OUT_MOVES_THE_SERVER = {"vllm": True, "sglang": False}
_REAL_STEP_LEARNING_RATE = 2e-4
_REAL_STEP_ROW = 3  # 256 tokens, inside every server window
# The clone and Adam's two moments of every parameter do not fit beside ep1's copies on a 60 GB checkpoint.
_REAL_STEP_STRIDE = 4
_TRAINER_GAP_MARGIN = 2
_TRAINER_GAP_FLOOR = 0.01


def parse_row(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=sorted(CHECKPOINTS), required=True)
    parser.add_argument("--backend", choices=rollout_backends(), required=True)
    parser.add_argument("--ep-size", type=int, default=1)
    parser.add_argument("--tp-size", type=int, default=1)
    return parser.parse_args(argv)


def _server_url(backend: str) -> str:
    return SGLANG_SERVER_URL if backend == "sglang" else env_str("VLLM_SERVER_URL") or "http://localhost:8000"


def _row_checkpoint(family: str) -> str | None:
    return env_str("HALO_TEST_PUSH_EXACT_MODEL") or CHECKPOINTS[family]


def skip_unless_a_fresh_server_holds_the_row(row: argparse.Namespace) -> None:
    """Decline to run on a server that took another suite's push, which no later push can tell from a moved one, or
    serves another checkpoint; the launcher reads exit 0 with a ``SKIP:`` line and no result as a skip."""
    if not env_flag("HALO_TEST_PUSH_EXACT_FRESH"):
        log("SKIP: set HALO_TEST_PUSH_EXACT_FRESH=1 on a server that has taken no other suite's push")
        sys.exit(0)
    checkpoint = _row_checkpoint(row.family)
    if checkpoint is None:
        log(f"SKIP: --family {row.family} has no default checkpoint (set HALO_TEST_STEP3P7_MODEL)")
        sys.exit(0)
    url = _server_url(row.backend)
    served = requests.get(f"{url}/v1/models", timeout=60).json()["data"][0]["id"]
    if served != checkpoint:
        log(f"SKIP: {url} serves {served}, not this row's {checkpoint} (--served-model-name spells it)")
        sys.exit(0)


def _token_rows(checkpoint: str, limit: int | None) -> list[list[int]]:
    """Whichever fit the server's window, leaving room for the one generated token vLLM insists on."""
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
    ids = tokenizer(_TEXT * _TEXT_REPEATS, add_special_tokens=False)["input_ids"]
    return [
        ids[index * _ROW_STRIDE : index * _ROW_STRIDE + length]
        for index, length in enumerate(_ROW_LENGTHS)
        if limit is None or length < limit
    ]


def _score(client, rows: list[list[int]]) -> list[float]:
    return [value for row in rows for value in client.score_completion_logprobs(row[:1], row[1:])]


def _difference(reference: list[float], scores: list[float]) -> dict[str, float]:
    gaps = [abs(a - b) for a, b in zip(reference, scores, strict=True)]
    finite = all(math.isfinite(value) for value in (*reference, *scores))
    return {
        "max": max(gaps) if finite else math.inf,
        "mean": sum(gaps) / len(gaps) if finite else math.inf,
        "differ": sum(gap != 0 for gap in gaps),
    }


@torch.no_grad()
def _trainer_scores(model, rows: list[list[int]], device) -> list[float]:
    """COLLECTIVE: every rank forwards the same rows in the same order. The same positions ``_score`` reads."""
    scores: list[float] = []
    for row in rows:
        ids = torch.tensor([row], device=device)
        logits = model(input_ids=ids, use_cache=False).logits[0, :-1].float()
        scores += logits.log_softmax(-1).gather(1, ids[0, 1:].unsqueeze(1)).squeeze(1).tolist()
    reshard_fsdp2_modules(model)
    return scores


def _take_one_real_step(model, stepped: list[tuple[str, torch.nn.Parameter]], row: list[int], device) -> None:
    """COLLECTIVE. Cross-entropy off the logits, as a family's ``labels`` path may want training-only inputs."""
    optimizer = AdamWBF16(stepped, lr=_REAL_STEP_LEARNING_RATE, weight_decay=0.0)
    ids = torch.tensor([row], device=device)
    F.cross_entropy(model(input_ids=ids, use_cache=False).logits[0, :-1].float(), ids[0, 1:]).backward()
    reshard_fsdp2_modules(model)
    optimizer.step()
    model.zero_grad(set_to_none=True)


class _RoundSender:
    """What one round's gather sends through: the real client, less the names ``leave_out`` ends, with the first control
    tensor scaled when ``scale`` is set."""

    def __init__(self, client, *, scale: bool = False, leave_out: tuple[str, ...] = ()):
        self._client = client
        self._leave_out = leave_out
        self._scale = scale
        self.scaled: str | None = None
        self.forwarded = 0

    def scope_co_load_groups(self, module_names) -> None:
        self._client.scope_co_load_groups(module_names)

    def update_named_param(self, name: str, weights: torch.Tensor) -> None:
        if self._leave_out and name.endswith(self._leave_out):
            return
        self.forwarded += 1
        if (
            self._scale
            and self.scaled is None
            and weights.ndim >= 2
            and _CONTROL_TENSOR.search(name)
            and not _NO_TEXT_ROW_REACHES.search(name)
        ):
            self.scaled = name
            weights = weights * _CONTROL_SCALE
        self._client.update_named_param(name, weights)

    def finish_weight_update(self) -> None:
        self._client.finish_weight_update()

    def abort_weight_update(self) -> None:
        self._client.abort_weight_update()


def run_push_exactness(ctx: Ctx, row: argparse.Namespace) -> dict:
    """COLLECTIVE: every rank gathers each round; rank 0 forwards and scores."""
    checkpoint = _row_checkpoint(row.family)
    model, _ = load_policy(
        checkpoint,
        PolicyLayout(ep_size=row.ep_size, tp_size=row.tp_size).parallelism_config(),
        None,
        **_TRAINER_LOAD_OVERRIDES.get(row.family, {}),
    )
    reject_unsyncable_model(model, row.backend)
    group_port = weight_transfer_port(f"HALO_TEST_{row.backend.upper()}_GROUP_PORT")
    declared = tuple(
        sorted(
            {
                attr
                for _, layer in named_ep_layers(model)
                if layer.FAMILY.engine_loaded_buffers is not None
                for attr in layer.FAMILY.engine_loaded_buffers.attrs
            }
        )
    )

    client = rows = baseline = None
    checks: dict[str, bool] = {}
    metrics: dict[str, float] = {}
    bar = _BAR_FLOOR
    if ctx.rank == 0:
        client = resolve_weight_sync_client(row.backend)(
            base_url=_server_url(row.backend), group_port=group_port, connection_timeout=600
        )
        client.init_communicator(device=ctx.device)
    window = [client.served_max_model_len() if ctx.rank == 0 else None]
    dist.broadcast_object_list(window, src=0)
    rows = _token_rows(checkpoint, window[0])
    if ctx.rank == 0:
        baseline, *repeats = [_score(client, rows) for _ in range(_BASELINE_RUNS)]
        noise = max(_difference(baseline, repeat)["max"] for repeat in repeats)
        bar = max(_NOISE_MARGIN * noise, _BAR_FLOOR)
        checks["baseline_is_finite"] = math.isfinite(noise)
        metrics |= {"noise": noise, "bar": bar, "positions": len(baseline)}
        log(f"  {checkpoint} on {row.backend}: {len(baseline)} positions, noise {noise:.3e}, bar {bar:.3e}")

    def pushed(label: str, **sender_args) -> tuple[dict[str, float], _RoundSender | None]:
        """One full push and, on rank 0, how far it left the served log-probs from the baseline."""
        sender = _RoundSender(client, **sender_args) if ctx.rank == 0 else None
        sync_weights_to_client(model, sender, ctx.rank == 0, True)
        ctx.barrier()
        if ctx.rank != 0:
            return {}, None
        moved = _difference(baseline, _score(client, rows))
        metrics.update({f"{label}/{key}": value for key, value in moved.items()})
        metrics[f"{label}/forwarded"] = sender.forwarded
        log(f"  {label}: {moved} over {sender.forwarded} forwarded tensors, scaled {sender.scaled}")
        return moved, sender

    def stays(moved: dict[str, float]) -> bool:
        return moved["max"] <= bar

    try:
        moved, _ = pushed("unchanged")
        if ctx.rank == 0:
            checks["unchanged_push_leaves_the_served_log_probs"] = stays(moved)

        moved, sender = pushed("scaled", scale=True)
        if ctx.rank == 0:
            checks["control_scaled_a_text_tensor"] = sender.scaled is not None
            checks["control_moved_past_the_bar"] = not stays(moved)

        moved, _ = pushed("restored_after_scaling")
        if ctx.rank == 0:
            checks["unchanged_push_restores_the_control"] = stays(moved)

        if declared:
            moved, _ = pushed("buffers_left_out", leave_out=declared)
            if ctx.rank == 0:
                checks["leaving_the_declared_buffers_out_acts_as_the_engine_loader_says"] = (
                    stays(moved) != _LEAVING_THEM_OUT_MOVES_THE_SERVER[row.backend]
                )

            moved, _ = pushed("restored_after_leaving_out")
            if ctx.rank == 0:
                checks["unchanged_push_restores_the_buffers"] = stays(moved)

        # TP's replicated parameters take the trainer's gradient sweep, which this one step does not run.
        if row.tp_size == 1:
            model.eval()
            before_step = _trainer_scores(model, rows, ctx.device)
            trainable = [(name, param) for name, param in model.named_parameters() if param.requires_grad]
            stepped = trainable[::_REAL_STEP_STRIDE]
            saved = [(param, param.detach().clone()) for _, param in stepped]
            _take_one_real_step(model, stepped, rows[_REAL_STEP_ROW], ctx.device)
            after_step = _trainer_scores(model, rows, ctx.device)
            moved, _ = pushed("after_a_real_step")
            if ctx.rank == 0:
                floor = _difference(before_step, baseline)
                stale = _difference(after_step, baseline)
                served = _difference(after_step, _score(client, rows))
                metrics |= {"trainer_gap/floor": floor["mean"], "trainer_gap/stale": stale["mean"]}
                metrics["trainer_gap/after_push"] = served["mean"]
                log(f"  trainer against served: floor {floor}, stale {stale}, after the push {served}")
                checks["a_real_step_moved_the_trainer_past_the_floor"] = (
                    stale["mean"] > _TRAINER_GAP_MARGIN * floor["mean"]
                )
                checks["a_push_after_a_real_step_moved_the_server"] = not stays(moved)
                checks["a_push_after_a_real_step_serves_the_trainer"] = served["mean"] <= max(
                    _TRAINER_GAP_MARGIN * floor["mean"], _TRAINER_GAP_FLOOR
                )
            with torch.no_grad():
                for param, weights in saved:
                    param.copy_(weights)
            moved, _ = pushed("restored_after_the_real_step")
            if ctx.rank == 0:
                checks["unchanged_push_restores_the_real_step"] = stays(moved)
    finally:
        if client is not None:
            client.close_communicator()
    return {"checks": ctx.broadcast_checks(checks), "metrics": metrics}
