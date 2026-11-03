"""Build-time checks of the vLLM server image: the stack pins, the runtime patches and the contracts the
weight-sync client relies on (the Triton-interpreter checks live beside their patches and plugins).

No test suite can run these: they import vLLM, which the training image does not have (ABI-incompatible
stacks), so the image build is the only gate. Each check fails the build on an upstream change that would
otherwise surface as a hung group, a silently reverted sync or a wrong serve-time result:

    python3 verify_image.py
"""

import asyncio
import dataclasses
import inspect
import typing
from importlib.metadata import requires, version
from unittest.mock import AsyncMock, MagicMock

import vllm
import vllm_layerwise_reload_patch as layerwise_patch
import vllm_reasoning_forced_end_patch as forced_end_patch
import vllm_thinking_budget_completion_patch as completion_patch
import vllm_weight_transfer_reinit_patch as reinit_patch
from packaging.requirements import Requirement
from vllm.config.kernel import KernelConfig
from vllm.config.reasoning import ReasoningConfig
from vllm.distributed.device_communicators.pynccl_wrapper import ncclUniqueId
from vllm.distributed.weight_transfer import packed_tensor
from vllm.distributed.weight_transfer.nccl_engine import (
    NCCLWeightTransferEngine,
    NCCLWeightTransferInitInfo,
    NCCLWeightTransferUpdateInfo,
)
from vllm.engine.arg_utils import EngineArgs
from vllm.entrypoints.rl.online.api_router import pause_generation, router
from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
from vllm.model_executor.model_loader.reload import layerwise
from vllm.model_executor.model_loader.reload.meta import SKIP_LOAD_TENSORS, SKIP_TENSORS
from vllm.model_executor.models.gemma4 import Gemma4Router
from vllm.model_executor.models.gpt_oss import OAIAttention
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.engine.core import EngineCore, EngineCoreProc
from vllm.v1.worker.gpu.sample.thinking_budget import ThinkingBudgetState

from halo.weight_sync.transport.nccl_library import ncclUniqueId as shim_unique_id


def check_engine_stack() -> None:
    """transformers and tokenizers stay at the base image's own, inside the range vLLM declares."""
    print(f"vLLM: {vllm.__version__}")
    reqs = [r for r in map(Requirement, requires("vllm")) if r.name in ("transformers", "tokenizers")]
    bad = [
        f"{r.name} {version(r.name)} not in {r.specifier}"
        for r in reqs
        if not r.specifier.contains(version(r.name), prereleases=True)
    ]
    assert len(reqs) == 2 and not bad, bad or reqs
    print("transformers/tokenizers within vLLM requirement:", {r.name: version(r.name) for r in reqs})


def check_unique_id_pickle_shim() -> None:
    """The trainer sends ``ncclUniqueId`` pickled as its own module's type, which must unpickle as vLLM's."""
    assert shim_unique_id is ncclUniqueId
    print("ncclUniqueId pickle compat: OK")


def check_layerwise_reload_patch() -> None:
    """vLLM's layerwise reload puts each layer's tensors on the meta device, so a layer that writes its
    weights with a direct ``param.copy_()`` reverts on every /update_weights unless the patch skips it."""
    assert layerwise_patch._APPLIED, "sitecustomize hook did not fire on layerwise import"
    assert layerwise.restore_layer_on_meta.__module__ == layerwise_patch.__name__, "restore_layer_on_meta not patched"
    assert layerwise.initialize_online_processing.__module__ == layerwise_patch.__name__, (
        "initialize_online_processing not patched"
    )
    print(f"layerwise reload patch: OK (skips {sorted(layerwise_patch.SKIP_LAYER_NAMES)})")


def check_layerwise_skip_coverage() -> None:
    """The reload skip list covers each direct-copy layer, by MRO, in this vLLM: an upstream rename fails
    the build, not the run."""

    def covered(cls) -> bool:
        return any(c.__name__ in layerwise_patch.SKIP_LAYER_NAMES for c in cls.__mro__)

    assert covered(RoutedExperts), "expert-weight owner RoutedExperts not covered by SKIP_LAYER_NAMES"
    assert covered(OAIAttention), "sinks owner OAIAttention not covered by SKIP_LAYER_NAMES"
    assert covered(Gemma4Router), "Gemma4Router (scale param + root_size buffer) not covered by SKIP_LAYER_NAMES"
    print("layerwise skip-list coverage: OK (RoutedExperts + OAIAttention + Gemma4Router)")
    # GLM-4's and LFM-2's router biases stay out of the push on this name alone; Ling's gate spells it expert_bias,
    # which the reload counts, so the family forwards it (MoEFamilySpec.engine_loaded_buffers).
    bias = "e_score_correction_bias"
    assert bias in SKIP_LOAD_TENSORS and bias in SKIP_TENSORS, f"the reload no longer skips {bias}"
    print(f"layerwise tensor skip list: OK ({bias})")


def check_weight_transfer_reinit_patch() -> None:
    """Re-initialization closes the previous NCCL communicator: unpatched, every trainer that reconnects
    without a clean shutdown strands a live one on each engine-core worker until ncclCommInitRank fails.
    The patch leaves shutdown to upstream, which destroys it."""
    assert reinit_patch._APPLIED, "sitecustomize hook did not fire on nccl_engine import"
    assert getattr(NCCLWeightTransferEngine.init_transfer_engine, reinit_patch.PATCH_MARKER, False), (
        "init_transfer_engine not patched"
    )
    assert "self.model_update_group.destroy()" in inspect.getsource(NCCLWeightTransferEngine.shutdown), (
        "shutdown no longer destroys the communicator"
    )
    print("weight-transfer re-init patch: OK (communicator destroyed on re-init; upstream shutdown destroys it)")


def check_weight_transfer_wire_contract() -> None:
    """The wire contract the client speaks: the packed geometry rides the init request (the engine
    otherwise receives tensor by tensor while the trainer broadcasts packed buffers), an update declares
    only names, dtypes and shapes, both sides cut chunks at the same byte budget, the consumer reuses its
    streams (per-call streams strand every sync's receive buffers in the caching allocator), and the
    routes exist. Any field change fails here, not as a hung group."""
    init = {f.name for f in dataclasses.fields(NCCLWeightTransferInitInfo)}
    assert init == {
        "master_address",
        "master_port",
        "rank_offset",
        "world_size",
        "nccl_unique_id_b64",
        "packed",
        "packed_buffer_size_bytes",
        "packed_num_buffers",
    }, f"init request fields changed: {sorted(init)}"
    update = {f.name for f in dataclasses.fields(NCCLWeightTransferUpdateInfo)}
    assert update == {"names", "dtype_names", "shapes"}, f"update request fields changed: {sorted(update)}"
    assert (packed_tensor.DEFAULT_PACKED_BUFFER_SIZE_BYTES, packed_tensor.DEFAULT_PACKED_NUM_BUFFERS) == (
        1 << 30,
        2,
    ), "packed defaults moved"
    consumer = inspect.getsource(packed_tensor.packed_nccl_broadcast_consumer)
    assert "_get_streams(" in consumer and hasattr(packed_tensor._get_streams, "cache_info"), (
        "packed consumer no longer reuses its streams"
    )
    assert "> target_packed_tensor_size" in consumer, "packed consumer changed its chunk boundary"
    paths = {getattr(r, "path", None) for r in router.routes}
    required = {
        "/pause",
        "/resume",
        "/is_paused",
        "/init_weight_transfer_engine",
        "/start_weight_update",
        "/update_weights",
        "/finish_weight_update",
        "/get_world_size",
    }
    assert required <= paths, f"weight-sync routes lost: {sorted(required - paths)}"
    print("weight-transfer wire contract: OK")


def check_thinking_budget_patches() -> None:
    """Unpatched, a gpt-oss cut stops at the channel token and the model picks the channel itself, where
    an analysis it reopens counts afresh."""
    assert forced_end_patch._APPLIED and getattr(
        ReasoningConfig.initialize_token_ids, forced_end_patch.PATCH_MARKER, False
    ), "forced-end patch not applied"
    assert completion_patch._APPLIED and all(
        getattr(getattr(ThinkingBudgetState, n), completion_patch.PATCH_MARKER, False)
        for n in ("__init__", "apply_staged_writes", "apply")
    ), "completion patch not applied"
    print("thinking-budget patches: OK")


def check_serve_flags() -> None:
    """The engine arguments this deployment requires, as dataclass fields since ``vllm serve --help``
    needs a GPU to infer the platform: a rename fails the build, not the run."""
    names = {f.name for f in dataclasses.fields(EngineArgs)}
    missing = {"weight_transfer_config", "enable_return_routed_experts"} - names
    assert not missing, f"EngineArgs lost: {missing}"
    knames = {f.name for f in dataclasses.fields(KernelConfig)}
    assert "moe_backend" in knames, "KernelConfig.moe_backend gone"
    print("serve flags: OK")


def check_pause_keep_clears_cache() -> None:
    """The client pauses with ``mode=keep`` (in-flight requests survive the sync) and leaves
    ``clear_cache`` at its default, so the pause resets the prefix cache and recomputes every in-flight
    request under the new weights: no KV computed with the old weights is read after a sync. Pinned by
    behaviour, since a keep pause that ignored ``clear_cache`` would serve that stale KV with no error."""
    params = inspect.signature(pause_generation).parameters

    def inner(annotation):
        return typing.get_args(annotation)[0] if typing.get_origin(annotation) is typing.Annotated else annotation

    assert "mode" in params and "keep" in typing.get_args(inner(params["mode"].annotation)), (
        "/pause mode no longer accepts keep"
    )
    assert (
        "clear_cache" in params
        and inner(params["clear_cache"].annotation) is bool
        and params["clear_cache"].default is True
    ), "/pause clear_cache changed"
    llm = MagicMock()
    llm.renderer.clear_mm_cache_async = AsyncMock()
    llm.engine_core.pause_scheduler_async = AsyncMock()

    def forwarded(clear: bool) -> bool:
        asyncio.run(AsyncLLM.pause_generation(llm, mode="keep", clear_cache=clear))
        return llm.engine_core.pause_scheduler_async.await_args.kwargs["clear_cache"]

    assert forwarded(True) is True and forwarded(False) is False, (
        "pause_generation no longer forwards clear_cache under mode=keep"
    )
    engine = MagicMock()
    EngineCoreProc.pause_scheduler(engine, mode="keep", clear_cache=True)
    assert engine._finish_pause.call_args.args == (True,), (
        "a keep pause no longer finishes with the clear_cache it was given"
    )
    engine = MagicMock()
    EngineCore._finish_pause(engine, False)
    assert not engine._reset_caches.called, "clear_cache=false still resets"
    engine = MagicMock()
    EngineCore._finish_pause(engine, True)
    assert engine._reset_caches.called, "clear_cache=true no longer resets"
    engine = MagicMock()
    EngineCore._reset_caches(engine)
    assert engine.reset_prefix_cache.call_args.kwargs.get("reset_running_requests") is True, (
        "pause reset no longer preempts running requests"
    )
    print("/pause mode=keep + clear_cache: OK")


CHECKS = (
    check_engine_stack,
    check_unique_id_pickle_shim,
    check_layerwise_reload_patch,
    check_layerwise_skip_coverage,
    check_weight_transfer_reinit_patch,
    check_weight_transfer_wire_contract,
    check_thinking_budget_patches,
    check_serve_flags,
    check_pause_keep_clears_cache,
)


if __name__ == "__main__":
    for check in CHECKS:
        check()
