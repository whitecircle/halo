"""Build-time checks of the vLLM server image: the stack pins, the runtime patches and the contracts the
weight-sync client relies on (the Triton-interpreter checks live beside their patches and plugins).

No test suite can run these: they import vLLM, which the training image does not have (ABI-incompatible
stacks), so the image build is the only gate. Each check fails the build on an upstream change that would
otherwise surface as a hung group, a silently reverted sync or a wrong serve-time result. ``Dockerfile.vllm`` passes
the tag it builds and the engine build that tag pins:

    python3 verify_image.py release 0.32.0
    python3 verify_image.py inkling2 0.1.dev25249+gb974b306f
"""

import asyncio
import dataclasses
import importlib
import importlib.util
import inspect
import sys
import typing
from importlib.metadata import requires, version
from unittest.mock import AsyncMock, MagicMock

import vllm
import vllm.envs as envs
import vllm_image_identity_patch as identity_patch
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
from vllm.entrypoints.serve.dev.server_info.api_router import _get_vllm_env_vars
from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
from vllm.model_executor.model_loader.reload import layerwise
from vllm.model_executor.model_loader.reload.meta import SKIP_LOAD_TENSORS, SKIP_TENSORS
from vllm.model_executor.models.gemma4 import Gemma4Router
from vllm.model_executor.models.gpt_oss import OAIAttention
from vllm.tool_parsers import ToolParser, ToolParserManager
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.engine.core import EngineCore, EngineCoreProc
from vllm.v1.worker.gpu.sample.thinking_budget import ThinkingBudgetState

from halo.weight_sync.transport.nccl_library import ncclUniqueId as shim_unique_id

# Dockerfile.vllm's VLLM_ENGINE values: the release tag every family is served from, and the Inkling 2 tag.
RELEASE, INKLING2 = "release", "inkling2"
ENGINE = RELEASE
# Request fields the Inkling 2 build declares beyond the release's; the client sends none of them.
INKLING2_INIT_FIELDS = {"use_load_plan", "metadata"}
INKLING2_UPDATE_FIELDS = {"src"}
RUNNER_SWITCH = "VLLM_USE_V2_MODEL_RUNNER"
V1_RUNNER_MODULE = "vllm.v1.worker.gpu_model_runner"
V2_RUNNER_MODULE = "vllm.v1.worker.gpu.model_runner"
SPLIT_GUARD_PATCH = "vllm_topp_split_guard_patch"


def check_engine_build(build: str) -> None:
    """The engine is the build this tag pins: the client refuses an identity row that names another one, and the
    Inkling 2 base is a development build with no release number to tell two of them apart."""
    assert vllm.__version__ == build, f"the base image runs vLLM {vllm.__version__}, and this tag pins {build}"
    print(f"vLLM: {vllm.__version__} ({ENGINE})")


def check_engine_stack() -> None:
    """transformers and tokenizers stay at the base image's own, inside the range vLLM declares."""
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
    init = {f.name: f.default for f in dataclasses.fields(NCCLWeightTransferInitInfo)}
    assert set(init) == {
        "master_address",
        "master_port",
        "rank_offset",
        "world_size",
        "nccl_unique_id_b64",
        "packed",
        "packed_buffer_size_bytes",
        "packed_num_buffers",
    } | (INKLING2_INIT_FIELDS if ENGINE == INKLING2 else set()), f"init request fields changed: {sorted(init)}"
    update = {f.name: f.default for f in dataclasses.fields(NCCLWeightTransferUpdateInfo)}
    assert set(update) == {"names", "dtype_names", "shapes"} | (
        INKLING2_UPDATE_FIELDS if ENGINE == INKLING2 else set()
    ), f"update request fields changed: {sorted(update)}"
    if ENGINE == INKLING2:
        # The client sends neither load-plan field, so the engine writes each received tensor through load_weights.
        assert (init["use_load_plan"], init["metadata"]) == (False, None), "the load plan is on by default"
        # The client sends no src: the trainer broadcasts as group rank 0.
        assert update["src"] == 0, "the update's broadcasting rank is not 0 by default"
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


def check_model_runner_v2() -> None:
    """Model Runner V2 is the runner the engine starts: it enforces thinking budgets (the completion patch targets its
    sampler), and it is the only runner R3 capture runs on. The release ships both runners behind a switch, which the
    image's ENV holds at V2; the Inkling 2 build ships V2 alone."""
    if RUNNER_SWITCH in envs.environment_variables:
        assert getattr(envs, RUNNER_SWITCH) is True, f"{RUNNER_SWITCH} selects the V1 runner in this image"
    else:
        assert importlib.util.find_spec(V1_RUNNER_MODULE) is None, (
            f"the engine ships the V1 runner ({V1_RUNNER_MODULE}) with no {RUNNER_SWITCH} to pin V2 by"
        )
    assert (RUNNER_SWITCH in envs.environment_variables) == (ENGINE == RELEASE), (
        f"{RUNNER_SWITCH} is declared on the Inkling 2 build or gone from the release"
    )
    runner = importlib.import_module(V2_RUNNER_MODULE)
    assert ThinkingBudgetState.__module__.startswith(f"{runner.__package__}."), (
        "the thinking-budget state the completion patch targets is not Model Runner V2's"
    )
    worker = importlib.import_module("vllm.v1.worker.gpu_worker")
    assert f"from {V2_RUNNER_MODULE} import GPUModelRunner" in inspect.getsource(worker), (
        "the GPU worker does not build Model Runner V2"
    )
    print("Model Runner V2: OK")


def check_image_identity() -> None:
    """``/server_info`` names the patches, on the route the trainer reads, or it refuses this image as upstream. The
    split top-p guard is the release tag's alone."""
    listed = _get_vllm_env_vars().get(identity_patch.IDENTITY_KEY, "").split(",")
    assert "vllm_layerwise_reload_patch" in listed, f"/server_info lists {listed!r} as the image's patches"
    assert (SPLIT_GUARD_PATCH in listed) == (ENGINE == RELEASE), f"/server_info lists {listed!r} on the {ENGINE} tag"
    print(f"image identity on /server_info: OK ({','.join(listed)})")


def check_tool_calling_grammar_off() -> None:
    """Under the image's env no parser takes a grammar at tool_choice auto, however a server is launched: the glm47
    parser, whose default strict level would, reads the same flag."""
    assert envs.VLLM_ENFORCE_STRICT_TOOL_CALLING is False, "the image no longer turns strict tool calling off"
    glm47 = ToolParserManager.get_tool_parser("glm47")
    assert glm47.default_tool_strict_level is not None, "glm47 lost the default strict level this flag guards"
    assert "VLLM_ENFORCE_STRICT_TOOL_CALLING" in inspect.getsource(ToolParser), (
        "the tool parser base no longer reads VLLM_ENFORCE_STRICT_TOOL_CALLING"
    )
    print("strict tool calling off: OK")


def _check_release_core_pause() -> None:
    engine = MagicMock()
    EngineCoreProc.pause_scheduler(engine, mode="keep", clear_cache=True)
    assert engine._finish_pause.call_args.args == (True,), "a keep pause finishes without the clear_cache it was given"
    engine = MagicMock()
    EngineCore._finish_pause(engine, False)
    assert not engine._reset_caches.called, "clear_cache=false resets"
    engine = MagicMock()
    EngineCore._finish_pause(engine, True)
    assert engine._reset_caches.called, "clear_cache=true does not reset"


def _check_inkling2_core_pause() -> None:
    engine = MagicMock()
    EngineCoreProc.pause_generation(engine, mode="keep", clear_cache=True)
    assert engine.pause_scheduler.call_args.args == ("keep", True), "the engine's pause drops mode or clear_cache"
    assert engine.reject_new_requests is False, "a keep pause rejects the requests it should queue"
    engine = MagicMock()
    engine._pause_complete.return_value = True
    EngineCoreProc.pause_scheduler(engine, mode="keep", clear_cache=True)
    assert engine._finish_pause.call_args.args[0] is True, "a keep pause finishes without the clear_cache it was given"
    engine = MagicMock()
    engine._checkpoint_state = None
    EngineCore._finish_pause(engine, False, True)
    assert not engine._reset_caches.called, "clear_cache=false resets"
    engine = MagicMock()
    engine._checkpoint_state = None
    EngineCore._finish_pause(engine, True, True)
    assert engine._reset_caches.called, "clear_cache=true does not reset"


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
    # The engine-core call a keep pause forwards to differs between the two builds.
    forward = "pause_generation_async" if ENGINE == INKLING2 else "pause_scheduler_async"
    llm = MagicMock()
    llm.renderer.clear_mm_cache_async = AsyncMock()
    setattr(llm.engine_core, forward, AsyncMock())

    def forwarded(clear: bool) -> bool:
        asyncio.run(AsyncLLM.pause_generation(llm, mode="keep", clear_cache=clear))
        return getattr(llm.engine_core, forward).await_args.kwargs["clear_cache"]

    assert forwarded(True) is True and forwarded(False) is False, (
        "pause_generation does not forward clear_cache under mode=keep"
    )
    (_check_inkling2_core_pause if ENGINE == INKLING2 else _check_release_core_pause)()
    engine = MagicMock()
    EngineCore._reset_caches(engine)
    assert engine.reset_prefix_cache.call_args.kwargs.get("reset_running_requests") is True, (
        "pause reset does not preempt running requests"
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
    check_model_runner_v2,
    check_image_identity,
    check_tool_calling_grammar_off,
    check_pause_keep_clears_cache,
)


if __name__ == "__main__":
    ENGINE, pinned_build = sys.argv[1:]
    assert ENGINE in (RELEASE, INKLING2), f"unknown engine {ENGINE!r}"
    check_engine_build(pinned_build)
    for check in CHECKS:
        check()
