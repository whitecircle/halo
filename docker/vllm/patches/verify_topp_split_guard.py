"""Build-time check of the split top-p guard (``vllm_topp_split_guard_patch``).

Unpatched, a p-only row of a batch of at most 64 whose budget forced a token comes back all -inf, so the
engine samples token 0 with a NaN log-prob; the patch's ``self_check`` also fails once upstream stops
wiping that row, and the patch goes then. The interpreter never reads kernel source, so the rebuilt kernel
is also loaded as the compiled JITFunction the engine builds; a failure there would only be logged at
server start. The kernel runs on the Triton interpreter, since the build has no GPU:

    CUDA_VISIBLE_DEVICES= TRITON_INTERPRET=1 python3 verify_topp_split_guard.py
"""

import vllm_topp_split_guard_patch as guard_patch
from triton.runtime.jit import JITFunction
from vllm.v1.sample.ops import topk_topp_triton

if __name__ == "__main__":
    kernel = topk_topp_triton._topp_split_mask.kernel
    assert guard_patch._APPLIED and getattr(kernel, guard_patch.PATCH_MARKER, False), "top-p split guard not applied"
    assert guard_patch._GUARD in JITFunction(kernel.fn).src, "guarded kernel does not load as a compiled Triton kernel"
    guard_patch.self_check()
    print(
        "top-p split guard: OK (forced row kept, ordinary row masked as upstream, upstream still wipes the forced row)"
    )
