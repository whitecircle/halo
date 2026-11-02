// Stub specializations for FA2 split-K kernels on Hopper (sm_90).
//
// CUDA 13.2's ptxas hangs indefinitely (>>1h, no output, RSS growing) when
// compiling the original csrc/flash_attn/src/flash_fwd_split_hdim*_*_sm80.cu
// kernel templates targeted to sm_90. The hang reproduces 100% on every fresh
// build with FLASH_ATTN_CUDA_ARCHS=90; the same files compile fine for sm_100
// (Blackwell), so it appears to be a CUDA 13.2 SM90-specific codegen
// regression on these particular CUTLASS kernel templates.
//
// Workaround for the Hopper image: drop the split .cu files from FA2's setup.py
// source list (the `flash_fwd_split_` prefix, both the hdim and the newer
// align_hdim sets) and replace them with this single .cpp providing full template
// specializations of run_mha_fwd_splitkv_dispatch<T, HD, Causal> that throw at
// runtime. The .so links cleanly (no undefined symbols); the varlen forward and
// the whole backward are unaffected.
//
// SCOPE OF THE THROW. flash_api.cpp's mha_fwd, and mha_varlen_fwd for a one-token
// query under GQA (seqlenq_ngroups_swapped), pass num_splits=0 to set_params_splitkv(),
// whose occupancy heuristic picks split-K whenever dropout is 0 and
// batch*heads*ceil(seq_q/64) < 0.8*2*SMs: small unpadded batches and decode, training
// included. The Dockerfile rewrites both calls to num_splits=1 and asserts it, so
// neither forward reaches these stubs. Only flash_attn_with_kvcache, whose caller
// picks num_splits, still can; nothing in Halo calls it.
//
// If/when CUDA 13.3+ fixes the ptxas regression, drop this file and restore
// the original split .cu kernels.

// cuda_runtime.h before flash.h: flash.h uses cudaStream_t in declarations
// without including the CUDA runtime headers itself. The original split .cu
// files get cudaStream_t via flash_fwd_launch_template.h's chain; we pull it
// in directly so our .cpp parses flash.h's primary template declaration
// correctly (otherwise our specializations won't match it).
#include <cuda_runtime.h>
#include "namespace_config.h"
#include "flash.h"
#include <cutlass/numeric_types.h>
#include <c10/util/Exception.h>

namespace FLASH_NAMESPACE {

#define STUB_SPLIT(T, HD, CAUSAL)                                                     \
    template<>                                                                        \
    void run_mha_fwd_splitkv_dispatch<T, HD, CAUSAL>(Flash_fwd_params&, cudaStream_t) \
    {                                                                                 \
        TORCH_CHECK(false,                                                            \
            "FlashAttention-2 split-K kernels are disabled in this Hopper image. "    \
            "(CUDA 13.2 ptxas hangs on these templates for sm_90.) "                  \
            "Use flash_attention_3 for inference paths that need split-K.");          \
    }

STUB_SPLIT(cutlass::half_t,     32,  false)
STUB_SPLIT(cutlass::half_t,     32,  true)
STUB_SPLIT(cutlass::half_t,     64,  false)
STUB_SPLIT(cutlass::half_t,     64,  true)
STUB_SPLIT(cutlass::half_t,     96,  false)
STUB_SPLIT(cutlass::half_t,     96,  true)
STUB_SPLIT(cutlass::half_t,    128,  false)
STUB_SPLIT(cutlass::half_t,    128,  true)
STUB_SPLIT(cutlass::half_t,    192,  false)
STUB_SPLIT(cutlass::half_t,    192,  true)
STUB_SPLIT(cutlass::half_t,    256,  false)
STUB_SPLIT(cutlass::half_t,    256,  true)
STUB_SPLIT(cutlass::bfloat16_t, 32,  false)
STUB_SPLIT(cutlass::bfloat16_t, 32,  true)
STUB_SPLIT(cutlass::bfloat16_t, 64,  false)
STUB_SPLIT(cutlass::bfloat16_t, 64,  true)
STUB_SPLIT(cutlass::bfloat16_t, 96,  false)
STUB_SPLIT(cutlass::bfloat16_t, 96,  true)
STUB_SPLIT(cutlass::bfloat16_t, 128, false)
STUB_SPLIT(cutlass::bfloat16_t, 128, true)
STUB_SPLIT(cutlass::bfloat16_t, 192, false)
STUB_SPLIT(cutlass::bfloat16_t, 192, true)
STUB_SPLIT(cutlass::bfloat16_t, 256, false)
STUB_SPLIT(cutlass::bfloat16_t, 256, true)

#undef STUB_SPLIT

}  // namespace FLASH_NAMESPACE
