"""Megatron Bridge 0.6.2 full-parameter SFT benchmark for google/gemma-4-26B-A4B-it on 2 GPUs.

Starts from the Bridge's own Gemma 4 26B-A4B recipe (gemma4_vl_26b_sft_8gpu_h100_bf16_config — the only
26B-A4B recipe it ships; Bridge 0.6.2 maps the MoE checkpoint only to Gemma4VLModel) and overrides it for the
benchmark protocol: text-only chat data, seq 2048, MBS 1, GBS 2, 25 iters, lr 1e-5 constant, clip 1.0,
no eval / no checkpoint saving. Parallel layout / kernels are selected with env vars (see LAUNCH vars below).

Launch: torchrun --nproc_per_node=2 train_gemma4.py
"""

import json
import os
import time

import torch
from megatron.bridge.data.sources.hf import HFDatasetSourceConfig
from megatron.bridge.recipes.gemma4_vl.h100.gemma4_vl import gemma4_vl_26b_sft_8gpu_h100_bf16_config
from megatron.bridge.training.callbacks import Callback
from megatron.bridge.training.finetune import finetune
from megatron.bridge.training.vlm_step import forward_step

BENCH_ROOT = os.environ["BENCH_ROOT"]  # see paths.env


def env(name, default):
    return os.environ.get(name, default)


# ---- launch knobs ---------------------------------------------------------------------------
TP = int(env("MB_TP", "1"))
EP = int(env("MB_EP", "2"))
ETP = int(env("MB_ETP", "1"))
RECOMPUTE = env("MB_RECOMPUTE", "none")  # none | full | selective:<mod1,mod2>
DISPATCHER = env("MB_DISPATCHER", "alltoall")  # alltoall | allgather | flex
ATTN = env("MB_ATTN", "auto")  # auto | fused | unfused | flash | local
GRAD_FP32 = env("MB_GRAD_REDUCE_FP32", "1") == "1"
OVERLAP = env("MB_OVERLAP", "0") == "1"
PRECISION_AWARE = env("MB_PRECISION_AWARE_OPT", "0") == "1"  # bf16 Adam moments via TE FusedAdam
MEGATRON_FSDP = env("MB_MEGATRON_FSDP", "0") == "1"
DATA = env("MB_DATA", f"{BENCH_ROOT}/data/ultrachat_gemma4_2k.jsonl")
CKPT = env("MB_CKPT", f"{BENCH_ROOT}/megatron/ckpt/gemma4-26b-a4b-it")
OUT_JSON = env("MB_OUT_JSON", f"{BENCH_ROOT}/megatron/runs/last_run.json")
ITERS = int(env("MB_ITERS", env("BENCH_STEPS", "25")))
SEQ = int(env("BENCH_SEQ", "2048"))


class BenchCallback(Callback):
    """Per-step wall time (CUDA-synced), loss, grad norm, peak memory."""

    def __init__(self):
        self.step_times, self.losses, self.grad_norms = [], [], []
        self._t0 = None

    def on_train_step_start(self, context):
        torch.cuda.synchronize()
        self._t0 = time.perf_counter()

    def on_train_step_end(self, context):
        torch.cuda.synchronize()
        self.step_times.append(time.perf_counter() - self._t0)
        loss = None
        if context.loss_dict:
            v = next(iter(context.loss_dict.values()))
            loss = float(v.item() if torch.is_tensor(v) else v)
        self.losses.append(loss)
        self.grad_norms.append(None if context.grad_norm is None else float(context.grad_norm))
        if torch.distributed.get_rank() == 0:
            print(
                f"[bench] step {len(self.step_times)} time {self.step_times[-1]:.3f}s loss {loss} "
                f"gnorm {self.grad_norms[-1]}",
                flush=True,
            )

    def on_train_end(self, context):
        alloc = torch.tensor([torch.cuda.max_memory_allocated() / 2**30], device="cuda")
        resv = torch.tensor([torch.cuda.max_memory_reserved() / 2**30], device="cuda")
        torch.distributed.all_reduce(alloc, op=torch.distributed.ReduceOp.MAX)
        torch.distributed.all_reduce(resv, op=torch.distributed.ReduceOp.MAX)
        if torch.distributed.get_rank() != 0:
            return
        world = torch.distributed.get_world_size()
        tokens_per_step = SEQ * world  # MBS 1 per DP rank; GBS = 2
        measured = self.step_times[5:]
        result = {
            "step_times_s": self.step_times,
            "losses": self.losses,
            "grad_norms": self.grad_norms,
            "tokens_per_step": tokens_per_step,
            "cluster_tokens_per_second": tokens_per_step * len(measured) / sum(measured) if measured else None,
            "peak_mem_allocated_gib": alloc.item(),
            "peak_mem_reserved_gib": resv.item(),
            "knobs": {k: v for k, v in os.environ.items() if k.startswith("MB_")},
        }
        os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
        with open(OUT_JSON, "w") as f:
            json.dump(result, f, indent=2)
        print("[bench] RESULT", json.dumps(result), flush=True)


def build_config():
    cfg = gemma4_vl_26b_sft_8gpu_h100_bf16_config()
    m = cfg.model

    # Parallel layout (2 GPUs)
    m.tensor_model_parallel_size = TP
    m.expert_model_parallel_size = EP
    m.expert_tensor_parallel_size = ETP
    m.pipeline_model_parallel_size = 1
    m.context_parallel_size = 1
    m.sequence_parallel = TP > 1
    m.seq_length = SEQ

    # MoE kernels: TE GroupedLinear (moe_grouped_gemm) + fused permute; dispatcher selectable
    # MB_GROUPED_GEMM=0 -> MCore SequentialMLP (per-expert TE linears in a Python loop); the old grouped_gemm-package
    # MCore 0.19 has no GroupedMLP
    m.moe_grouped_gemm = env("MB_GROUPED_GEMM", "1") == "1"
    m.moe_permute_fusion = True
    m.moe_token_dispatcher_type = DISPATCHER
    if DISPATCHER == "flex":
        m.moe_flex_dispatcher_backend = env("MB_FLEX_BACKEND", "deepep")

    # Attention: Gemma 4 global layers have head_dim 512 (FlashAttention rejects it)
    m.attention_backend = ATTN
    # gradient_accumulation_fusion needs APEX's fused_weight_gradient_mlp_cuda for the non-TE output layer
    m.gradient_accumulation_fusion = env("MB_GRAD_ACC_FUSION", "1") == "1"
    # Optional: TE GEGLU op for the expert/shared-expert GLU. MCore maps (F.gelu, gated) -> te.ops.GEGLU, and
    # TE's GEGLU is the tanh approximation, i.e. numerically the same function as Gemma's gelu_pytorch_tanh.
    if env("MB_TE_ACT", "0") == "1":
        m.activation_func = torch.nn.functional.gelu
        m.use_te_activation_func = True
    if "MB_BIAS_ACT_FUSION" in os.environ:
        m.bias_activation_fusion = os.environ["MB_BIAS_ACT_FUSION"] == "1"

    # Activation recompute
    if RECOMPUTE == "none":
        m.recompute_granularity = None
        m.recompute_method = None
        m.recompute_num_layers = None
        m.recompute_modules = None
    elif RECOMPUTE == "full":
        m.recompute_granularity = "full"
        m.recompute_method = "uniform"
        m.recompute_num_layers = 1
        m.recompute_modules = None
    elif RECOMPUTE.startswith("selective:"):
        m.recompute_granularity = "selective"
        m.recompute_method = None
        m.recompute_num_layers = None
        m.recompute_modules = RECOMPUTE.split(":", 1)[1].split(",")

    # Text-only SFT: vision tower + projection are not exercised; keep them frozen (recipe freezes the tower)
    m.freeze_language_model = False
    m.freeze_vision_model = True
    m.freeze_vision_projection = True

    # Data: local chat jsonl ({"messages": [...]}), Gemma 4 chat template via the HF processor
    d = cfg.dataset
    d.seq_length = SEQ
    d.source = HFDatasetSourceConfig(path_or_dataset="json", split="train", load_kwargs={"data_files": DATA})
    d.do_validation = False
    d.do_test = False
    d.enable_in_batch_packing = False
    d.pad_to_max_length = True
    d.num_workers = 2

    # Training schedule
    t = cfg.train
    t.train_iters = ITERS
    t.global_batch_size = 2
    t.micro_batch_size = 1
    cfg.validation.eval_interval = None
    cfg.validation.eval_iters = 0

    # Optimizer: Megatron distributed (fused TE) Adam, lr 1e-5 constant, clip 1.0
    o = cfg.optimizer
    o.lr = 1e-5
    o.min_lr = 1e-5
    o.clip_grad = 1.0
    o.use_distributed_optimizer = True
    if PRECISION_AWARE:
        o.use_precision_aware_optimizer = True
        o.main_params_dtype = torch.float32
        o.main_grads_dtype = torch.float32
        o.exp_avg_dtype = torch.bfloat16
        o.exp_avg_sq_dtype = torch.bfloat16
    s = cfg.scheduler
    s.lr_decay_style = "constant"
    s.lr_warmup_iters = 0
    s.lr_decay_iters = ITERS

    # DDP
    cfg.ddp.use_distributed_optimizer = True
    cfg.ddp.grad_reduce_in_fp32 = GRAD_FP32
    cfg.ddp.overlap_grad_reduce = OVERLAP
    cfg.ddp.overlap_param_gather = OVERLAP
    if MEGATRON_FSDP:
        cfg.ddp.use_megatron_fsdp = True
        cfg.ddp.data_parallel_sharding_strategy = "optim_grads_params"
        cfg.checkpoint.ckpt_format = "fsdp_dtensor"

    # Checkpoint: load converted weights only, never save
    cfg.checkpoint.pretrained_checkpoint = CKPT
    cfg.checkpoint.load = None
    cfg.checkpoint.save = None
    cfg.checkpoint.save_interval = None

    cfg.logger.log_interval = 1
    cfg.logger.tensorboard_dir = None
    cfg.logger.wandb_project = None
    return cfg


if __name__ == "__main__":
    finetune(build_config(), forward_step, callbacks=[BenchCallback()])
