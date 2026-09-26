#!/bin/bash
# the protocol (PROTOCOL.txt) Axolotl 0.19.0 runs inside Docker, one command per config.
# GPUs via CDI ($GPU_FLAGS); GPU lock held only for the docker run. Each run writes
#   ${BENCH_ROOT}/axolotl/out/<name>.json  (step times, losses, losses_token_weighted, grad_norms, row ids, data mismatches, mem)
#   ${BENCH_ROOT}/axolotl/logs/<name>.log
# Prereqs: stage.sh (builds $BENCH_ROOT/axolotl/data_axolotl.jsonl, sha256 9835ddc6d50db117c76892164961d8a821738fce3a1316629774264cf147ee99,
#   the prepared/ cache: 512 rows, in order, 0 mismatches vs canonical; and the two derived images). The plugins
#   (benchplugin/) and configs are read from scripts/benchmarks; configs are rendered with render_config (paths.env).
# HF_TOKEN is passed through from the caller's environment (never written here). The model is read from $HF_CACHE.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../../paths.env"
set -u
ROOT=${BENCH_ROOT}/axolotl
OFFICIAL=axolotlai/axolotl:0.19.0-py3.12-cu130-2.13.0@sha256:b38049f2454ceafc58c7a8269c7b400a75416e15b0f308a33850d48ae11e384d
OFFICIAL_T211=axolotlai/axolotl:0.19.0-py3.12-cu130-2.11.0@sha256:35c1379c265709acef0eb198b181c0713e4d4cc25b180f5cda5670b8df11cce0
# Derived images (Dockerfiles in docker/, build context = docker/; stage.sh builds them):
#   docker build -f Dockerfile.deepep        -t axolotl-bench:0.19.0-deepep .          # + DeepEP 4214430 (for expert_parallel_size: 2)
#   docker build -f Dockerfile.sonic_triton36 -t axolotl-bench:0.19.0-sonic-triton360 . # Triton 3.7.1 -> 3.6.0 for SonicMoE on sm_103
DEEPEP=axolotl-bench:0.19.0-deepep            # measured with image id sha256:80f6348914571db0e3f2e1b9243032d9eee9a4c3d27435ae9f954dc4b2d7bd3f
SONIC=axolotl-bench:0.19.0-sonic-triton360    # measured with image id sha256:17707eced910e94fba438fdfcb8a0bef8d6623edb336da1bc6c9c4ae5368dcc6

run() {  # run <name> <image> <config>
  local NAME=$1 IMG=$2 CFG
  CFG=$(render_config "$BUNDLE/gemma4_sft/axolotl/$3")
  mkdir -p $ROOT/logs $ROOT/out
  flock ${GPU_LOCK} bash -c "echo LOAD_BEFORE \$(cat /proc/loadavg) > $ROOT/logs/$NAME.load; \
   timeout 1800 docker run --rm --name axolotl-$NAME \
    $GPU_FLAGS --ipc=host --ulimit memlock=-1 --shm-size=128g \
    "${BENCH_DOCKER_MOUNTS[@]}" -e HF_TOKEN \
    -e PYTHONPATH=$BUNDLE/gemma4_sft/axolotl -e TRITON_CACHE_DIR=$ROOT/triton_cache_docker_$(echo $IMG | tr ':/@' '___' | cut -c1-60) \
    -e WANDB_DISABLED=true -e WANDB_MODE=disabled -e AXOLOTL_DO_NOT_TRACK=1 \
    -e BENCH_EXIT_AFTER=1 -e BENCH_OUT=$ROOT/out/$NAME.json \
    -w $ROOT --entrypoint bash $IMG \
    -lc 'torchrun --nproc_per_node 2 --master_port 29611 -m axolotl.cli.train $CFG' > $ROOT/logs/$NAME.log 2>&1; \
   rc=\$?; echo LOAD_AFTER \$(cat /proc/loadavg) >> $ROOT/logs/$NAME.load; exit \$rc"
  echo "exit $?" >> $ROOT/logs/$NAME.log
  echo "$NAME: $(grep -E '^exit' $ROOT/logs/$NAME.log) $(grep -o '\[bench\] RESULT.*' $ROOT/logs/$NAME.log | cut -c1-250)"
}
# Throughput is valid only when the host is idle: runs record /proc/loadavg before/after in logs/<name>.load; a 1-min
# load above ~20 marks that run's throughput "invalid: host load" (losses / data / order checks stay valid).
# NOTE: do not export DO_NOT_TRACK=1 (breaks kernels-hub downloads of the FA2 hub kernel: malformed user-agent).

# 1. Best FSDP2 config (grouped_mm, no GC, hybrid FA2/sdpa, no-reshard) and its ScatterMoE twin - 2 repeats each
run dk_L_rep1 $OFFICIAL configs/L_fsdp2_groupedmm_nogc_hybridfa2_noreshard.yaml
run dk_L_rep2 $OFFICIAL configs/L_fsdp2_groupedmm_nogc_hybridfa2_noreshard.yaml
run dk_N_rep1 $OFFICIAL configs/N_fsdp2_scattermoe_nogc_hybridfa2_noreshard.yaml
run dk_N_rep2 $OFFICIAL configs/N_fsdp2_scattermoe_nogc_hybridfa2_noreshard.yaml
# 2. Expert parallel EP2 (pure EP: expert_parallel_size == world_size == 2, DDP for non-expert params, DeepEP dispatch/combine).
#    Needs the DeepEP-derived image AND the EPDistInitPlugin workaround (benchplugin/ep_dist_init.py): without it
#    axolotl 0.19.0 silently skips expert sharding in this path (torch.distributed not yet initialized at post_model_build).
#    Check the log for "Sharded 30 Experts module(s) ... ep_size=2" and "[bench] post_model_load ... ((64, 1408, 2816), 64)".
run dk_P_ep2_groupedmm $DEEPEP configs/P_ep2_groupedmm_nogc_hybridfa2.yaml
run dk_Q_ep2_scattermoe $DEEPEP configs/Q_ep2_scattermoe_nogc_hybridfa2.yaml
# 3. SonicMoE (best layout otherwise): official image as-is (expected: "B300 (sm_103) requires Triton 3.6.x, but found 3.7.1"),
#    Triton-3.6.0 derived image, and the official torch-2.11 tag (ships Triton 3.6.0).
run dk_S_sonic_official $OFFICIAL configs/S_fsdp2_sonicmoe_nogc_hybridfa2_noreshard.yaml
run dk_S_sonic_triton360 $SONIC configs/S_fsdp2_sonicmoe_nogc_hybridfa2_noreshard.yaml
run dk_S_sonic_official_t211 $OFFICIAL_T211 configs/S_fsdp2_sonicmoe_nogc_hybridfa2_noreshard.yaml
