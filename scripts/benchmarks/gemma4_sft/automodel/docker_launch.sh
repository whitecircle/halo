#!/bin/bash
# the protocol launch for NeMo AutoModel in the official container (GPUs via CDI).
# Usage: docker_launch.sh <run_name> [0|1 = independent token-weighted CE check] [extra --dotted.overrides]
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../../paths.env"
set -u
NAME=$1; CHECK=${2:-0}; shift; [ $# -gt 0 ] && shift
B=$BUNDLE/gemma4_sft/automodel          # run_bench.py, bench_data.py and gemma4_26b_a4b_ep2_bench.yaml, read from scripts/benchmarks
R=${BENCH_ROOT}/automodel/runs; mkdir -p $R; rm -f $R/$NAME.order.rank*
IMG=nvcr.io/nvidia/nemo-automodel:26.08.00  # sha256:b0bd0e50b29cb0a3ec69c3cbbb49652d3c1aabedbfe22ee0fa6d053b6bb33aee
flock ${GPU_LOCK} bash -c 'echo "$(date -Is) $(cat /proc/loadavg) | $(uptime)" > '$R/$NAME.load_before'; exec timeout 1800 "$@"' _ docker run --rm --name automodel_$NAME \
  $GPU_FLAGS --ipc=host --ulimit memlock=-1 --shm-size=128g \
  "${BENCH_DOCKER_MOUNTS[@]}" -e HF_HOME=${HF_CACHE} -e HF_HUB_OFFLINE=1 -e TMPDIR=${BENCH_TMP} \
  -e CUDA_DEVICE_MAX_CONNECTIONS=1 -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e TORCHINDUCTOR_CACHE_DIR=${BENCH_TMP}/inductor_automodel_container -e WANDB_MODE=disabled \
  -e PYTHONPATH=$B -e BENCH_OUT=$R/$NAME.json \
  -e BENCH_ORDER_LOG=$R/$NAME.order -e BENCH_TW_CHECK=$CHECK \
  -w $B $IMG \
  torchrun --nproc_per_node=2 --master_port=29579 run_bench.py --config "$(render_config $B/gemma4_26b_a4b_ep2_bench.yaml)" "$@" \
  > $R/$NAME.log 2>&1
rc=$?
echo "$(date -Is) $(cat /proc/loadavg) | $(uptime)" > $R/$NAME.load_after
echo "exit $rc" >> $R/$NAME.log
