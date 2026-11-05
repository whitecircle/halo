#!/usr/bin/env python
"""A push of the unchanged checkpoint leaves a live server's log-probs where they were, and a push after a real
optimizer step leaves it computing what the trainer computes, family by family on both engines; the rounds are in
:mod:`tests.common.weight_push_exact`. Each pass needs a server on the row's checkpoint that
has taken no other suite's push (a row ends on a restoring push, so the rows of one family share a server), on a GPU the
two ranks do not own.

vLLM (all GPUs visible, masked by CUDA_VISIBLE_DEVICES; ``--trust-remote-code`` for Ling-mini-2.0 and Step-3.7, plus
``--max-model-len 1024`` for Step-3.7's tiny checkpoint):
    docker run -d --gpus all -e CUDA_VISIBLE_DEVICES=7 -e VLLM_SERVER_DEV_MODE=1 -e VLLM_ENFORCE_STRICT_TOOL_CALLING=0 \\
        --network host --ipc host \\
        -v $HF_HOME:/root/.cache/huggingface:ro -e HF_HUB_OFFLINE=1 -e HF_MODULES_CACHE=/root/hf_modules \\
        -e NCCL_IB_DISABLE=1 -e NCCL_NET=Socket -e NCCL_P2P_LEVEL=NVL \\
        halo:vllm-0.32.0 $MODEL --port 8000 --dtype auto --weight-transfer-config '{"backend": "nccl"}' \\
        --moe-backend triton --return-tokens-as-token-ids --logprobs-mode processed_logprobs --enable-prefix-caching

SGLang (``SGLANG_TRUST_REMOTE_CODE=1`` for Ling-mini-2.0; ``SGLANG_ATTENTION_BACKEND=triton
SGLANG_EXTRA_ARGS=--enable-deterministic-inference`` for GLM-4):
    SGLANG_MODEL=$MODEL SGLANG_MODEL_DIR=$ANY_DIR SGLANG_CUDA_DEVICES=7 docker compose -f docker-compose.sglang.yml up -d

Then, for one family (``-k`` narrows ``pytest -m 'gpu and vllm_server' tests/gpu`` the same way):
    HALO_TEST_PUSH_EXACT_FRESH=1 VLLM_SERVER_URL=http://localhost:8000 VLLM_GROUP_HOST=127.0.0.1 NCCL_IB_DISABLE=1 \\
        NCCL_NET=Socket CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 \\
        tests/gpu/trainers/grpo/test_weight_push_exact.py --family bailing_moe --backend vllm --ep-size 2

``--reach`` adds one push per forwarded name, on Laguna and Gemma 4 especially, whose real-step bar is gross;
``--reach-leave-out model.layers.N.mlp.gate.weight`` is its negative control and must fail.

GLM-5 Next serves a tiny random checkpoint written first (the Hub supplies the tokenizer and processor files only):
    python tests/gpu/trainers/grpo/test_weight_push_exact.py --family glm5_next --write-checkpoint $DIR
then vLLM with ``--max-model-len 1024 --enforce-eager --gpu-memory-utilization 0.2``, or SGLang with ``SGLANG_MODEL=$DIR``,
and ``HALO_TEST_GLM5_NEXT_MODEL=$DIR`` on the trainer. The name-for-name ``--reach`` rows leave GLM-5 Next out: scaling a
DSA indexer weight cannot move the log-probs (the top-k ranking is scale-invariant).

A server launched from a local path takes ``--served-model-name`` (or SGLang's ``SGLANG_MODEL``) equal to
``HALO_TEST_PUSH_EXACT_MODEL``, the same path for the trainer.
"""

from tests.common.harness import gpu_test_main
from tests.common.weight_push_exact import (
    WORLD_SIZE,
    parse_row,
    run_push_exactness,
    skip_unless_a_fresh_server_holds_the_row,
    write_tiny_checkpoint_if_asked,
)


def run(ctx) -> dict:
    return run_push_exactness(ctx, parse_row())


main = gpu_test_main(exact_world_size=WORLD_SIZE, prefix="weight_push_exact")(run)

if __name__ == "__main__":
    row = parse_row()
    write_tiny_checkpoint_if_asked(row)
    skip_unless_a_fresh_server_holds_the_row(row)
    main()
