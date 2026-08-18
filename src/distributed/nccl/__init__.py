"""NCCL weight sync + HTTP generation for the supported rollout engines (vendored from vLLM v0.18.0,
Apache-2.0). The training image installs neither vllm nor sglang — their transformers pins conflict
with the trainer's — so each client speaks its engine's HTTP control plane directly and broadcasts over
NCCL. ``registry`` resolves a backend key to its client."""
