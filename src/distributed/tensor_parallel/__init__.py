"""Tensor parallelism, two mechanisms: HF-native ``tp_plan="auto"`` for dense models and selective attention-only TP
for MoE models (EP owns the experts), plus the plain-tensor shards both leave on the model."""
