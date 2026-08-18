"""The one spelling of the CP→HF attention key mapping.

A torch-free leaf so the serializers that need it — the EP gathered save, the PEFT adapter
save — can spell the mapping without importing the CP wrapper stack (and through it the
collators, the attention patcher, and every per-family CP wrapper module).
"""


def strip_cp_attention_prefix(key: str) -> str:
    """Map a CP-patched state-dict key back to the unwrapped HF layout.

    Ulysses patching replaces each attention module with a wrapper that holds the original as
    ``original_attention``, so every projection is reported one level deeper than HF names it.
    Every serializer that reads a state dict out of a CP-patched module tree — the CP and EP
    gathered saves, the pipeline stage shard, PEFT adapters — must apply this before writing, or
    ``from_pretrained`` finds no attention weights and leaves them randomly initialized.
    """
    return key.replace(".original_attention.", ".")
