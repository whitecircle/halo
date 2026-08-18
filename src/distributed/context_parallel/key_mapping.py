"""CP → HF attention state-dict key mapping.

Torch-free leaf, so serializers (the EP gathered save, the PEFT adapter save) can apply the mapping
without importing the CP wrapper stack.
"""


def strip_cp_attention_prefix(key: str) -> str:
    """Map a CP-patched state-dict key back to the unwrapped HF layout.

    Ulysses patching holds the original module as ``original_attention``, so every projection is
    reported one level deeper than HF names it. Any serializer reading a state dict out of a
    CP-patched module tree (the CP and EP gathered saves, the pipeline stage shard, PEFT adapters)
    must apply this before writing, or ``from_pretrained`` leaves the attention weights randomly
    initialized.
    """
    return key.replace(".original_attention.", ".")
