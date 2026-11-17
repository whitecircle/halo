"""ms-swift runner for the protocol (PROTOCOL.txt): `swift sft` in-process (under torchrun) fed the canonical tokens/labels.

The dataset is messages.jsonl (so ms-swift's normal dataset pipeline, collator and trainer run unchanged), but
Template.encode is wrapped so each row's input_ids/labels are replaced by canonical_tokens.jsonl (looked up by the
row's message contents). ms-swift's stock gemma4 template tokenizes differently (it keeps the thought channel in
history turns, which HF's template strips), so the canonical tokens are injected instead of re-tokenizing.
"""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench_callback  # noqa: E402

bench_callback.install()

import torch  # noqa: E402

BENCH_ROOT = os.environ["BENCH_ROOT"]  # see paths.env
THOUGHT = "<|channel>thought\n<channel|>"
MSG = os.environ.get("BENCH_MESSAGES", f"{BENCH_ROOT}/data/messages.jsonl")


def _key(messages):
    return json.dumps([(m["role"], m["content"].replace(THOUGHT, "").strip()) for m in messages])


if os.environ.get("BENCH_FSDP_NO_RAM_EFFICIENT") == "1":
    # ms-swift forces FSDP_CPU_RAM_EFFICIENT_LOADING=True for FSDP2; accelerate's rank-0 broadcast path crashes on
    # Gemma 4's persistent nn.Buffer `layer_scalar` ('Tensor' object has no attribute 'device_mesh').
    import swift.arguments.sft_args as _sft_args

    def _no_ram_efficient():
        os.environ.setdefault("ACCELERATE_USE_FSDP", "True")
        os.environ["FSDP_CPU_RAM_EFFICIENT_LOADING"] = "False"

    _sft_args.enable_fsdp_ram_efficient_loading = _no_ram_efficient

if os.environ.get("BENCH_CANONICAL_INJECT", "1") == "1":
    from swift.template.base import Template

    canon, _ = bench_callback.load_canonical()
    key2row = {_key(json.loads(line)["messages"]): i for i, line in enumerate(Path(MSG).read_text().splitlines())}
    assert len(key2row) == len(canon), "message keys are not unique"
    _orig_encode = Template.encode

    def encode(self, inputs, *args, **kwargs):
        encoded = _orig_encode(self, inputs, *args, **kwargs)
        if self.mode != "train" or not isinstance(encoded, dict):
            return encoded
        msgs = inputs["messages"] if isinstance(inputs, dict) else inputs.chosen.messages
        row = canon[key2row[_key(msgs)]]
        n = len(row["input_ids"])
        encoded["input_ids"] = list(row["input_ids"])
        encoded["labels"] = list(row["labels"])
        if encoded.get("mm_token_type_ids") is not None:
            encoded["mm_token_type_ids"] = torch.zeros_like(encoded["mm_token_type_ids"])[..., :n]
        if encoded.get("loss_scale") is not None:
            encoded["loss_scale"] = [1.0 if x != -100 else 0.0 for x in row["labels"]]
        for k in ("length", "lengths"):
            if k in encoded:
                encoded[k] = n if isinstance(encoded[k], int) else [n]
        return encoded

    Template.encode = encode

from swift.pipelines import sft_main  # noqa: E402

if __name__ == "__main__":
    sft_main(sys.argv[1:])
