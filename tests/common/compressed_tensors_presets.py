"""What compressed-tensors serializes as the ``quantization_config`` of each block-scaled format's preset, which the
training image does not install. The record is taken in the upstream image ``Dockerfile.vllm`` builds on,
``vllm/vllm-openai:v<VLLM_VERSION>``, whose compressed-tensors the Halo layers leave as is. Run there from the repo root
with ``--entrypoint python3``, ``-m tests.common.compressed_tensors_presets`` checks the record and ``--regen``
rewrites it."""

from __future__ import annotations

import importlib.metadata
import json
from pathlib import Path

import torch
from compressed_tensors.compressors.format import infer_module_format
from compressed_tensors.quantization import QuantizationConfig, QuantizationScheme, QuantizationStatus
from compressed_tensors.quantization.quant_scheme import PRESET_SCHEMES

from tests.common.goldens import regen_main

RECORD_PATH = Path(__file__).with_suffix(".json")
# MX trains with both GEMM operands quantized; nvfp4 exports weight-only.
PRESETS = {"mxfp8": "MXFP8", "mxfp4": "MXFP4", "nvfp4": "NVFP4A16"}
# Carried verbatim into each config, so the test hands the export the same targets.
TARGETS = [r"re:(?:.*\.)?layers\.(?:0)\.mlp\.gate_proj$"]


def _serialized_preset(preset: str) -> dict:
    """The preset as one config group over ``TARGETS``, under the compressor the library picks for it on a Linear."""
    compression_format = infer_module_format(
        torch.nn.Linear, QuantizationScheme(targets=TARGETS, **PRESET_SCHEMES[preset])
    ).value
    scheme = QuantizationScheme(targets=TARGETS, format=compression_format, **PRESET_SCHEMES[preset])
    config = QuantizationConfig(
        config_groups={"group_0": scheme},
        format=compression_format,
        quantization_status=QuantizationStatus.COMPRESSED,
    )
    return json.loads(config.model_dump_json())


def render_record() -> str:
    record = {
        "vllm": importlib.metadata.version("vllm"),
        "compressed_tensors": importlib.metadata.version("compressed-tensors"),
        "targets": TARGETS,
        "presets": {fmt: _serialized_preset(preset) for fmt, preset in PRESETS.items()},
    }
    return json.dumps(record, indent=2) + "\n"


if __name__ == "__main__":
    regen_main(RECORD_PATH, render_record)
