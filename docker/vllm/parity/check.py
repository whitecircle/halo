"""Build-time config-schema parity gate for the rollout server image.

The server parses every checkpoint it serves with its own transformers, pinned below the training
image's line (``Dockerfile.vllm``). Each directory under ``fixtures/`` holds the ``config.json`` the
toolkit exports for one served family, written by the training image's transformers through every
export-side rewrite a checkpoint writer applies, plus the remote-code config module the export ships
where the family is parsed through ``auto_map`` (see ``generate.py``). It must parse here, its text
config must survive the global attention reads every vLLM model loader makes, and its
``architectures`` must be registered in this vLLM. A directory under ``unparseable/`` is a negative
control, the same config without the rewrite that makes it servable, and must not parse; one that
starts to parse fails the build, since the pin and that rewrite are then due for review.

No test suite can run this: it needs the server's transformers and vLLM, which the training image
does not have (ABI-incompatible stacks). Run against a checkout's fixtures:

    python3 check.py docker/vllm/parity
"""

import json
import os
import sys

from transformers import AutoConfig
from vllm.model_executor.models.registry import ModelRegistry

# The global attention geometry every vLLM loader reads off ``hf_text_config``. A config still
# carrying transformers 5.16's ``per_layer_config`` is refused at parse on the pinned line; one whose
# parse survives it raises AmbiguousGlobalPerLayerAttributeError on these reads instead. A field a
# family does not spell (Step-3.7's vendor config names its own) is not a finding.
# ``rope_parameters`` is the rope schema half: 5.16 exports spell RoPE there and vLLM 0.26.0 reads it
# off the config, so a future divergence in that spelling fails the build rather than a live run.
GEOMETRY_FIELDS = (
    "hidden_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
    "rope_parameters",
)


def load_fixture(path: str):
    """Parse the fixture the way the server does, then read what its loaders read."""
    with open(os.path.join(path, "config.json")) as f:
        payload = json.load(f)
    config = AutoConfig.from_pretrained(path, trust_remote_code="auto_map" in payload)
    text_config = config.get_text_config()
    for field in GEOMETRY_FIELDS:
        getattr(text_config, field, None)
    return payload, config


def fixture_dirs(root: str) -> list[str]:
    """Every directory under ``root``. Empty is fatal, since a gate over no fixtures checks nothing."""
    directories = sorted(
        os.path.join(root, name) for name in os.listdir(root) if os.path.isdir(os.path.join(root, name))
    )
    if not directories:
        raise SystemExit(f"no fixture directories under {root} — the parity gate would certify nothing")
    return directories


def main(parity_root: str) -> int:
    supported_archs = set(ModelRegistry.get_supported_archs())
    failures: list[str] = []

    def check(label: str, ok: bool, detail: str = "") -> None:
        print(f"    {'OK  ' if ok else 'FAIL'} {label}{f' — {detail}' if detail and not ok else ''}")
        if not ok:
            failures.append(label)

    for path in fixture_dirs(os.path.join(parity_root, "fixtures")):
        label = f"fixtures/{os.path.basename(path)}"
        if not os.path.isfile(os.path.join(path, "config.json")):
            check(f"{label} ships a config.json", False, "a fixture directory the gate cannot read")
            continue
        try:
            payload, config = load_fixture(path)
        except Exception as error:
            # Any parse-time failure is the finding: the server would fail on this checkpoint too.
            check(f"{label} parses", False, f"{type(error).__name__}: {error}")
            continue
        check(f"{label} parses as {type(config).__name__}", True)
        architectures = payload.get("architectures") or ()
        check(f"{label} declares architectures", bool(architectures))
        for arch in architectures:
            check(f"{label} architecture {arch} is registered in this vLLM", arch in supported_archs)

    for path in fixture_dirs(os.path.join(parity_root, "unparseable")):
        label = f"unparseable/{os.path.basename(path)}"
        try:
            load_fixture(path)
        except Exception as error:
            # The expected outcome; its type is the report.
            check(f"{label} is refused ({type(error).__name__})", True)
        else:
            check(
                f"{label} is refused",
                False,
                "it parsed — the transformers pin or the export rewrite is due for review",
            )

    if failures:
        print(f"\n{len(failures)} config-parity check(s) failed:\n  " + "\n  ".join(failures))
        return 1
    print("\nconfig-schema parity: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__))))
