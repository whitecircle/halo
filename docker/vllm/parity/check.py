"""Build-time config-schema parity gate for the rollout server image.

The server parses every checkpoint it serves with its own transformers, behind the training image's
line (``Dockerfile.vllm``). Each directory under ``fixtures/`` holds the ``config.json`` the toolkit
exports for one served family, written by the training image's transformers through every
export-side rewrite a checkpoint writer applies, plus the remote-code config module the export ships
where the family is parsed through ``auto_map`` (see ``generate.py``). It must resolve through vLLM's
own ``ModelConfig`` to a per-layer attention geometry, the reads every vLLM model loader makes, and
its ``architectures`` must be registered in this vLLM. A directory under ``unparseable/`` is the same
config without the rewrite: it must be refused, or resolve to exactly the geometry of its rewritten
fixture, in which case the rewrite is lossless here and only kept for older readers.

No test suite can run this: it needs the server's transformers and vLLM, which the training image
does not have (ABI-incompatible stacks). Run against a checkout's fixtures:

    python3 check.py docker/vllm/parity
"""

import json
import os
import sys

from vllm.config import ModelConfig
from vllm.model_executor.models.registry import ModelRegistry


def load_fixture(path: str) -> tuple[dict, ModelConfig, list[tuple[int, int]]]:
    """Parse the fixture the way the server does, then read the per-layer geometry its loaders read."""
    with open(os.path.join(path, "config.json")) as f:
        payload = json.load(f)
    config = ModelConfig(
        model=path, skip_tokenizer_init=True, trust_remote_code="auto_map" in payload, dtype="bfloat16"
    )
    arch = config.model_arch_config
    geometry = [
        (arch[layer].head_size, arch[layer].total_num_kv_heads) for layer in range(arch.total_num_hidden_layers)
    ]
    return payload, config, geometry


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
    geometries: dict[str, list[tuple[int, int]]] = {}

    def check(label: str, ok: bool, detail: str = "") -> None:
        print(f"    {'OK  ' if ok else 'FAIL'} {label}{f' — {detail}' if detail and not ok else ''}")
        if not ok:
            failures.append(label)

    for path in fixture_dirs(os.path.join(parity_root, "fixtures")):
        name = os.path.basename(path)
        label = f"fixtures/{name}"
        if not os.path.isfile(os.path.join(path, "config.json")):
            check(f"{label} ships a config.json", False, "a fixture directory the gate cannot read")
            continue
        try:
            payload, config, geometry = load_fixture(path)
        except Exception as error:
            # Any parse-time failure is the finding: the server would fail on this checkpoint too.
            check(f"{label} parses", False, f"{type(error).__name__}: {error}")
            continue
        geometries[name] = geometry
        check(f"{label} parses as {type(config.hf_config).__name__}", True)
        architectures = payload.get("architectures") or ()
        check(f"{label} declares architectures", bool(architectures))
        for arch in architectures:
            check(f"{label} architecture {arch} is registered in this vLLM", arch in supported_archs)

    for path in fixture_dirs(os.path.join(parity_root, "unparseable")):
        name = os.path.basename(path)
        label = f"unparseable/{name}"
        try:
            _, _, geometry = load_fixture(path)
        except Exception as error:
            # The expected outcome; its type is the report.
            check(f"{label} is refused ({type(error).__name__})", True)
            continue
        check(
            f"{label} resolves to its rewritten fixture's geometry",
            geometry == geometries.get(name),
            "it parsed to a different model than the export serves — the export rewrite is due for review",
        )

    if failures:
        print(f"\n{len(failures)} config-parity check(s) failed:\n  " + "\n  ".join(failures))
        return 1
    print("\nconfig-schema parity: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__))))
