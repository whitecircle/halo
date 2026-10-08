"""The Hub files the CPU tier reads: configs, tokenizers, chat templates and remote code, never weights.

The repos are derived, not listed: every Hub id a shipped example trains (``model_name_or_path``
under ``examples/``), every checkpoint constant in :mod:`tests.common.models` (where a test names what
it loads), and the snapshots its ``PINNED_REVISIONS`` pins. A test that reads a repo outside that set
skips, and fails under ``HALO_TEST_REQUIRE_HUB_CACHE`` (:mod:`tests.common.tokenizers`).

    python -m tests.common.hub_seed          # download the seed into HF_HOME
    python -m tests.common.hub_seed --list   # one ``repo`` or ``repo@revision`` per line
"""

import argparse
import functools
import re
from pathlib import Path

import yaml
from huggingface_hub import snapshot_download

from tests.common import models
from tests.common.utils import REPO_ROOT

ALLOW_PATTERNS = ["*.json", "*.jinja", "*.txt", "*.model", "*.py", "*.tiktoken"]
# The Hub refuses these to an anonymous client, so the seed leaves them out and their tests skip. A
# repo that stops being gated leaves this set, or its tests go on skipping where they could run.
GATED_REPOS = frozenset({models.GEMMA3_4B_IT})
# ``namespace/name``; an absolute path or a deeper directory tree never matches.
_HUB_ID = re.compile(r"[A-Za-z0-9][\w.-]*/[\w.-]+")


@functools.cache
def _example_configs() -> tuple[dict, ...]:
    return tuple(
        yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        for path in sorted((REPO_ROOT / "examples").rglob("*.yaml"))
    )


@functools.cache
def local_output_roots() -> frozenset[str]:
    """The top-level directories the example runs write into (``output_dir``).

    A later stage trains the checkpoint an earlier one wrote, so ``checkpoints/<run>`` is a local
    path with the shape of a Hub id and must not be fetched from the Hub.
    """
    return frozenset(
        Path(config["output_dir"]).parts[0]
        for config in _example_configs()
        if isinstance(config.get("output_dir"), str) and not Path(config["output_dir"]).is_absolute()
    )


def is_hub_repo(reference: str) -> bool:
    """Whether ``reference`` names a Hub repo rather than a local checkpoint."""
    return (
        bool(_HUB_ID.fullmatch(reference))
        and reference.split("/", 1)[0] not in local_output_roots()
        and not Path(reference).exists()
    )


def example_repos() -> set[str]:
    """The Hub ids the shipped example configs train."""
    references = (config.get("model_name_or_path") for config in _example_configs())
    return {ref for ref in references if isinstance(ref, str) and is_hub_repo(ref)}


def roster_repos() -> set[str]:
    """The Hub ids :mod:`tests.common.models` names."""
    return {
        value
        for name, value in vars(models).items()
        if name.isupper() and isinstance(value, str) and is_hub_repo(value)
    }


def seed() -> list[tuple[str, str | None]]:
    """Every ``(repo, revision)`` to fetch; ``None`` is the repo's main."""
    latest = sorted((example_repos() | roster_repos()) - GATED_REPOS)
    return [(repo, None) for repo in latest] + sorted(models.PINNED_REVISIONS.items())


def download() -> None:
    """Fetch the whole seed, then exit naming every repo that could not be fetched."""
    failures = []
    for repo, revision in seed():
        try:
            snapshot_download(repo, revision=revision, allow_patterns=ALLOW_PATTERNS)
        except Exception as e:
            failures.append(f"  {repo}@{revision or 'main'}: {type(e).__name__}: {e}")
    if failures:
        raise SystemExit("hub seed: cannot fetch\n" + "\n".join(failures))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("--list", action="store_true", help="print the seed instead of downloading it")
    if parser.parse_args().list:
        for repo, revision in seed():
            print(repo if revision is None else f"{repo}@{revision}")
    else:
        download()


if __name__ == "__main__":
    main()
