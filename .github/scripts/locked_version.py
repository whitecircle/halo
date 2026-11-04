"""Print the version ``uv.lock`` pins a package at, for a hosted job that installs that package alone.

python .github/scripts/locked_version.py <package>
"""

import argparse
import sys
import tomllib
from pathlib import Path

UV_LOCK = Path(__file__).resolve().parents[2] / "uv.lock"


def locked_version(package: str) -> str:
    """The version ``uv.lock`` pins ``package`` at; fails unless it pins exactly one."""
    with UV_LOCK.open("rb") as lock:
        versions = [entry["version"] for entry in tomllib.load(lock)["package"] if entry["name"] == package]
    if len(versions) != 1:
        raise SystemExit(f"::error::uv.lock pins {len(versions)} versions of {package}, not one: {versions}")
    return versions[0]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("package")
    sys.stdout.write(f"{locked_version(parser.parse_args(argv).package)}\n")


if __name__ == "__main__":
    main()
