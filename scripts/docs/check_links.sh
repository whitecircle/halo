#!/usr/bin/env bash
# The docs link gate `make docs` and the docs workflow run: every relative link and heading anchor
# in the plain-GitHub-rendered doc trees must resolve, and every path and package name their code
# names must exist. The checker is scripts/docs/check_links.py (stdlib only, Python 3.12+), run as a
# module of the repo root so it imports its siblings; path arguments stay relative to the caller.
#
# usage: scripts/docs/check_links.sh [path ...]   (default: agent-docs, human-docs, skills + root markdown)
root="$(cd "$(dirname "$0")/../.." && pwd)"
export PYTHONPATH="$root${PYTHONPATH:+:$PYTHONPATH}"
exec "$(command -v python3.12 || command -v python3)" -m scripts.docs.check_links "$@"
