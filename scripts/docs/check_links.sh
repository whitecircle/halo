#!/usr/bin/env bash
# The docs link gate `make docs` and the docs workflow run: every relative link and heading anchor
# in the plain-GitHub-rendered doc trees must resolve. The checker is check_links.py (stdlib only).
#
# usage: scripts/docs/check_links.sh [path ...]   (default: agent-docs, human-docs, skills + root markdown)
exec python3 "$(dirname "$0")/check_links.py" "$@"
