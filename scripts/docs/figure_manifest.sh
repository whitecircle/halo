#!/usr/bin/env bash
# The stale-figure gate. `make diagrams` writes agent-docs/assets/figures.sha256, the SHA-256 of every
# figure PNG and of everything it is rendered from: its page, the data folders the page names, the
# visuals kit and the render scripts. `make docs` and the docs workflow check the tree against it, so a
# page, kit or data edit committed without a re-render, or a PNG changed by hand, fails without the
# licensed fonts a render needs.
#
# usage: scripts/docs/figure_manifest.sh write|check
set -euo pipefail

cd "$(dirname "$0")/../.."
manifest=agent-docs/assets/figures.sha256

# Every file the manifest covers, one repo-relative path per line, sorted.
list_files() {
  local page data
  {
    for page in agent-docs/assets/*/*.html; do
      echo "$page"
      echo "${page%.html}.png"
      for data in "$(dirname "$page")"/*/; do
        [[ -d $data ]] || continue
        if grep -qF "$(basename "$data")/" "$page"; then
          find "${data%/}" -type f ! -name '.*'
        fi
      done
    done
    find skills/visuals/kit -type f ! -name '.*'
    echo skills/visuals/scripts/render.sh
    echo skills/visuals/scripts/canvas-size.sh
  } | LC_ALL=C sort -u
}

case "${1:-}" in
  write)
    list_files | xargs shasum -a 256 > "$manifest"
    ;;
  check)
    remedy="Re-render with 'make diagrams' (on a host with Chrome, the ABC fonts and OptiPNG) and commit the PNGs with $manifest."
    if [[ ! -f $manifest ]]; then
      echo "$manifest is missing. $remedy" >&2
      exit 1
    fi
    unlisted=$(LC_ALL=C comm -23 <(list_files) <(awk '{print $2}' "$manifest" | LC_ALL=C sort))
    if [[ -n $unlisted ]]; then
      echo "Not in $manifest, so never rendered with it:" >&2
      echo "$unlisted" >&2
      echo "$remedy" >&2
      exit 1
    fi
    if ! shasum -a 256 --check --quiet "$manifest" >&2; then
      echo "These files changed since the last 'make diagrams', so their figures may be stale. $remedy" >&2
      exit 1
    fi
    ;;
  *)
    echo "usage: $0 write|check" >&2
    exit 2
    ;;
esac
