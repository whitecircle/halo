#!/usr/bin/env bash
# Resolve every relative markdown link in the plain-GitHub-rendered files and fail on a broken
# target. `mkdocs build --strict` already covers agent-docs/; human-docs/ and the root markdown
# have no build, so this is what keeps their links honest.
#
# usage: scripts/docs/check_links.sh [path ...]   (default: human-docs, skills + root markdown)
set -uo pipefail

paths=("$@")
[ ${#paths[@]} -eq 0 ] && paths=(human-docs skills README.md CONTRIBUTING.md CLAUDE.md AGENTS.md SECURITY.md CODE_OF_CONDUCT.md)

# A path that no longer exists would leave `find` scanning nothing and the check passing vacuously.
missing=0
for path in "${paths[@]}"; do
    if [ ! -e "$path" ]; then
        echo "MISSING PATH  $path"
        missing=$((missing + 1))
    fi
done
if [ "$missing" -gt 0 ]; then
    echo "FAIL: $missing path(s) to check do not exist"
    exit 1
fi

broken=0
while IFS= read -r file; do
    dir=$(dirname "$file")
    while IFS= read -r target; do
        case "$target" in
            http://* | https://* | mailto:* | '#'*) continue ;;
        esac
        target=${target%%#*}     # drop the anchor
        target=${target%% *}     # drop a link title
        [ -z "$target" ] && continue
        if [ ! -e "$dir/$target" ]; then
            echo "BROKEN  $file  ->  $target"
            broken=$((broken + 1))
        fi
    done < <(grep -oE '\]\([^)]+\)' "$file" | sed -E 's/^\]\(//; s/\)$//')
done < <(find -L "${paths[@]}" -name '*.md' -type f | sort)

if [ "$broken" -gt 0 ]; then
    echo "FAIL: $broken broken relative link(s)"
    exit 1
fi
echo "OK: every relative link resolves"
