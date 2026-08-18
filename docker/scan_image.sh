#!/bin/bash
# Pre-publication image scan: refuse to push an image carrying credentials, identity state, or
# authenticated remotes. Bounded to the directories such state realistically lands in — a scan of
# the full 30 GB filesystem would take minutes for no additional coverage.
#
# usage: docker/scan_image.sh <image>          exits 1 (with a listing) on any finding
set -uo pipefail

IMAGE=${1:?usage: docker/scan_image.sh <image>}

docker run --rm -i --entrypoint bash "$IMAGE" -s <<'SCAN'
set -uo pipefail
fail=0
hit() { echo "SECRET-SCAN FINDING: $*"; fail=1; }

# Files that must not exist in a published image.
for f in /root/.claude.json /root/.netrc /root/.git-credentials /root/.bash_history \
         /root/.python_history /root/.wget-hsts /workspace/.env; do
    [ -e "$f" ] && hit "file present: $f"
done
[ -d /root/.aws ] && [ -n "$(ls -A /root/.aws 2>/dev/null)" ] && hit "non-empty /root/.aws"
[ -d /root/.ssh ] && ls /root/.ssh/id_* >/dev/null 2>&1 && hit "SSH key material in /root/.ssh"

# Content patterns in the places state lands: HOME, the shipped repo, and system config.
PATTERNS='-----BEGIN (RSA|OPENSSH|EC|DSA|PGP) PRIVATE KEY|AKIA[0-9A-Z]{16}|hf_[A-Za-z0-9]{30,}|sk-ant-[A-Za-z0-9_-]{20,}|ghp_[A-Za-z0-9]{36}|github_pat_[A-Za-z0-9_]{36,}|xox[bap]-[A-Za-z0-9-]{10,}|AIza[0-9A-Za-z_-]{35}'
matches=$(grep -rIlE "$PATTERNS" /root /workspace /etc/profile.d 2>/dev/null | head -20)
[ -n "$matches" ] && hit "credential-shaped content in: $matches"

# Internal identifiers that must never ship: a stale /workspace baked from a pre-release tree is
# exactly what these catch (the credential regexes above cannot).
INTERNAL='whitecircle-research|846247541878|internal-halo|research-model-spec|research-halo-internal|mintaka|policies_judge|prompts_gradient'
imatches=$(grep -rIlE "$INTERNAL" /workspace 2>/dev/null | head -20)
[ -n "$imatches" ] && hit "internal identifier in: $imatches"

# Authenticated git remotes (user:token@host) anywhere git config lives.
auth=$(grep -rIlE 'url *= *https?://[^/@ ]+:[^/@ ]+@' /root/.gitconfig /workspace/.git /opt 2>/dev/null | head -5)
[ -n "$auth" ] && hit "authenticated git remote in: $auth"

if [ "$fail" -eq 0 ]; then echo "SECRET-SCAN CLEAN"; fi
exit "$fail"
SCAN
rc=$?
if [ "$rc" -ne 0 ]; then
    echo "docker/scan_image.sh: $IMAGE FAILED the pre-publication scan — fix the Dockerfile layer that introduced the finding (an rm in a later layer still ships the bytes)." >&2
fi
exit "$rc"
