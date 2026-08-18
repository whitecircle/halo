#!/usr/bin/env python
"""NVLink health check: per-link hard errors, FEC correction rate, and active bandwidth.

A marginal NVLink lane shows up as thousands of nonfatal ``Xid 145`` (``RLW_RXPIPE``) lines per hour
in ``dmesg`` while the link keeps running at full speed, because forward error correction absorbs the
damage. This separates that noise from the counters that mean data was at risk.

A link is unhealthy when it shows hard errors (Rx, symbol, link recovery, link integrity: the ones
FEC did not absorb). Corrections reaching the deepest FEC bins, or a correction rate high enough to
warrant a physical inspection, mark it marginal rather than failed. Plain correctable churn is
reported only.

Exits non-zero when any link is unhealthy, so it works as a preflight gate before a long run:

    python scripts/profiling/nvlink_health.py                 # summary, exit 1 if unhealthy
    python scripts/profiling/nvlink_health.py --per-link      # every link, not just flagged ones
    python scripts/profiling/nvlink_health.py --json          # machine-readable

Runs against ``nvidia-smi`` only (no GPU allocation, no torch), so it is safe to run alongside a
live training job.
"""

import argparse
import json
import re
import subprocess
import sys
from collections import defaultdict

# Counters that mean FEC did not absorb the error: any non-zero value is a fabric fault.
HARD_ERROR_KEYS = (
    "Malformed packet Errors",
    "Buffer overrun Errors",
    "Rx Errors",
    "Rx remote Errors",
    "Rx General Errors",
    "Local link integrity Errors",
    "Tx discards",
    "Link recovery successful events",
    "Link recovery failed events",
    "Effective Errors",
    "Symbol Errors",
)

# FEC bin N = codewords that needed N symbol corrections. Deep bins mean individual codewords are
# consuming much of the correction budget, but a handful across ~1e13 codewords is normal link noise,
# so depth alone is a watch signal rather than a fault.
DEEP_FEC_BIN = 3

# Corrected-codeword rate above which a link deserves a physical look. Healthy links on this class
# of hardware sit around 1e-9; a marginal-but-corrected lane runs ~1e-7.
CORRECTION_RATE_WARN = 1e-8

_COUNTER_RE = re.compile(r"Link (\d+): (.+?):\s*(\d+)")


def _nvidia_smi(*args: str) -> str:
    """Run nvidia-smi, raising with its stderr rather than returning partial output."""
    result = subprocess.run(("nvidia-smi", *args), capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"nvidia-smi {' '.join(args)} failed ({result.returncode}): {result.stderr.strip()}")
    return result.stdout


def gpu_count() -> int:
    return len([line for line in _nvidia_smi("--list-gpus").splitlines() if line.strip()])


def link_counters(gpu: int) -> dict[int, dict[str, int]]:
    """``{link_index: {counter_name: value}}`` for one GPU."""
    counters: dict[int, dict[str, int]] = defaultdict(dict)
    for match in _COUNTER_RE.finditer(_nvidia_smi("nvlink", "-e", "-i", str(gpu))):
        link, name, value = match.groups()
        counters[int(link)][name.strip()] = int(value)
    return dict(counters)


def link_speeds(gpu: int) -> dict[int, str]:
    """``{link_index: "53.125 GB/s"}`` for the links that are up."""
    speeds = {}
    for line in _nvidia_smi("nvlink", "-s", "-i", str(gpu)).splitlines():
        match = re.search(r"Link (\d+):\s*(.+)", line)
        if match:
            speeds[int(match.group(1))] = match.group(2).strip()
    return speeds


def assess_link(counters: dict[str, int]) -> dict:
    """Classify one link from its counters.

    Only errors FEC did not absorb mean data was at risk, so those alone make a link unhealthy.
    Pre-FEC churn (a high correction rate, or codewords reaching the deep bins) is reported as
    ``marginal``: the link carries data correctly at full bandwidth but has less physical margin than
    its peers, and is the usual source of nonfatal Xid 145 messages.
    """
    hard = {key: counters[key] for key in HARD_ERROR_KEYS if counters.get(key)}
    fec = {int(m.group(1)): v for k, v in counters.items() if (m := re.fullmatch(r"FEC Errors - (\d+)", k))}
    if not fec and not any(key in counters for key in HARD_ERROR_KEYS):
        # A driver spelling the counters differently (the pre-FEC Replay/Recovery/CRC set) would
        # otherwise read as a clean link with nothing checked.
        raise RuntimeError(
            f"unrecognized NVLink counter set {sorted(counters)}: none of the hard-error or FEC counters "
            "this check reads is present, so the nvidia-smi nvlink -e format is not the one parsed here."
        )
    clean, corrected = fec.get(0, 0), sum(v for bin_index, v in fec.items() if bin_index > 0)
    deep = {b: v for b, v in fec.items() if b >= DEEP_FEC_BIN and v}
    total = clean + corrected
    rate = corrected / total if total else 0.0
    return {
        "hard_errors": hard,
        "deep_fec_bins": deep,
        "correction_rate": rate,
        "corrected_codewords": corrected,
        "unhealthy": bool(hard),
        "marginal": not hard and (rate > CORRECTION_RATE_WARN or bool(deep)),
    }


def collect() -> dict[int, dict[int, dict]]:
    report: dict[int, dict[int, dict]] = {}
    for gpu in range(gpu_count()):
        speeds = link_speeds(gpu)
        report[gpu] = {
            link: {**assess_link(counters), "speed": speeds.get(link, "down")}
            for link, counters in link_counters(gpu).items()
        }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--per-link", action="store_true", help="print every link, not only flagged ones")
    parser.add_argument("--json", action="store_true", help="emit the raw report as JSON")
    args = parser.parse_args()

    report = collect()
    if args.json:
        print(json.dumps(report, indent=2))

    unhealthy = [(g, ln) for g, links in report.items() for ln, r in links.items() if r["unhealthy"]]
    marginal = [(g, ln) for g, links in report.items() for ln, r in links.items() if r["marginal"]]

    if not args.json:
        for gpu, links in sorted(report.items()):
            for link, result in sorted(links.items()):
                flagged = result["unhealthy"] or result["marginal"]
                if not (args.per_link or flagged):
                    continue
                status = "UNHEALTHY" if result["unhealthy"] else ("marginal" if result["marginal"] else "ok")
                print(
                    f"GPU {gpu} link {link:>2}: {status:<9} {result['speed']:>12}  "
                    f"corrected={result['corrected_codewords']:>12,} rate={result['correction_rate']:.2e}"
                )
                if result["hard_errors"]:
                    print(f"    hard errors: {result['hard_errors']}")
                if result["deep_fec_bins"]:
                    print(f"    deep FEC bins (>={DEEP_FEC_BIN}): {result['deep_fec_bins']}")

        total_links = sum(len(links) for links in report.values())
        print(
            f"\n{total_links} links checked: {len(unhealthy)} unhealthy, {len(marginal)} marginal "
            f"(corrected-only, FEC absorbing it), {total_links - len(unhealthy) - len(marginal)} clean"
        )
        if marginal and not unhealthy:
            print(
                "Marginal links carry data correctly at full bandwidth — FEC is absorbing the errors, "
                "and they are the usual source of nonfatal Xid 145 log spam. Track the rate over time; "
                "a physical reseat is the fix if it climbs or reaches the deep FEC bins."
            )
    return 1 if unhealthy else 0


if __name__ == "__main__":
    sys.exit(main())
