#!/usr/bin/env python
"""Create the policy sets of a JSON file in a White Circle environment and print the matching ``rewards:`` block, one
``whitecircle`` term per set.

A set is a list of policies, each a separate verdict, or a checklist: one judgment whose items are checked one by one
and whose confidence is the fraction of items hit. Each item becomes a policy named under the checklist
(``Defensive handling · Unvalidated request input``) with the checklist's shared ``allowed_content`` added to its own,
and its term lists them as ``checklist:``. The judge reads a policy's name, so an item's name states what it checks.

    WHITECIRCLE_API_KEY=wc-... halo run create-whitecircle-policies \\
        --environment-id <id> --policies examples/grpo/online/whitecircle/policies.json

One condition, ``halo-grpo``, selects every policy of the file for the environment. A policy whose name already exists
is reused when its rules match the file's, so a rerun creates nothing twice. With other rules, one that ``halo-grpo``
alone selects (this tool's own) is replaced by the file's version; any other stops the run before it changes anything,
naming every such policy, as they are not the file's to change. ``--dry-run`` prints what a run would reuse, replace
and create, and changes nothing."""

import argparse
import json
import urllib.error
import urllib.request
from typing import NamedTuple

from halo.env import env_str
from halo.rewards.scorers.whitecircle import API_KEY_VARIABLE, API_VERSION
from halo.rewards.terms import WhiteCircleRegion

CONDITION_NAME = "halo-grpo"


class Client:
    def __init__(self, region: str, key: str):
        self.base = f"https://{region}.whitecircle.com"
        self.headers = {
            "Authorization": f"Bearer {key}",
            "whitecircle-version": API_VERSION,
            "Content-Type": "application/json",
        }

    def call(self, method: str, path: str, body: dict | None = None):
        data = None if body is None else json.dumps(body).encode()
        request = urllib.request.Request(self.base + path, data=data, headers=self.headers, method=method)
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return json.load(response)
        except urllib.error.HTTPError as e:
            raise SystemExit(f"{method} {path} answered {e.code}: {e.read().decode()[:300]}") from None


class SyncPlan(NamedTuple):
    """What a run does with the file's policies, by name: ``reused`` and ``replaced`` map to the existing policy's id,
    ``created`` names the ones no policy exists for."""

    reused: dict[str, str]
    replaced: dict[str, str]
    created: tuple[str, ...]


def _rules(spec: dict) -> list[dict]:
    rules = [{"flagged_content": text} for text in spec["flagged_content"]]
    return rules + [{"allowed_content": text} for text in spec.get("allowed_content", [])]


def _rule_texts(rules: list[dict]) -> list[tuple[str, str]]:
    return sorted(
        (kind, rule[kind]) for rule in rules for kind in ("flagged_content", "allowed_content") if rule.get(kind)
    )


def plan_sync(client: Client, specs: list[dict], existing: dict[str, str], owned: set[str]) -> SyncPlan:
    """Reads only. A same-named policy outside ``owned`` with other rules stops the run, every such policy named."""
    reused: dict[str, str] = {}
    replaced: dict[str, str] = {}
    foreign: list[str] = []
    for spec in specs:
        name = spec["name"]
        policy_id = existing.get(name)
        if policy_id is None:
            continue
        current = client.call("GET", f"/api/policy/get_by_id?id={policy_id}")["rules"]
        if _rule_texts(current) == _rule_texts(_rules(spec)):
            reused[name] = policy_id
        elif policy_id in owned:
            replaced[name] = policy_id
        else:
            foreign.append(name)
    if foreign:
        raise SystemExit(
            f"policies named {foreign} already exist with other rules; rename them in the file or in White Circle, or "
            "delete the existing ones"
        )
    created = tuple(dict.fromkeys(spec["name"] for spec in specs if spec["name"] not in existing))
    return SyncPlan(reused, replaced, created)


def policy_specs(entry: list[dict] | dict) -> list[dict]:
    """A set's policies; a checklist's items as policies named under it, each with the shared allowed content."""
    if isinstance(entry, list):
        return entry
    shared = entry.get("allowed_content", [])
    return [
        {
            "name": f"{entry['checklist']} · {item['name']}",
            "flagged_content": item["flagged_content"],
            "allowed_content": [*item.get("allowed_content", []), *shared],
        }
        for item in entry["items"]
    ]


def sync(client: Client, environment_id: str, sets: dict[str, list[dict] | dict], *, dry_run: bool) -> dict:
    """Each set's policy ids once ``halo-grpo`` selects exactly the file's policies; a dry run returns the
    :class:`SyncPlan` and sends no write. The condition never selects a deleted policy: one to replace leaves the
    condition before its delete, and its replacement joins with the final update."""
    conditions = client.call("GET", "/api/condition/list")
    condition = next((c for c in conditions if c["name"] == CONDITION_NAME), None)
    borrowed = {policy_id for c in conditions if c is not condition for policy_id in c["policy_ids"]}
    owned = set(condition["policy_ids"]) - borrowed if condition else set()
    existing = {policy["name"]: policy["id"] for policy in client.call("GET", "/api/policy/list")}
    set_specs = {name: policy_specs(entry) for name, entry in sets.items()}
    specs = [spec for members in set_specs.values() for spec in members]
    plan = plan_sync(client, specs, existing, owned)
    if dry_run:
        return plan

    stale = set(plan.replaced.values())
    if stale:
        kept = [policy_id for policy_id in condition["policy_ids"] if policy_id not in stale]
        # With every selected policy stale there is nothing to keep, and the final update is the one that repoints.
        if kept:
            client.call("POST", "/api/condition/update", {"id": condition["id"], "policy_ids": kept})
        for policy_id in stale:
            # The API's update keeps a policy's rule set, so the file's version replaces it.
            client.call("POST", "/api/policy/delete", {"id": policy_id})
    ids = dict(plan.reused)
    for spec in specs:
        if spec["name"] not in ids:
            body = {
                "name": spec["name"],
                "rules": _rules(spec),
                "content_type": "any",
                "applies_to": "output",
                "mode": "prod",
            }
            ids[spec["name"]] = client.call("POST", "/api/policy/create", body)["id"]
    every_id = list(dict.fromkeys(ids[spec["name"]] for spec in specs))

    if condition is None:
        body = {"name": CONDITION_NAME, "environment_ids": [environment_id], "policy_ids": every_id}
        client.call("POST", "/api/condition/create", body)
    else:
        # The condition selects exactly the file's policies, so one the file dropped stops scoring.
        environment_ids = sorted(set(condition["environment_ids"]) | {environment_id})
        body = {"id": condition["id"], "policy_ids": every_id, "environment_ids": environment_ids}
        client.call("POST", "/api/condition/update", body)
    return {name: [ids[spec["name"]] for spec in members] for name, members in set_specs.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--environment-id", required=True, help="White Circle environment the policies attach to")
    parser.add_argument(
        "--policies",
        required=True,
        help="JSON mapping each set's name to its policies (name, flagged_content, allowed_content) or to a checklist "
        "(checklist, allowed_content, items of name, flagged_content, allowed_content)",
    )
    regions = [region.value for region in WhiteCircleRegion]
    parser.add_argument("--region", default=WhiteCircleRegion.US.value, choices=regions, help="API region")
    parser.add_argument("--dry-run", action="store_true", help="print what a run would change, and change nothing")
    args = parser.parse_args()

    key = env_str(API_KEY_VARIABLE)
    if not key:
        raise SystemExit(f"set {API_KEY_VARIABLE}")
    with open(args.policies) as fh:
        sets = json.load(fh)
    outcome = sync(Client(args.region, key), args.environment_id, sets, dry_run=args.dry_run)
    if isinstance(outcome, SyncPlan):
        for verb, names in (("reuse", outcome.reused), ("replace", outcome.replaced), ("create", outcome.created)):
            print(f"{verb} ({len(names)}):", *names, sep="\n  ")
        return

    print("rewards:")
    for name, ids in outcome.items():
        print(f"  - source: whitecircle\n    name: {name}\n    environment_id: {args.environment_id}")
        if args.region != WhiteCircleRegion.US:
            print(f"    region: {args.region}")
        field = "policies" if isinstance(sets[name], list) else "checklist"
        print(f"    {field}: [{', '.join(ids)}]")


if __name__ == "__main__":
    main()
