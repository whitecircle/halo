#!/usr/bin/env python
"""Create the policy sets of a JSON file in a White Circle environment and print the matching ``rewards:`` block, one
``whitecircle`` term per set.

    WHITECIRCLE_API_KEY=wc-... halo run create-whitecircle-policies \\
        --environment-id <id> --policies examples/grpo/online/whitecircle/policies.json

One condition, ``halo-grpo``, selects every policy of the file for the environment. A policy whose name already exists
is reused when its rules match the file's, so a rerun creates nothing twice. With other rules, one ``halo-grpo`` selects
(this tool's own) is replaced by the file's version; any other stops the run, naming it, as it is not the file's to
change."""

import argparse
import json
import os
import urllib.error
import urllib.request

API_VERSION = "2026-06-01"
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


def _rule_texts(rules: list[dict]) -> list[tuple[str, str]]:
    return sorted(
        (kind, rule[kind]) for rule in rules for kind in ("flagged_content", "allowed_content") if rule.get(kind)
    )


def ensure_policies(client: Client, specs: list[dict], existing: dict[str, str], owned: set[str]) -> list[str]:
    """Each policy's id: reused when its rules match, replaced when it is one of ``owned`` (``halo-grpo``'s), else
    created; a same-named policy outside ``owned`` with other rules stops the run."""
    ids = []
    for spec in specs:
        rules = [{"flagged_content": text} for text in spec["flagged_content"]]
        rules += [{"allowed_content": text} for text in spec.get("allowed_content", [])]
        policy_id = existing.get(spec["name"])
        if policy_id is not None:
            current = client.call("GET", f"/api/policy/get_by_id?id={policy_id}")["rules"]
            if _rule_texts(current) != _rule_texts(rules):
                if policy_id not in owned:
                    raise SystemExit(
                        f"a policy named {spec['name']!r} already exists with other rules; rename it in the file or "
                        "in White Circle, or delete the existing one"
                    )
                # The API's update keeps a policy's rule set, so the file's version replaces it.
                client.call("POST", "/api/policy/delete", {"id": policy_id})
                policy_id = None
        if policy_id is None:
            body = {
                "name": spec["name"],
                "rules": rules,
                "content_type": "any",
                "applies_to": "output",
                "mode": "prod",
            }
            policy_id = existing[spec["name"]] = client.call("POST", "/api/policy/create", body)["id"]
        ids.append(policy_id)
    return ids


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--environment-id", required=True, help="White Circle environment the policies attach to")
    parser.add_argument(
        "--policies",
        required=True,
        help="JSON mapping each set's name to its policies (name, flagged_content, allowed_content)",
    )
    parser.add_argument("--region", default="us", choices=["us", "eu"], help="API region (default: us)")
    args = parser.parse_args()

    key = os.environ.get("WHITECIRCLE_API_KEY")
    if not key:
        raise SystemExit("set WHITECIRCLE_API_KEY")
    client = Client(args.region, key)
    with open(args.policies) as fh:
        sets = json.load(fh)

    condition = next((c for c in client.call("GET", "/api/condition/list") if c["name"] == CONDITION_NAME), None)
    owned = set(condition["policy_ids"]) if condition else set()
    existing = {policy["name"]: policy["id"] for policy in client.call("GET", "/api/policy/list")}
    set_ids = {name: ensure_policies(client, specs, existing, owned) for name, specs in sets.items()}
    every_id = [policy_id for ids in set_ids.values() for policy_id in ids]

    if condition is None:
        body = {"name": CONDITION_NAME, "environment_ids": [args.environment_id], "policy_ids": every_id}
        client.call("POST", "/api/condition/create", body)
    else:
        # The condition selects exactly the file's policies, so one the file dropped stops scoring.
        environment_ids = sorted(set(condition["environment_ids"]) | {args.environment_id})
        body = {"id": condition["id"], "policy_ids": every_id, "environment_ids": environment_ids}
        client.call("POST", "/api/condition/update", body)

    print("rewards:")
    for name, ids in set_ids.items():
        print(f"  - source: whitecircle\n    name: {name}\n    environment_id: {args.environment_id}")
        print(f"    policies: [{', '.join(ids)}]")


if __name__ == "__main__":
    main()
