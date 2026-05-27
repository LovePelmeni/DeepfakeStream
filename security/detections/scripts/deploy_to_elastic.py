#!/usr/bin/env python3
import argparse
import requests
import sys
from pathlib import Path

def deploy(host, api_key, rules_dir, action):
    headers = {
        "Content-Type": "application/json",
        "kbn-xsrf": "true",
        "Authorization": f"ApiKey {api_key}"
    }
    base_url = host.rstrip('/')
    api_url = f"{base_url}/api/detection_engine/rules"

    rule_files = list(Path(rules_dir).glob("*.eql"))
    if not rule_files:
        print(f"No .eql files found in {rules_dir}")
        sys.exit(1)

    for rule_file in rule_files:
        with open(rule_file) as f:
            eql_query = f.read().strip()
        rule_name = rule_file.stem
        payload = {
            "rule_id": rule_name,
            "name": rule_name,
            "description": f"Auto-deployed from Sigma: {rule_name}",
            "type": "eql",
            "query": eql_query,
            "risk_score": 50,
            "severity": "medium",
            "interval": "5m",
            "from": "now-6m",
            "enabled": True
        }
        # Try to create
        resp = requests.post(api_url, json=payload, headers=headers)
        if resp.status_code not in (200, 201):
            # If conflict, try update
            resp = requests.put(f"{api_url}?rule_id={rule_name}", json=payload, headers=headers)
            if resp.status_code not in (200, 201):
                print(f"Failed to deploy {rule_name}: {resp.text}")
                sys.exit(1)
        print(f"Deployed {rule_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", required=True)
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--rules-dir", required=True)
    parser.add_argument("--action", default="upsert")
    args = parser.parse_args()
    deploy(args.host, args.api_key, args.rules_dir, args.action)