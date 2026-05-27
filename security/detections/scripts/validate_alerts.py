#!/usr/bin/env python3
import sys, json, requests, argparse, xml.etree.ElementTree as ET
from datetime import datetime, timedelta

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--host", required=True)
    p.add_argument("--api-key", required=True)
    p.add_argument("--attack-report", required=True)
    p.add_argument("--output", required=True)
    return p.parse_args()

def main():
    args = parse_args()
    # Parse Atomic report to get list of executed technique IDs
    tree = ET.parse(args.attack_report)
    techniques = []
    for testcase in tree.findall(".//testcase"):
        name = testcase.get("name", "")
        # Extract technique ID (e.g., "T1059")
        techniques.append(name.split()[0])  # simplified
    # Query Elasticsearch for alerts triggered in the last 30 minutes
    headers = {"Authorization": f"ApiKey {args.api_key}"}
    now = datetime.utcnow()
    time_from = (now - timedelta(minutes=30)).isoformat() + "Z"
    query = {"query": {"range": {"@timestamp": {"gte": time_from}}}}
    resp = requests.get(f"{args.host}/.siem-signals-*/_search", json=query, headers=headers)
    alerts = resp.json()
    triggered_techniques = set()
    for hit in alerts.get("hits", {}).get("hits", []):
        for tag in hit.get("_source", {}).get("tags", []):
            if tag.startswith("attack."):
                triggered_techniques.add(tag.split(".")[-1].upper())
    # Determine pass/fail per technique
    total = len(techniques)
    passed = 0
    failures = []
    for tech in techniques:
        if tech in triggered_techniques:
            passed += 1
        else:
            failures.append(tech)
    # Generate JUnit report
    report_xml = f'''<?xml version="1.0" encoding="UTF-8"?>
    <testsuite name="Detection Validation" tests="{total}" failures="{len(failures)}" errors="0" skipped="0">
    '''
    for tech in techniques:
        if tech in triggered_techniques:
            report_xml += f'  <testcase name="{tech}" classname="detection.atomic"/>\n'
        else:
            report_xml += f'''  <testcase name="{tech}" classname="detection.atomic">
                <failure>Technique {tech} not detected</failure>
            </testcase>
            '''
    report_xml += '</testsuite>'
    with open(args.output, 'w') as f:
        f.write(report_xml)
    if failures:
        print(f"Failed techniques: {failures}")
        sys.exit(1)
    else:
        print("All techniques detected successfully")

if __name__ == "__main__":
    main()