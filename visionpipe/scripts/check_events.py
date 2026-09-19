#!/usr/bin/env python
"""Assert event counts in a JSONL file. Used by the CI smoke test.

Usage: python scripts/check_events.py outputs/cam0/events.jsonl --expect loitering=1 zone_intrusion=2
"""
import argparse
import json
import sys
from collections import Counter


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("events")
    ap.add_argument("--expect", nargs="+", required=True, metavar="TYPE=COUNT")
    args = ap.parse_args()
    counts = Counter(json.loads(line)["type"] for line in open(args.events) if line.strip())
    ok = True
    for item in args.expect:
        t, n = item.split("=")
        if counts.get(t, 0) != int(n):
            print(f"FAIL {t}: expected {n}, got {counts.get(t, 0)}")
            ok = False
    print(dict(counts), "OK" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
