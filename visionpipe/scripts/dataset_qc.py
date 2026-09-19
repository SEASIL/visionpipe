#!/usr/bin/env python
"""Automated quality checks for a YOLO-format dataset. Exit code 1 on errors (CI/DVC friendly).

Usage:
  python scripts/dataset_qc.py --images data/train/images --labels data/train/labels --num-classes 3 \
         --out metrics/qc_train.json
"""
import argparse
import json
import sys
from pathlib import Path

from visionpipe.datatools import run_qc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--num-classes", type=int)
    ap.add_argument("--min-side", type=float, default=0.005, help="Warn on boxes smaller than this (normalised)")
    ap.add_argument("--check-images", action="store_true", help="Also try to decode every image")
    ap.add_argument("--strict", action="store_true", help="Treat warnings as failures")
    ap.add_argument("--out", help="Write the JSON report here")
    args = ap.parse_args()

    rep = run_qc(args.images, args.labels, args.num_classes, min_side_norm=args.min_side, check_images=args.check_images)
    d = rep.to_dict()
    print(f"images={d['images']} boxes={d['boxes']} background={d['background_images']} classes={d['class_counts']}")
    print(f"errors={d['errors']} warnings={d['warnings']}")
    for i in rep.issues:
        if i.severity != "info":
            print(f"  [{i.severity}] {i.code}: {i.file} - {i.message}")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(d, indent=2))
    return 1 if rep.errors or (args.strict and rep.warnings) else 0


if __name__ == "__main__":
    sys.exit(main())
