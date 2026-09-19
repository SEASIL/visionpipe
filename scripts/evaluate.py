#!/usr/bin/env python
"""Evaluate a detector on a dataset split and gate on quality (use in CI / DVC).

Fails (exit 1) if mAP@0.5 is below --min-map50, or if it dropped more than --max-drop versus a
stored baseline JSON. That turns 'did the new model get worse?' into an automatic check.

Usage:
  python scripts/evaluate.py --weights models/best.pt --data data/dataset/data.yaml \
      --min-map50 0.5 --baseline metrics/baseline.json --max-drop 0.02 --out metrics/eval.json
"""
import argparse
import json
import sys
from pathlib import Path

from ultralytics import YOLO


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help=".pt / .onnx / .engine")
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--min-map50", type=float)
    ap.add_argument("--baseline", help="JSON from a previous evaluation to compare against")
    ap.add_argument("--max-drop", type=float, default=0.02)
    ap.add_argument("--out", default="metrics/eval.json")
    args = ap.parse_args()

    res = YOLO(args.weights).val(data=args.data, split=args.split, imgsz=args.imgsz, verbose=False)
    box = res.box
    metrics = {
        "map50": float(box.map50),
        "map50_95": float(box.map),
        "precision": float(box.mp),
        "recall": float(box.mr),
        "per_class_ap50": {res.names[int(c)]: float(v) for c, v in zip(box.ap_class_index, box.ap50)},
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(metrics, indent=2))
    print(json.dumps({k: v for k, v in metrics.items() if k != "per_class_ap50"}, indent=2))

    failed = False
    if args.min_map50 is not None and metrics["map50"] < args.min_map50:
        print(f"FAIL: mAP50 {metrics['map50']:.3f} < required {args.min_map50}")
        failed = True
    if args.baseline and Path(args.baseline).exists():
        base = json.loads(Path(args.baseline).read_text())["map50"]
        if metrics["map50"] < base - args.max_drop:
            print(f"FAIL: mAP50 dropped {base - metrics['map50']:.3f} vs baseline {base:.3f} (allowed {args.max_drop})")
            failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
