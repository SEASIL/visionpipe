#!/usr/bin/env python
"""Pseudo-label images with a detector and build an active-learning review queue.

Workflow: auto_label -> import labels into CVAT -> annotators fix images in review_queue.txt first
-> retrain -> repeat. Images the model is least sure about (confidences near 0.5) are reviewed first.

Usage:
  python scripts/auto_label.py --weights yolov8n.pt --images data/unlabeled --out data/pseudo_labels \
         --classes person car --conf-keep 0.5 --review-top 200
"""
import argparse
import json
from pathlib import Path

from visionpipe.datatools import pseudo_label, select_for_review
from visionpipe.detect import build_detector


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help=".pt (needs ultralytics) or .onnx")
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--classes", nargs="*", help="Class names to keep (default: all)")
    ap.add_argument("--conf-keep", type=float, default=0.5, help="Detections at or above this become labels")
    ap.add_argument("--conf-detect", type=float, default=0.1, help="Detector threshold (low, so borderline cases are seen)")
    ap.add_argument("--review-top", type=int, default=100)
    args = ap.parse_args()

    det = build_detector({"weights": args.weights, "conf": args.conf_detect, "classes": args.classes})
    scores = pseudo_label(det, args.images, args.out, conf_keep=args.conf_keep)
    queue = select_for_review(scores, args.review_top)
    out = Path(args.out)
    (out / "review_queue.txt").write_text("\n".join(queue) + "\n")
    (out / "uncertainty.json").write_text(json.dumps(scores, indent=2))
    print(f"labelled {len(scores)} images; {len(queue)} queued for human review -> {out / 'review_queue.txt'}")


if __name__ == "__main__":
    main()
