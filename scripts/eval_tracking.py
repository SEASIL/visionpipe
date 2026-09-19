#!/usr/bin/env python
"""Score tracker output against MOTChallenge-format ground truth (MOTA, IDF1, ID switches...).

Ground truth: MOT format `frame,id,x,y,w,h,conf,class,vis` (1-based frames).
Tracker output: written by the pipeline via `output.mot_txt`.

Usage:
  python scripts/eval_tracking.py --gt data/synthetic.gt.txt --pred outputs/cam0/tracks.txt
  # MOT17 sequence: --gt MOT17-04-FRCNN/gt/gt.txt (filter to pedestrians with --gt-class 1)
"""
import argparse
import json

import motmetrics as mm
import numpy as np

from visionpipe.geometry import iou_matrix


def load_mot(path: str, cls: int | None = None):
    rows = np.loadtxt(path, delimiter=",", ndmin=2)
    if cls is not None and rows.shape[1] > 7:
        rows = rows[rows[:, 7] == cls]
    frames = {}
    for r in rows:
        frames.setdefault(int(r[0]), []).append((int(r[1]), r[2:6]))
    return frames


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--gt-class", type=int, default=None, help="Keep only this class id from GT (MOT17: 1 = pedestrian)")
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--out", help="Write metrics JSON here")
    ap.add_argument("--min-mota", type=float, help="Exit 1 if MOTA is below this (regression gate)")
    ap.add_argument("--min-idf1", type=float, help="Exit 1 if IDF1 is below this")
    args = ap.parse_args()

    gt, pred = load_mot(args.gt, args.gt_class), load_mot(args.pred)
    acc = mm.MOTAccumulator(auto_id=True)
    for f in range(1, max(list(gt) + list(pred)) + 1):
        g, p = gt.get(f, []), pred.get(f, [])
        # own IoU (motmetrics' helper breaks on NumPy 2): distance = 1 - IoU, NaN = "cannot match"
        gb = np.array([[x, y, x + w, y + h] for _, (x, y, w, h) in g]).reshape(-1, 4)
        pb = np.array([[x, y, x + w, y + h] for _, (x, y, w, h) in p]).reshape(-1, 4)
        iou = iou_matrix(gb, pb)
        dist = np.where(iou >= args.iou, 1.0 - iou, np.nan)
        acc.update([i for i, _ in g], [i for i, _ in p], dist)

    mh = mm.metrics.create()
    names = ["mota", "motp", "idf1", "precision", "recall", "num_switches", "num_fragmentations", "mostly_tracked", "mostly_lost"]
    summ = mh.compute(acc, metrics=names, name="seq")
    print(mm.io.render_summary(summ, formatters=mh.formatters, namemap=mm.io.motchallenge_metric_names))
    if args.out:
        with open(args.out, "w") as f:
            json.dump({k: float(v) for k, v in summ.iloc[0].items()}, f, indent=2)
    row = summ.iloc[0]
    if (args.min_mota is not None and row["mota"] < args.min_mota) or (args.min_idf1 is not None and row["idf1"] < args.min_idf1):
        print("FAIL: tracking metrics below required thresholds")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
