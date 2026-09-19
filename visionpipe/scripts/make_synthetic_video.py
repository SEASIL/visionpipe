#!/usr/bin/env python
"""Generate a deterministic synthetic surveillance clip + ground-truth annotations.

Scene (640x480 @ 25 fps, ~22 s), textured rectangles stand in for people:
  #1 walks top -> bottom at x=150 and crosses the counting line (y=250)
  #2 walks into the restricted zone and lingers there (should trigger intrusion + loitering)
  #3 runs left -> right through the zone in ~2 s (intrusion, but NOT loitering)

Used by the zero-download demo and the CI smoke test. Real footage is still needed for real claims.
Usage: python scripts/make_synthetic_video.py --out data/synthetic.mp4
"""
import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np

W, H, FPS, N_FRAMES = 640, 480, 25, 560
ACTOR_W, ACTOR_H = 40, 90


def actor_path(actor_id: int, f: int):
    """Return (cx, cy) or None if the actor is not in the scene at frame f."""
    if actor_id == 1:
        if 25 <= f <= 275:
            return 150.0, 60.0 + (f - 25) * 1.6
    elif actor_id == 2:
        if 60 <= f < 160:
            return 700.0 - (f - 60) * 2.2, 350.0
        if 160 <= f < 420:
            return 480.0 + 40.0 * math.sin((f - 160) * 2 * math.pi / 100), 350.0
        if 420 <= f < 520:
            return 480.0 + (f - 420) * 2.2, 350.0
    elif actor_id == 3:
        if 200 <= f <= 330:
            return -30.0 + (f - 200) * 5.6, 230.0
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/synthetic.mp4")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    bg = np.tile(np.linspace(90, 140, W, dtype=np.float32), (H, 1))
    bg = np.stack([bg, bg * 0.95, bg * 0.9], -1) + rng.normal(0, 3, (H, W, 3))
    textures = {i: rng.integers(30, 255, (ACTOR_H, ACTOR_W, 3), dtype=np.uint8) for i in (1, 2, 3)}

    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    gt = []
    for f in range(N_FRAMES):
        img = np.clip(bg + rng.normal(0, 2, (H, W, 3)), 0, 255).astype(np.uint8)
        objs = []
        for aid in (1, 2, 3):
            pos = actor_path(aid, f)
            if pos is None:
                continue
            cx, cy = pos
            x1, y1 = int(round(cx - ACTOR_W / 2)), int(round(cy - ACTOR_H / 2))
            x2, y2 = x1 + ACTOR_W, y1 + ACTOR_H
            sx1, sy1, sx2, sy2 = max(x1, 0), max(y1, 0), min(x2, W), min(y2, H)
            if sx2 > sx1 and sy2 > sy1:
                img[sy1:sy2, sx1:sx2] = textures[aid][sy1 - y1 : sy2 - y1, sx1 - x1 : sx2 - x1]
                objs.append({"id": aid, "box": [sx1, sy1, sx2, sy2]})
        gt.append({"frame": f + 1, "objects": objs})
        writer.write(img)
    writer.release()

    gt_path = Path(args.out).with_suffix(".gt.json")
    gt_path.write_text(json.dumps(gt))
    # MOTChallenge-format ground truth for scripts/eval_tracking.py
    mot_path = Path(args.out).with_suffix(".gt.txt")
    with open(mot_path, "w") as fh:
        for rec in gt:
            for o in rec["objects"]:
                x1, y1, x2, y2 = o["box"]
                fh.write(f"{rec['frame']},{o['id']},{x1},{y1},{x2 - x1},{y2 - y1},1,1,1\n")
    print(f"wrote {args.out} ({N_FRAMES} frames), {gt_path.name}, {mot_path.name}")


if __name__ == "__main__":
    main()
