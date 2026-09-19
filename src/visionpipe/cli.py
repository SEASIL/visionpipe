from __future__ import annotations

import argparse
import json
import logging
import sys

from .config import build_pipeline, load_config


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="visionpipe", description="Run the video analytics pipeline.")
    ap.add_argument("--config", "-c", help="YAML config (see configs/)")
    ap.add_argument("--source", "-s", help="Video file, webcam index or rtsp:// URL (overrides config)")
    ap.add_argument("--camera-id", help="Camera identifier written into every event")
    ap.add_argument("--max-frames", type=int, help="Stop after N frames")
    ap.add_argument("--events-out", help="Write events as JSON lines to this path")
    ap.add_argument("--video-out", help="Write annotated video to this path (.mp4)")
    ap.add_argument("--show", action="store_true", help="Display a live window (press q to quit)")
    ap.add_argument("--stats-out", help="Write run statistics (FPS, per-stage latency) to this JSON file")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(asctime)s %(levelname)s %(message)s")

    overrides: dict = {}
    if args.source:
        overrides["source"] = {"uri": int(args.source) if args.source.isdigit() else args.source}
    if args.camera_id:
        overrides["camera_id"] = args.camera_id
    out = {k: v for k, v in (("events_jsonl", args.events_out), ("video", args.video_out)) if v}
    if out:
        overrides["output"] = out

    cfg = load_config(args.config, overrides)
    if cfg.get("cameras"):  # multi-camera mode
        from .multicam import run_multi

        all_stats = run_multi(cfg, max_frames=args.max_frames)
        payload = json.dumps([s.to_dict() for s in all_stats], indent=2)
    else:
        stats = build_pipeline(cfg, show=args.show).run(max_frames=args.max_frames)
        payload = json.dumps(stats.to_dict(), indent=2)
    print(payload)
    if args.stats_out:
        with open(args.stats_out, "w") as f:
            f.write(payload)
    return 0


if __name__ == "__main__":
    sys.exit(main())
