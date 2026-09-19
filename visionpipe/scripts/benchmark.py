#!/usr/bin/env python
"""Benchmark detector backends on the same footage: latency, FPS, memory and (optionally) mAP.

Latency is END-TO-END per frame for the detector stage (pre-processing + inference + post-processing),
measured after warm-up, one frame at a time (batch size 1 - the real-time streaming case).

Usage:
  python scripts/benchmark.py --models models/best.pt models/best.onnx models/best_fp16.onnx \
        --source data/site_clip.mp4 --frames 300 --data data/dataset/data.yaml --out docs/BENCHMARKS.md
Run it on the hardware you claim numbers for, and record that hardware in the output table.
"""
import argparse
import json
import platform
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import psutil

from visionpipe.detect import build_detector
from visionpipe.io import VideoSource


def load_frames(source: str, n: int):
    frames = []
    for f in VideoSource(source).frames():
        frames.append(f.image)
        if len(frames) >= n:
            break
    if not frames:
        raise SystemExit(f"no frames read from {source}")
    return frames


def gpu_mem_mb():
    if not shutil.which("nvidia-smi"):
        return None
    try:
        out = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True)
        return float(out.strip().splitlines()[0])
    except Exception:
        return None


def map_for(model_path: str, data: str, imgsz: int):
    try:
        from ultralytics import YOLO

        r = YOLO(model_path).val(data=data, imgsz=imgsz, verbose=False)
        return float(r.box.map50), float(r.box.map)
    except Exception as exc:  # keep benchmarking even if validation is unavailable for a format
        print(f"  (mAP skipped for {model_path}: {exc})")
        return None, None


def bench(model_path: str, frames, warmup: int, imgsz: int, conf: float):
    proc = psutil.Process()
    rss_before = proc.memory_info().rss
    det = build_detector({"weights": model_path, "imgsz": imgsz, "conf": conf})
    for f in frames[: max(warmup, 1)]:
        det.detect(f)
    lat = []
    for f in frames:
        t0 = time.perf_counter()
        det.detect(f)
        lat.append((time.perf_counter() - t0) * 1000)
    lat = np.array(lat)
    return {
        "model": Path(model_path).name,
        "backend": ",".join(getattr(det, "providers", ["ultralytics"])),
        "mean_ms": float(lat.mean()),
        "p50_ms": float(np.percentile(lat, 50)),
        "p95_ms": float(np.percentile(lat, 95)),
        "fps": float(1000.0 / lat.mean()),
        "rss_mb": (proc.memory_info().rss - rss_before) / 1e6,
        "gpu_mem_mb": gpu_mem_mb(),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--frames", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--data", help="dataset yaml -> also report mAP (needs ultralytics)")
    ap.add_argument("--out", help="write a Markdown table here (and .json next to it)")
    args = ap.parse_args()

    frames = load_frames(args.source, args.frames)
    rows = []
    for m in args.models:
        print(f"benchmarking {m} ...")
        r = bench(m, frames, args.warmup, args.imgsz, args.conf)
        r["map50"], r["map50_95"] = map_for(m, args.data, args.imgsz) if args.data else (None, None)
        rows.append(r)

    fmt = lambda v, p=1: "-" if v is None else f"{v:.{p}f}"  # noqa: E731
    hw = f"{platform.processor() or platform.machine()}, {psutil.cpu_count()} logical CPUs, {platform.platform()}"
    lines = [
        f"Hardware: {hw}  |  frames={len(frames)}  imgsz={args.imgsz}  batch=1",
        "",
        "| Model | Backend | mean ms | p50 ms | p95 ms | FPS | +RSS MB | GPU MB | mAP50 | mAP50-95 |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['model']} | {r['backend']} | {fmt(r['mean_ms'])} | {fmt(r['p50_ms'])} | {fmt(r['p95_ms'])} | {fmt(r['fps'])} | "
            f"{fmt(r['rss_mb'], 0)} | {fmt(r['gpu_mem_mb'], 0)} | {fmt(r['map50'], 3)} | {fmt(r['map50_95'], 3)} |"
        )
    table = "\n".join(lines)
    print("\n" + table)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text("# Benchmarks\n\n" + table + "\n")
        Path(args.out).with_suffix(".json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
