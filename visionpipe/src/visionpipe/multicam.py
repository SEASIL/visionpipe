"""Run several camera pipelines concurrently, with optional cross-camera global IDs.

Threading model: one thread per camera (OpenCV decode, NumPy and ONNX Runtime release the GIL for
their heavy work). A stateless neural detector is shared behind a lock so the model is loaded
once; stateful detectors (background subtraction) are created per camera.
Scaling beyond a handful of streams would call for batched inference (e.g. DeepStream nvinfer
or a batching inference server) - see docs/ARCHITECTURE.md.
"""
from __future__ import annotations

import threading
from typing import Any, Dict, List

from .config import build_pipeline, deep_merge
from .detect import build_detector
from .detect.base import Detector
from .pipeline import RunStats
from .reid import GlobalIdRegistry


class LockedDetector(Detector):
    def __init__(self, inner: Detector):
        self.inner, self._lock = inner, threading.Lock()
        self.names = inner.names

    def detect(self, image):
        with self._lock:
            return self.inner.detect(image)


def run_multi(cfg: Dict[str, Any], max_frames: int | None = None, reid: bool = True) -> List[RunStats]:
    cameras = cfg["cameras"]
    base = {k: v for k, v in cfg.items() if k != "cameras"}
    shared = None
    if cfg.get("detector", {}).get("type") != "motion":
        shared = LockedDetector(build_detector(cfg["detector"]))
    registry = GlobalIdRegistry() if reid else None

    results: List[RunStats] = []
    errors: List[BaseException] = []
    lock = threading.Lock()

    def worker(cam_cfg: Dict[str, Any]) -> None:
        try:
            merged = deep_merge(base, {"camera_id": cam_cfg["id"], "source": {"uri": cam_cfg["uri"]}})
            if "rules" in cam_cfg:  # per-camera rules replace the global ones
                merged["rules"] = cam_cfg["rules"]
            stats = build_pipeline(merged, detector=shared, reid=registry).run(max_frames=max_frames)
            with lock:
                results.append(stats)
        except BaseException as exc:  # surface thread failures to the caller
            with lock:
                errors.append(exc)

    threads = [threading.Thread(target=worker, args=(c,), name=f"cam-{c['id']}") for c in cameras]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if errors:
        raise errors[0]
    return sorted(results, key=lambda s: s.camera_id)
