from __future__ import annotations

from typing import Any, Dict

from .base import Detector
from .motion import MotionDetector


def build_detector(cfg: Dict[str, Any]) -> Detector:
    """Create a detector from a config dict. `type` may be yolo | onnx | motion.

    If type is omitted it is inferred from the weights extension (.onnx -> onnx, else yolo).
    """
    cfg = dict(cfg)
    kind = cfg.pop("type", None)
    weights = cfg.get("weights")
    if kind is None:
        kind = "onnx" if weights and str(weights).endswith(".onnx") else "yolo"
    if kind == "motion":
        return MotionDetector(**{k: v for k, v in cfg.items() if k in ("min_area", "label", "history", "var_threshold")})
    if kind == "onnx":
        from .onnx_yolo import ONNXYoloDetector

        keys = ("conf", "iou", "imgsz", "classes", "providers", "intra_op_threads")
        return ONNXYoloDetector(cfg["weights"], **{k: v for k, v in cfg.items() if k in keys})
    if kind == "yolo":
        from .yolo import YOLODetector

        keys = ("weights", "conf", "iou", "imgsz", "classes", "device", "half")
        return YOLODetector(**{k: v for k, v in cfg.items() if k in keys})
    raise ValueError(f"Unknown detector type '{kind}'")


__all__ = ["Detector", "MotionDetector", "build_detector"]
