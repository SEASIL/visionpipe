"""Ultralytics YOLO wrapper. Works with .pt, .onnx, .engine (TensorRT) and OpenVINO exports."""
from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

from ..types import Detection
from .base import Detector, resolve_class_ids


class YOLODetector(Detector):
    def __init__(
        self,
        weights: str = "yolov8n.pt",
        conf: float = 0.25,
        iou: float = 0.5,
        imgsz: int = 640,
        classes: Optional[Sequence] = None,
        device: Optional[str] = None,
        half: bool = False,
    ):
        from ultralytics import YOLO  # lazy import: heavy, and optional for ONNX/motion deployments

        self.model = YOLO(weights)
        self.names = dict(self.model.names)
        self.conf, self.iou, self.imgsz = conf, iou, imgsz
        self.class_ids = resolve_class_ids(self.names, classes)
        self.device, self.half = device, half

    def detect(self, image: np.ndarray) -> List[Detection]:
        res = self.model.predict(
            image,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            classes=self.class_ids,
            device=self.device,
            half=self.half,
            verbose=False,
        )[0]
        if res.boxes is None or len(res.boxes) == 0:
            return []
        xyxy = res.boxes.xyxy.cpu().numpy()
        conf = res.boxes.conf.cpu().numpy()
        cls = res.boxes.cls.cpu().numpy().astype(int)
        return [Detection(b.astype(float), float(s), int(c), self.names[int(c)]) for b, s, c in zip(xyxy, conf, cls)]
