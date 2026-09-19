"""Classical background-subtraction detector (OpenCV MOG2).

Needs no model weights, so it is used for the zero-download demo and the CI smoke test.
It also doubles as a baseline to compare deep detectors against on static-camera footage.
"""
from __future__ import annotations

from typing import List

import cv2
import numpy as np

from ..types import Detection
from .base import Detector


class MotionDetector(Detector):
    def __init__(self, min_area: int = 400, label: str = "object", history: int = 200, var_threshold: float = 32.0):
        self.min_area = min_area
        self.label = label
        self.names = {0: label}
        self._bg = cv2.createBackgroundSubtractorMOG2(
            history=history, varThreshold=var_threshold, detectShadows=True
        )
        self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    def detect(self, image: np.ndarray) -> List[Detection]:
        mask = self._bg.apply(image)
        mask = (mask == 255).astype(np.uint8) * 255  # drop MOG2's gray "shadow" pixels
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._kernel, iterations=3)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        dets = []
        for c in contours:
            if cv2.contourArea(c) < self.min_area:
                continue
            x, y, w, h = cv2.boundingRect(c)
            dets.append(Detection(np.array([x, y, x + w, y + h], dtype=float), 0.9, 0, self.label))
        return dets
