"""Core data types shared by every stage of the pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np


@dataclass
class Frame:
    camera_id: str
    index: int
    timestamp: float  # seconds. File sources: index / fps. Live sources: wall clock.
    image: np.ndarray  # BGR uint8 (H, W, 3)


@dataclass
class Detection:
    box: np.ndarray  # [x1, y1, x2, y2] in pixels
    score: float
    cls_id: int
    cls_name: str


@dataclass
class Track:
    track_id: int
    box: np.ndarray  # [x1, y1, x2, y2]
    score: float
    cls_id: int
    cls_name: str
    hits: int = 1
    age: int = 1
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def center(self) -> Tuple[float, float]:
        x1, y1, x2, y2 = self.box
        return float((x1 + x2) / 2), float((y1 + y2) / 2)

    @property
    def foot(self) -> Tuple[float, float]:
        """Bottom-center of the box: the point that touches the ground plane."""
        x1, _, x2, y2 = self.box
        return float((x1 + x2) / 2), float(y2)

    def anchor(self, kind: str = "bottom_center") -> Tuple[float, float]:
        return self.foot if kind == "bottom_center" else self.center


@dataclass
class Event:
    type: str
    camera_id: str
    frame_index: int
    timestamp: float
    track_id: Optional[int] = None
    cls_name: Optional[str] = None
    zone: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type,
            "camera_id": self.camera_id,
            "frame_index": self.frame_index,
            "timestamp": round(float(self.timestamp), 3),
            "track_id": self.track_id,
            "class": self.cls_name,
            "zone": self.zone,
            "details": self.details,
        }
