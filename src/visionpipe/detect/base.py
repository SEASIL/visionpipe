from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Sequence

import numpy as np

from ..types import Detection


class Detector(ABC):
    names: Dict[int, str]

    @abstractmethod
    def detect(self, image: np.ndarray) -> List[Detection]:
        """Run inference on one BGR frame and return detections in pixel coordinates."""

    def warmup(self, shape=(480, 640, 3), n: int = 3) -> None:
        dummy = np.zeros(shape, np.uint8)
        for _ in range(n):
            self.detect(dummy)


def resolve_class_ids(names: Dict[int, str], wanted: Optional[Sequence]) -> Optional[List[int]]:
    """Accepts class names or ids; returns ids (or None = keep everything)."""
    if not wanted:
        return None
    inv = {v: k for k, v in names.items()}
    ids = []
    for w in wanted:
        if isinstance(w, int):
            ids.append(w)
        elif w in inv:
            ids.append(inv[w])
        else:
            raise ValueError(f"Unknown class '{w}'. Model classes: {sorted(inv)[:20]}...")
    return ids
