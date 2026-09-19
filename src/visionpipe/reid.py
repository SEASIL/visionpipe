"""Baseline cross-camera re-identification using colour histograms.

HONEST SCOPE: an HSV torso/legs histogram is a weak appearance descriptor. It works when people
wear distinct colours and camera colour balance is similar, and it fails on uniforms or very
different lighting. It exists to demonstrate the multi-camera identity plumbing (global IDs,
per-camera track -> global identity assignment, thread safety). To get production-grade
re-ID, swap `HSVEmbedder` for a learned embedding (e.g. OSNet) - `GlobalIdRegistry` only
needs a function `(image, box) -> 1-D vector` and a matching distance.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .types import Frame, Track


class HSVEmbedder:
    """Concatenated H-S histograms of the upper (torso) and lower (legs) half of the box."""

    def __init__(self, bins: Tuple[int, int] = (8, 8), min_size: int = 16):
        self.bins = bins
        self.min_size = min_size

    def __call__(self, image: np.ndarray, box: np.ndarray) -> Optional[np.ndarray]:
        H, W = image.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in box]
        x1, y1, x2, y2 = max(x1, 0), max(y1, 0), min(x2, W), min(y2, H)
        if x2 - x1 < self.min_size or y2 - y1 < self.min_size:
            return None
        crop = cv2.cvtColor(image[y1:y2, x1:x2], cv2.COLOR_BGR2HSV)
        mid = crop.shape[0] // 2
        parts = []
        for half in (crop[:mid], crop[mid:]):
            hist = cv2.calcHist([half], [0, 1], None, list(self.bins), [0, 180, 0, 256]).flatten()
            parts.append(hist)
        v = np.concatenate(parts).astype(np.float64)
        s = v.sum()
        return v / s if s > 0 else None


def bhattacharyya(p: np.ndarray, q: np.ndarray) -> float:
    """Distance in [0, 1] between two L1-normalised histograms (0 = identical)."""
    bc = float(np.sum(np.sqrt(p * q)))
    return float(np.sqrt(max(0.0, 1.0 - bc)))


@dataclass
class _Sample:
    total: np.ndarray = field(default_factory=lambda: np.zeros(0))
    n: int = 0

    def add(self, v: np.ndarray) -> None:
        self.total = v.copy() if self.n == 0 else self.total + v
        self.n += 1

    @property
    def mean(self) -> np.ndarray:
        m = self.total / max(self.n, 1)
        return m / m.sum()


@dataclass
class _Identity:
    gid: int
    embed: _Sample
    members: Dict[str, int] = field(default_factory=dict)  # camera_id -> track_id


class GlobalIdRegistry:
    """Thread-safe mapping (camera_id, track_id) -> global identity."""

    def __init__(
        self,
        embedder: Optional[HSVEmbedder] = None,
        classes: Sequence[str] = ("person",),
        threshold: float = 0.35,
        min_samples: int = 5,
        sample_every: int = 3,
        min_age: int = 5,
    ):
        self.embedder = embedder or HSVEmbedder()
        self.classes = set(classes)
        self.threshold = threshold
        self.min_samples = min_samples
        self.sample_every = sample_every
        self.min_age = min_age
        self._lock = threading.Lock()
        self._samples: Dict[Tuple[str, int], _Sample] = {}
        self._assigned: Dict[Tuple[str, int], int] = {}
        self._identities: List[_Identity] = []

    def update(self, frame: Frame, tracks: Sequence[Track]) -> None:
        if frame.index % self.sample_every != 0:
            return
        for t in tracks:
            if t.cls_name not in self.classes or t.age < self.min_age:
                continue
            key = (frame.camera_id, t.track_id)
            if key in self._assigned:
                continue
            emb = self.embedder(frame.image, t.box)
            if emb is None:
                continue
            with self._lock:
                s = self._samples.setdefault(key, _Sample())
                s.add(emb)
                if s.n >= self.min_samples:
                    self._assign(key, s)

    def _assign(self, key: Tuple[str, int], sample: _Sample) -> None:
        cam, tid = key
        emb = sample.mean
        best, best_d = None, self.threshold
        for ident in self._identities:
            if cam in ident.members:  # one identity cannot be two people in the same camera
                continue
            d = bhattacharyya(emb, ident.embed.mean)
            if d < best_d:
                best, best_d = ident, d
        if best is None:
            best = _Identity(gid=len(self._identities) + 1, embed=_Sample())
            self._identities.append(best)
        best.embed.add(emb)
        best.members[cam] = tid
        self._assigned[key] = best.gid

    def global_id(self, camera_id: str, track_id: int) -> Optional[int]:
        with self._lock:
            return self._assigned.get((camera_id, track_id))
