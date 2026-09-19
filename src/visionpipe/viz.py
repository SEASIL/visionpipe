from __future__ import annotations

from typing import Iterable, List, Sequence

import cv2
import numpy as np

from .events.rules import Rule
from .types import Event, Track


def _color(track_id: int):
    rng = np.random.default_rng(track_id * 7919)
    return tuple(int(c) for c in rng.integers(60, 255, 3))


def draw_rules(img: np.ndarray, rules: Iterable[Rule]) -> None:
    for r in rules:
        poly = getattr(r, "polygon", None)
        if poly:
            pts = np.array(poly, np.int32).reshape(-1, 1, 2)
            overlay = img.copy()
            cv2.fillPoly(overlay, [pts], (0, 0, 255))
            cv2.addWeighted(overlay, 0.15, img, 0.85, 0, img)
            cv2.polylines(img, [pts], True, (0, 0, 255), 2)
            cv2.putText(img, r.name, tuple(pts[0, 0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        if hasattr(r, "p1") and hasattr(r, "p2"):
            p1, p2 = tuple(map(int, r.p1)), tuple(map(int, r.p2))
            cv2.line(img, p1, p2, (0, 255, 255), 2)
            counts = getattr(r, "counts", {})
            label = f"{r.name} F:{counts.get('forward', 0)} B:{counts.get('backward', 0)}"
            cv2.putText(img, label, (p1[0], max(p1[1] - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)


def draw_tracks(img: np.ndarray, tracks: Sequence[Track]) -> None:
    for t in tracks:
        x1, y1, x2, y2 = map(int, t.box)
        col = _color(t.track_id)
        cv2.rectangle(img, (x1, y1), (x2, y2), col, 2)
        label = f"#{t.track_id} {t.cls_name} {t.score:.2f}"
        cv2.putText(img, label, (x1, max(y1 - 5, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)


def draw_event_banner(img: np.ndarray, recent: List[Event], fps: float | None = None) -> None:
    y = 22
    if fps is not None:
        cv2.putText(img, f"{fps:.1f} FPS", (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y += 22
    for e in recent[-4:]:
        txt = f"{e.type.upper()} {e.zone or ''}" + (f" #{e.track_id}" if e.track_id is not None else "")
        cv2.putText(img, txt, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        y += 22
