from __future__ import annotations

from typing import List, Sequence

from ..types import Event, Frame, Track
from .rules import Context, Rule


class EventEngine:
    """Runs all rules on each frame's tracks and collects the emitted events."""

    def __init__(self, rules: Sequence[Rule]):
        self.rules = list(rules)

    def update(self, frame: Frame, tracks: List[Track]) -> List[Event]:
        h, w = frame.image.shape[:2]
        ctx = Context(frame.camera_id, frame.index, frame.timestamp, tracks, frame_width=w, frame_height=h)
        events: List[Event] = []
        for rule in self.rules:
            events.extend(rule.update(ctx))
        return events
