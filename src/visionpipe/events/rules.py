"""Rule-based complex event processing on top of tracks.

Every rule is a small state machine that consumes the tracks of one frame and may emit
Events. The false-positive filters that make the output trustworthy live here:

* persistence     - condition must hold for N consecutive frames before an event fires
* debouncing      - short exits from a zone do not reset dwell time (tracker jitter, occlusion)
* fire-once       - one event per track per episode, not one per frame
* hysteresis      - crowd alerts use different on/off conditions to avoid flapping
* min-gap         - a track jittering on a counting line is not counted repeatedly
* stale-state GC  - per-track state is dropped once a track has disappeared
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from ..geometry import (
    Point,
    distance,
    point_in_polygon,
    point_line_distance,
    segments_properly_intersect,
    side_of_line,
)
from ..types import Event, Track

STALE_FRAMES = 150  # forget per-track state after this many frames without seeing the track


@dataclass
class Context:
    camera_id: str
    frame_index: int
    timestamp: float
    tracks: List[Track]
    frame_width: int
    frame_height: int


def _prune(states: Dict[int, object], frame_index: int, stale: int = STALE_FRAMES) -> None:
    dead = [k for k, s in states.items() if frame_index - s.last_seen > stale]  # type: ignore[attr-defined]
    for k in dead:
        del states[k]


def _class_ok(track: Track, classes: Optional[Sequence[str]]) -> bool:
    return classes is None or track.cls_name in classes


def _scale_polygon(polygon: Sequence[Point], w: int, h: int) -> List[Point]:
    if not polygon:
        return list(polygon)
    if max(p[0] for p in polygon) <= 1.0 and max(p[1] for p in polygon) <= 1.0:
        return [(p[0] * w, p[1] * h) for p in polygon]
    return list(polygon)


def _scale_point(p: Point, w: int, h: int) -> Point:
    if p[0] <= 1.0 and p[1] <= 1.0:
        return (p[0] * w, p[1] * h)
    return p


class Rule(ABC):
    type: str = "rule"

    def __init__(self, name: str, classes: Optional[Sequence[str]] = None):
        self.name = name
        self.classes = list(classes) if classes else None

    def _event(self, ctx: Context, track: Optional[Track], **details) -> Event:
        return Event(
            type=self.type,
            camera_id=ctx.camera_id,
            frame_index=ctx.frame_index,
            timestamp=ctx.timestamp,
            track_id=track.track_id if track else None,
            cls_name=track.cls_name if track else None,
            zone=self.name,
            details=details,
        )

    @abstractmethod
    def update(self, ctx: Context) -> List[Event]: ...


# --------------------------------------------------------------------------- zone presence
@dataclass
class _Presence:
    last_seen: int = 0
    inside_run: int = 0
    outside_run: int = 0
    entered_ts: Optional[float] = None
    fired: bool = False


class _ZoneRule(Rule):
    """Shared debounced 'is this track in the polygon' logic."""

    def __init__(
        self,
        name: str,
        polygon: Sequence[Point],
        classes: Optional[Sequence[str]] = None,
        anchor: str = "bottom_center",
        exit_frames: int = 15,
    ):
        super().__init__(name, classes)
        self.polygon = [tuple(map(float, p)) for p in polygon]
        self.anchor = anchor
        self.exit_frames = exit_frames
        self._state: Dict[int, _Presence] = {}
        self._scaled = False

    def _observe(self, ctx: Context) -> List[Tuple[Track, _Presence]]:
        """Update presence state; return (track, state) for tracks currently inside."""
        if not self._scaled:
            self.polygon = _scale_polygon(self.polygon, ctx.frame_width, ctx.frame_height)
            self._scaled = True
        inside_now: List[Tuple[Track, _Presence]] = []
        for tr in ctx.tracks:
            if not _class_ok(tr, self.classes):
                continue
            st = self._state.setdefault(tr.track_id, _Presence())
            st.last_seen = ctx.frame_index
            if point_in_polygon(tr.anchor(self.anchor), self.polygon):
                st.inside_run += 1
                st.outside_run = 0
                if st.entered_ts is None:
                    st.entered_ts = ctx.timestamp
                inside_now.append((tr, st))
            else:
                st.outside_run += 1
                st.inside_run = 0
                if st.outside_run >= self.exit_frames:  # really left: allow a new episode
                    st.entered_ts = None
                    st.fired = False
        _prune(self._state, ctx.frame_index)
        return inside_now


class ZoneIntrusion(_ZoneRule):
    type = "zone_intrusion"

    def __init__(self, *args, min_frames: int = 5, **kwargs):
        super().__init__(*args, **kwargs)
        self.min_frames = min_frames

    def update(self, ctx: Context) -> List[Event]:
        events = []
        for tr, st in self._observe(ctx):
            if st.inside_run >= self.min_frames and not st.fired:
                st.fired = True
                events.append(self._event(ctx, tr, confidence=round(tr.score, 3)))
        return events


class Loitering(_ZoneRule):
    type = "loitering"

    def __init__(self, *args, dwell_seconds: float = 10.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.dwell_seconds = dwell_seconds

    def update(self, ctx: Context) -> List[Event]:
        events = []
        for tr, st in self._observe(ctx):
            dwell = ctx.timestamp - (st.entered_ts if st.entered_ts is not None else ctx.timestamp)
            if dwell >= self.dwell_seconds and not st.fired:
                st.fired = True
                events.append(self._event(ctx, tr, dwell_seconds=round(dwell, 2)))
        return events


# --------------------------------------------------------------------------- line crossing
@dataclass
class _LineState:
    last_seen: int = 0
    last_point: Optional[Point] = None
    last_side: int = 0
    last_cross_frame: int = -10**9


class LineCrossing(Rule):
    """Counts tracks crossing a segment.

    Direction: 'forward' means moving from the negative to the positive side of the
    line p1->p2 (sign of the 2-D cross product in image coordinates), 'backward' the reverse.
    """

    type = "line_crossing"

    def __init__(
        self,
        name: str,
        p1: Point,
        p2: Point,
        classes: Optional[Sequence[str]] = None,
        anchor: str = "center",
        direction: str = "any",
        min_gap_frames: int = 15,
        hysteresis_px: float = 0.0,
    ):
        super().__init__(name, classes)
        self.p1, self.p2 = tuple(map(float, p1)), tuple(map(float, p2))
        self.anchor = anchor
        self.direction = direction
        self.min_gap_frames = min_gap_frames
        self.hysteresis_px = hysteresis_px
        self.counts = {"forward": 0, "backward": 0}
        self._state: Dict[int, _LineState] = {}
        self._scaled = False

    def update(self, ctx: Context) -> List[Event]:
        if not self._scaled:
            self.p1 = _scale_point(self.p1, ctx.frame_width, ctx.frame_height)
            self.p2 = _scale_point(self.p2, ctx.frame_width, ctx.frame_height)
            self.hysteresis_px = self.hysteresis_px  # Not scaled, assuming absolute pixels
            self._scaled = True

        events = []
        for tr in ctx.tracks:
            if not _class_ok(tr, self.classes):
                continue
            st = self._state.setdefault(tr.track_id, _LineState())
            st.last_seen = ctx.frame_index
            pt = tr.anchor(self.anchor)
            side = side_of_line(pt, self.p1, self.p2)
            if side == 0:  # exactly on the line: wait for the next frame
                continue
            if self.hysteresis_px > 0 and point_line_distance(pt, self.p1, self.p2) <= self.hysteresis_px:
                continue

            if (
                st.last_point is not None
                and st.last_side != 0
                and side != st.last_side
                and segments_properly_intersect(st.last_point, pt, self.p1, self.p2)
                and ctx.frame_index - st.last_cross_frame >= self.min_gap_frames
            ):
                direction = "forward" if side > 0 else "backward"
                if self.direction in ("any", direction):
                    self.counts[direction] += 1
                    st.last_cross_frame = ctx.frame_index
                    events.append(self._event(ctx, tr, direction=direction, counts=dict(self.counts)))
            
            st.last_point, st.last_side = pt, side
        _prune(self._state, ctx.frame_index)
        return events


# --------------------------------------------------------------------------- crowd
class CrowdDetection(Rule):
    type = "crowd_detected"

    def __init__(
        self,
        name: str,
        polygon: Sequence[Point],
        classes: Optional[Sequence[str]] = ("person",),
        threshold: int = 5,
        min_frames: int = 10,
        clear_frames: int = 30,
        anchor: str = "bottom_center",
    ):
        super().__init__(name, classes)
        self.polygon = [tuple(map(float, p)) for p in polygon]
        self.threshold = threshold
        self.min_frames = min_frames
        self.clear_frames = clear_frames
        self.anchor = anchor
        self._high = 0
        self._low = 0
        self._active = False
        self._scaled = False

    def update(self, ctx: Context) -> List[Event]:
        if not self._scaled:
            self.polygon = _scale_polygon(self.polygon, ctx.frame_width, ctx.frame_height)
            self._scaled = True

        count = sum(
            1
            for t in ctx.tracks
            if _class_ok(t, self.classes) and point_in_polygon(t.anchor(self.anchor), self.polygon)
        )
        events: List[Event] = []
        if count >= self.threshold:
            self._high, self._low = self._high + 1, 0
            if not self._active and self._high >= self.min_frames:
                self._active = True
                events.append(self._event(ctx, None, count=count, threshold=self.threshold))
        else:
            self._low, self._high = self._low + 1, 0
            if self._active and self._low >= self.clear_frames:
                self._active = False
                events.append(
                    Event(
                        type="crowd_cleared",
                        camera_id=ctx.camera_id,
                        frame_index=ctx.frame_index,
                        timestamp=ctx.timestamp,
                        zone=self.name,
                        details={"count": count},
                    )
                )
        return events


# --------------------------------------------------------------------------- abandoned object
@dataclass
class _ObjState:
    last_seen: int = 0
    ref_point: Optional[Point] = None
    stationary_since: float = 0.0
    unattended_since: Optional[float] = None
    fired: bool = False


class AbandonedObject(Rule):
    """A bag-like object that stays put while no person is nearby for `stationary_seconds`."""

    type = "abandoned_object"

    def __init__(
        self,
        name: str,
        object_classes: Sequence[str] = ("backpack", "handbag", "suitcase"),
        person_classes: Sequence[str] = ("person",),
        stationary_seconds: float = 30.0,
        move_tol_px: float = 15.0,
        owner_radius_px: float = 150.0,
    ):
        super().__init__(name, object_classes)
        self.person_classes = list(person_classes)
        self.stationary_seconds = stationary_seconds
        self.move_tol_px = move_tol_px
        self.owner_radius_px = owner_radius_px
        self._state: Dict[int, _ObjState] = {}

    def update(self, ctx: Context) -> List[Event]:
        people = [t.center for t in ctx.tracks if t.cls_name in self.person_classes]
        events = []
        for tr in ctx.tracks:
            if not _class_ok(tr, self.classes):
                continue
            st = self._state.setdefault(tr.track_id, _ObjState(stationary_since=ctx.timestamp))
            st.last_seen = ctx.frame_index
            c = tr.center
            if st.ref_point is None or distance(c, st.ref_point) > self.move_tol_px:
                st.ref_point, st.stationary_since = c, ctx.timestamp
                st.unattended_since = None
            owner_near = any(distance(c, p) <= self.owner_radius_px for p in people)
            if owner_near:
                st.unattended_since = None
            elif st.unattended_since is None:
                st.unattended_since = ctx.timestamp
            if st.unattended_since is not None and not st.fired:
                idle = ctx.timestamp - max(st.stationary_since, st.unattended_since)
                if idle >= self.stationary_seconds:
                    st.fired = True
                    events.append(self._event(ctx, tr, unattended_seconds=round(idle, 1)))
        _prune(self._state, ctx.frame_index)
        return events


RULE_REGISTRY = {
    "zone_intrusion": ZoneIntrusion,
    "loitering": Loitering,
    "line_crossing": LineCrossing,
    "crowd": CrowdDetection,
    "abandoned_object": AbandonedObject,
}


def build_rules(specs: Sequence[dict]) -> List[Rule]:
    rules = []
    for spec in specs:
        spec = dict(spec)
        rtype = spec.pop("type")
        if rtype not in RULE_REGISTRY:
            raise ValueError(f"Unknown rule type '{rtype}'. Options: {sorted(RULE_REGISTRY)}")
        rules.append(RULE_REGISTRY[rtype](**spec))
    return rules
