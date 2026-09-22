"""End-to-end pipeline: ingest -> detect -> track -> events -> sinks."""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

import cv2

from .detect.base import Detector
from .events.engine import EventEngine
from .events.sinks import EventSink
from .io.source import VideoSource
from .timing import StageTimer
from .track.bytetrack import ByteTracker
from .types import Event, Frame, Track
from .viz import draw_event_banner, draw_rules, draw_tracks

log = logging.getLogger(__name__)


@dataclass
class RunStats:
    camera_id: str
    frames: int = 0
    wall_seconds: float = 0.0
    dropped_frames: int = 0
    events: int = 0
    events_by_type: Dict[str, int] = field(default_factory=dict)
    stages: Dict[str, Dict[str, float]] = field(default_factory=dict)

    @property
    def fps(self) -> float:
        return self.frames / self.wall_seconds if self.wall_seconds > 0 else 0.0

    def to_dict(self) -> dict:
        return {
            "camera_id": self.camera_id,
            "frames": self.frames,
            "wall_seconds": round(self.wall_seconds, 3),
            "fps": round(self.fps, 2),
            "dropped_frames": self.dropped_frames,
            "events": self.events,
            "events_by_type": self.events_by_type,
            "stage_latency": self.stages,
        }


class Outputs:
    """Optional file sinks. Each one is enabled by giving it a path."""

    def __init__(
        self,
        events_jsonl: Optional[str] = None,
        metadata_jsonl: Optional[str] = None,
        mot_txt: Optional[str] = None,
        video: Optional[str] = None,
    ):
        for p in (events_jsonl, metadata_jsonl, mot_txt, video):
            if p:
                Path(p).parent.mkdir(parents=True, exist_ok=True)
        self.events_f = open(events_jsonl, "a") if events_jsonl else None
        self.meta_f = open(metadata_jsonl, "w") if metadata_jsonl else None
        self.mot_f = open(mot_txt, "w") if mot_txt else None
        self.video_path = video
        self.writer: Optional[cv2.VideoWriter] = None

    def write_frame_outputs(self, frame: Frame, tracks: List[Track]) -> None:
        if self.meta_f:
            rec = {
                "camera_id": frame.camera_id,
                "frame": frame.index,
                "ts": round(frame.timestamp, 3),
                "objects": [
                    {"id": t.track_id, "class": t.cls_name, "score": round(t.score, 3), "box": [round(float(v), 1) for v in t.box]}
                    for t in tracks
                ],
            }
            self.meta_f.write(json.dumps(rec) + "\n")
        if self.mot_f:  # MOTChallenge format: frame,id,x,y,w,h,conf,-1,-1,-1 (frames are 1-based)
            for t in tracks:
                x1, y1, x2, y2 = t.box
                self.mot_f.write(f"{frame.index + 1},{t.track_id},{x1:.2f},{y1:.2f},{x2 - x1:.2f},{y2 - y1:.2f},{t.score:.3f},-1,-1,-1\n")

    def write_events(self, events: List[Event]) -> None:
        if self.events_f:
            for e in events:
                self.events_f.write(json.dumps(e.to_dict()) + "\n")
            self.events_f.flush()

    def write_video(self, img, fps: float) -> None:
        if not self.video_path:
            return
        if self.writer is None:
            h, w = img.shape[:2]
            self.writer = cv2.VideoWriter(self.video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        self.writer.write(img)

    def close(self) -> None:
        for f in (self.events_f, self.meta_f, self.mot_f):
            if f:
                f.close()
        if self.writer:
            self.writer.release()


class Pipeline:
    def __init__(
        self,
        source: VideoSource,
        detector: Detector,
        tracker: ByteTracker,
        engine: EventEngine,
        outputs: Optional[Outputs] = None,
        on_event: Optional[Callable[[Event], None]] = None,
        sinks: Optional[List[EventSink]] = None,
        reid=None,
        show: bool = False,
    ):
        self.source, self.detector, self.tracker, self.engine = source, detector, tracker, engine
        self.outputs = outputs or Outputs()
        self.on_event = on_event
        self.sinks = sinks or []
        self.reid = reid
        self.show = show
        self.timer = StageTimer()

    def run(self, max_frames: Optional[int] = None) -> RunStats:
        cam = self.source.camera_id
        stats = RunStats(camera_id=cam)
        recent: List[Event] = []
        t_start = time.perf_counter()
        frames_iter = self.source.frames()
        try:
            while True:
                with self.timer.measure("ingest"):
                    frame = next(frames_iter, None)
                if frame is None:
                    break
                with self.timer.measure("detect"):
                    dets = self.detector.detect(frame.image)
                with self.timer.measure("track"):
                    tracks = self.tracker.update(dets)
                with self.timer.measure("events"):
                    if self.reid is not None:
                        self.reid.update(frame, tracks)
                    events = self.engine.update(frame, tracks)
                    if self.reid is not None:
                        for e in events:
                            gid = self.reid.global_id(cam, e.track_id) if e.track_id is not None else None
                            if gid is not None:
                                e.details["global_id"] = gid

                self.outputs.write_frame_outputs(frame, tracks)
                self.outputs.write_events(events)
                for e in events:
                    stats.events += 1
                    stats.events_by_type[e.type] = stats.events_by_type.get(e.type, 0) + 1
                    recent.append(e)
                    log.info("EVENT %s", json.dumps(e.to_dict()))
                    if self.on_event:
                        self.on_event(e)
                    for sink in self.sinks:
                        sink.send(e)
                recent = [e for e in recent if frame.index - e.frame_index < 40]  # banner lifetime (frames)

                if self.outputs.video_path or self.show:
                    vis = frame.image.copy()
                    draw_rules(vis, self.engine.rules)
                    draw_tracks(vis, tracks)
                    elapsed = time.perf_counter() - t_start
                    draw_event_banner(vis, recent, (stats.frames + 1) / elapsed if elapsed > 0 else None)
                    self.outputs.write_video(vis, self.source.fps)
                    if self.show:
                        cv2.imshow(cam, vis)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            break

                stats.frames += 1
                if max_frames and stats.frames >= max_frames:
                    break
        finally:
            self.source.close()
            self.outputs.close()
            for sink in self.sinks:
                sink.close()
            if self.show:
                cv2.destroyAllWindows()
        stats.dropped_frames = self.source.dropped_frames
        stats.wall_seconds = time.perf_counter() - t_start
        stats.stages = self.timer.summary()
        return stats
