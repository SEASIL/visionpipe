"""ByteTrack-style multi-object tracker.

Key idea from the ByteTrack paper (Zhang et al., 2022): do NOT throw away low-confidence
detections. Associate high-confidence detections first, then use the low-confidence ones
to rescue tracks that would otherwise be lost to occlusion or blur.

This is a compact re-implementation (Kalman motion model + IoU + Hungarian matching),
intentionally readable rather than a line-by-line port of the reference code.
"""
from __future__ import annotations

from enum import IntEnum
from typing import List, Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from ..geometry import cxcywh_to_xyxy, iou_matrix, xyxy_to_cxcywh
from ..types import Detection, Track
from .kalman import KalmanBoxFilter


class _State(IntEnum):
    TRACKED = 1
    LOST = 2


class _Tracklet:
    def __init__(self, det: Detection, track_id: int, kf: KalmanBoxFilter):
        self.id = track_id
        self.mean, self.cov = kf.initiate(xyxy_to_cxcywh(det.box))
        self.score = det.score
        self.cls_id = det.cls_id
        self.cls_name = det.cls_name
        self.hits = 1
        self.age = 1
        self.time_since_update = 0
        self.state = _State.TRACKED

    @property
    def box(self) -> np.ndarray:
        return cxcywh_to_xyxy(self.mean)

    def predict(self, kf: KalmanBoxFilter) -> None:
        self.mean, self.cov = kf.predict(self.mean, self.cov)
        self.age += 1
        self.time_since_update += 1

    def update(self, det: Detection, kf: KalmanBoxFilter) -> None:
        self.mean, self.cov = kf.update(self.mean, self.cov, xyxy_to_cxcywh(det.box))
        self.score = det.score
        self.cls_id, self.cls_name = det.cls_id, det.cls_name
        self.hits += 1
        self.time_since_update = 0
        self.state = _State.TRACKED


class ByteTracker:
    def __init__(
        self,
        high_thresh: float = 0.5,
        low_thresh: float = 0.1,
        new_track_thresh: float = 0.6,
        match_iou: float = 0.2,
        second_match_iou: float = 0.5,
        max_lost: int = 30,
        min_hits: int = 2,
    ):
        self.high_thresh = high_thresh
        self.low_thresh = low_thresh
        self.new_track_thresh = new_track_thresh
        self.match_iou = match_iou
        self.second_match_iou = second_match_iou
        self.max_lost = max_lost
        self.min_hits = min_hits
        self.kf = KalmanBoxFilter()
        self._tracklets: List[_Tracklet] = []
        self._next_id = 1

    def reset(self) -> None:
        self._tracklets.clear()
        self._next_id = 1

    @staticmethod
    def _associate(
        tracks: Sequence[_Tracklet], dets: Sequence[Detection], min_iou: float
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        if not tracks or not dets:
            return [], list(range(len(tracks))), list(range(len(dets)))
        iou = iou_matrix(np.array([t.box for t in tracks]), np.array([d.box for d in dets]))
        t_cls = np.array([t.cls_id for t in tracks])
        d_cls = np.array([d.cls_id for d in dets])
        iou[t_cls[:, None] != d_cls[None, :]] = 0.0  # never match across classes
        rows, cols = linear_sum_assignment(1.0 - iou)
        matches = [(r, c) for r, c in zip(rows, cols) if iou[r, c] >= min_iou]
        matched_t = {r for r, _ in matches}
        matched_d = {c for _, c in matches}
        un_t = [i for i in range(len(tracks)) if i not in matched_t]
        un_d = [i for i in range(len(dets)) if i not in matched_d]
        return matches, un_t, un_d

    def update(self, detections: Sequence[Detection]) -> List[Track]:
        highs = [d for d in detections if d.score >= self.high_thresh]
        lows = [d for d in detections if self.low_thresh <= d.score < self.high_thresh]

        for t in self._tracklets:
            t.predict(self.kf)

        # Stage 1: high-confidence detections vs. all tracks (tracked + recently lost)
        pool = list(self._tracklets)
        m1, un_t, un_d = self._associate(pool, highs, self.match_iou)
        for ti, di in m1:
            pool[ti].update(highs[di], self.kf)

        # Stage 2: low-confidence detections rescue still-tracked, unmatched tracks
        remaining = [pool[i] for i in un_t if pool[i].state == _State.TRACKED]
        m2, un_t2, _ = self._associate(remaining, lows, self.second_match_iou)
        for ti, di in m2:
            remaining[ti].update(lows[di], self.kf)
        for i in un_t2:
            remaining[i].state = _State.LOST

        # New tracks only from confident, unmatched detections
        for di in un_d:
            d = highs[di]
            if d.score >= self.new_track_thresh:
                self._tracklets.append(_Tracklet(d, self._next_id, self.kf))
                self._next_id += 1

        self._tracklets = [t for t in self._tracklets if t.time_since_update <= self.max_lost]

        return [
            Track(
                track_id=t.id,
                box=t.box,
                score=t.score,
                cls_id=t.cls_id,
                cls_name=t.cls_name,
                hits=t.hits,
                age=t.age,
            )
            for t in self._tracklets
            if t.state == _State.TRACKED and t.time_since_update == 0 and t.hits >= self.min_hits
        ]
