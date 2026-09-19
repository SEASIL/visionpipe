import numpy as np

from tests.helpers import make_det
from visionpipe.track import ByteTracker


def run(tracker, frames):
    return [tracker.update(dets) for dets in frames]


def test_single_object_keeps_id():
    tr = ByteTracker(min_hits=1)
    out = run(tr, [[make_det(100 + 5 * i, 100)] for i in range(30)])
    ids = {t.track_id for frame in out for t in frame}
    assert ids == {1}
    assert len(out[-1]) == 1
    # Kalman-smoothed box should follow the object
    assert abs(out[-1][0].center[0] - (100 + 5 * 29)) < 5


def test_two_crossing_objects_keep_distinct_ids():
    tr = ByteTracker(min_hits=1)
    frames = []
    for i in range(40):
        frames.append([make_det(50 + 8 * i, 100 + 2 * i), make_det(370 - 8 * i, 100 + 2 * i)])
    out = run(tr, frames)
    ids_first = {t.track_id: t.center for t in out[0]}
    ids_last = {t.track_id: t.center for t in out[-1]}
    assert len(ids_last) == 2
    # left object at start must still be the one with the same id at the end (it moves right)
    left_id = min(ids_first, key=lambda k: ids_first[k][0])
    assert ids_last[left_id][0] > 300


def test_low_confidence_detection_rescues_track():
    tr = ByteTracker(min_hits=1, high_thresh=0.5, low_thresh=0.1)
    frames = [[make_det(100 + 3 * i, 100, score=0.9)] for i in range(10)]
    frames += [[make_det(100 + 3 * i, 100, score=0.3)] for i in range(10, 15)]  # occluded / blurry
    frames += [[make_det(100 + 3 * i, 100, score=0.9)] for i in range(15, 25)]
    out = run(tr, frames)
    ids = {t.track_id for f in out for t in f}
    assert ids == {1}, "ID should survive a run of low-confidence frames"
    assert all(len(f) == 1 for f in out[10:15])


def test_track_survives_short_gap_but_is_dropped_after_max_lost():
    tr = ByteTracker(min_hits=1, max_lost=5)
    for _ in range(10):
        tr.update([make_det(100, 100)])
    for _ in range(3):
        assert tr.update([]) == []
    back = tr.update([make_det(100, 100)])
    assert [t.track_id for t in back] == [1]  # same id after 3 missed frames
    for _ in range(8):
        tr.update([])
    new = tr.update([make_det(100, 100)])
    assert new and new[0].track_id != 1  # id 1 expired


def test_no_cross_class_matching():
    tr = ByteTracker(min_hits=1)
    tr.update([make_det(100, 100, cls_id=0, cls_name="person")])
    out = tr.update([make_det(100, 100, cls_id=2, cls_name="car")])
    assert out[0].cls_name == "car" and out[0].track_id != 1


def test_min_hits_suppresses_single_frame_flicker():
    tr = ByteTracker(min_hits=3)
    assert tr.update([make_det(100, 100)]) == []
    assert tr.update([make_det(102, 100)]) == []
    assert len(tr.update([make_det(104, 100)])) == 1
    # a one-off false positive never surfaces
    tr2 = ByteTracker(min_hits=3)
    assert tr2.update([make_det(300, 300)]) == []
    assert tr2.update([]) == []


def test_empty_frames_are_fine():
    tr = ByteTracker()
    assert tr.update([]) == []
    assert isinstance(np.zeros(1), np.ndarray)
