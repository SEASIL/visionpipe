import numpy as np

from visionpipe.types import Detection, Frame, Track


def make_track(tid, cx, cy, w=40, h=80, cls="person", score=0.9):
    box = np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dtype=float)
    return Track(track_id=tid, box=box, score=score, cls_id=0 if cls == "person" else 1, cls_name=cls)


def make_det(cx, cy, w=40, h=80, score=0.9, cls_id=0, cls_name="person"):
    box = np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dtype=float)
    return Detection(box=box, score=score, cls_id=cls_id, cls_name=cls_name)


def frame(i, fps=25.0, cam="cam0"):
    return Frame(camera_id=cam, index=i, timestamp=i / fps, image=np.zeros((10, 10, 3), np.uint8))
