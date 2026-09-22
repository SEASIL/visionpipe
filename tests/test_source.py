import subprocess
import sys
import time
from pathlib import Path

import pytest

from visionpipe.io import VideoSource, is_live_uri

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def video(tmp_path_factory):
    out = tmp_path_factory.mktemp("src") / "v.mp4"
    subprocess.run([sys.executable, str(ROOT / "scripts" / "make_synthetic_video.py"), "--out", str(out)], check=True, capture_output=True)
    return str(out)


def test_uri_classification():
    assert is_live_uri("rtsp://10.0.0.5/stream1")
    assert is_live_uri(0)
    assert not is_live_uri("clip.mp4")


def test_file_source_is_deterministic(video):
    src = VideoSource(video, camera_id="c1")
    frames = []
    for f in src.frames():
        frames.append(f)
        if len(frames) == 30:
            break
    assert [f.index for f in frames] == list(range(30))
    assert frames[25].timestamp == pytest.approx(1.0, abs=1e-6)  # 25 fps
    assert frames[0].camera_id == "c1" and frames[0].image.shape == (480, 640, 3)


def test_loop_restarts_file(video):
    src = VideoSource(video, loop=True)
    n = 0
    for _ in src.frames():
        n += 1
        if n > 600:  # more than the 560 frames in the clip
            break
    assert n > 600
    src.close()


def test_live_reader_delivers_frames_and_terminates(video):
    src = VideoSource(video, live=True, reconnect=False, max_queue=2)
    idx = [f.index for f in src.frames()]
    assert idx == list(range(len(idx))) and len(idx) > 0
    # queue is tiny and drops oldest frames when the consumer is slow: never all 560 arrive
    assert len(idx) <= 560


def test_live_drops_old_frames_when_consumer_is_slow(video):
    """Producer (file, far faster than real time) outruns a slow consumer: the tiny queue must drop
    the oldest frames instead of buffering all 560, keeping latency bounded."""
    src = VideoSource(video, live=True, reconnect=False, max_queue=2)
    stamps = []
    for f in src.frames():
        stamps.append(f.timestamp)
        time.sleep(0.02)
    assert 1 <= len(stamps) < 280  # far fewer than the 560 frames produced
    assert stamps == sorted(stamps)  # order preserved
    assert src.dropped_frames > 0


def test_live_unreachable_without_reconnect_ends_cleanly():
    src = VideoSource("rtsp://127.0.0.1:1/none", reconnect=False)
    assert list(src.frames()) == []
