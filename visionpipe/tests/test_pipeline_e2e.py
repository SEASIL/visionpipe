"""End-to-end: synthetic video -> motion detector -> ByteTrack -> event engine -> JSONL/MOT outputs."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

from visionpipe.config import build_pipeline, load_config

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def synthetic_video(tmp_path_factory):
    out = tmp_path_factory.mktemp("vid") / "synthetic.mp4"
    subprocess.run([sys.executable, str(ROOT / "scripts" / "make_synthetic_video.py"), "--out", str(out)], check=True, capture_output=True)
    return out


def test_demo_pipeline_emits_expected_events(synthetic_video, tmp_path):
    cfg = load_config(
        str(ROOT / "configs" / "demo.yaml"),
        {
            "source": {"uri": str(synthetic_video)},
            "output": {
                "events_jsonl": str(tmp_path / "events.jsonl"),
                "metadata_jsonl": str(tmp_path / "meta.jsonl"),
                "mot_txt": str(tmp_path / "tracks.txt"),
                "video": None,
            },
        },
    )
    stats = build_pipeline(cfg).run()
    assert stats.frames == 560
    assert set(stats.stages) == {"ingest", "detect", "track", "events"}

    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    by_type = {}
    for e in events:
        by_type.setdefault(e["type"], []).append(e)
    # Scene design: #2 lingers in the zone, #3 runs through it, #1 crosses the counting line.
    assert len(by_type["loitering"]) == 1
    assert len(by_type["zone_intrusion"]) == 2
    assert len(by_type["line_crossing"]) == 1
    assert by_type["line_crossing"][0]["details"]["direction"] == "forward"
    loiterer = by_type["loitering"][0]["track_id"]
    runner = next(e["track_id"] for e in by_type["zone_intrusion"] if e["track_id"] != loiterer)
    assert loiterer != runner  # the runner must not also be flagged as loitering

    meta = (tmp_path / "meta.jsonl").read_text().splitlines()
    assert len(meta) == 560 and "objects" in json.loads(meta[0])
    assert (tmp_path / "tracks.txt").read_text().count("\n") > 500


def test_missing_source_raises():
    cfg = load_config(None, {"source": {"uri": "does_not_exist.mp4"}})
    with pytest.raises(FileNotFoundError):
        build_pipeline(cfg).run()


def test_no_source_configured_is_a_clear_error():
    with pytest.raises(ValueError, match="No video source"):
        build_pipeline(load_config(None))
