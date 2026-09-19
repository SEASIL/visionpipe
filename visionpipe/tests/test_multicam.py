import json
import subprocess
import sys
from pathlib import Path

from visionpipe.config import load_config
from visionpipe.multicam import run_multi

ROOT = Path(__file__).resolve().parents[1]


def test_two_cameras_run_concurrently_with_own_rules(tmp_path):
    vid = tmp_path / "s.mp4"
    subprocess.run([sys.executable, str(ROOT / "scripts" / "make_synthetic_video.py"), "--out", str(vid)], check=True, capture_output=True)
    cfg = load_config(str(ROOT / "configs" / "multicam.yaml"))
    for cam in cfg["cameras"]:
        cam["uri"] = str(vid)
    cfg["output"]["events_jsonl"] = str(tmp_path / "{camera_id}" / "events.jsonl")

    stats = run_multi(cfg)
    assert [s.camera_id for s in stats] == ["lobby", "warehouse"]
    assert all(s.frames == 560 for s in stats)

    def types(cam):
        return sorted(json.loads(x)["type"] for x in (tmp_path / cam / "events.jsonl").read_text().splitlines())

    assert types("lobby") == ["line_crossing"]  # per-camera rules are isolated
    assert types("warehouse") == ["loitering", "zone_intrusion", "zone_intrusion"]
