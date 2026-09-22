import os
import shutil
import subprocess
import sys
from pathlib import Path


def run_cmd(cmd, hide_output=False):
    print(f"Running: {' '.join(cmd)}")
    kwargs = {}
    if hide_output:
        kwargs["stdout"] = subprocess.DEVNULL
    subprocess.run(cmd, check=True, **kwargs)


def main():
    root_dir = Path(__file__).resolve().parent.parent
    os.chdir(root_dir)

    # 1. Generate synthetic video
    run_cmd([sys.executable, "scripts/make_synthetic_video.py", "--out", "data/synthetic.mp4"])

    # 2. Clean outputs directory
    outputs_dir = root_dir / "outputs"
    if outputs_dir.exists():
        shutil.rmtree(outputs_dir, ignore_errors=True)

    # 3. Run pipeline
    run_cmd([sys.executable, "-m", "visionpipe", "-c", "configs/demo.yaml", "--stats-out", "outputs/stats.json"], hide_output=True)

    # 4. Check events
    run_cmd([sys.executable, "scripts/check_events.py", "outputs/cam0/events.jsonl", "--expect", "loitering=1", "zone_intrusion=2", "line_crossing=1"])
    run_cmd([sys.executable, "scripts/check_events.py", "outputs/cam1/events.jsonl", "--expect", "loitering=1", "zone_intrusion=2", "line_crossing=1"])

    # 5. Evaluate tracking
    run_cmd([sys.executable, "scripts/eval_tracking.py", "--gt", "data/synthetic.gt.txt", "--pred", "outputs/cam0/tracks.txt", "--min-mota", "0.9", "--min-idf1", "0.9"])
    run_cmd([sys.executable, "scripts/eval_tracking.py", "--gt", "data/synthetic.gt.txt", "--pred", "outputs/cam1/tracks.txt", "--min-mota", "0.9", "--min-idf1", "0.9"])
    print("Smoke test passed successfully!")


if __name__ == "__main__":
    main()
