# VisionPipe: real-time multi-camera video analytics

Turns raw video streams into **reliable, structured business events**: zone intrusion, loitering,
line-crossing counts, crowd detection and abandoned objects. Pipeline: RTSP/file ingestion → detection →
multi-object tracking → rule-based event engine → JSON/annotated video output.

```mermaid
flowchart LR
    A[RTSP / file / webcam<br/>threaded reader, drop-oldest,<br/>auto-reconnect] --> B[Detector<br/>YOLO via Ultralytics or<br/>ONNX Runtime TRT/CUDA/OpenVINO/CPU]
    B --> C[ByteTrack-style tracker<br/>Kalman + 2-stage IoU matching]
    C --> D[Event engine<br/>zones, lines, dwell, crowd,<br/>abandoned object]
    D --> E[(events.jsonl<br/>metadata.jsonl<br/>MOT tracks.txt<br/>annotated.mp4)]
    C -.-> F[Cross-camera re-ID<br/>global IDs]
    F -.-> D
```

**What is in the box**

| Area | Implementation |
|---|---|
| Ingestion | `io/source.py`: file + live streams, bounded queue that drops stale frames, exponential-backoff reconnect |
| Detection | `detect/`: Ultralytics YOLO wrapper, **hand-written ONNX Runtime YOLO detector** (letterbox, decode, class-aware NMS, provider auto-selection), background-subtraction baseline |
| Tracking | `track/`: own ByteTrack-style tracker (constant-velocity Kalman filter, Hungarian matching, low-confidence rescue stage, no cross-class matches) |
| Event engine | `events/rules.py`: persistence, debouncing, fire-once, hysteresis, min-gap and stale-state GC to keep false positives down |
| Multi-camera | `multicam.py` (thread per camera, shared detector) and `reid.py` (baseline colour-histogram global IDs) |
| Dataset tooling | `datatools.py`: label QC, auto-labelling (pseudo-labels for CVAT), uncertainty-based active-learning review queue |
| MLOps | DVC pipeline (`dvc.yaml`), MLflow logging (`scripts/train.py`), mAP regression gate (`scripts/evaluate.py`), GitHub Actions CI, Docker (CPU + GPU) |
| Optimisation | `scripts/export_model.py` (ONNX FP32/FP16, OpenVINO, TensorRT, INT8) and `scripts/benchmark.py` (latency/FPS/memory/mAP table) |
| Evaluation | MOTA / IDF1 / ID-switches via `scripts/eval_tracking.py`, per-stage latency in every run |

## Quick start (no downloads, no GPU)

```bash
pip install -e ".[dev]"
make demo          # generates a synthetic clip, runs the pipeline, writes outputs/cam0/*
make test          # 49 unit + end-to-end tests
make smoke         # CI gate: expected events + MOTA/IDF1 thresholds
```

The demo uses a background-subtraction detector so it works with no model weights. The synthetic scene has
three actors, and the run produces exactly the events the scene was designed to trigger:
one line crossing, one loitering alert (the actor that lingers) and two intrusions (the lingerer and the
runner), with the runner correctly **not** flagged as loitering.

## Run on real footage

```bash
pip install -e ".[train]"                       # Ultralytics, MLflow, DVC ...
python -m visionpipe --source rtsp://user:pass@10.0.0.5/stream1 --config configs/site.yaml   # copy of configs/site.example.yaml --show
python -m visionpipe --source clip.mp4 --config configs/site.yaml   # copy of configs/site.example.yaml --video-out out.mp4 --events-out events.jsonl
python -m visionpipe -c configs/multicam.yaml   # several cameras at once
```

Use `detector: {type: yolo, weights: yolov8n.pt}` or, for the lightweight runtime, `{type: onnx, weights: models/best.onnx}`.
Rules are declared in YAML (see `configs/demo.yaml`). Draw zone polygons in pixel coordinates of the stream.

Example event:

```json
{"type": "loitering", "camera_id": "cam0", "frame_index": 246, "timestamp": 9.84, "track_id": 2,
 "class": "person", "zone": "restricted_area", "details": {"dwell_seconds": 6.0}}
```

## Train, evaluate, optimise (the MLOps loop)

```bash
python scripts/dataset_qc.py --images data/dataset/images/train --labels data/dataset/labels/train --num-classes 3
python scripts/auto_label.py --weights yolov8n.pt --images data/unlabeled --out data/pseudo --classes person car
dvc repro                                                   # qc -> train (MLflow) -> evaluate (gate)
python scripts/export_model.py --weights models/best.pt --format onnx --half
python scripts/benchmark.py --models models/best.pt models/best.onnx --source clip.mp4 --data data/dataset/data.yaml --out docs/BENCHMARKS.md
python scripts/eval_tracking.py --gt MOT17-04/gt/gt.txt --gt-class 1 --pred outputs/cam0/tracks.txt
```

## How false positives are controlled

| Technique | Where | Effect |
|---|---|---|
| `min_hits` | tracker | one-frame detector flicker never becomes a track |
| Low-confidence rescue | tracker | occluded / blurry objects keep their ID instead of fragmenting |
| Persistence (`min_frames`) | intrusion, crowd | condition must hold N consecutive frames |
| Exit debouncing (`exit_frames`) | zone rules | tracker jitter at the polygon edge does not reset dwell time |
| Fire-once per episode | all rules | one alert per person per visit, not one per frame |
| Hysteresis | crowd | different on/off conditions, so alerts do not flap |
| `min_gap_frames` | line crossing | a track jittering on the line is not counted repeatedly |
| Owner-proximity check | abandoned object | a bag with its owner nearby is never flagged |

## Results

> **Fill this in with numbers from YOUR runs. Do not leave the placeholders in a public repo.**

| Metric | Value | Dataset / hardware |
|---|---|---|
| mAP@0.5, pretrained baseline | _TBD_ | _your dataset_ |
| mAP@0.5, fine-tuned | _TBD_ | _your dataset_ |
| MOTA / IDF1 / ID switches | _TBD_ | _MOT17 sequence or your clip_ |
| End-to-end FPS, PyTorch vs ONNX vs TensorRT/OpenVINO | _TBD_ | _your GPU / CPU_ |
| False-alert reduction from event filtering | _TBD_ | _count alerts with and without persistence/debounce on the same clip_ |

Numbers that ARE produced by this repo's CI on the synthetic clip (a sanity check, not a benchmark):
MOTA 98.2%, IDF1 99.1%, 0 ID switches. The synthetic scene is easy by construction, so do not quote these as performance.

See `docs/FAILURE_ANALYSIS.md` (fill in per condition), `docs/ARCHITECTURE.md` and `docs/BENCHMARKS.md`.

## Known limitations

* Colour-histogram re-ID is a weak baseline; replace with a learned embedding (OSNet) for real deployments.
* Multi-camera mode shares one detector behind a lock, with no batching. For many streams, use DeepStream or a batching inference server.
* Zone polygons are static, in pixel coordinates; there is no camera-motion compensation or calibration to ground-plane coordinates.
* The tracker is a compact re-implementation, not the reference ByteTrack code; validate on MOT17 before comparing with published numbers.

## Repo layout

```
src/visionpipe/   io/ detect/ track/ events/ pipeline.py multicam.py reid.py datatools.py config.py cli.py
scripts/          make_synthetic_video, train, evaluate, export_model, benchmark, eval_tracking, dataset_qc, auto_label, check_events
configs/          demo.yaml  multicam.yaml
tests/            geometry, tracker, rules, ONNX detector, e2e pipeline, source, re-ID, datatools, multi-camera
dvc.yaml params.yaml Dockerfile docker/Dockerfile.gpu Makefile .github/workflows/ci.yml
```
