# Architecture and design decisions

## Data flow

`VideoSource` yields `Frame`s → `Detector.detect()` returns `Detection`s → `ByteTracker.update()` returns
`Track`s with stable IDs → `EventEngine.update()` runs every `Rule` and returns `Event`s → sinks write
JSONL / MOT / annotated video. Every stage is timed (`StageTimer`), so each run reports p50/p95 latency per stage.

## Key decisions

**Bounded, drop-oldest queue for live streams.** If inference is slower than the camera frame rate, a normal queue
grows without bound and latency climbs forever. Dropping the oldest frame keeps the pipeline near real time.
File sources are read synchronously so results are reproducible.

**Event logic is separate from the model.** Rules consume tracks, not pixels, so they are unit-tested with
hand-built tracks (no video, no GPU) and stay valid when the detector is swapped.

**Foot point for zones, centre point for lines.** The bottom-centre of a person box is where they touch the ground,
which is what a floor-plane polygon means. Line counting uses the centre, which is more stable frame to frame.

**Hand-written ONNX detector.** Removes PyTorch from the deployment image and exposes the whole inference path.
It reads class names from the ONNX metadata written by Ultralytics, so fine-tuned models work without changes.

**Tracker.** ByteTrack's insight is to keep low-confidence detections and use them in a second matching stage.
The Kalman filter noise scales with box height, so near and far objects behave consistently.

## Scaling notes (what would change for many cameras)

* Batch frames from several cameras into one inference call (DeepStream `nvinfer`, Triton, or ORT with batch>1).
* Hardware decode (NVDEC) via GStreamer instead of software decode in OpenCV.
* Move each camera to its own process, or use a message queue between ingest, inference and event stages.
* Replace the colour-histogram re-ID with learned embeddings and a vector index.
