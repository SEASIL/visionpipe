# Benchmarks

Hardware: AMD64 Family 25 Model 117 Stepping 2, AuthenticAMD, 16 logical CPUs, Windows-10-10.0.26200-SP0 | NVIDIA GeForce RTX 4060 Laptop GPU, 592.82, P8 | CUDA 12.1 | frames=200 imgsz=640 batch=1

| Model | Backend | mean ms | p50 ms | p95 ms | FPS | Warmup ms | +RSS MB | GPU MB | mAP50 | mAP50-95 |
|---|---|---|---|---|---|---|---|---|---|---|
| yolov8n.pt | ultralytics | 22.9 | 19.2 | 39.7 | 43.6 | 7009 | 1084 | 41 | - | - |
| yolov8n.onnx | CPUExecutionProvider | 44.6 | 46.6 | 52.5 | 22.4 | 1036 | 105 | - | - | - |
