# Experiment Log

| Metric | Value | Command | Output File | Date | Hardware |
|---|---|---|---|---|---|
| mAP50 (Tiny Dataset Validation) | 0.000 | `python scripts/evaluate.py --weights models/best.pt --data data/tiny_dataset/data.yaml --min-map50 0.0 --out metrics/eval.json` | `metrics/eval.json` | 2026-09-19 | NVIDIA RTX 4060 Laptop GPU |
| MOTA (MOT17-04-FRCNN, yolov8n.onnx detector) | 29.6% | `python scripts/eval_tracking.py --gt data/MOT17/train/MOT17-04-FRCNN/gt/gt.txt --pred outputs/cam0/tracks.txt --gt-class 1 --out metrics/tracking_eval.json` | `metrics/tracking_eval.json` | 2026-09-22 | NVIDIA RTX 4050 Laptop |
| IDF1 (MOT17-04-FRCNN, yolov8n.onnx detector) | 43.0% | `python scripts/eval_tracking.py --gt data/MOT17/train/MOT17-04-FRCNN/gt/gt.txt --pred outputs/cam0/tracks.txt --gt-class 1 --out metrics/tracking_eval.json` | `metrics/tracking_eval.json` | 2026-09-22 | NVIDIA RTX 4050 Laptop |
| ID Switches (MOT17-04-FRCNN, yolov8n.onnx detector) | 28 | `python scripts/eval_tracking.py --gt data/MOT17/train/MOT17-04-FRCNN/gt/gt.txt --pred outputs/cam0/tracks.txt --gt-class 1 --out metrics/tracking_eval.json` | `metrics/tracking_eval.json` | 2026-09-22 | NVIDIA RTX 4050 Laptop |
| Fragmentations (MOT17-04-FRCNN, yolov8n.onnx detector) | 118 | `python scripts/eval_tracking.py --gt data/MOT17/train/MOT17-04-FRCNN/gt/gt.txt --pred outputs/cam0/tracks.txt --gt-class 1 --out metrics/tracking_eval.json` | `metrics/tracking_eval.json` | 2026-09-22 | NVIDIA RTX 4050 Laptop |
| mAP50 yolov8n baseline | 0.988 | python scripts/evaluate.py --weights yolov8n.pt ... | metrics/eval_yolov8n_base.json | 2026-09-22 | NVIDIA RTX 4050 Laptop |
| mAP50 yolov8n fine-tuned | 0.694 | dvc repro (yolov8n) | metrics/eval.json | 2026-09-22 | NVIDIA RTX 4050 Laptop |
| mAP50 yolov8s baseline | 0.949 | python scripts/evaluate.py --weights yolov8s.pt ... | metrics/eval_yolov8s_base.json | 2026-09-22 | NVIDIA RTX 4050 Laptop |
| mAP50 yolov8s fine-tuned | 0.947 | dvc repro (yolov8s) | metrics/eval.json | 2026-09-22 | NVIDIA RTX 4050 Laptop |
| FPS yolov8n (PyTorch) | 43.6 | python scripts/benchmark.py --models yolov8n.pt yolov8n.onnx --providers CPUExecutionProvider --source data/synthetic.mp4 | docs/BENCHMARKS.md | 2026-09-22 | NVIDIA RTX 4050 Laptop |
| FPS yolov8n (ONNX CPU) | 22.4 | python scripts/benchmark.py --models yolov8n.pt yolov8n.onnx --providers CPUExecutionProvider --source data/synthetic.mp4 | docs/BENCHMARKS.md | 2026-09-22 | NVIDIA RTX 4050 Laptop |
