# Benchmarks

Generate with `scripts/benchmark.py` on the hardware you want to claim numbers for:

```bash
python scripts/export_model.py --weights models/best.pt --format onnx
python scripts/export_model.py --weights models/best.pt --format onnx --half     # GPU only
python scripts/export_model.py --weights models/best.pt --format openvino
python scripts/benchmark.py --models models/best.pt models/best.onnx models/best_openvino_model \
    --source data/site_clip.mp4 --frames 300 --data data/dataset/data.yaml --out docs/BENCHMARKS.md
```

Overwrite this file with the generated table. Report the latency figures next to the mAP of each variant,
because a faster model that lost accuracy is not an improvement. Note the exact GPU/CPU used.
