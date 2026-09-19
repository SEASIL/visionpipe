#!/usr/bin/env python
"""Export a trained model for deployment (ONNX, OpenVINO, TensorRT).

Examples:
  python scripts/export_model.py --weights models/best.pt --format onnx                 # FP32
  python scripts/export_model.py --weights models/best.pt --format onnx --half          # FP16 (GPU)
  python scripts/export_model.py --weights models/best.pt --format openvino --int8 --data data/dataset/data.yaml
  python scripts/export_model.py --weights models/best.pt --format engine --half        # needs NVIDIA GPU + TensorRT
INT8 needs a calibration dataset (--data). Always re-measure mAP after quantisation
(scripts/benchmark.py --data ...) - accuracy loss is model- and data-dependent.
"""
import argparse

from ultralytics import YOLO


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--format", required=True, choices=["onnx", "openvino", "engine"])
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--half", action="store_true", help="FP16")
    ap.add_argument("--int8", action="store_true", help="INT8 (requires --data for calibration)")
    ap.add_argument("--data", help="dataset yaml, used for INT8 calibration")
    ap.add_argument("--dynamic", action="store_true", help="dynamic input shapes (ONNX)")
    args = ap.parse_args()

    kw = {"format": args.format, "imgsz": args.imgsz, "half": args.half, "int8": args.int8}
    if args.data:
        kw["data"] = args.data
    if args.format == "onnx":
        kw.update(dynamic=args.dynamic, simplify=True)
    out = YOLO(args.weights).export(**kw)
    print(f"exported -> {out}")


if __name__ == "__main__":
    main()
