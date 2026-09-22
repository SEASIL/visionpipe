import cv2
import numpy as np

from visionpipe.detect.onnx_yolo import ONNXYoloDetector
from visionpipe.detect.yolo import YOLODetector


def main():
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    # Draw something so it's not totally blank
    cv2.rectangle(img, (100, 100), (300, 300), (255, 255, 255), -1)

    print("Running YOLODetector on yolov8n.pt...", flush=True)
    uda = YOLODetector("yolov8n.pt", conf=0.25, iou=0.45)
    dets_pt = uda.detect(img)
    print(f"PT detections: {len(dets_pt)}", flush=True)
    for d in dets_pt:
        print(f"  {d.cls_name}: {d.score:.3f} {d.box}", flush=True)

    print("Running ONNXYoloDetector on yolov8n.onnx...", flush=True)
    oda = ONNXYoloDetector("yolov8n.onnx", conf=0.25, iou=0.45)
    
    print("Intercepting raw output...", flush=True)
    # Let's intercept the raw output to confirm layout (1, 4+nc, N)
    ort_inputs = {oda.session.get_inputs()[0].name: oda.preprocess(img)[0]}
    raw_out = oda.session.run(None, ort_inputs)[0]
    print(f"ONNX raw output shape: {raw_out.shape} -> Expected (1, 4+80, 8400)", flush=True)
    
    print("Running detect...", flush=True)
    dets_onnx = oda.detect(img)
    print(f"ONNX detections: {len(dets_onnx)}", flush=True)
    for d in dets_onnx:
        print(f"  {d.cls_name}: {d.score:.3f} {d.box}", flush=True)

if __name__ == "__main__":
    main()
