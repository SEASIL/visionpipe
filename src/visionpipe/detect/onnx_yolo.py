"""YOLOv8/YOLO11 detector running on ONNX Runtime with hand-written pre/post-processing.

Why hand-written? It removes the PyTorch/Ultralytics dependency from the deployment image
(tiny container, fast start-up) and makes every step of the inference path explicit:
letterbox -> normalise -> session.run -> decode -> undo letterbox -> class-aware NMS.

Expected output layout (Ultralytics export): (1, 4 + num_classes, num_anchors), boxes as cx,cy,w,h.
Execution providers are picked automatically: TensorRT > CUDA > OpenVINO > CPU.
"""
from __future__ import annotations

import ast
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from ..types import Detection
from .base import Detector, resolve_class_ids
from .coco import COCO_NAMES

PROVIDER_PREFERENCE = [
    "TensorrtExecutionProvider",
    "CUDAExecutionProvider",
    "OpenVINOExecutionProvider",
    "CPUExecutionProvider",
]


def letterbox(img: np.ndarray, size: Tuple[int, int], color: int = 114):
    """Resize keeping aspect ratio, pad to `size` (h, w). Returns image, scale, pad_left, pad_top."""
    h, w = img.shape[:2]
    r = min(size[0] / h, size[1] / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top, left = (size[0] - nh) // 2, (size[1] - nw) // 2
    out = np.full((size[0], size[1], 3), color, np.uint8)
    out[top : top + nh, left : left + nw] = resized
    return out, r, left, top


class ONNXYoloDetector(Detector):
    def __init__(
        self,
        model_path: str,
        conf: float = 0.25,
        iou: float = 0.5,
        imgsz: int = 640,
        classes: Optional[Sequence] = None,
        providers: Optional[List[str | tuple]] = None,
        intra_op_threads: int = 0,
    ):
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if intra_op_threads:
            so.intra_op_num_threads = intra_op_threads
            
        if providers:
            chosen = providers
        else:
            available = ort.get_available_providers()
            chosen = [p for p in PROVIDER_PREFERENCE if p in available]
            
        self.session = ort.InferenceSession(model_path, sess_options=so, providers=chosen)
        self.providers = self.session.get_providers()

        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        self.fp16 = "float16" in inp.type
        shape = inp.shape  # e.g. [1, 3, 640, 640] or dynamic names
        h = shape[2] if isinstance(shape[2], int) else imgsz
        w = shape[3] if isinstance(shape[3], int) else imgsz
        self.input_hw = (h, w)

        self.names = self._read_names()
        self.conf, self.iou = conf, iou
        self.class_ids = resolve_class_ids(self.names, classes)

    def _read_names(self) -> Dict[int, str]:
        meta = self.session.get_modelmeta().custom_metadata_map
        if "names" in meta:  # Ultralytics stores class names here, so fine-tuned models just work
            try:
                return {int(k): v for k, v in ast.literal_eval(meta["names"]).items()}
            except (ValueError, SyntaxError):
                pass
        return dict(enumerate(COCO_NAMES))

    # ------------------------------------------------------------------ steps
    def preprocess(self, image: np.ndarray):
        lb, r, left, top = letterbox(image, self.input_hw)
        x = lb[:, :, ::-1].transpose(2, 0, 1)[None]  # BGR->RGB, HWC->NCHW
        x = np.ascontiguousarray(x, dtype=np.float32) / 255.0
        return x.astype(np.float16 if self.fp16 else np.float32, copy=False), r, left, top

    def postprocess(self, raw: np.ndarray, r: float, left: int, top: int, shape) -> List[Detection]:
        pred = np.squeeze(raw, 0).T.astype(np.float32)  # (anchors, 4 + nc)
        scores_all = pred[:, 4:]
        cls_ids = scores_all.argmax(1)
        scores = scores_all[np.arange(len(pred)), cls_ids]
        keep = scores >= self.conf
        if self.class_ids is not None:
            keep &= np.isin(cls_ids, self.class_ids)
        pred, cls_ids, scores = pred[keep], cls_ids[keep], scores[keep]
        if len(pred) == 0:
            return []

        cx, cy, w, h = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
        boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1)
        boxes[:, [0, 2]] = (boxes[:, [0, 2]] - left) / r
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] - top) / r
        H, W = shape[:2]
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, W - 1)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, H - 1)

        # class-aware NMS via coordinate offset trick (boxes of different classes never overlap)
        offset = cls_ids[:, None].astype(np.float32) * 10000.0
        nms_boxes = boxes + offset
        xywh = np.stack(
            [nms_boxes[:, 0], nms_boxes[:, 1], nms_boxes[:, 2] - nms_boxes[:, 0], nms_boxes[:, 3] - nms_boxes[:, 1]], 1
        )
        idx = cv2.dnn.NMSBoxes(xywh.tolist(), scores.tolist(), self.conf, self.iou)
        idx = np.array(idx).reshape(-1).astype(int)
        return [
            Detection(boxes[i].astype(float), float(scores[i]), int(cls_ids[i]), self.names.get(int(cls_ids[i]), str(cls_ids[i])))
            for i in idx
        ]

    def detect(self, image: np.ndarray) -> List[Detection]:
        x, r, left, top = self.preprocess(image)
        raw = self.session.run(None, {self.input_name: x})[0]
        return self.postprocess(raw, r, left, top, image.shape)
