"""Dataset management utilities for YOLO-format detection datasets.

* run_qc            - automated label quality checks (broken lines, out-of-range boxes, duplicates, ...)
* pseudo_label      - auto-annotate images with a detector (output can be imported into CVAT for review)
* uncertainty_score / select_for_review - a simple active-learning loop: send the images the model is
                      least sure about to human annotators first
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .geometry import iou_matrix
from .types import Detection

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class Issue:
    severity: str  # "error" | "warning" | "info"
    code: str
    file: str
    message: str


@dataclass
class QCReport:
    n_images: int = 0
    n_boxes: int = 0
    n_background: int = 0
    class_counts: Counter = field(default_factory=Counter)
    issues: List[Issue] = field(default_factory=list)

    @property
    def errors(self) -> List[Issue]:
        return [i for i in self.issues if i.severity == "error"]

    @property
    def warnings(self) -> List[Issue]:
        return [i for i in self.issues if i.severity == "warning"]

    def to_dict(self) -> dict:
        return {
            "images": self.n_images,
            "boxes": self.n_boxes,
            "background_images": self.n_background,
            "class_counts": dict(sorted(self.class_counts.items())),
            "errors": len(self.errors),
            "warnings": len(self.warnings),
            "issues": [i.__dict__ for i in self.issues],
        }


def _yolo_to_xyxy(b: np.ndarray) -> np.ndarray:
    cx, cy, w, h = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    return np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1)


def run_qc(
    images_dir: str,
    labels_dir: str,
    num_classes: Optional[int] = None,
    min_side_norm: float = 0.005,
    dup_iou: float = 0.95,
    imbalance_ratio: float = 10.0,
    check_images: bool = False,
) -> QCReport:
    images_dir_p, labels_dir_p = Path(images_dir), Path(labels_dir)
    rep = QCReport()
    images = sorted(p for p in images_dir_p.rglob("*") if p.suffix.lower() in IMG_EXT)
    labels = {p.stem: p for p in labels_dir_p.rglob("*.txt")}
    rep.n_images = len(images)
    image_stems = {p.stem for p in images}

    for stem, lp in labels.items():
        if stem not in image_stems:
            rep.issues.append(Issue("error", "orphan_label", str(lp), "label has no matching image"))

    for img in images:
        if check_images and cv2.imread(str(img)) is None:
            rep.issues.append(Issue("error", "corrupt_image", str(img), "image cannot be decoded"))
        lp = labels.get(img.stem)
        if lp is None:
            rep.n_background += 1
            rep.issues.append(Issue("info", "missing_label", str(img), "no label file (treated as background)"))
            continue
        rows, per_file_ok = [], True
        for ln, line in enumerate(lp.read_text().splitlines(), 1):
            if not line.strip():
                continue
            parts = line.split()
            try:
                if len(parts) != 5:
                    raise ValueError(f"expected 5 fields, got {len(parts)}")
                cls = int(parts[0])
                vals = [float(v) for v in parts[1:]]
            except ValueError as exc:
                rep.issues.append(Issue("error", "malformed_line", f"{lp}:{ln}", str(exc)))
                per_file_ok = False
                continue
            cx, cy, w, h = vals
            if cls < 0 or (num_classes is not None and cls >= num_classes):
                rep.issues.append(Issue("error", "bad_class_id", f"{lp}:{ln}", f"class id {cls} out of range"))
                per_file_ok = False
                continue
            if w <= 0 or h <= 0:
                rep.issues.append(Issue("error", "non_positive_box", f"{lp}:{ln}", f"w={w}, h={h}"))
                per_file_ok = False
                continue
            eps = 1e-4
            if cx - w / 2 < -eps or cy - h / 2 < -eps or cx + w / 2 > 1 + eps or cy + h / 2 > 1 + eps:
                rep.issues.append(Issue("error", "out_of_bounds", f"{lp}:{ln}", "box extends outside the image"))
                per_file_ok = False
                continue
            if w < min_side_norm or h < min_side_norm:
                rep.issues.append(Issue("warning", "tiny_box", f"{lp}:{ln}", f"w={w:.4f}, h={h:.4f} (normalised)"))
            rows.append((cls, cx, cy, w, h))
            rep.class_counts[cls] += 1
            rep.n_boxes += 1
        if not rows and per_file_ok:
            rep.n_background += 1
        # duplicate boxes: same class, near-identical extent
        if len(rows) > 1:
            arr = np.array(rows)
            for c in np.unique(arr[:, 0]):
                sub = arr[arr[:, 0] == c][:, 1:]
                if len(sub) < 2:
                    continue
                iou = iou_matrix(_yolo_to_xyxy(sub), _yolo_to_xyxy(sub))
                iu = np.triu(iou, k=1)
                n_dup = int((iu > dup_iou).sum())
                if n_dup:
                    rep.issues.append(Issue("warning", "duplicate_boxes", str(lp), f"{n_dup} duplicate box pair(s) for class {int(c)}"))

    if rep.class_counts:
        counts = dict(rep.class_counts)
        if num_classes is not None:
            for c in range(num_classes):
                if counts.get(c, 0) == 0:
                    rep.issues.append(Issue("warning", "empty_class", "dataset", f"class {c} has no instances"))
        present = [v for v in counts.values() if v > 0]
        if len(present) > 1 and max(present) / min(present) > imbalance_ratio:
            rep.issues.append(
                Issue("warning", "class_imbalance", "dataset", f"max/min class count = {max(present) / min(present):.1f} (> {imbalance_ratio})")
            )
    return rep


# ----------------------------------------------------------------------- auto-labelling / active learning
def write_yolo_labels(dets: Sequence[Detection], image_shape: Tuple[int, int], path: Path) -> None:
    H, W = image_shape[:2]
    lines = []
    for d in dets:
        x1, y1, x2, y2 = d.box
        lines.append(
            f"{d.cls_id} {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} {(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + ("\n" if lines else ""))


def uncertainty_score(dets: Sequence[Detection]) -> float:
    """Highest when the model's most ambiguous detection sits near confidence 0.5. Range [0, 1]."""
    if not dets:
        return 0.0
    return float(max(1.0 - 2.0 * abs(d.score - 0.5) for d in dets))


def select_for_review(scores: Dict[str, float], top_k: int) -> List[str]:
    return [k for k, _ in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_k]]


def pseudo_label(detector, images_dir: str, out_dir: str, conf_keep: float = 0.5) -> Dict[str, float]:
    """Auto-annotate every image. Keeps detections >= conf_keep as labels; returns per-image
    uncertainty computed over ALL detections the detector produced (so borderline ones count)."""
    scores: Dict[str, float] = {}
    for p in sorted(Path(images_dir).rglob("*")):
        if p.suffix.lower() not in IMG_EXT:
            continue
        img = cv2.imread(str(p))
        if img is None:
            continue
        dets = detector.detect(img)
        write_yolo_labels([d for d in dets if d.score >= conf_keep], img.shape, Path(out_dir) / f"{p.stem}.txt")
        scores[p.name] = uncertainty_score(dets)
    return scores
