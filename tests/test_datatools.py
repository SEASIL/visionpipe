import cv2
import numpy as np

from tests.helpers import make_det
from visionpipe.datatools import pseudo_label, run_qc, select_for_review, uncertainty_score


def _mk(tmp_path, labels: dict, n_images=None):
    (tmp_path / "images").mkdir()
    (tmp_path / "labels").mkdir()
    names = list(labels) if n_images is None else [f"img{i}" for i in range(n_images)]
    for n in names:
        cv2.imwrite(str(tmp_path / "images" / f"{n}.jpg"), np.zeros((64, 64, 3), np.uint8))
    for n, text in labels.items():
        (tmp_path / "labels" / f"{n}.txt").write_text(text)
    return str(tmp_path / "images"), str(tmp_path / "labels")


def test_clean_dataset_has_no_errors(tmp_path):
    imgs, lbls = _mk(tmp_path, {"a": "0 0.5 0.5 0.2 0.3\n1 0.2 0.2 0.1 0.1\n", "b": "0 0.5 0.5 0.4 0.4\n"})
    rep = run_qc(imgs, lbls, num_classes=2)
    assert rep.errors == [] and rep.n_boxes == 3 and rep.class_counts == {0: 2, 1: 1}


def test_detects_common_labelling_problems(tmp_path):
    imgs, lbls = _mk(
        tmp_path,
        {
            "bad_fmt": "0 0.5 0.5 0.2\n",  # 4 fields
            "bad_cls": "7 0.5 0.5 0.2 0.2\n",  # class out of range
            "oob": "0 0.95 0.5 0.3 0.2\n",  # extends past right edge
            "zero": "0 0.5 0.5 0 0.2\n",
            "tiny": "0 0.5 0.5 0.001 0.001\n",
            "dup": "0 0.5 0.5 0.2 0.2\n0 0.5 0.5 0.2 0.2\n",
        },
    )
    (tmp_path / "labels" / "orphan.txt").write_text("0 0.5 0.5 0.2 0.2\n")
    rep = run_qc(imgs, lbls, num_classes=2)
    codes = {i.code for i in rep.errors}
    assert {"malformed_line", "bad_class_id", "out_of_bounds", "non_positive_box", "orphan_label"} <= codes
    wcodes = {i.code for i in rep.warnings}
    assert {"tiny_box", "duplicate_boxes", "empty_class"} <= wcodes


def test_class_imbalance_and_background(tmp_path):
    lab = {f"i{k}": "0 0.5 0.5 0.2 0.2\n" * 12 for k in range(2)}
    lab["rare"] = "1 0.5 0.5 0.2 0.2\n"
    imgs, lbls = _mk(tmp_path, lab)
    cv2.imwrite(str(tmp_path / "images" / "bg.jpg"), np.zeros((64, 64, 3), np.uint8))  # no label -> background
    rep = run_qc(imgs, lbls, num_classes=2)
    assert rep.n_background == 1
    assert "class_imbalance" in {i.code for i in rep.warnings}


def test_uncertainty_prefers_borderline_confidence():
    sure = [make_det(10, 10, score=0.99)]
    unsure = [make_det(10, 10, score=0.5)]
    assert uncertainty_score(unsure) > uncertainty_score(sure) > uncertainty_score([])
    assert select_for_review({"a": 0.1, "b": 0.9, "c": 0.5}, 2) == ["b", "c"]


class _FixedDetector:
    def detect(self, image):
        return [make_det(32, 32, w=20, h=20, score=0.9), make_det(10, 10, w=8, h=8, score=0.45)]


def test_pseudo_label_writes_yolo_files_and_scores(tmp_path):
    imgs, _ = _mk(tmp_path, {}, n_images=2)
    scores = pseudo_label(_FixedDetector(), imgs, str(tmp_path / "out"), conf_keep=0.5)
    assert set(scores) == {"img0.jpg", "img1.jpg"}
    lines = (tmp_path / "out" / "img0.txt").read_text().strip().splitlines()
    assert len(lines) == 1  # the 0.45 detection is below conf_keep
    cls, cx, cy, w, h = lines[0].split()
    assert cls == "0" and float(cx) == 0.5 and float(w) == 0.3125
    assert scores["img0.jpg"] > 0.8  # the 0.45 detection is very uncertain -> high review priority
