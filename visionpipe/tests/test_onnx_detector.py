"""Verifies letterbox -> decode -> undo-letterbox -> class-aware NMS against a tiny synthetic ONNX model
whose output is a known constant tensor (so no real weights are needed)."""
import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

from visionpipe.detect import build_detector
from visionpipe.detect.onnx_yolo import ONNXYoloDetector, letterbox


def _make_model(path, preds: np.ndarray, names="{0: 'person', 1: 'car'}"):
    """preds: (4 + nc, anchors) in letterboxed 640x640 coordinates (cx, cy, w, h, scores...)."""
    out = numpy_helper.from_array(preds[None].astype(np.float32), name="const")
    node = helper.make_node("Constant", [], ["output0"], value=out)
    inp = helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 640, 640])
    outp = helper.make_tensor_value_info("output0", TensorProto.FLOAT, list(preds[None].shape))
    graph = helper.make_graph([node], "fake_yolo", [inp], [outp])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    meta = model.metadata_props.add()
    meta.key, meta.value = "names", names
    onnx.save(model, str(path))


@pytest.fixture()
def model_path(tmp_path):
    #            cx     cy    w     h    person car
    cols = [
        (320, 320, 100, 200, 0.90, 0.05),  # person
        (324, 322, 100, 200, 0.80, 0.05),  # near-duplicate of the person -> must be removed by NMS
        (100, 200, 50, 50, 0.10, 0.70),  # car
        (500, 500, 40, 40, 0.10, 0.10),  # below confidence threshold
    ]
    p = tmp_path / "fake.onnx"
    _make_model(p, np.array(cols, dtype=np.float32).T)
    return str(p)


def test_letterbox_geometry():
    img = np.zeros((480, 640, 3), np.uint8)
    out, r, left, top = letterbox(img, (640, 640))
    assert out.shape == (640, 640, 3) and r == 1.0 and left == 0 and top == 80


def test_detect_end_to_end(model_path):
    det = ONNXYoloDetector(model_path, conf=0.25, iou=0.5)
    assert det.names == {0: "person", 1: "car"}  # read from ONNX metadata
    assert det.providers[0] == "CPUExecutionProvider"
    dets = det.detect(np.zeros((480, 640, 3), np.uint8))
    assert sorted(d.cls_name for d in dets) == ["car", "person"]  # duplicate suppressed, weak one dropped
    person = next(d for d in dets if d.cls_name == "person")
    car = next(d for d in dets if d.cls_name == "car")
    # boxes are mapped back from the letterboxed 640x640 space to the 480x640 image (top padding = 80)
    assert np.allclose(person.box, [270, 140, 370, 340], atol=1)
    assert np.allclose(car.box, [75, 95, 125, 145], atol=1)
    assert person.score == pytest.approx(0.9, abs=1e-3)


def test_class_filter_by_name(model_path):
    det = build_detector({"weights": model_path, "classes": ["car"]})
    dets = det.detect(np.zeros((480, 640, 3), np.uint8))
    assert [d.cls_name for d in dets] == ["car"]


def test_unknown_class_name_raises(model_path):
    with pytest.raises(ValueError):
        ONNXYoloDetector(model_path, classes=["dragon"])
