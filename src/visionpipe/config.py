from __future__ import annotations

import copy
from typing import Any, Dict, Optional

import yaml

from .detect import build_detector
from .events import EventEngine, build_rules
from .io import VideoSource
from .pipeline import Outputs, Pipeline
from .track import ByteTracker

DEFAULTS: Dict[str, Any] = {
    "camera_id": "cam0",
    "source": {"uri": None, "loop": False, "reconnect": True},
    "detector": {"type": "motion"},
    "tracker": {},
    "rules": [],
    "output": {"events_jsonl": None, "metadata_jsonl": None, "mot_txt": None, "video": None},
}


def deep_merge(base: Dict[str, Any], over: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in over.items():
        out[k] = deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_config(path: Optional[str] = None, overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    cfg = copy.deepcopy(DEFAULTS)
    if path:
        with open(path) as f:
            cfg = deep_merge(cfg, yaml.safe_load(f) or {})
    if overrides:
        cfg = deep_merge(cfg, overrides)
    return cfg


def build_pipeline(cfg: Dict[str, Any], detector=None, reid=None, show: bool = False) -> Pipeline:
    """Assemble a Pipeline from a config dict. A detector can be injected (shared across cameras)."""
    cam = cfg["camera_id"]
    src = cfg["source"]
    if src.get("uri") is None:
        raise ValueError("No video source. Set source.uri in the config or pass --source.")
    source = VideoSource(src["uri"], camera_id=cam, loop=src.get("loop", False), reconnect=src.get("reconnect", True))
    detector = detector or build_detector(cfg["detector"])
    tracker = ByteTracker(**cfg.get("tracker", {}))
    engine = EventEngine(build_rules(cfg.get("rules", [])))
    outputs = Outputs(**{k: v.format(camera_id=cam) for k, v in cfg.get("output", {}).items() if v})
    return Pipeline(source, detector, tracker, engine, outputs, reid=reid, show=show)
