from __future__ import annotations

import time
from collections import defaultdict
from contextlib import contextmanager
from typing import Dict, List

import numpy as np


class StageTimer:
    """Collects per-stage wall-clock latencies (ms) so every run yields a benchmark for free."""

    def __init__(self) -> None:
        self._t: Dict[str, List[float]] = defaultdict(list)

    @contextmanager
    def measure(self, stage: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self._t[stage].append((time.perf_counter() - t0) * 1000.0)

    def summary(self) -> Dict[str, Dict[str, float]]:
        out = {}
        for stage, v in self._t.items():
            a = np.asarray(v)
            out[stage] = {
                "mean_ms": round(float(a.mean()), 3),
                "p50_ms": round(float(np.percentile(a, 50)), 3),
                "p95_ms": round(float(np.percentile(a, 95)), 3),
            }
        return out
