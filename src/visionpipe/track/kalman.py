"""Constant-velocity Kalman filter over a bounding box.

State  x = [cx, cy, w, h, vx, vy, vw, vh]
Meas.  z = [cx, cy, w, h]
Noise is scaled by the box height, following the DeepSORT / ByteTrack convention,
so the filter behaves consistently for near and far objects.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np


class KalmanBoxFilter:
    def __init__(self, std_pos: float = 1 / 20, std_vel: float = 1 / 160):
        self.std_pos = std_pos
        self.std_vel = std_vel
        self.F = np.eye(8)
        for i in range(4):
            self.F[i, i + 4] = 1.0  # dt = 1 frame
        self.H = np.eye(4, 8)

    def initiate(self, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        h = max(float(z[3]), 1.0)
        mean = np.concatenate([z, np.zeros(4)])
        std = np.array([2 * self.std_pos * h] * 4 + [10 * self.std_vel * h] * 4)
        return mean, np.diag(np.square(std))

    def predict(self, mean: np.ndarray, cov: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        h = max(float(mean[3]), 1.0)
        std = np.array([self.std_pos * h] * 4 + [self.std_vel * h] * 4)
        Q = np.diag(np.square(std))
        mean = self.F @ mean
        cov = self.F @ cov @ self.F.T + Q
        mean[2:4] = np.maximum(mean[2:4], 1e-3)  # keep w, h positive
        return mean, cov

    def update(self, mean: np.ndarray, cov: np.ndarray, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        h = max(float(mean[3]), 1.0)
        R = np.diag(np.square([self.std_pos * h] * 4))
        S = self.H @ cov @ self.H.T + R
        K = np.linalg.solve(S, self.H @ cov).T  # = cov H^T S^-1 (cov symmetric)
        mean = mean + K @ (z - self.H @ mean)
        cov = (np.eye(8) - K @ self.H) @ cov
        mean[2:4] = np.maximum(mean[2:4], 1e-3)
        return mean, cov
