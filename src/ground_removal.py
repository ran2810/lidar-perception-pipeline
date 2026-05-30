"""
ground_removal.py
=================
Separate ground from obstacles — pure NumPy RANSAC

Methods
-------
  ransac            — robust plane fit; handles slightly sloped roads
  height_threshold  — fast z-cutoff fallback
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class GroundConfig:
    method: str = "ransac"
    distance_threshold: float = 0.20   # metres: points within → ground
    max_iterations: int = 100
    height_threshold: float = -1.6     # used only with method="height_threshold"
    min_normal_z: float = 0.85         # sanity check: normal must point upward


PlaneModel = Tuple[float, float, float, float]   # (a, b, c, d)  ax+by+cz+d=0


class GroundRemover:
    """
    Remove ground plane from a preprocessed point cloud.
    """

    def __init__(self, config: GroundConfig | dict | None = None):
        if config is None:
            self.cfg = GroundConfig()
        elif isinstance(config, dict):
            self.cfg = GroundConfig(**config)
        else:
            self.cfg = config

    def remove(
        self, points: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, PlaneModel]:
        if points.shape[0] < 10:
            return (
                np.empty((0, points.shape[1]), dtype=np.float32),
                points,
                (0.0, 0.0, 1.0, 0.0),
            )
        if self.cfg.method == "ransac":
            return self._ransac(points)
        return self._height_threshold(points)

    #  RANSAC method
    def _ransac(self, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, PlaneModel]:
        """
        Fit a plane to `points` with RANSAC.

        Algorithm
        ---------
        For each iteration:
          1. Pick 3 random points -> compute plane normal via cross product
          2. Count inliers (|signed distance| ≤ threshold)
          3. Keep the plane with the most inliers
        After finding the best plane, refit with all inliers (least-squares)
        for a cleaner estimate.
        """
        xyz = points[:, :3].astype(np.float64)
        n = len(xyz)
        thr = self.cfg.distance_threshold
        best_inliers = np.empty(0, dtype=np.int64)
        best_plane: PlaneModel = (0.0, 0.0, 1.0, 0.0)
        rng = np.random.default_rng(42)

        for _ in range(self.cfg.max_iterations):
            # pick 3 random points
            idx = rng.choice(n, 3, replace=False)
            p0, p1, p2 = xyz[idx[0]], xyz[idx[1]], xyz[idx[2]]

            # Plane normal via cross product
            v1 = p1 - p0
            v2 = p2 - p0
            normal = np.cross(v1, v2)
            norm_len = np.linalg.norm(normal)
            if norm_len < 1e-8:
                continue                    # degenerate sample
            normal /= norm_len

            # Reject planes whose normal is not mostly vertical
            if abs(normal[2]) < self.cfg.min_normal_z:
                continue

            d = -np.dot(normal, p0)
            # compute inliers
            dists = np.abs(xyz @ normal + d)
            inliers = np.where(dists <= thr)[0]

            # keep the best plane
            if len(inliers) > len(best_inliers):
                best_inliers = inliers
                best_plane = (normal[0], normal[1], normal[2], d)

        # Least-squares refit on all inliers for a cleaner plane
        if len(best_inliers) >= 3:
            best_plane = self._lstsq_plane(xyz[best_inliers])

        # Final labelling
        a, b, c, d = best_plane
        norm_len = np.sqrt(a*a + b*b + c*c) + 1e-8
        dists = np.abs(xyz[:, 0]*a + xyz[:, 1]*b + xyz[:, 2]*c + d) / norm_len
        ground_mask = dists <= thr

        # Sanity: if RANSAC failed (too few ground pts), fall back
        if ground_mask.sum() < 50:
            print("RANSAC found very few ground points; falling back to height threshold")
            return self._height_threshold(points)

        ground    = points[ground_mask]
        obstacles = points[~ground_mask]
        print(f"RANSAC: {ground.shape[0]}  ground /  {obstacles.shape[0]} obstacle | plane=( {a} ,{b}, {c}, {d})")

        return ground, obstacles, best_plane

    @staticmethod
    def _lstsq_plane(xyz: np.ndarray) -> PlaneModel:
        """
        Least-squares plane fit: minimise ||Ax - b||
        Plane: z = px*x + py*y + d  → rearranged to ax+by+cz+d=0 form.
        """
        try:
            A = np.c_[xyz[:, 0], xyz[:, 1], np.ones(len(xyz))]
            b = xyz[:, 2]
            result, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            px, py, d = result
            # Convert to ax+by+cz+d=0  with c=-1
            # px*x + py*y - z + d = 0
            normal = np.array([px, py, -1.0])
            norm_len = np.linalg.norm(normal) + 1e-8
            a, b_, c = normal / norm_len
            return (float(a), float(b_), float(c), float(d / norm_len))
        except Exception:
            return (0.0, 0.0, 1.0, 0.0)

    # Height threshold 
    def _height_threshold(
        self, points: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, PlaneModel]:
        mask = points[:, 2] <= self.cfg.height_threshold
        return (
            points[mask],
            points[~mask],
            (0.0, 0.0, 1.0, -self.cfg.height_threshold),
        )

    # # Util func
    # @staticmethod
    # def point_to_plane_dist(points: np.ndarray, plane: PlaneModel) -> np.ndarray:
    #     a, b, c, d = plane
    #     norm = np.sqrt(a*a + b*b + c*c) + 1e-8
    #     return (points[:, 0]*a + points[:, 1]*b + points[:, 2]*c + d) / norm
