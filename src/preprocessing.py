"""
preprocessing.py -point cloud preprocessing

"""

from __future__ import annotations # forward reference

from dataclasses import dataclass
from typing import Tuple

import numpy as np


# @dataclass(frozen=True) - to avoid modification
@dataclass
class PreprocessConfig:
    voxel_size: float = 0.10                        # metres; but smaller = more detail but becoming slower
    roi_x: Tuple[float, float] = (-40.0, 40.0)      # forward / backward (m)
    roi_y: Tuple[float, float] = (-20.0, 20.0)      # left / right (m)
    roi_z: Tuple[float, float] = (-3.0, 5.0)        # up / down (m)
    min_intensity: float = 0.02                     # drop sensor noise (0–255 scale)
    remove_ego_sphere: bool = True                  # remove points inside vehicle body
    ego_radius: float = 1.5                         # metres
    voxel_use_centroids: bool = False               # use centroids for the voxel representation point


class Preprocessor:
    """
    Apply a configurable preprocessing pipeline to raw point clouds.

    """

    def __init__(self, config: PreprocessConfig | dict | None = None):
        if config is None:
            self.cfg = PreprocessConfig()
        elif isinstance(config, dict):
            self.cfg = PreprocessConfig(**config)
        else:
            self.cfg = config
        

    def process(self, points: np.ndarray) -> np.ndarray:
        """Run the full preprocessing pipeline."""
        if points.shape[0] == 0:
            return points

        pts = points.copy()
        pts = self._remove_nan(pts)
        if self.cfg.remove_ego_sphere:
            pts = self._remove_ego(pts)
        pts = self._roi_crop(pts)
        pts = self._intensity_filter(pts)
        pts = self._voxel_downsample(pts)

        return pts
 
    # Steps 

    # remove Nan entries
    @staticmethod
    def _remove_nan(points: np.ndarray) -> np.ndarray:
        return points[np.isfinite(points[:, :3]).all(axis=1)]

    # remove point cloud from ego sphere
    def _remove_ego(self, points: np.ndarray) -> np.ndarray:
        r = self.cfg.ego_radius
        dist2 = (points[:, 0] ** 2 + points[:, 1] ** 2 + points[:, 2] ** 2)
        return points[dist2 > r * r]

     # crop roi
    def _roi_crop(self, points: np.ndarray) -> np.ndarray:
        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        mask = (
            (x >= self.cfg.roi_x[0]) & (x <= self.cfg.roi_x[1])
            & (y >= self.cfg.roi_y[0]) & (y <= self.cfg.roi_y[1])
            & (z >= self.cfg.roi_z[0]) & (z <= self.cfg.roi_z[1])
        )
        return points[mask]

    # filter for intensity for lanes, objects 
    def _intensity_filter(self, points: np.ndarray) -> np.ndarray:
        if points.shape[1] < 4:
            return points
        return points[points[:, 3] >= self.cfg.min_intensity]

    def _voxel_downsample(self, points: np.ndarray) -> np.ndarray:
        """
        voxel grid downsampling - reduce dense pcd into voxels(3d cube) - keep one representative point per voxel
        avoid open3d due to GPU usage. Complexity: O(N log N)
        Each voxel keeps the point closest to the voxel centroid.
        """

        # early exit
        if self.cfg.voxel_size <= 0 or points.shape[0] == 0:
            return points

        vs = self.cfg.voxel_size
        xyz = points[:, :3]

        # Assign each point to a voxel index (integer grid coordinates)
        voxel_coords = np.floor(xyz / vs).astype(np.int32)

        # Shift to positive range first to avoid sign issues
        mins = voxel_coords.min(axis=0)
        shifted = voxel_coords - mins

        # Encode (ix, iy, iz) as a single int64 key for fast grouping
        span = shifted.max(axis=0) + 1
        keys = (shifted[:, 0].astype(np.int64) * span[1] * span[2]
                + shifted[:, 1].astype(np.int64) * span[2]
                + shifted[:, 2].astype(np.int64))

        # Sort by key so equal-voxel points are contiguous
        order = np.argsort(keys, kind="stable")
        sorted_keys = keys[order]
        sorted_points = points[order]

        # use centroids to align with open3d in-built function -> open3d.geometry.PointCloud.voxel_down_sample()
        # more objects are detected -> therefore by default its false
        if self.cfg.voxel_use_centroids:
            print("Using centroids for voxel down sampling")
            unique_keys, inverse, counts = np.unique(
                keys, return_inverse=True, return_counts=True)
            num_voxels = unique_keys.shape[0]
            dim = points.shape[1]  # e.g. x,y,z,(intensity,...)
            
            # Accumulate sums per voxel
            sums = np.zeros((num_voxels, dim), dtype=points.dtype)
            np.add.at(sums, inverse, points)

            # Compute centroids
            centroids = sums / counts[:, None]
            return centroids
        else:
            # find first index of each voxel
            _, first_indices = np.unique(sorted_keys, return_index=True)
            
            # Map back to original point indices
            representative_indices = order[first_indices]
            return points[representative_indices]
