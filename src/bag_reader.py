"""
bag_reader.py

Read sensor_msgs/PointCloud2 messages from ROS1 or ROS2 bag files.
Uses the pure-Python `rosbags` library without ROS installation needed.

Supports:
  - ROS1  .bag files (rosbag format - to be tested)
  - ROS2  .db3 / directory-based bags

"""

from __future__ import annotations

from pathlib import Path
from typing import Generator, Optional, Tuple

import numpy as np


################################################################ 
# PointCloud2 field decoder                                    # 
################################################################ 

_DTYPE_MAP = {
    1: np.int8,
    2: np.uint8,
    3: np.int16,
    4: np.uint16,
    5: np.int32,
    6: np.uint32,
    7: np.float32,
    8: np.float64,
}


def _pc2_to_numpy(msg) -> np.ndarray:
    """
    Convert a sensor_msgs/PointCloud2 message to an (N, 4) float32 array.
    Columns: x, y, z, intensity (intensity set to 0 if field absent).
    """
    fields = {f.name: f for f in msg.fields}
    point_step = msg.point_step
    data = np.frombuffer(msg.data, dtype=np.uint8)

    n_points = msg.width * msg.height
    if n_points == 0:
        return np.empty((0, 4), dtype=np.float32)

    # Build structured dtype from fields
    struct_dtype = []
    for f in msg.fields:
        np_type = _DTYPE_MAP.get(f.datatype, np.float32)
        struct_dtype.append((f.name, np_type, (f.count,) if f.count > 1 else ()))
    # generates -> [("x", float32), ("y", float32), ("z", float32), ("intensity", float32)]

    # Pad to point_step
    total_named = sum(
        np.dtype(_DTYPE_MAP.get(f.datatype, np.float32)).itemsize * max(f.count, 1)
        for f in msg.fields
    )
    if total_named < point_step:
        struct_dtype.append(("_pad", np.uint8, (point_step - total_named,)))

    # vectorized and zero-copy
    try:
        pc = np.frombuffer(bytes(msg.data), dtype=np.dtype(struct_dtype))
    except Exception:
        # Fallback: manual extraction
        pc = None

    # if structured dtype fails
    if pc is not None:
        x = pc["x"].astype(np.float32)
        y = pc["y"].astype(np.float32)
        z = pc["z"].astype(np.float32)
        intensity = pc["intensity"].astype(np.float32) if "intensity" in fields else np.zeros(n_points, dtype=np.float32)
    else:
        # Slow fallback path
        x, y, z, intensity = [], [], [], []
        for i in range(n_points):
            base = i * point_step
            chunk = bytes(data[base : base + point_step])
            xv = np.frombuffer(chunk[fields["x"].offset : fields["x"].offset + 4], np.float32)[0]
            yv = np.frombuffer(chunk[fields["y"].offset : fields["y"].offset + 4], np.float32)[0]
            zv = np.frombuffer(chunk[fields["z"].offset : fields["z"].offset + 4], np.float32)[0]
            iv = (
                np.frombuffer(chunk[fields["intensity"].offset : fields["intensity"].offset + 4], np.float32)[0]
                if "intensity" in fields
                else 0.0
            )
            x.append(xv); y.append(yv); z.append(zv); intensity.append(iv)
        x, y, z, intensity = map(np.array, [x, y, z, intensity])

    # stack into array as (N,4)
    cloud = np.stack([x, y, z, intensity], axis=1)

    # Remove NaN / Inf
    valid = np.isfinite(cloud).all(axis=1)
    return cloud[valid].astype(np.float32)

################################################################ 
# BagReader                                                    # 
################################################################ 

class BagReader:
    """
    Read LiDAR point clouds from a ROS1 or ROS2 bag file.

    Parameters
    ----------
    bag_path : str | Path
    topic    : str   — PointCloud2 topic name
    max_frames : int | None — limit number of frames (None = all)
    start_frame : int — skip this many frames at the start
    """

    def __init__(
        self,
        bag_path: str | Path,
        topic: str = "/velodyne_points",
        max_frames: Optional[int] = None,
        start_frame: int = 0,
    ):
        self.bag_path = Path(bag_path)
        self.topic = topic
        self.max_frames = max_frames
        self.start_frame = start_frame
        self._format: str = self._detect_format()

    # Public 
    def iter_frames(self) -> Generator[Tuple[int, float, np.ndarray], None, None]: 
        """
        return: 
        Yield (frame_index (int), timestamp_sec(float), points_array) for every
        PointCloud2 message on the configured topic.
        points_array : np.ndarray shape (N, 4) — x, y, z, intensity
        """
        if self._format == "ros1":
            yield from self._iter_ros1()
        elif self._format == "ros2":
            yield from self._iter_ros2()
        else:
            raise ValueError(f"Unsupported bag format at {self.bag_path}")

    # def frame_count(self) -> int:
    #     """Return total number of PointCloud2 messages on topic."""
    #     count = 0
    #     if self._format == "ros1":
    #         try:
    #             from rosbags.rosbag1 import Reader
    #             with Reader(self.bag_path) as bag:
    #                 for conn in bag.connections:
    #                     if conn.topic == self.topic:
    #                         count = conn.msgcount
    #                         break
    #         except Exception:
    #             pass
    #     return count

    def topics(self) -> list[str]:
        """List all topics in the bag file based on file type."""
        topics = []
        try:
            if self._format == "ros1":
                from rosbags.rosbag1 import Reader
                with Reader(self.bag_path) as bag:
                    topics = list({c.topic for c in bag.connections})
            else:
                from rosbags.rosbag2 import Reader
                with Reader(self.bag_path) as bag:
                    topics = list({c.topic for c in bag.connections})
        except Exception as e:
            print("Could not list topics:", e)
        return sorted(topics)

    # Private helpers 
    def _detect_format(self) -> str:
        """ detect the format as ros2/ros1 and accordingly import rosbag1/rosbag2 """
        p = self.bag_path
        if p.is_dir():
            return "ros2"
        if p.suffix == ".bag":
            return "ros1"
        if p.suffix == ".db3":
            return "ros2"
        return "ros1"

    # untested
    def _iter_ros1(self) -> Generator[Tuple[int, float, np.ndarray], None, None]:
        """ iterate the data and yield -> frame id, timestamp and point cloud"""

        from rosbags.rosbag1 import Reader
        from rosbags.typesys import get_typestore, Stores

        typestore = get_typestore(Stores.ROS1_NOETIC)

        frame_idx = 0
        emitted = 0

        with Reader(self.bag_path) as bag:
            connections = [c for c in bag.connections if c.topic == self.topic]
            if not connections:
                available = [c.topic for c in bag.connections]
                raise RuntimeError(
                    f"Topic '{self.topic}' not found. "
                    f"Available: {available}"
                )

            for conn, timestamp, rawdata in bag.messages(connections=connections):
                if frame_idx < self.start_frame:
                    frame_idx += 1
                    continue
                if self.max_frames is not None and emitted >= self.max_frames:
                    break

                try:
                    msg = typestore.deserialize_ros1(rawdata, conn.msgtype)
                    points = _pc2_to_numpy(msg)
                    ts_sec = timestamp * 1e-9
                    yield frame_idx, ts_sec, points
                    emitted += 1
                except Exception as exc:
                    print("Frame \n", frame_idx)
                    print("decode error:", exc)
                finally:
                    frame_idx += 1


    def _iter_ros2(self) -> Generator[Tuple[int, float, np.ndarray], None, None]:
        """ iterate the data return yield -> frame id, timestamp and point cloud"""

        from rosbags.rosbag2 import Reader
        from rosbags.typesys import get_typestore, Stores

        typestore = get_typestore(Stores.ROS2_HUMBLE)

        frame_idx = 0
        emitted = 0

        # read the bag and check for topic 
        with Reader(self.bag_path) as bag:
            connections = [c for c in bag.connections if c.topic == self.topic]
            if not connections:
                available = [c.topic for c in bag.connections]
                raise RuntimeError(
                    f"Topic '{self.topic}' not found. Available: {available}"
                )

            # iterate through messages with particular connection & topic id
            for conn, timestamp, rawdata in bag.messages(connections=connections):
                if frame_idx < self.start_frame:
                    frame_idx += 1
                    continue
                if self.max_frames is not None and emitted >= self.max_frames:
                    break

                try:
                    msg = typestore.deserialize_cdr(rawdata, conn.msgtype)
                    points = _pc2_to_numpy(msg)
                    ts_sec = timestamp * 1e-9 # convert to sec
                    yield frame_idx, ts_sec, points
                    emitted += 1
                except Exception as exc:
                    print("Frame \n", frame_idx)
                    print("decode error:", exc)
                finally:
                    frame_idx += 1
