#!/usr/bin/env python3
"""
bag_to_hdf5.py

Convert a ROS1 (.bag) or ROS2 (.db3 / directory) bag file to HDF5 without ROS installation package

Existing converters from github need ROS installation package. 
https://github.com/strawlab/bag2hdf5/blob/master/bag2hdf5
https://gitlab.com/rosbag-development-team/rosbag-hdf5

"sample run cmd: python .\scripts/bag_to_hdf5.py --bag "H:/GitHub/lidar-perception-pipeline/kitti-data/sample/test/test.bag.db3"
"""

import argparse
import datetime
import sys
import time
from pathlib import Path
from typing import Generator, Optional, Tuple

import numpy as np

import h5py                                          
from tqdm import tqdm                                
from rosbags.highlevel import AnyReader              
from rosbags.typesys import get_typestore, Stores    

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 


################################################################ 
# PointCloud2 → numpy                                          # 
################################################################ 
_DTYPE_MAP = {
    1: np.int8,   2: np.uint8,  3: np.int16,  4: np.uint16,
    5: np.int32,  6: np.uint32, 7: np.float32, 8: np.float64,
}


def pc2_to_numpy(msg) -> np.ndarray:
    """
    Decode a sensor_msgs/PointCloud2 message into an (N, 4) float32 array.
    Columns: x, y, z, intensity (0.0 if the field is absent).
    Works for both ROS1 and ROS2 message objects from rosbags.
    """
    fields = {f.name: f for f in msg.fields}
    n_points = int(msg.width * msg.height)
    if n_points == 0:
        return np.empty((0, 4), dtype=np.float32)

    point_step = int(msg.point_step)

    # Fast path: try structured dtype decode
    try:
        struct_parts = []
        for f in sorted(msg.fields, key=lambda x: x.offset):
            np_type = _DTYPE_MAP.get(int(f.datatype), np.float32)
            count = max(int(f.count), 1)
            name = f.name
            struct_parts.append((name, np_type, (count,) if count > 1 else ()))

        named_size = sum(
            np.dtype(_DTYPE_MAP.get(int(f.datatype), np.float32)).itemsize * max(int(f.count), 1)
            for f in msg.fields
        )
        if named_size < point_step:
            struct_parts.append(("_pad", np.uint8, (point_step - named_size,)))

        raw = bytes(msg.data)
        pc = np.frombuffer(raw, dtype=np.dtype(struct_parts))

        x         = pc["x"].astype(np.float32).ravel()
        y         = pc["y"].astype(np.float32).ravel()
        z         = pc["z"].astype(np.float32).ravel()
        intensity = (pc["intensity"].astype(np.float32).ravel()
                     if "intensity" in fields else np.zeros(n_points, np.float32))

    except Exception:
        # Slow fallback: manual field extraction
        raw = np.frombuffer(bytes(msg.data), dtype=np.uint8)
        x, y, z, intensity = (np.empty(n_points, np.float32) for _ in range(4))
        for i in range(n_points):
            base = i * point_step
            chunk = bytes(raw[base: base + point_step])
            def _f32(offset: int) -> float:
                return np.frombuffer(chunk[offset: offset + 4], np.float32)[0]
            x[i] = _f32(fields["x"].offset)
            y[i] = _f32(fields["y"].offset)
            z[i] = _f32(fields["z"].offset)
            intensity[i] = _f32(fields["intensity"].offset) if "intensity" in fields else 0.0

    cloud = np.stack([x, y, z, intensity], axis=1)
    return cloud[np.isfinite(cloud).all(axis=1)]   # drop NaN / Inf


################################################################ 
# Bag reader — yields (frame_idx, timestamp_sec, points)       # 
################################################################ 
def _detect_typestore(bag_path: Path):
    """Pick the right typestore for ROS1 vs ROS2 bags."""
    # AnyReader handles both; we still need a typestore for deserialization
    p = str(bag_path)
    if bag_path.is_dir() or p.endswith(".db3") or p.endswith(".mcap"):
        return get_typestore(Stores.ROS2_HUMBLE)
    return get_typestore(Stores.ROS1_NOETIC)


def iter_pointcloud2(
    bag_path: Path,
    topic: str,
    max_frames: Optional[int] = None,
) -> Generator[Tuple[int, float, np.ndarray], None, None]:
    """
    Yield (frame_index, timestamp_seconds, points_array) for every
    PointCloud2 message on `topic`.
    Works with both ROS1 and ROS2 bags via rosbags AnyReader.
    """
    typestore = _detect_typestore(bag_path)

    with AnyReader([bag_path]) as reader:
        connections = [c for c in reader.connections if c.topic == topic]
        if not connections:
            available = sorted({c.topic for c in reader.connections})
            print(f"\n[ERROR] Topic '{topic}' not found in bag.")
            print( "  Available topics:")
            for t in available:
                print(f"    {t}")
            print(f"\n  Re-run with --topic <one of the above>\n")
            sys.exit(1)

        frame_idx = 0
        for conn, timestamp_ns, rawdata in reader.messages(connections=connections):
            if max_frames is not None and frame_idx >= max_frames:
                break
            try:
                msg = reader.deserialize(rawdata, conn.msgtype)
                points = pc2_to_numpy(msg)
                yield frame_idx, timestamp_ns * 1e-9, points
                frame_idx += 1
            except Exception as exc:
                print(f"  [WARN] Frame {frame_idx} decode error: {exc}")
                frame_idx += 1


def count_messages(bag_path: Path, topic: str) -> int:
    """Count total PointCloud2 messages on topic without deserializing."""
    with AnyReader([bag_path]) as reader:
        return sum(1 for c in reader.connections if c.topic == topic
                   for _ in [c.msgcount])


def list_topics(bag_path: Path) -> list[tuple[str, str, int]]:
    """Return [(topic, msgtype, count), ...] sorted by topic name."""
    with AnyReader([bag_path]) as reader:
        return sorted(
            [(c.topic, c.msgtype, c.msgcount) for c in reader.connections],
            key=lambda x: x[0],
        )


################################################################ 
# HDF5 writer                                                  # 
################################################################ 

def convert(
    bag_path: Path,
    topic: str,
    output_path: Path,
    compress: bool = False,
    max_frames: Optional[int] = None,
) -> None:
    print(f"  Bag     : {bag_path}")
    print(f"  Topic   : {topic}")
    print(f"  Output  : {output_path}")
    print(f"  Compress: {'gzip level 4' if compress else 'none (fastest)'}")
    if max_frames:
        print(f"  Frames  : {max_frames} (limited)")
    print()

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # compression kwargs forwarded to h5py dataset creation
    comp_kwargs: dict = {"compression": "gzip", "compression_opts": 4} if compress else {}

    t_start = time.perf_counter()
    total_points = 0
    n_frames = 0

    with h5py.File(output_path, "w") as hf:
        frames_grp = hf.create_group("frames")

        #  Count messages first so tqdm shows a real ETA 
        print("Counting frames...", end=" ", flush=True)
        try:
            n_total = count_messages(bag_path, topic)
            if max_frames:
                n_total = min(n_total, max_frames)
            print(f"{n_total} frames found.")
        except Exception:
            n_total = None
            print("(unknown)")

        #  Main conversion loop 
        for frame_idx, ts_sec, points in tqdm(
            iter_pointcloud2(bag_path, topic, max_frames),
            total=n_total,
            desc="Converting",
            unit="frame",
        ):
            grp_name = f"{frame_idx:06d}"
            grp = frames_grp.create_group(grp_name)

            # Core dataset: (N, 4) float32  [x, y, z, intensity]
            grp.create_dataset(
                "points",
                data=points.astype(np.float32),
                dtype=np.float32,
                **comp_kwargs,
            )
            # Timestamp as a scalar attribute (fast to read without loading points)
            grp.attrs["timestamp"] = ts_sec
            grp.attrs["n_points"]  = points.shape[0]

            total_points += points.shape[0]
            n_frames += 1

        #  Metadata group 
        meta = hf.create_group("metadata")
        meta.attrs["topic"]                  = topic
        meta.attrs["n_frames"]               = n_frames
        meta.attrs["bag_path"]               = str(bag_path)
        meta.attrs["converted_at"]           = datetime.datetime.utcnow().isoformat()
        meta.attrs["total_points"]           = total_points
        meta.attrs["mean_points_per_frame"]  = total_points / max(n_frames, 1)
        meta.attrs["field_names"]            = ["x", "y", "z", "intensity"]
        meta.attrs["compression"]            = "gzip" if compress else "none"

    elapsed   = time.perf_counter() - t_start
    file_mb   = output_path.stat().st_size / 1e6

    print(f" Done!")
    print(f"  Frames written    : {n_frames:,}")
    print(f"  Total points      : {total_points:,}")
    print(f"  Avg pts / frame   : {total_points // max(n_frames,1):,}")
    print(f"  HDF5 file size    : {file_mb:.1f} MB")
    print(f"  Time              : {elapsed:.1f}s  ({n_frames/elapsed:.1f} frames/s)")
    print(f"  Output            : {output_path.resolve()}")


################################################################ 
# --list-topics helper                                         # 
################################################################ 

def cmd_list_topics(bag_path: Path) -> None:
    print(f"\nTopics in: {bag_path}\n")
    print(f"  {'Topic':<45} {'Type':<45} {'Msgs':>6}")
    print(f"  {'-'*45} {'-'*45} {'-'*6}")
    for topic, msgtype, count in list_topics(bag_path):
        print(f"  {topic:<45} {msgtype:<45} {count:>6}")
    print()


################################################################ 
# CLI                                                          # 
################################################################ 

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert ROS bag → HDF5  (no ROS needed)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--bag",         required=True, help="Input .bag / .db3 / bag directory")
    p.add_argument("--topic",       default="/velodyne_points")
    p.add_argument("--output",      default=None,  help="Output .h5 path (default: same name as bag)")
    p.add_argument("--compress",    action="store_true", help="Enable gzip compression (level 4)")
    p.add_argument("--max-frames",  type=int, default=None)
    p.add_argument("--list-topics", action="store_true", help="Print all topics and exit")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    bag_path = Path(args.bag)

    if not bag_path.exists():
        print(f"[ERROR] Bag not found: {bag_path}")
        sys.exit(1)

    if args.list_topics:
        cmd_list_topics(bag_path)
        sys.exit(0)

    output = Path(args.output) if args.output else bag_path.with_suffix(".h5")

    convert(
        bag_path=bag_path,
        topic=args.topic,
        output_path=output,
        compress=args.compress,
        max_frames=args.max_frames,
    )
