#!/usr/bin/env python3
"""
kitti_to_bag.py

Convert a KITTI Raw Data sequence to a ROS2 bag (.db3) WITHOUT needing
any ROS installation. 

sample cmd: python .\scripts\kitti_to_bag.py --kitti-dir H:\GitHub\lidar-perception-pipeline\kitti-data --date 2011_10_03 --drive 0027 --output kitti-data\sample\test1.bag  

"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Optional

import numpy as np

import pykitti                                       
from tqdm import tqdm                                
from rosbags.rosbag2 import Writer                   
from rosbags.typesys import get_typestore, Stores    

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

################################################################ 
# Type store (ROS2 Humble message definitions)                                                          # 
################################################################ 
TYPESTORE = get_typestore(Stores.ROS2_HUMBLE)

# Convenient aliases into the type store
_PC2    = TYPESTORE.types["sensor_msgs/msg/PointCloud2"]
_Field  = TYPESTORE.types["sensor_msgs/msg/PointField"]
_Imu    = TYPESTORE.types["sensor_msgs/msg/Imu"]
_Fix    = TYPESTORE.types["sensor_msgs/msg/NavSatFix"]
_Header = TYPESTORE.types["std_msgs/msg/Header"]
_Stamp  = TYPESTORE.types["builtin_interfaces/msg/Time"]
_Quat   = TYPESTORE.types["geometry_msgs/msg/Quaternion"]
_Vec3   = TYPESTORE.types["geometry_msgs/msg/Vector3"]
_Covariance = lambda: np.zeros(9, dtype=np.float64)  


################################################################ 
# Helper: build a Header                                       # 
################################################################ 
def _make_header(timestamp_ns: int, frame_id: str) -> object:
    secs  = int(timestamp_ns // 1_000_000_000)
    nsecs = int(timestamp_ns %  1_000_000_000)
    return _Header(
        stamp=_Stamp(sec=secs, nanosec=nsecs),
        frame_id=frame_id,
    )


################################################################ 
# PointCloud2 builder                                          # 
################################################################ 
# KITTI Velodyne .bin layout: float32 x4 → [x, y, z, intensity]
_POINT_STEP = 16          # 4 fields × 4 bytes
_FIELD_DEFS = [
    ("x",         0,  7, 1),   # float32
    ("y",         4,  7, 1),
    ("z",         8,  7, 1),
    ("intensity", 12, 7, 1),
]

def _build_fields() -> list:
    return [
        _Field(name=name, offset=offset, datatype=dtype, count=count)
        for name, offset, dtype, count in _FIELD_DEFS
    ]

_CACHED_FIELDS = _build_fields()


def make_pointcloud2(points: np.ndarray, timestamp_ns: int) -> object:
    """
    Build a sensor_msgs/PointCloud2 from an (N, 4) float32 array.
    points columns: x, y, z, intensity
    """
    n = len(points)
    data_bytes = points.astype(np.float32).tobytes()
    return _PC2(
        header=_make_header(timestamp_ns, "velodyne"),
        height=1,
        width=n,
        fields=_CACHED_FIELDS,
        is_bigendian=False,
        point_step=_POINT_STEP,
        row_step=n * _POINT_STEP,
        data=np.frombuffer(data_bytes, dtype=np.uint8),
        is_dense=True,
    )

################################################################ 
# IMU builder                                                  # 
################################################################ 
def make_imu(oxts_packet, timestamp_ns: int) -> object:
    """Build sensor_msgs/Imu from a pykitti OXTS packet."""
    ax = float(oxts_packet.ax)
    ay = float(oxts_packet.ay)
    az = float(oxts_packet.az)
    wx = float(oxts_packet.wx)
    wy = float(oxts_packet.wy)
    wz = float(oxts_packet.wz)

    # Euler → quaternion (roll, pitch, yaw from OXTS)
    roll  = float(oxts_packet.roll)
    pitch = float(oxts_packet.pitch)
    yaw   = float(oxts_packet.yaw)
    cr, sr = np.cos(roll/2),  np.sin(roll/2)
    cp, sp = np.cos(pitch/2), np.sin(pitch/2)
    cy, sy = np.cos(yaw/2),   np.sin(yaw/2)
    qw = cr*cp*cy + sr*sp*sy
    qx = sr*cp*cy - cr*sp*sy
    qy = cr*sp*cy + sr*cp*sy
    qz = cr*cp*sy - sr*sp*cy

    cov9  = _Covariance()
    cov9[0] = cov9[4] = cov9[8] = 0.01

    return _Imu(
        header=_make_header(timestamp_ns, "imu_link"),
        orientation=_Quat(x=qx, y=qy, z=qz, w=qw),
        orientation_covariance=cov9.copy(),
        angular_velocity=_Vec3(x=wx, y=wy, z=wz),
        angular_velocity_covariance=cov9.copy(),
        linear_acceleration=_Vec3(x=ax, y=ay, z=az),
        linear_acceleration_covariance=cov9.copy(),
    )


################################################################ 
# NavSatFix builder                                            # 
################################################################ 
_STATUS_FIX  = TYPESTORE.types["sensor_msgs/msg/NavSatStatus"]
_FIX_STATUS_FIX = 0

def make_navsatfix(oxts_packet, timestamp_ns: int) -> object:
    return _Fix(
        header=_make_header(timestamp_ns, "gps_link"),
        status=_STATUS_FIX(status=_FIX_STATUS_FIX, service=1),
        latitude=float(oxts_packet.lat),
        longitude=float(oxts_packet.lon),
        altitude=float(oxts_packet.alt),
        position_covariance=np.zeros(9, dtype=np.float64),
        position_covariance_type=0,
    )


################################################################ 
# Timestamp helpers                                            # 
################################################################ 
def _datetime_to_ns(dt) -> int:
    """Convert a Python datetime to nanoseconds since Unix epoch."""
    import datetime
    epoch = datetime.datetime(1970, 1, 1, tzinfo=datetime.timezone.utc)
    # pykitti timestamps are timezone-aware
    if dt.tzinfo is None:
        import datetime as dt_mod
        dt = dt.replace(tzinfo=dt_mod.timezone.utc)
    delta = dt - epoch
    return int(delta.total_seconds() * 1e9)


################################################################ 
# Core converter                                               # 
################################################################ 
def convert(
    kitti_dir: Path,
    date: str,
    drive: str,
    output_path: Path,
    lidar_only: bool = False,
    max_frames: Optional[int] = None,
) -> None:
    print(f"  Source : {kitti_dir / date / f'{date}_drive_{drive}_sync'}")
    print(f"  Output : {output_path}")
    print(f"  Mode   : {'LiDAR only' if lidar_only else 'LiDAR + IMU + GPS'}")
    if max_frames:
        print(f"  Frames : {max_frames} (limited)")
    print()

    # Load KITTI sequence via pykitti
    print("Loading KITTI sequence metadata...")
    kitti = pykitti.raw(str(kitti_dir), date, drive)

    n_frames = len(kitti.velo_files)
    if max_frames:
        n_frames = min(n_frames, max_frames)

    print(f"  Found {len(kitti.velo_files)} LiDAR frames, converting {n_frames}")

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        import shutil
        shutil.rmtree(output_path) if output_path.is_dir() else output_path.unlink()

    # Write bag (provide version as parameter based rosbag lib version)
    with Writer(output_path, version=9) as bag:

        # Register connections
        velo_conn = bag.add_connection(
            "/velodyne_points",
            "sensor_msgs/msg/PointCloud2",
            typestore=TYPESTORE,
        )

        if not lidar_only:
            imu_conn = bag.add_connection(
                "/imu/data",
                "sensor_msgs/msg/Imu",
                typestore=TYPESTORE,
            )
            gps_conn = bag.add_connection(
                "/gps/fix",
                "sensor_msgs/msg/NavSatFix",
                typestore=TYPESTORE,
            )

        # Frame loop 
        t_start = time.perf_counter()
        total_points = 0

        for i in tqdm(range(n_frames), desc="Converting frames", unit="frame"):
            # LiDAR timestamp
            velo_ts = kitti.timestamps[i]
            ts_ns   = _datetime_to_ns(velo_ts)

            # Load .bin file: flat float32 array → (N, 4)
            velo_path = Path(kitti.velo_files[i])
            raw = np.fromfile(str(velo_path), dtype=np.float32)
            points = raw.reshape(-1, 4)          # x, y, z, intensity
            total_points += len(points)

            # Build and write PointCloud2
            pc2_msg = make_pointcloud2(points, ts_ns)
            raw_bytes = TYPESTORE.serialize_cdr(pc2_msg, "sensor_msgs/msg/PointCloud2")
            bag.write(velo_conn, ts_ns, raw_bytes)

            # IMU + GPS (from OXTS)
            if not lidar_only:
                try:
                    oxts = kitti.oxts[i]
                    oxts_packet = oxts.packet

                    imu_msg = make_imu(oxts_packet, ts_ns)
                    imu_raw = TYPESTORE.serialize_cdr(imu_msg, "sensor_msgs/msg/Imu")
                    bag.write(imu_conn, ts_ns, imu_raw)

                    fix_msg = make_navsatfix(oxts_packet, ts_ns)
                    fix_raw = TYPESTORE.serialize_cdr(fix_msg, "sensor_msgs/msg/NavSatFix")
                    bag.write(gps_conn, ts_ns, fix_raw)
                except Exception as exc:
                    # OXTS missing for some sequences — skip silently
                    pass

        elapsed = time.perf_counter() - t_start

    # Summary 
    bag_size_mb = sum(f.stat().st_size for f in output_path.rglob("*") if f.is_file()) / 1e6
    print(f"\n{'─'*60}")
    print(f"  Conversion complete!")
    print(f"  Frames converted  : {n_frames}")
    print(f"  Total LiDAR pts   : {total_points:,}")
    print(f"  Bag size          : {bag_size_mb:.1f} MB")
    print(f"  Time taken        : {elapsed:.1f}s  ({n_frames/elapsed:.1f} frames/sec)")
    print(f"  Output            : {output_path.resolve()}")
    print(f"\n  Run your pipeline:")
    print(f"    python scripts/run_pipeline.py \\")
    print(f"      --bag \"{output_path.resolve()}\" \\")
    print(f"      --topic /velodyne_points")
    print(f"{'─'*60}\n")


################################################################ 
# CLI                                                          # 
################################################################ 
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert KITTI raw data to ROS2 bag (no ROS needed)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--kitti-dir", "-d",
        required=True,
        help="Root folder containing the date subfolder (e.g. C:/data/kitti)",
    )
    p.add_argument(
        "--date", "-t",
        required=True,
        help="Recording date, e.g. 2011_10_03",
    )
    p.add_argument(
        "--drive", "-r",
        required=True,
        help="Drive number as 4-digit string, e.g. 0027",
    )
    p.add_argument(
        "--output", "-o",
        default=None,
        help="Output .bag path. Default: ./<date>_drive_<drive>.bag",
    )
    p.add_argument(
        "--lidar-only",
        action="store_true",
        help="Skip IMU and GPS",
    )
    p.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Convert only the first N frames",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    kitti_dir = Path(args.kitti_dir)
    output    = Path(args.output) if args.output else Path(f"{args.date}_drive_{args.drive}.bag")

    convert(
        kitti_dir=kitti_dir,
        date=args.date,
        drive=args.drive,
        output_path=output,
        lidar_only=args.lidar_only,
        max_frames=args.max_frames,
    )
