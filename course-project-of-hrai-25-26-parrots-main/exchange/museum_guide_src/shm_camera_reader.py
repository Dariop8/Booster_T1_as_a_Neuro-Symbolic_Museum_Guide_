#!/usr/bin/env python3
"""
Reads the latest RGB / depth frames written by circus to shared memory.

Circus writes camera frames to:
  /dev/shm/circus_ipc/<robot_name>_rgb.shm
  /dev/shm/circus_ipc/<robot_name>_depth.shm

Ring-buffer layout (all native 64-bit, no padding between members):
  Header {
    uint32  magic       offset  0
    uint32  version     offset  4
    uint32  width       offset  8
    uint32  height      offset 12
    uint32  channels    offset 16
    uint32  slot_count  offset 20
    size_t  slot_bytes  offset 24  (8 bytes on 64-bit Linux)
    uint64  seq         offset 32
  }                              total: 40 bytes
  followed by slot_count * slot_bytes of pixel data
  (RGB: 3ch uint8, row-major top-to-bottom.
   Depth: 1ch uint8, same layout — see read_latest_depth_frame for decoding.)
"""

import mmap
import struct
from pathlib import Path

import numpy as np

_HEADER_FMT = "@IIIIIIQQ"
_HEADER_SIZE = struct.calcsize(_HEADER_FMT)
_MAGIC = 0x5348514D

_DEPTH_MAX_METERS = 10.0


def _read_shm_slot(robot_name: str, suffix: str) -> tuple[np.ndarray, int, int, int] | None:
    """Read the latest raw slot from a circus IPC ring buffer. Returns
    (raw_bytes_as_uint8_array, width, height, channels) or None."""
    path = Path(f"/dev/shm/circus_ipc/{robot_name}_{suffix}.shm")
    if not path.exists():
        return None

    try:
        with path.open("r+b") as f:
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            if mm.size() < _HEADER_SIZE:
                mm.close()
                return None

            header_bytes = mm[:_HEADER_SIZE]
            magic, version, width, height, channels, slot_count, slot_bytes, seq = \
                struct.unpack(_HEADER_FMT, header_bytes)

            if magic != _MAGIC or seq == 0 or width == 0 or height == 0:
                mm.close()
                return None

            slot_idx = int(seq % slot_count)
            offset = _HEADER_SIZE + slot_idx * slot_bytes
            frame_bytes = mm[offset: offset + slot_bytes]
            mm.close()

        raw = np.frombuffer(frame_bytes, dtype=np.uint8).copy()
        return raw, width, height, channels

    except (OSError, ValueError, struct.error):
        return None


def read_latest_frame(robot_name: str = "red_Booster-T1_0") -> np.ndarray | None:
    """
    Return the latest RGB frame from the circus shared-memory ring buffer.
    Returns an np.ndarray of shape (H, W, 3) uint8, or None if unavailable.
    """
    result = _read_shm_slot(robot_name, "rgb")
    if result is None:
        return None
    raw, width, height, channels = result
    return raw.reshape(height, width, channels)


def read_latest_depth_frame(robot_name: str = "red_Booster-T1_0") -> np.ndarray | None:
    """
    Return the latest depth frame from the circus shared-memory ring buffer,
    decoded to metric meters. Returns an np.ndarray of shape (H, W) float32,
    or None if unavailable. Pixel-aligned with read_latest_frame()'s RGB
    frame — both are rendered from the same camera pose (see CameraDepth.h).
    """
    result = _read_shm_slot(robot_name, "depth")
    if result is None:
        return None
    raw, width, height, channels = result
    depth8 = raw.reshape(height, width, channels)[:, :, 0]
    return depth8.astype(np.float32) / 255.0 * _DEPTH_MAX_METERS


def read_latest_pose(robot_name: str = "red_Booster-T1_0") -> dict | None:
    """
    Ground-truth base pose (world frame), written every simulation tick by
    circus/include/robots/BoosterT1.h — see that file's update() for the
    packing (6 float32: x, y, z, roll, pitch, yaw). Path planning doesn't
    need SLAM/LIDAR to localize the robot in simulation: MuJoCo already
    knows exactly where it is, this just exposes it to Python the same way
    read_latest_frame() exposes the camera.

    Returns {"x","y","z","roll","pitch","yaw"} (meters/radians), or None if
    unavailable (is circus running?).
    """
    result = _read_shm_slot(robot_name, "pose")
    if result is None:
        return None
    raw, width, height, channels = result
    if width * height * channels != 6 * 4:
        return None
    values = np.frombuffer(raw.tobytes(), dtype=np.float32)
    return {
        "x": float(values[0]), "y": float(values[1]), "z": float(values[2]),
        "roll": float(values[3]), "pitch": float(values[4]), "yaw": float(values[5]),
    }


if __name__ == "__main__":
    frame = read_latest_frame()
    if frame is None:
        print("No frame available (is circus running?)")
    else:
        print(f"Frame captured: {frame.shape}  dtype={frame.dtype}")
        try:
            import cv2
            cv2.imwrite("/tmp/circus_frame.png", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            print("Saved to /tmp/circus_frame.png")
        except ImportError:
            print("cv2 not available — cannot save PNG")
