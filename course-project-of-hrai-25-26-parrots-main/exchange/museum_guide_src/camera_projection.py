#!/usr/bin/env python3
"""
Back-projects a (pixel, depth) reading from the head camera into a 3D point
in the robot base ("Trunk") frame — used to localize paintings during a scan.

Camera intrinsics/pose come from ik/t1_kinematics.xml's "cam_site", copied
from the "rgb_cam" MuJoCo camera in
circus/resources/robots/Booster-T1/instance.xml. Circus's CameraDepth
(circus/include/sensors/CameraDepth.h) is instantiated reusing the RGB
camera's pose to render depth too, so RGB and depth pixels are exactly
aligned — read_latest_depth_frame()'s array indexes the same way as
read_latest_frame()'s.

We never read the head's live joint state here: during a scan the head is
driven to known commanded angles (SCAN_YAW_RAD / kHeadTargets), so those
commanded values ARE the camera pose, with no sensor round-trip needed.
"""

from pathlib import Path

import mujoco
import numpy as np

_MODEL_PATH = Path(__file__).parent / "ik" / "t1_kinematics.xml"
_model = mujoco.MjModel.from_xml_path(str(_MODEL_PATH))
_data = mujoco.MjData(_model)

FOVY_DEG = 70.0
IMG_WIDTH = 640
IMG_HEIGHT = 480

_YAW_JID = mujoco.mj_name2id(_model, mujoco.mjtObj.mjOBJ_JOINT, "AAHead_yaw")
_PITCH_JID = mujoco.mj_name2id(_model, mujoco.mjtObj.mjOBJ_JOINT, "Head_pitch")
_CAM_SITE_ID = mujoco.mj_name2id(_model, mujoco.mjtObj.mjOBJ_SITE, "cam_site")


def camera_pose(head_yaw: float, head_pitch: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Camera position (3,) and rotation matrix (3,3) in the robot base frame,
    for the given COMMANDED head joint angles.
    """
    mujoco.mj_resetData(_model, _data)
    _data.qpos[_model.jnt_qposadr[_YAW_JID]] = head_yaw
    _data.qpos[_model.jnt_qposadr[_PITCH_JID]] = head_pitch
    mujoco.mj_forward(_model, _data)
    pos = _data.site_xpos[_CAM_SITE_ID].copy()
    rotmat = _data.site_xmat[_CAM_SITE_ID].reshape(3, 3).copy()
    return pos, rotmat


def backproject(u: float, v: float, depth_m: float, head_yaw: float, head_pitch: float,
                 img_width: int = IMG_WIDTH, img_height: int = IMG_HEIGHT,
                 fovy_deg: float = FOVY_DEG) -> np.ndarray:
    """
    Back-project pixel (u, v) — u: column from left, v: row from top — with
    metric depth (meters, perpendicular camera-space Z, matching the
    standard OpenGL depth-buffer convention used by CameraDepth.h) into a 3D
    point in the robot base (Trunk) frame.
    """
    f = (img_height / 2.0) / np.tan(np.radians(fovy_deg) / 2.0)
    cx, cy = img_width / 2.0, img_height / 2.0

    z_cam = -depth_m
    x_cam = (u - cx) / f * depth_m
    y_cam = (cy - v) / f * depth_m
    point_cam = np.array([x_cam, y_cam, z_cam])

    cam_pos, cam_rot = camera_pose(head_yaw, head_pitch)
    return cam_pos + cam_rot @ point_cam


def to_world_frame(local_xyz, robot_pose: dict) -> list[float]:
    """
    Converts a point in the robot's own base ("Trunk") frame — e.g.
    backproject()'s output — into world frame, given the robot's actual
    pose at the moment that point was observed (shm_camera_reader
    .read_latest_pose()'s dict: {"x","y","z","yaw",...}).

    Needed once the robot walks between scan stops: backproject() is always
    base-relative regardless of where the base actually is, so a painting
    localized while standing at one spot and a painting localized while
    standing at another aren't directly comparable — or usable for pointing
    later from yet a THIRD position (back at the reception desk) — until
    both are expressed in one common (world) frame. Planar transform only
    (rotate by yaw, ignore roll/pitch): reasonable for a bipedal robot that
    stays upright while observing, same simplification locomotion.py's
    heading control already makes.
    """
    lx, ly, lz = local_xyz
    yaw = robot_pose["yaw"]
    cos_y, sin_y = np.cos(yaw), np.sin(yaw)
    return [
        robot_pose["x"] + lx * cos_y - ly * sin_y,
        robot_pose["y"] + lx * sin_y + ly * cos_y,
        robot_pose["z"] + lz,
    ]


def to_base_frame(world_xyz, robot_pose: dict) -> list[float]:
    """
    Inverse of to_world_frame(): converts a world-frame point (e.g. a
    painting position discovered during a walking scan, from a DIFFERENT
    standing spot) into the robot's CURRENT base frame, given its CURRENT
    pose — what run_robot_point_at()/ik_solver.solve_arm() require, since
    they always operate in base-relative coordinates regardless of where
    the base actually is in the room.
    """
    dx = world_xyz[0] - robot_pose["x"]
    dy = world_xyz[1] - robot_pose["y"]
    dz = world_xyz[2] - robot_pose["z"]
    yaw = robot_pose["yaw"]
    cos_y, sin_y = np.cos(yaw), np.sin(yaw)
    return [
        dx * cos_y + dy * sin_y,
        -dx * sin_y + dy * cos_y,
        dz,
    ]


def median_depth_in_bbox(depth_frame: np.ndarray, bbox: tuple[int, int, int, int]) -> float | None:
    """
    Robust depth sample for a detected painting: median over the bbox
    (x0, y0, x1, y1), ignoring far-plane/background pixels (>= 9.5 m, the
    "nothing there" clip value from CameraDepth's 10 m max range).
    """
    x0, y0, x1, y1 = bbox
    patch = depth_frame[y0:y1, x0:x1]
    valid = patch[patch < 9.5]
    if valid.size == 0:
        return None
    return float(np.median(valid))


def pointcloud_centroid_in_bbox(
    depth_frame: np.ndarray, bbox: tuple[int, int, int, int],
    head_yaw: float, head_pitch: float,
    img_width: int = IMG_WIDTH, img_height: int = IMG_HEIGHT,
    fovy_deg: float = FOVY_DEG, max_depth_m: float = 9.5,
    stride: int = 2, outlier_k: float = 2.5,
) -> np.ndarray | None:
    """
    Point-cloud-based localization, the actual detect -> cloud -> centroid
    pipeline taught in Geometric2Semantic (point cloud filtering/statistical
    outlier removal, camera projection intrinsics/extrinsics) and used by
    LOST-3DSG's geometric re-projector (HRAI 5 - Scene Graphs, slides
    69-92): back-project EVERY valid pixel inside bbox (not just its center)
    into a base-frame 3D point, drop depth outliers, then return the cloud's
    centroid. Supersedes the older single-pixel/median-depth shortcut
    (median_depth_in_bbox + backproject on the box center) with a real,
    if lightweight, point cloud.

    Outlier removal is depth-based statistical filtering (PCL's
    StatisticalOutlierRemoval), simplified to 1-D since a painting is
    planar and shot near-frontally: the dominant outlier source isn't 3D
    structure on the canvas itself, it's a sliver of wall creeping in at
    the bbox edges, which reads as noticeably farther than the canvas
    surface. Points farther than outlier_k * MAD from the median depth are
    dropped before computing the centroid.

    stride subsamples the bbox for speed (stride=2 keeps 1/4 of the
    pixels — still hundreds of points for a >=100x100 px painting bbox).
    """
    x0, y0, x1, y1 = bbox
    patch = depth_frame[y0:y1:stride, x0:x1:stride]
    if patch.size == 0:
        return None
    vs, us = np.mgrid[y0:y1:stride, x0:x1:stride]
    valid = patch < max_depth_m
    if not np.any(valid):
        return None
    depths = patch[valid].astype(float)
    us = us[valid].astype(float)
    vs = vs[valid].astype(float)

    med = np.median(depths)
    mad = np.median(np.abs(depths - med)) + 1e-6
    keep = np.abs(depths - med) <= outlier_k * mad
    if not np.any(keep):
        return None
    depths, us, vs = depths[keep], us[keep], vs[keep]

    f = (img_height / 2.0) / np.tan(np.radians(fovy_deg) / 2.0)
    cx, cy = img_width / 2.0, img_height / 2.0
    x_cam = (us - cx) / f * depths
    y_cam = (cy - vs) / f * depths
    z_cam = -depths
    points_cam = np.stack([x_cam, y_cam, z_cam], axis=1)

    cam_pos, cam_rot = camera_pose(head_yaw, head_pitch)
    points_base = cam_pos + points_cam @ cam_rot.T
    return points_base.mean(axis=0)
