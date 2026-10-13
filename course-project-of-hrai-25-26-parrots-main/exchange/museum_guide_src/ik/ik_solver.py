#!/usr/bin/env python3
"""
Inverse kinematics for pointing the Booster-T1 arm/head at a 3D target.

All coordinates are in the robot base ("Trunk") frame, matching
ik/t1_kinematics.xml: +X forward, +Y left, +Z up, origin at Trunk.

The arm only reaches ~0.55 m, but painting targets are meters away, so
"pointing at" a target means orienting the outstretched arm along the
shoulder->target direction, not literally reaching the point. We solve
position IK toward a proxy point projected onto the arm's reachable sphere
along that direction.

Elbow_Pitch is kept fixed at 0 (fully extended), matching the hand-tuned
canned gestures in kArmTargets (point_left/point_right also use
elbow_pitch=0) — only Shoulder_Pitch/Shoulder_Roll are solved. With a rigid
straight forearm, 2 DOF is exactly enough to aim anywhere on the reachable
sphere. Solving all 4 joints (including elbow) is under-constrained for a
pure position target: the optimizer can (and did, before this fix) converge
to a bent-elbow solution that lands the fingertip on the target while the
forearm itself points somewhere else entirely — technically "reaching" the
point but visually not reading as pointing at it.
"""

from pathlib import Path

import mujoco
import numpy as np

_MODEL_PATH = Path(__file__).parent / "t1_kinematics.xml"
_model = mujoco.MjModel.from_xml_path(str(_MODEL_PATH))
_data = mujoco.MjData(_model)

_ARM_JOINTS = {
    "left": ["Left_Shoulder_Pitch", "Left_Shoulder_Roll", "Left_Elbow_Pitch", "Left_Elbow_Yaw"],
    "right": ["Right_Shoulder_Pitch", "Right_Shoulder_Roll", "Right_Elbow_Pitch", "Right_Elbow_Yaw"],
}
_AIM_JOINTS = {
    "left": ["Left_Shoulder_Pitch", "Left_Shoulder_Roll"],
    "right": ["Right_Shoulder_Pitch", "Right_Shoulder_Roll"],
}
_ARM_SITE = {"left": "left_hand_tip", "right": "right_hand_tip"}
_SHOULDER_BODY = {"left": "AL1", "right": "AR1"}

_NEUTRAL_ARM = {
    "left": [0.0, -1.5, 0.05, -0.3],
    "right": [0.0, 1.5, 0.05, 0.3],
}

_HEAD_JOINTS = ["AAHead_yaw", "Head_pitch"]


def _joint_qpos_and_ids(joint_names):
    jids = [mujoco.mj_name2id(_model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in joint_names]
    qadr = [_model.jnt_qposadr[j] for j in jids]
    return qadr, jids


def _shoulder_pos(arm: str) -> np.ndarray:
    bid = mujoco.mj_name2id(_model, mujoco.mjtObj.mjOBJ_BODY, _SHOULDER_BODY[arm])
    return np.array(_model.body_pos[bid])


def _straight_pose(arm: str) -> list[float]:
    """Elbow fully extended (pitch=0), keeping the neutral pose's elbow-yaw
    twist — the same convention as kArmTargets' point_left/point_right."""
    neutral = _NEUTRAL_ARM[arm]
    return [neutral[0], neutral[1], 0.0, neutral[3]]


def _reach_length(arm: str) -> float:
    """Distance from the shoulder pivot to the hand tip with the arm fully
    extended (elbow_pitch=0) — constant regardless of shoulder angles, since
    shoulder rotation is a rigid rotation about the shoulder pivot."""
    qadr, jids = _joint_qpos_and_ids(_ARM_JOINTS[arm])
    mujoco.mj_resetData(_model, _data)
    for i, a in zip(qadr, _straight_pose(arm)):
        _data.qpos[i] = a
    mujoco.mj_forward(_model, _data)
    site_id = mujoco.mj_name2id(_model, mujoco.mjtObj.mjOBJ_SITE, _ARM_SITE[arm])
    return float(np.linalg.norm(_data.site_xpos[site_id] - _shoulder_pos(arm)))


def solve_arm(target_xyz, arm: str | None = None, iters: int = 200,
              damping: float = 1e-2, tol: float = 1e-3):
    """
    Solve Shoulder_Pitch/Shoulder_Roll for `arm` ("left"/"right") that orient
    the fully-extended (straight-elbow) arm toward target_xyz (robot base
    frame). Auto-picks the arm from target_xyz[1] sign if not given (y>0 ->
    left, matching the Left_* joints' positive-Y shoulder offset).

    Returns dict: arm, q4 (list[4], elbow fixed straight), residual_m, converged.
    """
    target = np.asarray(target_xyz, dtype=float)
    if arm is None:
        arm = "left" if target[1] >= 0 else "right"

    shoulder = _shoulder_pos(arm)
    direction = target - shoulder
    dist = np.linalg.norm(direction)
    if dist < 1e-6:
        direction = np.array([1.0, 0.0, 0.0])
        dist = 1.0
    direction = direction / dist
    proxy_target = shoulder + direction * _reach_length(arm)

    all_joint_names = _ARM_JOINTS[arm]
    qadr, jids = _joint_qpos_and_ids(all_joint_names)
    aim_qadr, aim_jids = _joint_qpos_and_ids(_AIM_JOINTS[arm])
    aim_dof_idx = [_model.jnt_dofadr[j] for j in aim_jids]
    site_id = mujoco.mj_name2id(_model, mujoco.mjtObj.mjOBJ_SITE, _ARM_SITE[arm])

    mujoco.mj_resetData(_model, _data)
    q_fixed = _straight_pose(arm)
    for i, a in zip(qadr, q_fixed):
        _data.qpos[i] = a
    q_aim = np.array(q_fixed[:2], dtype=float)

    jacp = np.zeros((3, _model.nv))
    residual = None
    for _ in range(iters):
        mujoco.mj_forward(_model, _data)
        cur = _data.site_xpos[site_id].copy()
        err = proxy_target - cur
        residual = float(np.linalg.norm(err))
        if residual < tol:
            break
        mujoco.mj_jacSite(_model, _data, jacp, None, site_id)
        J = jacp[:, aim_dof_idx]
        JJt = J @ J.T + damping * np.eye(3)
        dq = J.T @ np.linalg.solve(JJt, err)
        q_aim = q_aim + dq
        for k, jid in enumerate(aim_jids):
            lo, hi = _model.jnt_range[jid]
            q_aim[k] = np.clip(q_aim[k], lo, hi)
        for i, a in zip(aim_qadr, q_aim):
            _data.qpos[i] = a

    q4 = [q_aim[0], q_aim[1], 0.0, q_fixed[3]]

    converged = residual is not None and residual < 0.06
    return {"arm": arm, "q4": q4, "residual_m": residual, "converged": converged}


def solve_arm_targets_8(target_xyz, arm: str | None = None) -> dict:
    """Full 8-float arm target array in kArmTargets order (idle/neutral pose
    on the unused arm), ready to pass as --arm-q to arm_head_goal_ramp_actions."""
    result = solve_arm(target_xyz, arm)
    left = result["q4"] if result["arm"] == "left" else _NEUTRAL_ARM["left"]
    right = result["q4"] if result["arm"] == "right" else _NEUTRAL_ARM["right"]
    result["arm8"] = [*left, *right]
    return result


def solve_head_look_at(target_xyz) -> tuple[float, float]:
    """
    Closed-form head yaw/pitch aiming the camera roughly at target_xyz
    (robot base frame). Sign convention verified against t1_kinematics.xml:
    positive Head_pitch tilts the camera down/forward.
    """
    h1_id = mujoco.mj_name2id(_model, mujoco.mjtObj.mjOBJ_BODY, "H1")
    h1_pos = np.array(_model.body_pos[h1_id])
    dx, dy, dz = (np.asarray(target_xyz, dtype=float) - h1_pos).tolist()

    yaw_jid = mujoco.mj_name2id(_model, mujoco.mjtObj.mjOBJ_JOINT, "AAHead_yaw")
    pitch_jid = mujoco.mj_name2id(_model, mujoco.mjtObj.mjOBJ_JOINT, "Head_pitch")

    yaw = float(np.clip(np.arctan2(dy, dx), *_model.jnt_range[yaw_jid]))
    horiz = float(np.hypot(dx, dy))
    pitch = float(np.clip(-np.arctan2(dz, horiz), *_model.jnt_range[pitch_jid]))
    return yaw, pitch
