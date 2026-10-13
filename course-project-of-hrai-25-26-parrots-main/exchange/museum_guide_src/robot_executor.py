#!/usr/bin/env python3

import os
import shlex
import subprocess
from pathlib import Path

import console
from ik.ik_solver import solve_arm_targets_8, solve_head_look_at


def find_repo_root() -> Path:
    current = Path(__file__).resolve()
    for parent in [current.parent] + list(current.parents):
        if (parent / "exchange").exists() and (parent / "circus").exists():
            return parent
    raise RuntimeError("Could not find repository root containing exchange/ and circus/.")


REPO_ROOT = find_repo_root()
CIRCUS_DIR = REPO_ROOT / "circus"

_VALID_GESTURES = {
    "point_left",
    "point_right",
    "point_front_left",
    "point_front_right",
    "point_left_mid",
    "point_right_mid",
}

_ARM_TO_HEAD = {
    "point_left":        "look_point_left",
    "point_right":       "look_point_right",
    "point_front_left":  "look_point_front_left",
    "point_front_right": "look_point_front_right",
    "point_left_mid":    "look_point_left_mid",
    "point_right_mid":   "look_point_right_mid",
}

_MODE_ALIASES = {
    "trajectory": "trajectory",
    "traj":       "trajectory",
    "hpp":        "trajectory",
    "retarget":   "trajectory",
    "goal":       "goal",
    "ramp":       "goal",
    "simple":     "goal",
}


def _get_gesture_mode() -> str:
    raw = os.environ.get("GESTURE_MODE", "trajectory").strip().lower()
    mode = _MODE_ALIASES.get(raw)
    if mode is None:
        console.warn(f"[Robot Executor] Unknown GESTURE_MODE='{raw}'. Using 'trajectory'.")
        mode = "trajectory"
    return mode


def _script_for_mode(mode: str) -> Path:
    if mode == "goal":
        return CIRCUS_DIR / "debug_scripts" / "run_cpp_arm_head_goal_action.sh"
    return CIRCUS_DIR / "debug_scripts" / "run_cpp_arm_hpp_action.sh"


def _extra_args_for_mode(mode: str) -> list[str]:
    if mode == "goal":
        return shlex.split(os.environ.get("GESTURE_GOAL_ARGS", ""))
    return shlex.split(os.environ.get("GESTURE_TRAJECTORY_ARGS", ""))


def prepare_gesture_backend() -> bool:
    """Pre-compila il backend prima della prima gesture."""
    mode = _get_gesture_mode()
    script = _script_for_mode(mode)

    if not script.exists():
        console.error(f"[Robot Executor] Script not found: {script}")
        return False

    if mode == "goal":
        try:
            console.detail("[Robot Executor] Building arm+head backend...")
            subprocess.run([str(script), "--build-only"], cwd=str(CIRCUS_DIR), check=True)
            console.detail("[Robot Executor] Backend ready.")
            return True
        except subprocess.CalledProcessError as e:
            console.error(f"[Robot Executor] Backend build failed: {e}")
            return False
    else:
        console.detail("[Robot Executor] Trajectory backend: warmup skipped.")
        return True


def _run_movement(cmd: list[str], label: str) -> bool:
    """
    Runs a movement subprocess. Its stdout (compiler cache notes, measured
    poses, tracking errors) is captured instead of streamed: gestures run in a
    background thread while the robot's speech is being printed, and the two
    outputs would interleave mid-sentence on the terminal. Full output is
    shown on failure, or always with GESTURE_VERBOSE=1.
    """
    verbose = os.environ.get("GESTURE_VERBOSE", "0") == "1"
    try:
        result = subprocess.run(
            cmd, cwd=str(CIRCUS_DIR), check=True,
            capture_output=not verbose, text=True,
        )
        if verbose and result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        console.error(f"[Robot Executor] {label} failed: {e}")
        for stream in (e.stdout, e.stderr):
            if stream:
                console.error(stream[-2000:])
        return False


def run_robot_gesture(gesture_name: str) -> bool:
    """
    Esegue braccio + testa contemporaneamente tramite il programma combinato.
    Singolo publisher DDS → nessun conflitto.
    """
    if gesture_name not in _VALID_GESTURES:
        console.warn(f"[Robot Executor] Unknown gesture '{gesture_name}'. Valid: {_VALID_GESTURES}")
        return False

    mode = _get_gesture_mode()
    script = _script_for_mode(mode)

    if not script.exists():
        console.error(f"[Robot Executor] Script not found: {script}")
        return False

    extra_args = _extra_args_for_mode(mode)

    if mode == "goal":
        head_action = _ARM_TO_HEAD[gesture_name]
        cmd = [str(script), gesture_name, head_action] + extra_args
    else:
        cmd = [str(script), gesture_name] + extra_args

    return _run_movement(cmd, f"Gesture {gesture_name}")


def run_robot_point_at(xyz: tuple[float, float, float]) -> bool:
    """
    Punta braccio + testa verso una posizione 3D (SEMPRE frame base del
    robot CORRENTE — discovered_paintings.json[...]["xyz"] è invece in
    frame MONDO dalla riscrittura a scan camminato: va convertito con
    camera_projection.to_base_frame() + shm_camera_reader.read_latest_pose()
    prima di chiamare questa funzione, come fa
    guida_museale_core.handle_recommend()) calcolata via IK, invece di
    scegliere uno dei 4 gesti canned.

    Richiede GESTURE_MODE=goal: è l'unico backend che accetta target di
    giunto arbitrari (--arm-q/--head-q su arm_head_goal_ramp_actions.cpp);
    la modalità trajectory (default) riproduce solo traiettorie pre-registrate
    per nome e non può eseguire un punto continuo. In quel caso, o se l'IK
    non converge, ritorna False: il chiamante ricade sul gesto canned
    (robot_actions.pointing_skill), come da
    robot_capabilities.yaml -> fallback_actions.cannot_point.
    """
    mode = _get_gesture_mode()
    if mode != "goal":
        console.warn(f"[Robot Executor] run_robot_point_at richiede GESTURE_MODE=goal "
                     f"(attuale: '{mode}') — fallback al gesto canned.")
        return False

    script = _script_for_mode("goal")
    if not script.exists():
        console.error(f"[Robot Executor] Script not found: {script}")
        return False

    result = solve_arm_targets_8(xyz)
    if not result["converged"]:
        console.warn(f"[Robot Executor] IK non converge per target {xyz} "
                     f"(residuo {result['residual_m']:.3f} m) — fallback al gesto canned.")
        return False

    head_yaw, head_pitch = solve_head_look_at(xyz)
    arm_q = ",".join(f"{v:.6f}" for v in result["arm8"])
    head_q = f"{head_yaw:.6f},{head_pitch:.6f}"

    extra_args = _extra_args_for_mode("goal")
    cmd = [str(script), "point_at", "point_at", "--arm-q", arm_q, "--head-q", head_q] + extra_args

    return _run_movement(cmd, f"Point at {xyz}")


def run_neutral_pose() -> bool:
    """
    Riporta braccio e testa alla posa neutra in un singolo movimento.
    Usa sempre il backend goal (non esiste neutral.hpp).
    """
    script = CIRCUS_DIR / "debug_scripts" / "run_cpp_arm_head_goal_action.sh"

    if not script.exists():
        console.error(f"[Robot Executor] Script not found: {script}")
        return False

    return _run_movement([str(script), "neutral", "center"], "Neutral pose")
