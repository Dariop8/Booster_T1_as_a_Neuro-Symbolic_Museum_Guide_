#!/usr/bin/env python3
"""
Museum visual explorer.

The robot WALKS to a standing spot in front of each painting, one at a
time, looks at it, captures a frame from the circus shared-memory camera,
sends it to a vision LLM, and builds a knowledge-graph-ready dict with the
discovered painting attributes. (Earlier version turned only the head from
a fixed standing position — replaced so the robot gets a square-on view of
each canvas instead of the oblique angles a head-only turn gives at the far
ones.)

Run once at startup (before the main guide loop) to populate the KG.
While the scan runs, the same live viewer used by guida_museale_core
(viewer/kg_viewer.html, port KG_VIEWER_PORT, default 8009) shows the scene
graph growing after every scanned direction.

Needs a live circus/T1 (rebuilt with BoosterT1.h's pose_writer_) AND the T2
walk-ready sequence (gm, hcm-start, gm) — walking needs ground-truth pose
and a working locomotion channel, neither of which the old head-only scan
required.

Usage (from circus/ directory):
    pixi run python ../exchange/museum_guide_src/museum_explorer.py

Results are saved to exchange/data/discovered_paintings.json.
"""

import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml

import console

from shm_camera_reader import read_latest_frame, read_latest_depth_frame, read_latest_pose
from painting_recognizer import analyze_painting_voted
from knowledge_graph import (
    SCAN_YAW_RAD,
    build_graph_from_scan,
    export_graph_snapshot,
    spatial_map_from_placement,
    _load_registry,
)
from viewer_server import start_viewer_server
from camera_projection import pointcloud_centroid_in_bbox, to_world_frame, to_base_frame
from locomotion import navigate_to_pose
from ik.ik_solver import solve_head_look_at

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CIRCUS_DIR = _REPO_ROOT / "circus"
_SCRIPT = _CIRCUS_DIR / "debug_scripts" / "run_cpp_head_goal_action.sh"
_DATA_DIR = _REPO_ROOT / "exchange" / "data"
_OUTPUT = _DATA_DIR / "discovered_paintings.json"
_SPATIAL_OUTPUT = _DATA_DIR / "discovered_scene_map.yaml"

_VIEWER_DIR = Path(__file__).resolve().parent / "viewer"
_SNAPSHOT_PATH = _VIEWER_DIR / "kg_snapshot.json"
_VIEWER_PORT = int(os.environ.get("KG_VIEWER_PORT", "8009"))

_TEXTURES_DIR = _CIRCUS_DIR / "resources" / "textures"
_GROUND_TRUTH = _DATA_DIR / "scene_ground_truth.yaml"


def _warn_if_textures_stale() -> None:
    """
    scene_randomizer.py writes scene_ground_truth.yaml and wall_p*.png
    together in the same run, but circus only reads wall_p*.png once, at
    scene load — if the randomizer ran again WITHOUT a circus restart
    afterward, the registry now names a hanging the walls don't actually
    show yet. The robot's camera (and therefore the VLM) still correctly
    reads the stale-but-really-rendered painting, so this looks exactly like
    a classification failure — comparing wall_p*.png's mtime against the
    registry's catches it before wasting a scan chasing a phantom VLM bug.
    """
    if not _GROUND_TRUTH.exists():
        return
    registry_mtime = _GROUND_TRUTH.stat().st_mtime
    stale_grace_s = 60.0
    stale = [f.name for f in _TEXTURES_DIR.glob("wall_p[0-9]*.png")
             if f.stat().st_mtime < registry_mtime - stale_grace_s]
    if stale:
        console.warn(f"[Explorer] ⚠️  {_GROUND_TRUTH.name} is newer than {len(stale)} wall texture(s) "
                     f"({', '.join(sorted(stale))}) — scene_randomizer.py ran but circus was never "
                     f"restarted since, so the room still shows the OLD hanging. The scan will "
                     f"correctly read what's on the walls right now, which will look like a "
                     f"mismatch against the registry. Restart circus (T1) before scanning, or ignore "
                     f"if that's expected.")

def _load_candidates() -> list[dict]:
    """
    The FULL museum catalog (paintings.yaml) — passed to the VLM as the closed
    set of possible answers. The hanging is randomized (scene_randomizer.py),
    so the robot does not know which 4 of these are actually on the walls:
    identifying which catalog painting each direction shows IS the task.
    Closed-set classification over the catalog still prevents free-recall
    misidentifications (e.g. Las Meninas -> The Night Watch).
    """
    paintings = yaml.safe_load((_DATA_DIR / "paintings.yaml").read_text())["paintings"]
    return [
        {"id": p["id"], "title": p["title"], "artist": p["artist"], "movement": p.get("style", "")}
        for p in paintings
    ]

_HEAD_PITCH_RAD = 0.0

WORLD_FRAME_ORIGIN_POSE = {"x": 4.8, "y": 0.0, "z": 0.79, "yaw": math.pi}

_HEAD_TURN_SCAN_PLAN = [
    {
        "gesture_id": "point_left_mid",
        "head": "look_point_left_mid",
        "crop_x": (0.4375, 0.59375),
        "crop_y": (0.375, 0.729),
        "depth_bbox": (280, 195, 380, 335),
        "focus": "center",
        "label": "left wall, intermediate position (canvas p6)",
    },
    {
        "gesture_id": "point_left",
        "head": "look_point_left",
        "crop_x": (0.0, 0.32),
        "crop_y": (0.177, 0.698),
        "depth_bbox": (0, 100, 195, 320),
        "focus": "center",
        "label": "left wall position (canvas p4)",
    },
    {
        "gesture_id": "point_front_left",
        "head": "look_point_front_left",
        "crop_x": (0.383, 0.586),
        "crop_y": (0.396, 0.660),
        "depth_bbox": (268, 210, 347, 297),
        "focus": "center",
        "label": "front-left position (canvas p1)",
    },
    {
        "gesture_id": "point_front_right",
        "head": "look_point_front_right",
        "crop_x": (0.50, 0.68),
        "crop_y": (0.392, 0.650),
        "depth_bbox": (330, 208, 415, 292),
        "focus": "center",
        "label": "front-right position (canvas p2)",
    },
    {
        "gesture_id": "point_right",
        "head": "look_point_right",
        "crop_x": (0.75, 1.0),
        "crop_y": (0.156, 0.635),
        "depth_bbox": (503, 90, 633, 290),
        "focus": "center",
        "label": "right wall position (canvas p3)",
    },
    {
        "gesture_id": "point_right_mid",
        "head": "look_point_right_mid",
        "crop_x": (0.469, 0.648),
        "crop_y": (0.375, 0.729),
        "depth_bbox": (300, 195, 415, 335),
        "focus": "center",
        "label": "right wall, intermediate position (canvas p5)",
    },
]


def _move_head(head: str) -> None:
    """Head to one of the named scan directions.

    The reason for the ramp/hold values below is worth keeping: shortening
    them (0.6/0.15) to speed up the scan pushed the measured tracking error
    from ~0.07 rad to 0.11-0.24 rad — the head's PD gains are not stiff enough
    to settle that fast — which drifted the camera outside the hand-tuned crop
    windows and caused intermittent VLM misclassification."""
    subprocess.run(
        [str(_SCRIPT), head, "--ramp-time", "1.2", "--final-hold-time", "0.3"],
        cwd=str(_CIRCUS_DIR), check=True,
    )


def _localize_painting_head_turn(robot_name: str, gesture_id: str, depth_bbox: tuple | None) -> list[float] | None:
    """
    Original (pre-locomotion) localization: estimate the 3D position of the
    canvas from the live depth camera using the head yaw we ourselves just
    commanded (SCAN_YAW_RAD) — base-frame by construction, since the base
    never moves in this fallback. Uses pointcloud_centroid_in_bbox() (a
    real per-pixel point cloud over the bbox, outlier-filtered, centroid
    taken) rather than a single center-pixel/median-depth sample. Wrapped
    through to_world_frame() with WORLD_FRAME_ORIGIN_POSE before being returned
    to the caller, so the stored result is in the same WORLD frame the
    walking scan produces (see WORLD_FRAME_ORIGIN_POSE's docstring for why
    that's a safe assumption here).
    """
    if depth_bbox is None:
        return None
    depth_frame = read_latest_depth_frame(robot_name)
    if depth_frame is None:
        return None
    yaw = SCAN_YAW_RAD.get(gesture_id, 0.0)
    local_point = pointcloud_centroid_in_bbox(
        depth_frame, depth_bbox, head_yaw=yaw, head_pitch=_HEAD_PITCH_RAD,
    )
    if local_point is None:
        return None
    world_point = to_world_frame(local_point, WORLD_FRAME_ORIGIN_POSE)
    return [round(float(c), 4) for c in world_point]


def _explore_museum_head_turn_fallback(
    api_key: str, candidates: list[dict], results: dict, robot_name: str,
) -> dict:
    """
    Original scan mechanism, unchanged: robot stays put at its fixed spawn
    position and turns only the head through 6 preset directions. Used when
    explore_museum() can't get a live pose for the walking scan.
    """
    console.warn("[Explorer] Falling back to the head-turn scan (fixed position, no walking).")
    for step in _HEAD_TURN_SCAN_PLAN:
        gesture_id = step["gesture_id"]
        console.detail(f"\n[Explorer] → {step['label']}")

        _move_head(step["head"])

        time.sleep(0.5)

        frame = read_latest_frame(robot_name)
        if frame is None:
            console.error(f"[Explorer]   No frame — is circus running?")
            results[gesture_id] = {"title": "Unknown", "error": "no_frame"}
            _trace("perception", f"{gesture_id}: no camera frame — direction skipped")
            _export_scan_snapshot(results)
            continue

        xyz = _localize_painting_head_turn(robot_name, gesture_id, step.get("depth_bbox"))

        frame = _crop_frame(frame, step.get("crop_x"), step.get("crop_y"))

        _save_debug_png(frame, gesture_id)

        focus = step.get("focus", "center")
        console.detail(f"[Explorer]   Frame {frame.shape} focus={focus} → querying VLM...")
        try:
            info = analyze_painting_voted(frame, api_key, focus=focus, candidates=candidates)
        except Exception as e:
            console.error(f"[Explorer]   VLM error: {e}")
            info = {"title": "Unknown", "error": str(e)}

        info["gesture_id"] = gesture_id
        if xyz is not None:
            info["xyz"] = xyz
            console.detail(f"[Explorer]   3D position (world frame, via assumed home pose): "
                           f"[{xyz[0]:.2f}, {xyz[1]:.2f}, {xyz[2]:.2f}] m")
        if not info.get("error") and not info.get("painting_id"):
            info["error"] = "unrecognized_title"
        results[gesture_id] = info
        console.event(f"   [scan] {gesture_id}: {info.get('title','?')} — "
                      f"{info.get('artist','?')} [{info.get('movement','?')}]")

        yaw = SCAN_YAW_RAD.get(gesture_id, 0.0)
        if info.get("error"):
            _trace("perception", f"{gesture_id}: VLM error ({info['error']})")
        else:
            _trace("perception",
                   f'{gesture_id} (head yaw {yaw:+.2f} rad) → VLM: "{info.get("title", "?")}" '
                   f'by {info.get("artist", "?")}')
        _export_scan_snapshot(results)

    console.detail("\n[Explorer] Returning to neutral pose.")
    _neutral()

    return results


_SLOT_WORLD_XYZ = {
    "point_front_left":  [-1.33, -2.4, 1.4],
    "point_front_right": [-1.33, 2.4, 1.4],
    "point_right":       [3.0, 3.83, 1.4],
    "point_left":        [3.0, -3.83, 1.4],
    "point_right_mid":   [0.75, 3.83, 1.4],
    "point_left_mid":    [0.75, -3.83, 1.4],
}

_STAND_POSE = {
    "point_left_mid":    {"x": 0.75, "y": -2.8, "yaw": -math.pi / 2},
    "point_left":        {"x": 3.0, "y": -2.8, "yaw": -math.pi / 2},
    "point_front_left":  {"x": -0.3, "y": -2.4, "yaw": math.pi},
    "point_front_right": {"x": -0.3, "y": 2.4, "yaw": math.pi},
    "point_right":       {"x": 3.0, "y": 2.8, "yaw": math.pi / 2},
    "point_right_mid":   {"x": 0.75, "y": 2.8, "yaw": math.pi / 2},
}


_HOME_POSE = WORLD_FRAME_ORIGIN_POSE

_SCAN_ORDER = [
    "point_left", "point_left_mid", "point_front_left",
    "point_front_right", "point_right_mid", "point_right",
]

_DEFAULT_CROP_X = (0.15, 0.85)
_DEFAULT_CROP_Y = (0.10, 0.90)
_DEFAULT_DEPTH_BBOX = (160, 80, 480, 400)

_SCAN_PLAN = [
    {
        "gesture_id": gesture_id,
        "crop_x": _DEFAULT_CROP_X,
        "crop_y": _DEFAULT_CROP_Y,
        "depth_bbox": _DEFAULT_DEPTH_BBOX,
        "focus": "center",
        "label": f"standing in front of slot ({gesture_id})",
    }
    for gesture_id in _SCAN_ORDER
]


def _look_at_slot(gesture_id: str) -> tuple[float, float] | None:
    """
    Small head adjustment once standing at the slot: computes the local
    (base-frame) direction to the KNOWN slot position from the robot's
    ACTUAL current pose (not the planned stand pose — walking converges
    within a tolerance, not exactly), via ik_solver.solve_head_look_at.
    Commanded through the existing run_cpp_head_goal_action.sh (--yaw/
    --pitch raw override) — no new C++.

    Returns the commanded (yaw, pitch), or None if pose is unavailable.
    """
    pose = read_latest_pose()
    if pose is None:
        return None
    local_target = to_base_frame(_SLOT_WORLD_XYZ[gesture_id], pose)
    yaw, pitch = solve_head_look_at(local_target)
    subprocess.run(
        [str(_SCRIPT), "center", "--yaw", f"{yaw:.4f}", "--pitch", f"{pitch:.4f}",
         "--ramp-time", "1.0", "--final-hold-time", "0.3"],
        cwd=str(_CIRCUS_DIR), check=True,
    )
    return yaw, pitch


def _neutral() -> None:
    _move_head("center")


_SETTLE_POLL_INTERVAL_S = 0.1
_SETTLE_POSITION_EPS_M = 0.01
_SETTLE_YAW_EPS_RAD = 0.02
_SETTLE_STABLE_READS = 3
_SETTLE_MAX_WAIT_S = 2.5


def _wait_for_settle(robot_name: str, max_wait_s: float = _SETTLE_MAX_WAIT_S) -> None:
    """
    Blocks until the base's live pose stops changing between consecutive
    reads, instead of a fixed sleep.

    navigate_to_pose() (locomotion.py) only checks position/heading against
    _WAYPOINT_TOLERANCE/_HEADING_TOLERANCE_RAD — it says nothing about
    residual velocity, so it can return the moment the base first crosses
    that tolerance while still carrying momentum from the last walk burst.
    A fixed sleep has to be long enough for the WORST case, but _SCAN_ORDER
    isn't uniform: point_front_left -> point_front_right is a ~4.8m crossing
    through the middle of the room (the "cross at the entrance" leg),
    clearly longer than the ~2m hops between adjacent wall slots, so it
    plausibly still has more residual sway to damp out than 1.0s fixed
    covers — exactly the slot reported to intermittently fail to localize/
    misclassify. Polling actual pose stability adapts to whichever slot it
    is instead of guessing one constant for all of them.

    Falls back to max_wait_s if pose never stabilizes (e.g. read_latest_pose
    unavailable, or a jittery reading) so a scan can't hang forever on one
    slot — same fail-safe spirit as navigate_to_pose's own tolerance loop.
    """
    start = time.monotonic()
    stable = 0
    last = read_latest_pose(robot_name)
    while time.monotonic() - start < max_wait_s:
        time.sleep(_SETTLE_POLL_INTERVAL_S)
        pose = read_latest_pose(robot_name)
        if pose is None or last is None:
            stable = 0
            last = pose
            continue
        settled = (
            abs(pose["x"] - last["x"]) < _SETTLE_POSITION_EPS_M
            and abs(pose["y"] - last["y"]) < _SETTLE_POSITION_EPS_M
            and abs(pose["yaw"] - last["yaw"]) < _SETTLE_YAW_EPS_RAD
        )
        stable = stable + 1 if settled else 0
        last = pose
        if stable >= _SETTLE_STABLE_READS:
            return


_TRACE: list[dict] = []
_VIEWER_ON: bool = False
_ROBOT_NAME: str = "red_Booster-T1_0"


def _trace(stage: str, detail: str) -> None:
    _TRACE.append({"t": time.strftime("%H:%M:%S"), "stage": stage, "detail": detail})


def _scan_geometry(results: dict) -> list[dict]:
    """
    M (real xyz, world frame) for the viewer's top-down semantic-map panel,
    assembled from the scan IN PROGRESS — so the map fills in painting by
    painting, alongside the node-link graph, instead of staying "not scanned
    yet" until the guide is started.

    Built the same way guida_museale_core._export_kg_snapshot builds it from
    the SAVED scan: the position is keyed by gesture (a purely geometric fact
    — where the depth camera looked), then joined with the registry-grounded
    gesture->painting_id, so a VLM misidentification doesn't strand the
    position on an id nothing resolves to (see knowledge_graph._load_scan_xyz).
    Directions not localized yet (no depth reading, or not visited yet) simply
    have no dot.

    Shaped as the painting dicts export_graph_snapshot expects
    (robot_actions.target_xyz), so the same code path serves both callers.
    """
    registry = _load_registry(_DATA_DIR)
    title_by_pid = {c["id"]: c["title"] for c in _load_candidates()}
    geometry = []
    for gesture_id, info in (results or {}).items():
        if info.get("error"):
            continue
        xyz = info.get("xyz")
        pid = registry.get(gesture_id) or info.get("painting_id")
        if not xyz or not pid:
            continue
        geometry.append({
            "id": pid,
            "title": title_by_pid.get(pid, info.get("title", pid)),
            "robot_actions": {"pointing_skill": gesture_id, "target_xyz": xyz},
        })
    return geometry


def _safe_pose() -> dict | None:
    """Robot pose for the map panel — never let a shared-memory hiccup here
    take down the whole snapshot export (the graph is the important half)."""
    try:
        return read_latest_pose(_ROBOT_NAME)
    except Exception:
        return None


def _export_scan_snapshot(results: dict) -> None:
    if not _VIEWER_ON:
        return
    try:
        scene_map = build_spatial_map(results) if results else None
        g = build_graph_from_scan(_DATA_DIR, results, scene_map)
        export_graph_snapshot(g, _SNAPSHOT_PATH, pipeline_trace=_TRACE,
                              paintings=_scan_geometry(results),
                              robot_pose=_safe_pose())
    except Exception as e:
        console.error(f"[Explorer] Viewer snapshot export failed: {e}")


def explore_museum(
    api_key: str | None = None,
    robot_name: str | None = None,
    live_viewer: bool = True,
    serve_viewer: bool = True,
) -> dict:
    """
    Walks to each of the 6 painting slots in turn (left to right — see
    _SCAN_ORDER), looks at it, and VLM-identifies + depth-localizes it.
    Returns a dict keyed by gesture_id with the VLM analysis.

    Needs a live circus/T1 (rebuilt with pose_writer_) + T2 walk-ready
    (gm, hcm-start, gm) — walking needs ground-truth pose and a working
    locomotion channel; the old head-only scan didn't.

    live_viewer  — write scan snapshots so the viewer animates the graph
                   growing painting by painting.
    serve_viewer — also start our own HTTP server for viewer/. Set False when
                   another process already serves the page on the port (e.g.
                   guida_museale_core during a staff rescan): we then only write
                   the shared kg_snapshot.json it is already serving. Safe from a
                   write race because that caller is blocked waiting on us.
    """
    global _VIEWER_ON, _ROBOT_NAME
    if api_key is None:
        api_key = os.environ.get("GROQ_API_KEY", "")
    if robot_name is None:
        robot_name = "red_Booster-T1_0"
    _ROBOT_NAME = robot_name

    _warn_if_textures_stale()
    candidates = _load_candidates()
    results: dict = {}

    if live_viewer:
        _VIEWER_ON = start_viewer_server(_VIEWER_DIR, _VIEWER_PORT) if serve_viewer else True
    else:
        _VIEWER_ON = False
    _trace("init", "Walking scan started — base graph only, nothing discovered yet")
    _export_scan_snapshot(results)

    if read_latest_pose(robot_name) is None:
        console.warn("[Explorer] ⚠️  No ground-truth pose available — is circus running with the "
                     "rebuilt BoosterT1.h (pose_writer_)? Can't walk without it (the old head-only "
                     "scan didn't need it) — falling back to the fixed-position head-turn scan.")
        results = _explore_museum_head_turn_fallback(api_key, candidates, results, robot_name)
        _trace("kg_update", "Head-turn fallback scan complete")
        _export_scan_snapshot(results)
        return results

    for step in _SCAN_PLAN:
        gesture_id = step["gesture_id"]
        stand = _STAND_POSE[gesture_id]
        console.detail(f"\n[Explorer] → {step['label']}  (walking to {stand['x']:.1f},{stand['y']:.1f})")

        if not navigate_to_pose(stand["x"], stand["y"], stand["yaw"], robot_name=robot_name, verbose=False):
            console.warn(f"[Explorer]   Failed to reach standing spot — skipping this slot.")
            results[gesture_id] = {"title": "Unknown", "error": "navigation_failed"}
            _trace("perception", f"{gesture_id}: could not walk to standing spot — direction skipped")
            _export_scan_snapshot(results)
            continue

        look = _look_at_slot(gesture_id)
        head_yaw, head_pitch = look if look is not None else (0.0, 0.0)
        _wait_for_settle(robot_name)

        frame = read_latest_frame(robot_name)
        if frame is None:
            console.error(f"[Explorer]   No frame — is circus running?")
            results[gesture_id] = {"title": "Unknown", "error": "no_frame"}
            _trace("perception", f"{gesture_id}: no camera frame — direction skipped")
            _export_scan_snapshot(results)
            continue

        xyz = _localize_painting(robot_name, step.get("depth_bbox"), head_yaw, head_pitch)

        frame = _crop_frame(frame, step.get("crop_x"), step.get("crop_y"))

        _save_debug_png(frame, gesture_id)

        focus = step.get("focus", "center")
        console.detail(f"[Explorer]   Frame {frame.shape} focus={focus} → querying VLM...")
        try:
            info = analyze_painting_voted(frame, api_key, focus=focus, candidates=candidates)
        except Exception as e:
            console.error(f"[Explorer]   VLM error: {e}")
            info = {"title": "Unknown", "error": str(e)}

        info["gesture_id"] = gesture_id
        if xyz is not None:
            info["xyz"] = xyz
            console.detail(f"[Explorer]   3D position (world frame): "
                           f"[{xyz[0]:.2f}, {xyz[1]:.2f}, {xyz[2]:.2f}] m")
        if not info.get("error") and not info.get("painting_id"):
            info["error"] = "unrecognized_title"
        results[gesture_id] = info
        console.event(f"   [scan] {gesture_id}: {info.get('title','?')} — "
                      f"{info.get('artist','?')} [{info.get('movement','?')}]")

        if info.get("error"):
            _trace("perception", f"{gesture_id}: VLM error ({info['error']})")
        else:
            _trace("perception",
                   f'{gesture_id} (standing at {stand["x"]:.1f},{stand["y"]:.1f}) → VLM: '
                   f'"{info.get("title", "?")}" by {info.get("artist", "?")}')
        _export_scan_snapshot(results)


    console.detail(f"\n[Explorer] Walking back to home position "
                   f"({_HOME_POSE['x']:.1f},{_HOME_POSE['y']:.1f}).")
    navigate_to_pose(_HOME_POSE["x"], _HOME_POSE["y"], _HOME_POSE["yaw"],
                      robot_name=robot_name, verbose=False)
    _neutral()

    return results


def save_scan_results(results: dict) -> bool:
    """
    Persist a completed scan to disk: discovered_paintings.json (semantic
    enrichment + xyz per direction) and discovered_scene_map.yaml (spatial
    scene graph derived from it). Shared by main() and
    guida_museale_core.handle_staff_rescan() — the staff-triggered live
    rescan reuses the exact same persistence path as a standalone run,
    rather than duplicating it.

    Returns False (and writes nothing) if no direction was recognized: a
    failed scan (camera down, VLM unreachable...) must not overwrite good
    discovered_* files from a previous scan, since knowledge_graph.py prefers
    them over the static registry/legacy map.
    """
    ok_results = {k: v for k, v in results.items() if not v.get("error")}
    if not ok_results:
        console.error("\n[Explorer] ❌ Nessuna direzione riconosciuta — file discovered NON aggiornati.")
        console.error("[Explorer]    Controlla che circus sia in esecuzione e GROQ_API_KEY sia valida.")
        return False

    _OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(_OUTPUT, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    console.detail(f"\n[Explorer] Semantic enrichment saved → {_OUTPUT}")

    spatial_map = build_spatial_map(results)
    with open(_SPATIAL_OUTPUT, "w", encoding="utf-8") as f:
        yaml.safe_dump(spatial_map, f, sort_keys=False, allow_unicode=True)
    console.detail(f"[Explorer] Spatial scene graph saved → {_SPATIAL_OUTPUT}")
    return True


def build_spatial_map(scan_results: dict) -> dict:
    """
    HRAI 4 - Geometric2Semantic / HRAI 5 - Scene Graphs: turn the robot's own
    scan (head yaw per direction + VLM-recognized painting per direction)
    into a Room -> Wall -> Painting scene graph with leftOf/rightOf edges,
    DERIVED from perception instead of hand-typed. With the randomized
    hanging, the painting identity per direction comes from the VLM
    recognition itself (info["painting_id"]) — the robot genuinely does not
    know in advance what hangs where.

    The geometric derivation (yaw ordering, wall grouping by |yaw| — the
    latter a simplifying heuristic in lieu of depth-based plane fitting)
    lives in knowledge_graph.spatial_map_from_placement, shared with the
    registry-based fallback used when no scan has been run yet.
    """
    placement = {
        gesture_id: info["painting_id"]
        for gesture_id, info in scan_results.items()
        if not info.get("error") and info.get("painting_id")
    }
    return spatial_map_from_placement(placement)


def _localize_painting(
    robot_name: str, depth_bbox: tuple | None, head_yaw: float, head_pitch: float,
) -> list[float] | None:
    """
    Estimate the WORLD-frame 3D position of the canvas at this scan stop
    from the live depth camera. pointcloud_centroid_in_bbox() (a real
    per-pixel point cloud over the bbox, outlier-filtered, centroid taken —
    see camera_projection.py) is unchanged in its base-frame semantics and
    still returns a robot-BASE-frame point (from head_yaw/head_pitch — the
    angles _look_at_slot just commanded, known exactly since we set them)
    — but unlike the old fixed-base scan, the base is now in a DIFFERENT
    place at every stop, so that base-frame point means something different
    each time. Converted to world frame (via the robot's ACTUAL current base
    pose — read live, not the planned stand pose, since walking converges
    within a tolerance rather than exactly) so every stop's result is
    directly comparable, and so it can be converted back to whatever base
    frame the robot is in later, when pointing from the reception desk (see
    guida_museale_core.handle_recommend).
    """
    if depth_bbox is None:
        return None
    depth_frame = read_latest_depth_frame(robot_name)
    if depth_frame is None:
        return None
    pose = read_latest_pose(robot_name)
    if pose is None:
        return None
    local_point = pointcloud_centroid_in_bbox(
        depth_frame, depth_bbox, head_yaw=head_yaw, head_pitch=head_pitch,
    )
    if local_point is None:
        return None
    world_point = to_world_frame(local_point, pose)
    return [round(float(c), 4) for c in world_point]


def _crop_frame(frame: "np.ndarray", crop_x: tuple | None, crop_y: tuple | None = None) -> "np.ndarray":
    """
    Crop frame to isolate a specific painting.
    crop_x, crop_y: (start_frac, end_frac) as fractions of frame width/height,
    or None for no crop on that axis.

    crop_y matters more than it looks: these crop_x-only windows are narrow
    (~100-200 px) against the full 480 px height, so without a matching
    vertical crop the image sent to the VLM is extremely tall and narrow
    (~1:4). Resized to whatever square-ish input the vision model expects,
    that squashes the painting horizontally enough to cause misclassification
    on visually subtle styles (confirmed on the p5 pointillist canvas: same
    crop_x, 0/4 correct without crop_y, 4/4 correct with it added).
    """
    if crop_y is not None:
        h = frame.shape[0]
        y0 = int(h * crop_y[0])
        y1 = int(h * crop_y[1])
        frame = frame[y0:y1]
    if crop_x is None:
        return frame
    w = frame.shape[1]
    x0 = int(w * crop_x[0])
    x1 = int(w * crop_x[1])
    return frame[:, x0:x1]


def _save_debug_png(frame, gesture_id: str) -> None:
    try:
        import cv2
        debug_dir = _REPO_ROOT / "exchange" / "data" / "explorer_frames"
        debug_dir.mkdir(parents=True, exist_ok=True)
        path = debug_dir / f"{gesture_id}.png"
        cv2.imwrite(str(path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        console.detail(f"[Explorer]   Saved frame → {path}")
    except Exception:
        pass


def main() -> None:
    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        print("ERROR: GROQ_API_KEY not set.")
        return

    print("=" * 60)
    print("Museum Visual Explorer")
    print("Robot scans all 4 paintings and builds knowledge graph.")
    print("=" * 60)

    results = explore_museum(api_key)

    if not save_scan_results(results):
        return
    spatial_map = build_spatial_map(results)

    registry = _load_registry(_DATA_DIR)
    print("\nSummary:")
    for gesture_id, info in results.items():
        line = f"  {gesture_id:25s} → {info.get('title','?')} by {info.get('artist','?')}"
        if registry:
            expected = registry.get(gesture_id)
            observed = info.get("painting_id")
            if info.get("error"):
                line += "   [scan failed]"
            elif expected == observed:
                line += "   ✓ matches registry"
            else:
                line += f"   ✗ REGISTRY SAYS {expected} — VLM hallucination"
        print(line)
    order = " → ".join(o["id"] for o in spatial_map["scene"]["objects"])
    print(f"\nDiscovered left-to-right order: {order}")

    _trace("kg_update", f"Scan complete — spatial scene graph derived (order: {order})")
    _export_scan_snapshot(results)

    if _VIEWER_ON:
        print("\n🌐 Scansione completata — chiudo il viewer dell'explorer così "
              "la guida può prendere la porta e continuare ad aggiornare il grafo.")


if __name__ == "__main__":
    main()
