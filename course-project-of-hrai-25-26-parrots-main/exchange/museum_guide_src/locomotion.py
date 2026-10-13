#!/usr/bin/env python3
"""
Closed-loop waypoint navigation for the museum guide robot.

HRAI - lecture1 (Intro + Benchmarking) "Common Functionality Benchmark
Tools" slide lists "Planning: pathfinding" and "Control: trajectory
tracking" as separate functionality domains — this module is where they
meet: nav_map.py + path_planner.py (the plan) driven closed-loop against
read_latest_pose() (ground-truth localization — no SLAM/LIDAR needed in
simulation, see shm_camera_reader.read_latest_pose's docstring) and
run_cpp_walk_leg.sh (the actuation primitive, same role
arm_head_goal_ramp_actions.cpp plays for pointing gestures).

Obstacle avoidance comes in two layers, because the room comes in two parts.
The walls, the desk and the bench were surveyed once and never move, so they
are planned around from the map. People were not: they are kept out of that map
on purpose (nav_map.PERCEIVED_ROLES) and have to be SENSED while walking, which
on a T1 with no lidar and no person detector means the simulator publishes their
ground truth and people_sensor.py decides — by range and line of sight — how
much of it the robot is allowed to know. Underneath both sits the depth-camera
strip, the last-resort reactive stop for anything neither layer saw coming.

Some of those people walk, and that is what makes the second layer a layer
rather than a second map. Three things here exist only because of them:
_person_blocking compares each track against the position the CURRENT plan
allowed for it, so somebody already routed around who then steps into the path
is caught rather than exempted; the belief the plan is built on decays with
time-since-seen, so a remembered position of a walking person is not treated as
a fact; and a blocker who is measurably moving buys a short stand-still before
the robot commits to a detour, because you cannot out-plan a moving obstacle by
routing around where it used to be. None of it changes what happens in a room
of stationary visitors: every one of the three is keyed on measured motion, and
a track that has never moved takes the old path through all of them.

Standalone capability, not yet wired into the museum-guide dialogue — see
navigation_demo.py for a CLI to exercise it directly.
"""
import math
import subprocess
from pathlib import Path

import numpy as np

import people_sensor
from nav_map import build_occupancy_grid
from path_planner import plan_path
from people_sensor import PERSON_CLEARANCE_M, _segment_hits_circle
from shm_camera_reader import read_latest_depth_frame, read_latest_pose

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CIRCUS_DIR = _REPO_ROOT / "circus"
_WALK_SCRIPT = _CIRCUS_DIR / "debug_scripts" / "run_cpp_walk_leg.sh"

_ROBOT_RADIUS = 0.30

_MAX_VX = 0.35
_MIN_VX = 0.15
_MAX_VYAW = 0.3
_MIN_VYAW = 0.25
_MIN_TURN_BURST_S = 0.3

_FALLEN_HEIGHT_M = 0.45


def _has_fallen(pose: dict) -> bool:
    return pose["z"] < _FALLEN_HEIGHT_M
_HEADING_KP = 1.2
_POSITION_KP = 0.8
_HEADING_GATE_RAD = 0.6
_WAYPOINT_TOLERANCE = 0.15
_BURST_DURATION = 1.0

_OBSTACLE_STOP_METERS = 0.5
_FORWARD_STRIP_FRAC = 0.15

_DEPTH_FAR_SENTINEL = 9.5

_OBSTACLE_PERCENTILE = 10
_MIN_VALID_DEPTH_PIXELS = 200

def _wrap_angle(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def _turn_command(heading_error: float) -> tuple[float, float]:
    """(vyaw, burst_duration) for a pure turn of `heading_error` radians.

    Above the deadzone this is the plain proportional command over a full
    burst. Below it, the rate is held at _MIN_VYAW and the burst is cut short
    to deliver the same rotation — see _MIN_VYAW for what happens when a
    sub-deadzone rate is sent instead.
    """
    vyaw = max(-_MAX_VYAW, min(_MAX_VYAW, _HEADING_KP * heading_error))
    if abs(vyaw) >= _MIN_VYAW:
        return vyaw, _BURST_DURATION
    vyaw = math.copysign(_MIN_VYAW, heading_error)
    duration = min(_BURST_DURATION, max(_MIN_TURN_BURST_S, abs(heading_error) / _MIN_VYAW))
    return vyaw, duration


class _WalkSession:
    """
    Keeps ONE walk_leg process alive for an entire navigate_to() call.

    Observed live: an earlier design spawned a fresh walk_leg process (and
    therefore a fresh ChangeMode(kWalking) call) for every single velocity
    burst — the robot never got past the walking-mode transition (visible
    knee-lift, no net translation), because every burst re-triggered that
    transition before the previous one had time to settle into an actual
    gait. teleop.cpp calls ChangeMode(kWalking) exactly ONCE and then sends
    Move() repeatedly for the whole session — this mirrors that, via
    run_cpp_walk_leg.sh's persistent "docker exec -i" mode (walk_leg.cpp
    reads velocity commands from stdin in a loop).
    """

    def __init__(self):
        self._proc = subprocess.Popen(
            [str(_WALK_SCRIPT)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, bufsize=1,
        )

    def burst(self, vx: float, vy: float, vyaw: float, duration: float = _BURST_DURATION) -> bool:
        """Sends one velocity command and blocks until walk_leg confirms it
        finished (reads its "done" acknowledgment line) — so the caller's
        next pose read reflects the full burst, not a partial one."""
        if self._proc.poll() is not None or self._proc.stdin is None:
            return False
        try:
            self._proc.stdin.write(f"{vx:.3f} {vy:.3f} {vyaw:.3f} {duration:.2f}\n")
            self._proc.stdin.flush()
            while True:
                line = self._proc.stdout.readline()
                if not line:
                    return False
                if line.startswith("walk_leg: done"):
                    return True
        except (BrokenPipeError, OSError):
            return False

    def close(self) -> None:
        if self._proc.poll() is None:
            try:
                if self._proc.stdin:
                    self._proc.stdin.write("STOP\n")
                    self._proc.stdin.flush()
            except (BrokenPipeError, OSError):
                pass
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.terminate()


def _forward_obstacle_distance(robot_name: str) -> float | None:
    """
    Distance to the nearest thing occupying a real share of the view straight
    ahead, or None when there is nothing to report. See _OBSTACLE_PERCENTILE
    for why this is a percentile over finite readings rather than a median
    over everything.
    """
    depth = read_latest_depth_frame(robot_name)
    if depth is None:
        return None
    h, w = depth.shape
    half = max(1, int(w * _FORWARD_STRIP_FRAC))
    strip = depth[:, w // 2 - half: w // 2 + half]
    valid = strip[(strip > 0) & (strip < _DEPTH_FAR_SENTINEL)]
    if valid.size < _MIN_VALID_DEPTH_PIXELS:
        return None
    return float(np.percentile(valid, _OBSTACLE_PERCENTILE))


def _people_layer(tracker, start_xy: tuple[float, float]) -> tuple[list[tuple], dict]:
    """
    The tracker's belief turned into planner input: ("circle", x, y, r)
    obstacles, plus, per track id, the position that circle was drawn around.

    Positions and not just ids, because a plan is only clear of a person while
    that person is where they were when it was made. Keeping the snapshot is
    what lets _person_blocking tell "already routed around" from "already
    routed around, and has since walked somewhere else" — without it a visitor
    who steps into the planned path after the plan is made is invisible to the
    very check that exists to catch them.

    A person whose inflated disc already contains the robot is dropped. It is
    what every real costmap does with its own footprint, and here it is load
    bearing: plan_path() refuses outright when the start cell is occupied, so
    one visitor standing a little too close would not slow the robot down, it
    would end navigation entirely with "nothing to plan from".
    """
    if tracker is None:
        return [], {}
    obstacles, planned = [], {}
    for track in tracker.known():
        radius = track["radius"] + _ROBOT_RADIUS + PERSON_CLEARANCE_M
        if math.hypot(track["x"] - start_xy[0], track["y"] - start_xy[1]) <= radius:
            continue
        obstacles.append(("circle", track["x"], track["y"], radius))
        planned[track["track"]] = (track["x"], track["y"])
    return obstacles, planned


_PLAN_STALE_M = 0.5 * PERSON_CLEARANCE_M + 0.25


def _person_blocking(pose: dict, waypoint: tuple[float, float], tracker, planned_tracks: dict) -> dict | None:
    """
    A person standing across the leg the robot is about to walk, whom the
    CURRENT plan does not already route around.

    Restricting it to tracks the plan didn't know about is what keeps this from
    looping: the path in hand was planned clear of everyone in planned_tracks,
    so re-reporting them would abort the same leg forever. Only a genuinely new
    detection is worth throwing the plan away for — someone who came into view
    or out from behind the desk since the plan was made, OR someone who was in
    the plan and has since walked _PLAN_STALE_M away from where it left room
    for them. The second case is the whole of dynamic obstacle avoidance: a
    check keyed on identity alone gives a moving visitor a permanent exemption
    from the one test that would notice them stepping into the robot's path.
    """
    if tracker is None:
        return None
    here = (pose["x"], pose["y"])
    for track in tracker.known():
        planned_at = planned_tracks.get(track["track"])
        if planned_at is not None and math.hypot(track["x"] - planned_at[0],
                                                 track["y"] - planned_at[1]) < _PLAN_STALE_M:
            continue
        radius = track["radius"] + _ROBOT_RADIUS + PERSON_CLEARANCE_M
        if math.hypot(track["x"] - here[0], track["y"] - here[1]) <= radius:
            continue
        if _segment_hits_circle(here, waypoint, track["x"], track["y"], radius):
            return track
    return None


_MOVING_SPEED_MS = 0.08

_YIELD_SECONDS = 1.5
_MAX_YIELDS = 4


_MAX_BURSTS_PER_WAYPOINT = 40


def _follow_path(
    path: list[tuple[float, float]], tolerance: float, robot_name: str,
    session: _WalkSession, verbose: bool, tracker=None, planned_tracks: dict | None = None,
) -> tuple[str, object] | None:
    """
    Walks `path` waypoint by waypoint (path[0] is the robot's own starting
    position, skipped). Returns None once the final waypoint is reached, or a
    (reason, payload) pair when the walk is abandoned to replan:

      ("depth",  (x, y))  — the depth camera saw something close ahead that
                            nothing in the plan accounted for; the caller
                            remembers the spot, since nothing else will.
      ("person", track)   — the people layer detected somebody across the next
                            leg. No coordinates come back: the tracker already
                            holds them and the next plan reads them from there.
    """
    for wp in path[1:]:
        bursts_this_waypoint = 0
        while True:
            bursts_this_waypoint += 1
            if bursts_this_waypoint > _MAX_BURSTS_PER_WAYPOINT:
                if verbose:
                    print(f"[locomotion] Stuck on waypoint {wp} after "
                          f"{_MAX_BURSTS_PER_WAYPOINT} bursts — replanning from here.")
                return ("stuck", None)

            pose = read_latest_pose(robot_name)
            if pose is None:
                if verbose:
                    print("[locomotion] Lost pose mid-walk — stopping.")
                return None
            if _has_fallen(pose):
                if verbose:
                    print(f"[locomotion] Robot is down (roll={pose['roll']:.2f}, "
                          f"pitch={pose['pitch']:.2f}) — no point walking. Stand it up "
                          f"and retry.")
                return ("fallen", None)

            dx, dy = wp[0] - pose["x"], wp[1] - pose["y"]
            dist = math.hypot(dx, dy)
            if dist < tolerance:
                break

            if tracker is not None:
                tracker.update(pose)
                blocker = _person_blocking(pose, wp, tracker, planned_tracks or {})
                if blocker is not None:
                    if verbose:
                        moving = (f", walking at {blocker['speed']:.2f} m/s"
                                  if blocker.get("speed", 0.0) >= _MOVING_SPEED_MS else "")
                        print(f"[locomotion] Person {blocker['track']} at "
                              f"({blocker['x']:.2f},{blocker['y']:.2f}) is across the next leg "
                              f"({blocker['distance']:.2f} m away{moving}) — replanning around them.")
                    return ("person", blocker)

            target_heading = math.atan2(dy, dx)
            heading_error = _wrap_angle(target_heading - pose["yaw"])
            vyaw = max(-_MAX_VYAW, min(_MAX_VYAW, _HEADING_KP * heading_error))
            driving_forward = abs(heading_error) < _HEADING_GATE_RAD

            if driving_forward:
                obstacle_dist = _forward_obstacle_distance(robot_name)
                if obstacle_dist is not None and obstacle_dist < _OBSTACLE_STOP_METERS:
                    ox = pose["x"] + obstacle_dist * math.cos(pose["yaw"])
                    oy = pose["y"] + obstacle_dist * math.sin(pose["yaw"])
                    if verbose:
                        print(f"[locomotion] Unexpected obstacle ~{obstacle_dist:.2f}m ahead "
                              f"(~({ox:.2f},{oy:.2f})) — replanning.")
                    return ("depth", (ox, oy))

            vx = max(_MIN_VX, min(_MAX_VX, _POSITION_KP * dist)) if driving_forward else 0.0

            if verbose:
                print(f"[locomotion]   pose=({pose['x']:.2f},{pose['y']:.2f},"
                      f"yaw={pose['yaw']:.2f}) -> wp=({wp[0]:.2f},{wp[1]:.2f}) "
                      f"dist={dist:.2f} vx={vx:.2f} vyaw={vyaw:.2f}")
            if not session.burst(vx, 0.0, vyaw):
                if verbose:
                    print("[locomotion] walk session died mid-burst — stopping.")
                return None

    return None


def navigate_to(
    goal_xy: tuple[float, float],
    tolerance: float = _WAYPOINT_TOLERANCE,
    max_replans: int = 8,
    robot_name: str = "red_Booster-T1_0",
    verbose: bool = True,
    session: "_WalkSession | None" = None,
    avoid_people: bool = True,
) -> bool:
    """
    Plans a path over the room's known static map (nav_map.py) and walks
    it, correcting heading/position every burst from the robot's own
    ground-truth pose, replanning around anything the depth camera sees
    that the static map didn't expect. One persistent walking session for
    the whole call (see _WalkSession) — ChangeMode(kWalking) happens once.

    avoid_people — run the people layer (people_sensor.py): visitors are sensed
    while walking, folded into the plan as obstacles, and a leg is abandoned
    mid-way when somebody new turns up across it — or when somebody the plan
    HAD accounted for walks somewhere else (see _person_blocking). They are
    deliberately absent from the static map (nav_map.PERCEIVED_ROLES), so with
    this off the robot plans straight through them. Set False only to measure
    what the avoidance is worth by taking it away.

    The replan budget is per call and deliberately not tight: against a visitor
    who is walking, replanning is the loop doing its job rather than a symptom
    of it failing, and each pass is one plan, not a busy loop.

    session — pass an existing _WalkSession (e.g. from navigate_to_pose(),
    which also needs one for its final turn-in-place) to reuse it instead of
    opening/closing a new one: every fresh session re-triggers the
    walking-mode transition (see _WalkSession's docstring), so chaining
    calls through the same session avoids paying that cost twice. When None
    (the default), a session is opened and closed locally, unchanged from
    before.

    Returns True if the goal was reached within tolerance, False otherwise
    (no path found, too many replans, or pose/circus unavailable).
    """
    grid, origin, resolution = build_occupancy_grid(robot_radius=_ROBOT_RADIUS)
    extra_obstacles: list[tuple] = []
    replans = 0
    yields = 0
    tracker = people_sensor.default_tracker() if avoid_people else None

    pose = read_latest_pose(robot_name)
    if pose is None:
        if verbose:
            print("[locomotion] No pose available — is circus running with the "
                  "rebuilt binary (see BoosterT1.h's pose_writer_)?")
        return False

    owns_session = session is None
    if session is None:
        session = _WalkSession()
    try:
        while True:
            pose = read_latest_pose(robot_name)
            if pose is None:
                if verbose:
                    print("[locomotion] Lost pose — stopping.")
                return False

            start_xy = (pose["x"], pose["y"])
            if tracker is not None:
                seen = tracker.update(pose)
                if verbose and seen:
                    print("[locomotion] People in sight: "
                          + ", ".join(f"{d['track']}@{d['distance']:.1f}m" for d in seen))
            people_obstacles, planned_tracks = _people_layer(tracker, start_xy)
            try:
                path = plan_path(start_xy, goal_xy, grid, origin, resolution,
                                  extra_obstacles=extra_obstacles + people_obstacles)
            except ValueError as e:
                if verbose:
                    print(f"[locomotion] {e}")
                return False
            if path is None:
                if verbose:
                    print(f"[locomotion] No path from {start_xy} to {goal_xy}")
                return False
            path[-1] = goal_xy
            if verbose:
                print(f"[locomotion] Path: {len(path)} waypoints from {start_xy} to {goal_xy}"
                      + (f", clear of {len(people_obstacles)} person(s)" if people_obstacles else ""))

            outcome = _follow_path(path, tolerance, robot_name, session, verbose,
                                    tracker, planned_tracks)

            if outcome is None:
                pose = read_latest_pose(robot_name)
                reached = (pose is not None
                           and math.hypot(pose["x"] - goal_xy[0], pose["y"] - goal_xy[1]) < tolerance)
                if verbose:
                    print(f"[locomotion] {'Reached goal.' if reached else 'Stopped without reaching goal.'}")
                return reached

            reason, payload = outcome
            if reason == "fallen":
                return False
            if reason == "stuck":
                pass
            elif reason == "depth":
                extra_obstacles.append(("circle", payload[0], payload[1], 0.35))
            elif reason == "person":
                if payload.get("speed", 0.0) >= _MOVING_SPEED_MS and yields < _MAX_YIELDS:
                    yields += 1
                    if verbose:
                        print(f"[locomotion] Letting {payload['track']} pass "
                              f"({yields}/{_MAX_YIELDS}) — holding {_YIELD_SECONDS:.1f}s.")
                    session.burst(0.0, 0.0, 0.0, _YIELD_SECONDS)
                    pose = read_latest_pose(robot_name)
                    if pose is not None:
                        tracker.update(pose)
                    continue
            replans += 1
            if replans > max_replans:
                if verbose:
                    print("[locomotion] Too many replans — giving up.")
                return False
    finally:
        if owns_session:
            session.close()


_HEADING_TOLERANCE_RAD = 0.1
_MAX_TURN_BURSTS = 20


def _turn_to_heading(
    target_yaw: float, session: _WalkSession, robot_name: str, verbose: bool,
) -> bool:
    """Turns in place until facing target_yaw within _HEADING_TOLERANCE_RAD.
    Position isn't actively held (vx=0 throughout) — walking in place turns
    on a real bipedal gait isn't perfectly stationary, but the drift over a
    few burst is small relative to _WAYPOINT_TOLERANCE."""
    for _ in range(_MAX_TURN_BURSTS):
        pose = read_latest_pose(robot_name)
        if pose is None:
            if verbose:
                print("[locomotion] Lost pose during final turn — stopping.")
            return False
        heading_error = _wrap_angle(target_yaw - pose["yaw"])
        if abs(heading_error) < _HEADING_TOLERANCE_RAD:
            return True
        vyaw, duration = _turn_command(heading_error)
        if verbose:
            print(f"[locomotion]   turning: yaw={pose['yaw']:.2f} -> target={target_yaw:.2f} "
                  f"error={heading_error:.2f} vyaw={vyaw:.2f} for {duration:.2f}s")
        if not session.burst(0.0, 0.0, vyaw, duration):
            if verbose:
                print("[locomotion] walk session died mid-turn — stopping.")
            return False

    pose = read_latest_pose(robot_name)
    if pose is not None and abs(_wrap_angle(target_yaw - pose["yaw"])) < _HEADING_TOLERANCE_RAD:
        return True
    if verbose:
        print("[locomotion] Final turn didn't converge within the burst budget.")
    return False


def navigate_to_pose(
    goal_x: float, goal_y: float, goal_yaw: float,
    tolerance: float = _WAYPOINT_TOLERANCE,
    heading_tolerance: float = _HEADING_TOLERANCE_RAD,
    robot_name: str = "red_Booster-T1_0",
    verbose: bool = True,
    avoid_people: bool = True,
) -> bool:
    """
    navigate_to() followed by a turn-in-place to face goal_yaw (radians,
    world frame) — navigate_to() alone only guarantees position, the final
    facing direction is whatever the last leg of the walk happened to be.
    Needed for the walking painting scan (museum_explorer.py): the robot
    must not just arrive near a painting, it must end up SQUARE-ON to it
    for the camera framing to make sense.

    One shared _WalkSession for both the walk and the turn — see
    navigate_to()'s `session` parameter docstring for why that matters.
    """
    session = _WalkSession()
    try:
        reached = navigate_to(
            (goal_x, goal_y), tolerance=tolerance, robot_name=robot_name,
            verbose=verbose, session=session, avoid_people=avoid_people,
        )
        if not reached:
            return False
        return _turn_to_heading(goal_yaw, session, robot_name, verbose)
    finally:
        session.close()
