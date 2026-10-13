#!/usr/bin/env python3
"""
Simulated person detector for the Booster T1.

The T1 in circus carries an RGB-D head camera and nothing else: no lidar, no
person detector, no tracker. Obstacle avoidance around people therefore has no
sensor to be driven by, so the simulator supplies the measurement instead —
circus publishes the ground-truth footprint of every scene object tagged
`role: person` into /dev/shm once per simulation step
(circus/include/sensors/PeopleGroundTruthWriter.h), and this module turns that
raw truth into something the robot could plausibly have measured.

That second half is the whole point. Handing the planner the full list would
not be obstacle avoidance, it would be a static map with people baked into it:
the robot would route around someone standing behind a wall it has never
looked at, and the demonstration would prove nothing. So the raw feed passes
through a sensor model before anything else sees it:

  * RANGE — a person beyond SENSOR_RANGE_M is not reported at all.
  * LINE OF SIGHT — the segment robot->person is tested against the room's
    known static geometry (nav_map.get_obstacles_2d()) and against the other
    people; anything crossing it hides them, exactly as it would hide a real
    sensor's view. A visitor in the next alcove is not detected until the
    robot comes round the corner.
  * MEMORY — a track that stops being detected is kept for FORGET_SECONDS
    before being dropped, so brushing past an occluder does not make an
    obstacle blink out from under the planner mid-step.
  * DECAY — while remembered rather than seen, the belief widens at the speed
    that track was last measured moving (UNCERTAINTY_MAX_M). Memory of a
    walking person is not a position, it is a shrinking claim about one.

People who move are the same problem, not a different one. The simulator
publishes the position it reads out of MuJoCo each step, so a visitor walking
a scripted route (circus/include/PeopleWalker.h) arrives here as a track whose
position changes — the sensor model above is unchanged, and everything that
makes a moving person harder than a standing one shows up as a consequence of
it: they can walk out of range, behind the desk, or into a corridor the robot
had already planned through. What this module adds for them is a velocity
estimate per track, which is what lets a stale memory decay honestly and lets
locomotion.py tell "someone is in the way" from "someone is walking past".

360 degrees rather than the head camera's cone, chosen deliberately. A detector
bolted to a head that spends the whole museum circuit turned toward the
paintings would leave the robot blind sideways at precisely the moment it walks
past the visitors, and it would collide with people it simply never pointed the
camera at — a limitation of that mounting, not a property of obstacle avoidance
worth reproducing. Range-and-occlusion in every direction is what a real
base-mounted safety scanner gives, and it is the honest floor of what a robot
sent into a room with people in it would ship with.

Not published on purpose: identities. The shared-memory payload is positions
and radii, no names — a detector returns tracks, not the scene file's variable
names — so tracks are numbered in the order the simulator emits them.
"""
import math
import time
from pathlib import Path

import numpy as np

from nav_map import get_obstacles_2d
from shm_camera_reader import _read_shm_slot

_SCENE_OWNER = "scene"
_PEOPLE_SUFFIX = "people"
_PERSON_STRIDE = 5

SENSOR_RANGE_M = 5.0

FORGET_SECONDS = 8.0

UNCERTAINTY_MAX_M = 0.9

_VELOCITY_MIN_DT_S = 0.2
_VELOCITY_SMOOTHING = 0.6

PERSON_CLEARANCE_M = 0.15


def read_people_ground_truth() -> list[dict] | None:
    """
    The raw simulator feed: every person in the scene, with no sensor model
    applied. Returns a list of {"x","y","z","radius","height"}, or None when
    the channel is absent (circus not running, or a scene with no people).

    Callers that are steering the robot want detect_people()/PeopleTracker
    instead — this is the omniscient view, and using it directly would give the
    robot knowledge it has no way to have.
    """
    result = _read_shm_slot(_SCENE_OWNER, _PEOPLE_SUFFIX)
    if result is None:
        return None
    raw, width, height, channels = result
    if height != 1 or channels != _PERSON_STRIDE * 4 or width == 0:
        return None
    if raw.nbytes != width * _PERSON_STRIDE * 4:
        return None

    values = np.frombuffer(raw.tobytes(), dtype=np.float32).reshape(width, _PERSON_STRIDE)
    return [
        {"x": float(v[0]), "y": float(v[1]), "z": float(v[2]),
         "radius": float(v[3]), "height": float(v[4])}
        for v in values
    ]


def _segment_hits_circle(p0, p1, cx, cy, r) -> bool:
    """True if segment p0->p1 comes within r of (cx, cy)."""
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    fx, fy = p0[0] - cx, p0[1] - cy
    denom = dx * dx + dy * dy
    if denom < 1e-12:
        return fx * fx + fy * fy <= r * r
    t = max(0.0, min(1.0, -(fx * dx + fy * dy) / denom))
    px, py = p0[0] + t * dx - cx, p0[1] + t * dy - cy
    return px * px + py * py <= r * r


def _segment_hits_rect(p0, p1, xmin, xmax, ymin, ymax) -> bool:
    """
    True if segment p0->p1 intersects the axis-aligned box, by Liang-Barsky
    clipping: walk the four half-planes narrowing the parameter interval the
    segment is allowed to live in, and see whether anything is left.
    """
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    t0, t1 = 0.0, 1.0
    for p, q in ((-dx, p0[0] - xmin), (dx, xmax - p0[0]),
                 (-dy, p0[1] - ymin), (dy, ymax - p0[1])):
        if abs(p) < 1e-12:
            if q < 0:
                return False
            continue
        t = q / p
        if p < 0:
            if t > t1:
                return False
            t0 = max(t0, t)
        else:
            if t < t0:
                return False
            t1 = min(t1, t)
    return t0 <= t1


def _line_of_sight(p0, p1, occluders) -> bool:
    for occ in occluders:
        if occ[0] == "circle":
            if _segment_hits_circle(p0, p1, occ[1], occ[2], occ[3]):
                return False
        else:
            if _segment_hits_rect(p0, p1, occ[1], occ[2], occ[3], occ[4]):
                return False
    return True


def detect_people(
    pose: dict,
    people: list[dict] | None = None,
    max_range: float = SENSOR_RANGE_M,
    scene_path: Path | None = None,
) -> list[dict]:
    """
    One sensor reading: the people the robot can actually perceive from `pose`,
    as {"x","y","radius","distance","bearing"} (bearing in radians, relative to
    the robot's own heading, purely for logging — nothing steers by it).

    `pose` is read_latest_pose()'s dict. `people` defaults to the live feed;
    pass one in to evaluate the sensor model offline against a fixed scene.
    """
    if pose is None:
        return []
    if people is None:
        people = read_people_ground_truth()
    if not people:
        return []

    origin = (pose["x"], pose["y"])
    static_obstacles = get_obstacles_2d(scene_path)

    detections = []
    for i, person in enumerate(people):
        target = (person["x"], person["y"])
        distance = math.hypot(target[0] - origin[0], target[1] - origin[1])
        if distance > max_range:
            continue

        others = [("circle", p["x"], p["y"], max(p["radius"] - 0.02, 0.01))
                  for j, p in enumerate(people) if j != i]
        if distance > 1e-6:
            ux = (target[0] - origin[0]) / distance
            uy = (target[1] - origin[1]) / distance
            reach = max(distance - person["radius"], 0.0)
            probe = (origin[0] + ux * reach, origin[1] + uy * reach)
        else:
            probe = target
        if not _line_of_sight(origin, probe, static_obstacles + others):
            continue

        bearing = math.atan2(target[1] - origin[1], target[0] - origin[0]) - pose["yaw"]
        detections.append({
            "track": f"person_{i}",
            "x": target[0], "y": target[1], "radius": person["radius"],
            "distance": distance,
            "bearing": (bearing + math.pi) % (2 * math.pi) - math.pi,
        })
    return detections


class PeopleTracker:
    """
    Short-term memory over detect_people(): what the robot currently believes
    is standing in the room, which is not the same as what it can see this
    instant.

    Without it the obstacle set would flicker — a visitor drops out the moment
    the reception desk passes between them and the robot, the next replan routes
    straight back through where they are standing, and the robot walks into
    someone it saw two seconds ago. Keeping a track for FORGET_SECONDS is the
    cheapest form of the persistence any real people layer has.
    """

    def __init__(self, forget_seconds: float = FORGET_SECONDS, max_range: float = SENSOR_RANGE_M):
        self.forget_seconds = forget_seconds
        self.max_range = max_range
        self._tracks: dict[str, dict] = {}

    def update(self, pose: dict, people: list[dict] | None = None) -> list[dict]:
        """Fold one sensor reading in and expire stale tracks. Returns the
        detections from THIS reading (not the whole belief — see known()).

        Each track also carries a velocity estimated from consecutive
        sightings, because the identity of the ith person in the feed is
        stable: a track that moves is the same track at a new position, not a
        new track. Nothing steers by the velocity directly — it is what tells
        known() how fast a remembered position is going stale, and locomotion
        whether waiting for somebody to pass is a plausible thing to do."""
        now = time.monotonic()
        detections = detect_people(pose, people=people, max_range=self.max_range)
        for det in detections:
            prev = self._tracks.get(det["track"])
            vx, vy = 0.0, 0.0
            if prev is not None:
                dt = now - prev["last_seen"]
                if dt >= _VELOCITY_MIN_DT_S:
                    a = _VELOCITY_SMOOTHING
                    vx = a * (det["x"] - prev["x"]) / dt + (1 - a) * prev["vx"]
                    vy = a * (det["y"] - prev["y"]) / dt + (1 - a) * prev["vy"]
                else:
                    vx, vy = prev["vx"], prev["vy"]
            self._tracks[det["track"]] = {
                **det, "last_seen": now, "vx": vx, "vy": vy, "speed": math.hypot(vx, vy),
            }
        for track, entry in list(self._tracks.items()):
            if now - entry["last_seen"] > self.forget_seconds:
                del self._tracks[track]
        return detections

    def known(self) -> list[dict]:
        """
        Everything currently believed to be in the room, seen or remembered.

        "radius" is the belief's radius, not the person's: for a track not
        currently in sight it is the body radius grown by how far that person
        could have walked since (see UNCERTAINTY_MAX_M). "body_radius" keeps
        the raw measurement for anyone who needs it. Callers get the grown
        figure by default on purpose — planning around where somebody might be
        is the entire job of a people layer, and a caller that has to remember
        to ask for it is a caller that will forget.
        """
        now = time.monotonic()
        belief = []
        for track in self._tracks.values():
            age = now - track["last_seen"]
            if age > self.forget_seconds:
                continue
            uncertainty = min(track["speed"] * age, UNCERTAINTY_MAX_M)
            belief.append({**track,
                           "body_radius": track["radius"],
                           "radius": track["radius"] + uncertainty,
                           "age": age,
                           "uncertainty": uncertainty})
        return belief

    def obstacles(self, robot_radius: float, clearance: float = PERSON_CLEARANCE_M) -> list[tuple]:
        """
        The people layer as plan_path() extra_obstacles: ("circle", x, y, r)
        with r already inflated by the robot's own radius, since plan_path
        rasterizes extra obstacles at face value.
        """
        return [("circle", t["x"], t["y"], t["radius"] + robot_radius + clearance)
                for t in self.known()]

    def forget_all(self) -> None:
        self._tracks.clear()


_DEFAULT_TRACKER: PeopleTracker | None = None


def default_tracker() -> PeopleTracker:
    """Process-wide tracker, so what the robot learned walking to one painting
    is still known when it sets off for the next — a robot that forgets the room
    between goals is not one that has a people layer."""
    global _DEFAULT_TRACKER
    if _DEFAULT_TRACKER is None:
        _DEFAULT_TRACKER = PeopleTracker()
    return _DEFAULT_TRACKER


if __name__ == "__main__":
    import sys

    from shm_camera_reader import read_latest_pose

    if "--watch" in sys.argv:
        args = [a for a in sys.argv[1:] if a != "--watch"]
        seconds = float(args[0]) if args else 20.0
        tracker = PeopleTracker()
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            pose = read_latest_pose()
            if pose is None:
                print("No robot pose — is circus running?")
                raise SystemExit(1)
            tracker.update(pose)
            rows = " | ".join(
                f"{t['track']}: ({t['x']:+.2f},{t['y']:+.2f}) {t['speed']:.2f} m/s"
                + (f" r={t['radius']:.2f} (unseen {t['age']:.1f}s)" if t["age"] > 0.5 else "")
                for t in sorted(tracker.known(), key=lambda t: t["track"]))
            print(f"[{time.monotonic() - deadline + seconds:5.1f}s] "
                  + (rows or "nobody in sight"))
            time.sleep(0.5)
        raise SystemExit(0)

    truth = read_people_ground_truth()
    if truth is None:
        print("No people channel (is circus running with a scene that has "
              "`role: person` objects, and a binary rebuilt with "
              "PeopleGroundTruthWriter?)")
        raise SystemExit(1)

    print(f"Simulator ground truth: {len(truth)} people")
    for i, p in enumerate(truth):
        print(f"  person_{i}: ({p['x']:.2f}, {p['y']:.2f})  r={p['radius']:.2f}  h={p['height']:.2f}")

    pose = read_latest_pose()
    if pose is None:
        print("\nNo robot pose — cannot apply the sensor model.")
        raise SystemExit(1)

    print(f"\nRobot at ({pose['x']:.2f}, {pose['y']:.2f}) yaw={pose['yaw']:.2f}")
    detections = detect_people(pose)
    print(f"Detected from here: {len(detections)}/{len(truth)} "
          f"(range {SENSOR_RANGE_M} m + line of sight)")
    for det in detections:
        print(f"  {det['track']}: ({det['x']:.2f}, {det['y']:.2f})  "
              f"{det['distance']:.2f} m  bearing {math.degrees(det['bearing']):+.0f} deg")
    hidden = {f"person_{i}" for i in range(len(truth))} - {d["track"] for d in detections}
    if hidden:
        print(f"  hidden (occluded or out of range): {', '.join(sorted(hidden))}")
