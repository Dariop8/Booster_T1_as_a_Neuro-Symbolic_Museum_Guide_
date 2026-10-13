#!/usr/bin/env python3
"""
Static 2D obstacle map of the museum room, derived from the same scene YAML
that builds the MuJoCo world (circus/resources/scenes/museo_receptionist.yaml)
— the room layout is authored by hand, so there is no need to discover it via
SLAM/LIDAR: it's already fully known.

HRAI - lecture1 (Intro + Benchmarking), "Common Functionality Benchmark
Tools" slide lists "Planning: pathfinding" as a functionality domain — this
module is the map half of that; path_planner.py is the pathfinding half.

Every object with an explicit "size" (box half-extents, or cylinder/sphere
radius) whose vertical extent overlaps the robot's own body height band is
treated as a floor obstacle; objects without "size" (the painting canvas
meshes, sized by their .obj file instead) and objects entirely above/below
the band (floor slab, door lintel, wall-mounted plaques/frames — thin and
flush against the wall, so even where they technically overlap the band
they add no obstacle footprint beyond what the wall itself already blocks)
are skipped. Generic by construction: no hardcoded object names, works for
any object added to the scene later.
"""
from pathlib import Path

import numpy as np
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_SCENE_PATH = _REPO_ROOT / "circus" / "resources" / "scenes" / "museo_receptionist.yaml"

ROOM_BOUNDS = {"x_min": -1.5, "x_max": 6.0, "y_min": -4.0, "y_max": 4.0}

_ROBOT_BODY_Z_MIN = 0.05
_ROBOT_BODY_Z_MAX = 1.2

PERCEIVED_ROLES = frozenset({"person"})

def _load_scene_objects(scene_path: Path) -> list[dict]:
    scene = yaml.safe_load(scene_path.read_text(encoding="utf-8"))
    return scene.get("objects", [])


def _object_footprint_2d(obj: dict) -> tuple | None:
    """
    Returns a 2D obstacle primitive for `obj`, or None if it isn't a floor
    obstacle:
      ("rect", x_min, x_max, y_min, y_max)  — box-type objects
      ("circle", cx, cy, radius)            — cylinder/sphere-type objects
    """
    if obj.get("role") in PERCEIVED_ROLES:
        return None

    size = obj.get("size")
    if size is None:
        return None

    x, y, z = obj["position"]
    sx, sy, sz = size
    z_lo, z_hi = z - sz, z + sz
    if z_hi < _ROBOT_BODY_Z_MIN or z_lo > _ROBOT_BODY_Z_MAX:
        return None

    if obj.get("type") in ("cylinder", "sphere"):
        radius = max(sx, sy)
        return ("circle", x, y, radius)
    return ("rect", x - sx, x + sx, y - sy, y + sy)


def get_obstacles_2d(scene_path: Path | None = None) -> list[tuple]:
    """All floor-level obstacle primitives in the scene (see _object_footprint_2d)."""
    scene_path = scene_path or _DEFAULT_SCENE_PATH
    obstacles = []
    for obj in _load_scene_objects(scene_path):
        fp = _object_footprint_2d(obj)
        if fp is not None:
            obstacles.append(fp)
    return obstacles


def get_perceived_objects(scene_path: Path | None = None) -> list[dict]:
    """
    Ground truth for the objects deliberately left OUT of the map (see
    PERCEIVED_ROLES), as {"name", "role", "x", "y", "radius", "motion"}.

    The planner must never call this — it exists so a test can score whether
    the robot's own perception found something whose real position it was
    never told.

    For anyone carrying a "motion" route, x/y is the SPAWN position and nothing
    more: they are somewhere along that route by the time the robot is walking.
    Scoring a run against them means comparing what the robot perceived to
    people_sensor.read_people_ground_truth() at that moment, not to this file —
    which is the honest situation, since a scene file cannot be the answer key
    for where a moving person is.
    """
    scene_path = scene_path or _DEFAULT_SCENE_PATH
    found = []
    for obj in _load_scene_objects(scene_path):
        if obj.get("role") not in PERCEIVED_ROLES:
            continue
        x, y, _ = obj["position"]
        size = obj.get("size") or [0.0, 0.0, 0.0]
        found.append({"name": obj.get("name", "?"), "role": obj["role"],
                      "x": x, "y": y, "radius": max(size[0], size[1]),
                      "motion": obj.get("motion")})
    return found


def build_occupancy_grid(
    scene_path: Path | None = None,
    resolution: float = 0.05,
    robot_radius: float = 0.30,
) -> tuple[np.ndarray, tuple[float, float], float]:
    """
    Rasterizes the room into a 2D occupancy grid for path_planner.py.

    robot_radius inflates every obstacle so a planned path that stays in
    free cells is automatically collision-free for a circular robot
    footprint of that radius — simpler and cheaper than checking robot
    geometry against the path after the fact.

    Returns (grid, origin, resolution):
      grid       — np.ndarray[bool] of shape (nx, ny), True = occupied
      origin     — (x0, y0) world coordinates of grid cell (0, 0)'s center-ish
                   corner (see world_to_grid/grid_to_world in path_planner.py)
      resolution — meters per cell
    """
    obstacles = get_obstacles_2d(scene_path)
    x0, x1 = ROOM_BOUNDS["x_min"], ROOM_BOUNDS["x_max"]
    y0, y1 = ROOM_BOUNDS["y_min"], ROOM_BOUNDS["y_max"]
    nx = int(np.ceil((x1 - x0) / resolution))
    ny = int(np.ceil((y1 - y0) / resolution))

    xs = x0 + (np.arange(nx) + 0.5) * resolution
    ys = y0 + (np.arange(ny) + 0.5) * resolution
    xx, yy = np.meshgrid(xs, ys, indexing="ij")

    grid = np.zeros((nx, ny), dtype=bool)
    for fp in obstacles:
        if fp[0] == "rect":
            _, xmin, xmax, ymin, ymax = fp
            mask = (
                (xx >= xmin - robot_radius) & (xx <= xmax + robot_radius)
                & (yy >= ymin - robot_radius) & (yy <= ymax + robot_radius)
            )
        else:
            _, cx, cy, r = fp
            mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= (r + robot_radius) ** 2
        grid |= mask

    return grid, (x0, y0), resolution


if __name__ == "__main__":
    grid, origin, res = build_occupancy_grid()
    occupied = int(grid.sum())
    total = grid.size
    print(f"Grid: {grid.shape[0]}x{grid.shape[1]} cells @ {res}m — "
          f"{occupied}/{total} occupied ({occupied / total:.1%})")
    print(f"Origin: {origin}")
    print(f"Obstacles found: {len(get_obstacles_2d())}")
    try:
        import cv2
        img = (~grid).astype(np.uint8).T[::-1] * 255
        cv2.imwrite("/tmp/nav_map_preview.png", img)
        print("Preview saved to /tmp/nav_map_preview.png")
    except ImportError:
        pass
