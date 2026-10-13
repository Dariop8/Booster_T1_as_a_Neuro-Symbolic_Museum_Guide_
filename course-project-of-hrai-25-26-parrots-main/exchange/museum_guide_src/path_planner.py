#!/usr/bin/env python3
"""
Grid-based A* path planning over nav_map.py's occupancy grid.

HRAI - lecture1 (Intro + Benchmarking), "Common Functionality Benchmark
Tools" slide: "Planning: pathfinding" — this is the pathfinding half (see
nav_map.py for the map half). Pure geometry/graph search, no simulator, no
LLM, no docker — fully offline and repeatable.
"""
import heapq
import math
from pathlib import Path

import numpy as np

from nav_map import build_occupancy_grid, get_obstacles_2d

_NEIGHBOR_OFFSETS = [
    (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
    (-1, -1, 1.4142135), (-1, 1, 1.4142135), (1, -1, 1.4142135), (1, 1, 1.4142135),
]


def world_to_grid(xy: tuple[float, float], origin: tuple[float, float], resolution: float) -> tuple[int, int]:
    x0, y0 = origin
    return (int((xy[0] - x0) / resolution), int((xy[1] - y0) / resolution))


def grid_to_world(ij: tuple[int, int], origin: tuple[float, float], resolution: float) -> tuple[float, float]:
    x0, y0 = origin
    return (x0 + (ij[0] + 0.5) * resolution, y0 + (ij[1] + 0.5) * resolution)


def _astar(grid: np.ndarray, start: tuple[int, int], goal: tuple[int, int]) -> list[tuple[int, int]] | None:
    nx, ny = grid.shape

    def heuristic(a, b):
        return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5

    open_heap = [(heuristic(start, goal), start)]
    came_from: dict = {}
    g_score = {start: 0.0}
    visited = set()

    while open_heap:
        _, current = heapq.heappop(open_heap)
        if current in visited:
            continue
        visited.add(current)
        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path

        for di, dj, cost in _NEIGHBOR_OFFSETS:
            neighbor = (current[0] + di, current[1] + dj)
            ni, nj = neighbor
            if not (0 <= ni < nx and 0 <= nj < ny) or grid[ni, nj]:
                continue
            tentative_g = g_score[current] + cost
            if tentative_g < g_score.get(neighbor, float("inf")):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                heapq.heappush(open_heap, (tentative_g + heuristic(neighbor, goal), neighbor))

    return None


def _simplify(waypoints: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Collapses runs of collinear waypoints — grid A* produces one waypoint
    per cell, most of them redundant along straight stretches."""
    if len(waypoints) <= 2:
        return waypoints
    simplified = [waypoints[0]]
    for i in range(1, len(waypoints) - 1):
        p0, p1, p2 = simplified[-1], waypoints[i], waypoints[i + 1]
        cross = (p1[0] - p0[0]) * (p2[1] - p1[1]) - (p1[1] - p0[1]) * (p2[0] - p1[0])
        if abs(cross) > 1e-6:
            simplified.append(p1)
    simplified.append(waypoints[-1])
    return simplified


_MAX_SEGMENT_M = 0.9


def _split_long_segments(
    waypoints: list[tuple[float, float]], max_len: float = _MAX_SEGMENT_M
) -> list[tuple[float, float]]:
    """Re-inserts evenly spaced points so no segment exceeds max_len."""
    if len(waypoints) < 2:
        return waypoints
    out = [waypoints[0]]
    for a, b in zip(waypoints, waypoints[1:]):
        pieces = int(math.ceil(math.hypot(b[0] - a[0], b[1] - a[1]) / max_len))
        for k in range(1, pieces):
            t = k / pieces
            out.append((a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t))
        out.append(b)
    return out


_START_ESCAPE_RADIUS_M = 0.45


def _cell_centers(grid: np.ndarray, origin: tuple[float, float], resolution: float):
    nx, ny = grid.shape
    xs = origin[0] + (np.arange(nx) + 0.5) * resolution
    ys = origin[1] + (np.arange(ny) + 0.5) * resolution
    return np.meshgrid(xs, ys, indexing="ij")


def _clear_start_inflation(
    grid: np.ndarray, start_xy: tuple[float, float], origin: tuple[float, float],
    resolution: float, scene_path: Path | None,
) -> np.ndarray:
    """
    Free the cells around start_xy that are occupied only by inflation, leaving
    every cell inside real geometry occupied. See _START_ESCAPE_RADIUS_M.
    """
    xx, yy = _cell_centers(grid, origin, resolution)
    near = (xx - start_xy[0]) ** 2 + (yy - start_xy[1]) ** 2 <= _START_ESCAPE_RADIUS_M ** 2

    solid = np.zeros_like(grid)
    for fp in get_obstacles_2d(scene_path):
        if fp[0] == "rect":
            _, xmin, xmax, ymin, ymax = fp
            solid |= (xx >= xmin) & (xx <= xmax) & (yy >= ymin) & (yy <= ymax)
        else:
            _, cx, cy, r = fp
            solid |= (xx - cx) ** 2 + (yy - cy) ** 2 <= r ** 2

    grid = grid.copy()
    grid[near & ~solid] = False
    return grid


def plan_path(
    start_xy: tuple[float, float],
    goal_xy: tuple[float, float],
    grid: np.ndarray | None = None,
    origin: tuple[float, float] | None = None,
    resolution: float | None = None,
    scene_path: Path | None = None,
    extra_obstacles: list[tuple] | None = None,
) -> list[tuple[float, float]] | None:
    """
    A* path from start_xy to goal_xy (robot base frame, meters) over the
    room's known static map. Returns a simplified list of (x, y) waypoints,
    or None if no path exists.

    Pass an existing (grid, origin, resolution) to avoid rebuilding the
    static map on every call (e.g. from locomotion.py's control loop);
    otherwise one is built fresh from scene_path (nav_map.py's default if
    omitted).

    extra_obstacles — additional ("circle", cx, cy, r) primitives (e.g. a
    dynamic obstacle the depth camera just detected) OR'd into the grid for
    this call only, without mutating the caller's cached grid.
    """
    if grid is None or origin is None or resolution is None:
        grid, origin, resolution = build_occupancy_grid(scene_path)

    start_ij = world_to_grid(start_xy, origin, resolution)
    goal_ij = world_to_grid(goal_xy, origin, resolution)

    if grid[start_ij]:
        grid = _clear_start_inflation(grid, start_xy, origin, resolution, scene_path)

    if extra_obstacles:
        grid = grid.copy()
        xx, yy = _cell_centers(grid, origin, resolution)
        for _, cx, cy, r in extra_obstacles:
            grid |= (xx - cx) ** 2 + (yy - cy) ** 2 <= r ** 2

    if grid[start_ij]:
        raise ValueError(f"start {start_xy} falls inside an obstacle — nothing to plan from")
    if grid[goal_ij]:
        raise ValueError(f"goal {goal_xy} falls inside an (inflated) obstacle — unreachable")

    path_ij = _astar(grid, start_ij, goal_ij)
    if path_ij is None:
        return None

    waypoints = [grid_to_world(ij, origin, resolution) for ij in path_ij]
    return _split_long_segments(_simplify(waypoints))


if __name__ == "__main__":
    import sys

    if len(sys.argv) == 5:
        start = (float(sys.argv[1]), float(sys.argv[2]))
        goal = (float(sys.argv[3]), float(sys.argv[4]))
    else:
        start = (4.8, 0.0)
        goal = (0.75, -2.8)

    path = plan_path(start, goal)
    if path is None:
        print(f"No path found from {start} to {goal}")
    else:
        print(f"Path from {start} to {goal}: {len(path)} waypoints")
        for wp in path:
            print(f"  ({wp[0]:.2f}, {wp[1]:.2f})")
