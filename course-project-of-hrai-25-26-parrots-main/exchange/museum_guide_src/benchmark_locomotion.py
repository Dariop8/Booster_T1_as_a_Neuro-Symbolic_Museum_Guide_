#!/usr/bin/env python3
"""
Functionality benchmark — Control domain (locomotion).

HRAI - lecture1 (Intro + Benchmarking), "Common Functionality Benchmark
Tools" slide: "Control: trajectory tracking, compliance, energy efficiency"
— a distinct functionality domain from the existing benchmark_planning.py
(which only covers the ARM's IK, never the walking control loop). This
isolates locomotion.navigate_to()/navigate_to_pose() — path planning +
closed-loop walking — from perception (no VLM calls) and dialogue (no LLM
calls), same "without necessarily completing a full task" spirit as the
other functionality benchmarks.

Fixed, small set of goals (unlike the arm IK benchmark's many cheap
repeats): each one costs real walking time, so this trades repeat count for
covering qualitatively different situations — short open-space hop, a long
crossing of the room, a route that's forced around the reception
desk/bench cluster, and a goal with a specific final heading (reuses real
museum_explorer.py standing spots for the last two, tying the benchmark to
the actual production use case instead of arbitrary points).

Metrics (slide "Common Metrics"): Accuracy (final position/heading error),
Robustness (success rate — reached within tolerance), Efficiency (time to
arrive, and path ratio = planned path length / straight-line distance — >1
means the planner had to detour around something).

Needs a live circus/T1 (rebuilt with BoosterT1.h's pose_writer_) AND the T2
walk-ready sequence (gm, hcm-start, gm) — same requirement as
navigation_demo.py / museum_explorer.py.

Usage (from circus/ directory):
    pixi run python ../exchange/museum_guide_src/benchmark_locomotion.py
"""
import argparse
import json
import math
import time
from pathlib import Path

from locomotion import navigate_to, navigate_to_pose, _WAYPOINT_TOLERANCE, _HEADING_TOLERANCE_RAD
from nav_map import build_occupancy_grid
from path_planner import plan_path
from shm_camera_reader import read_latest_pose
from museum_explorer import _STAND_POSE, _HOME_POSE

_REPO_ROOT = Path(__file__).resolve().parents[2]
_LOG_DIR = _REPO_ROOT / "exchange" / "logs"

_GOALS = [
    {
        "name": "short_open",
        "goal": (3.5, -1.6),
        "heading": None,
        "description": "Short hop in open space near spawn — baseline, no obstacles nearby.",
    },
    {
        "name": "long_cross_room",
        "goal": (_STAND_POSE["point_front_left"]["x"], _STAND_POSE["point_front_left"]["y"]),
        "heading": None,
        "description": "Long walk across most of the room (spawn to the far entrance wall).",
    },
    {
        "name": "obstacle_avoidance",
        "goal": (_STAND_POSE["point_right_mid"]["x"], _STAND_POSE["point_right_mid"]["y"]),
        "heading": None,
        "description": "Forces a route around the reception desk/bench cluster blocking "
                        "the direct line.",
    },
    {
        "name": "pose_with_heading",
        "goal": (_STAND_POSE["point_right"]["x"], _STAND_POSE["point_right"]["y"]),
        "heading": _STAND_POSE["point_right"]["yaw"],
        "description": "Tests navigate_to_pose()'s final turn-in-place — position AND "
                        "facing direction both matter here.",
    },
]


def _path_length(path: list[tuple[float, float]]) -> float:
    return sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(path, path[1:]))


def benchmark_locomotion(robot_name: str = "red_Booster-T1_0", verbose: bool = True) -> dict:
    pose = read_latest_pose(robot_name)
    if pose is None:
        raise RuntimeError(
            "No ground-truth pose available — is circus running with the rebuilt "
            "BoosterT1.h (pose_writer_)? Restart T1 if it was already running before "
            "that change."
        )

    grid, origin, resolution = build_occupancy_grid()
    results = []

    for spec in _GOALS:
        start_pose = read_latest_pose(robot_name)
        start_xy = (start_pose["x"], start_pose["y"])
        if verbose:
            print(f"\n[Locomotion benchmark] {spec['name']} — {spec['description']}")
            print(f"  {start_xy} -> {spec['goal']}"
                  + (f" (heading {spec['heading']:.2f} rad)" if spec["heading"] is not None else ""))

        straight_line = math.hypot(spec["goal"][0] - start_xy[0], spec["goal"][1] - start_xy[1])
        try:
            planned = plan_path(start_xy, spec["goal"], grid, origin, resolution)
        except ValueError as e:
            if verbose:
                print(f"  Planning failed: {e}")
            results.append({"name": spec["name"], "error": str(e)})
            continue
        planned_length = _path_length(planned) if planned else None

        t0 = time.monotonic()
        if spec["heading"] is not None:
            reached = navigate_to_pose(spec["goal"][0], spec["goal"][1], spec["heading"],
                                        robot_name=robot_name, verbose=False)
        else:
            reached = navigate_to(spec["goal"], robot_name=robot_name, verbose=False)
        elapsed_s = time.monotonic() - t0

        final_pose = read_latest_pose(robot_name)
        position_error_m = (math.hypot(final_pose["x"] - spec["goal"][0], final_pose["y"] - spec["goal"][1])
                             if final_pose else None)
        heading_error_rad = None
        if spec["heading"] is not None and final_pose is not None:
            heading_error_rad = abs((final_pose["yaw"] - spec["heading"] + math.pi) % (2 * math.pi) - math.pi)

        entry = {
            "name": spec["name"],
            "description": spec["description"],
            "reached": bool(reached),
            "position_error_cm": round(100 * position_error_m, 1) if position_error_m is not None else None,
            "heading_error_rad": round(heading_error_rad, 3) if heading_error_rad is not None else None,
            "elapsed_s": round(elapsed_s, 1),
            "straight_line_m": round(straight_line, 2),
            "planned_path_m": round(planned_length, 2) if planned_length is not None else None,
            "path_ratio": (round(planned_length / straight_line, 2)
                            if planned_length and straight_line > 1e-3 else None),
        }
        results.append(entry)
        if verbose:
            print(f"  reached={entry['reached']} pos_error={entry['position_error_cm']}cm "
                  f"heading_error={entry['heading_error_rad']} time={entry['elapsed_s']}s "
                  f"path_ratio={entry['path_ratio']}")

    if verbose:
        print(f"\n[Locomotion benchmark] Walking back to home position "
              f"({_HOME_POSE['x']:.1f},{_HOME_POSE['y']:.1f}).")
    navigate_to_pose(_HOME_POSE["x"], _HOME_POSE["y"], _HOME_POSE["yaw"], robot_name=robot_name, verbose=False)

    ok_results = [r for r in results if "error" not in r]
    n = len(_GOALS)
    n_reached = sum(1 for r in ok_results if r["reached"])
    position_errors = [r["position_error_cm"] for r in ok_results if r["position_error_cm"] is not None]
    heading_errors = [r["heading_error_rad"] for r in ok_results if r["heading_error_rad"] is not None]
    times = [r["elapsed_s"] for r in ok_results]
    path_ratios = [r["path_ratio"] for r in ok_results if r["path_ratio"] is not None]

    report = {
        "domain": "control_locomotion",
        "n_goals": n,
        "waypoint_tolerance_m": _WAYPOINT_TOLERANCE,
        "heading_tolerance_rad": _HEADING_TOLERANCE_RAD,
        "robustness_success_rate": round(n_reached / n, 3) if n else None,
        "accuracy_mean_position_error_cm": round(sum(position_errors) / len(position_errors), 1) if position_errors else None,
        "accuracy_mean_heading_error_rad": round(sum(heading_errors) / len(heading_errors), 3) if heading_errors else None,
        "efficiency_mean_time_s": round(sum(times) / len(times), 1) if times else None,
        "efficiency_mean_path_ratio": round(sum(path_ratios) / len(path_ratios), 2) if path_ratios else None,
        "per_goal": results,
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Functionality benchmark — control (locomotion)")
    parser.add_argument("--robot-name", default="red_Booster-T1_0")
    args = parser.parse_args()

    report = benchmark_locomotion(args.robot_name)

    print("\n" + "=" * 60)
    print("Functionality Benchmark — Control (locomotion)")
    print("=" * 60)
    print(f"  Goals: {report['n_goals']}")
    print(f"  Robustness — success rate: {report['robustness_success_rate']:.0%}")
    print(f"  Accuracy   — mean position error: {report['accuracy_mean_position_error_cm']} cm")
    print(f"  Accuracy   — mean heading error: {report['accuracy_mean_heading_error_rad']} rad")
    print(f"  Efficiency — mean time: {report['efficiency_mean_time_s']} s")
    print(f"  Efficiency — mean path ratio (planned/straight-line): {report['efficiency_mean_path_ratio']}")
    print()
    for g in report["per_goal"]:
        if "error" in g:
            print(f"    {g['name']:20s} ERROR: {g['error']}")
        else:
            print(f"    {g['name']:20s} reached={g['reached']}  pos_err={g['position_error_cm']}cm  "
                  f"heading_err={g['heading_error_rad']}  time={g['elapsed_s']}s  "
                  f"path_ratio={g['path_ratio']}")

    _LOG_DIR.mkdir(parents=True, exist_ok=True)
    existing = sorted(_LOG_DIR.glob("functionality_locomotion_*.json"))
    index = len(existing) + 1
    out_path = _LOG_DIR / f"functionality_locomotion_{index:03d}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n📝 Log salvato in: {out_path}")


if __name__ == "__main__":
    main()
