#!/usr/bin/env python3
"""
Functionality benchmark — Planning/Control domain.

HRAI - lecture1 (Intro + Benchmarking), slide "Common Functionality
Benchmark Tools": functionality domains include "Control: trajectory
tracking, compliance, energy efficiency" and "Planning: pathfinding, task
planning, semantic reasoning" — this benchmark covers our system's instance
of both at once, since pointing-IK is a planning step (solve joint angles
for a target) evaluated by a control-style metric (position error).

Isolates the IK solver (ik/ik_solver.py) from the rest of the pipeline —
"without necessarily completing a full task" (same slide) — no circus/T1,
no simulator, no LLM: purely the geometric solve against a MuJoCo
kinematics-only model, so it's fast and fully repeatable (HRAI-lecture1
"Principles of Effective Benchmarking": Repeatability, Well-Defined
Metrics).

Ground truth: the 3D target positions the robot's own depth camera
back-projected during the last real scan (exchange/data/discovered_paintings.json,
read via knowledge_graph._load_scan_xyz — same source guida_museale_core.py
itself uses for run_robot_point_at). Falls back to a small fixed set of
plausible gallery-wall positions if no scan has ever been run, so the
benchmark is runnable even on a fresh checkout.

Since the walking scan (museum_explorer.py) stores these positions in WORLD
frame — the robot stands somewhere different for each painting while
scanning, so a base-relative position from scan time means nothing once it
walks away — solve_arm() (which needs base-relative coordinates) can't take
them directly, the same reason guida_museale_core.handle_recommend()
doesn't either. Converted here via camera_projection.to_base_frame() against
a FIXED assumed pose (museum_explorer.WORLD_FRAME_ORIGIN_POSE, the reception
spawn — R's fallback anchor in the SM = ⟨R, M, P⟩ formalism, see that
constant's docstring) rather than a live read_latest_pose() call, so the
benchmark keeps its "no circus/T1, no simulator" property — it answers "what
would the base-frame
target be if the robot were at its normal home position", which is the
realistic case (pointing happens right after returning from a scan, or in
any later session).

Usage (from circus/ directory):
    pixi run python ../exchange/museum_guide_src/benchmark_planning.py [--repeats N]
"""
import argparse
import json
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DATA_DIR = _REPO_ROOT / "exchange" / "data"
_LOG_DIR = _REPO_ROOT / "exchange" / "logs"

sys.path.insert(0, str(Path(__file__).resolve().parent / "ik"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ik_solver import solve_arm
from knowledge_graph import _load_scan_xyz
from camera_projection import to_base_frame
from museum_explorer import WORLD_FRAME_ORIGIN_POSE

_FALLBACK_TARGETS = {
    "point_left_mid":    [2.90, 3.05, 0.35],
    "point_left":        [1.55, 2.10, 0.55],
    "point_front_left":  [3.05, 1.75, 0.55],
    "point_front_right": [3.30, -1.75, 0.55],
    "point_right":       [1.55, -2.10, 0.55],
    "point_right_mid":   [2.90, -3.05, 0.35],
}

def benchmark_planning(n_repeats: int = 5) -> dict:
    """
    Runs solve_arm() n_repeats times per known target (IK is deterministic
    given the same target, so repeats measure solve-time variance/efficiency,
    not accuracy variance — still useful to catch outlier-slow solves).
    """
    targets = _load_scan_xyz(_DATA_DIR)
    if targets:
        source = "robot scan (discovered_paintings.json), world->base via fixed home pose"
        targets = {gesture: to_base_frame(xyz, WORLD_FRAME_ORIGIN_POSE) for gesture, xyz in targets.items()}
    else:
        targets = _FALLBACK_TARGETS
        source = "fallback fixed positions (no scan on disk)"

    per_target = []
    all_residuals = []
    all_times = []
    n_converged = 0
    n_total = 0

    for gesture, xyz in sorted(targets.items()):
        residuals = []
        times = []
        converged_flags = []
        for _ in range(n_repeats):
            t0 = time.perf_counter()
            result = solve_arm(xyz)
            dt = time.perf_counter() - t0
            residuals.append(result["residual_m"])
            times.append(dt)
            converged_flags.append(result["converged"])
            n_total += 1
            if result["converged"]:
                n_converged += 1
        all_residuals.extend(residuals)
        all_times.extend(times)
        per_target.append({
            "gesture": gesture,
            "target_xyz": list(xyz),
            "arm": result["arm"],
            "mean_residual_cm": round(1000 * sum(residuals) / len(residuals) / 10, 2),
            "converged_rate": round(sum(converged_flags) / len(converged_flags), 3),
            "mean_solve_time_ms": round(1000 * sum(times) / len(times), 2),
        })

    report = {
        "domain": "planning_control",
        "source": source,
        "n_targets": len(targets),
        "n_repeats_per_target": n_repeats,
        "accuracy_mean_residual_cm": round(100 * sum(all_residuals) / len(all_residuals), 2),
        "accuracy_max_residual_cm": round(100 * max(all_residuals), 2),
        "robustness_converged_rate": round(n_converged / n_total, 3),
        "efficiency_mean_solve_time_ms": round(1000 * sum(all_times) / len(all_times), 3),
        "per_target": per_target,
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Functionality benchmark — planning/control (IK)")
    parser.add_argument("--repeats", type=int, default=5, help="Solve attempts per target")
    args = parser.parse_args()

    report = benchmark_planning(args.repeats)

    print("=" * 60)
    print("Functionality Benchmark — Planning/Control (IK pointing)")
    print("=" * 60)
    print(f"  Source: {report['source']}")
    print(f"  Targets: {report['n_targets']}  x  {report['n_repeats_per_target']} repeats")
    print(f"  Accuracy   — mean residual: {report['accuracy_mean_residual_cm']} cm "
          f"(max: {report['accuracy_max_residual_cm']} cm)")
    print(f"  Robustness — converged rate: {report['robustness_converged_rate']}")
    print(f"  Efficiency — mean solve time: {report['efficiency_mean_solve_time_ms']} ms")
    print()
    for t in report["per_target"]:
        print(f"    {t['gesture']:20s} arm={t['arm']:5s} "
              f"residual={t['mean_residual_cm']:5.2f}cm  "
              f"converged={t['converged_rate']:.0%}  "
              f"solve={t['mean_solve_time_ms']:.2f}ms")

    _LOG_DIR.mkdir(parents=True, exist_ok=True)
    existing = sorted(_LOG_DIR.glob("functionality_planning_*.json"))
    index = len(existing) + 1
    out_path = _LOG_DIR / f"functionality_planning_{index:03d}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n📝 Log salvato in: {out_path}")


if __name__ == "__main__":
    main()
