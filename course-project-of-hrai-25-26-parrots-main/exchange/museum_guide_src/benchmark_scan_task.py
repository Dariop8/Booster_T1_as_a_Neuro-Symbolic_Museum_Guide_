#!/usr/bin/env python3
"""
Task benchmark — the full painting-scanning procedure, end to end.

HRAI - lecture1 (Intro + Benchmarking), slide "Benchmarking – Task
Benchmarking": "End-to-end evaluation; Includes perception, planning, and
actuation". Perception (benchmark_perception.py) and localisation
(benchmark_planning.py) are already tested, but both isolate their piece by
reimplementing the walking loop themselves — neither calls the real
production entrypoint (museum_explorer.explore_museum() +
save_scan_results()) end to end. This benchmark does: it runs
museum_explorer.py as a subprocess (same real script a curator would run
after rehanging the gallery) and checks the whole circuit, not just one
component in isolation.

Metrics:
  - Full-circuit completion: did all 6 directions succeed in ONE continuous
    run (not per-direction accuracy in isolation — a walking failure on
    direction 3 that strands the robot counts here even if directions 1-2
    were perfect).
  - Efficiency: total wall-clock time for the whole circuit, door to door —
    never measured at this level before (functionality benchmarks measure
    per-call latency, not the full walk+perceive+build-graph procedure).
  - Classification accuracy per direction, against the SAME curator
    registry benchmark_perception.py uses — but here read from the real
    entrypoint's own output file, not a reimplemented loop.
  - Scene-graph structural correctness: does discovered_scene_map.yaml
    assign each gesture_id to the wall the fixed geometry (SLOT_BEARING_RAD +
    _WALL_BOUNDARY_YAW, knowledge_graph.py) says it MUST be on? The
    assignment is mathematically deterministic, but no benchmark has ever
    actually exercised the code that builds it — a regression there would
    go unnoticed today.
  - Return-to-reception error: position error after the full circuit,
    vs. locomotion.py's isolated single-goal benchmark_locomotion.py, here
    as the LAST step of a much longer real task.

IMPORTANT — scene randomization is NOT re-triggered between (or before)
repeats here, unlike what a first draft of this benchmark assumed:
scene_randomizer.py's own docstring is explicit that textures are loaded
ONCE when circus starts the scene ("Run BEFORE starting the simulator") —
calling randomize_scene() while circus is already running would write new
textures the simulator never reloads, silently desynchronising
scene_ground_truth.yaml (updated) from what the VLM actually sees
(stale) and making the accuracy numbers meaningless. So: like
benchmark_perception.py, this benchmark reads whatever scene_ground_truth.yaml
already exists (from whenever circus itself was last started) and repeats
the SCAN procedure N times against that fixed, already-loaded hanging —
still a genuine repeatability test of the scanning procedure itself.

Needs GROQ_API_KEY AND a live circus/T1 (rebuilt with BoosterT1.h's
pose_writer_) AND the T2 walk-ready sequence (gm, hcm-start, gm) — same
requirements as museum_explorer.py/benchmark_perception.py. A real scan
takes minutes; keep --repeats low (default 3).

Usage (from circus/ directory):
    pixi run python ../exchange/museum_guide_src/benchmark_scan_task.py [--repeats N]
"""
import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml

from knowledge_graph import _load_registry
from shm_camera_reader import read_latest_pose

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CIRCUS_DIR = _REPO_ROOT / "circus"
_EXPLORER_SCRIPT = Path(__file__).resolve().parent / "museum_explorer.py"
_DATA_DIR = _REPO_ROOT / "exchange" / "data"
_LOG_DIR = _REPO_ROOT / "exchange" / "logs"

_DISCOVERED_PAINTINGS = _DATA_DIR / "discovered_paintings.json"
_DISCOVERED_SCENE_MAP = _DATA_DIR / "discovered_scene_map.yaml"

_HOME_POSE = {"x": 4.8, "y": 0.0}

_SCAN_ORDER = [
    "point_left", "point_left_mid", "point_front_left",
    "point_front_right", "point_right_mid", "point_right",
]

_EXPECTED_WALL = {
    "point_left_mid": "left_wall", "point_left": "left_wall",
    "point_front_left": "front_wall", "point_front_right": "front_wall",
    "point_right": "right_wall", "point_right_mid": "right_wall",
}


def _run_one_scan(robot_name: str = "red_Booster-T1_0", timeout_s: float = 900.0) -> dict:
    registry = _load_registry(_DATA_DIR)
    if not registry:
        raise RuntimeError(
            "No exchange/data/scene_ground_truth.yaml — run scene_randomizer.py "
            "BEFORE starting circus, then restart circus, before this benchmark."
        )

    before_mtime = _DISCOVERED_PAINTINGS.stat().st_mtime if _DISCOVERED_PAINTINGS.exists() else None

    cmd = [sys.executable, str(_EXPLORER_SCRIPT)]
    t0 = time.monotonic()
    try:
        proc = subprocess.run(
            cmd, cwd=str(_CIRCUS_DIR), capture_output=True, text=True,
            timeout=timeout_s, env=os.environ.copy(),
        )
    except subprocess.TimeoutExpired:
        return {"completed_full_circuit": False, "error": "timeout",
                "wall_s": round(time.monotonic() - t0, 1)}
    wall_s = round(time.monotonic() - t0, 1)

    after_mtime = _DISCOVERED_PAINTINGS.stat().st_mtime if _DISCOVERED_PAINTINGS.exists() else None
    if after_mtime is None or after_mtime == before_mtime:
        tail = (proc.stdout[-800:] + proc.stderr[-800:]).strip()
        return {
            "completed_full_circuit": False,
            "error": "discovered_paintings.json not refreshed by this run "
                     "(subprocess exit code {}) — tail of its output: {}".format(proc.returncode, tail[-500:]),
            "wall_s": wall_s,
        }
    discovered = json.loads(_DISCOVERED_PAINTINGS.read_text(encoding="utf-8"))

    failed = [g for g in _SCAN_ORDER if g not in discovered or discovered[g].get("error")]
    completed_full_circuit = not failed

    checked = [g for g in _SCAN_ORDER if g in discovered and not discovered[g].get("error")]
    correct = sum(1 for g in checked if discovered[g].get("painting_id") == registry.get(g))
    classification_accuracy = round(correct / len(_SCAN_ORDER), 3)

    wall_correct = wall_checked = 0
    if _DISCOVERED_SCENE_MAP.exists():
        scene_map = yaml.safe_load(_DISCOVERED_SCENE_MAP.read_text(encoding="utf-8"))
        objects = {o["id"]: o for o in scene_map.get("scene", {}).get("objects", [])}
        for g in checked:
            pid = discovered[g].get("painting_id")
            if pid not in objects:
                continue
            wall_checked += 1
            if objects[pid].get("wall") == _EXPECTED_WALL[g]:
                wall_correct += 1
    scene_graph_wall_accuracy = round(wall_correct / wall_checked, 3) if wall_checked else None

    pose = read_latest_pose(robot_name)
    return_to_home_error_m = None
    if pose is not None:
        return_to_home_error_m = round(math.hypot(
            pose["x"] - _HOME_POSE["x"], pose["y"] - _HOME_POSE["y"],
        ), 4)

    return {
        "completed_full_circuit": completed_full_circuit,
        "failed_directions": failed,
        "wall_s": wall_s,
        "classification_accuracy": classification_accuracy,
        "scene_graph_wall_accuracy": scene_graph_wall_accuracy,
        "return_to_home_error_m": return_to_home_error_m,
    }


def benchmark_scan_task(n_repeats: int = 3, robot_name: str = "red_Booster-T1_0") -> dict:
    runs = []
    for i in range(n_repeats):
        print(f"[Scan task benchmark] Run {i + 1}/{n_repeats}...", flush=True)
        run = _run_one_scan(robot_name)
        runs.append(run)
        status = "OK" if run["completed_full_circuit"] else f"FAILED ({run.get('error', run.get('failed_directions'))})"
        print(f"  {status}  ({run['wall_s']}s, classification={run.get('classification_accuracy')}, "
              f"wall_graph={run.get('scene_graph_wall_accuracy')}, "
              f"home_error_m={run.get('return_to_home_error_m')})")

    n = len(runs)
    completion_rate = round(sum(r["completed_full_circuit"] for r in runs) / n, 3) if n else None
    mean_wall_s = round(sum(r["wall_s"] for r in runs) / n, 1) if n else None
    class_acc = [r["classification_accuracy"] for r in runs if "classification_accuracy" in r]
    wall_acc = [r["scene_graph_wall_accuracy"] for r in runs if r.get("scene_graph_wall_accuracy") is not None]
    home_err = [r["return_to_home_error_m"] for r in runs if r.get("return_to_home_error_m") is not None]

    return {
        "n_repeats": n_repeats,
        "completion_rate": completion_rate,
        "mean_wall_s": mean_wall_s,
        "mean_classification_accuracy": round(sum(class_acc) / len(class_acc), 3) if class_acc else None,
        "mean_scene_graph_wall_accuracy": round(sum(wall_acc) / len(wall_acc), 3) if wall_acc else None,
        "mean_return_to_home_error_m": round(sum(home_err) / len(home_err), 4) if home_err else None,
        "runs": runs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Task benchmark — full painting-scanning procedure")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--robot-name", default="red_Booster-T1_0")
    args = parser.parse_args()

    report = benchmark_scan_task(args.repeats, args.robot_name)

    print("\n" + "=" * 60)
    print("Task Benchmark — full scanning procedure")
    print("=" * 60)
    print(f"  Completion rate (full 6/6 circuit): {report['completion_rate']:.0%}" if report["completion_rate"] is not None else "  Completion rate: n/a")
    print(f"  Mean wall time: {report['mean_wall_s']}s")
    print(f"  Mean classification accuracy: {report['mean_classification_accuracy']}")
    print(f"  Mean scene-graph wall accuracy: {report['mean_scene_graph_wall_accuracy']}")
    print(f"  Mean return-to-home error: {report['mean_return_to_home_error_m']} m")

    _LOG_DIR.mkdir(parents=True, exist_ok=True)
    existing = sorted(_LOG_DIR.glob("task_scan_*.json"))
    index = len(existing) + 1
    out_path = _LOG_DIR / f"task_scan_{index:03d}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n📝 Log salvato in: {out_path}")


if __name__ == "__main__":
    main()
