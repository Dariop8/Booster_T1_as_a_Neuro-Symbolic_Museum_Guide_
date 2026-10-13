#!/usr/bin/env python3
"""
Functionality benchmark — Perception domain.

HRAI - lecture1 (Intro + Benchmarking), slide "Common Functionality
Benchmark Tools": functionality domains include "Perception: image
classification, depth estimation, SLAM" — this benchmark isolates OUR image
classification component (painting_recognizer.analyze_painting, the
single-call VLM classification underneath museum_explorer's
analyze_painting_voted) from the rest of the pipeline: one walking scan
(same procedure as museum_explorer.py — walk to each of the 6 standing
spots, look at the slot) captures one frame per stop, then each cropped
frame is classified N times (cheap — no further walking) to measure the
underlying per-call accuracy that voting sits on top of.

Metrics (slide "Common Metrics"): Accuracy (against
exchange/data/scene_ground_truth.yaml — the curator ground truth, same
comparison guida_museale_core.kg._merge_scan_with_registry does), Robustness
(accuracy broken down per standing spot — the 6 differ somewhat in
distance/geometry, a real varying condition already present in one scan),
Efficiency (mean VLM call latency).

Needs GROQ_API_KEY AND a live circus/T1 (rebuilt with BoosterT1.h's
pose_writer_) AND the T2 walk-ready sequence (gm, hcm-start, gm) — the scan
walks between paintings now, same requirement as museum_explorer.py.

Usage (from circus/ directory):
    pixi run python ../exchange/museum_guide_src/benchmark_perception.py [--repeats N]
"""
import argparse
import json
import os
import time
from pathlib import Path

from shm_camera_reader import read_latest_frame
from painting_recognizer import analyze_painting
from knowledge_graph import _load_registry
from locomotion import navigate_to_pose
from museum_explorer import (
    _SCAN_PLAN, _STAND_POSE, _HOME_POSE, _look_at_slot, _crop_frame,
    _save_debug_png, _load_candidates, _warn_if_textures_stale,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DATA_DIR = _REPO_ROOT / "exchange" / "data"
_LOG_DIR = _REPO_ROOT / "exchange" / "logs"


def benchmark_perception(n_repeats: int = 5, robot_name: str = "red_Booster-T1_0") -> dict:
    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY not set")

    _warn_if_textures_stale()
    candidates = _load_candidates()
    registry = _load_registry(_DATA_DIR)
    if not registry:
        raise RuntimeError(
            "No exchange/data/scene_ground_truth.yaml — run scene_randomizer.py first "
            "so there is a ground truth to score against."
        )

    per_direction = []
    all_correct = 0
    all_calls = 0
    all_times = []

    print("[Perception benchmark] Walking scan (6 standing spots, one frame each)...")
    for step in _SCAN_PLAN:
        gesture_id = step["gesture_id"]
        stand = _STAND_POSE[gesture_id]
        expected_pid = registry.get(gesture_id)
        print(f"\n  → {step['label']}  (expected: {expected_pid})")

        if not navigate_to_pose(stand["x"], stand["y"], stand["yaw"], robot_name=robot_name, verbose=False):
            print("    Failed to reach standing spot — skipping direction.")
            per_direction.append({"gesture": gesture_id, "error": "navigation_failed"})
            continue
        _look_at_slot(gesture_id)
        time.sleep(1.0)

        frame = read_latest_frame(robot_name)
        if frame is None:
            print(f"    No frame — is circus running? Skipping direction.")
            per_direction.append({"gesture": gesture_id, "error": "no_frame"})
            continue

        frame = _crop_frame(frame, step.get("crop_x"), step.get("crop_y"))
        _save_debug_png(frame, gesture_id)

        answers = []
        times = []
        for _ in range(n_repeats):
            t0 = time.perf_counter()
            try:
                info = analyze_painting(frame, api_key, focus=step.get("focus", "center"),
                                         candidates=candidates)
            except Exception as e:
                info = {"painting_id": None, "error": str(e)}
            dt = time.perf_counter() - t0
            times.append(dt)
            got_pid = info.get("painting_id")
            answers.append(got_pid)
            all_calls += 1
            all_times.append(dt)
            if expected_pid is not None and got_pid == expected_pid:
                all_correct += 1

        n_ok = sum(1 for a in answers if a == expected_pid)
        print(f"    Answers: {answers}  ({n_ok}/{n_repeats} correct, "
              f"mean {1000*sum(times)/len(times):.0f}ms/call)")
        per_direction.append({
            "gesture": gesture_id,
            "expected_painting_id": expected_pid,
            "answers": answers,
            "accuracy": round(n_ok / n_repeats, 3),
            "mean_latency_ms": round(1000 * sum(times) / len(times), 1),
        })

    print(f"\n[Perception benchmark] Walking back to home position "
          f"({_HOME_POSE['x']:.1f},{_HOME_POSE['y']:.1f}).")
    navigate_to_pose(_HOME_POSE["x"], _HOME_POSE["y"], _HOME_POSE["yaw"], robot_name=robot_name, verbose=False)

    report = {
        "domain": "perception_vlm_classification",
        "n_directions": len(per_direction),
        "n_repeats_per_direction": n_repeats,
        "accuracy_overall": round(all_correct / all_calls, 3) if all_calls else None,
        "robustness_per_direction": {
            d["gesture"]: d.get("accuracy") for d in per_direction if "accuracy" in d
        },
        "efficiency_mean_latency_ms": round(1000 * sum(all_times) / len(all_times), 1) if all_times else None,
        "per_direction": per_direction,
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Functionality benchmark — perception (VLM classification)")
    parser.add_argument("--repeats", type=int, default=5, help="Classification calls per direction")
    parser.add_argument("--robot-name", default="red_Booster-T1_0")
    args = parser.parse_args()

    report = benchmark_perception(args.repeats, args.robot_name)

    print("\n" + "=" * 60)
    print("Functionality Benchmark — Perception (VLM painting classification)")
    print("=" * 60)
    print(f"  Directions: {report['n_directions']}  x  {report['n_repeats_per_direction']} repeats")
    print(f"  Accuracy (overall, single-call, unvoted): {report['accuracy_overall']:.1%}")
    print(f"  Efficiency — mean latency: {report['efficiency_mean_latency_ms']} ms")
    print("  Robustness — accuracy per direction (viewing angle):")
    for gesture, acc in report["robustness_per_direction"].items():
        print(f"    {gesture:20s} {acc:.0%}")

    _LOG_DIR.mkdir(parents=True, exist_ok=True)
    existing = sorted(_LOG_DIR.glob("functionality_perception_*.json"))
    index = len(existing) + 1
    out_path = _LOG_DIR / f"functionality_perception_{index:03d}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n📝 Log salvato in: {out_path}")


if __name__ == "__main__":
    main()
