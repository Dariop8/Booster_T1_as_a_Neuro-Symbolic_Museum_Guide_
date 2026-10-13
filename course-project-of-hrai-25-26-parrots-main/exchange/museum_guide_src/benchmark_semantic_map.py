#!/usr/bin/env python3
"""
Functionality benchmark — Semantic Mapping domain.

HRAI - Geometrical and Semantical Representations, "Building a Semantic
Map": SM = ⟨R, M, P⟩ and, given a built map SM1 ("env") and an expected/
ground-truth map SMm, δ(SM1, SMm) = f(|M1 ⊖ Mm|, |P1 ⊟ Pm|). This benchmark
instantiates that δ for our museum:
  - SM1 (M1/P1) comes from the robot's own RAW scan
    (exchange/data/discovered_paintings.json, via knowledge_graph._load_scan)
    — the "env" as actually perceived, BEFORE curator-registry grounding
    (resolve_placement()'s correction step is deliberately bypassed here:
    grounding on the registry would make SM1 trivially equal to SMm, since
    the registry IS the ground truth — see knowledge_graph._merge_scan_with_
    registry — which would defeat the point of measuring perception quality).
  - SMm (Mm/Pm) comes from exchange/data/scene_ground_truth.yaml (written by
    scene_randomizer.py), via knowledge_graph._load_registry.
  - R is shared by construction (see semantic_map_delta()'s docstring) and
    plays no part in the comparison.

Needs no live circus/T1 — both inputs are already on disk from the last real
scan (same "no circus/T1, no simulator" property benchmark_planning.py has),
so it's fast and fully repeatable (HRAI-lecture1 "Principles of Effective
Benchmarking": Repeatability, Well-Defined Metrics).

Usage (from circus/ directory):
    pixi run python ../exchange/museum_guide_src/benchmark_semantic_map.py
"""
import json
from pathlib import Path

from knowledge_graph import _load_scan, _load_registry, semantic_map_delta

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DATA_DIR = _REPO_ROOT / "exchange" / "data"
_LOG_DIR = _REPO_ROOT / "exchange" / "logs"


def benchmark_semantic_map() -> dict:
    scan = _load_scan(_DATA_DIR)
    if not scan:
        raise RuntimeError(
            "No exchange/data/discovered_paintings.json — run a scan "
            "(museum_explorer.explore_museum) first so there is a built map to score."
        )
    ground_truth = _load_registry(_DATA_DIR)
    if not ground_truth:
        raise RuntimeError(
            "No exchange/data/scene_ground_truth.yaml — run scene_randomizer.py first "
            "so there is a ground truth to score against."
        )

    result = semantic_map_delta(scan, ground_truth)
    return {
        "domain": "semantic_map_delta",
        "n_directions": len(set(scan) | set(ground_truth)),
        **result,
    }


def main() -> None:
    report = benchmark_semantic_map()

    print("=" * 60)
    print("Functionality Benchmark — Semantic Map δ(SM1, SMm)")
    print("=" * 60)
    print(f"  δ (overall, lower is better): {report['delta']}")
    print(f"  M discrepancy (|M1 ⊖ Mm|):    {report['m_delta']}")
    if report["m_mismatches"]:
        print("    mismatched directions:")
        for m in report["m_mismatches"]:
            print(f"      {m['gesture']:20s} built={m['built']!r:>6}  ground_truth={m['ground_truth']!r}")
    else:
        print("    (no mismatches — every direction agrees with ground truth)")
    print(f"  P discrepancy (|P1 ⊟ Pm|):    {report['p_delta']}")
    if report["p_sym_diff"]:
        print(f"    {len(report['p_sym_diff'])} relation(s) differ:")
        for t in report["p_sym_diff"]:
            print(f"      {t}")
    else:
        print("    (no relation differences — same scene-graph structure)")

    _LOG_DIR.mkdir(parents=True, exist_ok=True)
    existing = sorted(_LOG_DIR.glob("functionality_semantic_map_*.json"))
    index = len(existing) + 1
    out_path = _LOG_DIR / f"functionality_semantic_map_{index:03d}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n📝 Log salvato in: {out_path}")


if __name__ == "__main__":
    main()
