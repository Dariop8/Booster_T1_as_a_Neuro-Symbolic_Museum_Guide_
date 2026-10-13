#!/usr/bin/env python3
"""
HRAI - lecture1 (Intro + Benchmarking): aggregates the raw benchmark logs
into the categories taught in the slides ("Two types of benchmarks for
complex systems: functionality benchmarks... task benchmarks...", slide
"Benchmarking – Task Benchmarking"):

  - Functionality benchmarks (per-component, "without necessarily completing
    a full task" — slide "Measuring Specific Capabilities of Robotic
    Subsystems"), one per domain the slide lists under "Examples of
    Functionality Domains":
      - Control  : HRAI 9 self-awareness feasibility score calibration
                   (check_feasibility() vs actual gesture_executed outcome) —
                   read straight from session interaction_*.json logs.
      - Planning/Control: IK pointing accuracy (benchmark_planning.py)
      - Control (locomotion): walking accuracy/robustness/efficiency
                   (benchmark_locomotion.py) — trajectory tracking, distinct
                   from the arm-only Planning/Control benchmark above.
      - Perception: VLM painting-classification accuracy (benchmark_perception.py)
      - Interaction: NLU intent/language accuracy (benchmark_interaction.py)
  - Task benchmark: end-to-end session performance — completion, turns,
    duration, paintings shown per visit (opportunistic, from every session
    ever run) AND the standardized scripted-scenario success/completion
    rates (benchmark_task.py).
  - User-experience benchmark: Godspeed ratings from --survey sessions
    (HRI7-Evaluation, Lecture 7 — a different deck from the one above).

Reads exchange/logs/*.json — nothing here talks to the simulator or the
LLM, so it can be run anytime after a batch of benchmark runs, independent
of whether circus/the robot is running.

Usage (from circus/ directory):
    pixi run python ../exchange/museum_guide_src/benchmark_report.py
"""
import json
from pathlib import Path
from statistics import mean, pstdev

from scipy import stats as scipy_stats

_REPO_ROOT = Path(__file__).resolve().parents[2]
_LOG_DIR = _REPO_ROOT / "exchange" / "logs"


def _load_all(pattern: str) -> list[dict]:
    out = []
    for path in sorted(_LOG_DIR.glob(pattern)):
        try:
            out.append(json.loads(path.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError):
            continue
    return out


def functionality_selfawareness_benchmark(interactions: list[dict]) -> dict:
    """Per-recommendation: HRAI 9 feasibility score + gesture success rate —
    does the self-awareness score's prediction match what actually happened?"""
    scores = [
        i["feasibility"]["feasibility_score"] for i in interactions
        if i.get("feasibility", {}).get("feasibility_score") is not None
    ]
    fallbacks = [
        i["feasibility"]["fallback"] for i in interactions
        if i.get("feasibility", {}).get("fallback")
    ]
    gesture_flags = [i["gesture_executed"] for i in interactions if "gesture_executed" in i]

    return {
        "n_recommendations": len(interactions),
        "mean_feasibility_score": round(mean(scores), 3) if scores else None,
        "fallback_counts": {f: fallbacks.count(f) for f in sorted(set(fallbacks))},
        "gesture_success_rate": round(mean(gesture_flags), 3) if gesture_flags else None,
    }


def functionality_perception_benchmark(logs: list[dict]) -> dict:
    """Aggregates benchmark_perception.py runs (VLM painting classification)."""
    if not logs:
        return {"n_runs": 0}
    accs = [l["accuracy_overall"] for l in logs if l.get("accuracy_overall") is not None]
    lats = [l["efficiency_mean_latency_ms"] for l in logs if l.get("efficiency_mean_latency_ms") is not None]
    return {
        "n_runs": len(logs),
        "mean_accuracy": round(mean(accs), 3) if accs else None,
        "mean_latency_ms": round(mean(lats), 1) if lats else None,
    }


def functionality_interaction_benchmark(logs: list[dict]) -> dict:
    """Aggregates benchmark_interaction.py runs (NLU intent/language)."""
    if not logs:
        return {"n_runs": 0}
    return {
        "n_runs": len(logs),
        "mean_intent_accuracy": round(mean(l["accuracy_intent"] for l in logs), 3),
        "mean_language_accuracy": round(mean(l["accuracy_language"] for l in logs), 3),
        "mean_latency_ms": round(mean(l["efficiency_mean_latency_ms"] for l in logs), 1),
    }


def functionality_planning_benchmark(logs: list[dict]) -> dict:
    """Aggregates benchmark_planning.py runs (IK pointing accuracy)."""
    if not logs:
        return {"n_runs": 0}
    return {
        "n_runs": len(logs),
        "mean_accuracy_residual_cm": round(mean(l["accuracy_mean_residual_cm"] for l in logs), 2),
        "mean_robustness_converged_rate": round(mean(l["robustness_converged_rate"] for l in logs), 3),
        "mean_efficiency_solve_time_ms": round(mean(l["efficiency_mean_solve_time_ms"] for l in logs), 3),
    }


def functionality_semantic_map_benchmark(logs: list[dict]) -> dict:
    """Aggregates benchmark_semantic_map.py runs (δ(SM1, SMm) — built scan vs ground truth)."""
    if not logs:
        return {"n_runs": 0}
    return {
        "n_runs": len(logs),
        "mean_delta": round(mean(l["delta"] for l in logs), 3),
        "mean_m_delta": round(mean(l["m_delta"] for l in logs), 3),
        "mean_p_delta": round(mean(l["p_delta"] for l in logs), 3),
    }


def functionality_locomotion_benchmark(logs: list[dict]) -> dict:
    """Aggregates benchmark_locomotion.py runs (walking accuracy/robustness/efficiency)."""
    if not logs:
        return {"n_runs": 0}
    pos = [l["accuracy_mean_position_error_cm"] for l in logs if l.get("accuracy_mean_position_error_cm") is not None]
    heading = [l["accuracy_mean_heading_error_rad"] for l in logs if l.get("accuracy_mean_heading_error_rad") is not None]
    succ = [l["robustness_success_rate"] for l in logs if l.get("robustness_success_rate") is not None]
    times = [l["efficiency_mean_time_s"] for l in logs if l.get("efficiency_mean_time_s") is not None]
    ratios = [l["efficiency_mean_path_ratio"] for l in logs if l.get("efficiency_mean_path_ratio") is not None]
    return {
        "n_runs": len(logs),
        "mean_position_error_cm": round(mean(pos), 1) if pos else None,
        "mean_heading_error_rad": round(mean(heading), 3) if heading else None,
        "mean_success_rate": round(mean(succ), 3) if succ else None,
        "mean_time_s": round(mean(times), 1) if times else None,
        "mean_path_ratio": round(mean(ratios), 2) if ratios else None,
    }


def functionality_face_emotion_benchmark(logs: list[dict]) -> dict:
    """Aggregates benchmark_face_emotion.py runs (face emotion classification)."""
    if not logs:
        return {"n_runs": 0}
    raw = [l["accuracy_raw_label"] for l in logs if l.get("accuracy_raw_label") is not None]
    mood = [l["accuracy_mood"] for l in logs if l.get("accuracy_mood") is not None]
    conf_ok = [l["mean_confidence_when_correct"] for l in logs if l.get("mean_confidence_when_correct") is not None]
    conf_bad = [l["mean_confidence_when_wrong"] for l in logs if l.get("mean_confidence_when_wrong") is not None]
    return {
        "n_runs": len(logs),
        "mean_accuracy_raw_label": round(mean(raw), 3) if raw else None,
        "mean_accuracy_mood": round(mean(mood), 3) if mood else None,
        "mean_confidence_when_correct": round(mean(conf_ok), 3) if conf_ok else None,
        "mean_confidence_when_wrong": round(mean(conf_bad), 3) if conf_bad else None,
    }


def task_scan_benchmark(logs: list[dict]) -> dict:
    """Aggregates benchmark_scan_task.py runs (full painting-scanning procedure)."""
    if not logs:
        return {"n_runs": 0}
    completion = [l["completion_rate"] for l in logs if l.get("completion_rate") is not None]
    wall_s = [l["mean_wall_s"] for l in logs if l.get("mean_wall_s") is not None]
    classification = [l["mean_classification_accuracy"] for l in logs if l.get("mean_classification_accuracy") is not None]
    wall_graph = [l["mean_scene_graph_wall_accuracy"] for l in logs if l.get("mean_scene_graph_wall_accuracy") is not None]
    home_err = [l["mean_return_to_home_error_m"] for l in logs if l.get("mean_return_to_home_error_m") is not None]
    return {
        "n_runs": len(logs),
        "mean_completion_rate": round(mean(completion), 3) if completion else None,
        "mean_wall_s": round(mean(wall_s), 1) if wall_s else None,
        "mean_classification_accuracy": round(mean(classification), 3) if classification else None,
        "mean_scene_graph_wall_accuracy": round(mean(wall_graph), 3) if wall_graph else None,
        "mean_return_to_home_error_m": round(mean(home_err), 4) if home_err else None,
    }


def task_benchmark(sessions: list[dict]) -> dict:
    """End-to-end: completion, turns, duration, paintings shown — over EVERY
    session ever run (opportunistic, not just the standardized scenarios)."""
    if not sessions:
        return {"n_sessions": 0}
    return {
        "n_sessions": len(sessions),
        "mean_turns": round(mean(s["turns"] for s in sessions), 2),
        "mean_duration_s": round(mean(s["duration_s"] for s in sessions), 1),
        "mean_paintings_shown": round(mean(s["paintings_shown"] for s in sessions), 2),
        "mean_clarifications": round(mean(s["clarification_count"] for s in sessions), 2),
    }


def task_scenario_benchmark(logs: list[dict]) -> dict:
    """Aggregates benchmark_task.py runs: per-scenario completion/success
    rate across the STANDARDIZED, repeatable scripted visitor scenarios
    (as opposed to task_benchmark()'s opportunistic session log average)."""
    if not logs:
        return {"n_runs": 0, "scenarios": {}}
    by_scenario: dict[str, list[dict]] = {}
    for log in logs:
        for s in log.get("scenarios", []):
            by_scenario.setdefault(s["scenario"], []).append(s)
    scenarios = {
        name: {
            "mean_completion_rate": round(mean(e["completion_rate"] for e in entries), 3),
            "mean_success_rate": round(mean(e["success_rate"] for e in entries), 3),
        }
        for name, entries in sorted(by_scenario.items())
    }
    return {"n_runs": len(logs), "scenarios": scenarios}


def task_subgoal_benchmark(logs: list[dict]) -> dict:
    """
    Per-scenario, per-sub-goal pass rate across every benchmark_task.py run
    (see _SCENARIOS' "subgoals" lists there) — the ERL-style decomposition
    Suriani described (task -> separately evaluable sub-goals) made visible
    at the report level, one line per sub-goal instead of one line per
    scenario. Complements task_scenario_benchmark()'s single success_rate:
    a scenario stuck at success_rate=0.67 could be one sub-goal failing a
    third of the time, or three sub-goals each failing independently a
    ninth of the time — this is what tells them apart.
    """
    if not logs:
        return {"n_runs": 0, "scenarios": {}}
    by_scenario: dict[str, list[dict]] = {}
    for log in logs:
        for s in log.get("scenarios", []):
            if s.get("subgoal_rates"):
                by_scenario.setdefault(s["scenario"], []).append(s["subgoal_rates"])
    scenarios = {}
    for name, rate_dicts in sorted(by_scenario.items()):
        all_names = sorted({sg for d in rate_dicts for sg in d})
        scenarios[name] = {
            sg: round(mean(d[sg] for d in rate_dicts if sg in d), 3) for sg in all_names
        }
    return {"n_runs": len(logs), "scenarios": scenarios}


_BASELINE_SCENARIOS = {"novice_color", "expert_dramatic", "staff_ops", "social_safety_refusal"}
_STRESS_SCENARIOS = {
    "compound_intent", "contradictory_mood", "rapid_rejections", "multimodal_mood_conflict",
}


def task_quality_benchmark(logs: list[dict]) -> dict:
    """
    Aggregates benchmark_task.py's per-run "metrics" dict (recommendation
    optimality, turns/duration, clarification ratio, gesture success rate —
    see benchmark_task._standardized_metrics()) separately for the baseline
    and stress scenario groups.
    """
    groups: dict[str, list[dict]] = {"baseline": [], "stress": []}
    for log in logs:
        for s in log.get("scenarios", []):
            group = (
                "baseline" if s["scenario"] in _BASELINE_SCENARIOS else
                "stress" if s["scenario"] in _STRESS_SCENARIOS else None
            )
            if group is None:
                continue
            for run in s.get("runs", []):
                metrics = run.get("metrics")
                if metrics:
                    groups[group].append(metrics)

    def _agg(entries: list[dict]) -> dict:
        if not entries:
            return {"n": 0}
        optimality = [e["recommendation_optimality"] for e in entries if e.get("recommendation_optimality") is not None]
        clarification = [e["clarification_ratio"] for e in entries if e.get("clarification_ratio") is not None]
        gesture = [e["gesture_success_rate"] for e in entries if e.get("gesture_success_rate") is not None]
        turns = [e["turns"] for e in entries if e.get("turns") is not None]
        duration = [e["duration_s"] for e in entries if e.get("duration_s") is not None]
        return {
            "n": len(entries),
            "mean_recommendation_optimality": round(mean(optimality), 3) if optimality else None,
            "mean_turns": round(mean(turns), 2) if turns else None,
            "mean_duration_s": round(mean(duration), 1) if duration else None,
            "mean_clarification_ratio": round(mean(clarification), 3) if clarification else None,
            "mean_gesture_success_rate": round(mean(gesture), 3) if gesture else None,
        }

    return {"baseline": _agg(groups["baseline"]), "stress": _agg(groups["stress"])}


def survey_benchmark(surveys: list[dict]) -> dict:
    """Godspeed ratings, averaged per dimension (HRI7-Evaluation)."""
    if not surveys:
        return {"n_surveys": 0}
    dims: dict[str, list[int]] = {}
    for s in surveys:
        for k, v in s.get("godspeed_ratings", {}).items():
            dims.setdefault(k, []).append(v)
    return {
        "n_surveys": len(surveys),
        "mean_ratings": {k: round(mean(v), 2) for k, v in dims.items()},
    }


def _significance_stars(p_value: float) -> str:
    """HRI7-Evaluation "HRI Analysis" slide's own convention."""
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return "ns"


def synthetic_survey_benchmark(surveys: list[dict]) -> dict:
    """
    HRI7-Evaluation: between-subject comparison of the GenAI-simulated
    Godspeed ratings from synthetic_survey.py's H1/H2 conditions. These are
    NOT real participant data — see report/evaluation.tex's synthetic-data
    disclosure section — this function exists to demonstrate the analysis
    methodology the slides teach (independent t-test, significance stars),
    not to report a genuine empirical finding.

    Groups by (hypothesis, condition), computes per-dimension mean/stddev/N
    for each of the two conditions, then an independent (Welch, unequal
    variance) t-test between them per dimension — same test and the same
    p<0.05/0.01/0.001 star convention as the slide's own worked t-test
    example ("HRI Analysis").
    """
    if not surveys:
        return {"n_surveys": 0, "hypotheses": {}}

    grouped: dict[str, dict[str, dict[str, list[int]]]] = {}
    for s in surveys:
        iv = s.get("independent_variables", {})
        hyp, cond = iv.get("hypothesis"), iv.get("condition")
        if not hyp or not cond:
            continue
        for dim, val in s.get("godspeed_ratings", {}).items():
            grouped.setdefault(hyp, {}).setdefault(cond, {}).setdefault(dim, []).append(val)

    results: dict[str, dict] = {}
    for hyp, conditions in sorted(grouped.items()):
        cond_names = sorted(conditions.keys())
        if len(cond_names) != 2:
            results[hyp] = {"error": f"expected exactly 2 conditions, found {cond_names}"}
            continue
        cond_a, cond_b = cond_names
        dims = sorted(set(conditions[cond_a]) & set(conditions[cond_b]))
        per_dim = {}
        for dim in dims:
            a, b = conditions[cond_a][dim], conditions[cond_b][dim]
            if len(a) < 2 or len(b) < 2:
                continue
            t_stat, p_value = scipy_stats.ttest_ind(a, b, equal_var=False)
            per_dim[dim] = {
                cond_a: {"mean": round(mean(a), 2), "stddev": round(pstdev(a), 2), "n": len(a)},
                cond_b: {"mean": round(mean(b), 2), "stddev": round(pstdev(b), 2), "n": len(b)},
                "t_stat": round(float(t_stat), 3),
                "p_value": round(float(p_value), 5),
                "significance": _significance_stars(float(p_value)),
            }
        results[hyp] = {"conditions": cond_names, "dimensions": per_dim}

    return {"n_surveys": len(surveys), "hypotheses": results}


def main() -> None:
    interactions = _load_all("interaction_*.json")
    sessions = _load_all("session_*.json")
    surveys = _load_all("survey_*.json")
    synthetic_surveys = _load_all("synthetic_survey_*.json")
    perception_logs = _load_all("functionality_perception_*.json")
    interaction_logs = _load_all("functionality_interaction_*.json")
    planning_logs = _load_all("functionality_planning_*.json")
    locomotion_logs = _load_all("functionality_locomotion_*.json")
    face_emotion_logs = _load_all("functionality_face_emotion_*.json")
    semantic_map_logs = _load_all("functionality_semantic_map_*.json")
    task_scenario_logs = _load_all("task_scenario_*.json")
    task_scan_logs = _load_all("task_scan_*.json")

    print("=" * 60)
    print("HRAI Benchmark Report")
    print(f"Log directory: {_LOG_DIR}")
    print("=" * 60)

    print("\n[Functionality benchmark — Control (self-awareness calibration)]")
    for k, v in functionality_selfawareness_benchmark(interactions).items():
        print(f"  {k}: {v}")

    print("\n[Functionality benchmark — Perception (VLM classification)]")
    for k, v in functionality_perception_benchmark(perception_logs).items():
        print(f"  {k}: {v}")

    print("\n[Functionality benchmark — Interaction (NLU intent/language)]")
    for k, v in functionality_interaction_benchmark(interaction_logs).items():
        print(f"  {k}: {v}")

    print("\n[Functionality benchmark — Planning/Control (IK pointing)]")
    for k, v in functionality_planning_benchmark(planning_logs).items():
        print(f"  {k}: {v}")

    print("\n[Functionality benchmark — Control (locomotion)]")
    for k, v in functionality_locomotion_benchmark(locomotion_logs).items():
        print(f"  {k}: {v}")

    print("\n[Functionality benchmark — Perception (face emotion classification)]")
    for k, v in functionality_face_emotion_benchmark(face_emotion_logs).items():
        print(f"  {k}: {v}")

    print("\n[Functionality benchmark — Semantic Mapping (δ(SM1, SMm) vs ground truth)]")
    for k, v in functionality_semantic_map_benchmark(semantic_map_logs).items():
        print(f"  {k}: {v}")

    print("\n[Task benchmark — every session ever run (opportunistic)]")
    for k, v in task_benchmark(sessions).items():
        print(f"  {k}: {v}")

    print("\n[Task benchmark — standardized scripted scenarios]")
    scenario_report = task_scenario_benchmark(task_scenario_logs)
    print(f"  n_runs: {scenario_report['n_runs']}")
    for name, stats in scenario_report["scenarios"].items():
        print(f"  {name}: {stats}")

    print("\n[Task benchmark — sub-goal checklist per scenario]")
    subgoal_report = task_subgoal_benchmark(task_scenario_logs)
    for name, subgoals in subgoal_report["scenarios"].items():
        print(f"  {name}:")
        for sg_name, rate in subgoals.items():
            print(f"      {sg_name:32s} {rate:.0%}")

    print("\n[Task benchmark — full scanning procedure]")
    for k, v in task_scan_benchmark(task_scan_logs).items():
        print(f"  {k}: {v}")

    print("\n[Task benchmark — standardized quality metrics (baseline vs stress)]")
    quality_report = task_quality_benchmark(task_scenario_logs)
    for group_name, stats in quality_report.items():
        print(f"  {group_name}: {stats}")

    print("\n[User-experience benchmark — Godspeed survey]")
    for k, v in survey_benchmark(surveys).items():
        print(f"  {k}: {v}")

    print("\n[Evaluation — synthetic Godspeed comparison (GenAI-simulated "
          "participants, NOT real users — see report/evaluation.tex)]")
    synthetic_report = synthetic_survey_benchmark(synthetic_surveys)
    print(f"  n_surveys: {synthetic_report['n_surveys']}")
    for hyp, data in synthetic_report["hypotheses"].items():
        if "error" in data:
            print(f"  {hyp}: {data['error']}")
            continue
        print(f"  {hyp}  conditions={data['conditions']}")
        for dim, d in data["dimensions"].items():
            cond_a, cond_b = data["conditions"]
            print(f"    {dim}: {cond_a}={d[cond_a]}  {cond_b}={d[cond_b]}  "
                  f"t={d['t_stat']}  p={d['p_value']} {d['significance']}")


if __name__ == "__main__":
    main()
