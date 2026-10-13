#!/usr/bin/env python3
"""
Functionality benchmark — Interaction domain.

HRAI - lecture1 (Intro + Benchmarking), slide "Common Functionality
Benchmark Tools": functionality domains include "Interaction: speech
recognition, gesture interpretation" — here, the NLU half of our
interaction pipeline: llm_interpreter.analyze_input()'s intent
classification and language detection, isolated from dialogue routing
(dialogue_manager) and from utterance generation ("without necessarily
completing a full task", same slide as the domain list).

Ground truth: a small hand-labeled set of utterances fixed in this file
(not derived from live session logs), so the benchmark is a repeatable,
version-controlled dataset test — HRAI-lecture1 "Principles of Effective
Benchmarking": Repeatability + Well-Defined Metrics. Cases were chosen to
cover both the ordinary intents and the edge cases debugged live this
session (mood-shift-implies-new-recommendation, staff-claim-without-
passphrase must NOT be honored as staff_* by a visitor role, multilingual
greetings).

Needs GROQ_API_KEY; does not touch circus/T1/the simulator.

Usage (from circus/ directory):
    pixi run python ../exchange/museum_guide_src/benchmark_interaction.py
"""
import json
import time
from pathlib import Path

from llm_interpreter import analyze_input

_REPO_ROOT = Path(__file__).resolve().parents[2]
_LOG_DIR = _REPO_ROOT / "exchange" / "logs"

_TEST_SET = [
    ("Ciao!",                                                          "greeting",              "it", "visitor"),
    ("Hello!",                                                         "greeting",               "en", "visitor"),
    ("Bonjour!",                                                       "greeting",               "fr", "visitor"),
    ("Vorrei vedere un quadro colorato",                               "request_recommendation", "it", "visitor"),
    ("I'd like to see something dramatic",                             "request_recommendation", "en", "visitor"),
    ("Mi è piaciuto molto ma vorrei vedere qualcosa di più drammatico","request_recommendation", "it", "visitor"),
    ("Puoi dirmi di più sulla tecnica?",                                "wants_more_detail",      "it", "visitor"),
    ("Interesting, what about the artist?",                            "wants_more_detail",      "en", "visitor"),
    ("Mi è piaciuto molto!",                                           "feedback",               "it", "visitor"),
    ("It wasn't really my style",                                      "feedback",               "en", "visitor"),
    ("No grazie, sono soddisfatto",                                    "goodbye",                "it", "visitor"),
    ("No thank you, I'm fine",                                         "goodbye",                "en", "visitor"),
    ("Arrivederci!",                                                   "goodbye",                "it", "visitor"),
    ("Cos'è l'impressionismo?",                                        "general_question",       "it", "visitor"),
    ("What is this museum about?",                                     "general_question",       "en", "visitor"),
    ("sono dello staff",                                                "other",                  "it", "visitor"),
    ("1234, riscansiona la sala",                                       "staff_rescan",            "it", "staff"),
    ("il quadro Water Lilies è in manutenzione",                       "staff_disable_painting",  "it", "staff"),
    ("riattiva il quadro Water Lilies",                                "staff_enable_painting",   "it", "staff"),
    ("Hola, quiero ver un cuadro",                                     "request_recommendation", "es", "visitor"),
    ("Hallo, ich möchte ein Bild sehen",                                "request_recommendation", "de", "visitor"),
]


def benchmark_interaction() -> dict:
    results = []
    intent_correct = 0
    lang_correct = 0
    times = []

    for text, expected_intent, expected_lang, role in _TEST_SET:
        t0 = time.perf_counter()
        try:
            out = analyze_input(text, history=[], role=role)
            error = None
        except Exception as e:
            out = {"intent": "other", "user_model_update": {}}
            error = str(e)
        dt = time.perf_counter() - t0
        times.append(dt)

        got_intent = out.get("intent")
        got_lang = out.get("user_model_update", {}).get("language", "unknown")
        intent_ok = got_intent == expected_intent
        lang_ok = got_lang == expected_lang
        intent_correct += intent_ok
        lang_correct += lang_ok

        results.append({
            "text": text, "role": role,
            "expected_intent": expected_intent, "got_intent": got_intent, "intent_ok": intent_ok,
            "expected_lang": expected_lang, "got_lang": got_lang, "lang_ok": lang_ok,
            "latency_ms": round(1000 * dt, 1),
            "error": error,
        })

    n = len(_TEST_SET)
    report = {
        "domain": "interaction_nlu",
        "n_cases": n,
        "accuracy_intent": round(intent_correct / n, 3),
        "accuracy_language": round(lang_correct / n, 3),
        "efficiency_mean_latency_ms": round(1000 * sum(times) / len(times), 1),
        "cases": results,
    }
    return report


def main() -> None:
    report = benchmark_interaction()

    print("=" * 60)
    print("Functionality Benchmark — Interaction (NLU: intent + language)")
    print("=" * 60)
    print(f"  Cases: {report['n_cases']}")
    print(f"  Accuracy (intent):   {report['accuracy_intent']:.1%}")
    print(f"  Accuracy (language): {report['accuracy_language']:.1%}")
    print(f"  Efficiency — mean latency: {report['efficiency_mean_latency_ms']} ms")
    print()
    for c in report["cases"]:
        mark_i = "✓" if c["intent_ok"] else "✗"
        mark_l = "✓" if c["lang_ok"] else "✗"
        print(f"    {mark_i}{mark_l}  \"{c['text'][:45]:45s}\"  "
              f"intent={c['got_intent']:24s} (want {c['expected_intent']:24s})  "
              f"lang={c['got_lang']} (want {c['expected_lang']})")

    _LOG_DIR.mkdir(parents=True, exist_ok=True)
    existing = sorted(_LOG_DIR.glob("functionality_interaction_*.json"))
    index = len(existing) + 1
    out_path = _LOG_DIR / f"functionality_interaction_{index:03d}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n📝 Log salvato in: {out_path}")


if __name__ == "__main__":
    main()
