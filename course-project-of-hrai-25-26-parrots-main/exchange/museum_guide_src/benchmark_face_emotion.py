#!/usr/bin/env python3
"""
Functionality benchmark — Perception domain (face emotion classification).

HRAI - lecture1 (Intro + Benchmarking), slide "Common Functionality
Benchmark Tools": functionality domains include "Perception: image
classification" — this benchmark does for the face-emotion module
(hsemotion-onnx, guida_museale_core.detect_mood_from_frames) exactly what
benchmark_perception.py already does for the VLM painting classifier: feed
images with a KNOWN label and check the prediction, in isolation from the
rest of the pipeline (no circus/T1, no simulator — pure classifier test).
This module had ZERO test coverage before (verified: no "hsemotion" match
anywhere in benchmark_*.py).

Images: exchange/data/benchmark_faces/ (see SOURCES.md there for exact
provenance/labels and an honest account of how noisy the source dataset
turned out to be during selection).

Metrics (slide "Common Metrics"): Accuracy at two levels — the RAW
hsemotion label (happy/sad/surprise/neutral/angry/disgust/fear/contempt)
and the MOOD it maps to via guida_museale_core._EMOTION_TO_MOOD (a
many-to-one mapping, so mood accuracy should be >= raw-label accuracy).
Efficiency is not measured here (single deterministic forward pass, unlike
the VLM's LLM-backed, genuinely variable-latency calls) — repeating a call
on the same static image would not reveal anything new, so each image is
classified once, not N times like benchmark_perception.py's VLM calls.

Usage (from circus/ directory), no real GROQ_API_KEY/circus/T1 needed:
    pixi run python ../exchange/museum_guide_src/benchmark_face_emotion.py
"""
import json
import os
from pathlib import Path

import cv2

os.environ.setdefault("GROQ_API_KEY", "unused-placeholder-for-face-emotion-benchmark")

import guida_museale_core as guide

_REPO_ROOT = Path(__file__).resolve().parents[2]
_FACES_DIR = _REPO_ROOT / "exchange" / "data" / "benchmark_faces"
_LOG_DIR = _REPO_ROOT / "exchange" / "logs"

_LABELED_IMAGES = [
    ("happiness.jpg",       "happiness", "dynamic"),
    ("happiness_extra.jpg", "happiness", "dynamic"),
    ("sadness.jpg",         "sadness",   "serene"),
    ("neutral.jpg",         "neutral",   None),
    ("neutral_extra.jpg",   "neutral",   None),
    ("anger.jpg",           "anger",     "contemplative"),
    ("disgust.jpg",         "disgust",   "contemplative"),
    ("fear_undetected_example.jpg",       "fear",     "peaceful"),
    ("surprise_misclassified_example.jpg", "surprise", "sublime"),
]

_HAAR_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def _classify_raw(img) -> str | None:
    """
    Single-image raw hsemotion label, bypassing detect_mood_from_frames()'s
    multi-frame majority-voting wrapper (built for a live camera stream, not
    a single static image) — reuses the same face detector + recognizer
    (guida_museale_core._get_emotion_recognizer()) so this is testing the
    real model, not a reimplementation of it. detect_mood_from_frames()
    itself is used separately below for the mood-level (real system
    behaviour) accuracy check.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = _HAAR_CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48))
    if not len(faces):
        return None
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    face_crop = img[y:y + h, x:x + w]
    recognizer = guide._get_emotion_recognizer()
    emotion, _ = recognizer.predict_emotions(face_crop, logits=False)
    return emotion.lower()


def benchmark_face_emotion() -> dict:
    results = []
    for filename, expected_raw, expected_mood in _LABELED_IMAGES:
        path = _FACES_DIR / filename
        img = cv2.imread(str(path))
        if img is None:
            results.append({"file": filename, "error": "could not read image"})
            continue

        raw = _classify_raw(img)
        mood, conf = guide.detect_mood_from_frames([img])

        results.append({
            "file": filename,
            "expected_raw": expected_raw,
            "predicted_raw": raw,
            "raw_correct": raw == expected_raw,
            "expected_mood": expected_mood,
            "predicted_mood": mood,
            "mood_correct": mood == expected_mood,
            "confidence": conf,
        })

    n = len(results)
    raw_correct = sum(r.get("raw_correct", False) for r in results)
    mood_correct = sum(r.get("mood_correct", False) for r in results)
    conf_correct = [r["confidence"] for r in results if r.get("mood_correct")]
    conf_wrong = [r["confidence"] for r in results if "mood_correct" in r and not r["mood_correct"]]

    return {
        "domain": "perception_face_emotion",
        "n_images": n,
        "accuracy_raw_label": round(raw_correct / n, 3) if n else None,
        "accuracy_mood": round(mood_correct / n, 3) if n else None,
        "mean_confidence_when_correct": round(sum(conf_correct) / len(conf_correct), 3) if conf_correct else None,
        "mean_confidence_when_wrong": round(sum(conf_wrong) / len(conf_wrong), 3) if conf_wrong else None,
        "results": results,
    }


def main() -> None:
    report = benchmark_face_emotion()

    print("=" * 60)
    print("Functionality Benchmark — Perception (face emotion classification)")
    print("=" * 60)
    print(f"  Images: {report['n_images']}")
    print(f"  Accuracy (raw hsemotion label): {report['accuracy_raw_label']:.1%}")
    print(f"  Accuracy (mapped mood):         {report['accuracy_mood']:.1%}")
    print(f"  Mean confidence when correct: {report['mean_confidence_when_correct']}")
    print(f"  Mean confidence when wrong:   {report['mean_confidence_when_wrong']}")
    for r in report["results"]:
        if "error" in r:
            print(f"    {r['file']:38s} ERROR: {r['error']}")
            continue
        mark = "OK" if r["mood_correct"] else "FAIL"
        print(f"    {r['file']:38s} {mark:4s} expected={r['expected_mood']!r:14s} "
              f"got={r['predicted_mood']!r:14s} (raw expected={r['expected_raw']}, "
              f"got={r['predicted_raw']})")

    _LOG_DIR.mkdir(parents=True, exist_ok=True)
    existing = sorted(_LOG_DIR.glob("functionality_face_emotion_*.json"))
    index = len(existing) + 1
    out_path = _LOG_DIR / f"functionality_face_emotion_{index:03d}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n📝 Log salvato in: {out_path}")


if __name__ == "__main__":
    main()
