#!/usr/bin/env python3
"""
Sends a camera frame to a Groq vision model and returns structured painting info.
"""

import base64
import json
import os
import re
from typing import Any

import numpy as np

from models import VISION_MODEL as _VISION_MODEL, explain_model_error
import console

_REASONING_EFFORT = "none"

_PROMPT_TEMPLATE = """\
You are an art expert looking at a camera image inside a small museum room.
{focus_instruction}

This museum room contains EXACTLY these {n_candidates} paintings — there is no other possibility:
{candidates_block}

Identify which of these paintings is shown, based on its composition, colors, and subject matter.
Return a JSON object with exactly these fields:
  "title"       – must be EXACTLY one of the titles listed above, copied verbatim
  "artist"      – the matching artist from the list above
  "movement"    – art movement (e.g. "Impressionism", "Expressionism", "Baroque", "Romanticism")
  "mood"        – list of moods/emotions evoked (e.g. ["calm", "melancholy", "dramatic"])
  "themes"      – list of main subjects/themes (e.g. ["nature", "portrait", "war", "mythology"])
  "description" – one sentence describing what you see in that painting

Return ONLY the raw JSON object, no markdown, no code fences, no explanation.\
"""

_PROMPT_TEMPLATE_OPEN = """\
You are an art expert looking at a camera image inside a museum.
{focus_instruction}
Identify that painting and return a JSON object with exactly these fields:
  "title"       – painting title (or "Unknown")
  "artist"      – artist full name (or "Unknown")
  "movement"    – art movement (e.g. "Impressionism", "Expressionism", "Baroque", "Romanticism")
  "mood"        – list of moods/emotions evoked (e.g. ["calm", "melancholy", "dramatic"])
  "themes"      – list of main subjects/themes (e.g. ["nature", "portrait", "war", "mythology"])
  "description" – one sentence describing what you see in that painting

Return ONLY the raw JSON object, no markdown, no code fences, no explanation.\
"""

_FOCUS_INSTRUCTIONS = {
    "left":   "Focus ONLY on the painting on the LEFT side of the image. Ignore any other paintings.",
    "right":  "Focus ONLY on the painting on the RIGHT side of the image. Ignore any other paintings.",
    "center": "There is one main painting visible. Identify it.",
}


def _frame_to_base64(frame: np.ndarray) -> str:
    import cv2
    ok, buf = cv2.imencode(".png", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    if not ok:
        raise ValueError("PNG encoding of camera frame failed")
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def _title_from_raw(raw: str, candidates: list[dict] | None) -> dict[str, Any] | None:
    """
    Last-resort recovery from a reply that identified the painting but did not
    produce valid JSON: look for candidate titles in the raw text.

    Requires EXACTLY one match. A reply mentioning two catalog titles is a
    model comparing options, not answering, and guessing between them would
    turn a clean parse failure into a silent misidentification — which the
    registry would then report as a hallucination.
    """
    if not candidates:
        return None
    lowered = raw.lower()
    hits = [c for c in candidates if c["title"].strip().lower() in lowered]
    if len(hits) != 1:
        return None
    c = hits[0]
    return {
        "title": c["title"],
        "artist": c["artist"],
        "movement": c.get("movement", ""),
        "mood": [],
        "themes": [],
        "description": "",
        "painting_id": c.get("id"),
        "recovered_from_malformed_json": True,
        "raw_response": raw,
    }


def _snap_to_candidate(result: dict[str, Any], candidates: list[dict]) -> dict[str, Any]:
    """
    Safety net: if the VLM didn't copy a title verbatim (small wording/case
    differences), snap it to the closest candidate by title, and attach the
    candidate's catalog id as "painting_id" — the key downstream placement
    resolution (knowledge_graph.resolve_placement) is built on. A result
    without painting_id means the VLM answered outside the catalog.
    """
    observed = str(result.get("title", "")).strip().lower()
    if not observed:
        return result
    for c in candidates:
        expected = c["title"].strip().lower()
        if observed == expected or observed in expected or expected in observed:
            result["title"] = c["title"]
            result["artist"] = c["artist"]
            if "id" in c:
                result["painting_id"] = c["id"]
            return result
    return result


def analyze_painting(
    frame: np.ndarray,
    api_key: str | None = None,
    focus: str = "center",
    candidates: list[dict] | None = None,
) -> dict[str, Any]:
    """
    Send frame to Groq vision model.
    focus: "left" | "right" | "center" — which part of image to focus on.
    candidates: optional list of {"title", "artist", "movement"} dicts — the
      known closed set of paintings that could possibly be in view. Strongly
      recommended: turns this from open-ended recall into closed-set
      classification, which is far more reliable at this resolution/angle.
    Returns a dict with keys: title, artist, movement, mood, themes, description.
    """
    if api_key is None:
        api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        raise ValueError("GROQ_API_KEY not set")

    from groq import Groq
    client = Groq(api_key=api_key)

    b64 = _frame_to_base64(frame)
    focus_instruction = _FOCUS_INSTRUCTIONS.get(focus, _FOCUS_INSTRUCTIONS["center"])

    if candidates:
        candidates_block = "\n".join(
            f'{i}. "{c["title"]}" by {c["artist"]} ({c.get("movement", "")})'
            for i, c in enumerate(candidates, start=1)
        )
        prompt = _PROMPT_TEMPLATE.format(
            focus_instruction=focus_instruction,
            n_candidates=len(candidates),
            candidates_block=candidates_block,
        )
    else:
        prompt = _PROMPT_TEMPLATE_OPEN.format(focus_instruction=focus_instruction)

    response = client.chat.completions.create(
        model=_VISION_MODEL,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text",      "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            ],
        }],
        max_tokens=512,
        temperature=0.1,
        reasoning_effort=_REASONING_EFFORT,
    )

    raw = response.choices[0].message.content.strip()

    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        result = None
        if m:
            try:
                result = json.loads(m.group())
            except json.JSONDecodeError:
                result = None
        if result is None:
            recovered = _title_from_raw(raw, candidates)
            if recovered is not None:
                return recovered
            return {"title": "Unknown", "artist": "Unknown", "raw_response": raw}

    if candidates:
        result = _snap_to_candidate(result, candidates)
    return result


def analyze_painting_voted(
    frame: np.ndarray,
    api_key: str | None = None,
    focus: str = "center",
    candidates: list[dict] | None = None,
    n_votes: int = 3,
) -> dict[str, Any]:
    """
    Calls analyze_painting n_votes times on the same frame and keeps the
    majority answer (by painting_id, falling back to title when the VLM
    answered outside the catalog).

    A first attempt at this (see git history) was reverted because it didn't
    help a *deterministic* misclassification caused by a bad crop (an
    oblique/distant framing that consistently read as the wrong painting —
    all 3 votes agreed on the same wrong answer). That framing bug is fixed
    now (see museum_explorer._SCAN_PLAN's crop_x/crop_y). The residual
    errors observed since are NOT unanimous across repeated calls on the same
    frame — i.e. genuine per-call stochastic noise (temperature=0.1 is not
    0) — which is exactly the failure mode majority voting fixes.
    """
    results = []
    for _ in range(n_votes):
        try:
            results.append(analyze_painting(frame, api_key, focus, candidates))
        except Exception as e:
            console.warn(f"[PaintingRecognizer] Chiamata fallita: {type(e).__name__}: {e}")
            results.append({"title": "Unknown", "error": str(e)})

    votes: dict[str, int] = {}
    for r in results:
        key = r.get("painting_id") or r.get("title", "Unknown")
        votes[key] = votes.get(key, 0) + 1

    scores_str = "  ".join(f"{k}:{v}" for k, v in sorted(votes.items(), key=lambda x: -x[1]))
    console.detail(f"[PaintingRecognizer] Voti ({n_votes} chiamate): {scores_str}")

    winner_key = max(votes, key=votes.get)
    for r in results:
        key = r.get("painting_id") or r.get("title", "Unknown")
        if key == winner_key:
            return r
    return results[0]
