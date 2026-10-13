#!/usr/bin/env python3
"""
HRI8 - Cognitive Architectures: Social Supervisor (safety filter).

This is the front-of-pipeline "Toxicity trigger — action not performed" box of
the Minimal Robot Cognitive Architecture, and the "Use case 1: unacceptable
social request" of the lecture: before any intent parsing or action runs, the
visitor's message is classified against a reduced, Llama-Guard-style unsafe-
content taxonomy. An unsafe request is refused and NO downstream action is
performed (analyze_input / the FSM / the handlers are all skipped for that turn).

Deliberately a separate, dedicated LLM call (not folded into analyze_input): a
safety gate should be an explicit, auditable stage, not one field of a larger
best-effort classification — the same design reason staff authentication is a
deterministic gate rather than an LLM-extracted flag.

Toggle with the SOCIAL_SUPERVISOR env var (default on): SOCIAL_SUPERVISOR=0
disables it, mirroring the GESTURE_MODE env-toggle pattern used elsewhere.
"""
import json
import os
import console

from groq import Groq

client = Groq()
from models import CHAT_MODEL as _GUARD_MODEL, chat_kwargs, explain_model_error

UNSAFE_CATEGORIES = {
    "violence":           "requests to plan, threaten, or carry out physical harm",
    "hate_discrimination":"requests to produce hateful, discriminatory, or demeaning content toward a group or person",
    "sexual":             "sexual solicitation or explicit sexual content directed at the robot or a person",
    "self_harm":          "content encouraging or seeking help to carry out self-harm or suicide",
    "illegal_activity":   "requests to facilitate crimes, weapons, drugs, or other clearly illegal acts",
    "harassment":         "targeted insults, bullying, or degrading language aimed at a person",
    "privacy":            "requests to reveal or misuse private/personal data about identifiable people",
}


def is_enabled() -> bool:
    return os.environ.get("SOCIAL_SUPERVISOR", "1").strip().lower() not in ("0", "false", "off", "no")


def check_social_safety(user_text: str, history: list | None = None) -> dict:
    """
    Classify whether `user_text` is a socially acceptable request/statement for
    a public museum guide robot.

    Returns {"safe": bool, "category": str|None, "reason": str}. Fails OPEN
    (safe=True) if the classifier call errors out — a rate-limited safety check
    must not brick an otherwise harmless museum conversation; the refusal path
    is a guard-rail, not the primary behaviour.
    """
    if not is_enabled() or not user_text.strip():
        return {"safe": True, "category": None, "reason": "supervisor disabled or empty input"}

    categories_text = "\n".join(f"- {name}: {desc}" for name, desc in UNSAFE_CATEGORIES.items())
    system_prompt = (
        "You are the Social Supervisor safety filter of a robotic museum guide that interacts with "
        "the public. Classify ONLY the visitor's latest message: is it a socially acceptable thing to "
        "say to a museum assistant, or does it fall into one of the unsafe categories below?\n\n"
        f"Unsafe categories:\n{categories_text}\n\n"
        "IMPORTANT — avoid false positives: discussing artworks that DEPICT nudity, death, war, religion "
        "or violence is SAFE, that is normal museum conversation. Ordinary requests, small talk, "
        "criticism of the museum, and off-topic-but-harmless questions are SAFE. Mark a message unsafe "
        "ONLY when the request or statement itself is genuinely harmful, abusive, or illegal.\n\n"
        'Respond with a JSON object only: {"safe": true|false, "category": <one of the category names or '
        'null>, "reason": <one short sentence>}.'
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Visitor message: {user_text}"},
    ]
    try:
        response = client.chat.completions.create(
            messages=messages,
            **chat_kwargs(),
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        raw = json.loads(response.choices[0].message.content)
    except Exception as e:
        console.warn("[Social Supervisor] classifier error — failing open (safe): "
                     f"{explain_model_error(e, _GUARD_MODEL)}")
        return {"safe": True, "category": None, "reason": f"classifier error: {e}"}

    safe = bool(raw.get("safe", True))
    category = raw.get("category")
    if category not in UNSAFE_CATEGORIES:
        category = None
    if safe:
        category = None
    return {
        "safe": safe,
        "category": category,
        "reason": str(raw.get("reason", ""))[:200],
    }
