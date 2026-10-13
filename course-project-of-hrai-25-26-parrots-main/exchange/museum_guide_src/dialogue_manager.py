#!/usr/bin/env python3

CONFIDENCE_THRESHOLD = 0.65
MAX_CLARIFICATION_ATTEMPTS = 2


def initialize_state() -> dict:
    return {
        "phase": "greeting",
        "user_model": {
            "profile": "unknown",
            "interests": {},
            "available_time": "unknown",
            "preferred_mood": "unknown",
            "preferred_mood_source": None,
            "language": "en",
            "role": "visitor",
        },
        "confidence": 0.0,
        "history": [],
        "last_painting": None,
        "shown_painting_ids": [],
        "clarification_count": 0,
        "social_refusal_count": 0,
        "staff_command_executed": False,
        "auto_continue": False,
    }


def get_next_action(state: dict, intent: str, confidence: float) -> tuple[str, str]:
    """
    Pure routing: given current state, intent, and confidence,
    returns (action, next_phase).

    All state transitions live here — no phase updates elsewhere.

    Actions:
        greeting_ack        — acknowledge greeting, ask open question
        ask_clarification   — ask one targeted question to gather visitor info
        recommend           — run the full recommendation pipeline
        answer_question     — answer a general art/museum question
        answer_more_detail  — provide more detail on the last recommended painting
        close_or_restart    — visitor satisfied; offer to continue or end
        goodbye             — end session, reset for next visitor

    Phases:
        greeting    — first contact, no info gathered yet
        profiling   — at least one clarification asked, building user model
        presenting  — recommendation just made, awaiting visitor reaction
        follow_up   — satisfaction check done, deciding whether to continue
    """
    phase = state["phase"]
    clarification_count = state.get("clarification_count", 0)

    _um = state["user_model"]
    model_has_info = (
        _um.get("profile", "unknown") != "unknown"
        or bool(_um.get("interests", {}))
        or _um.get("available_time", "unknown") != "unknown"
    )

    if intent == "goodbye":
        return "goodbye", "greeting"

    if intent in ("staff_rescan", "staff_disable_painting", "staff_enable_painting"):
        if _um.get("role") == "staff":
            return intent, phase
        intent = "other"

    if intent == "asks_robot_preference":
        return "robot_preference", phase

    if intent == "wants_more_detail" and state.get("last_painting"):
        return "answer_more_detail", "follow_up"

    if phase == "greeting":
        if intent == "greeting":
            return "greeting_ack", "greeting"
        if intent == "general_question":
            return "answer_question", "greeting"
        if confidence >= CONFIDENCE_THRESHOLD or model_has_info:
            return "recommend", "presenting"
        return "ask_clarification", "profiling"

    if phase == "profiling":
        if intent == "general_question":
            return "answer_question", "profiling"
        if (confidence >= CONFIDENCE_THRESHOLD
                or model_has_info
                or clarification_count >= MAX_CLARIFICATION_ATTEMPTS):
            return "recommend", "presenting"
        return "ask_clarification", "profiling"

    if phase == "presenting":
        if intent == "general_question":
            return "answer_question", "presenting"
        if intent == "request_recommendation":
            return "recommend", "presenting"
        return "close_or_restart", "follow_up"

    if phase == "follow_up":
        if intent == "request_recommendation":
            return "recommend", "presenting"
        if intent == "general_question":
            return "answer_question", "follow_up"
        return "goodbye", "greeting"

    return "ask_clarification", "profiling"
