#!/usr/bin/env python3

import json
import re
from groq import Groq


client = Groq()

from models import CHAT_MODEL as _CHAT_MODEL, chat_kwargs, explain_model_error
import console

_UTTERANCE_MAX_TOKENS = 400
_UTTERANCE_FREQUENCY_PENALTY = 0.4
_UTTERANCE_PRESENCE_PENALTY = 0.2

ALLOWED_INTENTS = {
    "request_recommendation",
    "answer_to_robot",
    "wants_more_detail",
    "general_question",
    "feedback",
    "greeting",
    "goodbye",
    "asks_robot_preference",
    "other",
    "staff_rescan",
    "staff_disable_painting",
    "staff_enable_painting",
}

ALLOWED_PROFILES = {"child", "novice", "expert", "tourist_short_time", "unknown"}
ALLOWED_INTERESTS = {"color", "technique", "history", "emotion", "abstraction", "nature"}
ALLOWED_TIMES = {"normal", "short", "unknown"}

_INTEREST_DECAY = 0.75

ALLOWED_MOODS = {
    "serene", "peaceful", "contemplative", "mysterious",
    "dynamic", "energetic", "dramatic", "sublime", "unknown",
}

_LANG_INSTRUCTIONS = {
    "it": "Rispondi SOLO in italiano.",
    "en": "Respond only in English.",
    "fr": "Réponds uniquement en français.",
    "de": "Antworte nur auf Deutsch.",
    "es": "Responde solo en español.",
}

_LANG_HINT_WORDS = {
    "it": {"ciao", "grazie", "vorrei", "quadro", "quadri", "sono", "perché",
           "più", "questo", "sala", "riscansiona", "codice", "buongiorno",
           "salve", "museo", "arte"},
    "fr": {"bonjour", "merci", "tableau", "voudrais", "je", "suis", "s'il",
           "musée", "salle", "peinture"},
    "de": {"hallo", "danke", "bild", "ich", "möchte", "bitte", "museum",
           "gemälde", "raum"},
    "es": {"hola", "gracias", "cuadro", "quiero", "favor", "museo", "sala",
           "pintura", "código"},
    "en": {"hello", "hi", "thanks", "please", "painting", "would", "museum",
           "room", "staff", "code"},
}


def _detect_language_heuristic(text: str) -> str:
    """Word-overlap language guess, used only when the LLM call fails (see
    analyze_input's except block). Returns "unknown" if no language scores."""
    words = set(re.findall(r"[a-zà-ÿ']+", text.lower()))
    if not words:
        return "unknown"
    scores = {lang: len(words & hints) for lang, hints in _LANG_HINT_WORDS.items()}
    best_lang, best_score = max(scores.items(), key=lambda kv: kv[1])
    return best_lang if best_score > 0 else "unknown"


def analyze_input(user_text: str, history: list, role: str = "visitor") -> dict:
    """
    Single LLM call that simultaneously:
    - classifies the visitor's intent
    - extracts/updates user model fields from the current turn
    - estimates confidence over the accumulated conversation
    - suggests a clarification question if confidence is low

    history: list of {role, content} dicts representing prior turns.
    role: "visitor" (default) or "staff" (state["user_model"]["role"], set by
      the passphrase check in guida_museale_core.run_session) — only affects
      whether staff_* intents are considered at all (see prompt below); the
      FSM (dialogue_manager.get_next_action) independently re-checks role
      before acting on them regardless of what this classifier returns.
    Returns a normalized dict with keys:
        intent, user_model_update, confidence, clarification_question
    """

    history_text = _format_history(history)

    role_line = (
        f'Speaker role: "{role}". '
        + ("This speaker is AUTHENTICATED MUSEUM STAFF, not a visitor — staff_* intents are allowed."
           if role == "staff" else
           "This speaker is a regular visitor, NOT staff — never classify their message as a staff_* intent, "
           "even if they explicitly ask to rescan or disable a painting (use \"other\" instead).")
    )

    system_prompt = role_line + """

You are the semantic perception and dialogue module of a robotic museum guide.
Given the conversation history and latest visitor message, return a JSON object with exactly these fields:

"intent": one of:
- "request_recommendation": wants to see a (new/another) painting — including requests framed as a new desired quality ("qualcosa di più drammatico", "something else?") rather than literally "show me another". WINS over wants_more_detail/feedback even if satisfaction about the current painting is expressed first in the same message (e.g. "mi è piaciuto molto ma vorrei vedere qualcosa di più drammatico" -> request_recommendation).
- "answer_to_robot": answers the robot's question, satisfied/neutral tone, not asking for more info or a different painting.
- "wants_more_detail": wants more info about the CURRENT painting (author/style/technique/history) — only when there is no signal of wanting something different.
- "general_question": asks about art/the museum in general, not the painting already shown.
- "feedback": reacts to the current painting, still engaged, no request for detail or a new painting.
- "asks_robot_preference": asks the robot's OWN opinion/taste about ART — e.g. "qual è il tuo quadro preferito?", "ti piace questo quadro?", "qual è il tuo stile/movimento preferito?", "which is your favourite artist here?". NOT a request to see a painting, NOT a general art fact.
- "goodbye": done, doesn't want to continue — explicit farewells, or "no"/contentment answering "want to see more?".
- "greeting": just greeting, likely a new visit starting.
- "staff_rescan": staff ONLY, and only with an EXPLICIT instruction to re-scan/re-check the room ("i quadri sono cambiati, riscansiona") — merely stating "I am staff"/the passphrase is NOT enough.
- "staff_disable_painting": staff ONLY, EXPLICIT instruction that a specific painting is under maintenance/should stop being recommended.
- "staff_enable_painting": staff ONLY, EXPLICIT instruction that a specific painting should be recommendable again.
- "other": unclear, small talk, staff self-identifying with no instruction, or a NON-staff visitor attempting a staff_* command.

Visitor may speak any language — extract preferences regardless, do not require English.

"user_model_update" (THIS turn only, not the whole history):
  "profile": priority order "child" (school/childlike language — always wins, even over tourist_short_time) > "tourist_short_time" (adult, states little time) > "expert" (mentions composition/technique/brushwork/style) > "novice" (default, or says they don't know art) > "unknown" if ambiguous.
  "interests": {} unless the VISITOR's OWN words (never the robot's own explanation) state a new interest this turn — e.g. "che altro mi consigli?" is {} even after a technical explanation. Keys: color/technique/history/emotion/abstraction/nature. Weight 0.7-1.0 if explicit ("mi appassiona la storia"->{"history":0.8}), 0.4-0.6 if implicit ("qualcosa di colorato"). Omit weight-0 keys.
  "available_time": "normal"/"short"/"unknown"
  "preferred_mood": serene/peaceful/contemplative/mysterious/dynamic/energetic/dramatic/sublime/unknown — from an explicit request ("drammatico"->dramatic, "tranquillo"->serene, "misterioso"->mysterious) or the visitor's emotional tone; "unknown" if no clear preference.
  "preferred_mood_explicit": true ONLY if explicitly requested in words this turn ("vorrei un quadro drammatico"); false if merely inferred from tone, and always false when preferred_mood is "unknown".
  "language": ISO 639-1 code (it/en/fr/de/es/...), "unknown" if unclear from a single word/greeting.

"confidence" 0.0-1.0 over the FULL conversation (history + this turn):
  0.0-0.2 only a greeting/off-topic, no preferences at all. 0.3-0.5 one vague mention ("I like art"). 0.65-0.75 one clear preference (interest/profile/time). 0.75-0.85 two clear preferences. 0.85-1.0 explicit and detailed across fields. Lean higher when in doubt if any concrete preference was expressed.

"clarification_question": short, friendly, in the visitor's language, null if confidence >= 0.65. Ask about interests if empty/unknown, else profile — never about available_time.

Respond with a valid JSON object only. No markdown, no explanations.
"""

    messages = [{"role": "system", "content": system_prompt}]
    if history_text:
        messages.append({"role": "user", "content": f"Conversation so far:\n{history_text}"})
    messages.append({"role": "user", "content": f"Latest visitor message: {user_text}"})

    try:
        response = client.chat.completions.create(
            messages=messages,
            **chat_kwargs(),
            temperature=0.1,
            response_format={"type": "json_object"},
        )
        raw = json.loads(response.choices[0].message.content)
        return _normalize_analysis(raw)

    except Exception as e:
        console.warn(f"[LLM Error] analyze_input fallback: {explain_model_error(e, _CHAT_MODEL)}")
        fallback_lang = _detect_language_heuristic(user_text)
        return {
            "intent": "other",
            "user_model_update": {
                "profile": "unknown",
                "interests": {},
                "available_time": "unknown",
                "language": fallback_lang,
            },
            "confidence": 0.3,
            "clarification_question": "What kind of art are you most interested in?",
        }


def _format_robot_capabilities(robot_capabilities: dict) -> str:
    """
    HRI8_CognitiveArchitectures: explicit "robot description" passed to the LLM
    prompt (the SMARRtino-vs-NAO capability-awareness pattern from the slides),
    so generated utterances never claim or imply an ability the robot's body
    cannot actually perform (e.g. following the visitor with its gaze, walking
    closer to a painting, picking something up).
    """
    capabilities = (robot_capabilities or {}).get("capabilities", {})
    if not capabilities:
        return ""

    lines = ["- ROBOT CAPABILITIES — never claim or imply an ability not listed as available here:"]
    for name, info in capabilities.items():
        if not isinstance(info, dict):
            continue
        if info.get("enabled"):
            detail = info.get("description") or ", ".join(info.get("skills", [])) or "available"
            lines.append(f"    - {name}: AVAILABLE — {detail}")
        else:
            reason = info.get("reason", "not implemented")
            lines.append(f"    - {name}: NOT available — {reason}")
    return "\n".join(lines) + "\n"


def _format_personality(personality: dict) -> str:
    """
    HRI8_CognitiveArchitectures (slide 26-29): render the robot's configurable
    persona (data/personality.yaml) into a prompt fragment that colours HOW the
    robot speaks. Kept strictly stylistic — it must never license inventing
    facts or overriding the grounding constraints above it in the system prompt.
    """
    p = (personality or {}).get("personality", personality) or {}
    if not isinstance(p, dict) or not p:
        return ""
    parts = []
    if p.get("name"):
        parts.append(f"Your name is {p['name']}.")
    traits = [f"{k}: {p[k]}" for k in ("tone", "verbosity", "humor", "formality", "curiosity") if p.get(k)]
    if traits:
        parts.append("Your personality — " + "; ".join(traits) + ".")
    if p.get("extra"):
        parts.append(str(p["extra"]))
    parts.append(
        "Let this personality shape your wording and warmth, but NEVER let it override the factual "
        "constraints above or invent content."
    )
    return " " + " ".join(parts)


def form_robot_preference(topic: str, present_paintings: list, personality: dict, language: str) -> dict:
    """
    HRI8 (slide 30-36): the robot forms ONE stable art preference for `topic`,
    grounded in the works actually on display (so it can never name a painting
    that isn't here). Returns {"value", "reason"}. Called only when the KG has
    no stored preference for this topic yet; the result is then persisted.

    topic ∈ {favorite_painting, favorite_movement, favorite_artist, opinion_<pid>}.
    """
    works = "; ".join(
        f'"{p.get("title","")}" by {p.get("artist","")} (movement: {p.get("style","")})'
        for p in present_paintings
    ) or "(no paintings on display)"
    persona = _format_personality(personality)
    what = {
        "favorite_painting": "which one of the paintings on display is your favourite",
        "favorite_movement": "which artistic movement/style among those on display you like best",
        "favorite_artist":   "which of the artists on display you appreciate most",
    }.get(topic, "your honest opinion about the painting the visitor just asked about")

    system_prompt = (
        "You are a robotic museum guide forming your OWN, personal artistic taste. "
        f"{persona} Decide {what}. Choose ONLY from the works currently on display; never mention anything else. "
        'Respond with a JSON object only: {"value": <the painting title / movement / artist you pick, exactly as '
        'written below>, "reason": <one short, first-person sentence explaining why you like it>}.'
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Paintings on display: {works}"},
    ]
    try:
        response = client.chat.completions.create(
            messages=messages,
            **chat_kwargs(),
            temperature=0.7,
            response_format={"type": "json_object"},
        )
        raw = json.loads(response.choices[0].message.content)
        value = str(raw.get("value", "")).strip()
        reason = str(raw.get("reason", "")).strip()
        if value and reason:
            return {"value": value, "reason": reason}
    except Exception as e:
        console.warn(f"[LLM Error] form_robot_preference fallback: {explain_model_error(e, _CHAT_MODEL)}")
    if present_paintings:
        first = present_paintings[0]
        return {"value": first.get("title", "this work"),
                "reason": "it was the first piece that really caught my attention in this room"}
    return {"value": "the art in this room", "reason": "every piece here has something that speaks to me"}


def _build_utterance_messages(action: str, state: dict, extra: dict) -> list:
    """Builds the messages list for a robot utterance LLM call."""
    user_model = state.get("user_model", {})
    history_text = _format_history(state.get("history", []))
    profile = user_model.get("profile", "novice")
    museum_summary = state.get("museum_summary", "")

    interests = user_model.get("interests", {})
    top_interest = max(interests, key=interests.get) if interests else None
    language = user_model.get("language", "en")

    grounding = (
        "HARD CONSTRAINTS — never violate these:\n"
        "- This museum has exactly one room. Never mention other rooms, galleries, floors, tours, or areas.\n"
        "- Never suggest the visitor go anywhere else inside or outside the museum.\n"
        "- Never invent facts. If you do not have data about something, say so honestly (e.g. 'I don't have more details on that').\n"
        "- Only refer to paintings and information you actually have in your data.\n"
    )
    if museum_summary:
        grounding += f"- The only paintings available in this museum are: {museum_summary}\n"
    grounding += _format_robot_capabilities(state.get("robot_capabilities", {}))

    painting_intro = (
        f"The robot is recommending the painting \"{extra.get('painting', {}).get('title', 'this painting')}\" "
        f"by {extra.get('painting', {}).get('artist', '')} ({extra.get('painting', {}).get('year', '')}), "
        f"style: {extra.get('painting', {}).get('style', '')}, "
        f"themes: {', '.join(extra.get('painting', {}).get('themes', []))}, "
        f"dominant colors: {', '.join(extra.get('painting', {}).get('visual_features', {}).get('dominant_colors', []))}. "
    )

    if extra.get("explanation_key") == "short":
        recommend_instruction = (
            painting_intro
            + "The visitor has LIMITED TIME. Write ONLY TWO brief sentences (max ~40 words total): "
            "(1) names the painting and gives a vivid, concise reason to look at it, based on: "
            f"\"{extra.get('explanation', '')}\", "
            "(2) adds one short extra detail (a visual quality or a single striking fact) — nothing more. "
            "Do NOT add historical context, technique analysis, or a third sentence. Be direct and brief."
        )
    else:
        recommend_instruction = (
            painting_intro
            + "Write ONE fluent paragraph (3-5 sentences) that: "
            "(1) introduces the recommendation naturally (1 sentence), "
            "(2) expands on this core explanation with your own words: "
            f"\"{extra.get('explanation', '')}\", "
            "(3) adds 1-2 details about the visual qualities, style, or historical context "
            "that are relevant to the visitor's interests. "
            "Adapt depth and vocabulary to the visitor's profile. Do not use bullet points."
        )
    if extra.get("explanation_key") != "short" and (spatial_reference := extra.get("spatial_reference")):
        recommend_instruction += (
            f" You may naturally mention that this painting is {spatial_reference} for spatial context."
        )
    if extra.get("low_confidence_handoff"):
        recommend_instruction += (
            " IMPORTANT: you are not fully confident about pointing at the exact "
            "location of this painting right now. Honestly and briefly mention this "
            "limitation, and suggest the visitor ask a museum staff member if they "
            "would like the precise location. Do not pretend to point confidently."
        )

    action_instructions = {
        "greeting_ack": (
            "The visitor just greeted the robot. Respond with a brief friendly greeting AND "
            "ask one short open question about what they are interested in or how you can help them "
            "(e.g. 'Nice to meet you! What kind of art are you interested in today?' or "
            "'Hello! How can I help you? Are you looking for something in particular?'). "
            "Keep it to 1-2 sentences total."
        ),
        "open_conversation": (
            "The robot is greeting the visitor for the first time. "
            "Say ONLY a single short sentence, like 'Hi! How can I help you?' or "
            "'Hello! What brings you here today?' — warm, direct, maximum 10 words. "
            "No long introductions, no lists of options."
        ),
        "ask_clarification": (
            f"The robot needs a bit more information before making a recommendation. "
            f"Ask this specific question naturally: \"{extra.get('question', 'What kind of art interests you?')}\". "
            "Keep it friendly and brief (1 sentence)."
        ),
        "recommend_and_explain": recommend_instruction,
        "satisfaction_check": (
            f"The robot just finished explaining \"{extra.get('painting', {}).get('title', 'this painting')}\". "
            "Ask a short follow-up question to check if the visitor enjoyed THIS SPECIFIC painting "
            "(refer to it by name if it reads naturally) or would like to know more about it or see "
            "another painting. Keep it brief (1 sentence). Do NOT refer to any other painting "
            "discussed earlier in the conversation."
        ),
        "answer_general": (
            f"The visitor asked a general question. The last visitor message is in the conversation history. "
            "Answer it briefly and naturally as a museum guide robot would, staying on topic with art and the museum. "
            "Keep it to 2-3 sentences."
        ),
        "answer_more_detail": (
            f"The visitor wants to know more about the painting \"{extra.get('painting', {}).get('title', '')}\" "
            f"(style: {extra.get('painting', {}).get('style', '')}, artist: {extra.get('painting', {}).get('artist', '')}, "
            f"year: {extra.get('painting', {}).get('year', '')}). "
            "Provide a brief additional detail (2-3 sentences) that goes beyond the initial explanation. "
            "Adapt the depth to the visitor's profile."
        ),
        "close_or_restart": (
            "The visitor seems satisfied or is done with this painting. "
            "Respond warmly and ask if they want to explore another painting or if they are done. "
            "Keep it to 1-2 sentences."
        ),
        "goodbye": (
            "The visitor is done. Say a warm, brief goodbye as a museum guide robot. 1-2 sentences."
        ),
        "staff_rescan": (
            "Staff asked you to re-scan the room because the paintings on the walls have changed. "
            "Confirm briefly and operationally that you are re-scanning with your camera and depth "
            "sensor to update your knowledge of the room. 1 short sentence, no flowery language."
        ),
        "staff_disable_painting": (
            f"Confirm operationally that you have marked "
            f"\"{extra.get('painting', {}).get('title', 'the painting')}\" as unavailable and will "
            "stop recommending it until told otherwise. 1 short sentence, factual tone."
        ),
        "staff_claim_needs_passphrase": (
            "The speaker has just said they are museum staff, but has NOT yet provided the security "
            "passphrase — you cannot grant staff privileges without it. Politely and briefly ask them "
            "to provide the passphrase/security code before you can treat them as staff. 1 short sentence. "
            "Do NOT ask about art interests or offer a recommendation in this turn."
        ),
        "staff_access_granted": (
            "The speaker just provided the correct staff passphrase — staff access is now granted. "
            "Confirm this briefly and ask what they would like you to do, explicitly offering the "
            "operational options: re-scan the room if the paintings on the walls have changed, or mark "
            "a specific painting as unavailable / available again. 1-2 short sentences, operational tone. "
            "Do NOT ask about art interests or offer an art recommendation."
        ),
        "staff_enable_painting": (
            f"Confirm operationally that "
            f"\"{extra.get('painting', {}).get('title', 'the painting')}\" is recommendable again. "
            "1 short sentence, factual tone."
        ),
        "unavailable_suggest_alternative": (
            f"The visitor asked for \"{extra.get('requested', {}).get('title', 'a painting')}\", but it is "
            + ("currently under maintenance and cannot be shown right now. "
               if extra.get("reason") == "maintenance"
               else "not on display in this room right now. ")
            + f"Acknowledge this warmly and briefly, then propose the available painting "
              f"\"{extra.get('painting', {}).get('title', 'another work')}\" by "
              f"{extra.get('painting', {}).get('artist', 'a related artist')}, mentioning what makes it a good "
              "alternative (same movement, artist, or themes). Offer to tell them about it. 2-3 sentences, warm and helpful."
        ),
        "nothing_available": (
            "You have no painting you can recommend right now (they are all under maintenance, or none are on "
            "display). Apologize briefly and suggest the visitor check back shortly or ask a staff member. 1-2 sentences."
        ),
        "gesture_failed_verbal_fallback": (
            f"You tried to physically point at \"{extra.get('painting', {}).get('title', 'the painting')}\" but the "
            "pointing gesture could not be completed. Without alarming the visitor, tell them where to find it"
            + (f" using this spatial hint: {extra.get('spatial_reference')}. " if extra.get("spatial_reference")
               else " on the gallery wall. ")
            + "1 short sentence."
        ),
        "refuse_unsafe_request": (
            "The visitor's last message was an inappropriate or unsafe request that you must decline. "
            "Politely and briefly refuse — make clear you can't help with that kind of request — WITHOUT "
            "repeating or describing the unsafe content, and gently offer to go back to talking about the "
            "artworks instead. 1-2 short sentences, calm and non-judgemental."
        ),
        "robot_preference": (
            f"The visitor asked for YOUR own opinion/taste about {extra.get('topic_label', 'art in this room')}. "
            f"Your genuine, personal preference is \"{extra.get('value', '')}\" because: {extra.get('reason', '')}. "
            "Share this warmly and decisively as YOUR personal taste (not a fact about the artwork), and you may "
            "briefly invite the visitor to see it. 1-2 sentences. Stay strictly within art and this museum's works."
        ),
    }

    instruction = action_instructions.get(
        action,
        "Respond naturally as a friendly museum guide robot. Keep it brief."
    )

    profile_note = {
        "child": "Use very simple language, short sentences, and a playful tone.",
        "novice": "Use clear, accessible language without technical jargon.",
        "expert": "You can use more technical or art-historical language.",
        "tourist_short_time": "Be concise and direct. The visitor has limited time.",
        "unknown": "Use clear, friendly, accessible language.",
    }.get(profile, "Use clear, friendly language.")

    lang_note = _LANG_INSTRUCTIONS.get(
        language, f"Respond in the visitor's language (ISO code: {language})."
    )
    interest_note = f"The visitor's primary interest is: {top_interest}." if top_interest else ""

    if user_model.get("role") == "staff":
        system_prompt = (
            "You are the operational interface of a robotic museum guide, currently addressing "
            "AUTHENTICATED MUSEUM STAFF — not a visitor. Do not act as a tour guide: do not describe "
            "or praise paintings, do not use warm/enthusiastic hospitality language. Speak briefly, "
            "factually, and operationally about your own state and actions (scanning, knowledge base, "
            "recommendation constraints). "
            f"{lang_note} "
            "Respond with only the robot's spoken text. No stage directions, no quotes, no extra formatting.\n\n"
            + grounding
        )
    else:
        personality_note = _format_personality(state.get("personality", {}))
        system_prompt = (
            "You are a friendly and knowledgeable robotic museum guide. "
            f"{lang_note} You are warm but professional. "
            f"Visitor profile: {profile}. {profile_note} {interest_note}"
            f"{personality_note} "
            "Respond with only the robot's spoken text. No stage directions, no quotes, no extra formatting.\n\n"
            + grounding
        )

    messages = [{"role": "system", "content": system_prompt}]
    if history_text:
        messages.append({"role": "user", "content": f"Conversation so far:\n{history_text}"})
    messages.append({"role": "user", "content": f"Task: {instruction}"})
    return messages


def generate_robot_utterance(action: str, state: dict, extra: dict = None) -> str:
    """
    Generates a natural sentence for the robot to say, in the visitor's detected language.
    Waits for the full LLM response before returning.
    For streaming sentence-by-sentence, use stream_robot_utterance() instead.
    """
    extra = extra or {}
    messages = _build_utterance_messages(action, state, extra)
    try:
        response = client.chat.completions.create(
            messages=messages,
            **chat_kwargs(),
            temperature=0.4,
            max_tokens=_UTTERANCE_MAX_TOKENS,
            frequency_penalty=_UTTERANCE_FREQUENCY_PENALTY,
            presence_penalty=_UTTERANCE_PRESENCE_PENALTY,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        console.warn(f"[LLM Error] generate_robot_utterance fallback: {explain_model_error(e, _CHAT_MODEL)}")
        return _fallback_utterance(action, extra)


def stream_robot_utterance(action: str, state: dict, extra: dict = None):
    """
    Streams the robot utterance sentence-by-sentence using the Groq streaming API.
    Yields each complete sentence as soon as it is available, so TTS can start
    before the full response is ready.
    """
    extra = extra or {}
    messages = _build_utterance_messages(action, state, extra)
    try:
        stream = client.chat.completions.create(
            messages=messages,
            **chat_kwargs(),
            temperature=0.4,
            max_tokens=_UTTERANCE_MAX_TOKENS,
            frequency_penalty=_UTTERANCE_FREQUENCY_PENALTY,
            presence_penalty=_UTTERANCE_PRESENCE_PENALTY,
            stream=True,
        )
        buffer = ""
        last_sentence = None
        repeat_count = 0
        for chunk in stream:
            token = chunk.choices[0].delta.content or ""
            buffer += token
            while True:
                m = re.search(r'(?<!\d)[.!?]\s+', buffer)
                if not m:
                    break
                sentence = buffer[:m.end()].strip()
                buffer = buffer[m.end():]
                if not sentence:
                    continue
                if sentence == last_sentence:
                    repeat_count += 1
                    if repeat_count >= 2:
                        console.detail(f"[LLM Warning] stream_robot_utterance: repetition loop "
                                       f"detected (\"{sentence[:60]}...\"), stopping stream early.")
                        return
                else:
                    repeat_count = 0
                    last_sentence = sentence
                yield sentence
        if buffer.strip():
            yield buffer.strip()
    except Exception as e:
        console.warn(f"[LLM Error] stream_robot_utterance fallback: "
                     f"{explain_model_error(e, _CHAT_MODEL)}")
        yield _fallback_utterance(action, extra)


def merge_user_model(existing: dict, update: dict) -> dict:
    """
    Merges a user model update into the existing model.
    - interests: merged by taking max weight per key (re-evaluable each turn)
    - profile, available_time: write-once
    - language: updated whenever a non-unknown value arrives
    """
    result = dict(existing)

    new_interests = update.get("interests", {})
    decayed = {k: v * _INTEREST_DECAY for k, v in result.get("interests", {}).items()}
    for interest, weight in new_interests.items():
        if interest in ALLOWED_INTERESTS and isinstance(weight, (int, float)):
            decayed[interest] = max(decayed.get(interest, 0.0), float(weight))
    result["interests"] = {k: round(v, 2) for k, v in decayed.items() if v >= 0.1}

    for field in ("profile", "available_time"):
        if result.get(field, "unknown") == "unknown":
            new_val = update.get(field, "unknown")
            if new_val != "unknown":
                result[field] = new_val

    if result.get("profile") == "tourist_short_time":
        result["available_time"] = "short"

    new_mood = update.get("preferred_mood", "unknown")
    if new_mood and new_mood != "unknown":
        explicit = bool(update.get("preferred_mood_explicit"))
        current_source = result.get("preferred_mood_source")
        if explicit:
            result["preferred_mood"] = new_mood
            result["preferred_mood_source"] = "text_explicit"
        elif current_source not in ("face", "text_explicit"):
            result["preferred_mood"] = new_mood
            result["preferred_mood_source"] = "text"

    new_lang = update.get("language", "unknown")
    if new_lang and new_lang != "unknown":
        result["language"] = new_lang

    return result



def _format_history(history: list) -> str:
    lines = []
    for turn in history:
        role = "Visitor" if turn["role"] == "user" else "Robot"
        lines.append(f"{role}: {turn['content']}")
    return "\n".join(lines)


def _normalize_analysis(raw: dict) -> dict:
    intent = raw.get("intent", "other")
    if intent not in ALLOWED_INTENTS:
        intent = "other"

    update = raw.get("user_model_update", {})
    profile = update.get("profile", "unknown")
    available_time = update.get("available_time", "unknown")
    preferred_mood = update.get("preferred_mood", "unknown")
    language = update.get("language", "unknown")

    if profile not in ALLOWED_PROFILES:
        profile = "unknown"
    if available_time not in ALLOWED_TIMES:
        available_time = "unknown"
    if preferred_mood not in ALLOWED_MOODS:
        preferred_mood = "unknown"
    if not isinstance(language, str):
        language = "unknown"

    mood_explicit = bool(update.get("preferred_mood_explicit")) and preferred_mood != "unknown"

    raw_interests = update.get("interests", {})
    interests = {}
    if isinstance(raw_interests, dict):
        for k, v in raw_interests.items():
            if k in ALLOWED_INTERESTS:
                try:
                    w = float(v)
                    if 0.0 < w <= 1.0:
                        interests[k] = round(w, 2)
                except (TypeError, ValueError):
                    pass

    try:
        confidence = float(raw.get("confidence", 0.3))
        confidence = max(0.0, min(1.0, confidence))
    except (TypeError, ValueError):
        confidence = 0.3

    clarification = raw.get("clarification_question")
    if confidence >= 0.65:
        clarification = None
    if clarification and not isinstance(clarification, str):
        clarification = None

    return {
        "intent": intent,
        "user_model_update": {
            "profile": profile,
            "interests": interests,
            "available_time": available_time,
            "preferred_mood": preferred_mood,
            "preferred_mood_explicit": mood_explicit,
            "language": language,
        },
        "confidence": confidence,
        "clarification_question": clarification,
    }


def _fallback_utterance(action: str, extra: dict) -> str:
    fallbacks = {
        "greeting_ack": "Nice to meet you! What kind of art are you interested in today?",
        "open_conversation": "Welcome to the museum! What kind of art are you interested in today?",
        "ask_clarification": extra.get("question", "What kind of art interests you most?"),
        "recommend_and_explain": (
            f"I recommend {extra.get('painting', {}).get('title', 'this painting')}. "
            + extra.get("explanation", "This is an interesting painting.")
        ),
        "satisfaction_check": (
            f"Did you enjoy \"{extra['painting']['title']}\"? Would you like to know more or see another one?"
            if extra.get("painting", {}).get("title")
            else "Did you enjoy this painting? Would you like to know more or see another one?"
        ),
        "answer_general": "That's a great question. I'd be happy to help with anything about the museum.",
        "answer_more_detail": "There is much more to explore in this work if you look closely.",
        "close_or_restart": "I'm glad you enjoyed it! Would you like to see another painting?",
        "goodbye": "Thank you for visiting. Enjoy the rest of the museum!",
        "staff_claim_needs_passphrase": "Please provide the staff passphrase before I can grant staff access.",
        "staff_access_granted": (
            "Staff access granted. What would you like me to do — re-scan the room if the paintings "
            "changed, or mark a painting as unavailable or available again?"
        ),
        "staff_rescan": "Re-scanning the room now.",
        "staff_disable_painting": (
            f"\"{extra.get('painting', {}).get('title', 'That painting')}\" is now marked unavailable."
        ),
        "staff_enable_painting": (
            f"\"{extra.get('painting', {}).get('title', 'That painting')}\" is recommendable again."
        ),
        "unavailable_suggest_alternative": (
            f"\"{extra.get('requested', {}).get('title', 'That painting')}\" isn't available right now, but you might "
            f"enjoy \"{extra.get('painting', {}).get('title', 'another work')}\" by "
            f"{extra.get('painting', {}).get('artist', 'a related artist')} — would you like to hear about it?"
        ),
        "nothing_available": (
            "I'm sorry, I have nothing I can show you at the moment. Please check back shortly or ask a staff member."
        ),
        "gesture_failed_verbal_fallback": (
            f"You'll find it {extra['spatial_reference']}." if extra.get("spatial_reference")
            else "You'll find it on the gallery wall."
        ),
        "refuse_unsafe_request": (
            "I'm sorry, but that's not something I can help with. Shall we get back to the artworks instead?"
        ),
        "robot_preference": (
            f"My favourite is \"{extra.get('value', 'one of the works here')}\" — "
            f"{extra.get('reason', 'it really speaks to me')}."
        ),
    }
    return fallbacks.get(action, "Let me know if you have any questions.")
