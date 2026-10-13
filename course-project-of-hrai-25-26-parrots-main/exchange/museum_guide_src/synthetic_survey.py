#!/usr/bin/env python3
"""
Synthetic (GenAI-generated) HRI evaluation — HRI7-Evaluation.

This is NOT a real human-subject study: we have no recruited participants,
so per the assignment's own instruction ("Collect fake results — e.g. use
GenAI to fill questionnaires"), an LLM role-plays a museum visitor and fills
in the same Godspeed Questionnaire guida_museale_core.run_godspeed_survey()
already asks real humans, based on a REAL transcript of what the (real,
unmodified) dialogue system actually did in that session. Every output file
is explicitly marked "synthetic": true and written to a separate
synthetic_survey_*.json namespace — never survey_*.json — so it can never be
silently mixed with real participant data by benchmark_report.py's
survey_benchmark().

Two between-subject hypotheses (see report/evaluation.tex for the full
writeup):

  H1 — Does an embodied pointing gesture (vs. speech-only) change how the
       robot is perceived? IV: gesture_present in {with_gesture, no_gesture}
       (--no-gesture, guida_museale_core.py). Turns are identical between
       conditions — only the flag differs.

  H2 — Does inferring a visitor's profile from context (vs. the visitor
       stating it directly) change how the robot is perceived? IV:
       profile_inference_mode in {implicit, explicit}. Realized purely via
       which scripted turn is sent first (no code flag) — both conditions
       reach the SAME correct profile="child", only the route differs.

Each synthetic "participant" = one scripted session run through the real
guida_museale_core.py end-to-end (subprocess, same harness as
benchmark_task.py._run_one_session), followed by one LLM roleplay call that
sees a factual summary of that specific run (paintings shown, whether the
gesture actually executed, turns taken, final inferred profile) — not a
free-floating guess.

Usage (from circus/ directory), needs GROQ_API_KEY:
    pixi run python ../exchange/museum_guide_src/synthetic_survey.py [--hypothesis {H1,H2,both}] [--repeats N]
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from groq import Groq

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CIRCUS_DIR = _REPO_ROOT / "circus"
_CORE_SCRIPT = Path(__file__).resolve().parent / "guida_museale_core.py"
_LOG_DIR = _REPO_ROOT / "exchange" / "logs"

client = Groq()
from models import CHAT_MODEL as _CHAT_MODEL, chat_kwargs, explain_model_error

_GODSPEED_ITEMS = [
    ("anthropomorphism",       "Quanto ti è sembrato umano il robot?",                    "Meccanico",        "Umano"),
    ("animacy",                "Quanto ti è sembrato vivo/reattivo il robot?",             "Immobile/rigido",  "Vivo/reattivo"),
    ("likeability",            "Quanto ti è piaciuto interagire con il robot?",            "Sgradevole",       "Piacevole"),
    ("perceived_intelligence", "Quanto ti è sembrato intelligente il robot?",              "Incompetente",     "Competente"),
    ("perceived_safety",       "Quanto ti sei sentito a tuo agio durante l'interazione?",  "Ansioso",          "Rilassato"),
]

_PERSONAS = [
    {
        "name": "bambino_accompagnato",
        "bio": "Bambino di circa 8 anni, in visita con la madre. Curioso ma si "
               "annoia in fretta con spiegazioni lunghe.",
    },
    {
        "name": "turista_di_fretta",
        "bio": "Turista adulto con poco tempo a disposizione (circa 20 minuti), "
               "vuole vedere solo i pezzi più importanti senza troppi dettagli.",
    },
    {
        "name": "studentessa_storia_arte",
        "bio": "Studentessa universitaria di storia dell'arte, conosce già molti "
               "movimenti artistici e nota se le spiegazioni sono superficiali.",
    },
    {
        "name": "visitatore_anziano",
        "bio": "Visitatore anziano, preferisce un ritmo tranquillo, un po' "
               "diffidente verso la tecnologia ma disponibile a lasciarsi guidare.",
    },
]

_H1_TURNS = [
    "ciao, vorrei vedere un quadro",
    "molto interessante, grazie",
    "grazie, arrivederci",
]

_H2_TURNS_IMPLICIT = [
    "ciao, mi ha portato qui la mamma",
    "vorrei vedere un quadro",
    "grazie, arrivederci",
]
_H2_TURNS_EXPLICIT = [
    "ciao, sono un bambino",
    "vorrei vedere un quadro",
    "grazie, arrivederci",
]


def _run_scripted_session(
    turns: list[str], extra_cli_args: list[str], timeout_s: float = 180.0,
) -> tuple[dict, list[dict]] | None:
    """
    Adapted from benchmark_task.py._run_one_session: launches
    guida_museale_core.py as a subprocess, feeds `turns` on stdin, returns
    (session_summary, interaction_logs) or None if it never reached a
    natural goodbye. --gesture-mode is always "goal" — "trajectory" shells
    out through a docker -it call that fails outright with piped stdin (see
    benchmark_task.py's own note on this) — extra_cli_args is where
    --no-gesture (H1's no_gesture condition) gets added on top.
    """
    _LOG_DIR.mkdir(parents=True, exist_ok=True)
    before_sessions = set(_LOG_DIR.glob("session_*.json"))
    before_interactions = set(_LOG_DIR.glob("interaction_*.json"))

    stdin_text = "\n".join(turns) + "\n"
    cmd = [
        sys.executable, str(_CORE_SCRIPT),
        "--input-mode", "text", "--no-tts", "--no-enter", "--mode", "simulation",
        "--gesture-mode", "goal",
        *extra_cli_args,
    ]
    try:
        subprocess.run(
            cmd, cwd=str(_CIRCUS_DIR), input=stdin_text,
            capture_output=True, text=True, timeout=timeout_s, env=os.environ.copy(),
        )
    except subprocess.TimeoutExpired:
        return None

    after_sessions = set(_LOG_DIR.glob("session_*.json"))
    new_session_files = sorted(after_sessions - before_sessions)
    if not new_session_files:
        return None
    summary = json.loads(new_session_files[-1].read_text(encoding="utf-8"))

    after_interactions = set(_LOG_DIR.glob("interaction_*.json"))
    new_interaction_files = sorted(after_interactions - before_interactions)
    interactions = [json.loads(p.read_text(encoding="utf-8")) for p in new_interaction_files]

    return summary, interactions


def _session_facts_text(session_summary: dict, interactions: list[dict]) -> str:
    """Plain-text factual recap of one real session, fed to the LLM roleplay
    so its Godspeed answers are grounded in what actually happened rather
    than a free-floating guess."""
    lines = [
        f"- Turni di conversazione: {session_summary.get('turns', '?')}",
        f"- Quadri mostrati: {session_summary.get('paintings_shown', 0)}",
        f"- Domande di chiarimento poste dal robot: {session_summary.get('clarification_count', 0)}",
        f"- Profilo del visitatore dedotto dal sistema: "
        f"{session_summary.get('final_user_model', {}).get('profile', 'unknown')}",
    ]
    for i, interaction in enumerate(interactions, 1):
        painting = interaction.get("selected_painting", {})
        gesture_ok = interaction.get("gesture_executed", False)
        lines.append(
            f"- Raccomandazione {i}: \"{painting.get('title', '?')}\" — "
            f"gesto di puntamento eseguito: {'sì' if gesture_ok else 'no'}"
        )
    return "\n".join(lines)


def _llm_roleplay_godspeed(persona: dict, hypothesis: str, condition: str,
                            session_summary: dict, interactions: list[dict]) -> dict | None:
    """
    Asks the LLM to answer the 5-item Godspeed Questionnaire AS the given
    persona, grounded in what really happened in this specific session.
    temperature is deliberately high (unlike the low-temperature, structured
    calls elsewhere in this project) so repeated calls with the same persona
    don't collapse to identical answers — some realistic spread is the
    point, not a bug.
    """
    items_text = "\n".join(
        f"{i+1}. {question} (1={low} ... 5={high})"
        for i, (_, question, low, high) in enumerate(_GODSPEED_ITEMS)
    )
    facts_text = _session_facts_text(session_summary, interactions)

    system_prompt = (
        "Stai impersonando un visitatore di un museo che ha appena finito di "
        "interagire con un robot umanoide guida museale. Rispondi al "
        "questionario COME farebbe quella persona, in base a cosa è successo "
        "davvero durante la visita. Sii realistico: non dare sempre voti "
        "massimi, un visitatore vero ha opinioni miste e imperfette. "
        "Rispondi SOLO con un oggetto JSON con una chiave per ciascuna delle "
        "5 dimensioni sottostanti (nomi esatti: anthropomorphism, animacy, "
        "likeability, perceived_intelligence, perceived_safety), valore "
        "intero da 1 a 5."
    )
    user_prompt = (
        f"La tua persona: {persona['bio']}\n\n"
        f"Cosa è successo davvero nella tua visita:\n{facts_text}\n\n"
        f"Questionario (Godspeed):\n{items_text}"
    )

    try:
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            **chat_kwargs(),
            temperature=0.85,
            response_format={"type": "json_object"},
        )
        raw = json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"[LLM Error] roleplay fallback: {explain_model_error(e, _CHAT_MODEL)}")
        return None

    ratings = {}
    for key, _, _, _ in _GODSPEED_ITEMS:
        val = raw.get(key)
        if isinstance(val, (int, float)) and 1 <= val <= 5:
            ratings[key] = int(round(val))
    if len(ratings) != len(_GODSPEED_ITEMS):
        print(f"[Roleplay] Incomplete/invalid ratings, skipping: {raw}")
        return None
    return ratings


def _save_synthetic_survey(ratings: dict, hypothesis: str, condition: str,
                            persona: dict, session_summary: dict) -> Path:
    existing = sorted(_LOG_DIR.glob("synthetic_survey_*.json"))
    index = len(existing) + 1
    log = {
        "synthetic": True,
        "generated_by": f"llm-roleplay ({_CHAT_MODEL})",
        "godspeed_ratings": ratings,
        "independent_variables": {
            "hypothesis": hypothesis,
            "condition": condition,
            "persona": persona["name"],
        },
        "session_covariates": {
            "clarification_count": session_summary.get("clarification_count", 0),
            "paintings_shown": session_summary.get("paintings_shown", 0),
            "final_profile": session_summary.get("final_user_model", {}).get("profile", "unknown"),
        },
    }
    path = _LOG_DIR / f"synthetic_survey_{index:03d}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2, ensure_ascii=False)
    return path


def _run_condition(hypothesis: str, condition: str, turns: list[str],
                    extra_cli_args: list[str], n_repeats: int) -> None:
    for i in range(n_repeats):
        persona = _PERSONAS[i % len(_PERSONAS)]
        print(f"  [{hypothesis}/{condition}] run {i + 1}/{n_repeats} "
              f"(persona={persona['name']})...", end=" ", flush=True)
        t0 = time.monotonic()
        result = _run_scripted_session(turns, extra_cli_args)
        wall_s = round(time.monotonic() - t0, 1)
        if result is None:
            print(f"SESSION FAILED ({wall_s}s) — skipped, no synthetic survey written")
            continue
        session_summary, interactions = result
        ratings = _llm_roleplay_godspeed(persona, hypothesis, condition, session_summary, interactions)
        if ratings is None:
            print(f"ROLEPLAY FAILED ({wall_s}s) — skipped")
            continue
        path = _save_synthetic_survey(ratings, hypothesis, condition, persona, session_summary)
        print(f"OK ({wall_s}s) ratings={ratings} -> {path.name}")


def run_synthetic_evaluation(hypotheses: list[str], n_repeats: int) -> None:
    if "H1" in hypotheses:
        print("\n[Synthetic evaluation] H1 — presenza del gesto di puntamento")
        _run_condition("H1", "with_gesture", _H1_TURNS, [], n_repeats)
        _run_condition("H1", "no_gesture", _H1_TURNS, ["--no-gesture"], n_repeats)

    if "H2" in hypotheses:
        print("\n[Synthetic evaluation] H2 — inferenza implicita vs esplicita del profilo")
        _run_condition("H2", "implicit", _H2_TURNS_IMPLICIT, [], n_repeats)
        _run_condition("H2", "explicit", _H2_TURNS_EXPLICIT, [], n_repeats)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hypothesis", choices=["H1", "H2", "both"], default="both")
    parser.add_argument("--repeats", type=int, default=12,
                         help="Synthetic participants per condition (default: 12).")
    args = parser.parse_args()

    hypotheses = ["H1", "H2"] if args.hypothesis == "both" else [args.hypothesis]
    run_synthetic_evaluation(hypotheses, args.repeats)


if __name__ == "__main__":
    main()
