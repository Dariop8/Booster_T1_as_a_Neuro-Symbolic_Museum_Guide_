#!/usr/bin/env python3

import json
import os
import queue
import re
import subprocess
import tempfile
import threading
import time
import concurrent.futures
import yaml
import argparse
from datetime import datetime
from pathlib import Path

import speech_recognition as sr
from groq import Groq
try:
    from gtts import gTTS
except ModuleNotFoundError:
    gTTS = None

from llm_interpreter import analyze_input, generate_robot_utterance, stream_robot_utterance, merge_user_model, _detect_language_heuristic, form_robot_preference
from robot_executor import run_robot_gesture, run_robot_point_at, run_neutral_pose, prepare_gesture_backend
from dialogue_manager import initialize_state, get_next_action
from social_supervisor import check_social_safety, is_enabled as social_supervisor_enabled
from viewer_server import start_viewer_server
from shm_camera_reader import read_latest_pose
from camera_projection import to_base_frame
from models import TRANSCRIPTION_MODEL
import console
import knowledge_graph as kg



def find_repo_root() -> Path:
    current = Path(__file__).resolve()

    for parent in [current.parent] + list(current.parents):
        if (parent / "exchange").exists() and (parent / "circus").exists():
            return parent

    raise RuntimeError("Could not find repository root containing exchange/ and circus/.")


REPO_ROOT = find_repo_root()
EXCHANGE_DIR = REPO_ROOT / "exchange"
DATA_DIR = EXCHANGE_DIR / "data"
CONFIG_DIR = EXCHANGE_DIR / "config"
LOG_DIR = EXCHANGE_DIR / "logs"
ROBOT_PREFERENCES_PATH = DATA_DIR / "robot_preferences.json"
VIEWER_DIR = Path(__file__).resolve().parent / "viewer"
SNAPSHOT_PATH = VIEWER_DIR / "kg_snapshot.json"
VIEWER_PORT = int(os.environ.get("KG_VIEWER_PORT", "8009"))

LOG_DIR.mkdir(parents=True, exist_ok=True)

client = Groq()

_KB: dict | None = None

_STAFF_PASSPHRASE = "1234"

_STAFF_CLAIM_KEYWORDS = ("staff", "personale del museo", "personnel", "mitarbeiter")

_PIPELINE_TRACE: list[dict] = []
_TRACE_MAXLEN = 60


def _trace(stage: str, detail: str) -> None:
    _PIPELINE_TRACE.append({"t": datetime.now().strftime("%H:%M:%S"), "stage": stage, "detail": detail})
    del _PIPELINE_TRACE[: -_TRACE_MAXLEN]
    _export_kg_snapshot()


_VIEWER_IS_PRIMARY: bool = False
_VIEWER_RETRY_AT: float = 0.0
_VIEWER_RETRY_EVERY_S = 10.0

_ACTIVE_USER_MODEL: dict | None = None


def _export_kg_snapshot() -> None:
    global _VIEWER_IS_PRIMARY, _VIEWER_RETRY_AT
    if _KB is None:
        return
    if not _VIEWER_IS_PRIMARY:
        now = time.monotonic()
        if now < _VIEWER_RETRY_AT:
            return
        _VIEWER_RETRY_AT = now + _VIEWER_RETRY_EVERY_S
        _VIEWER_IS_PRIMARY = start_viewer_server(VIEWER_DIR, VIEWER_PORT, quiet_on_conflict=True)
        if not _VIEWER_IS_PRIMARY:
            return
        console.detail("[KG Viewer] Porta acquisita — il viewer ora mostra questa istanza.")
    try:
        kg.export_graph_snapshot(_KB["graph"], SNAPSHOT_PATH, pipeline_trace=_PIPELINE_TRACE,
                                 visitor_model=_ACTIVE_USER_MODEL, paintings=_KB["paintings"],
                                 robot_pose=read_latest_pose())
    except Exception as e:
        console.error(f"[KG Viewer] Errore nell'esportazione dello snapshot: {e}")


def start_kg_viewer_server(port: int = VIEWER_PORT) -> None:
    """
    Serves viewer/ (kg_viewer.html + the live kg_snapshot.json written by
    _export_kg_snapshot) on a background thread, so the knowledge-graph
    visualizer is reachable as soon as the pipeline starts — no separate
    process to launch by hand.
    """
    global _VIEWER_IS_PRIMARY
    if _VIEWER_IS_PRIMARY:
        return
    _VIEWER_IS_PRIMARY = start_viewer_server(VIEWER_DIR, port)


def load_yaml(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"File mancante: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_robot_preferences(graph) -> None:
    """HRI8: populate the KG with the robot's persisted art preferences (if any)."""
    if not ROBOT_PREFERENCES_PATH.exists():
        return
    try:
        with open(ROBOT_PREFERENCES_PATH, "r", encoding="utf-8") as f:
            prefs = json.load(f)
    except Exception as e:
        console.warn(f"[preferences] Impossibile leggere {ROBOT_PREFERENCES_PATH.name}: {e}")
        return
    for pref in prefs:
        if pref.get("topic") and pref.get("value"):
            kg.set_robot_preference(graph, pref["topic"], pref["value"], pref.get("reason", ""))


def _save_robot_preferences(graph) -> None:
    """Persist every current robot preference back to disk (long-term memory)."""
    try:
        with open(ROBOT_PREFERENCES_PATH, "w", encoding="utf-8") as f:
            json.dump(kg.all_robot_preferences(graph), f, indent=2, ensure_ascii=False)
    except Exception as e:
        console.warn(f"[preferences] Impossibile salvare {ROBOT_PREFERENCES_PATH.name}: {e}")


def init_kb() -> None:
    """Load YAML knowledge base files into module-level _KB. Idempotent."""
    global _KB
    if _KB is not None:
        return
    paintings       = load_yaml(DATA_DIR / "paintings.yaml")["paintings"]
    user_profiles   = load_yaml(DATA_DIR / "user_profiles.yaml")["user_profiles"]
    robot_caps      = load_yaml(DATA_DIR / "robot_capabilities.yaml")["robot"]
    personality_path = DATA_DIR / "personality.yaml"
    personality     = load_yaml(personality_path) if personality_path.exists() else {}

    placement, mismatches, placement_source = kg.resolve_placement(DATA_DIR)
    gesture_by_pid = {pid: gesture for gesture, pid in placement.items()}
    xyz_by_gesture = kg._load_scan_xyz(DATA_DIR)
    for p in paintings:
        gesture = gesture_by_pid.get(p["id"])
        if gesture:
            p["robot_actions"] = {"pointing_skill": gesture}
            xyz = xyz_by_gesture.get(gesture)
            if xyz:
                p["robot_actions"]["target_xyz"] = xyz
    present = [p for p in paintings if p["id"] in gesture_by_pid]

    museum_summary = ", ".join(
        f'"{p["title"]}" by {p["artist"]} ({p["year"]})' for p in present
    )

    graph = kg.init_kg(DATA_DIR)

    _load_robot_preferences(graph)

    _KB = {
        "paintings":          paintings,
        "present_paintings":  present,
        "user_profiles":      user_profiles,
        "robot_capabilities": robot_caps,
        "personality":        personality,
        "museum_summary":     museum_summary,
        "graph":              graph,
    }
    console.detail(f"✅ Knowledge base caricata. Catalogo: {len(paintings)} quadri, "
                   f"in sala: {len(present)} ({placement_source})")
    if mismatches:
        console.warn(f"⚠️  Scan/registro in disaccordo su: {sorted(mismatches)}")

    start_kg_viewer_server()

    _trace("init", f"Knowledge graph built ({len(graph)} triples, "
                   f"{len(present)}/{len(paintings)} paintings hung — {placement_source})")
    _export_kg_snapshot()


def reload_kb() -> None:
    """
    HRI6 - KR&Planning / Joint Intention Theory: re-perceive and rebuild
    belief state after staff notifies the robot that the world changed
    (handle_staff_rescan). Reuses init_kb()'s normal load path — safe to call
    mid-session because kg.init_kg()/resolve_placement() read
    discovered_paintings.json fresh from disk each call, the same file
    museum_explorer.save_scan_results() just wrote.
    """
    global _KB
    _KB = None
    init_kb()


def configure(robot_mode: bool = False, mic_device_index: int | None = None,
              silent_tts: bool = False, quiet: bool = False) -> None:
    """Set global runtime configuration (robot/simulation mode, microphone,
    silent debug mode, quiet demo output)."""
    global _ROBOT_MODE, _MIC_DEVICE_INDEX, _SILENT_TTS
    console.set_quiet(quiet)
    _SILENT_TTS = silent_tts
    _ROBOT_MODE = robot_mode and not _SILENT_TTS
    _MIC_DEVICE_INDEX = mic_device_index
    if _ROBOT_MODE and not _init_booster_tts():
        console.warn("⚠️  Impossibile connettersi al robot — uso gTTS come fallback.")
        _ROBOT_MODE = False


def _format_user_model(um: dict) -> str:
    """
    The visitor model as the robot currently believes it — one readable line
    instead of a raw JSON dump. Empty/unknown fields are dropped (they say
    nothing and push the fields that DID change off the side of the terminal),
    and the mood carries its provenance, which is the interesting part when
    the camera and the dialogue disagree. Verbose mode still gets the full
    dict, nothing is hidden there.
    """
    if not console.is_quiet():
        return json.dumps(um)
    bits = []
    for key in ("role", "profile", "available_time", "language"):
        val = um.get(key)
        if val and val != "unknown":
            bits.append(f"{key}={val}")
    interests = um.get("interests") or {}
    if interests:
        top = sorted(interests.items(), key=lambda kv: -kv[1])[:3]
        bits.append("interests=" + ",".join(f"{k}:{v:g}" for k, v in top))
    mood = um.get("preferred_mood")
    if mood and mood != "unknown":
        src = um.get("preferred_mood_source")
        bits.append(f"mood={mood}" + (f"({src})" if src else ""))
    return "  ".join(bits) if bits else "(ancora vuoto)"


def print_banner(input_mode: str, gesture_mode: str) -> None:
    """
    Startup header. Verbose mode dumps the full runtime configuration (the
    paths are what you check first when a run misbehaves); quiet mode keeps
    one title line — on a screen recording the config block is dead space the
    viewer scrolls past before anything happens.
    """
    if console.is_quiet():
        console.event("\n🤖  BOOSTER-T1 — GUIDA MUSEALE"
                      f"   [{input_mode} · {'robot speaker' if _ROBOT_MODE else ('muto' if _SILENT_TTS else 'gTTS')}]\n")
        return
    print("==================================================")
    print("🤖 BOOSTER-T1 - GUIDA MUSEALE ATTIVA")
    print("==================================================")
    print(f"Repo root:  {REPO_ROOT}")
    print(f"Data dir:   {DATA_DIR}")
    print(f"Config:     {CONFIG_DIR}")
    print(f"Input mode: {input_mode}")
    print(f"Gesture mode: {gesture_mode}")
    print(f"Audio mode: {'silent text only' if _SILENT_TTS else ('robot speaker (kSpeak)' if _ROBOT_MODE else 'gTTS (PC speaker)')}")
    print(f"Mood detect: {'camera ' + str(_CAMERA_INDEX) if _CAMERA_INDEX is not None else 'disabilitato'}")
    print(f"Survey (HRI7): {'attivo' if _SURVEY_ENABLED else 'disabilitato'}")
    print("==================================================\n")


def shutdown() -> None:
    """Clean up ROS2 resources. Call once when the application exits."""
    _shutdown_booster_tts()



_MIC_DEVICE_INDEX: int | None = None

_ARECORD_DEVICE    = os.environ.get("MUSEUM_ARECORD_DEVICE", "hw:1,7")
_ARECORD_FALLBACK  = "default"
_arecord_device_ok: str | None = None
_ARECORD_RATE      = 16000
_ARECORD_CHANNELS  = 2
_SPEECH_THRESHOLD = 150
_SILENCE_S        = 2.0
_MAX_RECORD_S     = 30

_WHISPER_HALLUCINATIONS = {
    "", " ", ".",
    "thank you.", "thank you", "thanks.", "thanks",
    "you.", "you",
    "bye.", "bye", "goodbye.", "goodbye",
    "see you.", "see you",
    "subtitles by", "subtitles by the", "amara.org",
}


def _resolve_arecord_device() -> str | None:
    """
    First arecord capture device that actually opens, cached for the process.

    Probed with a one-second capture rather than by parsing `arecord -l`: what
    matters is whether THIS format (16 kHz stereo S16_LE) opens on it, which
    the listing doesn't say — `hw:` devices do no rate/channel conversion, so
    a card that exists can still refuse the format.

    Returns None if arecord itself is missing, or if no candidate opens; the
    caller then falls back to PyAudio as before.
    """
    global _arecord_device_ok
    if _arecord_device_ok is not None:
        return _arecord_device_ok or None

    candidates = [_ARECORD_DEVICE]
    if _ARECORD_FALLBACK not in candidates:
        candidates.append(_ARECORD_FALLBACK)

    for device in candidates:
        try:
            probe = subprocess.run(
                ["arecord", "-D", device, "-f", "S16_LE", "-r", str(_ARECORD_RATE),
                 "-c", str(_ARECORD_CHANNELS), "-d", "1", "-q", "--"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=5,
            )
        except FileNotFoundError:
            _arecord_device_ok = ""
            return None
        except subprocess.TimeoutExpired:
            continue
        if probe.returncode == 0:
            if device != _ARECORD_DEVICE:
                console.detail(f"[Audio] '{_ARECORD_DEVICE}' non disponibile — "
                               f"uso il device di cattura '{device}'.")
            _arecord_device_ok = device
            return device

    console.warn(f"[Audio] Nessun device arecord utilizzabile fra {candidates} — "
                 f"provo con PyAudio.")
    _arecord_device_ok = ""
    return None


def _record_with_arecord() -> bytes | None:
    """
    Registra dal DMIC hw:1,7 usando il comando arecord di sistema.
    Attende voce, poi si ferma dopo _SILENCE_S secondi di silenzio.
    Ritorna bytes WAV, oppure None se arecord non è disponibile.
    """
    import struct, wave, io as _io

    device = _resolve_arecord_device()
    if device is None:
        return None

    chunk_bytes = _ARECORD_RATE * _ARECORD_CHANNELS * 2 // 10

    try:
        proc = subprocess.Popen(
            ["arecord", "-D", device, "-f", "S16_LE",
             "-r", str(_ARECORD_RATE), "-c", str(_ARECORD_CHANNELS), "-q", "--"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        )
    except FileNotFoundError:
        return None

    try:
        proc.stdout.read(chunk_bytes)
    except Exception:
        pass

    stereo_chunks: list[bytes] = []
    silence_start: float | None = None
    speech_started = False
    total_frames = 0
    dot_t = time.monotonic()

    try:
        while total_frames < _ARECORD_RATE * _MAX_RECORD_S:
            chunk = proc.stdout.read(chunk_bytes)
            if not chunk:
                break
            n = len(chunk) // (_ARECORD_CHANNELS * 2)
            left = struct.unpack_from(f"<{n}h", chunk, 0)
            energy = sum(abs(s) for s in left) / len(left)

            stereo_chunks.append(chunk)
            total_frames += n

            if energy > _SPEECH_THRESHOLD:
                if not speech_started:
                    console.event(" parlando...", end="", flush=True)
                speech_started = True
                silence_start = None
            elif speech_started:
                if silence_start is None:
                    silence_start = time.monotonic()
                elif time.monotonic() - silence_start >= _SILENCE_S:
                    break
            else:
                if time.monotonic() - dot_t >= 1.0:
                    console.event(".", end="", flush=True)
                    dot_t = time.monotonic()
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            proc.kill()

    console.event()
    if not speech_started or not stereo_chunks:
        return None

    raw_stereo = b"".join(stereo_chunks)
    n_frames = len(raw_stereo) // (_ARECORD_CHANNELS * 2)
    stereo_samples = struct.unpack_from(f"<{n_frames * _ARECORD_CHANNELS}h", raw_stereo)
    mono = [
        (stereo_samples[i * _ARECORD_CHANNELS] + stereo_samples[i * _ARECORD_CHANNELS + 1]) // 2
        for i in range(n_frames)
    ]
    mono_bytes = struct.pack(f"<{len(mono)}h", *mono)

    buf = _io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(_ARECORD_RATE)
        wf.writeframes(mono_bytes)
    return buf.getvalue()


def record_and_transcribe() -> tuple[str, str]:
    """
    Registra dal microfono e trascrive con Whisper.
    Usa arecord (DMIC hw:1,7) se disponibile, altrimenti PyAudio.
    Ritorna (testo, lingua_iso639) oppure ("", "unknown") su silenzio/errore.
    """
    time.sleep(1.2)

    console.event("\n🎤 In ascolto... (parla e fai una pausa di ~2 secondi)", end="", flush=True)
    wav_data = _record_with_arecord()

    if wav_data is None:
        console.detail("\n   [fallback PyAudio]", flush=True)
        recognizer = sr.Recognizer()
        recognizer.pause_threshold = 2.5
        with console.muted_stderr(), sr.Microphone(device_index=_MIC_DEVICE_INDEX) as source:
            recognizer.adjust_for_ambient_noise(source, duration=1)
            audio_data = recognizer.listen(source)
        wav_data = audio_data.get_wav_data()

    if not wav_data:
        return "", "unknown"

    console.detail("⏳ Trascrizione con Whisper in corso...")
    temp_audio_path = EXCHANGE_DIR / "temp_audio.wav"
    with open(temp_audio_path, "wb") as f:
        f.write(wav_data)

    try:
        with open(temp_audio_path, "rb") as file:
            result = client.audio.transcriptions.create(
                file=(str(temp_audio_path), file.read()),
                model=TRANSCRIPTION_MODEL,
                response_format="verbose_json",
            )

        segments = getattr(result, "segments", None) or []
        if segments:
            avg_no_speech = sum(s.get("no_speech_prob", 0) for s in segments) / len(segments)
            if avg_no_speech > 0.5:
                console.event(f"⚠️  Silenzio (no_speech_prob={avg_no_speech:.2f}) — ignoro.")
                return "", "unknown"

        text = (result.text or "").strip()
        lang = (getattr(result, "language", None) or "unknown").strip()

        if text.lower().rstrip(".! ") in _WHISPER_HALLUCINATIONS or text.lower() in _WHISPER_HALLUCINATIONS:
            console.event(f"⚠️  Hallucination Whisper: \"{text}\" — ignoro.")
            return "", "unknown"

        console.detail(f"   [Whisper] lingua={lang}  testo=\"{text}\"")
        return text, lang

    except Exception as e:
        console.error(f"[Errore Audio] Impossibile trascrivere: {e}")
        return "", "unknown"


def get_user_input(input_mode: str) -> tuple[str, str]:
    """Ritorna (testo, lingua_iso639). Lingua è 'unknown' in modalità testo."""
    if input_mode == "voice":
        return record_and_transcribe()

    if input_mode == "text":
        console.detail("\n⌨️  Modalità testo attiva.")
        text = input("Scrivi cosa direbbe il visitatore: ")
        return text, "unknown"

    raise ValueError(f"Unknown input mode: {input_mode}")





FEASIBILITY_WEIGHTS = {"perception": 0.2, "comprehension": 0.5, "projection": 0.3}
FEASIBLE_THRESHOLD = 0.75
HUMAN_HANDOFF_THRESHOLD = 0.4


def check_feasibility(selected_painting: dict, robot_capabilities: dict, graph=None):
    """
    Two-stage feasibility check.

    Stage 1 — affordance precondition (HRAI 7 - Affordances / HRI5 Sl. 8-10):
    does the painting even afford a pointing gesture at all
    (museo:affordsPointTo)? This is a binary gate, not something to average
    into a score — an affordance that doesn't exist can't be partially used.
    Same for the speech channel: if it's off, nothing can be communicated at all.

    Stage 2 — HRAI 9 self-awareness score Ft(T), given the affordance exists:
      P  (perception)    - kg.perception_coverage(): how much of what's needed
                            to talk about this painting the robot actually
                            perceived with its own camera (museum_explorer.py
                            VLM scan), rather than trusting static YAML alone.
      C  (comprehension) - does the robot's declared capability set
                            (robot_capabilities.yaml) cover the actions this
                            task needs (speech + this specific gesture)?
      Pr (projection)    - is this exact gesture inside the predefined,
                            safety-validated skill set, or would executing it
                            be an untested improvisation?
    Below HUMAN_HANDOFF_THRESHOLD the robot performs the HUMAN primitive from
    the slides: an explicit, honest hand-off to museum staff instead of a
    silent downgrade or an unvalidated improvisation.
    """
    painting_id = selected_painting.get("id")
    pointing_skill = selected_painting.get("robot_actions", {}).get("pointing_skill")
    capabilities = robot_capabilities.get("capabilities", {})

    affords_pointing = kg.can_point(graph, painting_id) if graph is not None else pointing_skill is not None
    if not affords_pointing:
        return {
            "feasible": False, "issue": "missing_pointing_skill", "fallback": "verbal_only",
            "feasibility_score": 0.0, "components": None,
        }

    speech_enabled = capabilities.get("speech", {}).get("enabled", False)
    if not speech_enabled:
        return {
            "feasible": False, "issue": "speech_not_enabled", "fallback": "gesture_only",
            "feasibility_score": 0.0, "components": None,
        }

    perception_score = kg.perception_coverage(graph, painting_id) if graph is not None else 0.5

    available_actions = set(robot_capabilities.get("available_interaction_actions", []))
    required_actions = {"say", pointing_skill}
    comprehension_score = len(required_actions & available_actions) / len(required_actions)

    predefined_skills = set(capabilities.get("pointing", {}).get("skills", []))
    projection_score = 1.0 if pointing_skill in predefined_skills else 0.0

    feasibility_score = (
        FEASIBILITY_WEIGHTS["perception"] * perception_score
        + FEASIBILITY_WEIGHTS["comprehension"] * comprehension_score
        + FEASIBILITY_WEIGHTS["projection"] * projection_score
    )
    components = {
        "perception": round(perception_score, 2),
        "comprehension": round(comprehension_score, 2),
        "projection": round(projection_score, 2),
    }

    if feasibility_score < HUMAN_HANDOFF_THRESHOLD:
        return {
            "feasible": False, "issue": "low_feasibility_score", "fallback": "ask_human",
            "feasibility_score": round(feasibility_score, 2), "components": components,
        }

    if feasibility_score < FEASIBLE_THRESHOLD or pointing_skill not in available_actions:
        return {
            "feasible": False, "issue": f"gesture_not_available:{pointing_skill}", "fallback": "verbal_only",
            "feasibility_score": round(feasibility_score, 2), "components": components,
        }

    return {
        "feasible": True, "issue": None, "fallback": None,
        "feasibility_score": round(feasibility_score, 2), "components": components,
    }



def select_explanation(
    painting: dict, user_model: dict, user_profiles: dict, htn_key: str | None = None
) -> tuple[str, str]:
    """
    HRI6 Sl. 9-12: explanation_key is preferably the task-decomposition result
    of kg.GuidanceHTN().decompose() (htn_key), which already accounts for both
    available_time and profile via its M1-M5 methods — falling back to the flat
    per-profile lookup only if no HTN plan was produced.
    """
    profile = user_model.get("profile", "novice")
    interests = user_model.get("interests", {})
    explanations = painting.get("explanations", {})

    if interests:
        top_interest = max(interests, key=interests.get)
        if top_interest == "emotion" and "emotional" in explanations:
            return "emotional", explanations["emotional"]

    if profile not in user_profiles:
        profile = "novice"

    explanation_key = htn_key or user_profiles[profile].get("recommended_explanation_key", "novice")
    explanation = (
        explanations.get(explanation_key)
        or explanations.get("novice")
        or "I recommend this painting because it matches your preferences."
    )

    return explanation_key, explanation



_CAMERA_INDEX: int | None = None

_FACE_IMAGE_PATH: str | None = None

MOOD_SAMPLE_INTERVAL_S = 10.0

_ACTIVE_MOOD_STOP: threading.Event | None = None

_SURVEY_ENABLED: bool = False

_GESTURE_DISABLED: bool = False

_EMOTION_TO_MOOD: dict[str, str | None] = {
    "happiness": "dynamic",
    "surprise":  "sublime",
    "neutral":   None,
    "sadness":   "serene",
    "fear":      "peaceful",
    "disgust":   "contemplative",
    "anger":     "contemplative",
    "contempt":  "contemplative",
}

_emotion_recognizer = None


def _get_emotion_recognizer():
    global _emotion_recognizer
    if _emotion_recognizer is None:
        from hsemotion_onnx.facial_emotions import HSEmotionRecognizer
        _emotion_recognizer = HSEmotionRecognizer(model_name="enet_b0_8_best_afew")
    return _emotion_recognizer


def detect_mood_from_frames(frames: list) -> tuple[str | None, float]:
    """
    Analizza l'emozione dominante su frame BGR (numpy array) già catturati.
    Usa hsemotion-onnx (AffectNet, EfficientNet-B0) — molto più accurato di FER2013.
    Ritorna (mood, confidence): confidence è la quota pesata di voti
    dell'emozione dominante (1.0 = tutti i frame concordano).
    """
    try:
        import cv2
        recognizer = _get_emotion_recognizer()
    except Exception as e:
        console.detail(f"[FaceDetect] hsemotion-onnx non disponibile: {e}")
        return None, 0.0

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    emotion_votes: dict[str, int] = {}
    valid = 0

    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48)
        )
        if not len(faces):
            continue

        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        face_crop = frame[y: y + h, x: x + w]

        try:
            emotion, _ = recognizer.predict_emotions(face_crop, logits=False)
            emotion_lower = emotion.lower()
            emotion_votes[emotion_lower] = emotion_votes.get(emotion_lower, 0) + 1
            valid += 1
        except Exception:
            pass

    if not valid:
        console.detail("[FaceDetect] Nessuna faccia trovata nei frame")
        return None, 0.0

    weighted = {k: (v * 0.4 if k == "neutral" else float(v)) for k, v in emotion_votes.items()}
    dominant = max(weighted, key=weighted.get)
    mood = _EMOTION_TO_MOOD.get(dominant)
    confidence = weighted[dominant] / sum(weighted.values()) if weighted else 0.0
    return mood, confidence


def detect_visitor_mood(camera_index: int = 0, n_frames: int = 3) -> tuple[str | None, float]:
    """
    Apre la camera, cattura n_frames e chiama detect_mood_from_frames.
    Solo per la modalità standalone (senza face_trigger.py).
    Ritorna (mood, confidence).
    """
    try:
        import cv2
    except ImportError:
        return None, 0.0

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        console.warn(f"[FaceDetect] Camera {camera_index} non disponibile")
        return None, 0.0

    console.event(f"📷 Rilevamento emozione (camera {camera_index})...", end="", flush=True)
    frames = []
    try:
        for _ in range(n_frames):
            time.sleep(0.5)
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
    finally:
        cap.release()

    if not frames:
        console.event(" nessuna cattura.")
        return None, 0.0

    return detect_mood_from_frames(frames)



class MoodMonitor:
    """
    HRI3/HRI5 — percezione continua del segnale sociale: il lato percezione
    (loop di face_trigger o thread camera standalone) deposita qui l'ultima
    stima (mood, confidence) del volto; il loop di dialogo la consuma una
    volta per turno con poll().

    Politica di aggiornamento ("per tempo o per confidence"):
      - una lettura ad ALTA confidence (>= HIGH_CONFIDENCE) può cambiare il
        mood in qualunque momento;
      - una lettura ordinaria (>= MIN_CONFIDENCE) lo cambia solo se il mood
        corrente è stabile da almeno MIN_HOLD_S secondi (evita sfarfallii);
      - sotto MIN_CONFIDENCE la lettura viene ignorata.

    Il mood facciale ha PRIORITÀ su quello inferito dal testo: una volta
    applicato (source="face"), llm_interpreter.merge_user_model non lo
    sovrascrive più con inferenze testuali.
    """

    MIN_CONFIDENCE = 0.45
    HIGH_CONFIDENCE = 0.75
    MIN_HOLD_S = 25.0

    def __init__(self):
        self._lock = threading.Lock()
        self._latest: tuple[str, float] | None = None
        self._last_change_t: float | None = None

    def reset(self) -> None:
        """Nuova sessione: dimentica letture e timer di stabilità."""
        with self._lock:
            self._latest = None
            self._last_change_t = None

    def submit(self, mood: str | None, confidence: float) -> None:
        """Chiamato dal lato percezione ogni volta che c'è una nuova stima."""
        if not mood:
            return
        with self._lock:
            self._latest = (mood, float(confidence))

    def poll(self, current_mood: str) -> tuple[str, float] | None:
        """
        Chiamato dal loop di dialogo. Ritorna (mood, confidence) se il mood
        va aggiornato adesso, altrimenti None. La lettura viene consumata.
        """
        now = time.monotonic()
        with self._lock:
            if self._latest is None:
                return None
            mood, conf = self._latest

            if mood == current_mood:
                self._latest = None
                return None
            if conf < self.MIN_CONFIDENCE:
                self._latest = None
                return None

            held_long_enough = (
                self._last_change_t is None
                or now - self._last_change_t >= self.MIN_HOLD_S
            )
            if conf >= self.HIGH_CONFIDENCE or held_long_enough:
                self._latest = None
                self._last_change_t = now
                return mood, conf
            return None


MOOD_MONITOR = MoodMonitor()


def _apply_mood_from_monitor(state: dict) -> None:
    """Applica al modello utente l'eventuale aggiornamento del mood facciale
    (e lo marca source='face', che vince sulle inferenze testuali).
    NON tocca un mood richiesto esplicitamente a parole dal visitatore
    (source='text_explicit'): la richiesta verbale è la fonte più forte."""
    if state["user_model"].get("preferred_mood_source") == "text_explicit":
        return
    update = MOOD_MONITOR.poll(state["user_model"].get("preferred_mood", "unknown"))
    if update is None:
        return
    mood, conf = update
    state["user_model"]["preferred_mood"] = mood
    state["user_model"]["preferred_mood_source"] = "face"
    console.event(f"   preferred_mood dal volto → {mood} (conf {conf:.2f})")
    kg.sync_visitor(_KB["graph"], state["user_model"])
    _trace("perception", f"Mood update from face → {mood} (conf {conf:.2f})")



_GTTS_LANG_MAP = {
    "it": "it", "en": "en", "fr": "fr", "de": "de",
    "es": "es", "pt": "pt", "nl": "nl", "pl": "pl",
}

_ROBOT_MODE: bool = False
_SILENT_TTS: bool = False

_ros2_node = None
_ros2_tts_client = None


def _init_booster_tts() -> bool:
    """Initialize a ROS2 node and kSpeak service client. Returns True on success."""
    global _ros2_node, _ros2_tts_client
    try:
        import rclpy
        from rclpy.node import Node
        from booster_interface.srv import RpcService

        if not rclpy.ok():
            rclpy.init()
        _ros2_node = Node("museum_tts_node")
        _ros2_tts_client = _ros2_node.create_client(RpcService, "booster_rtc_service")
        if not _ros2_tts_client.wait_for_service(timeout_sec=5.0):
            console.detail("[Booster TTS] booster_rtc_service non disponibile — uso gTTS")
            return False
        console.detail("[Booster TTS] Connesso a booster_rtc_service ✓")
        return True
    except ImportError:
        console.detail("[Booster TTS] rclpy/booster_interface non trovati — uso gTTS")
        return False
    except Exception as e:
        console.warn(f"[Booster TTS] Errore init: {e}")
        return False


def _shutdown_booster_tts() -> None:
    global _ros2_node, _ros2_tts_client
    try:
        if _ros2_node is not None:
            _ros2_node.destroy_node()
            _ros2_node = None
            _ros2_tts_client = None
        import rclpy
        if rclpy.ok():
            rclpy.shutdown()
    except Exception:
        pass


def _booster_speak(text: str) -> None:
    """Send kSpeak (API ID 2002) to the robot's booster_rtc_service speaker."""
    global _ros2_node, _ros2_tts_client
    if _ros2_node is None or _ros2_tts_client is None:
        _tts_speak(text)
        return
    try:
        import rclpy
        from booster_interface.msg import BoosterApiReqMsg
        from booster_interface.srv import RpcService

        msg = BoosterApiReqMsg()
        msg.api_id = 2002
        msg.body = json.dumps({"msg": text})
        request = RpcService.Request()
        request.msg = msg
        future = _ros2_tts_client.call_async(request)
        rclpy.spin_until_future_complete(_ros2_node, future, timeout_sec=60.0)
    except Exception as e:
        console.error(f"[Booster TTS Error] {e}")
        _tts_speak(text)


def _tts_speak(text: str, language: str = "en") -> None:
    if _SILENT_TTS:
        return
    if gTTS is None:
        console.detail("[TTS disabled] gTTS non installato: audio saltato.")
        return
    lang = _GTTS_LANG_MAP.get(language, "en")
    tmp_path = None
    try:
        tts = gTTS(text=text, lang=lang, slow=False)
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            tmp_path = f.name
        tts.save(tmp_path)
        import subprocess as _sp
        _sp.run(
            ["gst-launch-1.0", "-q", "playbin", f"uri=file://{tmp_path}"],
            check=False, timeout=60, capture_output=True,
        )
    except Exception as e:
        console.error(f"[TTS Error] {e}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


def robot_say(text: str, language: str = "en") -> None:
    console.event(f"\n🤖 Robot: {text}")
    if _SILENT_TTS:
        return
    if _ROBOT_MODE:
        _booster_speak(text)
    else:
        _tts_speak(text, language)


def _say(state: dict, text: str) -> None:
    robot_say(text, state["user_model"].get("language", "en"))


def _say_streamed(state: dict, action: str, extra: dict = None) -> str:
    """
    Streams the LLM response sentence-by-sentence and pipelines TTS with playback.

    Simulation mode — three-thread pipeline:
      main   : LLM stream → print each sentence → submit TTS job to thread pool
      feeder : wait for each TTS future in order → enqueue audio path to player
      player : play audio files in order (zero gap between sentences)

    Robot mode — sequential kSpeak per sentence (TTS handled by robot hardware):
      main   : LLM stream → print each sentence → kSpeak (blocking)

    Returns the full utterance text for conversation history.
    """
    language = state["user_model"].get("language", "en")
    parts: list[str] = []
    console.event(f"\n🤖 Robot: ", end="", flush=True)

    if _SILENT_TTS:
        for sentence in stream_robot_utterance(action, state, extra):
            if sentence := sentence.strip():
                parts.append(sentence)
                console.event(sentence, end=" ", flush=True)
        console.event()
        return " ".join(parts)

    if _ROBOT_MODE:
        for sentence in stream_robot_utterance(action, state, extra):
            if sentence := sentence.strip():
                parts.append(sentence)
                console.event(sentence, end=" ", flush=True)
                _booster_speak(sentence)
        console.event()
        return " ".join(parts)

    if gTTS is None:
        console.detail("[TTS disabled] gTTS non installato: audio saltato.")
        for sentence in stream_robot_utterance(action, state, extra):
            if sentence := sentence.strip():
                parts.append(sentence)
                console.event(sentence, end=" ", flush=True)
        console.event()
        return " ".join(parts)

    lang = _GTTS_LANG_MAP.get(language, "en")
    play_q: queue.Queue = queue.Queue()
    future_q: queue.Queue = queue.Queue()

    def _tts_job(text: str) -> str | None:
        if gTTS is None:
            return None
        try:
            tts = gTTS(text=text, lang=lang, slow=False)
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                path = f.name
            tts.save(path)
            return path
        except Exception as e:
            console.error(f"\n[TTS Error] {e}")
            return None

    def _feeder() -> None:
        while True:
            fut = future_q.get()
            if fut is None:
                play_q.put(None)
                return
            path = fut.result()
            if path:
                play_q.put(path)

    def _player() -> None:
        import subprocess as _sp
        while True:
            path = play_q.get()
            if path is None:
                return
            try:
                _sp.run(
                    ["gst-launch-1.0", "-q", "playbin", f"uri=file://{path}"],
                    check=False, timeout=60, capture_output=True,
                )
            finally:
                if os.path.exists(path):
                    os.unlink(path)

    feeder = threading.Thread(target=_feeder, daemon=True)
    player = threading.Thread(target=_player, daemon=True)
    feeder.start()
    player.start()

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for sentence in stream_robot_utterance(action, state, extra):
            if sentence := sentence.strip():
                parts.append(sentence)
                console.event(sentence, end=" ", flush=True)
                future_q.put(pool.submit(_tts_job, sentence))

    future_q.put(None)
    feeder.join()
    player.join()
    console.event()
    return " ".join(parts)



def save_interaction_log(log_data: dict):
    existing = sorted(LOG_DIR.glob("interaction_*.json"))
    index = len(existing) + 1

    log_path = LOG_DIR / f"interaction_{index:03d}.json"

    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log_data, f, indent=2, ensure_ascii=False)

    console.detail(f"📝 Log salvato in: {log_path}")


def save_session_summary(state: dict, input_mode: str, started_at: float) -> None:
    """
    HRAI-lecture1 (Intro + Benchmarking): task-level session summary —
    completion, turns, paintings shown, duration — as opposed to the
    per-recommendation *functionality* logs written by save_interaction_log().
    Read together by benchmark_report.py.
    """
    existing = sorted(LOG_DIR.glob("session_*.json"))
    index = len(existing) + 1
    summary = {
        "input_mode": input_mode,
        "duration_s": round(time.monotonic() - started_at, 1),
        "turns": state.get("turn_count", 0),
        "clarification_count": state.get("clarification_count", 0),
        "social_refusal_count": state.get("social_refusal_count", 0),
        "staff_command_executed": state.get("staff_command_executed", False),
        "paintings_shown": len(state.get("shown_painting_ids", [])),
        "final_user_model": state["user_model"],
    }
    log_path = LOG_DIR / f"session_{index:03d}.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    console.detail(f"📝 Session summary salvato in: {log_path}")


_GODSPEED_ITEMS = [
    ("anthropomorphism",      "Quanto ti è sembrato umano il robot?",              "Meccanico",       "Umano"),
    ("animacy",               "Quanto ti è sembrato vivo/reattivo il robot?",       "Immobile/rigido", "Vivo/reattivo"),
    ("likeability",           "Quanto ti è piaciuto interagire con il robot?",      "Sgradevole",      "Piacevole"),
    ("perceived_intelligence","Quanto ti è sembrato intelligente il robot?",        "Incompetente",    "Competente"),
    ("perceived_safety",      "Quanto ti sei sentito a tuo agio durante l'interazione?", "Ansioso",    "Rilassato"),
]


def run_godspeed_survey(state: dict) -> dict:
    """
    HRI7-Evaluation: administers the Godspeed Questionnaire right after
    goodbye (opt-in via --survey). Logs the ratings together with the
    independent variables (visitor profile/interests) and session covariates
    (clarification turns, paintings shown) needed for a between/within-subject
    statistical comparison (t-test/ANOVA), as taught in the slides.
    """
    console.event("\n" + "=" * 50)
    console.event("📋 Questionario Godspeed (facoltativo — CTRL+C per saltare)")
    console.event("   Rispondi con un numero da 1 (sinistra) a 5 (destra).")
    console.event("=" * 50)

    ratings: dict[str, int] = {}
    try:
        for key, question, low_anchor, high_anchor in _GODSPEED_ITEMS:
            while True:
                raw = input(f"\n{question}\n  1={low_anchor} ... 5={high_anchor}\n  > ").strip()
                if raw.isdigit() and 1 <= int(raw) <= 5:
                    ratings[key] = int(raw)
                    break
                console.event("  Inserisci un numero da 1 a 5.")
    except KeyboardInterrupt:
        console.event("\n[Survey] Saltato dal visitatore.")
        return {}

    survey_log = {
        "godspeed_ratings": ratings,
        "independent_variables": {
            "visitor_profile":   state["user_model"].get("profile", "unknown"),
            "visitor_interests": state["user_model"].get("interests", {}),
        },
        "session_covariates": {
            "clarification_count": state.get("clarification_count", 0),
            "paintings_shown":     len(state.get("shown_painting_ids", [])),
        },
    }

    existing = sorted(LOG_DIR.glob("survey_*.json"))
    index = len(existing) + 1
    survey_path = LOG_DIR / f"survey_{index:03d}.json"
    with open(survey_path, "w", encoding="utf-8") as f:
        json.dump(survey_log, f, indent=2, ensure_ascii=False)
    console.detail(f"📝 Survey salvato in: {survey_path}")
    return survey_log



def handle_recommend(state: dict, paintings: list,
                     robot_capabilities: dict, user_profiles: dict):
    """
    Runs the full recommendation pipeline and updates state.
    Returns the log entry for this recommendation.

    HRI6 Sl. 9-16 / HRAI 5 - Scene Graphs: painting selection and epistemic
    tracking are delegated to the RDF knowledge graph (knowledge_graph.py):
      - kg.sparql_recommend() replaces score_painting()/recommend(): semantic
        matching (theme/profile/mood) + epistemic filtering (notKnows) in SPARQL
      - kg.GuidanceHTN().decompose() replaces the ad-hoc explanation-key lookup
        with an explicit HTN task decomposition (guide_visitor -> primitive tasks)
      - kg.mark_shown() advances the visitor's epistemic state (notKnows -> knows)
    """
    user_model = state["user_model"]
    graph = _KB["graph"]

    effective_model = dict(user_model)
    if effective_model["profile"] == "unknown":
        effective_model["profile"] = "novice"
    if effective_model["available_time"] == "unknown":
        effective_model["available_time"] = "normal"

    console.event("\n🖼️  Calcolo raccomandazione (knowledge graph + SPARQL)...")
    kg.sync_visitor(graph, effective_model)
    _trace("kg_sync", f"profile={effective_model['profile']} interests={effective_model['interests']} "
                       f"mood={effective_model.get('preferred_mood')}")

    best_painting, candidate_scores = kg.sparql_recommend(graph, paintings)
    if best_painting is None:
        kg.reopen_gallery(graph)
        _trace("kg_update", "Gallery reopened (all paintings already shown)")
        best_painting, candidate_scores = kg.sparql_recommend(graph, paintings)

    if best_painting is None:
        _trace("feasibility", "Nessun quadro raccomandabile (tutti in manutenzione / sala vuota)")
        text = _say_streamed(state, "nothing_available")
        state["history"].append({"role": "assistant", "content": text})
        state["auto_continue"] = False
        return {"selected_painting": None, "robot_messages": {"main": text}}

    pointing_skill = best_painting["robot_actions"]["pointing_skill"]
    target_xyz = best_painting["robot_actions"].get("target_xyz")
    _trace("kg_query", f'SPARQL → "{best_painting["title"]}" ({best_painting["id"]})')
    task_sequence, htn_key = kg.GuidanceHTN().decompose(graph)
    _trace("htn", f"plan={task_sequence} key={htn_key}")

    console.event(f"   Quadro: {best_painting['title']}  (gesture: {pointing_skill})")
    console.event(f"   HTN plan: {' → '.join(task_sequence)}  (explanation_key: {htn_key})")

    feasibility = check_feasibility(best_painting, robot_capabilities, graph)
    _trace("feasibility", f"Ft={feasibility['feasibility_score']} "
                           f"components={feasibility['components']} fallback={feasibility['fallback']}")
    explanation_key, explanation = select_explanation(best_painting, effective_model, user_profiles, htn_key)
    spatial_reference = kg.spatial_relation_text(graph, best_painting["id"])

    console.event(f"   Feasibility Ft={feasibility['feasibility_score']}  "
                  f"components={feasibility['components']}  fallback={feasibility['fallback']}")

    if target_xyz is not None:
        pose = read_latest_pose()
        if pose is not None:
            target_xyz = to_base_frame(target_xyz, pose)

    _gesture_result = [False]
    gesture_thread = None
    if _GESTURE_DISABLED:
        console.event("   GESTURE SKIPPED — experimental --no-gesture condition (speech only)")
        _trace("action", "say(...) only — gesture disabled (H1 no_gesture condition)")
    elif feasibility["feasible"]:
        def _gesture_task():
            if target_xyz is None or not run_robot_point_at(target_xyz):
                _gesture_result[0] = run_robot_gesture(pointing_skill)
            else:
                _gesture_result[0] = True
        gesture_thread = threading.Thread(target=_gesture_task, daemon=True)
        gesture_thread.start()
        if target_xyz is not None:
            console.event(f"   GESTURE started: point_at {target_xyz} (fallback: {pointing_skill})")
        else:
            console.event(f"   GESTURE started: {pointing_skill}")
        _trace("action", f"say(recommend_and_explain) + gesture({pointing_skill}, xyz={target_xyz})")
    elif feasibility["fallback"] == "ask_human":
        console.event(f"   GESTURE SKIPPED — HUMAN hand-off ({feasibility['issue']})")
        _trace("action", "say(...) + HUMAN hand-off (no gesture)")
    else:
        console.event(f"   GESTURE SKIPPED: {feasibility['issue']}")
        _trace("action", f"say(...) only — gesture skipped ({feasibility['issue']})")

    main_text = _say_streamed(
        state, "recommend_and_explain",
        extra={
            "painting": best_painting,
            "explanation": explanation,
            "explanation_key": explanation_key,
            "spatial_reference": spatial_reference,
            "low_confidence_handoff": feasibility["fallback"] == "ask_human",
        },
    )

    if gesture_thread is not None:
        gesture_thread.join()
    gesture_executed = _gesture_result[0]

    if feasibility["feasible"] and not gesture_executed:
        _trace("action", "gesture failed at runtime → verbal spatial fallback")
        recovery_text = _say_streamed(
            state, "gesture_failed_verbal_fallback",
            extra={"painting": best_painting, "spatial_reference": spatial_reference},
        )
        main_text = f"{main_text} {recovery_text}"

    if gesture_executed:
        run_neutral_pose()

    followup_text = _say_streamed(state, "satisfaction_check", extra={"painting": best_painting})

    state["history"].append({"role": "assistant", "content": f"{main_text} {followup_text}"})
    state["last_painting"] = best_painting
    state.setdefault("shown_painting_ids", []).append(best_painting["id"])
    kg.mark_shown(graph, best_painting["id"])
    _trace("kg_update", f'{best_painting["id"]}: notKnows → knows ("{best_painting["title"]}")')
    state["auto_continue"] = True

    return {
        "selected_painting": {
            "id": best_painting["id"],
            "title": best_painting["title"],
            "htn_plan": task_sequence,
            "pointing_skill": pointing_skill,
            "explanation_key": explanation_key,
        },
        "feasibility": feasibility,
        "gesture_executed": gesture_executed,
        "robot_messages": {
            "main": main_text,
            "follow_up": followup_text,
        },
    }



def handle_greeting_ack(state: dict) -> None:
    text = _say_streamed(state, "greeting_ack")
    state["history"].append({"role": "assistant", "content": text})


def handle_clarification(state: dict, clarification_q: str) -> None:
    text = _say_streamed(state, "ask_clarification", extra={"question": clarification_q})
    state["history"].append({"role": "assistant", "content": text})
    state["clarification_count"] = state.get("clarification_count", 0) + 1


def _find_mentioned_painting(text: str, paintings: list) -> dict | None:
    """
    Heuristic: scan a free-form robot utterance for a painting title.

    answer_general lets the LLM freely name a painting (e.g. when asked a
    general question about "history"), but that mention never went through
    handle_recommend(), so state["last_painting"] would otherwise stay unset —
    and a follow-up "tell me more" would have nothing to attach to and would
    incorrectly fall back to a brand-new recommendation instead of expanding
    on the painting just named. Only acts when exactly one title matches, to
    avoid guessing wrong when several paintings are mentioned in passing.
    """
    text_lower = text.lower()
    matches = [p for p in paintings if p["title"].lower() in text_lower]
    return matches[0] if len(matches) == 1 else None


def _find_requested_catalog_painting(text: str) -> dict | None:
    """
    Match the visitor's message against the FULL catalog (not just what is hung)
    by painting title, then by artist. Used to detect a request for a specific
    work that may be unavailable (see handle_unavailable_request). Returns the
    painting dict only for a single unambiguous match, else None.
    """
    text_lower = text.lower()
    catalog = _KB["paintings"]
    by_title = [p for p in catalog if p["title"].lower() in text_lower]
    if by_title:
        return by_title[0] if len(by_title) == 1 else None
    by_artist = [p for p in catalog if p["artist"].lower() in text_lower]
    return by_artist[0] if len(by_artist) == 1 else None


def _painting_available(painting_id: str) -> bool:
    """Available iff the painting is currently hung AND not flagged under maintenance."""
    present_ids = {p["id"] for p in _KB["present_paintings"]}
    if painting_id not in present_ids:
        return False
    return not kg.is_under_maintenance(_KB["graph"], painting_id)


def handle_unavailable_request(state: dict, requested: dict) -> None:
    """
    Visitor-facing contingency (HRAI project guidelines "good example": the
    default plan fails and the robot must adapt). The visitor asked for a
    specific painting that is not available now — not on display, or flagged
    under maintenance by staff. Instead of ignoring the request or recommending
    something unrelated, acknowledge the situation and negotiate the closest
    available alternative (curated related_to links + shared movement/themes).
    If the visitor then refuses, the normal recommend flow proposes the next
    option — the "easy vs difficult visitor" pair the guidelines ask for.
    """
    graph   = _KB["graph"]
    present = _KB["present_paintings"]
    reason  = "maintenance" if kg.is_under_maintenance(graph, requested["id"]) else "not_on_display"
    alternative = kg.most_similar_available(graph, requested, present)

    if alternative is None:
        _trace("kg_query", f'"{requested["title"]}" non disponibile ({reason}) — nessuna alternativa disponibile')
        text = _say_streamed(state, "nothing_available", extra={"requested": requested})
        state["history"].append({"role": "assistant", "content": text})
        return

    _trace("kg_query", f'"{requested["title"]}" non disponibile ({reason}) → propongo "{alternative["title"]}"')
    text = _say_streamed(
        state, "unavailable_suggest_alternative",
        extra={"requested": requested, "reason": reason, "painting": alternative},
    )
    state["history"].append({"role": "assistant", "content": text})
    state["last_painting"] = alternative
    state.setdefault("shown_painting_ids", []).append(alternative["id"])
    kg.mark_shown(graph, alternative["id"])
    _trace("kg_update", f'{alternative["id"]}: notKnows → knows (proposed as alternative)')
    _export_kg_snapshot()
    state["phase"] = "presenting"
    state["auto_continue"] = True


def handle_staff_rescan(state: dict) -> None:
    """
    HRI6 - Joint Intention Theory: staff notifying the robot that a
    previously-adopted belief (the current hanging) is no longer valid — the
    textbook "team member discovers a goal/belief is stale, tells the team"
    trigger. Blocking on purpose: this is a deliberate operational command
    from an authenticated staff member, not a conversational turn, and
    museum_explorer.explore_museum() physically moves the head for ~30-60s —
    narrating it first sets that expectation instead of leaving staff
    wondering why the robot went quiet.
    """
    text = _say_streamed(state, "staff_rescan")
    state["history"].append({"role": "assistant", "content": text})

    from museum_explorer import explore_museum, save_scan_results
    results = explore_museum(live_viewer=True, serve_viewer=False)
    if save_scan_results(results):
        reload_kb()
        kg.sync_visitor(_KB["graph"], state["user_model"])
        _trace("staff", "Rescan completato — knowledge base ricaricata da discovered_paintings.json")
        _export_kg_snapshot()
        console.event("   [staff] Rescan completato, knowledge base ricaricata.")
    else:
        _trace("staff", "Rescan fallito — knowledge base invariata")
        console.event("   [staff] Rescan fallito (nessuna direzione riconosciuta), knowledge base invariata.")


def _handle_staff_painting_toggle(state: dict, user_text: str, flag: bool, action: str) -> None:
    """Shared body for staff_disable_painting/staff_enable_painting — the
    only difference is the maintenance flag and which action/prompt to use."""
    mentioned = _find_mentioned_painting(user_text, _KB["present_paintings"])
    if mentioned is None:
        text = _say_streamed(state, "ask_clarification", extra={
            "question": "Which painting are you referring to? Please say its title.",
        })
        state["history"].append({"role": "assistant", "content": text})
        return

    kg.set_maintenance(_KB["graph"], mentioned["id"], flag)
    state["staff_command_executed"] = True
    _trace("staff", f'{"Suspended from" if flag else "Restored to"} recommendations: '
                     f'"{mentioned["title"]}" ({mentioned["id"]})')
    text = _say_streamed(state, action, extra={"painting": mentioned})
    state["history"].append({"role": "assistant", "content": text})


def handle_staff_disable_painting(state: dict, user_text: str) -> None:
    _handle_staff_painting_toggle(state, user_text, True, "staff_disable_painting")


def handle_staff_enable_painting(state: dict, user_text: str) -> None:
    _handle_staff_painting_toggle(state, user_text, False, "staff_enable_painting")


def handle_answer_question(state: dict) -> None:
    text = _say_streamed(state, "answer_general")
    state["history"].append({"role": "assistant", "content": text})

    mentioned = _find_mentioned_painting(text, _KB["present_paintings"])
    if mentioned:
        state["last_painting"] = mentioned
        console.detail(f"   [context] quadro nominato nella risposta generale → last_painting: {mentioned['title']}")


def handle_answer_more_detail(state: dict) -> None:
    text = _say_streamed(state, "answer_more_detail", extra={"painting": state["last_painting"]})
    followup = _say_streamed(state, "satisfaction_check", extra={"painting": state["last_painting"]})
    state["history"].append({"role": "assistant", "content": f"{text} {followup}"})


_PREF_TOPIC_LABELS = {
    "favorite_painting": "your favourite painting",
    "favorite_movement": "your favourite artistic movement",
    "favorite_artist":   "your favourite artist",
}


def _infer_preference_topic(user_text: str, state: dict) -> str:
    """
    Which art preference is the visitor asking about? Cheap keyword heuristic
    (multi-language) — an "opinion on the current painting" gets its own per-
    painting topic so the robot keeps a stable opinion about each work.
    """
    t = user_text.lower()
    if any(w in t for w in ("this painting", "questo quadro", "questa opera", "ce tableau", "dieses bild", "este cuadro")) \
            and state.get("last_painting"):
        return f"opinion_{state['last_painting']['id']}"
    if any(w in t for w in ("movement", "style", "movimento", "stile", "corrente", "mouvement", "estilo")):
        return "favorite_movement"
    if any(w in t for w in ("artist", "painter", "artista", "pittore", "autore", "artiste", "peintre", "künstler")):
        return "favorite_artist"
    return "favorite_painting"


def handle_robot_preference(state: dict, user_text: str) -> None:
    """
    HRI8 - Cognitive Architectures (slide 30-36): answer the visitor's question
    about the robot's OWN art taste. Recall a stored preference if one exists;
    otherwise form one (grounded in the works on display), persist it to the KG
    and to disk, and reuse it in later turns — so the opinion stays stable.
    """
    graph = _KB["graph"]
    present = _KB["present_paintings"]
    topic = _infer_preference_topic(user_text, state)

    existing = kg.get_robot_preference(graph, topic)
    if existing is not None:
        value, reason = existing["value"], existing["reason"]
        _trace("preference", f"recalled {topic} → {value}")
    else:
        pref = form_robot_preference(topic, present, state.get("personality", {}),
                                     state["user_model"].get("language", "en"))
        value, reason = pref["value"], pref["reason"]
        kg.set_robot_preference(graph, topic, value, reason)
        _save_robot_preferences(graph)
        _trace("preference", f"formed {topic} → {value}")
        _export_kg_snapshot()

    topic_label = _PREF_TOPIC_LABELS.get(topic, "this work" if topic.startswith("opinion_") else "art in this room")
    text = _say_streamed(state, "robot_preference",
                         extra={"topic_label": topic_label, "value": value, "reason": reason})
    state["history"].append({"role": "assistant", "content": text})


def handle_close_or_restart(state: dict) -> None:
    text = _say_streamed(state, "close_or_restart")
    state["history"].append({"role": "assistant", "content": text})


def handle_goodbye(state: dict) -> None:
    _say_streamed(state, "goodbye")



def run_session(
    input_mode: str = "voice",
    proactive_greeting: bool = False,
    wait_for_enter: bool = True,
    detected_mood: str | None = None,
    mood_future=None,
) -> None:
    """
    Run one visitor session until goodbye or KeyboardInterrupt.

    proactive_greeting — robot speaks first (English open welcome); used by face trigger.
    wait_for_enter     — show "Press ENTER" prompt before each voice input turn.
    """
    if _KB is None:
        init_kb()

    session_started_at = time.monotonic()

    paintings       = _KB["present_paintings"]
    robot_caps      = _KB["robot_capabilities"]
    user_profiles   = _KB["user_profiles"]

    kg.reset_visitor(_KB["graph"], paintings)
    _trace("session", "New visitor — epistemic state reset (all paintings notKnows)")
    _export_kg_snapshot()

    state = initialize_state()
    global _ACTIVE_USER_MODEL
    _ACTIVE_USER_MODEL = state["user_model"]
    state["museum_summary"] = _KB["museum_summary"]
    state["robot_capabilities"] = _KB["robot_capabilities"]
    state["personality"] = _KB.get("personality", {})

    global _ACTIVE_MOOD_STOP
    MOOD_MONITOR.reset()
    if _ACTIVE_MOOD_STOP is not None:
        _ACTIVE_MOOD_STOP.set()
    _mood_stop = threading.Event()
    _ACTIVE_MOOD_STOP = _mood_stop

    if detected_mood:
        MOOD_MONITOR.submit(detected_mood, 0.9)

    if _CAMERA_INDEX is not None:
        def _mood_sampler():
            while not _mood_stop.is_set():
                mood, conf = detect_visitor_mood(_CAMERA_INDEX)
                MOOD_MONITOR.submit(mood, conf)
                _mood_stop.wait(MOOD_SAMPLE_INTERVAL_S)
        threading.Thread(target=_mood_sampler, daemon=True).start()

    if _FACE_IMAGE_PATH is not None:
        import cv2
        img = cv2.imread(_FACE_IMAGE_PATH)
        if img is None:
            console.warn(f"[face-image] Impossibile leggere '{_FACE_IMAGE_PATH}' — ignorato.")
        else:
            mood, conf = detect_mood_from_frames([img])
            MOOD_MONITOR.submit(mood, conf)
            console.event(f"   preferred_mood da --face-image → {mood} (conf {conf:.2f})")

    if proactive_greeting:
        text = _say_streamed(state, "open_conversation")
        state["history"].append({"role": "assistant", "content": text})

    if mood_future is not None:
        try:
            mood, conf = mood_future.result(timeout=1.0)
            MOOD_MONITOR.submit(mood, conf)
        except Exception:
            pass

    _apply_mood_from_monitor(state)

    while True:
        console.event("\n--------------------------------------------------")
        if not state.get("auto_continue", False) and wait_for_enter:
            if input_mode == "voice":
                input("👉 Premi [INVIO] per parlare, oppure CTRL+C per uscire...")
            else:
                input("👉 Premi [INVIO] per rispondere, oppure CTRL+C per uscire...")
        state["auto_continue"] = False

        user_text, whisper_lang = get_user_input(input_mode)
        if not user_text.strip():
            console.event("⚠️  Non ho sentito nulla. Riprova.")
            continue

        state["turn_count"] = state.get("turn_count", 0) + 1

        if whisper_lang not in ("unknown", "", "en") and state["user_model"].get("language", "en") == "en":
            state["user_model"]["language"] = whisper_lang
            console.event(f"   lingua rilevata da Whisper: {whisper_lang}")
        elif whisper_lang in ("unknown", ""):
            _heur_lang = _detect_language_heuristic(user_text)
            if _heur_lang != "unknown":
                state["user_model"]["language"] = _heur_lang

        console.event(f"\n🗣️  Visitatore: \"{user_text}\"")
        state["history"].append({"role": "user", "content": user_text})

        if social_supervisor_enabled():
            safety = check_social_safety(user_text, state["history"])
            if not safety["safe"]:
                console.event(f"   🛡️  Social Supervisor: UNSAFE ({safety['category']}) — azione non eseguita")
                _trace("social_supervisor", f"unsafe:{safety['category']} — refused, no action")
                state["social_refusal_count"] = state.get("social_refusal_count", 0) + 1
                text = _say_streamed(state, "refuse_unsafe_request",
                                     extra={"category": safety["category"], "reason": safety["reason"]})
                state["history"].append({"role": "assistant", "content": text})
                continue

        _user_text_lower = user_text.lower()
        _just_authenticated = False
        if (state["user_model"].get("role", "visitor") != "staff"
                and _STAFF_PASSPHRASE in _user_text_lower):
            state["user_model"]["role"] = "staff"
            _just_authenticated = True
            console.event("   [staff] Passphrase riconosciuta — role → staff")
            _trace("staff", "Autenticazione staff riuscita")
        elif (state["user_model"].get("role", "visitor") != "staff"
                and any(kw in _user_text_lower for kw in _STAFF_CLAIM_KEYWORDS)):
            console.event("   [staff] Rivendicazione di ruolo staff senza passphrase — richiedo il codice")
            text = _say_streamed(state, "staff_claim_needs_passphrase")
            state["history"].append({"role": "assistant", "content": text})
            continue

        _apply_mood_from_monitor(state)

        console.event("🧠 Analisi input...")
        analysis       = analyze_input(user_text, state["history"], role=state["user_model"].get("role", "visitor"))
        intent         = analysis["intent"]
        confidence     = analysis["confidence"]
        clarification_q = analysis["clarification_question"]
        console.event(f"   intent: {intent}  |  confidence: {confidence:.2f}")
        _trace("nlu", f'"{user_text[:60]}" → intent={intent} (conf={confidence:.2f})')

        _mood_before = (state["user_model"].get("preferred_mood"),
                        state["user_model"].get("preferred_mood_source"))
        state["user_model"] = merge_user_model(
            state["user_model"], analysis["user_model_update"]
        )
        _ACTIVE_USER_MODEL = state["user_model"]
        state["confidence"] = confidence
        console.event(f"   user_model: {_format_user_model(state['user_model'])}")
        if (state["user_model"].get("preferred_mood_source") == "text_explicit"
                and _mood_before != (state["user_model"]["preferred_mood"], "text_explicit")):
            overridden = " (overrides face)" if _mood_before[1] == "face" else ""
            _trace("nlu", f'Explicit mood request → {state["user_model"]["preferred_mood"]}{overridden}')
        kg.sync_visitor(_KB["graph"], state["user_model"])
        _export_kg_snapshot()

        action, next_phase = get_next_action(state, intent, confidence)
        console.event(f"   action: {action}  |  {state['phase']} → {next_phase}")
        _trace("fsm", f"{state['phase']} → {action} → {next_phase}")

        _STAFF_COMMANDS = ("staff_rescan", "staff_disable_painting", "staff_enable_painting")
        if _just_authenticated and action not in _STAFF_COMMANDS:
            text = _say_streamed(state, "staff_access_granted")
            state["history"].append({"role": "assistant", "content": text})
            _trace("fsm", "staff authenticated → prompt for operational command")
            continue

        if action not in _STAFF_COMMANDS and action != "goodbye":
            _requested = _find_requested_catalog_painting(user_text)
            if _requested is not None and not _painting_available(_requested["id"]):
                _trace("fsm", f'{state["phase"]} → unavailable_request("{_requested["title"]}")')
                handle_unavailable_request(state, _requested)
                continue

        if action == "greeting_ack":
            handle_greeting_ack(state)
        elif action == "ask_clarification":
            handle_clarification(state, clarification_q)
        elif action == "recommend":
            log_entry = handle_recommend(state, paintings, robot_caps, user_profiles)
            save_interaction_log({"user_text": user_text, "user_model": state["user_model"], **log_entry})
        elif action == "answer_question":
            handle_answer_question(state)
        elif action == "answer_more_detail":
            handle_answer_more_detail(state)
        elif action == "robot_preference":
            handle_robot_preference(state, user_text)
        elif action == "staff_rescan":
            handle_staff_rescan(state)
            paintings, robot_caps = _KB["present_paintings"], _KB["robot_capabilities"]
        elif action == "staff_disable_painting":
            handle_staff_disable_painting(state, user_text)
        elif action == "staff_enable_painting":
            handle_staff_enable_painting(state, user_text)
        elif action == "close_or_restart":
            handle_close_or_restart(state)
        elif action == "goodbye":
            handle_goodbye(state)
            _mood_stop.set()
            _trace("session", "Visitor ended the session (goodbye)")
            _export_kg_snapshot()
            if _SURVEY_ENABLED:
                run_godspeed_survey(state)
            save_session_summary(state, input_mode, session_started_at)
            console.event("\n✅ Sessione terminata.")
            return
        else:
            console.warn(f"[Warning] Unknown action: {action}")

        state["phase"] = next_phase
        kg.sync_visitor(_KB["graph"], state["user_model"])
        _export_kg_snapshot()



def parse_args():
    parser = argparse.ArgumentParser(description="Booster-T1 museum guide demo")
    parser.add_argument(
        "--input-mode",
        choices=["voice", "text"],
        default="voice",
        help="Choose how the visitor input is acquired: voice or text.",
    )
    parser.add_argument(
        "--mode",
        choices=["simulation", "robot"],
        default="simulation",
        help="simulation: gTTS on PC speakers.  robot: Booster's own speaker via ROS2 kSpeak.",
    )
    parser.add_argument(
        "--no-tts",
        action="store_true",
        help="Debug: non riprodurre audio, stampa solo le risposte del robot.",
    )
    parser.add_argument(
        "--no-enter",
        action="store_true",
        help="Debug: non chiedere ENTER prima di ogni input del visitatore.",
    )
    parser.add_argument(
        "--mic-device-index",
        type=int,
        default=None,
        metavar="N",
        help="Index of the microphone device to use (default: system default). "
             "Run with --list-mics to see available devices.",
    )
    parser.add_argument(
        "--list-mics",
        action="store_true",
        help="List available microphone devices and exit.",
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=None,
        metavar="N",
        help="Camera device index for facial emotion detection (default: disabled). "
             "Use 0 for the default webcam.",
    )
    parser.add_argument(
        "--gesture-mode",
        "-gesture-mode",
        choices=["trajectory", "goal"],
        default=os.environ.get("GESTURE_MODE", "trajectory"),
        help=(
            "Sceglie come eseguire i gesti: "
            "trajectory usa i file .hpp, goal usa un singolo target articolare con rampa."
        ),
    )
    parser.add_argument(
        "--survey",
        action="store_true",
        help="HRI7-Evaluation: chiedi un breve questionario Godspeed al termine di ogni sessione.",
    )
    parser.add_argument(
        "--no-gesture",
        action="store_true",
        help="HRI7-Evaluation (H1): disabilita ogni gesto di puntamento — il robot "
             "spiega solo a voce. Solo per l'evaluation sperimentale, di default disattivato.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Demo/registrazione: stampa solo la traccia interessante (parlato del "
             "visitatore, risposta del robot, rilevamenti sul volto, scelte del robot "
             "con confidence) e silenzia la telemetria tecnica.",
    )
    parser.add_argument(
        "--face-image",
        default=None,
        metavar="PATH",
        help="HRI7-Evaluation: sostituto statico di una webcam live — legge un'immagine "
             "di un volto da file e la passa alla stessa pipeline di rilevamento emozioni "
             "usata in modalità camera. Solo per benchmark/test, di default disattivato.",
    )
    return parser.parse_args()



def main():
    global _ROBOT_MODE, _MIC_DEVICE_INDEX, _CAMERA_INDEX, _SILENT_TTS, _SURVEY_ENABLED, _GESTURE_DISABLED, _FACE_IMAGE_PATH
    args = parse_args()

    console.set_quiet(args.quiet)
    os.environ["GESTURE_MODE"] = args.gesture_mode

    if args.list_mics:
        console.event("Dispositivi microfono disponibili:")
        for i, name in enumerate(sr.Microphone.list_microphone_names()):
            console.event(f"  [{i}] {name}")
        return

    input_mode = args.input_mode
    _SILENT_TTS = args.no_tts
    _ROBOT_MODE = (args.mode == "robot") and not _SILENT_TTS
    _MIC_DEVICE_INDEX = args.mic_device_index
    _CAMERA_INDEX = args.camera_index
    _SURVEY_ENABLED = args.survey
    _GESTURE_DISABLED = args.no_gesture
    _FACE_IMAGE_PATH = args.face_image

    print_banner(input_mode, args.gesture_mode)

    if _ROBOT_MODE and not _init_booster_tts():
        console.warn("⚠️  Impossibile connettersi al robot — uso gTTS come fallback.")
        _ROBOT_MODE = False

    try:
        init_kb()
    except Exception as e:
        console.error(f"❌ Errore nel caricamento della knowledge base: {e}")
        return
    
    console.detail("🔧 Preparazione gesture backend...")
    prepare_gesture_backend()

    while True:
        try:
            console.event("\nIn attesa del prossimo visitatore (premi CTRL+C per uscire)...")
            run_session(input_mode=input_mode, proactive_greeting=False, wait_for_enter=not args.no_enter)
            console.event("\n✅ Sessione terminata. In attesa del prossimo visitatore...")
        except KeyboardInterrupt:
            console.event("\nSpegnimento del sistema. Arrivederci!")
            _shutdown_booster_tts()
            break


if __name__ == "__main__":
    main()
