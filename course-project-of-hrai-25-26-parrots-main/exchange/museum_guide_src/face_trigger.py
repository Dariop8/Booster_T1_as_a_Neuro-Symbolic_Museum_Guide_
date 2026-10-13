#!/usr/bin/env python3
"""
face_trigger.py — Face-detection trigger for the museum guide (Booster T1 or TIAGo).

TWO MODES:

  --webcam  (test sul PC, nessun ROS2 richiesto)
    Legge dalla webcam del PC, apre una finestra di preview.
    python3 face_trigger.py --webcam [--input-mode voice|text]

  default  (robot reale / simulazione, dentro Docker con ROS2)
    Subscribes ai topic ROS2 del robot (vedi RGB_TOPIC/DEPTH_TOPIC piu sotto):
      /camera/camera/color/image_raw
      /camera/camera/aligned_depth_to_color/image_raw
    python3 face_trigger.py --mode simulation|robot [--input-mode voice|text]

Trigger: faccia visibile ≥ FACE_TIMER_S secondi + distanza stimata ≤ MAX_DIST_M
→ Robot saluta in inglese, poi passa alla lingua del visitatore dopo la prima risposta.
→ Cooldown di COOLDOWN_S secondi al termine della conversazione.
"""

import sys
import time
import threading
import argparse
from pathlib import Path
import os

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

import console

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("GLOG_minloglevel", "2")

_mp_det = None
_USE_MEDIAPIPE = False
try:
    import mediapipe as mp
    _mp_det = mp.solutions.face_detection.FaceDetection(
        model_selection=0,
        min_detection_confidence=0.5,
    )
    _USE_MEDIAPIPE = True
except (ImportError, AttributeError):
    pass

if not _USE_MEDIAPIPE:
    _haar = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    console.detail("[FaceTrigger] Uso Haar cascade di OpenCV per il rilevamento volti")

import guida_museale_core as guide
from robot_executor import prepare_gesture_backend

RGB_TOPIC = "/camera/camera/color/image_raw"
DEPTH_TOPIC = "/camera/camera/aligned_depth_to_color/image_raw"

PROXEMIC_ZONES_M = {
    "public":   3.6,
    "social":   1.2,
    "personal": 0.45,
}

FACE_TIMER_S = 2.5
MAX_DIST_M   = 2.0
COOLDOWN_S   = 10.0
MOOD_SAMPLE_S = 8.0


def _proxemic_zone(dist_m: float) -> str:
    """Classifica la distanza stimata secondo le zone prossemiche di Hall (1966)."""
    if dist_m > PROXEMIC_ZONES_M["public"]:
        return "outside"
    if dist_m > PROXEMIC_ZONES_M["social"]:
        return "public"
    if dist_m > PROXEMIC_ZONES_M["personal"]:
        return "social"
    return "personal"


def _face_timer_for_zone(zone: str) -> float:
    """
    HRI4-Motion-HRI: un visitatore già in zona personale sta chiaramente
    cercando l'interazione — non ha senso fargli aspettare il timer intero
    pensato per chi è appena entrato in zona sociale.
    """
    return FACE_TIMER_S * 0.5 if zone == "personal" else FACE_TIMER_S



def _detect_faces(bgr: np.ndarray) -> list[tuple[float, float, float, float]]:
    """
    Ritorna lista di (cx_frac, cy_frac, w_frac, h_frac) per ogni faccia trovata,
    dove le coordinate sono frazioni delle dimensioni dell'immagine.
    """
    if _USE_MEDIAPIPE and _mp_det is not None:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        results = _mp_det.process(rgb)
        if not results.detections:
            return []
        out = []
        for det in results.detections:
            bb = det.location_data.relative_bounding_box
            out.append((
                bb.xmin + bb.width / 2,
                bb.ymin + bb.height / 2,
                bb.width,
                bb.height,
            ))
        return out
    else:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        faces = _haar.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
        if not len(faces):
            return []
        h, w = bgr.shape[:2]
        return [
            ((x + fw / 2) / w, (y + fh / 2) / h, fw / w, fh / h)
            for (x, y, fw, fh) in faces
        ]


def _face_distance_from_size(fw_frac: float) -> float:
    """Stima la distanza dalla larghezza relativa della faccia (nessun depth disponibile)."""
    return 0.18 / fw_frac if fw_frac > 0 else 99.0



def run_webcam_mode(input_mode: str, wait_for_enter: bool) -> None:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Impossibile aprire la webcam (device 0).")

    face_first_seen: float | None = None
    cooldown_until: float = 0.0
    recent_frames: list = []
    _conv = {"active": False}

    _mood_buf: list = []
    _mood_job = {"last": 0.0, "busy": False}

    def _sample_mood_during_conversation(bgr, now: float) -> None:
        _mood_buf.append(bgr)
        if len(_mood_buf) > 5:
            _mood_buf.pop(0)
        if (now - _mood_job["last"] < MOOD_SAMPLE_S
                or _mood_job["busy"] or len(_mood_buf) < 3):
            return
        frames = list(_mood_buf)
        _mood_buf.clear()
        _mood_job["last"] = now
        _mood_job["busy"] = True

        def _job():
            try:
                mood, conf = guide.detect_mood_from_frames(frames)
                guide.MOOD_MONITOR.submit(mood, conf)
            finally:
                _mood_job["busy"] = False

        threading.Thread(target=_job, daemon=True).start()

    console.detail(f"\n[FaceTrigger] Webcam attiva — {'MediaPipe' if _USE_MEDIAPIPE else 'Haar cascade'}")
    console.detail(f"  Trigger: faccia ≥{FACE_TIMER_S}s + distanza ≤{MAX_DIST_M}m")
    console.detail(f"  Cooldown: {COOLDOWN_S}s  |  Premi [q] nella finestra per uscire\n")
    if console.is_quiet():
        console.event("\n👀  In attesa di un volto… (premi [q] nella finestra per uscire)\n")

    def _start_conversation():
        frames = list(recent_frames)

        import concurrent.futures as _cf
        _executor = _cf.ThreadPoolExecutor(max_workers=1)
        mood_future = _executor.submit(guide.detect_mood_from_frames, frames) if frames else None

        def _run():
            try:
                guide.run_session(
                    input_mode=input_mode,
                    proactive_greeting=True,
                    wait_for_enter=wait_for_enter,
                    mood_future=mood_future,
                )
            except Exception as exc:
                console.error(f"[FaceTrigger] Errore sessione: {exc}")
            finally:
                _conv["active"] = False
                _conv["cooldown_until"] = time.monotonic() + COOLDOWN_S
                if mood_future is not None:
                    _executor.shutdown(wait=False)
                console.event(f"[FaceTrigger] Sessione terminata. Cooldown {COOLDOWN_S:.0f}s.")

        _conv["active"] = True
        _conv["cooldown_until"] = 0.0
        threading.Thread(target=_run, daemon=True).start()

    _conv["cooldown_until"] = 0.0

    try:
        while True:
            ret, bgr = cap.read()
            if not ret:
                time.sleep(0.033)
                continue

            now = time.monotonic()
            display = bgr.copy()

            if _conv["active"]:
                _sample_mood_during_conversation(bgr, now)
                cv2.putText(display, "CONVERSAZIONE ATTIVA", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 0), 2)
                cv2.imshow("Museum Face Trigger", display)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                face_first_seen = None
                continue

            cooldown_until = _conv["cooldown_until"]
            if now < cooldown_until:
                remaining = cooldown_until - now
                cv2.putText(display, f"Cooldown: {remaining:.0f}s", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 2)
                cv2.imshow("Museum Face Trigger", display)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                face_first_seen = None
                continue

            faces = _detect_faces(bgr)

            if not faces:
                if face_first_seen is not None:
                    console.event("[FaceTrigger] Faccia persa — timer resettato.")
                    recent_frames.clear()
                face_first_seen = None
                cv2.putText(display, "Nessuna faccia", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 2)
                cv2.imshow("Museum Face Trigger", display)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue

            cx_frac, cy_frac, fw_frac, fh_frac = max(faces, key=lambda f: f[2])
            dist = _face_distance_from_size(fw_frac)

            h, w = bgr.shape[:2]
            cx = int(cx_frac * w)
            cy = int(cy_frac * h)
            fw = int(fw_frac * w)
            fh = int(fh_frac * h)

            if dist > MAX_DIST_M:
                cv2.rectangle(display, (cx - fw // 2, cy - fh // 2),
                              (cx + fw // 2, cy + fh // 2), (0, 0, 200), 2)
                cv2.putText(display, f"Troppo lontano (~{dist:.1f}m)", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 200), 2)
                cv2.imshow("Museum Face Trigger", display)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                if face_first_seen is not None:
                    recent_frames.clear()
                face_first_seen = None
                continue

            recent_frames.append(bgr)
            if len(recent_frames) > 10:
                recent_frames.pop(0)

            zone = _proxemic_zone(dist)
            timer_s = _face_timer_for_zone(zone)

            if face_first_seen is None:
                face_first_seen = now
                console.event(f"[FaceTrigger] Faccia rilevata a ~{dist:.1f}m (zona {zone}). Timer avviato ({timer_s:.1f}s).")

            elapsed = now - face_first_seen
            progress = min(elapsed / timer_s, 1.0)

            cv2.rectangle(display, (cx - fw // 2, cy - fh // 2),
                          (cx + fw // 2, cy + fh // 2), (0, 255, 0), 2)
            cv2.putText(display, f"Visitatore ~{dist:.1f}m [{zone}] | {elapsed:.1f}/{timer_s:.1f}s",
                        (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            bar_w = int(progress * (w - 20))
            cv2.rectangle(display, (10, h - 20), (10 + bar_w, h - 8), (0, 255, 0), -1)
            cv2.rectangle(display, (10, h - 20), (w - 10, h - 8), (200, 200, 200), 1)

            cv2.imshow("Museum Face Trigger", display)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            if elapsed >= timer_s:
                face_first_seen = None
                console.event(f"[FaceTrigger] TRIGGER — avvio conversazione (zona {zone})")
                _start_conversation()
                recent_frames.clear()

    finally:
        cap.release()
        cv2.destroyAllWindows()



class FaceTriggerNode:
    """ROS2 node — importa rclpy solo quando necessario."""

    def __init__(self, input_mode: str, wait_for_enter: bool):
        import rclpy
        from rclpy.node import Node
        from sensor_msgs.msg import Image
        from cv_bridge import CvBridge

        class _Node(Node):
            def __init__(self_inner):
                super().__init__("museum_face_trigger")
                self_inner._bridge = CvBridge()
                self_inner._depth_img: np.ndarray | None = None
                self_inner._depth_lock = threading.Lock()
                self_inner._face_first_seen: float | None = None
                self_inner._conv_active = False
                self_inner._cooldown_until: float = 0.0
                self_inner._last_cooldown_log: float = 0.0
                self_inner._input_mode = input_mode
                self_inner._wait_for_enter = wait_for_enter
                self_inner._recent_frames: list = []
                self_inner._mood_buf: list = []
                self_inner._mood_last: float = 0.0
                self_inner._mood_busy: bool = False

                self_inner.create_subscription(
                    Image, RGB_TOPIC, self_inner._rgb_cb, 10)
                self_inner.create_subscription(
                    Image, DEPTH_TOPIC, self_inner._depth_cb, 10)

                self_inner.get_logger().info(
                    f"Museum face trigger pronto su {RGB_TOPIC}  "
                    f"face_timer={FACE_TIMER_S}s  max_dist={MAX_DIST_M}m  cooldown={COOLDOWN_S}s"
                )

            def _depth_cb(self_inner, msg):
                arr = self_inner._bridge.imgmsg_to_cv2(
                    msg, desired_encoding="passthrough").astype(np.float32)
                if msg.encoding in ("16UC1", "16uc1"):
                    arr /= 1000.0
                with self_inner._depth_lock:
                    self_inner._depth_img = arr

            def _sample_mood_during_conv(self_inner, bgr, now: float) -> None:
                """Come nella modalità webcam: ogni MOOD_SAMPLE_S secondi stima
                il mood dagli ultimi frame e lo deposita nel MOOD_MONITOR."""
                self_inner._mood_buf = (self_inner._mood_buf + [bgr])[-5:]
                if (now - self_inner._mood_last < MOOD_SAMPLE_S
                        or self_inner._mood_busy or len(self_inner._mood_buf) < 3):
                    return
                frames = list(self_inner._mood_buf)
                self_inner._mood_buf = []
                self_inner._mood_last = now
                self_inner._mood_busy = True

                def _job():
                    try:
                        mood, conf = guide.detect_mood_from_frames(frames)
                        guide.MOOD_MONITOR.submit(mood, conf)
                    finally:
                        self_inner._mood_busy = False

                threading.Thread(target=_job, daemon=True).start()

            def _rgb_cb(self_inner, msg):
                now = time.monotonic()
                if self_inner._conv_active:
                    bgr = self_inner._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
                    self_inner._sample_mood_during_conv(bgr, now)
                    return
                if now < self_inner._cooldown_until:
                    if now - self_inner._last_cooldown_log >= 1.0:
                        remaining = self_inner._cooldown_until - now
                        self_inner.get_logger().info(f"Cooldown: {remaining:.0f}s")
                        self_inner._last_cooldown_log = now
                    return

                bgr = self_inner._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
                self_inner._recent_frames = (self_inner._recent_frames + [bgr])[-3:]
                faces = _detect_faces(bgr)

                if not faces:
                    if self_inner._face_first_seen is not None:
                        self_inner.get_logger().info("Faccia persa — timer resettato.")
                    self_inner._face_first_seen = None
                    return

                cx_frac, cy_frac, fw_frac, _ = max(faces, key=lambda f: f[2])
                h, w = bgr.shape[:2]
                dist = self_inner._estimate_distance(cx_frac, cy_frac, fw_frac, w, h)

                if dist > MAX_DIST_M:
                    self_inner._face_first_seen = None
                    return

                zone = _proxemic_zone(dist)
                timer_s = _face_timer_for_zone(zone)

                if self_inner._face_first_seen is None:
                    self_inner._face_first_seen = now
                    self_inner.get_logger().info(
                        f"Faccia rilevata a ~{dist:.2f}m (zona {zone}). Timer avviato ({timer_s:.1f}s).")
                    return

                elapsed = now - self_inner._face_first_seen
                if elapsed >= timer_s:
                    self_inner.get_logger().info(f"TRIGGER — avvio conversazione (zona {zone})")
                    self_inner._trigger()

            def _estimate_distance(self_inner, cx_frac, cy_frac, fw_frac, w, h):
                with self_inner._depth_lock:
                    depth = self_inner._depth_img
                if depth is not None:
                    dh, dw = depth.shape[:2]
                    cx = int(cx_frac * dw)
                    cy_d = int(cy_frac * dh)
                    r = max(4, int(fw_frac * dw * 0.2))
                    patch = depth[max(0, cy_d - r): cy_d + r, max(0, cx - r): cx + r]
                    valid = patch[(patch > 0.15) & (patch < 8.0)]
                    if valid.size >= 4:
                        return float(np.median(valid))
                return _face_distance_from_size(fw_frac)

            def _trigger(self_inner):
                self_inner._face_first_seen = None
                self_inner._conv_active = True
                frames = list(self_inner._recent_frames)
                detected_mood, mood_conf = (
                    guide.detect_mood_from_frames(frames) if frames else (None, 0.0)
                )
                self_inner.get_logger().info(
                    f"TRIGGER — avvio conversazione (mood={detected_mood}, conf={mood_conf:.2f})")
                threading.Thread(
                    target=lambda: self_inner._run_session(detected_mood), daemon=True
                ).start()

            def _run_session(self_inner, detected_mood=None):
                try:
                    guide.run_session(
                        input_mode=self_inner._input_mode,
                        proactive_greeting=True,
                        wait_for_enter=self_inner._wait_for_enter,
                        detected_mood=detected_mood,
                    )
                except Exception as exc:
                    self_inner.get_logger().error(f"Errore sessione: {exc}")
                finally:
                    self_inner._conv_active = False
                    self_inner._cooldown_until = time.monotonic() + COOLDOWN_S
                    self_inner._face_first_seen = None
                    self_inner.get_logger().info(
                        f"Sessione terminata. Cooldown {COOLDOWN_S:.0f}s.")

        rclpy.init()
        self._node = _Node()
        self._rclpy = rclpy

    def spin(self):
        try:
            self._rclpy.spin(self._node)
        except KeyboardInterrupt:
            console.event("\nSpegnimento. Arrivederci!")
        finally:
            self._node.destroy_node()
            self._rclpy.shutdown()
            guide.shutdown()



def parse_args():
    p = argparse.ArgumentParser(
        description="Booster-T1 museum guide — face-detection trigger",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--webcam", action="store_true",
                   help="Usa la webcam del PC (nessun ROS2 richiesto). Per test in simulazione.")
    p.add_argument("--mode", choices=["simulation", "robot"], default="simulation",
                   help="[Solo modalità ROS2] simulation: gTTS su casse PC.  robot: kSpeak Booster.")
    p.add_argument("--input-mode", choices=["voice", "text"], default="voice",
                   help="Come acquisire l'input del visitatore.")
    p.add_argument("--mic-device-index", type=int, default=None, metavar="N",
                   help="Indice del microfono (default: microfono di sistema).")
    p.add_argument("--wait-for-enter", action="store_true",
                   help="Richiede INVIO prima di ogni ascolto (debug).")
    p.add_argument("--no-tts", action="store_true",
                   help="Debug: non riprodurre audio, stampa solo le risposte del robot.")
    p.add_argument(
        "--gesture-mode",
        "-gesture-mode",
        choices=["trajectory", "goal"],
        default=os.environ.get("GESTURE_MODE", "trajectory"),
        help=(
            "Sceglie come eseguire i gesti: "
            "trajectory usa i file .hpp, goal usa un singolo target articolare con rampa."
        ),
    )
    p.add_argument(
        "--survey",
        action="store_true",
        help="HRI7-Evaluation: chiedi un breve questionario Godspeed al termine di ogni sessione.",
    )
    p.add_argument(
        "--quiet",
        action="store_true",
        help="Demo/registrazione: stampa solo la traccia interessante (rilevamento volto, "
             "parlato del visitatore, risposta del robot, scelte del robot con confidence) "
             "e silenzia la telemetria tecnica.",
    )
    return p.parse_args()


def main():
    args = parse_args()

    console.set_quiet(args.quiet)
    os.environ["GESTURE_MODE"] = args.gesture_mode
    console.detail(f"[FaceTrigger] Gesture mode: {args.gesture_mode}")

    guide.configure(
        robot_mode=(args.mode == "robot") and not args.webcam,
        mic_device_index=args.mic_device_index,
        silent_tts=args.no_tts,
        quiet=args.quiet,
    )
    guide._SURVEY_ENABLED = args.survey
    guide.init_kb()

    console.detail("🔧 Preparazione gesture backend...")
    prepare_gesture_backend()

    if args.webcam:
        run_webcam_mode(
            input_mode=args.input_mode,
            wait_for_enter=args.wait_for_enter,
        )
    else:
        node = FaceTriggerNode(
            input_mode=args.input_mode,
            wait_for_enter=args.wait_for_enter,
        )
        node.spin()


if __name__ == "__main__":
    main()
