#!/usr/bin/env python3
"""
Scene randomizer — "the curator rehangs the gallery".

Picks 6 random paintings from the museum catalog (exchange/data/paintings.yaml)
and generates the 6 wall textures (circus/resources/textures/wall_p1..6.png)
referenced by circus/resources/scenes/museo_receptionist.yaml, letterboxing
each artwork to its canvas aspect ratio so nothing is distorted.

It also:
  - writes exchange/data/scene_ground_truth.yaml — the curator's registry of
    what is hung where. The robot does NOT receive this as its perception: it
    is the ground truth used by knowledge_graph.py to ground the VLM scan
    (HRAI 8 — a VLM hallucination becomes an explicit perceptionMismatch)
    and as a fallback placement when no scan has been run yet.
  - deletes the previous scan results (discovered_paintings.json,
    discovered_scene_map.yaml): they describe the OLD hanging and would
    poison the knowledge graph of the new one.

Run BEFORE starting the simulator (circus loads the textures at scene load):

    pixi run python ../exchange/museum_guide_src/scene_randomizer.py [--seed N]

then `docker compose up` and re-run museum_explorer.py so the robot discovers
the new hanging with its own camera.
"""

import argparse
import random
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
_TEXTURES_DIR = _REPO_ROOT / "circus" / "resources" / "textures"
_DATA_DIR = _REPO_ROOT / "exchange" / "data"
_GROUND_TRUTH = _DATA_DIR / "scene_ground_truth.yaml"
_STALE_SCAN_FILES = (
    _DATA_DIR / "discovered_paintings.json",
    _DATA_DIR / "discovered_scene_map.yaml",
)

_PAD_COLOR = (38, 34, 30)

_PLAQUE_SIZE = (600, 200)
_PLAQUE_BG = (18, 18, 20)
_PLAQUE_BORDER = (40, 95, 120)
_PLAQUE_TITLE_COLOR = (230, 230, 235)
_PLAQUE_ARTIST_COLOR = (175, 175, 180)

POSITIONS = [
    {"position": "p1", "wall": "entrance_wall_left",  "gesture": "point_front_left",
     "texture_out": "wall_p1.png", "canvas_w": 1.200, "canvas_h": 1.200},
    {"position": "p2", "wall": "entrance_wall_right", "gesture": "point_front_right",
     "texture_out": "wall_p2.png", "canvas_w": 1.200, "canvas_h": 1.200},
    {"position": "p3", "wall": "right_side_wall",     "gesture": "point_right",
     "texture_out": "wall_p3.png", "canvas_w": 1.200, "canvas_h": 1.200},
    {"position": "p4", "wall": "left_side_wall",      "gesture": "point_left",
     "texture_out": "wall_p4.png", "canvas_w": 1.200, "canvas_h": 1.200},
    {"position": "p5", "wall": "right_side_wall_mid", "gesture": "point_right_mid",
     "texture_out": "wall_p5.png", "canvas_w": 1.200, "canvas_h": 1.200},
    {"position": "p6", "wall": "left_side_wall_mid",  "gesture": "point_left_mid",
     "texture_out": "wall_p6.png", "canvas_w": 1.200, "canvas_h": 1.200},
]


def _letterbox(img: np.ndarray, target_aspect: float) -> np.ndarray:
    """Pad img with a neutral border so its aspect ratio (w/h) matches the
    canvas mesh, keeping the artwork undistorted."""
    h, w = img.shape[:2]
    aspect = w / h
    if abs(aspect - target_aspect) < 1e-3:
        return img
    if aspect > target_aspect:
        new_h = int(round(w / target_aspect))
        pad = new_h - h
        top, bottom = pad // 2, pad - pad // 2
        return cv2.copyMakeBorder(img, top, bottom, 0, 0,
                                  cv2.BORDER_CONSTANT, value=_PAD_COLOR)
    new_w = int(round(h * target_aspect))
    pad = new_w - w
    left, right = pad // 2, pad - pad // 2
    return cv2.copyMakeBorder(img, 0, 0, left, right,
                              cv2.BORDER_CONSTANT, value=_PAD_COLOR)


def _fit_text(text: str, max_width: int, base_scale: float, thickness: int,
               font=cv2.FONT_HERSHEY_SIMPLEX) -> tuple[float, int, int]:
    """Largest font scale (down from base_scale) that keeps text within max_width."""
    scale = base_scale
    size, _ = cv2.getTextSize(text, font, scale, thickness)
    while size[0] > max_width and scale > 0.3:
        scale -= 0.05
        size, _ = cv2.getTextSize(text, font, scale, thickness)
    return scale, size[0], size[1]


def make_plaque(title: str, artist: str) -> np.ndarray:
    """Small dark museum-label texture: title + artist, centered, autofit."""
    w, h = _PLAQUE_SIZE
    img = np.full((h, w, 3), _PLAQUE_BG, dtype=np.uint8)
    cv2.rectangle(img, (4, 4), (w - 5, h - 5), _PLAQUE_BORDER, 2)

    margin = 28
    font = cv2.FONT_HERSHEY_SIMPLEX
    title_scale, title_w, title_h = _fit_text(title, w - 2 * margin, 1.05, 2, font)
    artist_scale, artist_w, artist_h = _fit_text(artist, w - 2 * margin, 0.7, 1, font)

    title_y = h // 2 - 10
    cv2.putText(img, title, ((w - title_w) // 2, title_y), font, title_scale,
                _PLAQUE_TITLE_COLOR, 2, cv2.LINE_AA)
    artist_y = title_y + title_h + 30
    cv2.putText(img, artist, ((w - artist_w) // 2, artist_y), font, artist_scale,
                _PLAQUE_ARTIST_COLOR, 1, cv2.LINE_AA)
    return img


def load_catalog() -> list[dict]:
    return yaml.safe_load((_DATA_DIR / "paintings.yaml").read_text())["paintings"]


def randomize_scene(seed: int | None = None) -> dict:
    rng = random.Random(seed)
    catalog = load_catalog()

    usable = [p for p in catalog if (_TEXTURES_DIR / p["texture"]).exists()]
    missing = [p["texture"] for p in catalog if p not in usable]
    if missing:
        print(f"[Randomizer] ⚠️  Texture mancanti (quadri esclusi): {missing}")
    if len(usable) < len(POSITIONS):
        raise RuntimeError(
            f"Servono almeno {len(POSITIONS)} texture, trovate {len(usable)}."
        )

    chosen = rng.sample(usable, len(POSITIONS))

    hanging = []
    for pos, painting in zip(POSITIONS, chosen):
        src = _TEXTURES_DIR / painting["texture"]
        img = cv2.imread(str(src), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"Impossibile leggere la texture {src}")
        out = _letterbox(img, pos["canvas_w"] / pos["canvas_h"])
        out_path = _TEXTURES_DIR / pos["texture_out"]
        cv2.imwrite(str(out_path), out)

        plaque = make_plaque(painting["title"], painting["artist"])
        plaque_path = _TEXTURES_DIR / f"plaque_{pos['position']}.png"
        cv2.imwrite(str(plaque_path), plaque)

        hanging.append({
            "position": pos["position"],
            "wall": pos["wall"],
            "gesture": pos["gesture"],
            "painting_id": painting["id"],
            "title": painting["title"],
            "artist": painting["artist"],
        })
        print(f"[Randomizer] {pos['position']} ({pos['gesture']:18s}) ← "
              f'"{painting["title"]}" — {painting["artist"]}  → {pos["texture_out"]}')

    ground_truth = {
        "scene_ground_truth": {
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "seed": seed,
            "hanging": hanging,
        }
    }
    with open(_GROUND_TRUTH, "w", encoding="utf-8") as f:
        yaml.safe_dump(ground_truth, f, sort_keys=False, allow_unicode=True)
    print(f"[Randomizer] Registro curatore salvato → {_GROUND_TRUTH}")

    for stale in _STALE_SCAN_FILES:
        if stale.exists():
            stale.unlink()
            print(f"[Randomizer] Scan precedente invalidato: rimosso {stale.name}")

    return ground_truth


def main() -> None:
    parser = argparse.ArgumentParser(description="Randomize the museum hanging")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for a reproducible hanging")
    args = parser.parse_args()

    print("=" * 60)
    print("Scene Randomizer — nuovo allestimento del museo")
    print("=" * 60)
    randomize_scene(args.seed)
    print()
    print("Prossimi passi:")
    print("  1. (Ri)avvia il simulatore: docker compose up  (la scena carica le nuove texture)")
    print("  2. Fai riscoprire la stanza al robot: pixi run python ../exchange/museum_guide_src/museum_explorer.py")


if __name__ == "__main__":
    main()
