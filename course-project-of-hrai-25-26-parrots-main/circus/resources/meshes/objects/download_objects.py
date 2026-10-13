#!/usr/bin/env python3
"""
Download Google Scanned Objects from Gazebo Fuel and prepare them for MuJoCo.

Each object's model.obj is downloaded and saved as <short_name>.obj in this directory.
An entry may also name the exact mesh to take (models that ship several — poses,
collision proxies) and a texture, which lands in resources/textures/<short_name>.png
ready to be referenced from a scene's `texture:` field.

Usage:
    python3 download_objects.py

Requirements:
    pip install requests
"""

import os
from typing import Optional
from urllib.parse import quote

import requests

FUEL_BASE_GSO      = "https://fuel.gazebosim.org/1.0/GoogleResearch/models"
FUEL_BASE_OPENROBO = "https://fuel.gazebosim.org/1.0/OpenRobotics/models"

# (short_name, fuel_base_url, exact_fuel_model_name[, mesh_path, texture_path])
#
# The two optional fields are for models that ship more than one mesh or that
# come with a texture. Without mesh_path the first .obj/.dae found is taken,
# which is right for single-mesh models and wrong for the visitors below (they
# ship collision proxies and several poses alongside the render mesh).
OBJECTS = [
    # Tableware (Google Scanned Objects)
    ("mug",      FUEL_BASE_GSO,      "Threshold_Porcelain_Coffee_Mug_All_Over_Bead_White"),
    ("bowl",     FUEL_BASE_GSO,      "Threshold_Porcelain_Serving_Bowl_Coupe_White"),
    ("plate",    FUEL_BASE_GSO,      "Threshold_Salad_Plate_Square_Rim_Porcelain"),
    ("teapot",   FUEL_BASE_GSO,      "Threshold_Porcelain_Teapot_White"),
    ("ramekin",  FUEL_BASE_GSO,      "Threshold_Ramekin_White_Porcelain"),
    ("mouse",    FUEL_BASE_GSO,      "Razer_Taipan_White_Ambidextrous_Gaming_Mouse"),
    ("keyboard", FUEL_BASE_GSO,      "Razer_Blackwidow_Tournament_Edition_Keyboard"),
    # Furniture (OpenRobotics)
    ("table",    FUEL_BASE_OPENROBO, "Dining Table"),
    # Museum visitors (OpenRobotics) — native .obj with ONE texture atlas each,
    # which is what MuJoCo needs: it applies a single material per mesh, so a
    # model split across several textures (most of the Fuel human models, which
    # ship separate maps for skin/clothes/eyes) cannot be textured correctly
    # here. Rigged COLLADA models are no good either — dae2obj.py does not apply
    # skeleton transforms and drops texture coordinates.
    ("visitor_female",     FUEL_BASE_OPENROBO, "FemaleVisitor",
     "/meshes/FemaleVisitor.obj", "/meshes/FemaleVisitor.png"),
    ("visitor_male_phone", FUEL_BASE_OPENROBO, "MaleVisitorOnPhone",
     "/meshes/MaleVisitorStatic.obj", "/meshes/MaleVisitor.png"),
]

TEXTURE_DIR = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "textures"))

FUEL_BASE = FUEL_BASE_GSO  # kept for get_mesh_path default

OUT_DIR = os.path.dirname(os.path.abspath(__file__))


def get_mesh_path(base_url: str, model_name: str) -> Optional[str]:
    """Return the path of the first .obj or .dae file in the model, or None."""
    url = f"{base_url}/{model_name}/1/files"
    resp = requests.get(url, timeout=30)
    if resp.status_code != 200:
        print(f"  [warn] Could not fetch file list ({resp.status_code})")
        return None

    def walk(node):
        if "children" in node:
            for c in node["children"]:
                result = walk(c)
                if result:
                    return result
        elif node["path"].endswith(".obj") or node["path"].endswith(".dae"):
            return node["path"]
        return None

    data = resp.json()
    for entry in data.get("file_tree", []):
        p = walk(entry)
        if p:
            return p
    return None


def download_texture(short_name: str, base_url: str, model_name: str, texture_path: str):
    """Fetch the model's texture into resources/textures/<short_name>.png."""
    out = os.path.join(TEXTURE_DIR, f"{short_name}.png")
    if os.path.exists(out):
        print(f"  [skip] textures/{short_name}.png already exists")
        return
    url = f"{base_url}/{model_name}/1/files{quote(texture_path)}"
    r = requests.get(url, timeout=60)
    if r.status_code != 200:
        print(f"  [warn] Texture download failed: {r.status_code}")
        return
    os.makedirs(TEXTURE_DIR, exist_ok=True)
    with open(out, "wb") as f:
        f.write(r.content)
    print(f"  [{short_name}] OK -> {out}")


def download_object(short_name: str, base_url: str, model_name: str,
                    mesh_path: Optional[str] = None, texture_path: Optional[str] = None):
    if texture_path:
        download_texture(short_name, base_url, model_name, texture_path)

    # Determine extension after finding the mesh
    if mesh_path:
        print(f"  [{short_name}] Using {mesh_path} from {model_name}")
    else:
        print(f"  [{short_name}] Finding mesh in {model_name} ...")
        mesh_path = get_mesh_path(base_url, model_name)
    if not mesh_path:
        print(f"  [warn] No mesh found for {model_name}")
        return

    is_dae = mesh_path.endswith(".dae")
    raw_ext = ".dae" if is_dae else ".obj"

    # MuJoCo does not support .dae — download it then convert to .obj
    final_ext = ".obj"
    final_path = os.path.join(OUT_DIR, f"{short_name}{final_ext}")
    if os.path.exists(final_path):
        print(f"  [skip] {short_name}{final_ext} already exists")
        return

    url = f"{base_url}/{model_name}/1/files{quote(mesh_path)}"
    print(f"  [{short_name}] Downloading {url} ...")
    r = requests.get(url, timeout=60)
    if r.status_code != 200:
        print(f"  [warn] Download failed: {r.status_code}")
        return

    if is_dae:
        # Save .dae temporarily, convert to .obj, remove .dae
        dae_path = os.path.join(OUT_DIR, f"{short_name}.dae")
        with open(dae_path, "wb") as f:
            f.write(r.content)
        from dae2obj import convert as dae2obj_convert
        dae2obj_convert(dae_path, final_path)
        os.remove(dae_path)
        print(f"  [{short_name}] OK -> {final_path}")
    else:
        with open(final_path, "wb") as f:
            f.write(r.content)
        print(f"  [{short_name}] OK -> {final_path}")


if __name__ == "__main__":
    print(f"Output dir: {OUT_DIR}\n")
    for entry in OBJECTS:
        # 3-tuples (the original entries) and 5-tuples both accepted.
        download_object(*entry[:3], *(entry[3:5] or ()))
    print("\nDone.")
