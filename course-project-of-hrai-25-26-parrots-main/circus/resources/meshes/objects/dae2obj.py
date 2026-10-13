#!/usr/bin/env python3
"""
Minimal COLLADA (.dae) to .obj converter.
Exports ALL geometries into a single .obj file.
Applies the unit scale from <asset><unit meter="..."> automatically.

Usage:
    python3 dae2obj.py input.dae output.obj
"""

import sys
import xml.etree.ElementTree as ET

NS = {"c": "http://www.collada.org/2005/11/COLLADASchema"}


def find_text(node, path):
    el = node.find(path, NS)
    return el.text.strip() if el is not None and el.text else ""


def parse_floats(text):
    return [float(x) for x in text.split()]


def parse_ints(text):
    return [int(x) for x in text.split()]


def convert(dae_path: str, obj_path: str):
    tree = ET.parse(dae_path)
    root = tree.getroot()

    # Unit scale (DAE units → metres)
    unit_el = root.find("c:asset/c:unit", NS)
    scale = float(unit_el.attrib.get("meter", "1.0")) if unit_el is not None else 1.0
    print(f"  unit scale: {scale} (1 DAE unit = {scale} m)")

    all_positions = []   # global list of (x,y,z)
    all_normals = []     # global list of (nx,ny,nz)
    all_faces = []       # list of [(global_vi, global_ni or None), ...]

    for geom in root.findall(".//c:library_geometries/c:geometry", NS):
        geom_name = geom.attrib.get("name", geom.attrib.get("id", "?"))
        mesh_el = geom.find("c:mesh", NS)
        if mesh_el is None:
            continue

        # --- Sources ---
        sources = {}
        for src in mesh_el.findall("c:source", NS):
            sid = src.attrib["id"]
            fa = src.find("c:float_array", NS)
            if fa is not None:
                sources[sid] = parse_floats(fa.text)

        # --- Vertices (resolves the <vertices> indirection) ---
        vertices_el = mesh_el.find("c:vertices", NS)
        if vertices_el is None:
            continue
        vert_id = vertices_el.attrib["id"]
        pos_input = vertices_el.find("c:input[@semantic='POSITION']", NS)
        if pos_input is None:
            continue
        pos_source_id = pos_input.attrib["source"].lstrip("#")
        positions_raw = sources.get(pos_source_id, [])

        v_offset = len(all_positions)  # base index for this geometry's vertices
        for i in range(0, len(positions_raw), 3):
            all_positions.append((
                positions_raw[i]     * scale,
                positions_raw[i + 1] * scale,
                positions_raw[i + 2] * scale,
            ))

        # --- Normals ---
        n_offset = len(all_normals)
        normal_source_id = None
        normal_offset_idx = None

        # --- Triangles / Polylist ---
        tris_el = mesh_el.find("c:triangles", NS)
        poly_el = mesh_el.find("c:polylist", NS)
        prim_el = tris_el if tris_el is not None else poly_el
        if prim_el is None:
            continue

        inputs = prim_el.findall("c:input", NS)
        vert_offset_idx = 0
        stride = 1
        for inp in inputs:
            sem = inp.attrib["semantic"]
            off = int(inp.attrib["offset"])
            src = inp.attrib["source"].lstrip("#")
            if sem == "VERTEX":
                vert_offset_idx = off
            elif sem == "NORMAL":
                normal_offset_idx = off
                normal_source_id = src
            stride = max(stride, off + 1)

        if normal_source_id and normal_source_id in sources:
            nr = sources[normal_source_id]
            for i in range(0, len(nr), 3):
                all_normals.append((nr[i], nr[i + 1], nr[i + 2]))

        p_text = find_text(prim_el, "c:p")
        if not p_text:
            continue
        p_data = parse_ints(p_text)

        if prim_el.tag.endswith("triangles"):
            for i in range(0, len(p_data), stride * 3):
                tri = []
                for j in range(3):
                    vi = p_data[i + j * stride + vert_offset_idx] + v_offset
                    ni = (p_data[i + j * stride + normal_offset_idx] + n_offset
                          if normal_offset_idx is not None else None)
                    tri.append((vi, ni))
                all_faces.append(tri)
        else:  # polylist
            vcount_text = find_text(prim_el, "c:vcount")
            vcounts = parse_ints(vcount_text)
            idx = 0
            for vc in vcounts:
                verts = []
                for j in range(vc):
                    vi = p_data[idx + j * stride + vert_offset_idx] + v_offset
                    ni = (p_data[idx + j * stride + normal_offset_idx] + n_offset
                          if normal_offset_idx is not None else None)
                    verts.append((vi, ni))
                for j in range(1, vc - 1):
                    all_faces.append([verts[0], verts[j], verts[j + 1]])
                idx += vc * stride

        print(f"  geometry '{geom_name}': {len(positions_raw)//3} verts, {len(all_faces)} faces so far")

    # --- Write OBJ ---
    with open(obj_path, "w") as f:
        f.write(f"# Converted from {dae_path}\n")
        for v in all_positions:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for n in all_normals:
            f.write(f"vn {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}\n")
        f.write("g mesh\n")
        for tri in all_faces:
            if all_normals and tri[0][1] is not None:
                idxs = " ".join(f"{vi+1}//{ni+1}" for vi, ni in tri)
            else:
                idxs = " ".join(str(vi + 1) for vi, _ in tri)
            f.write(f"f {idxs}\n")

    print(f"  total: {len(all_positions)} vertices, {len(all_faces)} triangles → {obj_path}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} input.dae output.obj")
        sys.exit(1)
    convert(sys.argv[1], sys.argv[2])
