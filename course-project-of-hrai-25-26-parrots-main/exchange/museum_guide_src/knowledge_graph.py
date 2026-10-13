#!/usr/bin/env python3
"""
RDF/OWL Knowledge Graph for the museum guide robot.

Maps to course slides as follows:
  HRI6 Sl. 4-8   Knowledge Representation (propositional/predicate logic,
                  semantic networks) → RDF triples, museo: OWL ontology
  HRI6 Sl. 9-11  STRIPS/HTN planning  → GuidanceHTN.decompose() task decomposition
  HRI6 Sl. 14-16 Epistemic/Human-aware planning → museo:knows / museo:notKnows,
                  SPARQL epistemic-goal query
  HRI5 Sl. 8-10  Scene Graphs / semantic affordances → museo:locatedAt,
                  museo:affordsPointTo, Room->Wall->Painting hierarchy + leftOf/rightOf
  HRAI 4         Geometric2Semantic → scene_semantic_map.yaml (geometric poses,
                  wall/relation data) lifted into semantic graph edges
  HRAI 9         Self-awareness feasibility score → see perception_coverage()
                  and guida_museale_core.check_feasibility()
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
import console
from rdflib import Graph, Literal, Namespace, RDF, RDFS, URIRef
from rdflib.namespace import OWL, XSD

MUSEO   = Namespace("http://museum.robot/onto#")
PAINTER = Namespace("http://museum.robot/painting/")
VISITOR = Namespace("http://museum.robot/visitor/")
ROOM    = Namespace("http://museum.robot/room/")
WALL    = Namespace("http://museum.robot/wall/")

VISITOR_URI: URIRef = VISITOR.CurrentVisitor

ROBOT_URI: URIRef = MUSEO.Robot

WORLD_FRAME_URI: URIRef = MUSEO.WorldFrame

_G: Graph | None = None

_YAML_STYLE_TO_MOVEMENT: dict[str, str] = {
    "impressionism": "Impressionism",
    "baroque":       "Baroque",
    "romanticism":   "Romanticism",
    "expressionism": "Expressionism",
    "neoclassicism": "Neoclassicism",
    "realism":       "Realism",
    "symbolism":     "Symbolism",
}

LEGACY_GESTURE_TO_PAINTING_ID: dict[str, str] = {
    "point_front_left":  "p1",
    "point_front_right": "p2",
    "point_right":       "p3",
    "point_left":        "p4",
    "point_right_mid":   "p5",
    "point_left_mid":    "p6",
}

SCAN_YAW_RAD: dict[str, float] = {
    "point_left_mid":    0.710,
    "point_left":        0.550,
    "point_front_left":  0.280,
    "point_front_right": -0.280,
    "point_right":       -0.550,
    "point_right_mid":   -0.710,
}

SLOT_BEARING_RAD: dict[str, float] = {
    "point_left":        1.131,
    "point_left_mid":    0.757,
    "point_front_left":  0.373,
    "point_front_right": -0.373,
    "point_right_mid":   -0.757,
    "point_right":       -1.131,
}

_WALL_BOUNDARY_YAW = 0.4


def _load_registry(data_dir: Path) -> dict[str, str]:
    """Curator registry: gesture → painting_id, or {} if no randomizer run yet."""
    path = data_dir / "scene_ground_truth.yaml"
    if not path.exists():
        return {}
    hanging = yaml.safe_load(path.read_text())["scene_ground_truth"]["hanging"]
    return {h["gesture"]: h["painting_id"] for h in hanging}


def _load_scan(data_dir: Path) -> dict[str, str]:
    """Robot scan: gesture → recognized painting_id (failed directions omitted)."""
    path = data_dir / "discovered_paintings.json"
    if not path.exists():
        return {}
    scan: dict[str, str] = {}
    for gesture, info in json.loads(path.read_text()).items():
        if info.get("error"):
            continue
        pid = info.get("painting_id") or LEGACY_GESTURE_TO_PAINTING_ID.get(gesture)
        if pid:
            scan[gesture] = pid
    return scan


def _load_scan_xyz(data_dir: Path) -> dict[str, list[float]]:
    """
    Robot scan: gesture -> 3D position, expressed in R (the world frame — see
    WORLD_FRAME_URI), from the same discovered_paintings.json written by
    museum_explorer.py's depth-based back-projection (camera_projection
    .backproject() + to_world_frame()). Directions where localization failed
    (no depth reading) simply have no "xyz" key. Callers that need a
    base-relative point (e.g. IK) must convert via camera_projection
    .to_base_frame() against the current pose first — see benchmark_planning
    .py and guida_museale_core.handle_recommend for the two call sites.

    Keyed by gesture (a purely geometric fact — where the depth camera looked
    — independent of what the VLM thought it saw), NOT by the scan's own
    painting_id: on a VLM hallucination (see _merge_scan_with_registry) the
    registry's painting_id wins and the scan's is discarded, so indexing by
    the scan's painting_id would silently strand the xyz on an id nothing
    resolves to. The caller joins this with the registry-corrected
    gesture->painting_id mapping (resolve_placement) instead.
    """
    path = data_dir / "discovered_paintings.json"
    if not path.exists():
        return {}
    xyz_by_gesture: dict[str, list[float]] = {}
    for gesture, info in json.loads(path.read_text()).items():
        xyz = info.get("xyz")
        if xyz:
            xyz_by_gesture[gesture] = xyz
    return xyz_by_gesture


def _merge_scan_with_registry(
    scan: dict[str, str], registry: dict[str, str]
) -> tuple[dict[str, str], dict[str, dict]]:
    """
    HRAI 8 - grounding: the registry is trusted prior knowledge, the scan is
    perception. Where they disagree the VLM hallucinated: the placement keeps
    the registry entry and the disagreement is reported as a mismatch (which
    build_graph turns into an explicit museo:perceptionMismatch triple).
    """
    if not registry:
        return dict(scan), {}
    placement = dict(registry)
    mismatches: dict[str, dict] = {}
    for gesture, pid in scan.items():
        expected = registry.get(gesture)
        if expected is not None and expected != pid:
            mismatches[gesture] = {"observed": pid, "expected": expected}
    return placement, mismatches


def resolve_placement(data_dir: Path) -> tuple[dict[str, str], dict[str, dict], str]:
    """
    Returns (placement, mismatches, source):
      placement  — gesture → painting_id of what is (believed to be) hung where
      mismatches — gesture → {observed, expected} where scan contradicts registry
      source     — human-readable provenance for logging
    """
    registry = _load_registry(data_dir)
    scan = _load_scan(data_dir)

    if scan and registry:
        placement, mismatches = _merge_scan_with_registry(scan, registry)
        return placement, mismatches, "robot scan grounded on curator registry"
    if scan:
        return dict(scan), {}, "robot scan (no registry)"
    if registry:
        return dict(registry), {}, "curator registry (no scan yet)"
    return dict(LEGACY_GESTURE_TO_PAINTING_ID), {}, "legacy static mapping"


def spatial_map_from_placement(placement: dict[str, str]) -> dict:
    """
    HRAI 4/5: derive the Room→Wall→Painting scene graph (plus leftOf/rightOf)
    from a gesture→painting placement, using the slot geometry: sorting
    directions by their true bearing from the robot's home pose (descending,
    positive = robot's left) gives the true left-to-right order; adjacency in
    that order gives the spatial edges.

    The bearing comes from SLOT_BEARING_RAD, not from the commanded head yaw
    (SCAN_YAW_RAD): the head is deliberately under-turned at the side-wall
    slots, so its targets are not monotone with the real left-to-right order
    there — see SLOT_BEARING_RAD.
    """
    scanned = [
        {"gesture": g, "painting_id": pid, "yaw": SLOT_BEARING_RAD[g]}
        for g, pid in placement.items() if g in SLOT_BEARING_RAD
    ]
    scanned.sort(key=lambda s: s["yaw"], reverse=True)

    walls: list[dict] = []
    wall_of: dict[str, str] = {}
    for s in scanned:
        if abs(s["yaw"]) >= _WALL_BOUNDARY_YAW:
            wall_id = "left_wall" if s["yaw"] > 0 else "right_wall"
        else:
            wall_id = "front_wall"
        wall_of[s["painting_id"]] = wall_id
        if not any(w["id"] == wall_id for w in walls):
            walls.append({"id": wall_id, "description": f"wall at yaw≈{s['yaw']:+.2f} rad"})

    objects = []
    for i, s in enumerate(scanned):
        relations = {"visible_from_robot": True, "visible_from_visitor": True}
        if i > 0:
            relations["right_of"] = scanned[i - 1]["painting_id"]
        if i < len(scanned) - 1:
            relations["left_of"] = scanned[i + 1]["painting_id"]
        objects.append({
            "id": s["painting_id"],
            "type": "painting",
            "wall": wall_of[s["painting_id"]],
            "relations": relations,
        })

    return {
        "scene": {
            "id": "museum_room_1",
            "name": "Museum Room (derived from placement)",
            "walls": walls,
            "objects": objects,
        }
    }


def _define_ontology(g: Graph) -> None:
    for cls in (MUSEO.Painting, MUSEO.Visitor, MUSEO.Robot, MUSEO.Preference):
        g.add((cls, RDF.type, OWL.Class))

    for prop in (
        MUSEO.hasTitle, MUSEO.hasArtist, MUSEO.hasYear, MUSEO.hasMovement,
        MUSEO.hasTheme, MUSEO.hasMood, MUSEO.suitableFor, MUSEO.difficulty,
        MUSEO.locatedAt, MUSEO.affordsPointTo,
        MUSEO.hasProfile, MUSEO.hasInterest,
        MUSEO.availableTime, MUSEO.preferredMood,
        MUSEO.perceivedMovement, MUSEO.perceivedMood, MUSEO.perceivedTheme,
        MUSEO.perceptionMismatch,
        MUSEO.preferenceTopic, MUSEO.preferenceValue,
        MUSEO.preferenceReason,
    ):
        g.add((prop, RDF.type, OWL.DatatypeProperty))

    for prop in (
        MUSEO.knows, MUSEO.notKnows,
        MUSEO.hasPreference,
    ):
        g.add((prop, RDF.type, OWL.ObjectProperty))


def _define_scene_graph_ontology(g: Graph) -> None:
    for cls in (MUSEO.Room, MUSEO.Wall, MUSEO.ReferenceFrame):
        g.add((cls, RDF.type, OWL.Class))
    for prop in (MUSEO.hasWall, MUSEO.locatedOnWall, MUSEO.leftOf, MUSEO.rightOf,
                 MUSEO.hasReferenceFrame):
        g.add((prop, RDF.type, OWL.ObjectProperty))


def _add_paintings(g: Graph, paintings: list[dict], gesture_by_pid: dict[str, str]) -> None:
    """Add the paintings currently hung in the room. The pointing affordance
    comes from the resolved placement (which gesture aims at the painting's
    position), not from the catalog: it depends on where the randomizer hung
    the painting, not on the painting itself."""
    for p in paintings:
        uri = PAINTER[p["id"]]
        g.add((uri, RDF.type,          MUSEO.Painting))
        g.add((uri, MUSEO.hasTitle,    Literal(p["title"])))
        g.add((uri, MUSEO.hasArtist,   Literal(p["artist"])))
        g.add((uri, MUSEO.hasYear,     Literal(p.get("year", 0), datatype=XSD.integer)))
        movement = _YAML_STYLE_TO_MOVEMENT.get(
            p.get("style", "").lower(), p.get("style", "Unknown")
        )
        g.add((uri, MUSEO.hasMovement, Literal(movement)))
        g.add((uri, MUSEO.difficulty,  Literal(p.get("difficulty", "medium"))))

        gesture = gesture_by_pid.get(p["id"], "")
        if gesture:
            g.add((uri, MUSEO.affordsPointTo, Literal(gesture)))

        for theme in p.get("themes", []):
            g.add((uri, MUSEO.hasTheme, Literal(theme)))
        for mood in p.get("mood", []):
            g.add((uri, MUSEO.hasMood, Literal(mood)))
        for profile in p.get("suitable_for", []):
            g.add((uri, MUSEO.suitableFor, Literal(profile)))


def _add_mismatch_triples(
    g: Graph, mismatches: dict[str, dict], catalog_by_id: dict[str, dict]
) -> None:
    """
    HRAI 8 - Commonsense grounding: a VLM used as a perception source can
    hallucinate. Where the scan contradicts the curator registry, the wrong
    perception becomes an explicit, auditable museo:perceptionMismatch triple
    on the painting actually hung there — and its enrichment is skipped, so
    perception_coverage() correctly reports 0.0 (feeding the HRAI 9 score).
    """
    for gesture, mm in mismatches.items():
        expected_id = mm["expected"]
        observed = catalog_by_id.get(mm["observed"], {}).get("title", mm["observed"])
        expected = catalog_by_id.get(expected_id, {}).get("title", expected_id)
        g.add((
            PAINTER[expected_id], MUSEO.perceptionMismatch,
            Literal(f'VLM said "{observed}", registry says "{expected}"'),
        ))
        console.warn(
            f'[KG] ⚠️  VLM hallucination at {gesture}: registry says "{expected}", '
            f'scan saw "{observed}" — keeping registry, skipping enrichment.'
        )


def _add_vlm_enrichment(
    g: Graph, discovered: dict, mismatched_gestures: set[str]
) -> None:
    """Merge VLM-perceived attributes (movement/mood/themes) for the gestures
    whose recognition was NOT contradicted by the registry. Failed directions
    ("error") mean "not perceived", not "perceived wrong": silently skipped."""
    for gesture_id, info in discovered.items():
        if info.get("error") or gesture_id in mismatched_gestures:
            continue
        pid = info.get("painting_id") or LEGACY_GESTURE_TO_PAINTING_ID.get(gesture_id)
        if pid is None:
            continue
        uri = PAINTER[pid]
        if movement := info.get("movement"):
            g.add((uri, MUSEO.perceivedMovement, Literal(movement)))
        for mood in info.get("mood", []):
            g.add((uri, MUSEO.perceivedMood, Literal(mood)))
        for theme in info.get("themes", []):
            g.add((uri, MUSEO.perceivedTheme, Literal(theme)))


def _add_scene_graph(g: Graph, scene_map: dict) -> None:
    """
    Build the Room -> Wall -> Painting hierarchy plus leftOf/rightOf edges
    from scene_semantic_map.yaml.

    HRAI 4 - Geometric2Semantic: the yaml holds geometric poses (x, y, z, yaw)
    for the room, walls and objects — here we lift only the *semantic* part
    (which wall, which object is next to which) into graph edges, the same
    geometric->semantic step the slides describe for point clouds / meshes.

    HRAI 5 - Scene Graphs: this is what turns museo:locatedAt (a flat string,
    kept for backward compatibility in _add_paintings) into an actual scene
    graph — typed nodes (Room, Wall, Painting) connected by typed edges
    (hasWall, locatedOnWall, leftOf, rightOf), in the spirit of Hydra /
    3D-Scene-Graph's building -> room -> wall -> object hierarchy.
    """
    scene = scene_map.get("scene", {})

    room_uri = ROOM[scene.get("id", "room")]
    g.add((room_uri, RDF.type, MUSEO.Room))

    g.add((WORLD_FRAME_URI, RDF.type, MUSEO.ReferenceFrame))
    g.add((WORLD_FRAME_URI, RDFS.comment, Literal(
        "MuJoCo ground-truth world frame ('world'), exposed live via "
        "shm_camera_reader.read_latest_pose() (circus/include/robots/"
        "BoosterT1.h::update()). Convention: planar (x, y) ground plane, "
        "+z up, yaw about z. Fallback anchor when no live pose is "
        "available: museum_explorer.WORLD_FRAME_ORIGIN_POSE."
    )))
    g.add((room_uri, MUSEO.hasReferenceFrame, WORLD_FRAME_URI))

    wall_uris: dict[str, URIRef] = {}
    for wall in scene.get("walls", []):
        wall_uri = WALL[wall["id"]]
        wall_uris[wall["id"]] = wall_uri
        g.add((wall_uri, RDF.type, MUSEO.Wall))
        g.add((room_uri, MUSEO.hasWall, wall_uri))

    objects = [o for o in scene.get("objects", []) if o.get("type") == "painting"]

    for obj in objects:
        wall_uri = wall_uris.get(obj.get("wall"))
        if wall_uri is not None:
            g.add((PAINTER[obj["id"]], MUSEO.locatedOnWall, wall_uri))

    for obj in objects:
        painting_uri = PAINTER[obj["id"]]
        relations = obj.get("relations", {})
        if left_id := relations.get("left_of"):
            g.add((painting_uri, MUSEO.leftOf, PAINTER[left_id]))
        if right_id := relations.get("right_of"):
            g.add((painting_uri, MUSEO.rightOf, PAINTER[right_id]))


def _init_visitor(g: Graph, paintings: list[dict]) -> None:
    """
    Create the CurrentVisitor node.

    HRI6 Sl. 14-16: all paintings start in notKnows — the robot's epistemic goal
    is to move them to knows by explaining them to the visitor.
    """
    g.add((VISITOR_URI, RDF.type,           MUSEO.Visitor))
    g.add((VISITOR_URI, MUSEO.hasProfile,    Literal("unknown")))
    g.add((VISITOR_URI, MUSEO.availableTime, Literal("unknown")))
    g.add((VISITOR_URI, MUSEO.preferredMood, Literal("unknown")))
    for p in paintings:
        g.add((VISITOR_URI, MUSEO.notKnows, PAINTER[p["id"]]))


def _load_paintings(data_dir: Path) -> list[dict]:
    return yaml.safe_load((data_dir / "paintings.yaml").read_text())["paintings"]


def _new_base_graph(present_paintings: list[dict], gesture_by_pid: dict[str, str]) -> Graph:
    """Namespaces + TBox + ABox of the paintings currently hung + fresh
    visitor — everything common to build_graph() (file-based) and
    build_graph_from_scan() (in-memory). Only PRESENT paintings enter the
    graph: the rest of the catalog stays in storage, like a real museum."""
    g = Graph()
    g.bind("museo",   MUSEO)
    g.bind("painter", PAINTER)
    g.bind("visitor", VISITOR)
    g.bind("room",    ROOM)
    g.bind("wall",    WALL)

    _define_ontology(g)
    _define_scene_graph_ontology(g)
    _add_paintings(g, present_paintings, gesture_by_pid)
    _init_visitor(g, present_paintings)
    return g


def build_graph(
    data_dir: Path,
    discovered_path: Path | None = None,
) -> Graph:
    """Build a fresh RDF graph: catalog + resolved placement + optional
    VLM-discovered data, with scan-vs-registry grounding."""
    catalog = _load_paintings(data_dir)
    catalog_by_id = {p["id"]: p for p in catalog}

    placement, mismatches, source = resolve_placement(data_dir)
    gesture_by_pid = {pid: g_ for g_, pid in placement.items()}
    present = [catalog_by_id[pid] for pid in placement.values() if pid in catalog_by_id]
    console.detail(f"[KG] Placement: {source} — "
                   + ", ".join(f'{g_}→"{catalog_by_id.get(pid, {}).get("title", pid)}"'
                               for g_, pid in sorted(placement.items())))

    g = _new_base_graph(present, gesture_by_pid)
    _add_mismatch_triples(g, mismatches, catalog_by_id)

    discovered_scene_map_path = data_dir / "discovered_scene_map.yaml"
    static_scene_map_path = data_dir / "scene_semantic_map.yaml"
    if discovered_scene_map_path.exists():
        scene_map = yaml.safe_load(discovered_scene_map_path.read_text())
        scene_src = "robot scan"
    elif placement:
        scene_map = spatial_map_from_placement(placement)
        scene_src = "derived from placement"
    elif static_scene_map_path.exists():
        scene_map = yaml.safe_load(static_scene_map_path.read_text())
        scene_src = "static file"
    else:
        scene_map = None
        scene_src = "none"
    if scene_map:
        _add_scene_graph(g, scene_map)
        n_objects = len(scene_map.get("scene", {}).get("objects", []))
        console.detail(f"[KG] Scene graph: {scene_src} ({n_objects} objects)")

    if discovered_path is None:
        discovered_path = data_dir / "discovered_paintings.json"
    if discovered_path.exists():
        discovered = json.loads(discovered_path.read_text())
        _add_vlm_enrichment(g, discovered, set(mismatches))
        console.detail(f"[KG] VLM enrichment loaded ({len(discovered)} directions observed by robot camera)")

    console.detail(f"[KG] Graph built: {len(g)} triples")
    return g


def build_graph_from_scan(
    data_dir: Path,
    discovered: dict,
    scene_map: dict | None = None,
) -> Graph:
    """
    In-memory variant of build_graph() for the live scan viewer: the placement
    comes from the scan in progress (museum_explorer.explore_museum), grounded
    against the curator registry if one exists, so a hallucination shows up
    live as a mismatch badge. Rebuilt after every scanned direction: the graph
    visibly grows painting by painting while the robot moves its head.
    """
    catalog = _load_paintings(data_dir)
    catalog_by_id = {p["id"]: p for p in catalog}

    scan = {}
    for gesture, info in (discovered or {}).items():
        if info.get("error"):
            continue
        pid = info.get("painting_id") or LEGACY_GESTURE_TO_PAINTING_ID.get(gesture)
        if pid:
            scan[gesture] = pid

    registry = _load_registry(data_dir)
    partial_registry = {g_: pid for g_, pid in registry.items() if g_ in scan}
    placement, mismatches = _merge_scan_with_registry(scan, partial_registry)

    gesture_by_pid = {pid: g_ for g_, pid in placement.items()}
    present = [catalog_by_id[pid] for pid in placement.values() if pid in catalog_by_id]

    g = _new_base_graph(present, gesture_by_pid)
    _add_mismatch_triples(g, mismatches, catalog_by_id)
    if scene_map is None and placement:
        scene_map = spatial_map_from_placement(placement)
    if scene_map:
        _add_scene_graph(g, scene_map)
    if discovered:
        _add_vlm_enrichment(g, discovered, set(mismatches))
    return g


_PFX = """
PREFIX museo:   <http://museum.robot/onto#>
PREFIX painter: <http://museum.robot/painting/>
PREFIX visitor: <http://museum.robot/visitor/>
PREFIX rdf:     <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX xsd:     <http://www.w3.org/2001/XMLSchema#>
"""

QUERY_EPISTEMIC_GOAL = _PFX + """
SELECT DISTINCT ?painting ?title WHERE {
    ?painting rdf:type      museo:Painting .
    ?painting museo:hasTitle ?title .
    ?painting museo:affordsPointTo ?gesture .
    visitor:CurrentVisitor museo:hasInterest ?theme .
    ?painting museo:hasTheme ?theme .
    FILTER NOT EXISTS { visitor:CurrentVisitor museo:knows ?painting }
}
"""

_QUERY_CANDIDATES = _PFX + """
SELECT DISTINCT ?painting ?title ?artist ?gesture ?difficulty WHERE {
    ?painting rdf:type         museo:Painting .
    ?painting museo:hasTitle   ?title .
    ?painting museo:hasArtist  ?artist .
    ?painting museo:affordsPointTo ?gesture .
    ?painting museo:difficulty ?difficulty .
    FILTER NOT EXISTS { visitor:CurrentVisitor museo:knows ?painting }
    FILTER NOT EXISTS { ?painting museo:underMaintenance ?anyMaintenance }
}
"""

_QUERY_THEME_MATCHES = _PFX + """
SELECT ?painting (COUNT(DISTINCT ?theme) AS ?count) WHERE {
    ?painting rdf:type      museo:Painting .
    ?painting museo:hasTheme ?theme .
    visitor:CurrentVisitor museo:hasInterest ?theme .
    FILTER NOT EXISTS { visitor:CurrentVisitor museo:knows ?painting }
} GROUP BY ?painting
"""

_QUERY_PROFILE_MATCHES = _PFX + """
SELECT ?painting WHERE {
    ?painting rdf:type          museo:Painting .
    ?painting museo:suitableFor ?profile .
    visitor:CurrentVisitor museo:hasProfile ?profile .
    FILTER NOT EXISTS { visitor:CurrentVisitor museo:knows ?painting }
}
"""

_QUERY_MOOD_MATCHES = _PFX + """
SELECT ?painting WHERE {
    ?painting rdf:type      museo:Painting .
    ?painting museo:hasMood ?mood .
    visitor:CurrentVisitor museo:preferredMood ?mood .
    FILTER NOT EXISTS { visitor:CurrentVisitor museo:knows ?painting }
}
"""


def sparql_recommend(g: Graph, yaml_paintings: list[dict]) -> tuple[dict | None, dict[str, int]]:
    """
    Select the best next painting for the current visitor via SPARQL queries.

    Replaces the flat Python loop in score_painting() + recommend():
      - SPARQL handles semantic matching and epistemic filtering (notKnows)
      - Python applies the weighted score on the filtered result set

    Returns (best painting's full YAML dict, or None if all shown; candidate_scores
    — {painting_id: weighted score} for every eligible candidate, the SAME scores
    already computed to pick the winner, just no longer discarded). candidate_scores
    powers benchmark_task.py's "recommendation optimality ratio" metric — how the
    chosen painting's score compares to the best available one, not just whether
    the chosen one passed some scenario-specific check.
    """
    id_to_yaml: dict[str, dict] = {PAINTER[p["id"]]: p for p in yaml_paintings}

    candidates: dict[str, dict] = {
        str(row.painting): {
            "uri":     str(row.painting),
            "title":   str(row.title),
            "gesture": str(row.gesture),
        }
        for row in g.query(_QUERY_CANDIDATES)
    }
    if not candidates:
        return None, {}

    theme_scores: dict[str, int] = {
        str(row.painting): int(row["count"])
        for row in g.query(_QUERY_THEME_MATCHES)
    }

    profile_matches: set[str] = {
        str(row.painting) for row in g.query(_QUERY_PROFILE_MATCHES)
    }

    mood_matches: set[str] = {
        str(row.painting) for row in g.query(_QUERY_MOOD_MATCHES)
    }

    scored = [
        (
            theme_scores.get(uri, 0) * 3
            + (2 if uri in profile_matches else 0)
            + (1 if uri in mood_matches else 0),
            uri,
        )
        for uri in candidates
    ]
    best_uri = max(scored, key=lambda x: x[0])[1]

    candidate_scores = {
        id_to_yaml[URIRef(uri)]["id"]: score
        for score, uri in scored
        if URIRef(uri) in id_to_yaml
    }
    return id_to_yaml.get(URIRef(best_uri)), candidate_scores


def mark_shown(g: Graph, painting_id: str) -> None:
    """
    Belief update after the robot has explained a painting:
      notKnows(visitor, painting) → knows(visitor, painting)

    This models the epistemic action explain(robot, visitor, painting)
    from human-aware planning: the robot's goal is achieved for this painting.
    """
    uri = PAINTER[painting_id]
    g.remove((VISITOR_URI, MUSEO.notKnows, uri))
    g.add((VISITOR_URI,    MUSEO.knows,    uri))


def set_maintenance(g: Graph, painting_id: str, flag: bool) -> None:
    """
    Staff-only operational constraint (HRI6 - KR&Planning: execution rules,
    PNP-RO style — "if painting_under_maintenance do skip_recommend" layered
    on top of the KB, not a Python-side filter list and not a catalog
    deletion). Suppresses a painting from sparql_recommend()'s candidate set
    (see the FILTER NOT EXISTS clause in _QUERY_CANDIDATES) without touching
    knows/notKnows or the YAML catalog — a visitor who already knows about
    it, or names it explicitly, can still be told about it
    (_find_mentioned_painting / answer_more_detail don't go through
    sparql_recommend).
    """
    uri = PAINTER[painting_id]
    g.remove((uri, MUSEO.underMaintenance, None))
    if flag:
        g.add((uri, MUSEO.underMaintenance, Literal(True)))


def is_under_maintenance(g: Graph, painting_id: str) -> bool:
    """True if staff has flagged this painting as under maintenance (see set_maintenance)."""
    return bool(list(g.triples((PAINTER[painting_id], MUSEO.underMaintenance, None))))


def most_similar_available(g: Graph, target: dict, present_paintings: list[dict]) -> dict | None:
    """
    Best AVAILABLE painting similar to `target`, or None if nothing is available.

    Used for the visitor-facing contingency (guida_museale_core.handle_unavailable_request):
    a visitor asks for a work that is not on display or under maintenance, so we
    negotiate the closest alternative that is actually available now. Available =
    currently hung AND not under maintenance. Similarity combines the curated
    related_to links, same movement/style, same artist and shared themes.
    """
    t_themes  = set(target.get("themes", []))
    t_style   = target.get("style")
    t_artist  = target.get("artist")
    t_related = set(target.get("related_to", []))
    t_id      = target.get("id")

    best, best_score = None, -1
    for p in present_paintings:
        if p["id"] == t_id or is_under_maintenance(g, p["id"]):
            continue
        score = 0
        if p["id"] in t_related or t_id in set(p.get("related_to", [])):
            score += 3
        if t_style and p.get("style") == t_style:
            score += 2
        if t_artist and p.get("artist") == t_artist:
            score += 1
        score += len(t_themes & set(p.get("themes", [])))
        if score > best_score:
            best, best_score = p, score
    return best


def reopen_gallery(g: Graph) -> None:
    """
    Move every painting the visitor 'knows' back to 'notKnows'.

    Used when the visitor has been shown every painting but wants to keep
    going — an epistemic reset scoped to knows/notKnows only, so profile,
    interests and mood (built up over the conversation) are left untouched.
    Contrast with reset_visitor(), which starts an entirely new session.
    """
    for _, _, o in list(g.triples((VISITOR_URI, MUSEO.knows, None))):
        g.remove((VISITOR_URI, MUSEO.knows, o))
        g.add((VISITOR_URI, MUSEO.notKnows, o))


def sync_visitor(g: Graph, user_model: dict) -> None:
    """
    Write the current Python user_model dict into the KG visitor node.

    Called after each dialogue turn so SPARQL queries see up-to-date beliefs.
    Interests with weight ≤ 0.3 are dropped (below threshold for actionable preference).
    """
    for pred in (MUSEO.hasProfile, MUSEO.availableTime, MUSEO.preferredMood, MUSEO.hasInterest):
        for _, _, o in list(g.triples((VISITOR_URI, pred, None))):
            g.remove((VISITOR_URI, pred, o))

    g.add((VISITOR_URI, MUSEO.hasProfile,    Literal(user_model.get("profile", "unknown"))))
    g.add((VISITOR_URI, MUSEO.availableTime, Literal(user_model.get("available_time", "unknown"))))
    g.add((VISITOR_URI, MUSEO.preferredMood, Literal(user_model.get("preferred_mood", "unknown"))))

    for theme, weight in (user_model.get("interests") or {}).items():
        if weight > 0.3:
            g.add((VISITOR_URI, MUSEO.hasInterest, Literal(theme)))


def reset_visitor(g: Graph, paintings: list[dict]) -> None:
    """Reset epistemic state for a new visitor session (all paintings back to notKnows)."""
    for pred in (
        MUSEO.hasProfile, MUSEO.availableTime, MUSEO.preferredMood,
        MUSEO.hasInterest, MUSEO.knows, MUSEO.notKnows,
    ):
        for s, p, o in list(g.triples((VISITOR_URI, pred, None))):
            g.remove((s, p, o))

    g.add((VISITOR_URI, MUSEO.hasProfile,    Literal("unknown")))
    g.add((VISITOR_URI, MUSEO.availableTime, Literal("unknown")))
    g.add((VISITOR_URI, MUSEO.preferredMood, Literal("unknown")))
    for p_dict in paintings:
        g.add((VISITOR_URI, MUSEO.notKnows, PAINTER[p_dict["id"]]))


def can_point(g: Graph, painting_id: str) -> bool:
    """
    Return True if the robot has a pointing gesture for this painting.

    Replaces check_feasibility(): the KG encodes 'affordsPointTo' as a scene-graph
    property — the painting affords the robot the 'point_to' action.
    """
    uri = PAINTER[painting_id]
    return bool(list(g.triples((uri, MUSEO.affordsPointTo, None))))


def _preference_node(topic: str) -> URIRef:
    return MUSEO[f"pref_{topic}"]


def set_robot_preference(g: Graph, topic: str, value: str, reason: str) -> None:
    """Store/overwrite the robot's preference for `topic` (idempotent per topic)."""
    node = _preference_node(topic)
    for _, _, o in list(g.triples((node, None, None))):
        g.remove((node, None, o))
    g.remove((ROBOT_URI, MUSEO.hasPreference, node))

    g.add((ROBOT_URI, RDF.type, MUSEO.Robot))
    g.add((ROBOT_URI, MUSEO.hasPreference, node))
    g.add((node, RDF.type,               MUSEO.Preference))
    g.add((node, MUSEO.preferenceTopic,  Literal(topic)))
    g.add((node, MUSEO.preferenceValue,  Literal(value)))
    g.add((node, MUSEO.preferenceReason, Literal(reason)))


def get_robot_preference(g: Graph, topic: str) -> dict | None:
    """Return {'value', 'reason'} for a stored preference, or None if unknown."""
    node = _preference_node(topic)
    if not list(g.triples((node, MUSEO.preferenceValue, None))):
        return None
    return {
        "value":  _get_literal(g, node, MUSEO.preferenceValue, ""),
        "reason": _get_literal(g, node, MUSEO.preferenceReason, ""),
    }


def all_robot_preferences(g: Graph) -> list[dict]:
    """Every stored robot preference, for persistence / the viewer."""
    prefs = []
    for _, _, node in g.triples((ROBOT_URI, MUSEO.hasPreference, None)):
        prefs.append({
            "topic":  _get_literal(g, node, MUSEO.preferenceTopic, ""),
            "value":  _get_literal(g, node, MUSEO.preferenceValue, ""),
            "reason": _get_literal(g, node, MUSEO.preferenceReason, ""),
        })
    return prefs


_PERCEPTION_REQUIRED_PREDICATES = (
    MUSEO.perceivedMovement, MUSEO.perceivedMood, MUSEO.perceivedTheme,
)


def perception_coverage(g: Graph, painting_id: str) -> float:
    """
    P(T_perceived, E_required) = |T ∩ E| / |E|  (HRAI 9, Perception of the
    Environment level of the self-awareness hierarchy).

    Fraction of the VLM-perceivable attributes (movement, mood, themes) the
    robot has actually observed with its own camera for this painting, out of
    the full set an explanation task requires. Used as one term of the
    feasibility score Ft(T) in guida_museale_core.check_feasibility().
    """
    uri = PAINTER[painting_id]
    observed = sum(
        1 for pred in _PERCEPTION_REQUIRED_PREDICATES
        if list(g.triples((uri, pred, None)))
    )
    return observed / len(_PERCEPTION_REQUIRED_PREDICATES)


def spatial_relation_text(g: Graph, painting_id: str) -> str | None:
    """
    Short natural-language spatial reference from the leftOf/rightOf scene-graph
    edges built in _add_scene_graph(), e.g. "to the left of \"Las Meninas\"".

    Returns None if no spatial relation is known for this painting — this is
    the practical difference between a flat museo:locatedAt literal and an
    actual graph edge: the edge lets us generate a relational description,
    the literal doesn't.
    """
    uri = PAINTER[painting_id]
    for _, _, obj in g.triples((uri, MUSEO.leftOf, None)):
        if title := _get_literal(g, obj, MUSEO.hasTitle, ""):
            return f'to the left of "{title}"'
    for _, _, obj in g.triples((uri, MUSEO.rightOf, None)):
        if title := _get_literal(g, obj, MUSEO.hasTitle, ""):
            return f'to the right of "{title}"'
    return None


def reference_frame_info(g: Graph, room_id: str | None = None) -> dict | None:
    """
    R in the SM = ⟨R, M, P⟩ formalism, made query-able (see WORLD_FRAME_URI's
    docstring). Returns {"room", "frame", "comment"} for the given room (or
    the first Room found, since today there is only ever one), or None if
    _add_scene_graph() hasn't run yet.
    """
    room_uri = ROOM[room_id] if room_id else next(g.subjects(RDF.type, MUSEO.Room), None)
    if room_uri is None:
        return None
    frame = next(g.objects(room_uri, MUSEO.hasReferenceFrame), None)
    if frame is None:
        return None
    return {
        "room":    str(room_uri),
        "frame":   str(frame),
        "comment": _get_literal(g, frame, RDFS.comment, ""),
    }


def _scene_relations(placement: dict[str, str]) -> set[tuple[str, str, str]]:
    """P as a flat triple set — locatedOnWall/leftOf/rightOf — derived from a
    gesture->painting_id placement via spatial_map_from_placement()'s head-
    yaw geometry. Used by both sides of semantic_map_delta() so a difference
    in wall-naming convention between sources can never be the cause of a
    mismatch — only a difference in WHICH painting is where can be."""
    objects = spatial_map_from_placement(placement)["scene"]["objects"]
    triples: set[tuple[str, str, str]] = set()
    for o in objects:
        triples.add((o["id"], "locatedOnWall", o["wall"]))
        rel = o["relations"]
        if "left_of" in rel:
            triples.add((o["id"], "leftOf", rel["left_of"]))
        if "right_of" in rel:
            triples.add((o["id"], "rightOf", rel["right_of"]))
    return triples


def semantic_map_delta(built_placement: dict[str, str], ground_truth_placement: dict[str, str]) -> dict:
    """
    δ(SM1, SMm) = f(|M1 ⊖ Mm|, |P1 ⊟ Pm|) — "Building a Semantic Map" slide.

    Compares a built semantic map SM1 = ⟨R, M1, P1⟩ (the "env" — what was
    actually perceived, e.g. the robot's own raw scan) against an expected
    one SMm = ⟨R, Mm, Pm⟩ (e.g. scene_ground_truth.yaml). R is shared by
    construction (both placements are keyed by the same fixed set of scan
    gestures, i.e. expressed in the same world frame — see WORLD_FRAME_URI)
    so it plays no part in the comparison, exactly as the formula implies
    (only M and P are compared).

    M1 ⊖ Mm: symmetric difference on the object level — for each scan
    direction (gesture), is the SAME painting_id present in both maps.
    P1 ⊟ Pm: symmetric difference on the relation level — see
    _scene_relations(). Wrong here means the WRONG STRUCTURE was inferred
    (adjacency/wall grouping), which can happen even when M agrees almost
    everywhere: one wrong painting_id shifts every leftOf/rightOf edge it
    touches, not just its own locatedOnWall triple.

    `f` is left undefined by the slide; here it's an equal-weighted average
    of the two normalized-to-[0,1] discrepancies, in the same [0,1] spirit
    as the deck's ORI/OPI map-quality metrics. Returns 0.0 for an exact
    match, 1.0 for total disagreement on both axes.
    """
    all_gestures = sorted(set(built_placement) | set(ground_truth_placement))
    mismatched = [g for g in all_gestures if built_placement.get(g) != ground_truth_placement.get(g)]
    m_delta = len(mismatched) / len(all_gestures) if all_gestures else 0.0

    p_built = _scene_relations(built_placement)
    p_truth = _scene_relations(ground_truth_placement)
    p_sym_diff = p_built ^ p_truth
    p_union = p_built | p_truth
    p_delta = len(p_sym_diff) / len(p_union) if p_union else 0.0

    return {
        "delta":         round(0.5 * m_delta + 0.5 * p_delta, 3),
        "m_delta":       round(m_delta, 3),
        "p_delta":       round(p_delta, 3),
        "m_mismatches":  [
            {"gesture": g, "built": built_placement.get(g), "ground_truth": ground_truth_placement.get(g)}
            for g in mismatched
        ],
        "p_sym_diff": sorted(" ".join(t) for t in p_sym_diff),
    }


def _get_literal(g: Graph, subject: URIRef, predicate: URIRef, default: str = "") -> str:
    for _, _, obj in g.triples((subject, predicate, None)):
        return str(obj)
    return default


class GuidanceHTN:
    """
    Hierarchical Task Network for museum guidance.

    Formalises the explanation-selection + action-sequencing logic that was
    previously scattered across select_explanation() and if-chains in
    handle_recommend() (HRI6 Sl. 10-12).

    Abstract task:  guide_visitor(visitor, painting)

    Methods (tried in priority order, first applicable wins):
      M1: available_time == "short"
          → [quick_overview, point_gesture, goodbye_offer]          key: short
      M2: profile == "child"
          → [tell_story, point_gesture, simple_question]            key: child
      M3: profile == "expert"
          → [historical_context, point_gesture,
             technical_detail, open_discussion]                      key: expert
      M4: profile == "tourist_short_time"
          → [quick_overview, point_gesture, goodbye_offer]          key: short
      M5: DEFAULT (novice / unknown)
          → [standard_explanation, point_gesture, satisfaction_check] key: novice

    Primitive tasks map to LLM actions in guida_museale_core.py.
    """

    TASK_MAP: dict[str, tuple[str | None, str | None]] = {
        "quick_overview":      ("short",  "recommend_and_explain"),
        "tell_story":          ("child",  "recommend_and_explain"),
        "historical_context":  ("novice", "recommend_and_explain"),
        "standard_explanation":("novice", "recommend_and_explain"),
        "point_gesture":       (None,     None),
        "technical_detail":    ("expert", "answer_more_detail"),
        "simple_question":     (None,     "satisfaction_check"),
        "open_discussion":     (None,     "satisfaction_check"),
        "satisfaction_check":  (None,     "satisfaction_check"),
        "goodbye_offer":       (None,     "close_or_restart"),
    }

    def decompose(
        self,
        g: Graph,
        visitor_uri: URIRef = VISITOR_URI,
    ) -> tuple[list[str], str]:
        """
        Decompose abstract task guide_visitor → concrete task sequence.

        Returns (task_sequence, primary_explanation_key).
        task_sequence: list of primitive task names to execute in order.
        explanation_key: which explanation variant to fetch from paintings.yaml.
        """
        time_val = _get_literal(g, visitor_uri, MUSEO.availableTime, "unknown")
        profile  = _get_literal(g, visitor_uri, MUSEO.hasProfile,    "novice")

        if time_val == "short":
            return ["quick_overview", "point_gesture", "goodbye_offer"], "short"
        if profile == "child":
            return ["tell_story", "point_gesture", "simple_question"], "child"
        if profile == "expert":
            return [
                "historical_context",
                "point_gesture",
                "technical_detail",
                "open_discussion",
            ], "expert"
        if profile == "tourist_short_time":
            return ["quick_overview", "point_gesture", "goodbye_offer"], "short"
        return ["standard_explanation", "point_gesture", "satisfaction_check"], "novice"


def init_kg(data_dir: Path, discovered_path: Path | None = None) -> Graph:
    """Build the knowledge graph and store it as the module singleton. Idempotent."""
    global _G
    if _G is not None:
        return _G
    _G = build_graph(data_dir, discovered_path)
    return _G


def get_graph() -> Graph:
    if _G is None:
        raise RuntimeError("KG not initialized — call init_kg() first.")
    return _G


def dump_turtle(g: Graph) -> str:
    """Serialize graph to Turtle for debugging."""
    return g.serialize(format="turtle")


def _short_id(uri: URIRef) -> str:
    """Last path segment of a URI, used as the node id in the viewer JSON."""
    return str(uri).rstrip("/").rsplit("/", 1)[-1]


def export_graph_snapshot(
    g: Graph,
    path: Path,
    pipeline_trace: list[dict] | None = None,
    visitor_model: dict | None = None,
    paintings: list[dict] | None = None,
    robot_pose: dict | None = None,
) -> None:
    """
    Serialize the graph's instance data (ABox) into the small JSON format
    consumed by viewer/kg_viewer.html — the live node-link visualizer used to
    show, turn by turn, how the RDF graph is populated/queried/updated.

    Only object-to-object relations become edges (hasWall, locatedOnWall,
    leftOf/rightOf, knows/notKnows); literal-valued properties (title, mood,
    themes, profile...) are folded into the node's own attributes instead of
    becoming separate literal nodes, so the diagram stays a clean Room/Wall/
    Painting/Visitor graph rather than a triple-store dump.

    `paintings`/`robot_pose` are M — real xyz, expressed in R (see
    WORLD_FRAME_URI) — for the small top-down "semantic map" panel the viewer
    draws next to the node-link graph. They don't belong in the RDF graph
    itself (P stays symbolic, see _add_scene_graph) so, like `visitor_model`,
    they're display-only data folded into the snapshot rather than the graph.
    """
    nodes: dict[str, dict] = {}
    edges: list[dict] = []

    def ensure(uri: URIRef, ntype: str) -> dict:
        nid = _short_id(uri)
        return nodes.setdefault(nid, {"id": nid, "type": ntype, "attrs": {}})

    for room_uri in g.subjects(RDF.type, MUSEO.Room):
        ensure(room_uri, "Room")

    for wall_uri in g.subjects(RDF.type, MUSEO.Wall):
        ensure(wall_uri, "Wall")

    for p_uri in g.subjects(RDF.type, MUSEO.Painting):
        n = ensure(p_uri, "Painting")
        mismatch = [str(o) for o in g.objects(p_uri, MUSEO.perceptionMismatch)]
        n["attrs"] = {
            "title":              _get_literal(g, p_uri, MUSEO.hasTitle),
            "artist":             _get_literal(g, p_uri, MUSEO.hasArtist),
            "movement":           _get_literal(g, p_uri, MUSEO.hasMovement),
            "difficulty":         _get_literal(g, p_uri, MUSEO.difficulty),
            "mood":               [str(o) for o in g.objects(p_uri, MUSEO.hasMood)],
            "themes":             [str(o) for o in g.objects(p_uri, MUSEO.hasTheme)],
            "perceived_movement": _get_literal(g, p_uri, MUSEO.perceivedMovement),
            "perceived_verified": bool(list(g.objects(p_uri, MUSEO.perceivedMovement))) and not mismatch,
            "perception_mismatch": mismatch[0] if mismatch else None,
            "under_maintenance":  bool(list(g.objects(p_uri, MUSEO.underMaintenance))),
        }

    for v_uri in g.subjects(RDF.type, MUSEO.Visitor):
        n = ensure(v_uri, "Visitor")
        n["attrs"] = {
            "profile":         _get_literal(g, v_uri, MUSEO.hasProfile, "unknown"),
            "available_time":  _get_literal(g, v_uri, MUSEO.availableTime, "unknown"),
            "preferred_mood":  _get_literal(g, v_uri, MUSEO.preferredMood, "unknown"),
        }
        if visitor_model:
            n["attrs"].update({
                "profile":               visitor_model.get("profile", n["attrs"]["profile"]),
                "available_time":        visitor_model.get("available_time", n["attrs"]["available_time"]),
                "preferred_mood":        visitor_model.get("preferred_mood", n["attrs"]["preferred_mood"]),
                "preferred_mood_source": visitor_model.get("preferred_mood_source", "—"),
                "language":              visitor_model.get("language", "unknown"),
                "role":                  visitor_model.get("role", "visitor"),
                "interests":             visitor_model.get("interests", {}),
            })
        for _, _, obj in g.triples((v_uri, MUSEO.knows, None)):
            edges.append({"source": _short_id(v_uri), "target": _short_id(obj), "label": "knows"})
        for _, _, obj in g.triples((v_uri, MUSEO.notKnows, None)):
            edges.append({"source": _short_id(v_uri), "target": _short_id(obj), "label": "notKnows"})

        for theme in g.objects(v_uri, MUSEO.hasInterest):
            theme_id = f"interest:{theme}"
            nodes.setdefault(theme_id, {"id": theme_id, "type": "Interest", "attrs": {"label": str(theme)}})
            edges.append({"source": _short_id(v_uri), "target": theme_id, "label": "hasInterest"})

    _STRUCTURAL_EDGES = {
        MUSEO.hasWall:       "hasWall",
        MUSEO.locatedOnWall: "locatedOnWall",
        MUSEO.leftOf:        "leftOf",
        MUSEO.rightOf:       "rightOf",
    }
    for s, p, o in g:
        label = _STRUCTURAL_EDGES.get(p)
        if label is None:
            continue
        source_id, target_id = _short_id(s), _short_id(o)
        if source_id not in nodes or target_id not in nodes:
            continue
        edges.append({"source": source_id, "target": target_id, "label": label})

    map_paintings = []
    for p in (paintings or []):
        xyz = (p.get("robot_actions") or {}).get("target_xyz")
        if xyz is None:
            continue
        wall = next(g.objects(PAINTER[p["id"]], MUSEO.locatedOnWall), None)
        map_paintings.append({
            "id":    p["id"],
            "title": p.get("title", p["id"]),
            "x":     xyz[0],
            "y":     xyz[1],
            "wall":  _short_id(wall) if wall is not None else None,
        })
    map_robot = None
    if robot_pose is not None:
        map_robot = {"x": robot_pose["x"], "y": robot_pose["y"], "yaw": robot_pose["yaw"]}

    snapshot = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "triple_count": len(g),
        "nodes": list(nodes.values()),
        "edges": edges,
        "pipeline_trace": pipeline_trace or [],
        "geometry": {"paintings": map_paintings, "robot": map_robot},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(snapshot, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp_path.replace(path)
