from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from osumapper.paths import project_root
from osumapper.training.beatmaps import ModernHitObject, parse_standard_beatmap
from osumapper.training.storage import write_json

PLAYFIELD_WIDTH = 512
PLAYFIELD_HEIGHT = 384


def _percentile(values: list[float], percentile: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile)) if values else 0.0


def _turn_angle(first: tuple[float, float], second: tuple[float, float]) -> float | None:
    first_length = math.hypot(*first)
    second_length = math.hypot(*second)
    if first_length <= 1e-6 or second_length <= 1e-6:
        return None
    cosine = (first[0] * second[0] + first[1] * second[1]) / (first_length * second_length)
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))


def _transition_rows(objects: list[ModernHitObject]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    previous_vector: tuple[float, float] | None = None
    for index, (previous, current) in enumerate(zip(objects, objects[1:], strict=False), start=1):
        delta_ms = current.time_ms - previous.time_ms
        dx = float(current.x - previous.x)
        dy = float(current.y - previous.y)
        distance = math.hypot(dx, dy)
        vector = (dx, dy)
        angle = _turn_angle(previous_vector, vector) if previous_vector is not None else None
        rows.append(
            {
                "object_index": index,
                "timestamp_ms": current.time_ms,
                "delta_time_ms": delta_ms,
                "distance_px": distance,
                "speed_px_per_second": distance / (delta_ms / 1000.0) if delta_ms > 0 else None,
                "turn_angle_degrees": angle,
                "x": current.x,
                "y": current.y,
                "kind": current.kind,
            }
        )
        if distance > 1e-6:
            previous_vector = vector
    return rows


def analyze_placement(
    map_path: Path,
    *,
    output: Path | None = None,
) -> dict[str, Any]:
    """Report deterministic playfield and movement heuristics for an osu!standard map."""
    source = map_path.expanduser().resolve()
    parsed = parse_standard_beatmap(source, source.parent)
    playable = [obj for obj in parsed.hit_objects if obj.kind != "spinner"]
    transitions = _transition_rows(playable)
    distances = [float(row["distance_px"]) for row in transitions]
    speeds = [
        float(row["speed_px_per_second"])
        for row in transitions
        if row["speed_px_per_second"] is not None
    ]
    angles = [
        float(row["turn_angle_degrees"])
        for row in transitions
        if row["turn_angle_degrees"] is not None
    ]
    offscreen = [
        obj
        for obj in playable
        if not 0 <= obj.x <= PLAYFIELD_WIDTH or not 0 <= obj.y <= PLAYFIELD_HEIGHT
    ]
    slider_points = [point for obj in playable for point in obj.slider_path]
    offscreen_slider_points = [
        point
        for point in slider_points
        if not 0 <= point[0] <= PLAYFIELD_WIDTH or not 0 <= point[1] <= PLAYFIELD_HEIGHT
    ]
    edge_objects = [
        obj
        for obj in playable
        if min(obj.x, PLAYFIELD_WIDTH - obj.x, obj.y, PLAYFIELD_HEIGHT - obj.y) < 24
    ]
    stack_count = sum(
        float(row["distance_px"]) < 8.0 and float(row["delta_time_ms"]) <= 1_000.0
        for row in transitions
    )
    unique_positions = len({(obj.x, obj.y) for obj in playable})
    object_count = len(playable)
    transition_count = len(transitions)
    edge_ratio = len(edge_objects) / object_count if object_count else 0.0
    stack_ratio = stack_count / transition_count if transition_count else 0.0
    unique_ratio = unique_positions / object_count if object_count else 0.0
    speed_p95 = _percentile(speeds, 95)

    issues: list[dict[str, Any]] = []

    def issue(severity: str, code: str, message: str, count: int) -> None:
        if count:
            issues.append({"severity": severity, "code": code, "message": message, "count": count})

    issue("error", "offscreen_objects", "Objects fall outside the osu! playfield.", len(offscreen))
    issue(
        "error",
        "offscreen_slider_points",
        "Slider control points fall outside the osu! playfield.",
        len(offscreen_slider_points),
    )
    issue(
        "warning",
        "extreme_jump_speed",
        "Transitions exceed the conservative 3,500 px/s review threshold.",
        sum(speed > 3_500.0 for speed in speeds),
    )
    if stack_ratio > 0.25:
        issue(
            "warning",
            "repetitive_stacking",
            "More than 25% of transitions reuse nearly the same position.",
            stack_count,
        )
    if edge_ratio > 0.35:
        issue(
            "warning",
            "excessive_edge_use",
            "More than 35% of objects are within 24 px of a playfield edge.",
            len(edge_objects),
        )
    if object_count >= 20 and unique_ratio < 0.25:
        issue(
            "warning",
            "low_position_diversity",
            "Fewer than 25% of object positions are unique.",
            object_count - unique_positions,
        )

    penalty = (
        20 * len(offscreen)
        + 10 * len(offscreen_slider_points)
        + 15 * int(speed_p95 > 3_500.0)
        + 10 * int(stack_ratio > 0.25)
        + 10 * int(edge_ratio > 0.35)
        + 10 * int(object_count >= 20 and unique_ratio < 0.25)
    )
    x_values = [obj.x for obj in playable]
    y_values = [obj.y for obj in playable]
    report: dict[str, Any] = {
        "version": 1,
        "analyzer": "placement-heuristics-v1",
        "heuristic": True,
        "map": str(source),
        "objects": len(parsed.hit_objects),
        "playable_objects": object_count,
        "quality_score": max(0, 100 - penalty),
        "metrics": {
            "offscreen_objects": len(offscreen),
            "offscreen_slider_points": len(offscreen_slider_points),
            "unique_position_ratio": unique_ratio,
            "stacked_transition_ratio": stack_ratio,
            "edge_object_ratio": edge_ratio,
            "mean_jump_distance_px": float(np.mean(distances)) if distances else 0.0,
            "median_jump_distance_px": _percentile(distances, 50),
            "p95_jump_distance_px": _percentile(distances, 95),
            "max_jump_distance_px": max(distances, default=0.0),
            "p95_jump_speed_px_per_second": speed_p95,
            "max_jump_speed_px_per_second": max(speeds, default=0.0),
            "median_turn_angle_degrees": _percentile(angles, 50),
            "position_spread_x": float(np.std(x_values)) if x_values else 0.0,
            "position_spread_y": float(np.std(y_values)) if y_values else 0.0,
        },
        "issues": issues,
        "transitions": transitions,
    }
    destination = (
        (output or project_root() / "output" / f"{source.stem}-placement-analysis.json")
        .expanduser()
        .resolve()
    )
    if destination == source:
        raise ValueError("Placement analysis must not overwrite the source beatmap.")
    write_json(destination, report)
    report["output"] = str(destination)
    return report
