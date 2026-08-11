from __future__ import annotations

import bisect
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from osumapper.errors import InputError
from osumapper.training.config import GridConfig
from osumapper.training.features import load_features


@dataclass(frozen=True, slots=True)
class GridCandidate:
    timestamp_ms: float
    bpm: float
    beat_position: float
    measure_position: float
    meter: int
    divisor: int
    subdivision_index: int
    label: int
    matched_object_index: int | None
    time_since_previous_object_ms: float
    target_density: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class TimingGrid:
    candidates: tuple[GridCandidate, ...]
    matched_objects: int
    unmatched_objects: int


def load_map_detail(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise InputError(f"Could not read parsed map data {path}: {exc}") from exc


def _red_points(detail: dict[str, Any]) -> list[dict[str, Any]]:
    points = [point for point in detail["timing_points"] if point["uninherited"]]
    if not points:
        raise InputError("Parsed map has no uninherited timing points.")
    return sorted(points, key=lambda point: float(point["time_ms"]))


def create_timing_grid(
    detail: dict[str, Any],
    *,
    config: GridConfig | None = None,
    inference_end_ms: float | None = None,
    target_density: float | None = None,
) -> TimingGrid:
    grid_config = config or GridConfig()
    objects = detail["hit_objects"]
    if not objects and inference_end_ms is None:
        raise InputError("Cannot create a training grid for a map with no hit objects.")
    red_points = _red_points(detail)
    start_ms = (
        max(0.0, min(float(objects[0]["time_ms"]), float(red_points[0]["time_ms"])))
        if objects
        else max(0.0, float(red_points[0]["time_ms"]))
    )
    end_ms = (
        max(float(obj["end_time_ms"]) for obj in objects) if objects else float(inference_end_ms)
    )
    raw_candidates: dict[int, tuple[float, float, float, int, int, int, float]] = {}
    for section_index, point in enumerate(red_points):
        origin = float(point["time_ms"])
        beat_length = float(point["beat_length_ms"])
        bpm = float(point["bpm"])
        meter = int(point["meter"])
        section_start = max(start_ms, origin)
        section_end = (
            min(end_ms, float(red_points[section_index + 1]["time_ms"]))
            if section_index + 1 < len(red_points)
            else end_ms
        )
        if section_end < section_start:
            continue
        for divisor in grid_config.subdivisions:
            first_tick = math.ceil((section_start - origin) / beat_length * divisor - 1e-9)
            last_tick = math.floor((section_end - origin) / beat_length * divisor + 1e-9)
            for tick in range(first_tick, last_tick + 1):
                timestamp = origin + tick * beat_length / divisor
                key = int(round(timestamp * 1000.0))
                beat_position = (timestamp - origin) / beat_length
                measure_position = (beat_position % meter) / meter
                existing = raw_candidates.get(key)
                value = (
                    timestamp,
                    bpm,
                    beat_position,
                    meter,
                    divisor,
                    tick % divisor,
                    measure_position,
                )
                if existing is None or divisor < existing[4]:
                    raw_candidates[key] = value
    ordered = sorted(raw_candidates.values(), key=lambda item: item[0])
    if not ordered:
        raise InputError("Timing points produced no musical-grid candidates.")
    candidate_times = [value[0] for value in ordered]
    matches: dict[int, int] = {}
    used_candidates: set[int] = set()
    for object_index, obj in enumerate(objects):
        object_time = float(obj["time_ms"])
        insertion = bisect.bisect_left(candidate_times, object_time)
        choices = [index for index in (insertion - 1, insertion) if 0 <= index < len(ordered)]
        choices.sort(key=lambda index: abs(candidate_times[index] - object_time))
        selected = next(
            (
                index
                for index in choices
                if index not in used_candidates
                and abs(candidate_times[index] - object_time) <= grid_config.hit_tolerance_ms
            ),
            None,
        )
        if selected is not None:
            used_candidates.add(selected)
            matches[selected] = object_index

    object_times = [float(obj["time_ms"]) for obj in objects]
    density = float(
        target_density if target_density is not None else detail["statistics"]["objects_per_second"]
    )
    candidates: list[GridCandidate] = []
    previous_object = -math.inf
    object_cursor = 0
    for index, value in enumerate(ordered):
        timestamp, bpm, beat_position, meter, divisor, subdivision_index, measure_position = value
        while object_cursor < len(object_times) and object_times[object_cursor] < timestamp:
            previous_object = object_times[object_cursor]
            object_cursor += 1
        time_since = timestamp - previous_object if math.isfinite(previous_object) else 60_000.0
        candidates.append(
            GridCandidate(
                timestamp_ms=timestamp,
                bpm=bpm,
                beat_position=beat_position,
                measure_position=measure_position,
                meter=meter,
                divisor=divisor,
                subdivision_index=subdivision_index,
                label=1 if index in matches else 0,
                matched_object_index=matches.get(index),
                time_since_previous_object_ms=min(60_000.0, max(0.0, time_since)),
                target_density=density,
            )
        )
    return TimingGrid(
        candidates=tuple(candidates),
        matched_objects=len(matches),
        unmatched_objects=len(objects) - len(matches),
    )


def align_features_to_grid(feature_file: Path, grid: TimingGrid, *, context_radius: int = 0) -> Any:
    import numpy as np

    if context_radius < 0:
        raise InputError("Audio context radius cannot be negative.")

    features = load_features(feature_file)
    times = features["frame_times_ms"]
    frame_values = np.concatenate(
        [
            features["mel"],
            features["onset"][:, None],
            features["rms"][:, None],
            features["spectral_flux"][:, None],
            features["beat_pulse"][:, None],
        ],
        axis=1,
    ).astype(np.float32)
    candidate_times = np.asarray(
        [candidate.timestamp_ms for candidate in grid.candidates], dtype=np.float64
    )
    indices = np.searchsorted(times, candidate_times, side="left")
    indices = np.clip(indices, 0, len(times) - 1)
    previous = np.maximum(indices - 1, 0)
    use_previous = np.abs(times[previous] - candidate_times) < np.abs(
        times[indices] - candidate_times
    )
    indices = np.where(use_previous, previous, indices)
    if context_radius == 0:
        return frame_values[indices]
    offsets = np.arange(-context_radius, context_radius + 1, dtype=np.int64)
    contextual_indices = np.clip(indices[:, None] + offsets[None, :], 0, len(times) - 1)
    contextual = frame_values[contextual_indices]
    return contextual.reshape(len(indices), -1)


def grid_feature_array(grid: TimingGrid) -> Any:
    import numpy as np

    return np.asarray(
        [
            [
                candidate.bpm / 300.0,
                math.sin(2 * math.pi * candidate.measure_position),
                math.cos(2 * math.pi * candidate.measure_position),
                candidate.divisor / 8.0,
                candidate.subdivision_index / max(1, candidate.divisor),
                candidate.target_density / 10.0,
            ]
            for candidate in grid.candidates
        ],
        dtype=np.float32,
    )
