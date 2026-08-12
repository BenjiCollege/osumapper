from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from osumapper.beatmap import BeatmapDocument
from osumapper.difficulty import (
    LEGACY_DIFFICULTY_FEATURES,
    STAR_DIFFICULTY_FEATURES,
    resolve_standard_difficulty,
)
from osumapper.errors import InputError
from osumapper.training.beatmaps import parse_timing_points
from osumapper.training.config import AudioFeatureConfig, GridConfig, prediction_threshold
from osumapper.training.features import extract_audio_features
from osumapper.training.grid import (
    align_features_to_grid,
    create_timing_grid,
    grid_feature_array,
)
from osumapper.training.loader import PreparedMap, difficulty_feature_array, window_probabilities
from osumapper.training.model import load_rhythm_model
from osumapper.training.storage import read_json
from osumapper.training.trainer import default_model_root

Progress = Callable[[str], None]


@lru_cache(maxsize=2)
def _cached_inference_model(path: str) -> Any:
    return load_rhythm_model(Path(path), compile_model=False)


@dataclass(frozen=True, slots=True)
class ModernRhythmPrediction:
    timestamps_ms: np.ndarray[Any, Any]
    probabilities: np.ndarray[Any, Any]
    candidate_indices: np.ndarray[Any, Any]
    target_density: float
    threshold: float
    target_stars: float | None = None
    difficulty_tier: str | None = None


_PREDICTION_CACHE_LIMIT = 64
_prediction_cache: dict[tuple[Any, ...], ModernRhythmPrediction] = {}


def _difficulty(
    document: BeatmapDocument,
    target_density: float,
    feature_names: tuple[str, ...],
    *,
    target_stars: float | None,
    difficulty_tier: str | None,
) -> np.ndarray[Any, Any]:
    values = document.values("Difficulty")

    def number(key: str, default: float) -> float:
        try:
            return float(values.get(key, default))
        except ValueError as exc:
            raise InputError(f"Invalid difficulty value {key} in {document.path}") from exc

    od = number("OverallDifficulty", 5.0)
    ar = number("ApproachRate", od)
    cs = number("CircleSize", 4.0)
    row = {
        "od": od,
        "ar": ar,
        "cs": cs,
        "objects_per_second": target_density,
        "star_rating": target_stars,
        "difficulty_tier": difficulty_tier,
    }
    if feature_names == STAR_DIFFICULTY_FEATURES and (
        target_stars is None or difficulty_tier is None
    ):
        raise InputError(
            "Conformer-v4 requires --difficulty-tier or --target-stars for generation."
        )
    return difficulty_feature_array(row, feature_names)


def _source_density(document: BeatmapDocument, duration_seconds: float) -> float:
    object_count = sum(
        bool(line.strip()) and not line.lstrip().startswith("//")
        for line in document.sections().get("HitObjects", [])
    )
    if object_count and duration_seconds > 0:
        return max(0.1, min(10.0, object_count / duration_seconds))
    return 2.0


def _timing_detail(document: BeatmapDocument) -> list[dict[str, Any]]:
    return [point.as_dict() for point in parse_timing_points(document)]


def _select_events(
    candidate_times: np.ndarray[Any, Any],
    probabilities: np.ndarray[Any, Any],
    *,
    threshold: float,
    target_density: float,
    duration_seconds: float,
    min_spacing_ms: float = 45.0,
) -> np.ndarray[Any, Any]:
    eligible = np.flatnonzero(probabilities >= threshold)
    ranked = sorted(eligible.tolist(), key=lambda index: (-probabilities[index], index))
    selected: list[int] = []
    for index in ranked:
        timestamp = candidate_times[index]
        if all(abs(timestamp - candidate_times[other]) >= min_spacing_ms for other in selected):
            selected.append(index)
    maximum = max(1, math.ceil(target_density * duration_seconds * 1.5))
    if len(selected) > maximum:
        selected = selected[:maximum]
    return np.asarray(sorted(selected, key=lambda index: candidate_times[index]), dtype=int)


def predict_modern_rhythm(
    document: BeatmapDocument,
    audio: Path,
    workspace: Path,
    *,
    model_root: Path | None = None,
    threshold: float | None = None,
    target_density: float | None = None,
    difficulty_tier: str | None = None,
    target_stars: float | None = None,
    progress: Progress = print,
) -> ModernRhythmPrediction:
    requested = (model_root or default_model_root()).expanduser().resolve()
    root = requested.parent if requested.suffix.casefold() == ".keras" else requested
    model_path = requested if requested.suffix.casefold() == ".keras" else root / "model.keras"
    config = read_json(root / "config.json", default=None)
    if config is None or not model_path.is_file():
        raise InputError(
            f"Modern rhythm model is missing under {root}. Run `osumapper train rhythm`."
        )
    feature_names = tuple(config.get("difficulty_features", LEGACY_DIFFICULTY_FEATURES))
    requested_difficulty = resolve_standard_difficulty(difficulty_tier, target_stars)
    if requested_difficulty is not None and feature_names != STAR_DIFFICULTY_FEATURES:
        raise InputError(
            "Fixed star difficulty generation requires a Conformer-v4 star-conditioned model."
        )
    if feature_names == STAR_DIFFICULTY_FEATURES and requested_difficulty is None:
        raise InputError(
            "Conformer-v4 requires --difficulty-tier or --target-stars for generation."
        )
    profile, selected_stars = (
        requested_difficulty if requested_difficulty is not None else (None, None)
    )
    raw_audio_config = config.get("audio_features") or {}
    audio_config = (
        AudioFeatureConfig(**raw_audio_config) if raw_audio_config else AudioFeatureConfig()
    )
    feature_file = workspace / "modern-rhythm-features.npz"
    progress("Extracting modern rhythm audio features")
    feature_result = extract_audio_features(audio, feature_file, config=audio_config)
    density = float(
        target_density
        if target_density is not None
        else profile.target_density
        if profile is not None
        else _source_density(document, feature_result.duration_seconds)
    )
    if not 0 < density <= 20:
        raise InputError("Target density must be greater than 0 and no more than 20 objects/sec.")
    selected_threshold = prediction_threshold(
        config,
        threshold,
        target_density=density,
        difficulty_tier=profile.key if profile is not None else None,
    )
    cache_key = (
        str(workspace.resolve()),
        str(model_path),
        round(density, 9),
        round(selected_threshold, 9),
        profile.key if profile is not None else None,
        round(selected_stars, 6) if selected_stars is not None else None,
    )
    cached = _prediction_cache.get(cache_key)
    if cached is not None:
        progress(
            "Reusing cached modern rhythm selection for fine star calibration "
            f"({len(cached.timestamps_ms)} timestamps)"
        )
        return cached
    grid_config = GridConfig(
        sequence_length=int(config["training"]["sequence_length"]),
        prediction_threshold=selected_threshold,
    )
    detail = {
        "timing_points": _timing_detail(document),
        "hit_objects": [],
        "statistics": {"objects_per_second": density},
    }
    grid = create_timing_grid(
        detail,
        config=grid_config,
        inference_end_ms=max(1.0, feature_result.duration_seconds * 1000.0 - 1.0),
        target_density=density,
    )
    aligned_audio = align_features_to_grid(
        feature_file,
        grid,
        context_radius=int(config.get("audio_context_radius", 0)),
    )
    prepared = PreparedMap(
        row={"map_path": str(document.path)},
        grid=grid,
        audio=aligned_audio,
        grid_features=grid_feature_array(grid),
        difficulty=_difficulty(
            document,
            density,
            feature_names,
            target_stars=selected_stars,
            difficulty_tier=profile.key if profile is not None else None,
        ),
        labels=np.zeros((len(grid.candidates), 1), dtype=np.float32),
    )
    progress(f"Running modern rhythm model across {len(grid.candidates)} grid positions")
    model = _cached_inference_model(str(model_path))
    probabilities = window_probabilities(model, prepared, grid_config.sequence_length)
    candidate_times = np.asarray(
        [candidate.timestamp_ms for candidate in grid.candidates], dtype=np.float64
    )
    selected = _select_events(
        candidate_times,
        probabilities,
        threshold=selected_threshold,
        target_density=density,
        duration_seconds=feature_result.duration_seconds,
    )
    if selected.size == 0:
        raise InputError(
            "Modern rhythm model selected no events. Lower --rhythm-threshold "
            "or train a better model."
        )
    result = ModernRhythmPrediction(
        timestamps_ms=np.rint(candidate_times[selected]).astype(int),
        probabilities=probabilities[selected],
        candidate_indices=selected,
        target_density=density,
        threshold=selected_threshold,
        target_stars=selected_stars,
        difficulty_tier=profile.key if profile is not None else None,
    )
    if len(_prediction_cache) >= _PREDICTION_CACHE_LIMIT:
        _prediction_cache.pop(next(iter(_prediction_cache)))
    _prediction_cache[cache_key] = result
    return result
