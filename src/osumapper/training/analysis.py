from __future__ import annotations

import csv
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from osumapper.errors import InputError
from osumapper.paths import project_root
from osumapper.training.beatmaps import parse_standard_beatmap
from osumapper.training.config import (
    AudioFeatureConfig,
    DatasetPaths,
    GridConfig,
    prediction_threshold,
)
from osumapper.training.evaluation import match_events
from osumapper.training.features import extract_audio_features
from osumapper.training.grid import (
    align_features_to_grid,
    create_timing_grid,
    grid_feature_array,
)
from osumapper.training.loader import PreparedMap, window_probabilities
from osumapper.training.model import load_rhythm_model
from osumapper.training.storage import read_json, write_json
from osumapper.training.trainer import default_model_root


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with temporary.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else [])
            if rows:
                writer.writeheader()
                writer.writerows(rows)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def analyze_map(
    map_path: Path,
    *,
    model_root: Path | None = None,
    dataset_root: Path | None = None,
    threshold: float | None = None,
    output: Path | None = None,
    timeline_format: str = "json",
) -> dict[str, Any]:
    source = map_path.expanduser().resolve()
    parsed = parse_standard_beatmap(source, source.parent)
    requested_model = (model_root or default_model_root()).expanduser().resolve()
    root = (
        requested_model.parent if requested_model.suffix.casefold() == ".keras" else requested_model
    )
    model_path = (
        requested_model if requested_model.suffix.casefold() == ".keras" else root / "model.keras"
    )
    model_config = read_json(root / "config.json", default=None)
    if model_config is None or not model_path.is_file():
        raise InputError(f"Modern rhythm model is missing under {root}; train it first.")
    stats = parsed.statistics
    selected_threshold = prediction_threshold(
        model_config,
        threshold,
        target_density=float(stats["objects_per_second"]),
    )
    grid_config = GridConfig(
        sequence_length=int(model_config["training"]["sequence_length"]),
        prediction_threshold=selected_threshold,
    )
    raw_audio_config = model_config.get("audio_features") or {}
    audio_config = (
        AudioFeatureConfig(**raw_audio_config) if raw_audio_config else AudioFeatureConfig()
    )
    paths = DatasetPaths.at(dataset_root)
    paths.create()
    feature_file = paths.features / f"analysis-{parsed.map_sha256[:20]}.npz"
    extract_audio_features(parsed.audio_path, feature_file, config=audio_config)
    detail = parsed.detail_dict()
    grid = create_timing_grid(detail, config=grid_config)
    aligned_audio = align_features_to_grid(
        feature_file,
        grid,
        context_radius=int(model_config.get("audio_context_radius", 0)),
    )
    grid_features = grid_feature_array(grid)
    labels = np.asarray([candidate.label for candidate in grid.candidates], dtype=np.float32)[
        :, None
    ]
    difficulty = np.asarray(
        [
            parsed.difficulty["od"] / 10.0,
            parsed.difficulty["ar"] / 10.0,
            parsed.difficulty["cs"] / 10.0,
            stats["objects_per_second"] / 10.0,
        ],
        dtype=np.float32,
    )
    prepared = PreparedMap(
        row={"map_path": str(source)},
        grid=grid,
        audio=aligned_audio,
        grid_features=grid_features,
        difficulty=difficulty,
        labels=labels,
    )
    model = load_rhythm_model(model_path, compile_model=False)
    probabilities = window_probabilities(model, prepared, grid_config.sequence_length)
    predictions = probabilities >= selected_threshold
    candidate_times = [candidate.timestamp_ms for candidate in grid.candidates]
    predicted_times = [candidate_times[index] for index in np.flatnonzero(predictions).tolist()]
    human_times = [obj.time_ms for obj in parsed.hit_objects]
    matched, timing_errors = match_events(predicted_times, human_times)
    precision = matched / len(predicted_times) if predicted_times else 0.0
    recall = matched / len(human_times) if human_times else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    per_frame_dimension = audio_config.n_mels + 4
    center_offset = int(model_config.get("audio_context_radius", 0)) * per_frame_dimension
    timeline = [
        {
            "timestamp_ms": round(candidate.timestamp_ms, 6),
            "beat_position": round(candidate.beat_position, 8),
            "beat_snap": f"1/{candidate.divisor}",
            "measure_position": round(candidate.measure_position, 8),
            "bpm": round(candidate.bpm, 6),
            "audio_onset_strength": round(
                float(aligned_audio[index, center_offset + audio_config.n_mels]), 8
            ),
            "human_object": bool(candidate.label),
            "model_probability": round(float(probabilities[index]), 8),
            "model_prediction": bool(predictions[index]),
        }
        for index, candidate in enumerate(grid.candidates)
    ]
    metadata = parsed.metadata
    report: dict[str, Any] = {
        "song": {
            "artist": metadata["artist_unicode"] or metadata["artist"],
            "title": metadata["title_unicode"] or metadata["title"],
            "creator": metadata["creator"],
            "difficulty": metadata["version"],
            "map_path": str(source),
            "audio_path": str(parsed.audio_path),
        },
        "model": str(model_path),
        "threshold": selected_threshold,
        "human_objects": len(human_times),
        "model_predicted": len(predicted_times),
        "matched_musical_events": matched,
        "missed": len(human_times) - matched,
        "extra_predictions": len(predicted_times) - matched,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_timing_error_ms": sum(timing_errors) / len(timing_errors) if timing_errors else None,
        "grid_matched_human_objects": grid.matched_objects,
        "grid_unmatched_human_objects": grid.unmatched_objects,
    }
    normalized_format = timeline_format.casefold()
    if normalized_format not in {"json", "csv", "both"}:
        raise InputError("Timeline format must be json, csv, or both.")
    default_base = project_root() / "output" / f"{source.stem}-rhythm-analysis"
    destination = (output or default_base.with_suffix(".json")).expanduser().resolve()
    if destination == source:
        raise InputError("Analysis output must not overwrite the source .osu file.")
    if normalized_format in {"json", "both"}:
        json_path = (
            destination
            if destination.suffix.casefold() == ".json"
            else destination.with_suffix(".json")
        )
        write_json(json_path, {**report, "timeline": timeline})
        report["json_output"] = str(json_path)
    if normalized_format in {"csv", "both"}:
        csv_path = (
            destination
            if destination.suffix.casefold() == ".csv"
            else destination.with_suffix(".csv")
        )
        _write_csv(csv_path, timeline)
        summary_path = csv_path.with_suffix(".report.json")
        write_json(summary_path, report)
        report["csv_output"] = str(csv_path)
        report["summary_output"] = str(summary_path)
    return report
