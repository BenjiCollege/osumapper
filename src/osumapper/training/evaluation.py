from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from osumapper.difficulty import LEGACY_DIFFICULTY_FEATURES
from osumapper.errors import DependencyError, InputError
from osumapper.training.config import DatasetPaths, GridConfig, prediction_threshold
from osumapper.training.loader import prepare_map, window_probabilities
from osumapper.training.model import load_rhythm_model
from osumapper.training.splits import load_split
from osumapper.training.storage import read_json, write_json
from osumapper.training.trainer import default_model_root


def _require_metrics() -> tuple[Any, Any, Any, Any]:
    try:
        from sklearn.metrics import (
            average_precision_score,
            f1_score,
            precision_score,
            recall_score,
        )
    except ImportError as exc:
        raise DependencyError("Evaluation requires scikit-learn.") from exc
    return precision_score, recall_score, f1_score, average_precision_score


def match_events(
    predicted_ms: list[float], human_ms: list[float], *, tolerance_ms: float = 64.0
) -> tuple[int, list[float]]:
    remaining = set(range(len(human_ms)))
    errors: list[float] = []
    for predicted in sorted(predicted_ms):
        choices = sorted(remaining, key=lambda index: abs(human_ms[index] - predicted))
        if not choices:
            break
        selected = choices[0]
        error = abs(human_ms[selected] - predicted)
        if error <= tolerance_ms:
            remaining.remove(selected)
            errors.append(error)
    return len(errors), errors


def evaluate_rhythm(
    *,
    dataset_root: Path | None = None,
    model_root: Path | None = None,
    threshold: float | None = None,
    output: Path | None = None,
) -> dict[str, Any]:
    paths = DatasetPaths.at(dataset_root)
    requested = (model_root or default_model_root()).expanduser().resolve()
    root = requested.parent if requested.suffix.casefold() == ".keras" else requested
    model_path = requested if requested.suffix.casefold() == ".keras" else root / "model.keras"
    config = read_json(root / "config.json", default=None)
    if config is None:
        raise InputError(f"Modern model configuration is missing under {root}.")
    model_manifest = read_json(root / "dataset_manifest.json", default=None)
    current_manifest = read_json(paths.splits / "manifest.json", default=None)
    if model_manifest is None or current_manifest is None:
        raise InputError(
            "Held-out evaluation requires both the model dataset manifest and the current "
            "split manifest."
        )
    training_split = model_manifest.get("split_manifest")
    if training_split != current_manifest:
        raise InputError(
            "The current dataset split does not match the model's training split. "
            "Refusing to report held-out metrics that may contain song leakage."
        )
    test_rows = load_split("test", dataset_root=paths.root)
    if not test_rows:
        raise InputError("Held-out test split is empty; add more rated songs and split again.")
    test_songs = {str(row["song_id"]) for row in test_rows}
    training_songs = set(model_manifest.get("train_song_ids", []))
    validation_songs = set(model_manifest.get("validation_song_ids", []))
    overlap = test_songs & (training_songs | validation_songs)
    if overlap:
        examples = ", ".join(sorted(overlap)[:5])
        raise InputError(f"Held-out evaluation detected song leakage: {examples}")
    grid_config = GridConfig(
        sequence_length=int(config["training"]["sequence_length"]),
        prediction_threshold=prediction_threshold(config, threshold),
    )
    model = load_rhythm_model(model_path, compile_model=False)
    difficulty_features = tuple(config.get("difficulty_features", LEGACY_DIFFICULTY_FEATURES))
    all_labels: list[int] = []
    all_probabilities: list[float] = []
    all_predictions: list[bool] = []
    timing_errors: list[float] = []
    density_errors: list[float] = []
    map_reports: list[dict[str, Any]] = []
    for row in test_rows:
        map_threshold = prediction_threshold(
            config,
            threshold,
            target_density=float(row["objects_per_second"]),
            difficulty_tier=str(row.get("difficulty_tier", "")),
        )
        prepared = prepare_map(
            row,
            dataset_root=paths.root,
            grid_config=grid_config,
            audio_context_radius=int(config.get("audio_context_radius", 0)),
            difficulty_features=difficulty_features,
        )
        probabilities = window_probabilities(model, prepared, grid_config.sequence_length)
        labels = prepared.labels[:, 0].astype(int)
        predictions = probabilities >= map_threshold
        candidate_times = [candidate.timestamp_ms for candidate in prepared.grid.candidates]
        predicted_times = [candidate_times[index] for index in np.flatnonzero(predictions).tolist()]
        human_times = [
            float(obj["time_ms"])
            for obj in read_json(Path(str(row["map_json_path"])), default={}).get("hit_objects", [])
        ]
        matched, errors = match_events(predicted_times, human_times)
        timing_errors.extend(errors)
        duration_seconds = max(1e-6, float(row["map_duration_ms"]) / 1000.0)
        density_error = abs(len(predicted_times) - len(human_times)) / duration_seconds
        density_errors.append(density_error)
        all_labels.extend(labels.tolist())
        all_probabilities.extend(probabilities.tolist())
        all_predictions.extend(predictions.tolist())
        map_reports.append(
            {
                "map_id": row["map_id"],
                "song_id": row["song_id"],
                "map_path": row["map_path"],
                "star_rating": row.get("star_rating"),
                "difficulty_tier": row.get("difficulty_tier"),
                "threshold": map_threshold,
                "human_objects": len(human_times),
                "predicted_objects": len(predicted_times),
                "matched_events": matched,
                "missed": len(human_times) - matched,
                "extra_predictions": len(predicted_times) - matched,
                "density_error_objects_per_second": density_error,
                "mean_timing_error_ms": sum(errors) / len(errors) if errors else None,
            }
        )
    precision_score, recall_score, f1_score, average_precision_score = _require_metrics()
    labels_array = np.asarray(all_labels, dtype=int)
    probability_array = np.asarray(all_probabilities, dtype=float)
    prediction_array = np.asarray(all_predictions, dtype=bool)
    report = {
        "split": "test",
        "model": str(model_path),
        "threshold": grid_config.prediction_threshold,
        "difficulty_thresholds": config.get("calibration", {}).get("difficulty_tiers", []),
        "dataset_sha256": current_manifest["dataset_sha256"],
        "test_songs": sorted(test_songs),
        "test_maps": len(test_rows),
        "candidate_positions": len(all_labels),
        "precision": float(precision_score(labels_array, prediction_array, zero_division=0)),
        "recall": float(recall_score(labels_array, prediction_array, zero_division=0)),
        "f1": float(f1_score(labels_array, prediction_array, zero_division=0)),
        "pr_auc": float(average_precision_score(labels_array, probability_array)),
        "mean_timing_error_ms": sum(timing_errors) / len(timing_errors) if timing_errors else None,
        "median_timing_error_ms": float(np.median(timing_errors)) if timing_errors else None,
        "mean_density_error_objects_per_second": sum(density_errors) / len(density_errors),
        "maps": map_reports,
    }
    for key, value in report.items():
        if isinstance(value, float) and not math.isfinite(value):
            raise InputError(f"Evaluation produced a non-finite metric: {key}")
    destination = (output or (root / "evaluation.json")).expanduser().resolve()
    write_json(destination, report)
    report["output"] = str(destination)
    return report
