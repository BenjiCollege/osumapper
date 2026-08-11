from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from osumapper.errors import DependencyError, InputError
from osumapper.training.config import DatasetPaths, GridConfig
from osumapper.training.loader import prepare_map, window_probabilities
from osumapper.training.model import load_rhythm_model
from osumapper.training.splits import load_split
from osumapper.training.storage import read_json, write_json
from osumapper.training.trainer import default_model_root


def _metrics() -> tuple[Any, Any]:
    try:
        from sklearn.metrics import average_precision_score, precision_recall_curve
    except ImportError as exc:
        raise DependencyError("Threshold calibration requires scikit-learn.") from exc
    return average_precision_score, precision_recall_curve


def _model_paths(model_root: Path | None) -> tuple[Path, Path]:
    requested = (model_root or default_model_root()).expanduser().resolve()
    root = requested.parent if requested.suffix.casefold() == ".keras" else requested
    model = requested if requested.suffix.casefold() == ".keras" else root / "model.keras"
    if not model.is_file():
        raise InputError(f"Modern rhythm model is missing: {model}")
    return root, model


def calibrate_threshold(
    *,
    dataset_root: Path | None = None,
    model_root: Path | None = None,
    output: Path | None = None,
    progress: Any | None = print,
) -> dict[str, Any]:
    """Choose a threshold by maximum candidate-level F1 on validation songs only."""
    paths = DatasetPaths.at(dataset_root)
    root, model_path = _model_paths(model_root)
    config_path = root / "config.json"
    config = read_json(config_path, default=None)
    model_manifest = read_json(root / "dataset_manifest.json", default=None)
    split_manifest = read_json(paths.splits / "manifest.json", default=None)
    if config is None or model_manifest is None or split_manifest is None:
        raise InputError("Calibration requires model configuration and split manifests.")
    if model_manifest.get("split_manifest") != split_manifest:
        raise InputError(
            "The current dataset split does not match the model training split; "
            "validation-only calibration would be unsafe."
        )
    rows = load_split("validation", dataset_root=paths.root)
    if not rows:
        raise InputError("Validation split is empty; threshold calibration is unavailable.")
    validation_songs = {str(row["song_id"]) for row in rows}
    expected_songs = set(model_manifest.get("validation_song_ids", []))
    training_songs = set(model_manifest.get("train_song_ids", []))
    if validation_songs != expected_songs or validation_songs & training_songs:
        raise InputError("Validation-only calibration detected a changed split or song leakage.")

    grid_config = GridConfig(sequence_length=int(config["training"]["sequence_length"]))
    model = load_rhythm_model(model_path, compile_model=False)
    labels: list[np.ndarray[Any, Any]] = []
    probabilities: list[np.ndarray[Any, Any]] = []
    for index, row in enumerate(rows, start=1):
        prepared = prepare_map(
            row,
            dataset_root=paths.root,
            grid_config=grid_config,
            audio_context_radius=int(config.get("audio_context_radius", 0)),
        )
        labels.append(prepared.labels[:, 0].astype(np.uint8))
        probabilities.append(window_probabilities(model, prepared, grid_config.sequence_length))
        if progress is not None and (index % 25 == 0 or index == len(rows)):
            progress(f"Calibrated {index}/{len(rows)} validation maps")
    label_array = np.concatenate(labels)
    probability_array = np.concatenate(probabilities)
    average_precision_score, precision_recall_curve = _metrics()
    precision, recall, thresholds = precision_recall_curve(label_array, probability_array)
    if thresholds.size == 0:
        raise InputError("Validation predictions did not produce a usable threshold.")
    f1 = np.divide(
        2.0 * precision[:-1] * recall[:-1],
        precision[:-1] + recall[:-1],
        out=np.zeros_like(thresholds, dtype=np.float64),
        where=(precision[:-1] + recall[:-1]) > 0,
    )
    best_f1 = float(np.max(f1))
    candidates = np.flatnonzero(np.isclose(f1, best_f1, rtol=0.0, atol=1e-12))
    # A higher threshold is the conservative tie-break: fewer unwanted objects.
    selected = int(candidates[np.argmax(thresholds[candidates])])
    threshold = float(thresholds[selected])
    report = {
        "version": 1,
        "source_split": "validation",
        "selection_metric": "candidate_f1",
        "tie_break": "highest_threshold",
        "dataset_sha256": split_manifest["dataset_sha256"],
        "model": str(model_path),
        "validation_songs": len(validation_songs),
        "validation_maps": len(rows),
        "candidate_positions": int(label_array.size),
        "positive_positions": int(label_array.sum()),
        "threshold": threshold,
        "precision": float(precision[selected]),
        "recall": float(recall[selected]),
        "f1": best_f1,
        "pr_auc": float(average_precision_score(label_array, probability_array)),
    }
    for key, value in report.items():
        if isinstance(value, float) and not math.isfinite(value):
            raise InputError(f"Calibration produced a non-finite value: {key}")
    destination = (output or (root / "threshold_calibration.json")).expanduser().resolve()
    write_json(destination, report)
    config["calibration"] = {
        key: report[key]
        for key in (
            "version",
            "source_split",
            "selection_metric",
            "tie_break",
            "dataset_sha256",
            "validation_songs",
            "validation_maps",
            "candidate_positions",
            "threshold",
            "precision",
            "recall",
            "f1",
            "pr_auc",
        )
    }
    write_json(config_path, config)
    report["output"] = str(destination)
    return report
