from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from osumapper.paths import project_root


@dataclass(frozen=True, slots=True)
class DatasetPaths:
    root: Path
    dataset: Path
    maps: Path
    features: Path
    windows: Path
    splits: Path
    ratings: Path
    skipped: Path
    config: Path

    @classmethod
    def at(cls, root: Path | None = None) -> DatasetPaths:
        resolved = (root or (project_root() / "training_data")).expanduser().resolve()
        return cls(
            root=resolved,
            dataset=resolved / "dataset.parquet",
            maps=resolved / "maps",
            features=resolved / "features",
            windows=resolved / "window_shards",
            splits=resolved / "splits",
            ratings=resolved / "ratings.json",
            skipped=resolved / "skipped.jsonl",
            config=resolved / "config.json",
        )

    def create(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.maps.mkdir(parents=True, exist_ok=True)
        self.features.mkdir(parents=True, exist_ok=True)
        self.windows.mkdir(parents=True, exist_ok=True)
        self.splits.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True, slots=True)
class QualityConfig:
    min_objects: int = 25
    min_duration_ms: int = 5_000
    max_duration_ms: int = 30 * 60 * 1_000
    min_bpm: float = 20.0
    max_bpm: float = 1_000.0
    reject_converted: bool = True

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class AudioFeatureConfig:
    sample_rate: int = 22_050
    hop_length: int = 512
    n_fft: int = 2_048
    n_mels: int = 128
    fmin: float = 20.0
    fmax: float = 11_025.0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class GridConfig:
    subdivisions: tuple[int, ...] = (1, 2, 3, 4, 6, 8)
    hit_tolerance_ms: float = 32.0
    sequence_length: int = 256
    prediction_threshold: float = 0.5

    def as_dict(self) -> dict[str, Any]:
        values = asdict(self)
        values["subdivisions"] = list(self.subdivisions)
        return values


@dataclass(frozen=True, slots=True)
class RhythmTrainingConfig:
    epochs: int = 50
    batch_size: int = 16
    learning_rate: float = 1e-3
    seed: int = 2026
    device: str = "auto"
    sequence_length: int = 256
    prediction_threshold: float = 0.5
    include_unrated: bool = False
    architecture: str = "conformer-v4"
    audio_context_radius: int = 0
    precision: str = "auto"
    xla: str = "auto"
    early_stopping_patience: int = 8
    learning_rate_patience: int = 3
    learning_rate_factor: float = 0.5
    minimum_learning_rate: float = 1e-5
    balance_songs: bool = True
    window_cache: str = "auto"
    weight_decay: float = 1e-4
    stream_weight: float = 1.0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def prediction_threshold(
    model_config: dict[str, Any],
    override: float | None = None,
    *,
    target_density: float | None = None,
    difficulty_tier: str | None = None,
) -> float:
    """Resolve an explicit, validation-calibrated, or training-default threshold."""
    calibration = model_config.get("calibration", {})
    value: Any = override
    if value is None and difficulty_tier is not None:
        matching_tier = next(
            (
                band
                for band in calibration.get("difficulty_tiers", [])
                if str(band.get("name")) == difficulty_tier
            ),
            None,
        )
        if matching_tier is not None:
            value = matching_tier["threshold"]
    if value is None and target_density is not None:
        matching = next(
            (
                band
                for band in calibration.get("density_bands", [])
                if float(band["minimum_density"]) <= target_density < float(band["maximum_density"])
            ),
            None,
        )
        if matching is not None:
            value = matching["threshold"]
    if value is None:
        value = calibration.get(
            "threshold",
            model_config.get("training", {}).get("prediction_threshold", 0.5),
        )
    threshold = float(value)
    if not 0 < threshold < 1:
        raise ValueError("Prediction threshold must be between 0 and 1.")
    return threshold
