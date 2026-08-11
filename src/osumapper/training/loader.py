from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from osumapper.errors import DependencyError, InputError
from osumapper.training.config import DatasetPaths, GridConfig
from osumapper.training.features import feature_path
from osumapper.training.grid import (
    TimingGrid,
    align_features_to_grid,
    create_timing_grid,
    grid_feature_array,
    load_map_detail,
)


@dataclass(frozen=True, slots=True)
class PreparedMap:
    row: dict[str, Any]
    grid: TimingGrid
    audio: np.ndarray[Any, Any]
    grid_features: np.ndarray[Any, Any]
    difficulty: np.ndarray[Any, Any]
    labels: np.ndarray[Any, Any]


@dataclass(frozen=True, slots=True)
class WindowExample:
    inputs: dict[str, np.ndarray[Any, Any]]
    labels: np.ndarray[Any, Any]
    mask: np.ndarray[Any, Any]


@dataclass(frozen=True, slots=True)
class LoaderSummary:
    maps: int
    windows: int
    positives: int
    negatives: int
    audio_dimension: int
    grid_dimension: int
    positive_weight: float


def prepare_map(
    row: dict[str, Any],
    *,
    dataset_root: Path | None = None,
    grid_config: GridConfig | None = None,
    audio_context_radius: int = 0,
) -> PreparedMap:
    paths = DatasetPaths.at(dataset_root)
    detail = load_map_detail(Path(str(row["map_json_path"])))
    grid = create_timing_grid(detail, config=grid_config)
    audio = align_features_to_grid(
        feature_path(str(row["song_id"]), dataset_root=paths.root),
        grid,
        context_radius=audio_context_radius,
    )
    grid_features = grid_feature_array(grid)
    labels = np.asarray([candidate.label for candidate in grid.candidates], dtype=np.float32)[
        :, None
    ]
    difficulty = np.asarray(
        [
            float(row["od"]) / 10.0,
            float(row["ar"]) / 10.0,
            float(row["cs"]) / 10.0,
            float(row["objects_per_second"]) / 10.0,
        ],
        dtype=np.float32,
    )
    return PreparedMap(row, grid, audio, grid_features, difficulty, labels)


def iter_windows(
    rows: Iterable[dict[str, Any]],
    *,
    dataset_root: Path | None = None,
    grid_config: GridConfig | None = None,
    audio_context_radius: int = 0,
) -> Iterator[WindowExample]:
    config = grid_config or GridConfig()
    length = config.sequence_length
    for row in rows:
        try:
            prepared = prepare_map(
                row,
                dataset_root=dataset_root,
                grid_config=config,
                audio_context_radius=audio_context_radius,
            )
        except InputError as exc:
            raise InputError(f"Could not prepare {row['map_path']}: {exc}") from exc
        count = prepared.labels.shape[0]
        for start in range(0, count, length):
            stop = min(count, start + length)
            valid = stop - start
            audio = np.zeros((length, prepared.audio.shape[1]), dtype=np.float32)
            grid = np.zeros((length, prepared.grid_features.shape[1]), dtype=np.float32)
            labels = np.zeros((length, 1), dtype=np.float32)
            mask = np.zeros(length, dtype=np.float32)
            audio[:valid] = prepared.audio[start:stop]
            grid[:valid] = prepared.grid_features[start:stop]
            labels[:valid] = prepared.labels[start:stop]
            mask[:valid] = 1.0
            yield WindowExample(
                inputs={
                    "audio_features": audio,
                    "grid_features": grid,
                    "difficulty": prepared.difficulty.copy(),
                },
                labels=labels,
                mask=mask,
            )


def summarize_loader(
    rows: list[dict[str, Any]],
    *,
    dataset_root: Path | None = None,
    grid_config: GridConfig | None = None,
    audio_context_radius: int = 0,
) -> LoaderSummary:
    windows = 0
    positives = 0
    valid_positions = 0
    audio_dimension = 0
    grid_dimension = 0
    for example in iter_windows(
        rows,
        dataset_root=dataset_root,
        grid_config=grid_config,
        audio_context_radius=audio_context_radius,
    ):
        windows += 1
        valid_positions += int(example.mask.sum())
        positives += int((example.labels[:, 0] * example.mask).sum())
        audio_dimension = example.inputs["audio_features"].shape[1]
        grid_dimension = example.inputs["grid_features"].shape[1]
    if windows == 0:
        raise InputError("The selected split produced no training windows.")
    negatives = valid_positions - positives
    if positives == 0:
        raise InputError("The selected split has no positive rhythm labels.")
    positive_weight = min(50.0, max(1.0, negatives / positives))
    return LoaderSummary(
        maps=len(rows),
        windows=windows,
        positives=positives,
        negatives=negatives,
        audio_dimension=audio_dimension,
        grid_dimension=grid_dimension,
        positive_weight=positive_weight,
    )


def make_tf_dataset(
    rows: list[dict[str, Any]],
    *,
    dataset_root: Path | None = None,
    grid_config: GridConfig | None = None,
    batch_size: int = 16,
    seed: int = 2026,
    shuffle: bool = False,
    audio_context_radius: int = 0,
) -> tuple[Any, LoaderSummary]:
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise DependencyError("Training requires TensorFlow. Run `uv sync --locked`.") from exc
    config = grid_config or GridConfig()
    summary = summarize_loader(
        rows,
        dataset_root=dataset_root,
        grid_config=config,
        audio_context_radius=audio_context_radius,
    )

    def generator() -> Iterator[tuple[dict[str, Any], Any, Any]]:
        for example in iter_windows(
            rows,
            dataset_root=dataset_root,
            grid_config=config,
            audio_context_radius=audio_context_radius,
        ):
            yield example.inputs, example.labels, example.mask

    signature = (
        {
            "audio_features": tf.TensorSpec(
                (config.sequence_length, summary.audio_dimension), tf.float32
            ),
            "grid_features": tf.TensorSpec(
                (config.sequence_length, summary.grid_dimension), tf.float32
            ),
            "difficulty": tf.TensorSpec((4,), tf.float32),
        },
        tf.TensorSpec((config.sequence_length, 1), tf.float32),
        tf.TensorSpec((config.sequence_length,), tf.float32),
    )
    dataset = tf.data.Dataset.from_generator(generator, output_signature=signature)
    options = tf.data.Options()
    options.experimental_deterministic = True
    dataset = dataset.with_options(options)
    if shuffle:
        dataset = dataset.shuffle(
            min(max(128, batch_size * 8), max(1, summary.windows)),
            seed=seed,
            reshuffle_each_iteration=True,
        )
    dataset = dataset.batch(batch_size, drop_remainder=False)
    # model.fit receives explicit step counts from the loader summary.  Repeating
    # here gives every epoch a fresh traversal instead of leaving the finite
    # from_generator dataset exhausted after the first epoch.
    dataset = dataset.repeat()
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset, summary


def window_probabilities(
    model: Any, prepared: PreparedMap, sequence_length: int
) -> np.ndarray[Any, Any]:
    probabilities: list[np.ndarray[Any, Any]] = []
    count = prepared.labels.shape[0]
    for start in range(0, count, sequence_length):
        stop = min(count, start + sequence_length)
        valid = stop - start
        audio = np.zeros((1, sequence_length, prepared.audio.shape[1]), dtype=np.float32)
        grid = np.zeros((1, sequence_length, prepared.grid_features.shape[1]), dtype=np.float32)
        audio[0, :valid] = prepared.audio[start:stop]
        grid[0, :valid] = prepared.grid_features[start:stop]
        prediction = model.predict(
            {
                "audio_features": audio,
                "grid_features": grid,
                "difficulty": prepared.difficulty[None, :],
            },
            verbose=0,
        )
        probabilities.append(np.asarray(prediction[0, :valid, 0], dtype=np.float32))
    return np.concatenate(probabilities) if probabilities else np.empty(0, dtype=np.float32)
