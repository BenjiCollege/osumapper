from __future__ import annotations

import hashlib
import json
import os
import tempfile
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
from osumapper.training.storage import read_json, write_json

WINDOW_CACHE_VERSION = 1


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
    song_id: str


@dataclass(frozen=True, slots=True)
class LoaderSummary:
    maps: int
    windows: int
    positives: int
    negatives: int
    audio_dimension: int
    grid_dimension: int
    positive_weight: float
    songs: int = 0
    cache_path: str | None = None
    song_balanced: bool = False


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
                song_id=str(row["song_id"]),
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
        songs=len({str(row["song_id"]) for row in rows}),
    )


def _window_cache_key(
    rows: list[dict[str, Any]],
    *,
    paths: DatasetPaths,
    grid_config: GridConfig,
    audio_context_radius: int,
) -> str:
    feature_manifest = read_json(paths.features / "manifest.json", default={})
    identity = {
        "version": WINDOW_CACHE_VERSION,
        "maps": [
            {
                "map_id": str(row["map_id"]),
                "map_sha256": str(row["map_sha256"]),
                "song_id": str(row["song_id"]),
            }
            for row in rows
        ],
        "grid": grid_config.as_dict(),
        "audio_context_radius": audio_context_radius,
        "audio_features": feature_manifest.get("config", {}),
    }
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()[:20]


def _write_window_shard(path: Path, examples: list[WindowExample], song_indices: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with temporary.open("wb") as stream:
            # Float16 halves disk traffic and cache size. Labels remain exact
            # integers and Keras promotes inputs according to the model policy.
            np.savez(
                stream,
                audio=np.asarray(
                    [example.inputs["audio_features"] for example in examples],
                    dtype=np.float16,
                ),
                grid=np.asarray(
                    [example.inputs["grid_features"] for example in examples],
                    dtype=np.float16,
                ),
                difficulty=np.asarray(
                    [example.inputs["difficulty"] for example in examples],
                    dtype=np.float16,
                ),
                labels=np.asarray([example.labels for example in examples], dtype=np.uint8),
                mask=np.asarray([example.mask for example in examples], dtype=np.uint8),
                song_index=np.asarray(song_indices, dtype=np.int32),
            )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _build_window_cache(
    rows: list[dict[str, Any]],
    *,
    paths: DatasetPaths,
    cache_root: Path,
    grid_config: GridConfig,
    audio_context_radius: int,
    progress: Any | None = print,
) -> dict[str, Any]:
    song_ids = sorted({str(row["song_id"]) for row in rows})
    song_lookup = {song_id: index for index, song_id in enumerate(song_ids)}
    song_windows = [0 for _ in song_ids]
    shards: list[str] = []
    buffer: list[WindowExample] = []
    buffer_song_indices: list[int] = []
    positives = 0
    valid_positions = 0
    audio_dimension = 0
    grid_dimension = 0
    shard_capacity: int | None = None

    def flush() -> None:
        if not buffer:
            return
        shard_name = f"shard-{len(shards):05d}.npz"
        _write_window_shard(cache_root / shard_name, buffer, buffer_song_indices)
        shards.append(shard_name)
        buffer.clear()
        buffer_song_indices.clear()

    for number, example in enumerate(
        iter_windows(
            rows,
            dataset_root=paths.root,
            grid_config=grid_config,
            audio_context_radius=audio_context_radius,
        ),
        start=1,
    ):
        if shard_capacity is None:
            bytes_per_window = (
                example.inputs["audio_features"].size
                + example.inputs["grid_features"].size
                + example.inputs["difficulty"].size
            ) * np.dtype(np.float16).itemsize
            bytes_per_window += example.labels.size + example.mask.size + 4
            shard_capacity = max(8, min(512, (64 * 1024 * 1024) // bytes_per_window))
        song_index = song_lookup[example.song_id]
        song_windows[song_index] += 1
        positives += int((example.labels[:, 0] * example.mask).sum())
        valid_positions += int(example.mask.sum())
        audio_dimension = int(example.inputs["audio_features"].shape[1])
        grid_dimension = int(example.inputs["grid_features"].shape[1])
        buffer.append(example)
        buffer_song_indices.append(song_index)
        if len(buffer) >= shard_capacity:
            flush()
        if progress is not None and number % 1_000 == 0:
            progress(f"Cached {number} training windows")
    flush()
    if not shards:
        raise InputError("The selected split produced no training windows.")
    negatives = valid_positions - positives
    if positives == 0:
        raise InputError("The selected split has no positive rhythm labels.")
    manifest = {
        "version": WINDOW_CACHE_VERSION,
        "maps": len(rows),
        "songs": len(song_ids),
        "windows": sum(song_windows),
        "positives": positives,
        "negatives": negatives,
        "audio_dimension": audio_dimension,
        "grid_dimension": grid_dimension,
        "positive_weight": min(50.0, max(1.0, negatives / positives)),
        "sequence_length": grid_config.sequence_length,
        "audio_context_radius": audio_context_radius,
        "song_ids": song_ids,
        "song_windows": song_windows,
        "shards": shards,
    }
    write_json(cache_root / "manifest.json", manifest)
    return manifest


def _summary_from_manifest(
    manifest: dict[str, Any], cache_root: Path, *, song_balanced: bool
) -> LoaderSummary:
    return LoaderSummary(
        maps=int(manifest["maps"]),
        windows=int(manifest["windows"]),
        positives=int(manifest["positives"]),
        negatives=int(manifest["negatives"]),
        audio_dimension=int(manifest["audio_dimension"]),
        grid_dimension=int(manifest["grid_dimension"]),
        positive_weight=float(manifest["positive_weight"]),
        songs=int(manifest["songs"]),
        cache_path=str(cache_root),
        song_balanced=song_balanced,
    )


def _cached_tf_dataset(
    tf: Any,
    *,
    cache_root: Path,
    manifest: dict[str, Any],
    grid_config: GridConfig,
    song_balanced: bool,
) -> Any:
    def shard_generator() -> Iterator[tuple[dict[str, Any], Any, Any, Any]]:
        for shard_name in manifest["shards"]:
            shard_path = cache_root / str(shard_name)
            if not shard_path.is_file():
                raise InputError(f"Window-cache shard is missing: {shard_path}")
            with np.load(shard_path, allow_pickle=False) as archive:
                yield (
                    {
                        "audio_features": archive["audio"].copy(),
                        "grid_features": archive["grid"].copy(),
                        "difficulty": archive["difficulty"].copy(),
                    },
                    archive["labels"].astype(np.float32),
                    archive["mask"].astype(np.float32),
                    archive["song_index"].copy(),
                )

    length = grid_config.sequence_length
    signature = (
        {
            "audio_features": tf.TensorSpec(
                (None, length, int(manifest["audio_dimension"])), tf.float16
            ),
            "grid_features": tf.TensorSpec(
                (None, length, int(manifest["grid_dimension"])), tf.float16
            ),
            "difficulty": tf.TensorSpec((None, 4), tf.float16),
        },
        tf.TensorSpec((None, length, 1), tf.float32),
        tf.TensorSpec((None, length), tf.float32),
        tf.TensorSpec((None,), tf.int32),
    )
    dataset = tf.data.Dataset.from_generator(shard_generator, output_signature=signature).unbatch()
    if song_balanced:
        counts = np.asarray(manifest["song_windows"], dtype=np.float32)
        mean = float(counts.mean())
        weights = np.clip(mean / np.maximum(counts, 1.0), 0.25, 4.0).astype(np.float32)
        song_weights = tf.constant(weights)

        def apply_weight(inputs: Any, labels: Any, mask: Any, song_index: Any) -> Any:
            return inputs, labels, mask * tf.gather(song_weights, song_index)

        return dataset.map(apply_weight, num_parallel_calls=tf.data.AUTOTUNE)

    def remove_song_index(inputs: Any, labels: Any, mask: Any, _song_index: Any) -> Any:
        return inputs, labels, mask

    return dataset.map(remove_song_index, num_parallel_calls=tf.data.AUTOTUNE)


def make_tf_dataset(
    rows: list[dict[str, Any]],
    *,
    dataset_root: Path | None = None,
    grid_config: GridConfig | None = None,
    batch_size: int = 16,
    seed: int = 2026,
    shuffle: bool = False,
    audio_context_radius: int = 0,
    cache_split: str | None = None,
    cache_mode: str = "auto",
    balance_songs: bool = False,
    progress: Any | None = print,
) -> tuple[Any, LoaderSummary]:
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise DependencyError("Training requires TensorFlow. Run `uv sync --locked`.") from exc
    config = grid_config or GridConfig()
    normalized_cache = cache_mode.casefold()
    if normalized_cache not in {"auto", "off", "rebuild"}:
        raise InputError("Window cache mode must be auto, off, or rebuild.")
    paths = DatasetPaths.at(dataset_root)
    use_cache = cache_split is not None and normalized_cache != "off"
    manifest: dict[str, Any] | None = None
    cache_root: Path | None = None
    if use_cache:
        cache_key = _window_cache_key(
            rows,
            paths=paths,
            grid_config=config,
            audio_context_radius=audio_context_radius,
        )
        cache_root = paths.windows / str(cache_split) / cache_key
        manifest_path = cache_root / "manifest.json"
        manifest = None if normalized_cache == "rebuild" else read_json(manifest_path, default=None)
        if manifest is None:
            if progress is not None:
                progress(f"Building deterministic {cache_split} window shards")
            manifest = _build_window_cache(
                rows,
                paths=paths,
                cache_root=cache_root,
                grid_config=config,
                audio_context_radius=audio_context_radius,
                progress=progress,
            )
        summary = _summary_from_manifest(manifest, cache_root, song_balanced=balance_songs)
        dataset = _cached_tf_dataset(
            tf,
            cache_root=cache_root,
            manifest=manifest,
            grid_config=config,
            song_balanced=balance_songs,
        )
    else:
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
