from __future__ import annotations

import bisect
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from osumapper.beatmap import BeatmapDocument
from osumapper.config import configure_determinism
from osumapper.errors import InputError
from osumapper.paths import project_root
from osumapper.training.beatmaps import parse_timing_points
from osumapper.training.config import DatasetPaths
from osumapper.training.model import _require_keras
from osumapper.training.splits import load_split
from osumapper.training.storage import read_json, write_json
from osumapper.training.trainer import _configure_device, _jit_compile, _safe_output

PLACEMENT_FEATURE_DIMENSION = 11
PLACEMENT_TARGET_DIMENSION = 8


@dataclass(frozen=True, slots=True)
class PlacementTrainingConfig:
    epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 5e-4
    sequence_length: int = 256
    seed: int = 2026
    device: str = "auto"
    precision: str = "auto"
    xla: str = "auto"
    early_stopping_patience: int = 8
    weight_decay: float = 1e-4
    balance_songs: bool = True

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PlacementDataset:
    features: np.ndarray[Any, Any]
    targets: np.ndarray[Any, Any]
    weights: np.ndarray[Any, Any]
    windows: int
    maps: int
    songs: int


def _snap_divisor(value: Any) -> int:
    try:
        return max(1, int(str(value).split("/", 1)[1]))
    except (IndexError, TypeError, ValueError):
        return 4


def _active_bpm(timestamp: float, red_points: list[dict[str, Any]]) -> float:
    times = [float(point["time_ms"]) for point in red_points]
    index = max(0, bisect.bisect_right(times, timestamp) - 1)
    return float(red_points[index].get("bpm") or 120.0)


def _temporal_features(
    timestamps: list[float],
    beat_positions: list[float],
    snap_divisors: list[int],
    bpms: list[float],
    *,
    density: float,
    od: float,
    ar: float,
    cs: float,
) -> np.ndarray[Any, Any]:
    count = len(timestamps)
    features = np.zeros((count, PLACEMENT_FEATURE_DIMENSION), dtype=np.float32)
    for index, timestamp in enumerate(timestamps):
        previous_delta = timestamp - timestamps[index - 1] if index else 2_000.0
        next_delta = timestamps[index + 1] - timestamp if index + 1 < count else 2_000.0
        phase = beat_positions[index] % 1.0
        features[index] = (
            min(2.0, max(0.0, previous_delta / 1_000.0)),
            min(2.0, max(0.0, next_delta / 1_000.0)),
            math.sin(2.0 * math.pi * phase),
            math.cos(2.0 * math.pi * phase),
            min(1.0, snap_divisors[index] / 8.0),
            min(2.0, bpms[index] / 300.0),
            min(2.0, density / 10.0),
            od / 10.0,
            ar / 10.0,
            cs / 10.0,
            index / max(1, count - 1),
        )
    return features


def _placement_targets(objects: list[dict[str, Any]]) -> np.ndarray[Any, Any]:
    targets = np.zeros((len(objects), PLACEMENT_TARGET_DIMENSION), dtype=np.float32)
    previous_direction = 0.0
    for index, obj in enumerate(objects):
        if index:
            previous = objects[index - 1]
            dx = float(obj["x"]) - float(previous["x"])
            dy = float(obj["y"]) - float(previous["y"])
            distance = math.hypot(dx, dy)
            direction = math.atan2(dy, dx) if distance > 1e-6 else previous_direction
            turn = math.atan2(
                math.sin(direction - previous_direction),
                math.cos(direction - previous_direction),
            )
            targets[index, 0] = min(1.0, distance / 640.0)
            targets[index, 1] = math.sin(turn)
            targets[index, 2] = math.cos(turn)
            if distance > 1e-6:
                previous_direction = direction
        else:
            targets[index, 2] = 1.0
        kind = str(obj.get("kind", "circle"))
        kind_index = 1 if kind == "slider" else 2 if kind == "spinner" else 0
        targets[index, 3 + kind_index] = 1.0
        targets[index, 6] = min(1.0, float(obj.get("slider_pixel_length") or 0.0) / 512.0)
        targets[index, 7] = float(bool(obj.get("new_combo", False)))
    return targets


def _map_arrays(row: dict[str, Any]) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    detail = read_json(Path(str(row["map_json_path"])), default=None)
    if not isinstance(detail, dict):
        raise InputError(f"Parsed map data is missing: {row['map_json_path']}")
    objects = list(detail.get("hit_objects", []))
    if len(objects) < 2:
        raise InputError(f"Placement training needs at least two objects: {row['map_path']}")
    red_points = sorted(
        (point for point in detail["timing_points"] if point["uninherited"]),
        key=lambda point: float(point["time_ms"]),
    )
    timestamps = [float(obj["time_ms"]) for obj in objects]
    beat_positions = [float(obj.get("beat_position") or 0.0) for obj in objects]
    divisors = [_snap_divisor(obj.get("beat_snap")) for obj in objects]
    bpms = [_active_bpm(timestamp, red_points) for timestamp in timestamps]
    features = _temporal_features(
        timestamps,
        beat_positions,
        divisors,
        bpms,
        density=float(row["objects_per_second"]),
        od=float(row["od"]),
        ar=float(row["ar"]),
        cs=float(row["cs"]),
    )
    return features, _placement_targets(objects)


def prepare_placement_dataset(
    rows: list[dict[str, Any]],
    *,
    sequence_length: int,
    balance_songs: bool,
) -> PlacementDataset:
    if sequence_length <= 0:
        raise InputError("Placement sequence length must be positive.")
    raw: list[tuple[str, np.ndarray[Any, Any], np.ndarray[Any, Any], int]] = []
    song_windows: dict[str, int] = {}
    for row in rows:
        features, targets = _map_arrays(row)
        windows = math.ceil(len(features) / sequence_length)
        song_id = str(row["song_id"])
        song_windows[song_id] = song_windows.get(song_id, 0) + windows
        raw.append((song_id, features, targets, windows))
    all_features: list[np.ndarray[Any, Any]] = []
    all_targets: list[np.ndarray[Any, Any]] = []
    all_weights: list[np.ndarray[Any, Any]] = []
    mean_windows = float(np.mean(list(song_windows.values()))) if song_windows else 1.0
    for song_id, features, targets, _windows in raw:
        song_weight = (
            float(np.clip(mean_windows / song_windows[song_id], 0.25, 4.0))
            if balance_songs
            else 1.0
        )
        for start in range(0, len(features), sequence_length):
            stop = min(len(features), start + sequence_length)
            valid = stop - start
            feature_window = np.zeros(
                (sequence_length, PLACEMENT_FEATURE_DIMENSION), dtype=np.float32
            )
            target_window = np.zeros(
                (sequence_length, PLACEMENT_TARGET_DIMENSION), dtype=np.float32
            )
            weight_window = np.zeros(sequence_length, dtype=np.float32)
            feature_window[:valid] = features[start:stop]
            target_window[:valid] = targets[start:stop]
            weight_window[:valid] = song_weight
            all_features.append(feature_window)
            all_targets.append(target_window)
            all_weights.append(weight_window)
    if not all_features:
        raise InputError("The selected split produced no placement windows.")
    return PlacementDataset(
        features=np.asarray(all_features, dtype=np.float32),
        targets=np.asarray(all_targets, dtype=np.float32),
        weights=np.asarray(all_weights, dtype=np.float32),
        windows=len(all_features),
        maps=len(rows),
        songs=len(song_windows),
    )


def _placement_loss(keras: Any) -> Any:
    @keras.saving.register_keras_serializable(package="osumapper")
    class PlacementLoss(keras.losses.Loss):
        def call(self, y_true: Any, y_pred: Any) -> Any:
            ops = keras.ops
            distance = ops.square(y_true[..., 0] - y_pred[..., 0])
            true_turn = y_true[..., 1:3]
            predicted_turn = y_pred[..., 1:3]
            predicted_turn /= ops.maximum(
                ops.sqrt(ops.sum(ops.square(predicted_turn), axis=-1, keepdims=True)), 1e-6
            )
            turn = 1.0 - ops.sum(true_turn * predicted_turn, axis=-1)
            kind = -ops.sum(
                y_true[..., 3:6] * ops.log(ops.clip(y_pred[..., 3:6], 1e-7, 1.0)),
                axis=-1,
            )
            slider = ops.square(y_true[..., 6] - y_pred[..., 6]) * y_true[..., 4]
            combo = keras.losses.binary_crossentropy(y_true[..., 7], y_pred[..., 7])
            return 2.0 * distance + turn + kind + slider + 0.25 * combo

    return PlacementLoss(name="placement_loss")


def build_placement_model(
    *,
    sequence_length: int = 256,
    learning_rate: float = 5e-4,
    weight_decay: float = 1e-4,
    jit_compile: bool | str = "auto",
) -> Any:
    keras = _require_keras()
    inputs = keras.Input(
        (sequence_length, PLACEMENT_FEATURE_DIMENSION),
        dtype="float32",
        name="placement_features",
    )
    sequence = keras.layers.Conv1D(128, 7, padding="same", activation="gelu")(inputs)
    valid = keras.ops.greater(inputs[..., 6], 0.0)
    attention_mask = keras.ops.logical_and(
        keras.ops.expand_dims(valid, axis=1), keras.ops.expand_dims(valid, axis=2)
    )
    for block in range(3):
        normalized = keras.layers.LayerNormalization(name=f"placement_attention_norm_{block}")(
            sequence
        )
        attention = keras.layers.MultiHeadAttention(
            num_heads=4,
            key_dim=32,
            dropout=0.15,
            name=f"placement_attention_{block}",
        )(normalized, normalized, attention_mask=attention_mask)
        sequence = keras.layers.Add()([sequence, attention])
        normalized = keras.layers.LayerNormalization(name=f"placement_ff_norm_{block}")(sequence)
        feedforward = keras.layers.Dense(256, activation="gelu")(normalized)
        feedforward = keras.layers.Dropout(0.15)(feedforward)
        feedforward = keras.layers.Dense(128)(feedforward)
        sequence = keras.layers.Add()([sequence, feedforward])
    sequence = keras.layers.LayerNormalization()(sequence)
    distance = keras.layers.Dense(1, activation="sigmoid", dtype="float32")(sequence)
    turn = keras.layers.Dense(2, activation="tanh", dtype="float32")(sequence)
    kind = keras.layers.Dense(3, activation="softmax", dtype="float32")(sequence)
    slider = keras.layers.Dense(1, activation="sigmoid", dtype="float32")(sequence)
    combo = keras.layers.Dense(1, activation="sigmoid", dtype="float32")(sequence)
    output = keras.layers.Concatenate(name="placement_output")(
        [distance, turn, kind, slider, combo]
    )
    model = keras.Model(inputs=inputs, outputs=output, name="osumapper_placement_v1")
    model.compile(
        optimizer=keras.optimizers.AdamW(
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            clipnorm=1.0,
        ),
        loss=_placement_loss(keras),
        jit_compile=jit_compile,
    )
    return model


def _tf_dataset(tf: Any, data: PlacementDataset, batch_size: int, seed: int, shuffle: bool) -> Any:
    dataset = tf.data.Dataset.from_tensor_slices((data.features, data.targets, data.weights))
    options = tf.data.Options()
    options.experimental_deterministic = True
    dataset = dataset.with_options(options)
    if shuffle:
        dataset = dataset.shuffle(data.windows, seed=seed, reshuffle_each_iteration=True)
    return dataset.batch(batch_size).repeat().prefetch(tf.data.AUTOTUNE)


def train_placement(
    *,
    dataset_root: Path | None = None,
    output: Path | None = None,
    config: PlacementTrainingConfig | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    training = config or PlacementTrainingConfig()
    keras, tf, resolved_precision = _configure_device(training.device, training.precision)
    jit_compile = _jit_compile(training.xla, gpu_enabled=bool(tf.config.get_visible_devices("GPU")))
    configure_determinism(training.seed)
    paths = DatasetPaths.at(dataset_root)
    split_manifest = read_json(paths.splits / "manifest.json", default=None)
    if split_manifest is None:
        raise InputError("Dataset has not been split. Run `osumapper dataset split` first.")
    output_root = _safe_output(output or project_root() / "models" / "modern" / "placement-v1")
    model_path = output_root / "model.keras"
    best_path = output_root / "best.keras"
    last_path = output_root / "last.keras"
    state_path = output_root / "training_state.json"
    history_path = output_root / "training_history.json"
    if not resume and any(path.exists() for path in (model_path, best_path, last_path, state_path)):
        raise InputError(f"Placement artifacts already exist under {output_root}; use --resume.")
    train_rows = load_split("train", dataset_root=paths.root)
    validation_rows = load_split("validation", dataset_root=paths.root)
    train_data = prepare_placement_dataset(
        train_rows,
        sequence_length=training.sequence_length,
        balance_songs=training.balance_songs,
    )
    validation_data = prepare_placement_dataset(
        validation_rows,
        sequence_length=training.sequence_length,
        balance_songs=False,
    )
    initial_epoch = 0
    if resume:
        saved_config = read_json(output_root / "config.json", default=None)
        saved_manifest = read_json(output_root / "dataset_manifest.json", default=None)
        if saved_config is None or saved_manifest is None:
            raise InputError("Placement resume metadata is missing.")
        if saved_manifest.get("split_manifest") != split_manifest:
            raise InputError("Dataset split changed; placement resume is unsafe.")
        if int(saved_config["training"]["sequence_length"]) != training.sequence_length:
            raise InputError("Placement sequence length must match the original run.")
        current_training = training.as_dict()
        changed = [
            name
            for name in (
                "batch_size",
                "learning_rate",
                "seed",
                "precision",
                "xla",
                "early_stopping_patience",
                "weight_decay",
                "balance_songs",
            )
            if saved_config["training"].get(name) != current_training[name]
        ]
        if changed:
            raise InputError("Placement resume settings changed: " + ", ".join(changed))
        initial_epoch = int(read_json(state_path, default={}).get("epochs_completed", 0))
        if initial_epoch >= training.epochs:
            raise InputError("Placement run already reached the requested epoch count.")
        checkpoint = next(
            (path for path in (last_path, model_path, best_path) if path.is_file()),
            None,
        )
        if checkpoint is None:
            raise InputError("No placement checkpoint is available to resume.")
        model = keras.models.load_model(checkpoint, compile=False)
        model.compile(
            optimizer=keras.optimizers.AdamW(
                learning_rate=training.learning_rate,
                weight_decay=training.weight_decay,
                clipnorm=1.0,
            ),
            loss=_placement_loss(keras),
            jit_compile=jit_compile,
        )
    else:
        model = build_placement_model(
            sequence_length=training.sequence_length,
            learning_rate=training.learning_rate,
            weight_decay=training.weight_decay,
            jit_compile=jit_compile,
        )
    callbacks = [
        keras.callbacks.ModelCheckpoint(best_path, monitor="val_loss", save_best_only=True),
        keras.callbacks.ModelCheckpoint(last_path),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5, verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=training.early_stopping_patience,
            restore_best_weights=False,
            verbose=1,
        ),
    ]
    fit = model.fit(
        _tf_dataset(tf, train_data, training.batch_size, training.seed, True),
        validation_data=_tf_dataset(tf, validation_data, training.batch_size, training.seed, False),
        steps_per_epoch=math.ceil(train_data.windows / training.batch_size),
        validation_steps=math.ceil(validation_data.windows / training.batch_size),
        initial_epoch=initial_epoch,
        epochs=training.epochs,
        callbacks=callbacks,
        verbose=2,
    )
    history = read_json(history_path, default={}) if resume else {}
    for key, values in fit.history.items():
        history.setdefault(key, []).extend(float(value) for value in values)
    write_json(history_path, history)
    completed = len(history.get("loss", []))
    best_model = keras.models.load_model(best_path, compile=False)
    best_model.save(model_path)
    write_json(
        output_root / "config.json",
        {
            "model_version": 1,
            "model_kind": "placement-v1",
            "feature_dimension": PLACEMENT_FEATURE_DIMENSION,
            "target_dimension": PLACEMENT_TARGET_DIMENSION,
            "precision": resolved_precision,
            "training": training.as_dict(),
        },
    )
    write_json(
        output_root / "dataset_manifest.json",
        {
            "split_manifest": split_manifest,
            "train_song_ids": sorted({str(row["song_id"]) for row in train_rows}),
            "validation_song_ids": sorted({str(row["song_id"]) for row in validation_rows}),
        },
    )
    write_json(
        state_path,
        {
            "epochs_completed": completed,
            "target_epochs": training.epochs,
            "stopped_early": completed < training.epochs,
        },
    )
    return {
        "model": str(model_path),
        "best_checkpoint": str(best_path),
        "epochs_completed": completed,
        "train_windows": train_data.windows,
        "validation_windows": validation_data.windows,
    }


def _inference_features(
    document: BeatmapDocument, timestamps: np.ndarray[Any, Any], density: float
) -> np.ndarray[Any, Any]:
    points = [point for point in parse_timing_points(document) if point.uninherited]
    red_points = [point.as_dict() for point in points]
    values = document.values("Difficulty")
    od = float(values.get("OverallDifficulty", 5.0))
    ar = float(values.get("ApproachRate", od))
    cs = float(values.get("CircleSize", 4.0))
    times = [float(value) for value in timestamps]
    beat_positions: list[float] = []
    divisors: list[int] = []
    bpms: list[float] = []
    red_times = [float(point["time_ms"]) for point in red_points]
    for timestamp in times:
        index = max(0, bisect.bisect_right(red_times, timestamp) - 1)
        point = red_points[index]
        beat_position = (timestamp - float(point["time_ms"])) / float(point["beat_length_ms"])
        candidates = (1, 2, 3, 4, 6, 8)
        divisor = min(
            candidates,
            key=lambda value: abs(beat_position * value - round(beat_position * value)),
        )
        beat_positions.append(beat_position)
        divisors.append(divisor)
        bpms.append(float(point.get("bpm") or 120.0))
    return _temporal_features(
        times,
        beat_positions,
        divisors,
        bpms,
        density=density,
        od=od,
        ar=ar,
        cs=cs,
    )


def predict_placement(
    document: BeatmapDocument,
    timestamps: np.ndarray[Any, Any],
    *,
    model_root: Path | None,
    target_density: float,
    seed: int,
) -> list[dict[str, Any]]:
    root = (
        (model_root or project_root() / "models" / "modern" / "placement-v1").expanduser().resolve()
    )
    model_path = root if root.suffix.casefold() == ".keras" else root / "model.keras"
    config_root = root.parent if root.suffix.casefold() == ".keras" else root
    config = read_json(config_root / "config.json", default=None)
    if config is None or not model_path.is_file():
        raise InputError(f"Placement-v1 model is missing under {config_root}; train it first.")
    sequence_length = int(config["training"]["sequence_length"])
    features = _inference_features(document, timestamps, target_density)
    model = _require_keras().models.load_model(model_path, compile=False)
    predictions: list[np.ndarray[Any, Any]] = []
    for start in range(0, len(features), sequence_length):
        stop = min(len(features), start + sequence_length)
        window = np.zeros((1, sequence_length, PLACEMENT_FEATURE_DIMENSION), dtype=np.float32)
        window[0, : stop - start] = features[start:stop]
        predictions.append(np.asarray(model.predict(window, verbose=0)[0, : stop - start]))
    predicted = np.concatenate(predictions) if predictions else np.empty((0, 8))
    if not len(predicted):
        raise InputError("Placement-v1 received no rhythm events.")
    phase = math.radians(seed % 360)
    x, y = 256.0, 192.0
    objects: list[dict[str, Any]] = []
    for index, (timestamp, values) in enumerate(zip(timestamps, predicted, strict=True)):
        if index:
            turn = math.atan2(float(values[1]), float(values[2]))
            phase += turn
            delta_ms = max(1.0, float(timestamp - timestamps[index - 1]))
            maximum = min(300.0, max(48.0, 1.8 * delta_ms))
            distance = min(maximum, max(12.0, float(values[0]) * 640.0))
            candidate_x = x + math.cos(phase) * distance
            candidate_y = y + math.sin(phase) * distance
            if not 24 <= candidate_x <= 488:
                phase = math.pi - phase
                candidate_x = x + math.cos(phase) * distance
            if not 24 <= candidate_y <= 360:
                phase = -phase
                candidate_y = y + math.sin(phase) * distance
            x = min(488.0, max(24.0, candidate_x))
            y = min(360.0, max(24.0, candidate_y))
        kind = int(np.argmax(values[3:6]))
        next_gap = float(timestamps[index + 1] - timestamp) if index + 1 < len(timestamps) else 0.0
        combo = 4 if index and float(values[7]) >= 0.6 else 0
        obj: dict[str, Any] = {
            "x": x,
            "y": y,
            "time": int(timestamp),
            "type": 1 | combo,
            "hitsounds": 0,
        }
        if kind == 2 and next_gap >= 1_000.0:
            obj.update(
                {
                    "x": 256,
                    "y": 192,
                    "type": 8 | combo,
                    "spinnerEndTime": int(timestamp + max(500.0, next_gap - 20.0)),
                }
            )
        elif kind == 1 and next_gap >= 180.0:
            length = min(240.0, max(60.0, float(values[6]) * 512.0))
            endpoint = [
                min(488.0, max(24.0, x + math.cos(phase) * length)),
                min(360.0, max(24.0, y + math.sin(phase) * length)),
            ]
            obj.update(
                {
                    "type": 2 | combo,
                    "sliderGenerator": {"endpoint": endpoint, "len": length},
                }
            )
        objects.append(obj)
    return objects


def evaluate_placement(
    *,
    dataset_root: Path | None = None,
    model_root: Path | None = None,
    output: Path | None = None,
) -> dict[str, Any]:
    """Evaluate Placement-v1 once on the held-out song-level test split."""
    paths = DatasetPaths.at(dataset_root)
    root = (
        (model_root or project_root() / "models" / "modern" / "placement-v1").expanduser().resolve()
    )
    model_path = root if root.suffix.casefold() == ".keras" else root / "model.keras"
    config_root = root.parent if root.suffix.casefold() == ".keras" else root
    config = read_json(config_root / "config.json", default=None)
    model_manifest = read_json(config_root / "dataset_manifest.json", default=None)
    split_manifest = read_json(paths.splits / "manifest.json", default=None)
    if config is None or model_manifest is None or not model_path.is_file():
        raise InputError(f"Placement-v1 model metadata is missing under {config_root}.")
    if model_manifest.get("split_manifest") != split_manifest:
        raise InputError("Placement evaluation split does not match the training split.")
    rows = load_split("test", dataset_root=paths.root)
    data = prepare_placement_dataset(
        rows,
        sequence_length=int(config["training"]["sequence_length"]),
        balance_songs=False,
    )
    model = _require_keras().models.load_model(model_path, compile=False)
    predicted = np.asarray(
        model.predict(data.features, batch_size=int(config["training"]["batch_size"]), verbose=0)
    )
    valid = data.weights > 0
    actual = data.targets[valid]
    estimate = predicted[valid]
    actual_turn = actual[:, 1:3]
    estimate_turn = estimate[:, 1:3]
    estimate_turn /= np.maximum(np.linalg.norm(estimate_turn, axis=1, keepdims=True), 1e-8)
    cosine = np.clip(np.sum(actual_turn * estimate_turn, axis=1), -1.0, 1.0)
    slider_mask = np.argmax(actual[:, 3:6], axis=1) == 1
    report = {
        "version": 1,
        "split": "test",
        "model": str(model_path),
        "test_songs": data.songs,
        "test_maps": data.maps,
        "positions": int(actual.shape[0]),
        "jump_distance_mae_px": float(np.mean(np.abs(actual[:, 0] - estimate[:, 0])) * 640.0),
        "turn_angle_mae_degrees": float(np.degrees(np.mean(np.arccos(cosine)))),
        "object_type_accuracy": float(
            np.mean(np.argmax(actual[:, 3:6], axis=1) == np.argmax(estimate[:, 3:6], axis=1))
        ),
        "slider_length_mae_px": (
            float(np.mean(np.abs(actual[slider_mask, 6] - estimate[slider_mask, 6])) * 512.0)
            if slider_mask.any()
            else None
        ),
        "new_combo_accuracy": float(np.mean((actual[:, 7] >= 0.5) == (estimate[:, 7] >= 0.5))),
    }
    destination = (output or config_root / "evaluation.json").expanduser().resolve()
    write_json(destination, report)
    report["output"] = str(destination)
    return report
