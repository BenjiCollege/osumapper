from __future__ import annotations

import bisect
import hashlib
import math
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from osumapper.beatmap import BeatmapDocument
from osumapper.config import configure_determinism
from osumapper.difficulty import (
    STANDARD_DIFFICULTY_KEYS,
    difficulty_for_stars,
    standard_difficulty,
)
from osumapper.errors import InputError
from osumapper.paths import project_root
from osumapper.training.beatmaps import parse_timing_points
from osumapper.training.config import DatasetPaths
from osumapper.training.model import _require_keras
from osumapper.training.splits import load_split
from osumapper.training.storage import read_json, write_json
from osumapper.training.trainer import _configure_device, _jit_compile, _safe_output

PLACEMENT_ARCHITECTURES = ("placement-v1", "placement-v2", "placement-v3")
DEFAULT_PLACEMENT_ARCHITECTURE = "placement-v3"
# v2 and v3 share an encoder, feature set, and target set; they differ only in how
# the loss is balanced and how far reconstruction trusts the position head.
_V2_FAMILY = {"placement-v2", "placement-v3"}
_MODEL_VERSIONS = {"placement-v1": 1, "placement-v2": 2, "placement-v3": 3}

PLACEMENT_FEATURE_DIMENSION = 11
PLACEMENT_TARGET_DIMENSION = 8

# Placement-v2 keeps every Placement-v1 signal, then adds the difficulty
# conditioning, musical-measure context, and window position that V1 could not
# see. The final two features are filled during windowing because they describe
# a position inside the attention window rather than a property of the map.
PLACEMENT_V2_FEATURE_DIMENSION = 26
PLACEMENT_V2_TARGET_DIMENSION = 12
_V2_WINDOW_POSITION_INDEX = 24
_V2_VALID_INDEX = 11

PLAYFIELD_WIDTH = 512.0
PLAYFIELD_HEIGHT = 384.0
# Direction comes from a blend of the predicted relative step and the predicted
# absolute position; magnitude always stays on the predicted step. The anchor
# removes the slow drift a purely relative walk accumulates over a long map
# without weakening the spacing control that star calibration tunes.
_ABSOLUTE_ANCHOR_WEIGHT = 0.3
# The position head only reached ~99px MAE on held-out songs, so v3 leans on it
# less: it is useful for stopping long-run drift, not for dictating each step.
_ANCHOR_WEIGHTS = {"placement-v2": 0.3, "placement-v3": 0.15}
_MINIMUM_JUMP_PX = 12.0
_MAXIMUM_JUMP_PX = 420.0


def placement_architecture(model_root: Path | None) -> str:
    """Report the architecture recorded beside a trained placement model."""

    root = _placement_root(model_root)
    config = read_json(root / "config.json", default=None)
    if not isinstance(config, dict):
        return "placement-unavailable"
    recorded = config.get("architecture") or config.get("model_kind")
    return str(recorded) if recorded else "placement-v1"


def _placement_root(model_root: Path | None) -> Path:
    root = (
        (model_root or project_root() / "models" / "modern" / DEFAULT_PLACEMENT_ARCHITECTURE)
        .expanduser()
        .resolve()
    )
    return root.parent if root.suffix.casefold() == ".keras" else root


def _normalize_architecture(architecture: str) -> str:
    value = architecture.strip().casefold()
    if value not in PLACEMENT_ARCHITECTURES:
        choices = ", ".join(PLACEMENT_ARCHITECTURES)
        raise InputError(f"Placement architecture must be one of {choices}.")
    return value


def feature_dimension(architecture: str) -> int:
    return (
        PLACEMENT_V2_FEATURE_DIMENSION
        if _normalize_architecture(architecture) in _V2_FAMILY
        else PLACEMENT_FEATURE_DIMENSION
    )


def target_dimension(architecture: str) -> int:
    return (
        PLACEMENT_V2_TARGET_DIMENSION
        if _normalize_architecture(architecture) in _V2_FAMILY
        else PLACEMENT_TARGET_DIMENSION
    )


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
    architecture: str = DEFAULT_PLACEMENT_ARCHITECTURE
    model_dimension: int = 192
    blocks: int = 5
    attention_heads: int = 6
    dropout: float = 0.15

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
    architecture: str = "placement-v1"
    tiers: tuple[str, ...] = ()


def _snap_divisor(value: Any) -> int:
    try:
        return max(1, int(str(value).split("/", 1)[1]))
    except (IndexError, TypeError, ValueError):
        return 4


def _active_point(timestamp: float, red_points: list[dict[str, Any]]) -> dict[str, Any]:
    times = [float(point["time_ms"]) for point in red_points]
    index = max(0, bisect.bisect_right(times, timestamp) - 1)
    return red_points[index]


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


def _tier_one_hot(difficulty_tier: str) -> tuple[float, ...]:
    key = standard_difficulty(difficulty_tier).key
    return tuple(1.0 if name == key else 0.0 for name in STANDARD_DIFFICULTY_KEYS)


def _v2_features(
    timestamps: list[float],
    beat_positions: list[float],
    snap_divisors: list[int],
    bpms: list[float],
    meters: list[int],
    *,
    density: float,
    od: float,
    ar: float,
    cs: float,
    star_rating: float,
    difficulty_tier: str,
) -> np.ndarray[Any, Any]:
    """Build the difficulty-conditioned Placement-v2 input sequence.

    Every value is derivable at generation time from the requested tier, the
    timing points, and the rhythm model's timestamps, so training and inference
    see exactly the same representation.
    """

    count = len(timestamps)
    base = _temporal_features(
        timestamps,
        beat_positions,
        snap_divisors,
        bpms,
        density=density,
        od=od,
        ar=ar,
        cs=cs,
    )
    features = np.zeros((count, PLACEMENT_V2_FEATURE_DIMENSION), dtype=np.float32)
    features[:, :PLACEMENT_FEATURE_DIMENSION] = base
    features[:, _V2_VALID_INDEX] = 1.0
    features[:, 12] = min(1.5, max(0.0, star_rating / 10.0))
    features[:, 13:19] = _tier_one_hot(difficulty_tier)
    for index, timestamp in enumerate(timestamps):
        meter = max(1, meters[index])
        measure_phase = (beat_positions[index] / meter) % 1.0
        beat_phase = beat_positions[index] % 1.0
        previous_delta = timestamp - timestamps[index - 1] if index else 2_000.0
        next_delta = timestamps[index + 1] - timestamp if index + 1 < count else 2_000.0
        beat_ms = 60_000.0 / max(1e-6, bpms[index])
        features[index, 19] = math.sin(2.0 * math.pi * measure_phase)
        features[index, 20] = math.cos(2.0 * math.pi * measure_phase)
        on_beat = min(beat_phase, 1.0 - beat_phase) < 0.05
        features[index, 21] = float(on_beat and measure_phase < 0.05)
        features[index, 22] = math.tanh(
            math.log(max(1e-3, next_delta) / max(1e-3, previous_delta))
        )
        features[index, 23] = min(1.0, beat_ms / (4.0 * max(1e-3, previous_delta)))
    return features


def _apply_window_positions(window: np.ndarray[Any, Any], sequence_length: int) -> None:
    """Stamp the within-window position that attention uses for local ordering."""

    positions = np.arange(sequence_length, dtype=np.float32) / max(1, sequence_length - 1)
    valid = window[:, _V2_VALID_INDEX] > 0.0
    window[:, _V2_WINDOW_POSITION_INDEX] = np.where(valid, positions, 0.0)
    window[:, _V2_WINDOW_POSITION_INDEX + 1] = np.where(
        valid, np.sin(2.0 * np.pi * positions), 0.0
    )


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


def _v2_targets(
    objects: list[dict[str, Any]],
    beat_lengths: list[float],
) -> np.ndarray[Any, Any]:
    """Extend the V1 targets with absolute position, repeats, and hold length."""

    base = _placement_targets(objects)
    targets = np.zeros((len(objects), PLACEMENT_V2_TARGET_DIMENSION), dtype=np.float32)
    targets[:, :PLACEMENT_TARGET_DIMENSION] = base
    for index, obj in enumerate(objects):
        targets[index, 8] = min(1.0, max(0.0, float(obj["x"]) / PLAYFIELD_WIDTH))
        targets[index, 9] = min(1.0, max(0.0, float(obj["y"]) / PLAYFIELD_HEIGHT))
        repeats = int(obj.get("slider_repeats") or 1)
        targets[index, 10] = min(1.0, max(0.0, (repeats - 1) / 4.0))
        hold_ms = max(0.0, float(obj.get("end_time_ms", obj["time_ms"])) - float(obj["time_ms"]))
        targets[index, 11] = min(1.0, hold_ms / max(1e-6, beat_lengths[index] * 8.0))
    return targets


def _map_arrays(
    row: dict[str, Any],
    architecture: str,
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
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
    active = [_active_point(timestamp, red_points) for timestamp in timestamps]
    bpms = [float(point.get("bpm") or 120.0) for point in active]
    if architecture in _V2_FAMILY:
        meters = [int(point.get("meter") or 4) for point in active]
        beat_lengths = [float(point.get("beat_length_ms") or 500.0) for point in active]
        features = _v2_features(
            timestamps,
            beat_positions,
            divisors,
            bpms,
            meters,
            density=float(row["objects_per_second"]),
            od=float(row["od"]),
            ar=float(row["ar"]),
            cs=float(row["cs"]),
            star_rating=float(row["star_rating"]),
            difficulty_tier=str(row["difficulty_tier"]),
        )
        return features, _v2_targets(objects, beat_lengths)
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
    architecture: str = "placement-v1",
) -> PlacementDataset:
    if sequence_length <= 0:
        raise InputError("Placement sequence length must be positive.")
    selected = _normalize_architecture(architecture)
    features_dimension = feature_dimension(selected)
    targets_dimension = target_dimension(selected)
    raw: list[tuple[str, np.ndarray[Any, Any], np.ndarray[Any, Any], int]] = []
    song_windows: dict[str, int] = {}
    tiers: list[str] = []
    for row in rows:
        features, targets = _map_arrays(row, selected)
        windows = math.ceil(len(features) / sequence_length)
        song_id = str(row["song_id"])
        song_windows[song_id] = song_windows.get(song_id, 0) + windows
        raw.append((song_id, features, targets, windows))
    all_features: list[np.ndarray[Any, Any]] = []
    all_targets: list[np.ndarray[Any, Any]] = []
    all_weights: list[np.ndarray[Any, Any]] = []
    mean_windows = float(np.mean(list(song_windows.values()))) if song_windows else 1.0
    for (song_id, features, targets, _windows), row in zip(raw, rows, strict=True):
        song_weight = (
            float(np.clip(mean_windows / song_windows[song_id], 0.25, 4.0))
            if balance_songs
            else 1.0
        )
        for start in range(0, len(features), sequence_length):
            stop = min(len(features), start + sequence_length)
            valid = stop - start
            feature_window = np.zeros((sequence_length, features_dimension), dtype=np.float32)
            target_window = np.zeros((sequence_length, targets_dimension), dtype=np.float32)
            weight_window = np.zeros(sequence_length, dtype=np.float32)
            feature_window[:valid] = features[start:stop]
            target_window[:valid] = targets[start:stop]
            weight_window[:valid] = song_weight
            if selected in _V2_FAMILY:
                _apply_window_positions(feature_window, sequence_length)
            all_features.append(feature_window)
            all_targets.append(target_window)
            all_weights.append(weight_window)
            tiers.append(str(row.get("difficulty_tier", "")))
    if not all_features:
        raise InputError("The selected split produced no placement windows.")
    return PlacementDataset(
        features=np.asarray(all_features, dtype=np.float32),
        targets=np.asarray(all_targets, dtype=np.float32),
        weights=np.asarray(all_weights, dtype=np.float32),
        windows=len(all_features),
        maps=len(rows),
        songs=len(song_windows),
        architecture=selected,
        tiers=tuple(tiers),
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
            # `keras.losses.binary_crossentropy` reduces the last axis, so passing
            # the already-squeezed (batch, step) combo channel collapsed it to
            # (batch,) and every Placement-v1 run died on a broadcast error before
            # finishing one epoch. Compute the element-wise term directly so the
            # loss keeps one value per step, as the other terms do.
            probability = ops.clip(y_pred[..., 7], 1e-7, 1.0 - 1e-7)
            combo = -(
                y_true[..., 7] * ops.log(probability)
                + (1.0 - y_true[..., 7]) * ops.log(1.0 - probability)
            )
            return 2.0 * distance + turn + kind + slider + 0.25 * combo

    return PlacementLoss(name="placement_loss")


def _placement_v2_loss(keras: Any, **overrides: Any) -> Any:
    @keras.saving.register_keras_serializable(package="osumapper")
    class PlacementV2Loss(keras.losses.Loss):
        """Balance the eight Placement-v2 predictions against their real scales.

        Huber terms keep rare extreme jumps and long sliders from dominating the
        gradient, the class weights stop circles from swamping the far rarer
        sliders and spinners, and the absolute-position term is what teaches the
        model where in the playfield human mappers actually build patterns.
        """

        def __init__(
            self,
            distance_weight: float = 2.0,
            turn_weight: float = 1.0,
            kind_weight: float = 1.0,
            slider_weight: float = 1.0,
            repeat_weight: float = 0.5,
            hold_weight: float = 0.5,
            combo_weight: float = 0.5,
            position_weight: float = 0.75,
            combo_positive_weight: float = 3.0,
            kind_class_weights: tuple[float, float, float] = (1.0, 1.25, 6.0),
            squared_distance: bool = False,
            name: str = "placement_v2_loss",
            **kwargs: Any,
        ) -> None:
            super().__init__(name=name, **kwargs)
            self.squared_distance = bool(squared_distance)
            self.distance_weight = float(distance_weight)
            self.turn_weight = float(turn_weight)
            self.kind_weight = float(kind_weight)
            self.slider_weight = float(slider_weight)
            self.repeat_weight = float(repeat_weight)
            self.hold_weight = float(hold_weight)
            self.combo_weight = float(combo_weight)
            self.position_weight = float(position_weight)
            self.combo_positive_weight = float(combo_positive_weight)
            self.kind_class_weights = tuple(float(value) for value in kind_class_weights)

        @staticmethod
        def _huber(ops: Any, error: Any, delta: float) -> Any:
            absolute = ops.absolute(error)
            return ops.where(
                absolute <= delta,
                0.5 * ops.square(absolute),
                delta * (absolute - 0.5 * delta),
            )

        def call(self, y_true: Any, y_pred: Any) -> Any:
            ops = keras.ops
            actual = ops.cast(y_true, y_pred.dtype)
            distance_error = actual[..., 0] - y_pred[..., 0]
            distance = (
                ops.square(distance_error)
                if self.squared_distance
                else self._huber(ops, distance_error, 0.05)
            )

            true_turn = actual[..., 1:3]
            predicted_turn = y_pred[..., 1:3]
            norm = ops.sqrt(ops.sum(ops.square(predicted_turn), axis=-1) + 1e-9)
            unit_turn = predicted_turn / ops.expand_dims(ops.maximum(norm, 1e-6), axis=-1)
            turn = 1.0 - ops.sum(true_turn * unit_turn, axis=-1)
            # Keep the tanh turn head on the unit circle so the decoded angle is
            # stable even where the training turn distribution is nearly uniform.
            turn = turn + 0.1 * ops.square(norm - 1.0)

            class_weights = ops.cast(
                ops.convert_to_tensor(list(self.kind_class_weights)), y_pred.dtype
            )
            true_kind = actual[..., 3:6]
            kind = -ops.sum(
                true_kind * ops.log(ops.clip(y_pred[..., 3:6], 1e-7, 1.0)), axis=-1
            ) * ops.sum(true_kind * class_weights, axis=-1)

            is_slider = actual[..., 4]
            is_hold = ops.clip(is_slider + actual[..., 5], 0.0, 1.0)
            slider = self._huber(ops, actual[..., 6] - y_pred[..., 6], 0.05) * is_slider
            repeats = self._huber(ops, actual[..., 10] - y_pred[..., 10], 0.1) * is_slider
            hold = self._huber(ops, actual[..., 11] - y_pred[..., 11], 0.05) * is_hold

            probability = ops.clip(y_pred[..., 7], 1e-7, 1.0 - 1e-7)
            combo = -(
                self.combo_positive_weight * actual[..., 7] * ops.log(probability)
                + (1.0 - actual[..., 7]) * ops.log(1.0 - probability)
            )

            position = self._huber(ops, actual[..., 8] - y_pred[..., 8], 0.05) + self._huber(
                ops, actual[..., 9] - y_pred[..., 9], 0.05
            )
            return (
                self.distance_weight * distance
                + self.turn_weight * turn
                + self.kind_weight * kind
                + self.slider_weight * slider
                + self.repeat_weight * repeats
                + self.hold_weight * hold
                + self.combo_weight * combo
                + self.position_weight * position
            )

        def get_config(self) -> dict[str, Any]:
            return {
                **super().get_config(),
                "distance_weight": self.distance_weight,
                "turn_weight": self.turn_weight,
                "kind_weight": self.kind_weight,
                "slider_weight": self.slider_weight,
                "repeat_weight": self.repeat_weight,
                "hold_weight": self.hold_weight,
                "combo_weight": self.combo_weight,
                "position_weight": self.position_weight,
                "combo_positive_weight": self.combo_positive_weight,
                "kind_class_weights": list(self.kind_class_weights),
                "squared_distance": self.squared_distance,
            }

    return PlacementV2Loss(**overrides)


def _placement_v3_loss(keras: Any) -> Any:
    """Placement-v2's objective with the two terms that measurably hurt it fixed.

    Measured against Placement-v1 on the held-out split, v2 won on object types,
    combos, turn angle, and slider length but lost jump-distance accuracy
    (50.8px -> 53.8px MAE) while shrinking its prediction spread (72.7px ->
    67.9px against a 93.4px target spread). Both symptoms point at the same
    cause: a Huber delta of 0.05 is 32px, so typical ~54px distance errors sat in
    the linear regime where the gradient no longer grows with the error, and the
    head hedged toward the mean. v3 restores squared error on distance so large
    misses are punished proportionally again, and demotes the absolute-position
    term, which only reached ~99px MAE and was competing for the same capacity.
    Every term v2 actually improved is carried over unchanged.
    """

    # Only the distance term changes shape. Slider length measurably improved
    # under Huber (29.0px -> 28.1px), so those terms keep it.
    return _placement_v2_loss(
        keras,
        squared_distance=True,
        distance_weight=3.0,
        position_weight=0.25,
        name="placement_v3_loss",
    )


def _loss_for(keras: Any, architecture: str) -> Any:
    if architecture == "placement-v3":
        return _placement_v3_loss(keras)
    if architecture == "placement-v2":
        return _placement_v2_loss(keras)
    return _placement_loss(keras)


def _build_placement_v1(
    keras: Any,
    *,
    sequence_length: int,
) -> Any:
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
    return keras.Model(inputs=inputs, outputs=output, name="osumapper_placement_v1")


def _build_placement_v2(
    keras: Any,
    *,
    sequence_length: int,
    dimension: int,
    blocks: int,
    heads: int,
    dropout: float,
) -> Any:
    """Conformer-style placement encoder with per-group prediction heads.

    The depthwise convolution captures the short local motifs that make streams,
    triangles, and stacks recognisable, while masked attention supplies the
    longer-range structure that Placement-v1's three plain attention blocks had
    to infer from local timing alone.
    """

    if dimension % heads:
        raise InputError("Placement model dimension must be divisible by attention heads.")
    inputs = keras.Input(
        (sequence_length, PLACEMENT_V2_FEATURE_DIMENSION),
        dtype="float32",
        name="placement_features",
    )
    sequence = keras.layers.Dense(dimension, name="placement_input_projection")(inputs)
    sequence = keras.layers.LayerNormalization(name="placement_input_norm")(sequence)
    valid = keras.ops.greater(inputs[..., _V2_VALID_INDEX], 0.5)
    attention_mask = keras.ops.logical_and(
        keras.ops.expand_dims(valid, axis=1), keras.ops.expand_dims(valid, axis=2)
    )
    for block in range(blocks):
        first_ff = keras.layers.LayerNormalization(name=f"placement_ff1_norm_{block}")(sequence)
        first_ff = keras.layers.Dense(
            dimension * 4, activation="swish", name=f"placement_ff1_expand_{block}"
        )(first_ff)
        first_ff = keras.layers.Dropout(dropout, name=f"placement_ff1_dropout_{block}")(first_ff)
        first_ff = keras.layers.Dense(dimension, name=f"placement_ff1_project_{block}")(first_ff)
        first_ff = keras.layers.Rescaling(0.5, name=f"placement_ff1_scale_{block}")(first_ff)
        sequence = keras.layers.Add(name=f"placement_ff1_residual_{block}")([sequence, first_ff])

        attention = keras.layers.LayerNormalization(name=f"placement_attention_norm_{block}")(
            sequence
        )
        attention = keras.layers.MultiHeadAttention(
            num_heads=heads,
            key_dim=dimension // heads,
            dropout=dropout,
            name=f"placement_attention_{block}",
        )(attention, attention, attention_mask=attention_mask)
        attention = keras.layers.Dropout(dropout, name=f"placement_attention_dropout_{block}")(
            attention
        )
        sequence = keras.layers.Add(name=f"placement_attention_residual_{block}")(
            [sequence, attention]
        )

        convolution = keras.layers.LayerNormalization(name=f"placement_conv_norm_{block}")(
            sequence
        )
        convolution_value = keras.layers.Dense(dimension, name=f"placement_conv_value_{block}")(
            convolution
        )
        convolution_gate = keras.layers.Dense(
            dimension, activation="sigmoid", name=f"placement_conv_gate_{block}"
        )(convolution)
        convolution = keras.layers.Multiply(name=f"placement_conv_glu_{block}")(
            [convolution_value, convolution_gate]
        )
        convolution = keras.layers.SeparableConv1D(
            dimension, kernel_size=15, padding="same", name=f"placement_depthwise_{block}"
        )(convolution)
        convolution = keras.layers.LayerNormalization(name=f"placement_conv_post_norm_{block}")(
            convolution
        )
        convolution = keras.layers.Activation("swish", name=f"placement_conv_activation_{block}")(
            convolution
        )
        convolution = keras.layers.Dense(dimension, name=f"placement_conv_project_{block}")(
            convolution
        )
        convolution = keras.layers.Dropout(dropout, name=f"placement_conv_dropout_{block}")(
            convolution
        )
        sequence = keras.layers.Add(name=f"placement_conv_residual_{block}")(
            [sequence, convolution]
        )

        second_ff = keras.layers.LayerNormalization(name=f"placement_ff2_norm_{block}")(sequence)
        second_ff = keras.layers.Dense(
            dimension * 4, activation="swish", name=f"placement_ff2_expand_{block}"
        )(second_ff)
        second_ff = keras.layers.Dropout(dropout, name=f"placement_ff2_dropout_{block}")(second_ff)
        second_ff = keras.layers.Dense(dimension, name=f"placement_ff2_project_{block}")(second_ff)
        second_ff = keras.layers.Rescaling(0.5, name=f"placement_ff2_scale_{block}")(second_ff)
        sequence = keras.layers.Add(name=f"placement_ff2_residual_{block}")([sequence, second_ff])

    sequence = keras.layers.LayerNormalization(name="placement_output_norm")(sequence)

    def head(name: str, units: int, activation: str, width: int) -> Any:
        hidden = keras.layers.Dense(width, activation="gelu", name=f"{name}_head")(sequence)
        hidden = keras.layers.Dropout(dropout, name=f"{name}_head_dropout")(hidden)
        return keras.layers.Dense(
            units, activation=activation, dtype="float32", name=f"{name}_output"
        )(hidden)

    distance = head("placement_distance", 1, "sigmoid", dimension // 2)
    turn = head("placement_turn", 2, "tanh", dimension // 2)
    kind = head("placement_kind", 3, "softmax", dimension // 2)
    slider = head("placement_slider_length", 1, "sigmoid", dimension // 4)
    combo = head("placement_combo", 1, "sigmoid", dimension // 4)
    position = head("placement_position", 2, "sigmoid", dimension // 2)
    repeats = head("placement_repeats", 1, "sigmoid", dimension // 4)
    hold = head("placement_hold", 1, "sigmoid", dimension // 4)
    output = keras.layers.Concatenate(name="placement_output")(
        [distance, turn, kind, slider, combo, position, repeats, hold]
    )
    return keras.Model(inputs=inputs, outputs=output, name="osumapper_placement_v2")


def build_placement_model(
    *,
    sequence_length: int = 256,
    learning_rate: float = 5e-4,
    weight_decay: float = 1e-4,
    jit_compile: bool | str = "auto",
    architecture: str = DEFAULT_PLACEMENT_ARCHITECTURE,
    model_dimension: int = 192,
    blocks: int = 5,
    attention_heads: int = 6,
    dropout: float = 0.15,
) -> Any:
    keras = _require_keras()
    selected = _normalize_architecture(architecture)
    if sequence_length <= 0:
        raise InputError("Placement sequence length must be positive.")
    if selected in _V2_FAMILY:
        if model_dimension <= 0 or blocks <= 0 or attention_heads <= 0:
            raise InputError("Placement dimension, blocks, and heads must be positive.")
        if not 0.0 <= dropout < 1.0:
            raise InputError("Placement dropout must be between 0 and 1.")
        model = _build_placement_v2(
            keras,
            sequence_length=sequence_length,
            dimension=model_dimension,
            blocks=blocks,
            heads=attention_heads,
            dropout=dropout,
        )
    else:
        model = _build_placement_v1(keras, sequence_length=sequence_length)
    model.compile(
        optimizer=keras.optimizers.AdamW(
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            clipnorm=1.0,
        ),
        loss=_loss_for(keras, selected),
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
    architecture = _normalize_architecture(training.architecture)
    keras, tf, resolved_precision = _configure_device(training.device, training.precision)
    jit_compile = _jit_compile(training.xla, gpu_enabled=bool(tf.config.get_visible_devices("GPU")))
    configure_determinism(training.seed)
    paths = DatasetPaths.at(dataset_root)
    split_manifest = read_json(paths.splits / "manifest.json", default=None)
    if split_manifest is None:
        raise InputError("Dataset has not been split. Run `osumapper dataset split` first.")
    output_root = _safe_output(output or project_root() / "models" / "modern" / architecture)
    model_path = output_root / "model.keras"
    best_path = output_root / "best.keras"
    last_path = output_root / "last.keras"
    state_path = output_root / "training_state.json"
    history_path = output_root / "training_history.json"
    if not resume and any(path.exists() for path in (model_path, best_path, last_path, state_path)):
        raise InputError(f"Placement artifacts already exist under {output_root}; use --resume.")
    initial_epoch = 0
    saved_config: dict[str, Any] | None = None
    if resume:
        # Validate resume identity before the expensive window build so a changed
        # split or architecture fails in seconds rather than after preparation.
        saved_config = read_json(output_root / "config.json", default=None)
        saved_manifest = read_json(output_root / "dataset_manifest.json", default=None)
        if saved_config is None or saved_manifest is None:
            raise InputError("Placement resume metadata is missing.")
        if saved_manifest.get("split_manifest") != split_manifest:
            raise InputError("Dataset split changed; placement resume is unsafe.")
        if str(saved_config.get("architecture", "placement-v1")) != architecture:
            raise InputError(
                "Placement architecture changed; use a new --output directory instead of --resume."
            )
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
                "model_dimension",
                "blocks",
                "attention_heads",
                "dropout",
            )
            if saved_config["training"].get(name) != current_training[name]
        ]
        if changed:
            raise InputError("Placement resume settings changed: " + ", ".join(changed))
        initial_epoch = int(read_json(state_path, default={}).get("epochs_completed", 0))
        if initial_epoch >= training.epochs:
            raise InputError("Placement run already reached the requested epoch count.")
    train_rows = load_split("train", dataset_root=paths.root)
    validation_rows = load_split("validation", dataset_root=paths.root)
    train_data = prepare_placement_dataset(
        train_rows,
        sequence_length=training.sequence_length,
        balance_songs=training.balance_songs,
        architecture=architecture,
    )
    validation_data = prepare_placement_dataset(
        validation_rows,
        sequence_length=training.sequence_length,
        balance_songs=False,
        architecture=architecture,
    )
    if resume:
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
            loss=_loss_for(keras, architecture),
            jit_compile=jit_compile,
        )
    else:
        model = build_placement_model(
            sequence_length=training.sequence_length,
            learning_rate=training.learning_rate,
            weight_decay=training.weight_decay,
            jit_compile=jit_compile,
            architecture=architecture,
            model_dimension=training.model_dimension,
            blocks=training.blocks,
            attention_heads=training.attention_heads,
            dropout=training.dropout,
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
            "model_version": _MODEL_VERSIONS[architecture],
            "model_kind": architecture,
            "architecture": architecture,
            "feature_dimension": feature_dimension(architecture),
            "target_dimension": target_dimension(architecture),
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
        "architecture": architecture,
        "best_checkpoint": str(best_path),
        "epochs_completed": completed,
        "train_windows": train_data.windows,
        "validation_windows": validation_data.windows,
    }


def _document_timing(
    document: BeatmapDocument, timestamps: list[float]
) -> tuple[list[float], list[float], list[int], list[int], list[float]]:
    """Resolve beat length, slider velocity, meter, and snap for each timestamp."""

    points = [point for point in parse_timing_points(document) if point.uninherited]
    if not points:
        raise InputError("Placement inference requires at least one uninherited timing point.")
    red_points = [point.as_dict() for point in points]
    red_times = [float(point["time_ms"]) for point in red_points]
    inherited = document.timing_points()["ts"]
    beat_lengths: list[float] = []
    slider_lengths: list[float] = []
    meters: list[int] = []
    divisors: list[int] = []
    beat_positions: list[float] = []
    for timestamp in timestamps:
        index = max(0, bisect.bisect_right(red_times, timestamp) - 1)
        point = red_points[index]
        beat_length = float(point["beat_length_ms"]) or 500.0
        beat_position = (timestamp - float(point["time_ms"])) / beat_length
        divisor = min(
            (1, 2, 3, 4, 6, 8),
            key=lambda value: abs(beat_position * value - round(beat_position * value)),
        )
        velocity = 100.0
        for entry in inherited:
            if timestamp >= float(entry["beginTime"]):
                velocity = float(entry["sliderLength"])
            else:
                break
        beat_lengths.append(beat_length)
        slider_lengths.append(max(1.0, velocity))
        meters.append(int(point.get("meter") or 4))
        divisors.append(divisor)
        beat_positions.append(beat_position)
    return beat_lengths, slider_lengths, meters, divisors, beat_positions


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


def _v2_inference_features(
    document: BeatmapDocument,
    timestamps: list[float],
    density: float,
    *,
    difficulty_tier: str | None,
    target_stars: float | None,
) -> tuple[np.ndarray[Any, Any], list[float], list[float]]:
    values = document.values("Difficulty")
    od = float(values.get("OverallDifficulty", 5.0))
    ar = float(values.get("ApproachRate", od))
    cs = float(values.get("CircleSize", 4.0))
    beat_lengths, slider_lengths, meters, divisors, beat_positions = _document_timing(
        document, timestamps
    )
    if difficulty_tier is not None:
        profile = standard_difficulty(difficulty_tier)
    elif target_stars is not None:
        profile = difficulty_for_stars(float(target_stars))
    else:
        # Match PatternPlanner-v1's neutral fallback so an unconditioned request
        # still produces a coherent mid-difficulty layout rather than an
        # arbitrary one-hot.
        profile = standard_difficulty("hard")
    stars = profile.default_stars if target_stars is None else float(target_stars)
    features = _v2_features(
        timestamps,
        beat_positions,
        divisors,
        [60_000.0 / max(1e-6, length) for length in beat_lengths],
        meters,
        density=density,
        od=od,
        ar=ar,
        cs=cs,
        star_rating=stars,
        difficulty_tier=profile.key,
    )
    return features, beat_lengths, slider_lengths


@lru_cache(maxsize=2)
def _cached_placement_model(path: str) -> Any:
    return _require_keras().models.load_model(Path(path), compile=False)


_PREDICTION_CACHE_LIMIT = 32
_prediction_cache: dict[tuple[Any, ...], np.ndarray[Any, Any]] = {}


def _prediction_cache_key(
    model_path: str,
    features: np.ndarray[Any, Any],
    sequence_length: int,
) -> tuple[Any, ...]:
    return (
        model_path,
        sequence_length,
        features.shape,
        hashlib.sha256(np.ascontiguousarray(features, dtype=np.float32)).hexdigest(),
    )


def _cached_predictions(
    model: Any,
    model_path: str,
    features: np.ndarray[Any, Any],
    sequence_length: int,
    dimension: int,
    *,
    stamp_positions: bool,
) -> np.ndarray[Any, Any]:
    """Reuse predictions across calibration attempts that only change spacing.

    Full-set calibration re-runs placement up to 24 times per tier while tuning
    the flow scale, but the flow scale is applied during reconstruction and never
    reaches the model input. Only a density, tier, or threshold change alters the
    features, so keying the cache on the feature array itself is both correct and
    enough to make the fine-spacing phase nearly free.
    """

    key = _prediction_cache_key(model_path, features, sequence_length)
    cached = _prediction_cache.get(key)
    if cached is not None:
        return cached
    predicted = _predict_windows(
        model, features, sequence_length, dimension, stamp_positions=stamp_positions
    )
    if len(_prediction_cache) >= _PREDICTION_CACHE_LIMIT:
        _prediction_cache.pop(next(iter(_prediction_cache)))
    _prediction_cache[key] = predicted
    return predicted


def _predict_windows(
    model: Any,
    features: np.ndarray[Any, Any],
    sequence_length: int,
    dimension: int,
    *,
    stamp_positions: bool,
) -> np.ndarray[Any, Any]:
    """Run every attention window in one batched call."""

    starts = list(range(0, len(features), sequence_length))
    batch = np.zeros((len(starts), sequence_length, dimension), dtype=np.float32)
    for row, start in enumerate(starts):
        stop = min(len(features), start + sequence_length)
        batch[row, : stop - start] = features[start:stop]
        if stamp_positions:
            _apply_window_positions(batch[row], sequence_length)
    predicted = np.asarray(model.predict(batch, verbose=0))
    pieces = [
        predicted[row, : min(len(features), start + sequence_length) - start]
        for row, start in enumerate(starts)
    ]
    return np.concatenate(pieces) if pieces else np.empty((0, predicted.shape[-1]))


def _playfield_margin(document: BeatmapDocument) -> float:
    try:
        circle_size = float(document.value("Difficulty", "CircleSize", "4") or 4)
    except ValueError:
        circle_size = 4.0
    return max(0.0, math.ceil(54.4 - 4.48 * circle_size))


def _place_at_distance(
    x: float,
    y: float,
    angle: float,
    distance: float,
    margin: float,
) -> tuple[float, float]:
    """Land exactly `distance` away, as close to `angle` as the playfield allows.

    Star calibration tunes spacing, so the requested distance must survive
    contact with the playfield edge. Rotating to the nearest legal direction
    preserves the distance exactly; clamping the endpoint would silently
    shorten the jump and flatten the calibration signal. Only when no direction
    can reach that far does the distance shrink, and then it shrinks to the
    farthest point actually reachable.
    """

    left, right = margin, PLAYFIELD_WIDTH - margin
    top, bottom = margin, PLAYFIELD_HEIGHT - margin
    for offset_index in range(37):
        for sign in (1.0, -1.0) if offset_index else (1.0,):
            candidate = angle + sign * math.radians(5.0 * offset_index)
            next_x = x + math.cos(candidate) * distance
            next_y = y + math.sin(candidate) * distance
            if left <= next_x <= right and top <= next_y <= bottom:
                return next_x, next_y
    farthest = max(
        ((left, top), (right, top), (left, bottom), (right, bottom)),
        key=lambda corner: math.hypot(corner[0] - x, corner[1] - y),
    )
    reach = math.hypot(farthest[0] - x, farthest[1] - y)
    if reach <= 1e-6:
        return min(right, max(left, x)), min(bottom, max(top, y))
    fraction = min(1.0, distance / reach)
    return x + (farthest[0] - x) * fraction, y + (farthest[1] - y) * fraction


def _reconstruct_v1(
    timestamps: np.ndarray[Any, Any],
    predicted: np.ndarray[Any, Any],
    seed: int,
) -> list[dict[str, Any]]:
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


def _reconstruct_v2(
    timestamps: list[float],
    predicted: np.ndarray[Any, Any],
    *,
    beat_lengths: list[float],
    slider_lengths: list[float],
    margin: float,
    flow_scale: float,
    seed: int,
    anchor_weight: float = _ABSOLUTE_ANCHOR_WEIGHT,
) -> list[dict[str, Any]]:
    """Turn Placement-v2 predictions into legal, spacing-accurate osu! objects.

    Direction blends the predicted relative step with the predicted absolute
    position so long maps do not drift, while the step magnitude is preserved
    exactly: that magnitude is the knob full-set star calibration tunes.
    """

    scale = min(3.75, max(0.45, float(flow_scale)))
    left, right = margin, PLAYFIELD_WIDTH - margin
    top, bottom = margin, PLAYFIELD_HEIGHT - margin
    x = min(right, max(left, float(predicted[0][8]) * PLAYFIELD_WIDTH))
    y = min(bottom, max(top, float(predicted[0][9]) * PLAYFIELD_HEIGHT))
    phase = math.radians(seed % 360)
    objects: list[dict[str, Any]] = []
    for index, timestamp in enumerate(timestamps):
        values = predicted[index]
        beat_ms = max(1.0, beat_lengths[index])
        next_gap = (
            float(timestamps[index + 1] - timestamp) if index + 1 < len(timestamps) else beat_ms
        )
        if index:
            previous_x, previous_y = x, y
            phase += math.atan2(float(values[1]), float(values[2]))
            delta_ms = max(1.0, float(timestamp - timestamps[index - 1]))
            # A jump is only playable if the cursor can cross it in the gap the
            # rhythm model chose, so the time-derived ceiling always wins.
            ceiling = min(_MAXIMUM_JUMP_PX, max(48.0, 1.9 * delta_ms))
            distance = min(
                ceiling, max(_MINIMUM_JUMP_PX, float(values[0]) * 640.0 * scale)
            )
            step_x = math.cos(phase) * distance
            step_y = math.sin(phase) * distance
            anchor_x = min(right, max(left, float(values[8]) * PLAYFIELD_WIDTH))
            anchor_y = min(bottom, max(top, float(values[9]) * PLAYFIELD_HEIGHT))
            blend_x = (1.0 - anchor_weight) * (previous_x + step_x) + anchor_weight * anchor_x
            blend_y = (1.0 - anchor_weight) * (previous_y + step_y) + anchor_weight * anchor_y
            direction_x = blend_x - previous_x
            direction_y = blend_y - previous_y
            if math.hypot(direction_x, direction_y) < 1e-6:
                direction_x, direction_y = step_x, step_y
            x, y = _place_at_distance(
                previous_x,
                previous_y,
                math.atan2(direction_y, direction_x),
                distance,
                margin,
            )
            phase = math.atan2(y - previous_y, x - previous_x)
        kind = int(np.argmax(values[3:6]))
        combo = 4 if index and float(values[7]) >= 0.6 else 0
        hold_ms = float(values[11]) * beat_ms * 8.0
        obj: dict[str, Any] = {
            "x": x,
            "y": y,
            "time": int(round(timestamp)),
            "type": 1 | combo,
            "hitsounds": 0,
        }
        if kind == 2 and index + 1 < len(timestamps) and next_gap >= max(1_000.0, beat_ms * 2.0):
            end = timestamp + min(max(500.0, hold_ms), next_gap - 20.0)
            obj.update(
                {
                    "x": 256.0,
                    "y": 192.0,
                    "type": 8 | (4 if index else 0),
                    "spinnerEndTime": int(round(max(timestamp + 1.0, end))),
                }
            )
        elif kind == 1 and index + 1 < len(timestamps):
            velocity = slider_lengths[index]
            available_ms = next_gap - 20.0
            length = min(240.0, max(40.0, float(values[6]) * 512.0))
            repeats = max(1, min(4, 1 + int(round(float(values[10]) * 4.0))))
            duration_ms = length / velocity * beat_ms
            while repeats > 1 and duration_ms * repeats > available_ms:
                repeats -= 1
            if duration_ms > available_ms:
                length = max(0.0, available_ms / beat_ms * velocity)
            if length >= 35.0:
                end_x, end_y = _place_at_distance(x, y, phase, length, margin)
                obj.update(
                    {
                        "type": 2 | combo,
                        "sliderGenerator": {
                            "type": 0,
                            "endpoint": [end_x, end_y],
                            "len": length,
                            "repeats": repeats,
                        },
                    }
                )
        objects.append(obj)
    return objects


def predict_placement(
    document: BeatmapDocument,
    timestamps: np.ndarray[Any, Any],
    *,
    model_root: Path | None,
    target_density: float,
    seed: int,
    difficulty_tier: str | None = None,
    target_stars: float | None = None,
    flow_scale: float = 1.0,
) -> list[dict[str, Any]]:
    root = (
        (model_root or project_root() / "models" / "modern" / DEFAULT_PLACEMENT_ARCHITECTURE)
        .expanduser()
        .resolve()
    )
    model_path = root if root.suffix.casefold() == ".keras" else root / "model.keras"
    config_root = root.parent if root.suffix.casefold() == ".keras" else root
    config = read_json(config_root / "config.json", default=None)
    if config is None or not model_path.is_file():
        raise InputError(
            f"A trained placement model is missing under {config_root}; train it first."
        )
    architecture = _normalize_architecture(
        str(config.get("architecture") or config.get("model_kind") or "placement-v1")
    )
    sequence_length = int(config["training"]["sequence_length"])
    times = [float(value) for value in timestamps]
    if not times:
        raise InputError("Placement received no rhythm events.")
    model = _cached_placement_model(str(model_path))
    if architecture in _V2_FAMILY:
        features, beat_lengths, slider_lengths = _v2_inference_features(
            document,
            times,
            target_density,
            difficulty_tier=difficulty_tier,
            target_stars=target_stars,
        )
        predicted = _cached_predictions(
            model,
            str(model_path),
            features,
            sequence_length,
            PLACEMENT_V2_FEATURE_DIMENSION,
            stamp_positions=True,
        )
        return _reconstruct_v2(
            times,
            predicted,
            beat_lengths=beat_lengths,
            slider_lengths=slider_lengths,
            margin=_playfield_margin(document),
            flow_scale=flow_scale,
            seed=seed,
            anchor_weight=_ANCHOR_WEIGHTS.get(architecture, _ABSOLUTE_ANCHOR_WEIGHT),
        )
    features = _inference_features(document, timestamps, target_density)
    predicted = _cached_predictions(
        model,
        str(model_path),
        features,
        sequence_length,
        PLACEMENT_FEATURE_DIMENSION,
        stamp_positions=False,
    )
    return _reconstruct_v1(timestamps, predicted, seed)


def _tier_metrics(
    data: PlacementDataset,
    predicted: np.ndarray[Any, Any],
) -> list[dict[str, Any]]:
    """Report accuracy per difficulty tier so no band hides behind the mean."""

    rows: list[dict[str, Any]] = []
    tiers = np.asarray(data.tiers)
    for tier in STANDARD_DIFFICULTY_KEYS:
        window_mask = tiers == tier
        if not window_mask.any():
            continue
        valid = data.weights[window_mask] > 0
        actual = data.targets[window_mask][valid]
        estimate = predicted[window_mask][valid]
        if not len(actual):
            continue
        rows.append(
            {
                "tier": tier,
                "positions": int(actual.shape[0]),
                "jump_distance_mae_px": float(
                    np.mean(np.abs(actual[:, 0] - estimate[:, 0])) * 640.0
                ),
                "object_type_accuracy": float(
                    np.mean(
                        np.argmax(actual[:, 3:6], axis=1) == np.argmax(estimate[:, 3:6], axis=1)
                    )
                ),
            }
        )
    return rows


def evaluate_placement(
    *,
    dataset_root: Path | None = None,
    model_root: Path | None = None,
    output: Path | None = None,
) -> dict[str, Any]:
    """Evaluate a trained placement model once on the held-out test split."""
    paths = DatasetPaths.at(dataset_root)
    root = (
        (model_root or project_root() / "models" / "modern" / DEFAULT_PLACEMENT_ARCHITECTURE)
        .expanduser()
        .resolve()
    )
    model_path = root if root.suffix.casefold() == ".keras" else root / "model.keras"
    config_root = root.parent if root.suffix.casefold() == ".keras" else root
    config = read_json(config_root / "config.json", default=None)
    model_manifest = read_json(config_root / "dataset_manifest.json", default=None)
    split_manifest = read_json(paths.splits / "manifest.json", default=None)
    if config is None or model_manifest is None or not model_path.is_file():
        raise InputError(f"Placement model metadata is missing under {config_root}.")
    if model_manifest.get("split_manifest") != split_manifest:
        raise InputError("Placement evaluation split does not match the training split.")
    architecture = _normalize_architecture(
        str(config.get("architecture") or config.get("model_kind") or "placement-v1")
    )
    rows = load_split("test", dataset_root=paths.root)
    data = prepare_placement_dataset(
        rows,
        sequence_length=int(config["training"]["sequence_length"]),
        balance_songs=False,
        architecture=architecture,
    )
    model = _cached_placement_model(str(model_path))
    predicted = np.asarray(
        model.predict(data.features, batch_size=int(config["training"]["batch_size"]), verbose=0)
    )
    valid = data.weights > 0
    actual = data.targets[valid]
    estimate = predicted[valid]
    actual_turn = actual[:, 1:3]
    estimate_turn = estimate[:, 1:3]
    estimate_turn = estimate_turn / np.maximum(
        np.linalg.norm(estimate_turn, axis=1, keepdims=True), 1e-8
    )
    cosine = np.clip(np.sum(actual_turn * estimate_turn, axis=1), -1.0, 1.0)
    actual_kind = np.argmax(actual[:, 3:6], axis=1)
    estimate_kind = np.argmax(estimate[:, 3:6], axis=1)
    slider_mask = actual_kind == 1
    report: dict[str, Any] = {
        "version": _MODEL_VERSIONS[architecture],
        "architecture": architecture,
        "split": "test",
        "model": str(model_path),
        "test_songs": data.songs,
        "test_maps": data.maps,
        "positions": int(actual.shape[0]),
        "jump_distance_mae_px": float(np.mean(np.abs(actual[:, 0] - estimate[:, 0])) * 640.0),
        "turn_angle_mae_degrees": float(np.degrees(np.mean(np.arccos(cosine)))),
        "object_type_accuracy": float(np.mean(actual_kind == estimate_kind)),
        "slider_length_mae_px": (
            float(np.mean(np.abs(actual[slider_mask, 6] - estimate[slider_mask, 6])) * 512.0)
            if slider_mask.any()
            else None
        ),
        "new_combo_accuracy": float(np.mean((actual[:, 7] >= 0.5) == (estimate[:, 7] >= 0.5))),
    }
    for index, name in enumerate(("circle", "slider", "spinner")):
        mask = actual_kind == index
        report[f"{name}_recall"] = (
            float(np.mean(estimate_kind[mask] == index)) if mask.any() else None
        )
    if architecture in _V2_FAMILY:
        report.update(
            {
                "position_mae_px": float(
                    np.mean(np.abs(actual[:, 8] - estimate[:, 8])) * PLAYFIELD_WIDTH
                    + np.mean(np.abs(actual[:, 9] - estimate[:, 9])) * PLAYFIELD_HEIGHT
                )
                / 2.0,
                "slider_repeat_mae": (
                    float(
                        np.mean(np.abs(actual[slider_mask, 10] - estimate[slider_mask, 10])) * 4.0
                    )
                    if slider_mask.any()
                    else None
                ),
                "hold_length_mae_beats": (
                    float(
                        np.mean(np.abs(actual[slider_mask, 11] - estimate[slider_mask, 11])) * 8.0
                    )
                    if slider_mask.any()
                    else None
                ),
                "difficulty_tiers": _tier_metrics(data, predicted),
            }
        )
    destination = (output or config_root / "evaluation.json").expanduser().resolve()
    write_json(destination, report)
    report["output"] = str(destination)
    return report
