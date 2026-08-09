from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from osumapper.config import configure_determinism
from osumapper.errors import DependencyError, GenerationError
from osumapper.paths import legacy_v7_root, modern_models_root, project_root


@dataclass(frozen=True, slots=True)
class ParityResult:
    source: str
    destination: str
    source_sha256: str
    destination_sha256: str
    seed: int
    probe_shapes: list[list[int]]
    max_absolute_error: float
    tensorflow_version: str
    keras_version: str


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def legacy_model_paths() -> list[Path]:
    return sorted((legacy_v7_root() / "models").glob("*/rhythm_model"))


def modern_model_path(source: Path) -> Path:
    return modern_models_root() / source.parent.name / "rhythm.keras"


def _require_ml() -> tuple[Any, Any, Any]:
    os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
    os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
    try:
        import keras
        import numpy as np
        import tensorflow as tf
    except ImportError as exc:
        raise DependencyError(
            "Model migration requires the locked ML environment: `uv sync`."
        ) from exc
    return np, tf, keras


def _load_legacy_hdf5(source: Path, tf: Any, temporary: Path) -> Any:
    h5_path = temporary / "legacy-model.h5"
    shutil.copy2(source, h5_path)
    try:
        return tf.keras.models.load_model(h5_path, compile=False)
    except Exception as exc:
        raise GenerationError(f"Could not load legacy HDF5 model {source}: {exc}") from exc


def _probe_inputs(model: Any, np: Any, seed: int) -> tuple[list[Any], list[list[int]]]:
    generator = np.random.default_rng(seed)
    probes: list[Any] = []
    shapes: list[list[int]] = []
    for tensor in model.inputs:
        shape = [
            2 if index == 0 and dim is None else 1 if dim is None else int(dim)
            for index, dim in enumerate(tensor.shape)
        ]
        shapes.append(shape)
        probes.append(generator.normal(0.0, 0.25, size=shape).astype("float32"))
    return probes, shapes


def _prediction_arrays(prediction: Any) -> list[Any]:
    return list(prediction) if isinstance(prediction, (list, tuple)) else [prediction]


def _build_modern_rhythm_model(legacy: Any, keras: Any) -> Any:
    """Recreate the small v7 CNN/LSTM using public Keras 3 layers."""
    audio_shape = tuple(int(dimension) for dimension in legacy.inputs[0].shape[1:])
    timing_shape = tuple(int(dimension) for dimension in legacy.inputs[1].shape[1:])
    output_units = int(legacy.outputs[0].shape[-1])

    audio = keras.Input(shape=audio_shape, name="audio_features")
    stream = keras.layers.TimeDistributed(
        keras.layers.Conv2D(16, (2, 2), data_format="channels_last")
    )(audio)
    stream = keras.layers.TimeDistributed(
        keras.layers.MaxPool2D((1, 2), data_format="channels_last")
    )(stream)
    stream = keras.layers.TimeDistributed(keras.layers.Activation("relu"))(stream)
    stream = keras.layers.TimeDistributed(keras.layers.Dropout(0.3))(stream)
    stream = keras.layers.TimeDistributed(
        keras.layers.Conv2D(16, (2, 3), data_format="channels_last")
    )(stream)
    stream = keras.layers.TimeDistributed(
        keras.layers.MaxPool2D((1, 2), data_format="channels_last")
    )(stream)
    stream = keras.layers.TimeDistributed(keras.layers.Activation("relu"))(stream)
    stream = keras.layers.TimeDistributed(keras.layers.Dropout(0.3))(stream)
    stream = keras.layers.TimeDistributed(keras.layers.Flatten())(stream)
    stream = keras.layers.LSTM(64, activation="tanh", return_sequences=True)(stream)

    timing = keras.Input(shape=timing_shape, name="timing_features")
    combined = keras.layers.Concatenate()([stream, timing])
    combined = keras.layers.Dense(71, activation="tanh")(combined)
    combined = keras.layers.Dense(71, activation="relu")(combined)
    output = keras.layers.Dense(output_units, activation="tanh")(combined)
    model = keras.Model(inputs=[audio, timing], outputs=output, name="osumapper_rhythm_v7")
    model.set_weights(legacy.get_weights())
    return model


def migrate_model(source: Path, *, seed: int = 2026, tolerance: float = 1e-5) -> ParityResult:
    np, tf, keras = _require_ml()
    configure_determinism(seed)
    source = source.resolve()
    destination = modern_model_path(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staged = destination.with_name("rhythm.staged.keras")

    with tempfile.TemporaryDirectory(prefix="osumapper-model-") as temp_name:
        model = _load_legacy_hdf5(source, tf, Path(temp_name))
        probes, probe_shapes = _probe_inputs(model, np, seed)
        legacy_prediction = _prediction_arrays(model.predict(probes, verbose=0))
        try:
            modern = _build_modern_rhythm_model(model, keras)
            transferred_prediction = _prediction_arrays(modern.predict(probes, verbose=0))
            staged.unlink(missing_ok=True)
            modern.save(staged)
            keras.backend.clear_session()
            migrated = keras.models.load_model(staged, compile=False)
            migrated_prediction = _prediction_arrays(migrated.predict(probes, verbose=0))
        except Exception:
            staged.unlink(missing_ok=True)
            raise

    if len(legacy_prediction) != len(migrated_prediction):
        staged.unlink(missing_ok=True)
        raise GenerationError(f"Model parity failed for {source}: output count changed.")
    transferred_errors = [
        float(np.max(np.abs(before - after)))
        for before, after in zip(legacy_prediction, transferred_prediction, strict=True)
    ]
    errors = [
        float(np.max(np.abs(before - after)))
        for before, after in zip(legacy_prediction, migrated_prediction, strict=True)
    ]
    maximum = max([*transferred_errors, *errors], default=0.0)
    transferred_matches = all(
        np.allclose(before, after, rtol=tolerance, atol=tolerance)
        for before, after in zip(legacy_prediction, transferred_prediction, strict=True)
    )
    reloaded_matches = all(
        np.allclose(before, after, rtol=tolerance, atol=tolerance)
        for before, after in zip(legacy_prediction, migrated_prediction, strict=True)
    )
    if not transferred_matches or not reloaded_matches:
        staged.unlink(missing_ok=True)
        raise GenerationError(
            f"Model parity failed for {source}: max absolute error {maximum:.8g}."
        )
    os.replace(staged, destination)
    result = ParityResult(
        source=source.relative_to(project_root()).as_posix(),
        destination=destination.relative_to(project_root()).as_posix(),
        source_sha256=sha256(source),
        destination_sha256=sha256(destination),
        seed=seed,
        probe_shapes=probe_shapes,
        max_absolute_error=maximum,
        tensorflow_version=str(tf.__version__),
        keras_version=str(keras.__version__),
    )
    manifest = destination.with_suffix(".parity.json")
    manifest.write_text(json.dumps(asdict(result), indent=2) + "\n", encoding="utf-8")
    tf.keras.backend.clear_session()
    return result


def migrate_all(*, seed: int = 2026, tolerance: float = 1e-5) -> list[ParityResult]:
    return [migrate_model(path, seed=seed, tolerance=tolerance) for path in legacy_model_paths()]


def verify_model(source: Path, *, tolerance: float = 1e-5) -> ParityResult:
    destination = modern_model_path(source)
    manifest_path = destination.with_suffix(".parity.json")
    if not destination.is_file() or not manifest_path.is_file():
        raise GenerationError(
            f"Migrated model or parity manifest is missing for {source.parent.name}."
        )
    recorded = json.loads(manifest_path.read_text(encoding="utf-8"))
    if recorded["source_sha256"] != sha256(source):
        raise GenerationError(f"Legacy model changed after migration: {source}")
    if recorded["destination_sha256"] != sha256(destination):
        raise GenerationError(f"Migrated model changed after parity verification: {destination}")
    if float(recorded["max_absolute_error"]) > tolerance:
        raise GenerationError(f"Recorded model parity exceeds tolerance: {destination}")
    return ParityResult(**recorded)


def verify_all(*, tolerance: float = 1e-5) -> list[ParityResult]:
    return [verify_model(path, tolerance=tolerance) for path in legacy_model_paths()]
