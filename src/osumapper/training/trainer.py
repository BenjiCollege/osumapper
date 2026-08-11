from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from osumapper.config import configure_determinism
from osumapper.errors import DependencyError, InputError
from osumapper.paths import legacy_v7_root, project_root
from osumapper.training import MODERN_RHYTHM_MODEL_VERSION
from osumapper.training.config import (
    DatasetPaths,
    GridConfig,
    RhythmTrainingConfig,
)
from osumapper.training.features import extract_dataset_features
from osumapper.training.loader import LoaderSummary, make_tf_dataset
from osumapper.training.model import (
    build_rhythm_model,
    compile_rhythm_model,
    load_rhythm_model,
)
from osumapper.training.splits import load_split
from osumapper.training.storage import read_json, write_json


@dataclass(frozen=True, slots=True)
class TrainingResult:
    output: Path
    model: Path
    best_checkpoint: Path
    epochs_completed: int
    train: LoaderSummary
    validation: LoaderSummary


def historical_empty_epochs(history: dict[str, Any]) -> list[int]:
    return [index + 1 for index, value in enumerate(history.get("loss", [])) if float(value) <= 0.0]


def default_model_root(architecture: str = "transformer-v1") -> Path:
    name = "rhythm-conformer-v2" if architecture == "conformer-v2" else "rhythm"
    return project_root() / "models" / "modern" / name


def _safe_output(path: Path) -> Path:
    output = path.expanduser().resolve()
    forbidden = (project_root() / "v6.2", legacy_v7_root())
    if any(output == root or output.is_relative_to(root) for root in forbidden):
        raise InputError("Modern training output must not be inside v6.2 or v7.0.")
    legacy_models = project_root() / "models"
    modern_models = legacy_models / "modern"
    if output.is_relative_to(legacy_models) and not output.is_relative_to(modern_models):
        raise InputError("Modern training output must not overwrite migrated legacy models.")
    output.mkdir(parents=True, exist_ok=True)
    return output


def _configure_device(device: str) -> tuple[Any, Any]:
    os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
    os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
    try:
        import keras
        import tensorflow as tf
    except ImportError as exc:
        raise DependencyError("Training requires TensorFlow and Keras.") from exc
    normalized = device.casefold()
    if normalized not in {"auto", "cpu", "gpu"}:
        raise InputError("--device must be auto, cpu, or gpu.")
    gpus = tf.config.list_physical_devices("GPU")
    try:
        if normalized == "cpu":
            tf.config.set_visible_devices([], "GPU")
        elif normalized == "gpu" and not gpus:
            raise InputError("--device gpu was requested, but TensorFlow found no GPU.")
    except RuntimeError as exc:
        raise InputError(
            f"Could not configure TensorFlow device before initialization: {exc}"
        ) from exc
    return keras, tf


def train_rhythm(
    *,
    dataset_root: Path | None = None,
    output: Path | None = None,
    config: RhythmTrainingConfig | None = None,
    resume: bool = False,
) -> TrainingResult:
    training_config = config or RhythmTrainingConfig()
    keras, _ = _configure_device(training_config.device)
    configure_determinism(training_config.seed)
    paths = DatasetPaths.at(dataset_root)
    paths.create()
    output_root = _safe_output(output or default_model_root(training_config.architecture))
    model_path = output_root / "model.keras"
    best_path = output_root / "best.keras"
    last_path = output_root / "last.keras"
    config_path = output_root / "config.json"
    history_path = output_root / "training_history.json"
    state_path = output_root / "training_state.json"
    resume_path = output_root / "resume_state"
    existing_artifacts = (
        model_path,
        best_path,
        last_path,
        config_path,
        history_path,
        state_path,
        resume_path,
    )
    if not resume and any(path.exists() for path in existing_artifacts):
        raise InputError(
            f"Training artifacts already exist under {output_root}; "
            "use --resume or another --output."
        )
    if resume:
        existing_history = read_json(history_path, default={})
        if isinstance(existing_history, dict):
            empty_epochs = historical_empty_epochs(existing_history)
            if empty_epochs:
                examples = ", ".join(str(epoch) for epoch in empty_epochs[:5])
                raise InputError(
                    "This run contains historical empty epochs "
                    f"({examples}) and is preserved as a baseline. Start a clean run "
                    "with another --output instead of resuming it."
                )
    split_manifest = read_json(paths.splits / "manifest.json", default=None)
    if split_manifest is None:
        raise InputError("Dataset has not been split. Run `osumapper dataset split` first.")

    feature_summary = extract_dataset_features(
        dataset_root=paths.root, progress=lambda message: print(f"[features] {message}")
    )
    if feature_summary.failed:
        raise InputError(
            f"Feature extraction failed for {feature_summary.failed} songs; "
            f"see {paths.features / 'errors.json'}."
        )
    train_rows = load_split("train", dataset_root=paths.root)
    validation_rows = load_split("validation", dataset_root=paths.root)
    if not validation_rows:
        raise InputError("Validation split is empty; add more rated songs and split again.")
    grid_config = GridConfig(
        sequence_length=training_config.sequence_length,
        prediction_threshold=training_config.prediction_threshold,
    )
    train_dataset, train_summary = make_tf_dataset(
        train_rows,
        dataset_root=paths.root,
        grid_config=grid_config,
        batch_size=training_config.batch_size,
        seed=training_config.seed,
        shuffle=True,
        audio_context_radius=training_config.audio_context_radius,
    )
    validation_dataset, validation_summary = make_tf_dataset(
        validation_rows,
        dataset_root=paths.root,
        grid_config=grid_config,
        batch_size=training_config.batch_size,
        seed=training_config.seed,
        shuffle=False,
        audio_context_radius=training_config.audio_context_radius,
    )

    history: dict[str, list[float]] = {}
    initial_epoch = 0
    if resume:
        previous_config = read_json(config_path, default=None)
        previous_manifest = read_json(output_root / "dataset_manifest.json", default=None)
        if previous_config is None or previous_manifest is None:
            raise InputError(
                f"Cannot resume because configuration or dataset metadata is missing under "
                f"{output_root}."
            )
        if previous_manifest.get("split_manifest") != split_manifest:
            raise InputError(
                "The dataset or song-level split changed after this run started; "
                "resume is unsafe. Use a new --output directory."
            )
        previous_training = previous_config.get("training", {})
        if int(previous_training.get("sequence_length", -1)) != training_config.sequence_length:
            raise InputError("--sequence-length must match the original run when resuming.")
        if int(previous_config.get("audio_dimension", -1)) != train_summary.audio_dimension:
            raise InputError("Cached audio feature dimensions changed; resume is unsafe.")
        if int(previous_config.get("grid_dimension", -1)) != train_summary.grid_dimension:
            raise InputError("Timing-grid feature dimensions changed; resume is unsafe.")
        previous_architecture = str(previous_config.get("architecture", "transformer-v1"))
        if previous_architecture == "conv_audio_encoder_transformer":
            previous_architecture = "transformer-v1"
        if previous_architecture != training_config.architecture:
            raise InputError("--architecture must match the original run when resuming.")
        if (
            int(previous_config.get("audio_context_radius", 0))
            != training_config.audio_context_radius
        ):
            raise InputError("--audio-context-radius must match the original run when resuming.")
        state = read_json(state_path, default={})
        initial_epoch = int(state.get("epochs_completed", 0))
        if initial_epoch >= training_config.epochs:
            raise InputError(
                f"This run already completed {initial_epoch} epochs; set --epochs to a larger "
                "total to continue."
            )
        history = read_json(history_path, default={})
        if not isinstance(history, dict):
            raise InputError(f"Malformed training history: {history_path}")
        checkpoint = next(
            (path for path in (last_path, model_path, best_path) if path.is_file()), None
        )
        if checkpoint is not None:
            model = load_rhythm_model(checkpoint, compile_model=False)
            compile_rhythm_model(
                model,
                learning_rate=training_config.learning_rate,
                positive_weight=train_summary.positive_weight,
            )
        elif resume_path.exists():
            model = build_rhythm_model(
                audio_dimension=train_summary.audio_dimension,
                grid_dimension=train_summary.grid_dimension,
                sequence_length=training_config.sequence_length,
                learning_rate=training_config.learning_rate,
                positive_weight=train_summary.positive_weight,
                architecture=training_config.architecture,
            )
        else:
            raise InputError(f"No resumable checkpoint exists under {output_root}.")
    else:
        model = build_rhythm_model(
            audio_dimension=train_summary.audio_dimension,
            grid_dimension=train_summary.grid_dimension,
            sequence_length=training_config.sequence_length,
            learning_rate=training_config.learning_rate,
            positive_weight=train_summary.positive_weight,
            architecture=training_config.architecture,
        )

    class EpochProgress(keras.callbacks.Callback):
        def on_train_begin(self, logs: dict[str, Any] | None = None) -> None:
            self.started = time.monotonic()

        def on_epoch_begin(self, epoch: int, logs: dict[str, Any] | None = None) -> None:
            self.batches_this_epoch = 0

        def on_train_batch_end(self, batch: int, logs: dict[str, Any] | None = None) -> None:
            self.batches_this_epoch += 1

        def on_epoch_end(self, epoch: int, logs: dict[str, Any] | None = None) -> None:
            if self.batches_this_epoch == 0:
                raise RuntimeError(
                    "Training epoch consumed no batches; refusing to record an empty epoch."
                )
            values = logs or {}
            for name, value in sorted(values.items()):
                try:
                    history.setdefault(name, []).append(float(value))
                except (TypeError, ValueError):
                    continue
            write_json(history_path, history)
            write_json(
                state_path,
                {
                    "epochs_completed": epoch + 1,
                    "target_epochs": training_config.epochs,
                    "last_checkpoint": str(last_path),
                },
            )
            elapsed = time.monotonic() - self.started
            complete = epoch + 1
            completed_this_run = max(1, complete - initial_epoch)
            remaining = elapsed / completed_this_run * max(0, training_config.epochs - complete)
            fields = [
                f"Epoch {complete}/{training_config.epochs}",
                f"loss={float(values.get('loss', 0.0)):.5f}",
                f"val_loss={float(values.get('val_loss', 0.0)):.5f}",
                f"precision={float(values.get('precision', 0.0)):.4f}",
                f"recall={float(values.get('recall', 0.0)):.4f}",
                f"f1={float(values.get('f1', 0.0)):.4f}",
                f"pr_auc={float(values.get('pr_auc', 0.0)):.4f}",
                f"val_f1={float(values.get('val_f1', 0.0)):.4f}",
                f"val_pr_auc={float(values.get('val_pr_auc', 0.0)):.4f}",
                f"eta={remaining / 60.0:.1f}m",
            ]
            print(" | ".join(fields), flush=True)

    previous_best = max(history.get("val_pr_auc", []), default=None)
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=best_path,
            monitor="val_pr_auc",
            mode="max",
            save_best_only=True,
            initial_value_threshold=previous_best,
        ),
        keras.callbacks.ModelCheckpoint(filepath=last_path, save_best_only=False),
        keras.callbacks.TensorBoard(log_dir=output_root / "tensorboard"),
        keras.callbacks.CSVLogger(output_root / "training.csv", append=resume),
        keras.callbacks.BackupAndRestore(
            backup_dir=resume_path,
            save_freq="epoch",
            delete_checkpoint=False,
        ),
        EpochProgress(),
    ]
    write_json(
        config_path,
        {
            "model_version": MODERN_RHYTHM_MODEL_VERSION,
            "architecture": training_config.architecture,
            "audio_context_radius": training_config.audio_context_radius,
            "training": training_config.as_dict(),
            "grid": grid_config.as_dict(),
            "audio_features": read_json(paths.features / "manifest.json", default={}).get(
                "config", {}
            ),
            "audio_dimension": train_summary.audio_dimension,
            "grid_dimension": train_summary.grid_dimension,
            "difficulty_features": ["OD", "AR", "CS", "objects_per_second"],
            "class_weight_positive": train_summary.positive_weight,
        },
    )
    dataset_manifest = {
        "split_manifest": split_manifest,
        "train_song_ids": sorted({str(row["song_id"]) for row in train_rows}),
        "validation_song_ids": sorted({str(row["song_id"]) for row in validation_rows}),
        "train_maps": len(train_rows),
        "validation_maps": len(validation_rows),
        "feature_manifest": read_json(paths.features / "manifest.json", default={}),
    }
    write_json(output_root / "dataset_manifest.json", dataset_manifest)
    model.fit(
        train_dataset,
        validation_data=validation_dataset,
        initial_epoch=initial_epoch,
        epochs=training_config.epochs,
        callbacks=callbacks,
        shuffle=False,
        steps_per_epoch=math.ceil(train_summary.windows / training_config.batch_size),
        validation_steps=math.ceil(validation_summary.windows / training_config.batch_size),
        verbose=0,
    )
    if best_path.is_file():
        best_model = load_rhythm_model(best_path, compile_model=False)
        best_model.save(model_path)
    else:
        last_model = load_rhythm_model(last_path, compile_model=False)
        last_model.save(model_path)
    completed_state = read_json(state_path, default={})
    return TrainingResult(
        output=output_root,
        model=model_path,
        best_checkpoint=best_path,
        epochs_completed=int(completed_state.get("epochs_completed", len(history.get("loss", [])))),
        train=train_summary,
        validation=validation_summary,
    )
