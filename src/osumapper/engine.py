from __future__ import annotations

import contextlib
import importlib
import json
import math
import os
import shutil
import sys
import tempfile
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

from osumapper.beatmap import BeatmapDocument
from osumapper.config import GameMode, GenerationConfig, PresetSpec, configure_determinism
from osumapper.difficulty import (
    STAR_TARGET_TOLERANCE,
    apply_standard_difficulty,
    calculate_standard_stars,
    label_standard_difficulty,
    resolve_standard_difficulty,
)
from osumapper.errors import DependencyError, GenerationError, InputError
from osumapper.paths import legacy_v7_root

Progress = Callable[[str], None]


@contextlib.contextmanager
def _legacy_context(workspace: Path) -> Iterator[None]:
    legacy = str(legacy_v7_root())
    previous = Path.cwd()
    sys.path.insert(0, legacy)
    os.chdir(workspace)
    try:
        yield
    finally:
        os.chdir(previous)
        if sys.path and sys.path[0] == legacy:
            sys.path.pop(0)


def _require_audio_stack() -> tuple[Any, Any]:
    try:
        import librosa
        import numpy as np
    except ImportError as exc:
        raise DependencyError("Generation dependencies are missing. Run `uv sync`.") from exc
    return np, librosa


def ensure_preview_time(document: BeatmapDocument, _audio: Path) -> BeatmapDocument:
    try:
        current = int(document.value("General", "PreviewTime", "-1") or -1)
    except ValueError:
        current = -1
    if current >= 0:
        return document
    timestamps: list[int] = []
    for line in document.sections().get("HitObjects", []):
        fields = line.split(",")
        if len(fields) < 3:
            continue
        with contextlib.suppress(ValueError):
            timestamps.append(max(0, int(round(float(fields[2])))))
    preview_ms = int(max(timestamps, default=0) * 0.4)
    return document.set_value("General", "PreviewTime", str(preview_ms))


def _audio_features(timestamps: Any, audio: Path, *, fft_size: int = 128) -> Any:
    np, librosa = _require_audio_stack()
    signal, sample_rate = librosa.load(audio, sr=None, mono=True)
    peak = float(np.max(np.abs(signal))) if signal.size else 0.0
    if peak > 0:
        signal = signal / peak
    times = np.asarray(timestamps)
    if times.size < 2:
        raise InputError("Audio/timing produced too few model ticks.")
    intervals = np.append(times[1:] - times[:-1], times[-1] - times[-2])

    def frequency_slice(timestamp: float) -> Any:
        index = int(timestamp / 1000 * sample_rate)
        wave = signal[max(0, index - fft_size // 2) : index + fft_size - fft_size // 2]
        transformed = np.fft.fft(wave, fft_size)[: fft_size // 2]
        # The legacy integer-frequency calculation returned 31 bins for 22.05 kHz
        # audio even though every rhythm model requires 32. A quarter FFT is the
        # intended, sample-rate-independent shape.
        transformed = transformed[: fft_size // 4]
        return np.asarray([np.abs(transformed), np.angle(transformed)])

    samples = []
    for snap in (-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3):
        samples.append(
            np.asarray(
                [
                    frequency_slice(
                        max(0.0, min(len(signal) / sample_rate * 1000, time + intervals[i] * snap))
                    )
                    for i, time in enumerate(times)
                ]
            )
        )
    raw = np.asarray(samples)
    mean = np.mean(raw, axis=1, keepdims=True)
    deviation = np.std(raw, axis=1, keepdims=True)
    deviation = np.where(deviation < 1e-8, 1.0, deviation)
    return np.swapaxes((raw - mean) / deviation, 0, 1)


def _slider_length_at(points: list[dict[str, Any]], timestamp: float) -> float:
    value = float(points[0]["sliderLength"])
    for point in points:
        if timestamp >= point["beginTime"]:
            value = float(point["sliderLength"])
        else:
            break
    return value


def _prepare_model_input(document: BeatmapDocument, audio: Path, destination: Path) -> None:
    np, librosa = _require_audio_stack()
    signal, sample_rate = librosa.load(audio, sr=None, mono=True)
    duration_ms = signal.shape[0] / sample_rate * 1000
    end_time = duration_ms - 3000
    timing = document.timing_points()
    uts = timing["uts"]
    if end_time <= float(uts[0]["beginTime"]):
        raise InputError(
            "Audio is too short; at least three seconds after the first timing point are required."
        )
    end_times = [float(point["beginTime"]) for point in uts[1:]] + [end_time]
    timestamp_groups = [
        np.arange(float(point["beginTime"]), end_times[index], float(point["tickLength"]) / 4)
        for index, point in enumerate(uts)
    ]
    nonempty = [group for group in timestamp_groups if group.size]
    if not nonempty:
        raise InputError("Timing points did not produce any model ticks.")
    timestamps = np.round(np.concatenate(nonempty)).astype(int)
    ticks = np.concatenate([np.arange(group.size) for group in nonempty])
    tick_lengths = np.concatenate(
        [
            np.full(group.size, float(uts[index]["tickLength"]))
            for index, group in enumerate(timestamp_groups)
            if group.size
        ]
    )
    slider_lengths = np.asarray([_slider_length_at(timing["ts"], time) for time in timestamps])
    features = _audio_features(timestamps, audio)
    np.savez_compressed(
        destination,
        ticks=ticks,
        timestamps=timestamps,
        wav=features,
        extra=np.asarray([60000.0 / tick_lengths, slider_lengths]),
    )


def _load_model(path: Path) -> Any:
    os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
    try:
        import keras
        import tensorflow as tf
    except ImportError as exc:
        raise DependencyError("TensorFlow is unavailable. Run `uv sync`.") from exc
    try:
        if path.suffix.casefold() == ".keras":
            return keras.models.load_model(path, compile=False)
        with tempfile.TemporaryDirectory(prefix="osumapper-h5-") as temp_name:
            h5_path = Path(temp_name) / "rhythm-model.h5"
            shutil.copy2(path, h5_path)
            return tf.keras.models.load_model(h5_path, compile=False)
    except Exception as exc:
        raise GenerationError(
            f"Could not load rhythm model {path}. Run `osumapper models migrate`.\n{exc}"
        ) from exc


def _deterministic_flow(
    converted: tuple[Any, ...], seed: int, flow_scale: float = 1.0
) -> tuple[Any, tuple[Any, ...]]:
    np, _ = _require_audio_stack()
    (
        objs,
        predictions,
        ticks,
        timestamps,
        is_slider,
        is_spinner,
        is_note_end,
        sv,
        slider_ticks,
        distance,
    ) = converted
    count = len(timestamps)
    phase = (seed % 360) * math.pi / 180
    golden = math.pi * (3 - math.sqrt(5))
    angles = phase + np.arange(count) * golden
    radii = 118 + 42 * np.sin(np.arange(count) * 0.61 + phase)
    scale = max(0.5, min(2.0, float(flow_scale)))
    x = np.clip(256 + radii * scale * np.cos(angles), 32, 480)
    y = np.clip(192 + radii * scale * 0.72 * np.sin(angles), 28, 356)
    maps = np.zeros((count, 6), dtype=float)
    for index in range(count):
        next_index = min(index + 1, count - 1)
        dx = x[next_index] - x[index]
        dy = y[next_index] - y[index]
        norm = max(1e-8, math.hypot(dx, dy))
        out_x, out_y = dx / norm, dy / norm
        length = float(sv[index] / 4 * slider_ticks[index]) if is_slider[index] else 0.0
        maps[index] = [
            x[index],
            y[index],
            out_x,
            out_y,
            np.clip(x[index] + out_x * length, 0, 512),
            np.clip(y[index] + out_y * length, 0, 384),
        ]
    slider_types = np.zeros_like(is_slider, dtype=int)
    slider_length_base = sv / 4
    data = (*converted, slider_types, slider_length_base)
    return maps, data


def _standard_objects(
    workspace: Path,
    preset: PresetSpec,
    config: GenerationConfig,
    progress: Progress,
) -> list[dict[str, Any]]:
    with _legacy_context(workspace):
        rhythm = importlib.import_module("act_rhythm_calc")
        model = _load_model(preset.rhythm_model)
        prepared = rhythm.read_npz("mapthis.npz")
        progress("Predicting rhythm")
        predicted = rhythm.step5_predict_notes(model, prepared, preset.rhythm_params)
        converted = rhythm.step5_convert_sliders(predicted, preset.rhythm_params)
    return _place_standard_objects(workspace, preset, config, progress, converted)


def _place_standard_objects(
    workspace: Path,
    preset: PresetSpec,
    config: GenerationConfig,
    progress: Progress,
    converted: tuple[Any, ...],
) -> list[dict[str, Any]]:
    with _legacy_context(workspace):
        rhythm = importlib.import_module("act_rhythm_calc")
        rhythm.step5_save_predictions(converted)
        flow_result: tuple[Any, tuple[Any, ...]] | None = None
        if config.flow_engine in {"auto", "legacy"}:
            try:
                progress("Generating legacy GAN flow")
                gan = importlib.import_module("act_gan")
                gan.step6_set_gan_params(dict(preset.gan_params))
                flow_result = gan.step6_run_all(str(preset.flow_dataset))
            except Exception as exc:
                if config.flow_engine == "legacy":
                    raise GenerationError(f"Legacy GAN flow failed: {exc}") from exc
                progress(f"Legacy GAN unavailable; using deterministic flow ({exc})")
        if flow_result is None:
            progress(f"Generating deterministic flow at {config.flow_scale:.3f}× spacing")
            flow_result = _deterministic_flow(converted, config.seed, config.flow_scale)
        osu_array, data = flow_result

        modding = importlib.import_module("act_modding")
        osu_array, data = modding.step7_modding(osu_array, data, dict(preset.modding_params))
        final = importlib.import_module("act_final")
        hitsounds = None
        if preset.mode is GameMode.TAIKO and preset.hitsound_dataset:
            taiko = importlib.import_module("act_taiko_hitsounds")
            params = taiko.step8_taiko_hitsounds_set_params(divisor=4, metronome_count=4)
            hitsounds = taiko.step8_apply_taiko_hitsounds(
                osu_array, data, hs_dataset=str(preset.hitsound_dataset), params=params
            )
        objects = final.convert_to_osu_obj(osu_array, data, hitsounds=hitsounds)
        is_spinner = data[5]
        timestamps = data[3]
        for index, spinner in enumerate(is_spinner):
            if spinner:
                objects[index]["type"] = 8
                next_time = (
                    timestamps[index + 1]
                    if index + 1 < len(timestamps)
                    else timestamps[index] + 1000
                )
                objects[index]["spinnerEndTime"] = max(
                    int(timestamps[index]) + 1, int(next_time) - 20
                )
        return objects


def _modern_converted(
    document: BeatmapDocument, timestamps: Any, preset: PresetSpec
) -> tuple[Any, ...]:
    np, _ = _require_audio_stack()
    values = np.asarray(timestamps, dtype=int)
    timing = document.timing_points()
    ticks: list[int] = []
    for timestamp in values:
        active = timing["uts"][0]
        for point in timing["uts"]:
            if timestamp >= point["beginTime"]:
                active = point
            else:
                break
        ticks.append(int(round((timestamp - active["beginTime"]) / active["tickLength"] * 4)))
    count = len(values)
    predictions = np.zeros((count, 5), dtype=float)
    predictions[:, 0] = 1.0
    predictions[:, 1] = 1.0
    objects = np.ones(count, dtype=int)
    flags = np.zeros(count, dtype=int)
    slider_velocity = np.asarray(
        [_slider_length_at(timing["ts"], float(timestamp)) for timestamp in values],
        dtype=float,
    )
    slider_ticks = np.zeros(count, dtype=int)
    distance_multiplier = float(preset.rhythm_params[0])
    return (
        objects,
        predictions,
        np.asarray(ticks, dtype=int),
        values,
        flags.copy(),
        flags.copy(),
        flags.copy(),
        slider_velocity,
        slider_ticks,
        distance_multiplier,
    )


def _mania_objects(
    workspace: Path,
    preset: PresetSpec,
    progress: Progress,
) -> list[dict[str, Any]]:
    with _legacy_context(workspace):
        rhythm = importlib.import_module("mania_act_rhythm_calc")
        model = _load_model(preset.rhythm_model)
        prepared = rhythm.read_npz("mapthis.npz")
        progress("Predicting mania rhythm")
        predicted = rhythm.step5_predict_notes(model, prepared, preset.rhythm_params)
        notes_each_key = rhythm.step5_build_pattern(
            predicted, preset.rhythm_params, pattern_dataset=str(preset.pattern_dataset)
        )
        notes_each_key = rhythm.mania_modding(notes_each_key, dict(preset.modding_params))
        notes, key_count = rhythm.merge_objects_each_key(notes_each_key)
        final = importlib.import_module("mania_act_final")
        return final.convert_to_osu_mania_obj(notes, key_count)


def generate_document(
    document: BeatmapDocument,
    audio: Path,
    workspace: Path,
    preset: PresetSpec,
    config: GenerationConfig,
    progress: Progress = print,
) -> BeatmapDocument:
    configure_determinism(config.seed)
    document = ensure_preview_time(document, audio)
    requested_difficulty = resolve_standard_difficulty(config.difficulty_tier, config.target_stars)
    if requested_difficulty is not None:
        profile, _target_stars = requested_difficulty
        if document.mode is not GameMode.STANDARD or preset.mode is not GameMode.STANDARD:
            raise InputError("Fixed star difficulty generation supports osu!standard only.")
        document = apply_standard_difficulty(document, profile)
    if config.mode is not None and config.mode is not preset.mode:
        produced = preset.mode.name.lower()
        requested = config.mode.name.lower()
        raise InputError(f"Preset {preset.name} produces {produced}, not {requested}.")
    (workspace / "mapthis.json").write_text(
        json.dumps(document.legacy_dict(), ensure_ascii=False), encoding="utf-8"
    )
    if config.rhythm_engine == "modern":
        if preset.mode is not GameMode.STANDARD:
            raise InputError("The modern rhythm model currently supports osu!standard only.")
        from osumapper.training.inference import predict_modern_rhythm

        prediction = predict_modern_rhythm(
            document,
            audio,
            workspace,
            model_root=config.modern_model,
            threshold=config.rhythm_threshold,
            target_density=config.target_density,
            difficulty_tier=config.difficulty_tier,
            target_stars=config.target_stars,
            progress=progress,
        )
        progress(f"Modern rhythm selected {len(prediction.timestamps_ms)} timestamps")
        if config.flow_engine == "placement":
            from osumapper.training.placement_learning import predict_placement

            progress("Generating learned Placement-v1 flow and object types")
            objects = predict_placement(
                document,
                prediction.timestamps_ms,
                model_root=config.placement_model,
                target_density=prediction.target_density,
                seed=config.seed,
            )
        elif config.flow_engine in {"auto", "deterministic"}:
            from osumapper.patterns import plan_standard_patterns

            progress(f"Planning deterministic patterns at {config.flow_scale:.3f}× spacing")
            pattern_plan = plan_standard_patterns(
                document,
                prediction.timestamps_ms,
                difficulty_tier=config.difficulty_tier,
                target_stars=config.target_stars,
                seed=config.seed,
                flow_scale=config.flow_scale,
            )
            objects = list(pattern_plan.objects)
            summary = ", ".join(f"{name}={count}" for name, count in pattern_plan.counts.items())
            progress(f"PatternPlanner-v1: {summary}")
        else:
            converted = _modern_converted(document, prediction.timestamps_ms, preset)
            objects = _place_standard_objects(workspace, preset, config, progress, converted)
    else:
        progress("Preparing legacy rhythm audio features")
        _prepare_model_input(document, audio, workspace / "mapthis.npz")
        if preset.mode is GameMode.MANIA:
            objects = _mania_objects(workspace, preset, progress)
        else:
            objects = _standard_objects(workspace, preset, config, progress)
    if not objects:
        raise GenerationError("The model did not produce any hit objects.")
    progress(f"Serializing {len(objects)} hit objects")
    preset_label = (
        f"{preset.name}-modern-rhythm" if config.rhythm_engine == "modern" else preset.name
    )
    generated = document.with_hit_objects(objects, preset=preset_label, seed=config.seed)
    if requested_difficulty is not None:
        profile, target_stars = requested_difficulty
        actual_stars = calculate_standard_stars(generated)
        target_missed = (
            not profile.contains(actual_stars)
            or abs(actual_stars - target_stars) > STAR_TARGET_TOLERANCE
        )
        if target_missed and config.enforce_star_target:
            raise GenerationError(
                f"Generated {actual_stars:.2f}★ for a requested {target_stars:.2f}★ "
                f"{profile.label} target ({profile.range_label}, allowed target error "
                f"±{STAR_TARGET_TOLERANCE:.2f}★). "
                "Adjust --target-density or retrain the selected star-conditioned model "
                "with more maps in this band."
            )
        status = "Measured" if target_missed else "Verified"
        progress(f"{status} {profile.label}: {actual_stars:.2f}★ for requested {target_stars:.2f}★")
        generated = label_standard_difficulty(generated, profile, actual_stars)
    return generated
