from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from osumapper.errors import DependencyError, InputError
from osumapper.training.config import AudioFeatureConfig, DatasetPaths
from osumapper.training.storage import read_parquet, write_json

Progress = Callable[[str], None]


@dataclass(frozen=True, slots=True)
class FeatureResult:
    path: Path
    cached: bool
    frames: int
    duration_seconds: float


@dataclass(frozen=True, slots=True)
class FeatureBatchSummary:
    songs: int
    extracted: int
    cached: int
    failed: int
    root: Path

    def as_dict(self) -> dict[str, Any]:
        return {
            "songs": self.songs,
            "extracted": self.extracted,
            "cached": self.cached,
            "failed": self.failed,
            "root": str(self.root),
        }


def _audio_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _config_json(config: AudioFeatureConfig) -> str:
    return json.dumps(config.as_dict(), sort_keys=True, separators=(",", ":"))


def _require_audio() -> tuple[Any, Any]:
    try:
        import librosa
        import numpy as np
    except ImportError as exc:
        raise DependencyError("Audio feature extraction requires librosa and NumPy.") from exc
    return np, librosa


def _normalize(values: Any, np: Any) -> Any:
    maximum = float(np.max(np.abs(values))) if values.size else 0.0
    return values / maximum if maximum > 1e-12 else np.zeros_like(values)


def _cached_metadata(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    np, _ = _require_audio()
    try:
        with np.load(path, allow_pickle=False) as archive:
            return json.loads(str(archive["metadata_json"].item()))
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return None


def extract_audio_features(
    audio_path: Path,
    output_path: Path,
    *,
    config: AudioFeatureConfig | None = None,
    force: bool = False,
) -> FeatureResult:
    audio = audio_path.expanduser().resolve()
    if not audio.is_file():
        raise InputError(f"Audio file does not exist: {audio}")
    output = output_path.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    feature_config = config or AudioFeatureConfig()
    source_sha = _audio_hash(audio)
    expected = {
        "version": 1,
        "audio_sha256": source_sha,
        "config_json": _config_json(feature_config),
    }
    cached = _cached_metadata(output)
    if (
        not force
        and cached is not None
        and all(cached.get(key) == value for key, value in expected.items())
    ):
        return FeatureResult(
            path=output,
            cached=True,
            frames=int(cached["frames"]),
            duration_seconds=float(cached["duration_seconds"]),
        )

    np, librosa = _require_audio()
    try:
        signal, sample_rate = librosa.load(
            audio, sr=feature_config.sample_rate, mono=True, dtype=np.float32
        )
    except Exception as exc:
        raise InputError(f"Could not decode audio {audio}: {exc}") from exc
    if signal.size == 0:
        raise InputError(f"Audio file is empty: {audio}")
    stft = np.abs(
        librosa.stft(
            signal,
            n_fft=feature_config.n_fft,
            hop_length=feature_config.hop_length,
            center=True,
        )
    )
    mel_power = librosa.feature.melspectrogram(
        S=stft**2,
        sr=sample_rate,
        n_mels=feature_config.n_mels,
        fmin=feature_config.fmin,
        fmax=min(feature_config.fmax, sample_rate / 2),
    )
    mel_db = librosa.power_to_db(mel_power, ref=np.max, top_db=80.0)
    mel = np.clip((mel_db + 80.0) / 80.0, 0.0, 1.0).T
    onset = librosa.onset.onset_strength(
        y=signal, sr=sample_rate, hop_length=feature_config.hop_length
    )
    rms = librosa.feature.rms(
        y=signal,
        frame_length=feature_config.n_fft,
        hop_length=feature_config.hop_length,
        center=True,
    )[0]
    flux = np.concatenate(
        [np.zeros(1, dtype=np.float32), np.maximum(0.0, np.diff(stft, axis=1)).mean(axis=0)]
    )
    tempo, beat_frames = librosa.beat.beat_track(
        onset_envelope=onset,
        sr=sample_rate,
        hop_length=feature_config.hop_length,
        units="frames",
    )
    frame_count = min(mel.shape[0], onset.shape[0], rms.shape[0], flux.shape[0])
    if frame_count < 2:
        raise InputError(f"Audio produced too few feature frames: {audio}")
    mel = mel[:frame_count].astype(np.float32)
    onset = _normalize(onset[:frame_count], np).astype(np.float32)
    rms = _normalize(rms[:frame_count], np).astype(np.float32)
    flux = _normalize(flux[:frame_count], np).astype(np.float32)
    beat_pulse = np.zeros(frame_count, dtype=np.float32)
    valid_beats = np.asarray(beat_frames, dtype=int)
    valid_beats = valid_beats[(valid_beats >= 0) & (valid_beats < frame_count)]
    beat_pulse[valid_beats] = 1.0
    frame_times_ms = (
        np.arange(frame_count, dtype=np.float64) * feature_config.hop_length / sample_rate * 1000.0
    )
    duration = float(signal.shape[0] / sample_rate)
    metadata = {
        **expected,
        "audio_path": str(audio),
        "sample_rate": sample_rate,
        "hop_length": feature_config.hop_length,
        "frames": frame_count,
        "duration_seconds": duration,
        "tempo": float(np.asarray(tempo).reshape(-1)[0]),
    }
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with temporary.open("wb") as stream:
            np.savez_compressed(
                stream,
                mel=mel,
                onset=onset,
                rms=rms,
                spectral_flux=flux,
                beat_pulse=beat_pulse,
                frame_times_ms=frame_times_ms,
                metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
            )
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return FeatureResult(output, False, frame_count, duration)


def feature_path(song_id: str, *, dataset_root: Path | None = None) -> Path:
    return DatasetPaths.at(dataset_root).features / f"{song_id}.npz"


def extract_dataset_features(
    *,
    dataset_root: Path | None = None,
    config: AudioFeatureConfig | None = None,
    force: bool = False,
    progress: Progress = print,
) -> FeatureBatchSummary:
    paths = DatasetPaths.at(dataset_root)
    paths.create()
    rows = read_parquet(paths.dataset)
    songs: dict[str, set[str]] = {}
    for row in rows:
        if not row["eligible"]:
            continue
        songs.setdefault(str(row["song_id"]), set()).add(str(row["audio_path"]))
    errors: list[dict[str, str]] = []
    extracted = 0
    cached_count = 0
    for index, (song_id, audio_paths) in enumerate(sorted(songs.items()), start=1):
        if len(audio_paths) != 1:
            errors.append(
                {
                    "song_id": song_id,
                    "error": "Mapset references multiple audio files",
                    "audio_paths": ";".join(sorted(audio_paths)),
                }
            )
            continue
        audio = Path(next(iter(audio_paths)))
        try:
            result = extract_audio_features(
                audio,
                paths.features / f"{song_id}.npz",
                config=config,
                force=force,
            )
            if result.cached:
                cached_count += 1
            else:
                extracted += 1
        except InputError as exc:
            errors.append({"song_id": song_id, "error": str(exc), "audio_paths": str(audio)})
        if index % 25 == 0:
            progress(f"Processed features for {index}/{len(songs)} songs")
    write_json(paths.features / "errors.json", errors)
    summary = FeatureBatchSummary(
        songs=len(songs),
        extracted=extracted,
        cached=cached_count,
        failed=len(errors),
        root=paths.features,
    )
    write_json(
        paths.features / "manifest.json",
        {"version": 1, "config": (config or AudioFeatureConfig()).as_dict(), **summary.as_dict()},
    )
    return summary


def load_features(path: Path) -> dict[str, Any]:
    np, _ = _require_audio()
    if not path.is_file():
        raise InputError(f"Feature cache is missing: {path}. Run `osumapper dataset features`.")
    try:
        with np.load(path, allow_pickle=False) as archive:
            return {name: archive[name].copy() for name in archive.files}
    except (OSError, ValueError, KeyError) as exc:
        raise InputError(f"Could not read feature cache {path}: {exc}") from exc
