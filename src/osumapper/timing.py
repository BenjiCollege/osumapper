from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from osumapper.config import GameMode
from osumapper.errors import DependencyError, InputError


@dataclass(frozen=True, slots=True)
class TimingEstimate:
    bpm: float
    offset_ms: int
    confidence: str


def estimate_timing(audio: Path) -> TimingEstimate:
    """Estimate a single timing section using maintained, cross-platform librosa APIs."""
    try:
        import librosa
        import numpy as np
    except ImportError as exc:
        raise DependencyError(
            "Automatic timing requires the project dependencies. Run `uv sync`."
        ) from exc

    try:
        signal, sample_rate = librosa.load(audio, sr=None, mono=True)
    except Exception as exc:
        raise InputError(f"Could not decode audio file {audio}: {exc}") from exc
    if signal.size == 0:
        raise InputError(f"Audio file is empty: {audio}")

    tempo, beat_frames = librosa.beat.beat_track(y=signal, sr=sample_rate, units="frames")
    tempo_value = float(np.asarray(tempo).reshape(-1)[0])
    if not np.isfinite(tempo_value) or tempo_value <= 0:
        raise InputError("Automatic timing could not find a stable tempo; provide --bpm.")
    if len(beat_frames):
        beat_times = librosa.frames_to_time(beat_frames, sr=sample_rate)
        offset = int(round(float(beat_times[0]) * 1000))
        confidence = "beat-tracked"
    else:
        offset = 0
        confidence = "tempo-only"
    return TimingEstimate(bpm=round(tempo_value, 6), offset_ms=offset, confidence=confidence)


def create_timed_beatmap(
    audio_name: str,
    *,
    mode: GameMode,
    bpm: float,
    offset_ms: int,
    key_count: int = 4,
    title: str = "Untitled",
    artist: str = "Unknown Artist",
) -> str:
    if bpm <= 0:
        raise InputError("BPM must be greater than zero.")
    circle_size = key_count if mode is GameMode.MANIA else 4
    beat_length = 60000.0 / bpm
    return f"""osu file format v14

[General]
AudioFilename: {audio_name}
AudioLeadIn: 0
PreviewTime: -1
Countdown: 0
SampleSet: Soft
StackLeniency: 0.5
Mode: {int(mode)}
LetterboxInBreaks: 0
WidescreenStoryboard: 1

[Editor]
DistanceSpacing: 1
BeatDivisor: 4
GridSize: 8
TimelineZoom: 1

[Metadata]
Title:{title}
TitleUnicode:{title}
Artist:{artist}
ArtistUnicode:{artist}
Creator:osumapper
Version:Auto-timed
Source:
Tags:osumapper generated

[Difficulty]
HPDrainRate:5
CircleSize:{circle_size}
OverallDifficulty:7
ApproachRate:8
SliderMultiplier:1.4
SliderTickRate:1

[Events]

[TimingPoints]
{offset_ms},{beat_length:.12g},4,2,1,100,1,0

[Colours]
Combo1 : 255,128,128
Combo2 : 128,192,255

[HitObjects]
"""
