from __future__ import annotations

import hashlib
import math
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from osumapper.beatmap import BeatmapDocument
from osumapper.config import GameMode
from osumapper.errors import InputError
from osumapper.training import DATASET_SCHEMA_VERSION


class BeatmapDataError(InputError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True, slots=True)
class ModernTimingPoint:
    time_ms: float
    beat_length_ms: float
    meter: int
    uninherited: bool
    bpm: float | None
    slider_velocity_multiplier: float
    sample_set: int
    sample_index: int
    volume: int
    effects: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ModernHitObject:
    time_ms: float
    end_time_ms: float
    x: int
    y: int
    kind: str
    type_flags: int
    new_combo: bool
    hitsound: int
    slider_duration_ms: float | None = None
    slider_curve_type: str | None = None
    slider_path: tuple[tuple[int, int], ...] = ()
    slider_repeats: int | None = None
    slider_pixel_length: float | None = None
    beat_snap: str | None = None
    beat_position: float | None = None

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["slider_path"] = [list(point) for point in self.slider_path]
        return value


@dataclass(frozen=True, slots=True)
class ParsedTrainingBeatmap:
    map_id: str
    map_sha256: str
    map_path: Path
    audio_path: Path
    song_id: str
    mapset_key: str
    metadata: dict[str, Any]
    difficulty: dict[str, float]
    timing_points: tuple[ModernTimingPoint, ...]
    hit_objects: tuple[ModernHitObject, ...]
    statistics: dict[str, Any]
    is_converted: bool

    def detail_dict(self) -> dict[str, Any]:
        return {
            "schema_version": DATASET_SCHEMA_VERSION,
            "map_id": self.map_id,
            "map_sha256": self.map_sha256,
            "map_path": str(self.map_path),
            "audio_path": str(self.audio_path),
            "song_id": self.song_id,
            "mapset_key": self.mapset_key,
            "metadata": self.metadata,
            "difficulty": self.difficulty,
            "timing_points": [point.as_dict() for point in self.timing_points],
            "hit_objects": [obj.as_dict() for obj in self.hit_objects],
            "statistics": self.statistics,
            "is_converted": self.is_converted,
        }


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_document(path: Path) -> BeatmapDocument:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise BeatmapDataError("unreadable_map", f"Could not read beatmap: {exc}") from exc
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        try:
            text = raw.decode("cp1252")
        except UnicodeDecodeError as exc:
            raise BeatmapDataError("unsupported_encoding", str(exc)) from exc
    try:
        return BeatmapDocument.from_text(text, path.resolve())
    except InputError as exc:
        raise BeatmapDataError("malformed_map", str(exc)) from exc


def read_training_document(path: Path) -> BeatmapDocument:
    """Read an osu! file with the encodings commonly found in stable Songs folders."""
    return _read_document(path.expanduser().resolve())


def _float(values: dict[str, str], key: str, default: float) -> float:
    try:
        value = float(values.get(key, default))
    except (TypeError, ValueError) as exc:
        raise BeatmapDataError("malformed_difficulty", f"Invalid {key}") from exc
    if not math.isfinite(value):
        raise BeatmapDataError("malformed_difficulty", f"Non-finite {key}")
    return value


def _integer(value: str | None, default: int = -1) -> int:
    try:
        return int(value) if value not in (None, "") else default
    except ValueError:
        return default


def parse_timing_points(document: BeatmapDocument) -> tuple[ModernTimingPoint, ...]:
    points: list[ModernTimingPoint] = []
    active_beat_length: float | None = None
    for line_number, line in enumerate(document.sections().get("TimingPoints", []), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("//"):
            continue
        fields = [field.strip() for field in stripped.split(",")]
        try:
            if len(fields) < 2:
                raise ValueError("fewer than two fields")
            time_ms = float(fields[0])
            raw_beat_length = float(fields[1])
            meter = int(fields[2] or 4) if len(fields) > 2 else 4
            sample_set = int(fields[3] or 0) if len(fields) > 3 else 0
            sample_index = int(fields[4] or 0) if len(fields) > 4 else 0
            volume = int(fields[5] or 100) if len(fields) > 5 else 100
            uninherited = len(fields) < 7 or fields[6] != "0"
            effects = int(fields[7] or 0) if len(fields) > 7 else 0
        except ValueError as exc:
            raise BeatmapDataError(
                "malformed_timing", f"Invalid timing point line {line_number}: {line}"
            ) from exc
        if not math.isfinite(time_ms) or not math.isfinite(raw_beat_length):
            raise BeatmapDataError("malformed_timing", "Non-finite timing point")
        if uninherited:
            if raw_beat_length <= 0:
                raise BeatmapDataError(
                    "malformed_timing", "Uninherited timing point has non-positive beat length"
                )
            active_beat_length = raw_beat_length
            bpm = 60_000.0 / raw_beat_length
            velocity = 1.0
        else:
            if active_beat_length is None:
                raise BeatmapDataError(
                    "malformed_timing", "Inherited point occurs before a timing section"
                )
            bpm = None
            velocity = min(10.0, max(0.1, -100.0 / raw_beat_length)) if raw_beat_length < 0 else 1.0
        points.append(
            ModernTimingPoint(
                time_ms=time_ms,
                beat_length_ms=active_beat_length,
                meter=max(1, meter),
                uninherited=uninherited,
                bpm=bpm,
                slider_velocity_multiplier=velocity,
                sample_set=sample_set,
                sample_index=sample_index,
                volume=max(0, min(100, volume)),
                effects=effects,
            )
        )
    if not any(point.uninherited for point in points):
        raise BeatmapDataError("malformed_timing", "No uninherited timing points")
    return tuple(sorted(points, key=lambda point: point.time_ms))


def active_timing(
    points: tuple[ModernTimingPoint, ...], timestamp: float
) -> tuple[ModernTimingPoint, float]:
    red: ModernTimingPoint | None = None
    velocity = 1.0
    for point in points:
        if point.time_ms > timestamp:
            break
        if point.uninherited:
            red = point
            velocity = 1.0
        else:
            velocity = point.slider_velocity_multiplier
    if red is None:
        red = next(point for point in points if point.uninherited)
    return red, velocity


def beat_snap(timestamp: float, points: tuple[ModernTimingPoint, ...]) -> tuple[str, float]:
    timing, _ = active_timing(points, timestamp)
    beat_position = (timestamp - timing.time_ms) / timing.beat_length_ms
    candidates = (1, 2, 3, 4, 6, 8)
    divisor = min(
        candidates,
        key=lambda value: (abs(beat_position * value - round(beat_position * value)), value),
    )
    return f"1/{divisor}", beat_position


def _slider_path(value: str) -> tuple[str, tuple[tuple[int, int], ...]]:
    parts = value.split("|")
    curve_type = parts[0] if parts and parts[0] else "L"
    points: list[tuple[int, int]] = []
    for raw_point in parts[1:]:
        try:
            raw_x, raw_y = raw_point.split(":", 1)
            points.append((int(float(raw_x)), int(float(raw_y))))
        except (ValueError, TypeError) as exc:
            raise BeatmapDataError(
                "malformed_objects", f"Invalid slider point: {raw_point}"
            ) from exc
    return curve_type, tuple(points)


def parse_hit_objects(
    document: BeatmapDocument,
    timing_points: tuple[ModernTimingPoint, ...],
    slider_multiplier: float,
) -> tuple[ModernHitObject, ...]:
    objects: list[ModernHitObject] = []
    for line_number, line in enumerate(document.sections().get("HitObjects", []), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("//"):
            continue
        fields = [field.strip() for field in stripped.split(",")]
        try:
            if len(fields) < 5:
                raise ValueError("fewer than five fields")
            x = int(float(fields[0]))
            y = int(float(fields[1]))
            timestamp = float(fields[2])
            type_flags = int(fields[3])
            hitsound = int(fields[4])
        except ValueError as exc:
            raise BeatmapDataError(
                "malformed_objects", f"Invalid hit object line {line_number}: {line}"
            ) from exc
        if not math.isfinite(timestamp) or timestamp < 0:
            raise BeatmapDataError("malformed_objects", "Invalid hit object timestamp")

        kind = "circle"
        end_time = timestamp
        slider_duration: float | None = None
        curve_type: str | None = None
        path: tuple[tuple[int, int], ...] = ()
        repeats: int | None = None
        pixel_length: float | None = None
        if type_flags & 2:
            kind = "slider"
            try:
                curve_type, path = _slider_path(fields[5])
                repeats = max(1, int(fields[6]))
                pixel_length = max(0.0, float(fields[7]))
            except (IndexError, ValueError) as exc:
                raise BeatmapDataError(
                    "malformed_objects", f"Invalid slider line {line_number}: {line}"
                ) from exc
            timing, velocity = active_timing(timing_points, timestamp)
            pixels_per_beat = max(1e-6, slider_multiplier * 100.0 * velocity)
            slider_duration = pixel_length / pixels_per_beat * timing.beat_length_ms * repeats
            end_time = timestamp + slider_duration
        elif type_flags & 8:
            kind = "spinner"
            try:
                end_time = max(timestamp, float(fields[5]))
            except (IndexError, ValueError) as exc:
                raise BeatmapDataError(
                    "malformed_objects", f"Invalid spinner line {line_number}: {line}"
                ) from exc
        elif type_flags & 128:
            kind = "hold"
            try:
                end_time = max(timestamp, float(fields[5].split(":", 1)[0]))
            except (IndexError, ValueError) as exc:
                raise BeatmapDataError(
                    "malformed_objects", f"Invalid hold line {line_number}: {line}"
                ) from exc

        snap, beat_position = beat_snap(timestamp, timing_points)
        objects.append(
            ModernHitObject(
                time_ms=timestamp,
                end_time_ms=end_time,
                x=x,
                y=y,
                kind=kind,
                type_flags=type_flags,
                new_combo=bool(type_flags & 4),
                hitsound=hitsound,
                slider_duration_ms=slider_duration,
                slider_curve_type=curve_type,
                slider_path=path,
                slider_repeats=repeats,
                slider_pixel_length=pixel_length,
                beat_snap=snap,
                beat_position=beat_position,
            )
        )
    if any(
        later.time_ms < earlier.time_ms
        for earlier, later in zip(objects, objects[1:], strict=False)
    ):
        raise BeatmapDataError("broken_object_order", "Hit objects are not time-sorted")
    return tuple(objects)


def _run_counts(intervals: list[float], threshold_ms: float = 150.0) -> tuple[int, int]:
    bursts = 0
    streams = 0
    run = 0
    for interval in [*intervals, math.inf]:
        if 0 < interval <= threshold_ms:
            run += 1
            continue
        object_count = run + 1
        if object_count >= 5:
            streams += 1
        elif object_count >= 3:
            bursts += 1
        run = 0
    return bursts, streams


def derive_statistics(
    objects: tuple[ModernHitObject, ...], timing_points: tuple[ModernTimingPoint, ...]
) -> dict[str, Any]:
    if objects:
        start = objects[0].time_ms
        end = max(obj.end_time_ms for obj in objects)
    else:
        start = end = 0.0
    duration = max(0.0, end - start)
    playable = [obj for obj in objects if obj.kind != "spinner"]
    jumps = [
        math.dist((first.x, first.y), (second.x, second.y))
        for first, second in zip(playable, playable[1:], strict=False)
    ]
    intervals = [
        second.time_ms - first.time_ms for first, second in zip(objects, objects[1:], strict=False)
    ]
    positive_intervals = [value for value in intervals if value > 0]
    bpms = [point.bpm for point in timing_points if point.uninherited and point.bpm is not None]
    snap_counts: dict[str, int] = {}
    for obj in objects:
        if obj.beat_snap:
            snap_counts[obj.beat_snap] = snap_counts.get(obj.beat_snap, 0) + 1
    bursts, streams = _run_counts(positive_intervals)
    circles = sum(obj.kind == "circle" for obj in objects)
    sliders = sum(obj.kind == "slider" for obj in objects)
    spinners = sum(obj.kind == "spinner" for obj in objects)
    return {
        "map_start_ms": start,
        "map_end_ms": end,
        "map_duration_ms": duration,
        "object_count": len(objects),
        "circle_count": circles,
        "slider_count": sliders,
        "spinner_count": spinners,
        "objects_per_second": len(objects) / (duration / 1000.0) if duration > 0 else 0.0,
        "slider_percentage": sliders / len(objects) if objects else 0.0,
        "average_spacing": statistics.fmean(jumps) if jumps else 0.0,
        "median_jump_distance": statistics.median(jumps) if jumps else 0.0,
        "max_jump_distance": max(jumps, default=0.0),
        "average_rhythm_interval": statistics.fmean(positive_intervals)
        if positive_intervals
        else 0.0,
        "median_rhythm_interval": statistics.median(positive_intervals)
        if positive_intervals
        else 0.0,
        "jump_distances": jumps,
        "rhythm_intervals": intervals,
        "beat_snap_counts": snap_counts,
        "burst_count": bursts,
        "stream_count": streams,
        "bpm_min": min(bpms, default=0.0),
        "bpm_max": max(bpms, default=0.0),
        "bpm_mean": statistics.fmean(bpms) if bpms else 0.0,
        "timing_change_count": len(bpms),
    }


def _resolve_audio(document: BeatmapDocument, songs_root: Path) -> Path:
    raw_name = document.audio_filename.strip().strip('"').replace("\\", "/")
    pure = PurePosixPath(raw_name)
    if pure.is_absolute() or ".." in pure.parts or not pure.parts:
        raise BeatmapDataError("unsafe_audio_path", f"Unsafe AudioFilename: {raw_name}")
    audio = (document.path.parent / Path(*pure.parts)).resolve()
    if not audio.is_relative_to(songs_root):
        raise BeatmapDataError("unsafe_audio_path", f"Audio escapes Songs root: {audio}")
    if not audio.is_file():
        raise BeatmapDataError("missing_audio", f"Referenced audio does not exist: {audio}")
    return audio


def parse_standard_beatmap(path: Path, songs_root: Path | None = None) -> ParsedTrainingBeatmap:
    path = path.expanduser().resolve()
    root = (songs_root or path.parent).expanduser().resolve()
    document = _read_document(path)
    if document.mode is not GameMode.STANDARD:
        raise BeatmapDataError("non_standard_mode", f"Mode is {int(document.mode)}, not 0")
    audio = _resolve_audio(document, root)
    metadata_values = document.values("Metadata")
    difficulty_values = document.values("Difficulty")
    difficulty = {
        "hp": _float(difficulty_values, "HPDrainRate", 5.0),
        "cs": _float(difficulty_values, "CircleSize", 4.0),
        "od": _float(difficulty_values, "OverallDifficulty", 5.0),
        "ar": _float(
            difficulty_values,
            "ApproachRate",
            _float(difficulty_values, "OverallDifficulty", 5.0),
        ),
        "slider_multiplier": _float(difficulty_values, "SliderMultiplier", 1.4),
        "slider_tick_rate": _float(difficulty_values, "SliderTickRate", 1.0),
    }
    if difficulty["slider_multiplier"] <= 0 or difficulty["slider_tick_rate"] <= 0:
        raise BeatmapDataError("malformed_difficulty", "Slider settings must be positive")
    timing_points = parse_timing_points(document)
    objects = parse_hit_objects(document, timing_points, difficulty["slider_multiplier"])
    statistics_value = derive_statistics(objects, timing_points)
    map_sha = file_sha256(path)
    beatmap_id = _integer(metadata_values.get("BeatmapID"))
    beatmapset_id = _integer(metadata_values.get("BeatmapSetID"))
    if beatmapset_id > 0:
        mapset_key = f"set:{beatmapset_id}"
        song_id = f"set-{beatmapset_id}"
    else:
        identity = str(audio).replace("\\", "/").casefold().encode("utf-8")
        audio_key = hashlib.sha256(identity).hexdigest()[:20]
        mapset_key = f"audio:{audio_key}"
        song_id = f"audio-{audio_key}"
    map_id = f"b{beatmap_id}" if beatmap_id > 0 else f"sha-{map_sha[:20]}"
    tags = metadata_values.get("Tags", "")
    version = metadata_values.get("Version", "Unknown")
    converted_text = f" {tags} {version} ".casefold()
    is_converted = " converted " in converted_text or "(converted)" in converted_text
    metadata: dict[str, Any] = {
        "artist": metadata_values.get("Artist", "Unknown"),
        "artist_unicode": metadata_values.get("ArtistUnicode", ""),
        "title": metadata_values.get("Title", "Unknown"),
        "title_unicode": metadata_values.get("TitleUnicode", ""),
        "creator": metadata_values.get("Creator", "Unknown"),
        "version": version,
        "source": metadata_values.get("Source", ""),
        "tags": tags,
        "beatmap_id": beatmap_id,
        "beatmapset_id": beatmapset_id,
        "mode": 0,
    }
    return ParsedTrainingBeatmap(
        map_id=map_id,
        map_sha256=map_sha,
        map_path=path,
        audio_path=audio,
        song_id=song_id,
        mapset_key=mapset_key,
        metadata=metadata,
        difficulty=difficulty,
        timing_points=timing_points,
        hit_objects=objects,
        statistics=statistics_value,
        is_converted=is_converted,
    )
