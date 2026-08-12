from __future__ import annotations

import math
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any

from osumapper.config import GameMode
from osumapper.difficulty import calculate_standard_stars, difficulty_for_stars
from osumapper.errors import InputError
from osumapper.training.beatmaps import (
    ModernHitObject,
    ModernTimingPoint,
    active_timing,
    parse_standard_beatmap,
    read_training_document,
)
from osumapper.training.storage import write_json

GENERAL_CRITERIA_URL = "https://osu.ppy.sh/wiki/en/Ranking_criteria"
STANDARD_CRITERIA_URL = "https://osu.ppy.sh/wiki/en/Ranking_criteria/osu%21"
CRITERIA_SNAPSHOT_DATE = "2026-08-11"
SUPPORTED_SNAP_DIVISORS = (1, 2, 3, 4, 6, 8, 12, 16)

_SETTING_GUIDELINES: dict[str, dict[str, tuple[float | None, float | None]]] = {
    "easy": {"ar": (None, 5), "od": (1, 3), "hp": (1, 3), "cs": (None, 4)},
    "normal": {"ar": (4, 6), "od": (3, 5), "hp": (3, 5), "cs": (None, 5)},
    "hard": {"ar": (6, 8), "od": (5, 7), "hp": (4, 6), "cs": (None, 6)},
    "insane": {"ar": (7, 9.3), "od": (7, 9), "hp": (5, 8), "cs": (None, 7)},
    "expert": {"ar": (8, None), "od": (8, None), "hp": (5, None), "cs": (None, 7)},
    "expert-plus": {
        "ar": (8, None),
        "od": (8, None),
        "hp": (5, None),
        "cs": (None, 7),
    },
}

_SPINNER_GUIDELINES = {
    "easy": (4.0, 4.0),
    "normal": (3.0, 2.0),
    "hard": (2.0, 1.0),
}

_MANUAL_CHECKS = (
    (
        "human_authorship",
        "Confirm that ranking-bound hit objects, timing, and hitsounds were created by "
        "direct human input; generated drafts are not eligible for ranking.",
    ),
    (
        "content_permission",
        "Confirm permission and required attribution for the song, background, video, "
        "storyboard, samples, and skin elements.",
    ),
    (
        "musical_timing",
        "Verify BPM, offsets, time signatures, rhythm snapping, and musical cue selection "
        "by listening and test playing.",
    ),
    (
        "hitsound_feedback",
        "Verify every actively clicked part has audible, suitable, low-latency hitsound feedback.",
    ),
    (
        "slider_readability",
        "Verify slider paths, reverse arrows, ends, overlaps, velocity, and feedback remain clear.",
    ),
    (
        "visual_readability",
        "Verify backgrounds, combo colours, slider colours, skins, video, and storyboards "
        "remain visible, safe, credited, and readable.",
    ),
    (
        "difficulty_and_spread",
        "Verify musical intensity, difficulty spikes, names, drain-time spread, and the full "
        "mapset progression with human reviewers.",
    ),
    (
        "audio_quality",
        "Listen for clipping or distortion and verify bitrate, sample rate, source quality, "
        "preview consistency, and any required audio edits.",
    ),
)


def _pattern_summary(objects: tuple[ModernHitObject, ...]) -> dict[str, Any]:
    type_counts = Counter(obj.kind for obj in objects)
    playable = [obj for obj in objects if obj.kind != "spinner"]
    transitions = list(zip(playable, playable[1:], strict=False))
    distances = [
        math.hypot(float(current.x - previous.x), float(current.y - previous.y))
        for previous, current in transitions
    ]
    stacks = sum(
        distance < 8.0 and current.time_ms - previous.time_ms <= 1_000.0
        for (previous, current), distance in zip(transitions, distances, strict=True)
    )
    fast_group_lengths: list[int] = []
    start = 0
    while start + 1 < len(objects):
        if objects[start + 1].time_ms - objects[start].time_ms > 150.0:
            start += 1
            continue
        end = start + 1
        while end + 1 < len(objects) and objects[end + 1].time_ms - objects[end].time_ms <= 150.0:
            end += 1
        length = end - start + 1
        if length >= 3:
            fast_group_lengths.append(length)
        start = end + 1
    unique_positions = len({(obj.x, obj.y) for obj in playable})
    return {
        "analyzer": "pattern-summary-v1",
        "object_types": {
            "circles": type_counts["circle"],
            "sliders": type_counts["slider"],
            "spinners": type_counts["spinner"],
        },
        "patterns": {
            "jumps_over_160px": sum(distance >= 160.0 for distance in distances),
            "bursts_3_to_4_objects": sum(length in {3, 4} for length in fast_group_lengths),
            "streams_5_plus_objects": sum(length >= 5 for length in fast_group_lengths),
            "stacks_below_8px": stacks,
        },
        "object_type_ratios": {
            "slider": type_counts["slider"] / len(objects) if objects else 0.0,
            "spinner": type_counts["spinner"] / len(objects) if objects else 0.0,
        },
        "unique_position_ratio": unique_positions / len(playable) if playable else 0.0,
    }


def _issue(
    issues: list[dict[str, Any]],
    severity: str,
    code: str,
    message: str,
    *,
    rule_type: str,
    count: int = 1,
    timestamps_ms: list[float] | None = None,
) -> None:
    if count <= 0:
        return
    item: dict[str, Any] = {
        "severity": severity,
        "rule_type": rule_type,
        "code": code,
        "message": message,
        "count": count,
    }
    if timestamps_ms:
        item["timestamps_ms"] = [round(value, 3) for value in timestamps_ms[:20]]
    issues.append(item)


def _nearest_snap_error_ms(
    timestamp: float,
    timing_points: tuple[ModernTimingPoint, ...],
) -> float:
    timing, _ = active_timing(timing_points, timestamp)
    position = (timestamp - timing.time_ms) / timing.beat_length_ms
    return min(
        abs(position * divisor - round(position * divisor)) * timing.beat_length_ms / divisor
        for divisor in SUPPORTED_SNAP_DIVISORS
    )


def _overlap_violations(
    objects: tuple[ModernHitObject, ...],
    timing_points: tuple[ModernTimingPoint, ...],
    difficulty: str,
) -> list[float]:
    beat_limits = {"easy": 1.0, "normal": 1.0, "hard": 0.5, "insane": 0.25}
    beat_limit = beat_limits.get(difficulty)
    if beat_limit is None:
        return []
    overlaps: list[float] = []
    for previous, current in zip(objects, objects[1:], strict=False):
        if previous.x != current.x or previous.y != current.y:
            continue
        timing, _ = active_timing(timing_points, previous.time_ms)
        beats = (current.time_ms - previous.time_ms) / timing.beat_length_ms
        if beats <= beat_limit + 1e-9:
            overlaps.append(current.time_ms)
    return overlaps


def _background(document: Any) -> str | None:
    for line in document.sections().get("Events", []):
        stripped = line.strip()
        if not stripped or stripped.startswith("//"):
            continue
        fields = [field.strip().strip('"') for field in stripped.split(",")]
        if len(fields) >= 3 and fields[0] == "0" and fields[1] == "0":
            return fields[2]
    return None


def _safe_relative_file(root: Path, value: str) -> Path | None:
    normalized = value.replace("\\", "/")
    pure = PurePosixPath(normalized)
    if pure.is_absolute() or ".." in pure.parts or not pure.parts:
        return None
    candidate = (root / Path(*pure.parts)).resolve()
    return candidate if candidate.is_relative_to(root.resolve()) else None


def audit_standard_criteria(
    map_path: Path,
    *,
    output: Path | None = None,
    map_label: str | None = None,
) -> dict[str, Any]:
    """Audit the deterministic subset of the current osu!standard ranking criteria.

    This is deliberately an aid, not a rankability certificate. Rules requiring
    listening, visual inspection, permission checks, or human judgement are listed
    separately and never reported as automatically passed.
    """
    source = map_path.expanduser().resolve()
    document = read_training_document(source)
    if document.mode is not GameMode.STANDARD:
        raise InputError("Ranking-criteria audit currently supports osu!standard only.")
    parsed = parse_standard_beatmap(source, source.parent)
    stars = calculate_standard_stars(document)
    profile = difficulty_for_stars(stars)
    issues: list[dict[str, Any]] = []

    creator = (document.value("Metadata", "Creator", "") or "").casefold()
    version = (document.value("Metadata", "Version", "") or "").casefold()
    generated = "osumapper" in creator or "osumapper" in version
    if generated:
        _issue(
            issues,
            "error",
            "generated_content_not_rankable",
            "Current osu! ranking criteria prohibit generative tooling for hit objects, "
            "timing, and hitsounds. Treat this output only as a local draft.",
            rule_type="policy",
        )

    objects = parsed.hit_objects
    duplicate_times = [
        timestamp
        for timestamp, count in Counter(round(obj.time_ms, 3) for obj in objects).items()
        if count > 1
    ]
    _issue(
        issues,
        "error",
        "objects_on_same_tick",
        "Multiple hit objects start at the same timestamp.",
        rule_type="rule",
        count=len(duplicate_times),
        timestamps_ms=duplicate_times,
    )

    circle_gap_violations: list[float] = []
    long_object_gap_violations: list[float] = []
    for previous, current in zip(objects, objects[1:], strict=False):
        if previous.kind == "circle" and current.time_ms - previous.time_ms < 10.0:
            circle_gap_violations.append(current.time_ms)
        if previous.kind in {"slider", "spinner"} and current.time_ms - previous.end_time_ms < 20.0:
            long_object_gap_violations.append(current.time_ms)
    _issue(
        issues,
        "error",
        "circle_gap_below_10ms",
        "A hit circle is followed by another object in less than 10 ms.",
        rule_type="rule",
        count=len(circle_gap_violations),
        timestamps_ms=circle_gap_violations,
    )
    _issue(
        issues,
        "error",
        "long_object_gap_below_20ms",
        "An object begins less than 20 ms after a slider or spinner ends.",
        rule_type="rule",
        count=len(long_object_gap_violations),
        timestamps_ms=long_object_gap_violations,
    )

    cs = parsed.difficulty["cs"]
    radius = max(0.0, 54.4 - 4.48 * cs)
    offscreen: list[float] = []
    for obj in objects:
        if obj.kind == "spinner":
            if (obj.x, obj.y) != (256, 192):
                offscreen.append(obj.time_ms)
            continue
        points = ((obj.x, obj.y), *obj.slider_path)
        if any(
            x - radius < 0 or x + radius > 512 or y - radius < 0 or y + radius > 384
            for x, y in points
        ):
            offscreen.append(obj.time_ms)
    _issue(
        issues,
        "error",
        "objects_partially_offscreen_4_3",
        "Object geometry extends beyond the 4:3 osu! playfield.",
        rule_type="rule",
        count=len(offscreen),
        timestamps_ms=offscreen,
    )

    overlaps = _overlap_violations(objects, parsed.timing_points, profile.key)
    if profile.key in {"easy", "normal", "hard"}:
        overlap_severity, overlap_type = "error", "rule"
    else:
        overlap_severity, overlap_type = "warning", "guideline"
    _issue(
        issues,
        overlap_severity,
        "difficulty_overlap",
        f"Fully overlapping consecutive objects violate the {profile.label} timing limit.",
        rule_type=overlap_type,
        count=len(overlaps),
        timestamps_ms=overlaps,
    )

    timing_groups = Counter(
        (round(point.time_ms, 3), point.uninherited) for point in parsed.timing_points
    )
    duplicate_timing = [time for (time, _kind), count in timing_groups.items() if count > 1]
    _issue(
        issues,
        "error",
        "duplicate_timing_points",
        "Two timing points of the same inheritance type share a timestamp.",
        rule_type="rule",
        count=len(duplicate_timing),
        timestamps_ms=duplicate_timing,
    )
    first_uninherited = next(point for point in parsed.timing_points if point.uninherited)
    if first_uninherited.effects & 1:
        _issue(
            issues,
            "error",
            "first_uninherited_point_enables_kiai",
            "The first uninherited timing point enables kiai.",
            rule_type="rule",
            timestamps_ms=[first_uninherited.time_ms],
        )

    unsnapped: list[float] = []
    for obj in objects:
        for timestamp in (
            (obj.time_ms, obj.end_time_ms) if obj.kind in {"slider", "spinner"} else (obj.time_ms,)
        ):
            if _nearest_snap_error_ms(timestamp, parsed.timing_points) >= 2.0:
                unsnapped.append(timestamp)
    _issue(
        issues,
        "warning",
        "objects_not_within_2ms_of_supported_tick",
        "Object starts or ends are not within 2 ms of a common editor snap; confirm "
        "unsupported divisors manually.",
        rule_type="rule_with_manual_exception",
        count=len(unsnapped),
        timestamps_ms=unsnapped,
    )

    if parsed.statistics["map_duration_ms"] < 30_000:
        _issue(
            issues,
            "error",
            "drain_time_below_30_seconds",
            "The difficulty has less than 30 seconds between its first and last object.",
            rule_type="rule",
        )

    audio_suffix = parsed.audio_path.suffix.casefold()
    if audio_suffix not in {".mp3", ".ogg"}:
        _issue(
            issues,
            "error",
            "unsupported_ranked_audio_format",
            "Ranking audio must use MP3 or Ogg Vorbis.",
            rule_type="rule",
        )
    preview = document.value("General", "PreviewTime")
    try:
        preview_time = int(preview or "-1")
    except ValueError:
        preview_time = -1
    if preview_time < 0:
        _issue(
            issues,
            "error",
            "preview_time_not_set",
            "A non-negative preview point is required and must match across difficulties.",
            rule_type="rule",
        )

    background_name = _background(document)
    background_path = (
        _safe_relative_file(source.parent, background_name) if background_name else None
    )
    if background_path is None or not background_path.is_file():
        _issue(
            issues,
            "error",
            "background_missing",
            "The difficulty must reference an existing background image.",
            rule_type="rule",
        )
    elif background_path.stat().st_size > 2_500_000:
        _issue(
            issues,
            "error",
            "background_above_2_5mb",
            "The referenced background image exceeds 2.5 MB.",
            rule_type="rule",
        )

    forced_default_skin = (
        document.value("General", "SkinPreference", "") or ""
    ).strip().casefold() == "default"
    combo_colours = [
        line
        for line in document.sections().get("Colours", [])
        if line.strip().casefold().startswith("combo") and ":" in line
    ]
    if not forced_default_skin and len(combo_colours) < 2:
        _issue(
            issues,
            "error",
            "insufficient_combo_colours",
            "At least two custom combo colours are required unless the default skin is forced.",
            rule_type="rule",
        )

    settings = _SETTING_GUIDELINES[profile.key]
    for name, (minimum, maximum) in settings.items():
        value = parsed.difficulty[name]
        if (minimum is not None and value < minimum) or (maximum is not None and value > maximum):
            bounds = (
                f"{minimum:g} or higher"
                if maximum is None
                else f"{maximum:g} or lower"
                if minimum is None
                else f"{minimum:g} to {maximum:g}"
            )
            _issue(
                issues,
                "warning",
                f"{profile.key}_{name}_outside_guideline",
                f"{profile.label} {name.upper()} is {value:g}; the guideline is {bounds}.",
                rule_type="guideline",
            )

    spinner_guideline = _SPINNER_GUIDELINES.get(profile.key)
    if spinner_guideline:
        minimum_length_beats, minimum_follow_gap_beats = spinner_guideline
        short_spinners: list[float] = []
        short_follow_gaps: list[float] = []
        for index, obj in enumerate(objects):
            if obj.kind != "spinner":
                continue
            timing, _ = active_timing(parsed.timing_points, obj.time_ms)
            length_beats = (obj.end_time_ms - obj.time_ms) / timing.beat_length_ms
            if length_beats < minimum_length_beats:
                short_spinners.append(obj.time_ms)
            if index + 1 < len(objects):
                follow_beats = (
                    objects[index + 1].time_ms - obj.end_time_ms
                ) / timing.beat_length_ms
                if follow_beats < minimum_follow_gap_beats:
                    short_follow_gaps.append(obj.end_time_ms)
        _issue(
            issues,
            "warning",
            "spinner_below_difficulty_guideline",
            f"{profile.label} spinners should last at least {minimum_length_beats:g} beats.",
            rule_type="guideline",
            count=len(short_spinners),
            timestamps_ms=short_spinners,
        )
        _issue(
            issues,
            "warning",
            "spinner_follow_gap_below_guideline",
            f"{profile.label} should leave at least {minimum_follow_gap_beats:g} "
            "beats after a spinner.",
            rule_type="guideline",
            count=len(short_follow_gaps),
            timestamps_ms=short_follow_gaps,
        )

    severity_counts = Counter(str(issue["severity"]) for issue in issues)
    structural_errors = sum(
        int(issue["count"])
        for issue in issues
        if issue["severity"] == "error" and issue["rule_type"] != "policy"
    )
    report: dict[str, Any] = {
        "version": 1,
        "audit": "osu-standard-ranking-criteria-subset",
        "criteria_snapshot_date": CRITERIA_SNAPSHOT_DATE,
        "sources": [GENERAL_CRITERIA_URL, STANDARD_CRITERIA_URL],
        "map": map_label or str(source),
        "mode": "osu!standard",
        "difficulty_tier": profile.key,
        "star_rating": stars,
        "generated_by_osumapper": generated,
        "rankability": (
            "not-rankable-generated-draft" if generated else "not-determined-manual-review-required"
        ),
        "automated_structural_pass": structural_errors == 0,
        "pattern_summary": _pattern_summary(objects),
        "summary": {
            "errors": severity_counts["error"],
            "warnings": severity_counts["warning"],
            "structural_error_occurrences": structural_errors,
            "manual_checks": len(_MANUAL_CHECKS),
        },
        "issues": issues,
        "manual_checks": [
            {"code": code, "status": "required", "message": message}
            for code, message in _MANUAL_CHECKS
        ],
        "disclaimer": (
            "This audit checks only a deterministic subset of the criteria. It is not a "
            "rankability certificate and does not replace Mapset Verifier, test play, modding, "
            "permission review, or the official osu! ranking process."
        ),
    }
    if output is not None:
        destination = output.expanduser().resolve()
        if destination == source:
            raise InputError("Criteria report must not overwrite the source beatmap.")
        write_json(destination, report)
        report["output"] = str(destination)
    return report
