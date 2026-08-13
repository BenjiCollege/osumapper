from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from osumapper.beatmap import BeatmapDocument
from osumapper.difficulty import difficulty_for_stars, standard_difficulty


@dataclass(frozen=True, slots=True)
class PatternPlan:
    """A deterministic standard-mode object plan and its audit-friendly summary."""

    objects: tuple[dict[str, Any], ...]
    counts: dict[str, int]


_TIER_INDEX = {
    "easy": 0,
    "normal": 1,
    "hard": 2,
    "insane": 3,
    "expert": 4,
    "expert-plus": 5,
}

_SLIDER_RATES = {
    "easy": 0.28,
    "normal": 0.25,
    "hard": 0.21,
    "insane": 0.17,
    "expert": 0.14,
    "expert-plus": 0.11,
}

_SPINNER_TIMING = {
    # (minimum spinner length, empty follow-up gap), both measured in beats.
    "easy": (4.0, 4.0),
    "normal": (3.0, 2.0),
    "hard": (2.0, 1.0),
    "insane": (1.5, 0.5),
    "expert": (1.0, 0.25),
    "expert-plus": (1.0, 0.25),
}


def _active_value(points: list[dict[str, Any]], timestamp: float, key: str) -> float:
    active = points[0]
    for point in points:
        if timestamp >= float(point["beginTime"]):
            active = point
        else:
            break
    return float(active[key])


def _circle_margin(document: BeatmapDocument) -> int:
    try:
        circle_size = float(document.value("Difficulty", "CircleSize", "4") or 4)
    except ValueError:
        circle_size = 4.0
    return max(0, math.ceil(54.4 - 4.48 * circle_size))


def _reflected_move(
    x: float,
    y: float,
    angle: float,
    distance: float,
    margin: int,
) -> tuple[float, float, float]:
    low_x, high_x = float(margin), float(512 - margin)
    low_y, high_y = float(margin), float(384 - margin)
    next_x = x + math.cos(angle) * distance
    next_y = y + math.sin(angle) * distance
    if next_x < low_x or next_x > high_x:
        angle = math.pi - angle
        next_x = x + math.cos(angle) * distance
    if next_y < low_y or next_y > high_y:
        angle = -angle
        next_y = y + math.sin(angle) * distance
    return (
        min(high_x, max(low_x, next_x)),
        min(high_y, max(low_y, next_y)),
        angle,
    )


def _fast_groups(
    timestamps: Sequence[int],
    uninherited: list[dict[str, Any]],
) -> tuple[list[int], list[int], dict[int, int]]:
    """Mark short bursts and streams using timing-relative, capped intervals."""

    membership = [-1] * len(timestamps)
    positions = [-1] * len(timestamps)
    lengths: dict[int, int] = {}
    group = 0
    start = 0
    while start + 1 < len(timestamps):
        beat = _active_value(uninherited, timestamps[start], "tickLength")
        fast_limit = min(150.0, beat * 0.38)
        if timestamps[start + 1] - timestamps[start] > fast_limit:
            start += 1
            continue
        end = start + 1
        while end + 1 < len(timestamps):
            beat = _active_value(uninherited, timestamps[end], "tickLength")
            fast_limit = min(150.0, beat * 0.38)
            if timestamps[end + 1] - timestamps[end] > fast_limit:
                break
            end += 1
        length = end - start + 1
        if length >= 3:
            for index in range(start, end + 1):
                membership[index] = group
                positions[index] = index - start
            lengths[group] = length
            group += 1
        start = end + 1
    return membership, positions, lengths


def _spinner_indices(
    timestamps: Sequence[int],
    uninherited: list[dict[str, Any]],
    tier: str,
    seed: int,
) -> set[int]:
    if len(timestamps) < 2:
        return set()
    eligible: list[tuple[float, int]] = []
    for index in range(len(timestamps) - 1):
        gap = timestamps[index + 1] - timestamps[index]
        beat = _active_value(uninherited, timestamps[index], "tickLength")
        length_beats, follow_beats = _SPINNER_TIMING[tier]
        required_gap = beat * (length_beats + follow_beats)
        crosses_timing_change = any(
            timestamps[index] < float(point["beginTime"]) <= timestamps[index + 1]
            for point in uninherited
        )
        if gap >= max(900.0, required_gap) and not crosses_timing_change:
            # Prefer the longest musical rests; the tiny seeded term resolves ties.
            tie_break = ((index * 1103515245 + seed) & 0xFFFF) / 65536.0
            eligible.append((float(gap) + tie_break, index))
    if not eligible:
        return set()
    duration = max(0, timestamps[-1] - timestamps[0])
    target = max(1, min(3, round(duration / 150_000)))
    chosen: set[int] = set()
    for _score, index in sorted(eligible, reverse=True):
        if all(abs(index - other) >= 24 for other in chosen):
            chosen.add(index)
            if len(chosen) >= target:
                break
    return chosen


def plan_standard_patterns(
    document: BeatmapDocument,
    timestamps_ms: Sequence[int] | Any,
    *,
    difficulty_tier: str | None,
    target_stars: float | None,
    seed: int,
    flow_scale: float = 1.0,
) -> PatternPlan:
    """Plan varied, legal standard objects without changing predicted timestamps.

    PatternPlanner-v1 is deliberately deterministic. It turns rhythm timestamps into
    circles, sliders, spinners, jumps, streams, bursts, and sparse controlled stacks.
    It remains the default; the learned placement models are opt-in alternatives.
    """

    timestamps = sorted({max(0, int(round(float(value)))) for value in timestamps_ms})
    if not timestamps:
        return PatternPlan(objects=(), counts={})
    if difficulty_tier is not None:
        profile = standard_difficulty(difficulty_tier)
    elif target_stars is not None:
        profile = difficulty_for_stars(target_stars)
    else:
        profile = standard_difficulty("hard")
    stars = profile.default_stars if target_stars is None else float(target_stars)
    complexity = min(1.0, max(0.0, (stars - 1.5) / 5.5))
    tier_index = _TIER_INDEX[profile.key]
    # Calibrated Expert tiers may need wider stream and jump spacing than lower
    # difficulties. The pipeline applies tier-specific bounds up to 3.75; this
    # final defensive cap prevents unbounded or off-screen geometry.
    scale = min(3.75, max(0.45, float(flow_scale)))
    rng = random.Random(seed + tier_index * 10_007)
    timing = document.timing_points()
    uninherited = timing["uts"]
    inherited = timing["ts"]
    margin = _circle_margin(document)
    fast_membership, fast_positions, fast_lengths = _fast_groups(timestamps, uninherited)
    spinner_indices = _spinner_indices(timestamps, uninherited, profile.key, seed)

    counts = {
        "circles": 0,
        "sliders": 0,
        "spinners": 0,
        "jumps": 0,
        "bursts": sum(length in {3, 4} for length in fast_lengths.values()),
        "streams": sum(length >= 5 for length in fast_lengths.values()),
        "stacks": 0,
        "new_combos": 0,
    }
    objects: list[dict[str, Any]] = []
    phase = rng.uniform(-math.pi, math.pi)
    x = 256.0 + math.cos(phase) * 42.0
    y = 192.0 + math.sin(phase) * 32.0
    previous_group = -1
    stack_period = max(29, 67 - tier_index * 7)

    for index, timestamp in enumerate(timestamps):
        beat = _active_value(uninherited, timestamp, "tickLength")
        previous_gap = timestamp - timestamps[index - 1] if index else beat
        next_gap = timestamps[index + 1] - timestamp if index + 1 < len(timestamps) else beat
        group = fast_membership[index]
        group_started = group >= 0 and group != previous_group
        stack_here = (
            index > 0
            and group < 0
            and index % stack_period == (seed + tier_index * 3) % stack_period
            and previous_gap <= beat * (1.25 if tier_index >= 4 else 1.05)
        )

        if index:
            if stack_here:
                # Lower tiers use readable near-stacks. Exact stacks are reserved for
                # Expert tiers, where the overlap interval is intentionally controlled.
                if tier_index >= 4:
                    distance = 0.0
                else:
                    distance = 7.0
                    phase += math.pi / 2
                x, y, phase = _reflected_move(x, y, phase, distance, margin)
                counts["stacks"] += 1
            elif group >= 0:
                run_position = fast_positions[index]
                if run_position == 1:
                    phase += rng.choice((-1.0, 1.0)) * rng.uniform(0.35, 0.75)
                else:
                    phase += rng.choice((-1.0, 1.0)) * rng.uniform(0.08, 0.22)
                distance = (30.0 + 40.0 * complexity) * scale
                x, y, phase = _reflected_move(x, y, phase, distance, margin)
            else:
                gap_beats = previous_gap / max(1.0, beat)
                if gap_beats >= 0.48:
                    distance = (92.0 + 155.0 * complexity) * scale
                    phase += rng.choice((-1.0, 1.0)) * rng.uniform(1.00, 2.15)
                    counts["jumps"] += 1
                else:
                    distance = (58.0 + 82.0 * complexity) * scale
                    phase += rng.choice((-1.0, 1.0)) * rng.uniform(0.40, 1.15)
                x, y, phase = _reflected_move(x, y, phase, distance, margin)

        bar_position = int(round((timestamp - float(uninherited[0]["beginTime"])) / beat))
        new_combo = (
            index == 0
            or group_started
            or index in spinner_indices
            or (group < 0 and bar_position % 4 == 0 and index % 3 == 0)
        )
        combo_flag = 4 if new_combo else 0
        if new_combo:
            counts["new_combos"] += 1

        if index in spinner_indices and index + 1 < len(timestamps):
            _minimum_length_beats, follow_beats = _SPINNER_TIMING[profile.key]
            objects.append(
                {
                    "x": 256,
                    "y": 192,
                    "time": timestamp,
                    "type": 8 | combo_flag,
                    "hitsounds": 0,
                    "spinnerEndTime": max(
                        timestamp + 1,
                        math.floor(timestamps[index + 1] - beat * follow_beats),
                    ),
                }
            )
            counts["spinners"] += 1
            previous_group = group
            continue

        slider_hash = ((index + 1) * 2654435761 + seed * 97 + tier_index * 541) & 0xFFFF
        slider_selected = slider_hash / 65535.0 < _SLIDER_RATES[profile.key]
        slider_length_per_beat = _active_value(inherited, timestamp, "sliderLength")
        available_beats = max(0.0, (next_gap - 20.0) / max(1.0, beat))
        desired_beats = 0.75 + 0.25 * complexity
        geometry_limit_beats = 200.0 / max(1.0, slider_length_per_beat)
        duration_limit = min(available_beats, desired_beats, geometry_limit_beats)
        duration_beats = next(
            (
                candidate
                for candidate in (1.0, 0.75, 0.5, 0.375, 1.0 / 3.0, 0.25)
                if candidate <= duration_limit + 1e-9
                and not any(
                    timestamp < float(point["beginTime"]) <= timestamp + candidate * beat
                    for point in uninherited
                )
            ),
            0.0,
        )
        slider_length = duration_beats * slider_length_per_beat
        slider_allowed = (
            slider_selected
            and group < 0
            and index + 1 < len(timestamps)
            and next_gap >= max(180.0, beat * 0.45) + 20.0
            and duration_beats > 0.0
            and slider_length >= 35.0
        )
        if slider_allowed:
            slider_angle = phase + rng.choice((-1.0, 1.0)) * rng.uniform(0.55, 1.20)
            endpoint_x, endpoint_y, _ = _reflected_move(
                x,
                y,
                slider_angle,
                slider_length,
                margin,
            )
            actual_length = max(1.0, math.hypot(endpoint_x - x, endpoint_y - y))
            objects.append(
                {
                    "x": x,
                    "y": y,
                    "time": timestamp,
                    "type": 2 | combo_flag,
                    "hitsounds": 0,
                    "sliderGenerator": {
                        "type": 0,
                        "dOut": [math.cos(slider_angle), math.sin(slider_angle)],
                        "endpoint": [endpoint_x, endpoint_y],
                        "len": min(slider_length, actual_length),
                    },
                }
            )
            counts["sliders"] += 1
        else:
            objects.append(
                {
                    "x": x,
                    "y": y,
                    "time": timestamp,
                    "type": 1 | combo_flag,
                    "hitsounds": 0,
                }
            )
            counts["circles"] += 1
        previous_group = group

    return PatternPlan(objects=tuple(objects), counts=counts)
