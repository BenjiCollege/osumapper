from __future__ import annotations

import math
import unittest
from pathlib import Path

from osumapper.beatmap import BeatmapDocument
from osumapper.difficulty import apply_standard_difficulty, standard_difficulty
from osumapper.patterns import plan_standard_patterns

FIXTURES = Path(__file__).parent / "fixtures"


def _representative_timestamps() -> list[int]:
    timestamps: list[int] = []
    current = 1_000
    for index in range(150):
        timestamps.append(current)
        if index in {18, 73}:
            # Five-object stream including the current object.
            timestamps.extend(current + offset for offset in (100, 200, 300, 400))
            current += 400
        elif index in {42, 112}:
            # Three-object burst including the current object.
            timestamps.extend(current + offset for offset in (110, 220))
            current += 220
        if index in {35, 95, 130}:
            current += 2_500
        else:
            current += 500
    return sorted(set(timestamps))


class PatternPlannerTests(unittest.TestCase):
    def setUp(self) -> None:
        source = BeatmapDocument.read(FIXTURES / "standard.osu")
        self.document = apply_standard_difficulty(source, standard_difficulty("expert-plus"))
        self.timestamps = _representative_timestamps()

    def test_expert_plan_contains_varied_legal_object_types_and_patterns(self) -> None:
        plan = plan_standard_patterns(
            self.document,
            self.timestamps,
            difficulty_tier="expert-plus",
            target_stars=7.0,
            seed=2026,
        )

        self.assertGreater(plan.counts["circles"], 0)
        self.assertGreater(plan.counts["sliders"], 0)
        self.assertGreater(plan.counts["spinners"], 0)
        self.assertGreater(plan.counts["jumps"], 0)
        self.assertGreater(plan.counts["bursts"], 0)
        self.assertGreater(plan.counts["streams"], 0)
        self.assertGreater(plan.counts["stacks"], 0)
        self.assertTrue(any(int(obj["type"]) & 2 for obj in plan.objects))
        self.assertTrue(any(int(obj["type"]) & 8 for obj in plan.objects))

        margin = math.ceil(54.4 - 4.48 * 4.0)
        for obj in plan.objects:
            self.assertGreaterEqual(float(obj["x"]), margin)
            self.assertLessEqual(float(obj["x"]), 512 - margin)
            self.assertGreaterEqual(float(obj["y"]), margin)
            self.assertLessEqual(float(obj["y"]), 384 - margin)
            if int(obj["type"]) & 2:
                endpoint = obj["sliderGenerator"]["endpoint"]
                self.assertGreaterEqual(float(endpoint[0]), margin)
                self.assertLessEqual(float(endpoint[0]), 512 - margin)
                self.assertGreaterEqual(float(endpoint[1]), margin)
                self.assertLessEqual(float(endpoint[1]), 384 - margin)

    def test_plan_is_reproducible_and_seed_changes_geometry(self) -> None:
        first = plan_standard_patterns(
            self.document,
            self.timestamps,
            difficulty_tier="expert-plus",
            target_stars=7.0,
            seed=2026,
        )
        repeated = plan_standard_patterns(
            self.document,
            self.timestamps,
            difficulty_tier="expert-plus",
            target_stars=7.0,
            seed=2026,
        )
        changed = plan_standard_patterns(
            self.document,
            self.timestamps,
            difficulty_tier="expert-plus",
            target_stars=7.0,
            seed=2027,
        )

        self.assertEqual(first, repeated)
        self.assertNotEqual(first.objects, changed.objects)

    def test_easy_uses_readable_near_stacks_instead_of_exact_overlaps(self) -> None:
        document = apply_standard_difficulty(self.document, standard_difficulty("easy"))
        plan = plan_standard_patterns(
            document,
            self.timestamps,
            difficulty_tier="easy",
            target_stars=1.5,
            seed=2026,
        )
        circle_positions = [
            (round(float(obj["x"]), 4), round(float(obj["y"]), 4))
            for obj in plan.objects
            if int(obj["type"]) & 1
        ]

        self.assertGreater(plan.counts["stacks"], 0)
        self.assertTrue(
            all(a != b for a, b in zip(circle_positions, circle_positions[1:], strict=False))
        )

    def test_long_objects_keep_supported_snaps_and_required_follow_gaps(self) -> None:
        document = apply_standard_difficulty(self.document, standard_difficulty("easy"))
        timestamps: list[int] = []
        current = 975
        for index in range(100):
            timestamps.append(current)
            current += 5_000 if index == 40 else 500
        plan = plan_standard_patterns(
            document,
            timestamps,
            difficulty_tier="easy",
            target_stars=1.5,
            seed=2026,
        )

        self.assertGreater(plan.counts["sliders"], 0)
        self.assertEqual(plan.counts["spinners"], 1)
        timing = document.timing_points()
        for index, obj in enumerate(plan.objects[:-1]):
            next_time = float(plan.objects[index + 1]["time"])
            if int(obj["type"]) & 8:
                follow_gap = next_time - float(obj["spinnerEndTime"])
                self.assertGreaterEqual(follow_gap, 4 * 500)
            if int(obj["type"]) & 2:
                timestamp = float(obj["time"])
                pixels_per_beat = next(
                    float(point["sliderLength"])
                    for point in reversed(timing["ts"])
                    if timestamp >= float(point["beginTime"])
                )
                duration = float(obj["sliderGenerator"]["len"]) / pixels_per_beat * 500
                end_time = timestamp + duration
                self.assertGreaterEqual(next_time - end_time, 20 - 1e-6)
                beat_position = (end_time + 25) / 500
                snap_error = min(
                    abs(beat_position * divisor - round(beat_position * divisor)) * 500 / divisor
                    for divisor in (1, 2, 3, 4, 6, 8, 12, 16)
                )
                self.assertLess(snap_error, 0.01)


if __name__ == "__main__":
    unittest.main()
