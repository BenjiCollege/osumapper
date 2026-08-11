from __future__ import annotations

import unittest
from pathlib import Path

from osumapper.beatmap import BeatmapDocument
from osumapper.difficulty import (
    STANDARD_DIFFICULTY_KEYS,
    apply_standard_difficulty,
    calculate_standard_stars,
    difficulty_for_stars,
    resolve_standard_difficulty,
    standard_difficulty,
)
from osumapper.errors import InputError

FIXTURES = Path(__file__).parent / "fixtures"


class StandardDifficultyTests(unittest.TestCase):
    def test_exact_star_boundaries_select_the_requested_tiers(self) -> None:
        values = (0.0, 2.0, 2.7, 4.0, 5.3, 6.5)

        self.assertEqual(
            tuple(difficulty_for_stars(value).key for value in values),
            STANDARD_DIFFICULTY_KEYS,
        )
        self.assertEqual(difficulty_for_stars(1.99).key, "easy")
        self.assertEqual(difficulty_for_stars(6.49).key, "expert")
        self.assertEqual(difficulty_for_stars(12.0).key, "expert-plus")

    def test_target_must_be_inside_selected_band(self) -> None:
        profile, stars = resolve_standard_difficulty("hard", 3.5) or (None, None)

        self.assertEqual(profile.key, "hard")
        self.assertEqual(stars, 3.5)
        with self.assertRaisesRegex(InputError, "Hard requires"):
            resolve_standard_difficulty("hard", 4.0)

    def test_rosu_calculates_real_standard_stars_and_profile_sets_mode(self) -> None:
        source = FIXTURES / "standard.osu"
        stars = calculate_standard_stars(source)
        document = apply_standard_difficulty(
            BeatmapDocument.read(source), standard_difficulty("expert+")
        )

        self.assertAlmostEqual(stars, 0.2457381539, places=6)
        self.assertEqual(difficulty_for_stars(stars).key, "easy")
        self.assertEqual(document.mode.value, 0)
        self.assertEqual(document.value("Difficulty", "ApproachRate"), "10")


if __name__ == "__main__":
    unittest.main()
