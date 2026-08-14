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
        values = (0.0, 2.0, 2.7, 4.0, 5.3, 6.5, 8.0, 9.0)

        self.assertEqual(
            tuple(difficulty_for_stars(value).key for value in values),
            STANDARD_DIFFICULTY_KEYS,
        )
        self.assertEqual(difficulty_for_stars(1.99).key, "easy")
        self.assertEqual(difficulty_for_stars(6.49).key, "expert")
        self.assertEqual(difficulty_for_stars(7.99).key, "expert-plus")
        self.assertEqual(difficulty_for_stars(8.99).key, "master")
        self.assertEqual(difficulty_for_stars(12.0).key, "legendary")

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


class TierFeatureCompatibilityTests(unittest.TestCase):
    def test_each_architecture_keeps_the_tier_width_it_was_trained_with(self) -> None:
        # Adding Master and Legendary must not widen the difficulty input of
        # models already trained on six tiers: that silently breaks every saved
        # V4/V5/V6 model, and it only surfaces once training or generation runs.
        from osumapper.difficulty import (
            STAR_DIFFICULTY_FEATURES,
            V5_DIFFICULTY_FEATURES,
            V7_DIFFICULTY_FEATURES,
        )
        from osumapper.training.loader import difficulty_feature_array

        for features, width in (
            (STAR_DIFFICULTY_FEATURES, 2),
            (V5_DIFFICULTY_FEATURES, 7),
            (V7_DIFFICULTY_FEATURES, 9),
        ):
            for tier in ("easy", "expert-plus", "master", "legendary"):
                with self.subTest(features=len(features), tier=tier):
                    row = {"star_rating": 8.5, "difficulty_tier": tier}
                    self.assertEqual(difficulty_feature_array(row, features).shape[0], width)

    def test_new_tiers_fold_onto_the_closest_tier_older_models_know(self) -> None:
        from osumapper.difficulty import V5_DIFFICULTY_FEATURES, V7_DIFFICULTY_FEATURES
        from osumapper.training.loader import difficulty_feature_array

        expert_plus = difficulty_feature_array(
            {"star_rating": 7.0, "difficulty_tier": "expert-plus"}, V5_DIFFICULTY_FEATURES
        )
        legendary = difficulty_feature_array(
            {"star_rating": 9.5, "difficulty_tier": "legendary"}, V5_DIFFICULTY_FEATURES
        )
        self.assertEqual(legendary[1:].argmax(), expert_plus[1:].argmax())

        # V7 gives them their own heads instead.
        v7_expert_plus = difficulty_feature_array(
            {"star_rating": 7.0, "difficulty_tier": "expert-plus"}, V7_DIFFICULTY_FEATURES
        )
        v7_legendary = difficulty_feature_array(
            {"star_rating": 9.5, "difficulty_tier": "legendary"}, V7_DIFFICULTY_FEATURES
        )
        self.assertNotEqual(v7_legendary[1:].argmax(), v7_expert_plus[1:].argmax())
