from __future__ import annotations

import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import numpy as np

from osumapper.config import GameMode
from osumapper.difficulty import (
    FULL_SET_DIFFICULTIES,
    STANDARD_DIFFICULTIES,
    apply_standard_difficulty,
    standard_difficulty,
)
from osumapper.engine import _deterministic_flow
from osumapper.errors import InputError
from osumapper.pipeline import (
    CalibrationAttempt,
    GenerationRequest,
    _density_strain_plateaued,
    _next_bracketed_threshold,
    _next_density,
    _next_full_set_controls,
    _next_plateau_threshold,
    _next_precision_flow,
    _rhythm_selection_plateaued,
    _threshold_selection_saturated,
    _threshold_strain_plateaued,
    generate_package,
)

FIXTURES = Path(__file__).parent / "fixtures"


class FakeStarCalculator:
    name = "rosu-pp-test"
    version = "test"

    def __init__(self, values: list[float]) -> None:
        self._values = iter(values)

    def calculate(self, _source: object) -> float:
        return next(self._values)

    def close(self) -> None:
        return None


class FullSetTests(unittest.TestCase):
    def test_density_correction_moves_toward_target_and_stays_bounded(self) -> None:
        self.assertGreater(_next_density(2.0, 5.0, 3.0), 2.0)
        self.assertLess(_next_density(2.0, 3.0, 5.0), 2.0)
        self.assertLessEqual(_next_density(19.9, 20.0, 0.1), 20.0)
        self.assertGreaterEqual(_next_density(0.1, 0.1, 20.0), 0.1)

    def test_expert_retry_increases_spacing_instead_of_only_adding_objects(self) -> None:
        profile = standard_difficulty("expert-plus")
        density, flow_scale = _next_full_set_controls(profile, 3.5, 1.0, 5.83)

        self.assertGreater(density, 3.5)
        self.assertLess(density, 3.85)
        self.assertGreater(flow_scale, 1.2)

    def test_easy_retry_reduces_density_and_spacing_together(self) -> None:
        profile = standard_difficulty("easy")
        density, flow_scale = _next_full_set_controls(profile, 0.7, 1.0, 2.82)

        self.assertLess(density, 0.5)
        self.assertLess(flow_scale, 0.8)
        self.assertGreaterEqual(flow_scale, 0.5)

    def test_precision_flow_interpolates_between_measured_star_bracket(self) -> None:
        history = [
            CalibrationAttempt(1, "fine-spacing", 1.2, 0.9, 3.20, 0.15),
            CalibrationAttempt(2, "fine-spacing", 1.2, 1.1, 3.50, 0.15),
        ]

        flow = _next_precision_flow(history, density=1.2, target_stars=3.35)

        self.assertAlmostEqual(flow, 1.0, places=6)

    def test_density_plateau_switches_to_spacing_after_three_stable_maps(self) -> None:
        history = [
            CalibrationAttempt(1, "coarse-density", 2.2, 1.0, 3.56, 1.09, 288),
            CalibrationAttempt(2, "coarse-density", 2.9, 1.0, 3.56, 1.09, 289),
            CalibrationAttempt(3, "coarse-density", 3.8, 1.0, 3.56, 1.09, 289),
        ]

        self.assertTrue(_rhythm_selection_plateaued(history))
        self.assertFalse(_rhythm_selection_plateaued(history[:2]))

    def test_density_plateau_does_not_trigger_while_object_count_is_growing(self) -> None:
        history = [
            CalibrationAttempt(1, "coarse-density", 1.0, 1.0, 2.5, 2.1, 100),
            CalibrationAttempt(2, "coarse-density", 1.5, 1.0, 3.0, 1.6, 145),
            CalibrationAttempt(3, "coarse-density", 2.0, 1.0, 3.5, 1.1, 190),
        ]

        self.assertFalse(_rhythm_selection_plateaued(history))

    def test_density_plateau_allows_small_pattern_strain_variation(self) -> None:
        history = [
            CalibrationAttempt(6, "coarse-density", 4.06, 2.0, 6.68, 0.32, 356),
            CalibrationAttempt(7, "coarse-density", 4.11, 2.0, 6.68, 0.32, 356),
            CalibrationAttempt(8, "coarse-density", 4.15, 2.0, 6.58, 0.42, 357),
        ]

        self.assertTrue(_rhythm_selection_plateaued(history))

    def test_threshold_saturation_detects_unchanged_timestamp_selection(self) -> None:
        history = [
            CalibrationAttempt(15, "coarse-threshold", 12.0, 1.0, 5.70, 1.30, 1014, 0.20),
            CalibrationAttempt(16, "coarse-threshold", 12.0, 1.0, 5.72, 1.28, 1015, 0.10),
            CalibrationAttempt(17, "coarse-threshold", 12.0, 1.0, 5.73, 1.27, 1014, 0.05),
        ]

        self.assertTrue(_threshold_selection_saturated(history))

    def test_threshold_saturation_does_not_trigger_while_objects_are_added(self) -> None:
        history = [
            CalibrationAttempt(15, "coarse-threshold", 12.0, 1.0, 5.10, 1.90, 522, 0.80),
            CalibrationAttempt(16, "coarse-threshold", 12.0, 1.0, 5.40, 1.60, 760, 0.40),
            CalibrationAttempt(17, "coarse-threshold", 12.0, 1.0, 5.70, 1.30, 1014, 0.20),
        ]

        self.assertFalse(_threshold_selection_saturated(history))

    def test_threshold_strain_plateau_preserves_budget_when_extra_objects_add_no_stars(
        self,
    ) -> None:
        history = [
            CalibrationAttempt(11, "coarse-threshold", 5.08, 1.0, 4.931, 0.969, 961, 0.65),
            CalibrationAttempt(12, "coarse-threshold", 5.08, 1.0, 4.947, 0.953, 1155, 0.55),
            CalibrationAttempt(13, "coarse-threshold", 5.08, 1.0, 5.023, 0.877, 1286, 0.46),
        ]

        self.assertTrue(_threshold_strain_plateaued(history, 5.90))

    def test_threshold_strain_plateau_keeps_searching_when_stars_improve(self) -> None:
        history = [
            CalibrationAttempt(11, "coarse-threshold", 5.08, 1.0, 4.50, 1.40, 961, 0.65),
            CalibrationAttempt(12, "coarse-threshold", 5.08, 1.0, 4.80, 1.10, 1155, 0.55),
            CalibrationAttempt(13, "coarse-threshold", 5.08, 1.0, 5.10, 0.80, 1286, 0.46),
        ]

        self.assertFalse(_threshold_strain_plateaued(history, 5.90))

    def test_threshold_strain_plateau_preserves_final_bracket_attempt(self) -> None:
        history = [
            CalibrationAttempt(15, "coarse-threshold", 20.0, 1.0, 5.299, 1.701, 765, 0.30),
            CalibrationAttempt(16, "coarse-threshold", 20.0, 1.0, 5.385, 1.615, 819, 0.22),
            CalibrationAttempt(17, "coarse-threshold", 20.0, 1.0, 5.424, 1.576, 856, 0.16),
        ]

        self.assertTrue(_threshold_strain_plateaued(history, 7.00))

    def test_max_spacing_density_plateau_preserves_threshold_budget(self) -> None:
        history = [
            CalibrationAttempt(8, "coarse-density", 9.0, 3.75, 5.049, 1.951, 259),
            CalibrationAttempt(9, "coarse-density", 12.0, 3.75, 5.046, 1.954, 268),
            CalibrationAttempt(10, "coarse-density", 15.0, 3.75, 5.169, 1.831, 270),
        ]

        self.assertTrue(_density_strain_plateaued(history, 7.00, 3.75))

    def test_expert_precision_flow_can_exceed_legacy_two_x_limit(self) -> None:
        history = [
            CalibrationAttempt(1, "fine-spacing", 12.0, 2.0, 5.10, 0.80, 1000, 0.05),
        ]

        flow = _next_precision_flow(
            history,
            density=12.0,
            target_stars=5.90,
            maximum_flow=3.25,
        )

        self.assertGreater(flow, 2.0)

    def test_precision_flow_uses_minimum_step_until_target_is_bracketed(self) -> None:
        history = [
            CalibrationAttempt(20, "fine-spacing", 20.0, 2.1, 6.95, 0.05, 856, 0.16),
        ]

        flow = _next_precision_flow(
            history,
            density=20.0,
            target_stars=7.00,
            maximum_flow=3.75,
        )

        self.assertGreaterEqual(flow, 2.18)

    def test_plateau_threshold_adds_candidates_when_stars_are_too_low(self) -> None:
        self.assertLess(_next_plateau_threshold(0.84, 4.65, 3.56), 0.84)

    def test_plateau_threshold_removes_candidates_when_stars_are_too_high(self) -> None:
        self.assertGreater(_next_plateau_threshold(0.84, 4.65, 6.0), 0.84)

    def test_threshold_search_bisects_sparse_and_dense_selections(self) -> None:
        lower, sparse, dense = _next_bracketed_threshold(0.82, 1.5, 0.9, None, None)
        self.assertLess(lower, 0.82)
        midpoint, sparse, dense = _next_bracketed_threshold(lower, 1.5, 2.1, sparse, dense)

        self.assertAlmostEqual(midpoint, (0.82 + lower) / 2.0)
        self.assertEqual(sparse, 0.82)
        self.assertEqual(dense, lower)

    def test_precision_flow_ignores_nonmonotonic_spacing_measurements(self) -> None:
        history = [
            CalibrationAttempt(1, "fine-spacing", 3.7, 1.9, 4.21, 0.44),
            CalibrationAttempt(2, "fine-spacing", 3.7, 2.0, 4.18, 0.47),
        ]

        flow = _next_precision_flow(history, density=3.7, target_stars=4.65)

        self.assertEqual(flow, 2.0)

    def test_deterministic_flow_scale_increases_mean_jump_distance(self) -> None:
        count = 24
        zeros = np.zeros(count, dtype=int)
        converted = (
            np.ones(count, dtype=int),
            np.zeros((count, 5), dtype=float),
            np.arange(count),
            np.arange(count) * 100,
            zeros.copy(),
            zeros.copy(),
            zeros.copy(),
            np.full(count, 140.0),
            zeros.copy(),
            1.0,
        )
        base, _data = _deterministic_flow(converted, 2026, 1.0)
        wider, _data = _deterministic_flow(converted, 2026, 1.25)
        base_mean = float(np.linalg.norm(np.diff(base[:, :2], axis=0), axis=1).mean())
        wider_mean = float(np.linalg.norm(np.diff(wider[:, :2], axis=0), axis=1).mean())

        self.assertGreater(wider_mean, base_mean * 1.1)

    def test_full_set_rejects_incompatible_controls_before_opening_source(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            base = {
                "source": root / "song.mp3",
                "output": root / "set.osz",
                "full_set": True,
            }
            requests = (
                GenerationRequest(**base, rhythm_engine="legacy"),
                GenerationRequest(
                    **base,
                    rhythm_engine="modern",
                    mode=GameMode.MANIA,
                ),
                GenerationRequest(
                    **base,
                    rhythm_engine="modern",
                    target_density=2.0,
                ),
                GenerationRequest(
                    **base,
                    rhythm_engine="modern",
                    difficulty_tier="hard",
                ),
            )

            for request in requests:
                with self.subTest(request=request), self.assertRaises(InputError):
                    generate_package(request, progress=lambda _message: None)

    def test_full_set_packages_exactly_six_maps_one_audio_and_reports(self) -> None:
        def fake_generate(document, _audio, _workspace, _preset, config, _progress):
            profile = next(
                item for item in STANDARD_DIFFICULTIES if item.key == config.difficulty_tier
            )
            return apply_standard_difficulty(document, profile).with_hit_objects(
                [
                    {"x": 128, "y": 192, "time": 1_000, "type": 1},
                    {"x": 384, "y": 192, "time": 32_000, "type": 1},
                ],
                preset="test",
                seed=config.seed,
            )

        audit = {
            "issues": [],
            "summary": {
                "errors": 0,
                "warnings": 0,
                "structural_error_occurrences": 0,
            },
        }
        targets = [profile.default_stars for profile in FULL_SET_DIFFICULTIES]
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            source = root / "standard.osu"
            source.write_bytes((FIXTURES / "standard.osu").read_bytes())
            (root / "audio.wav").write_bytes(b"RIFF-fixture")
            output = root / "full-set.osz"

            with (
                patch("osumapper.pipeline.generate_document", side_effect=fake_generate),
                patch(
                    "osumapper.pipeline.resolve_star_calculator",
                    return_value=FakeStarCalculator(targets),
                ),
                patch("osumapper.pipeline.audit_standard_criteria", return_value=audit),
            ):
                generate_package(
                    GenerationRequest(
                        source=source,
                        output=output,
                        mode=GameMode.STANDARD,
                        rhythm_engine="modern",
                        full_set=True,
                    ),
                    progress=lambda _message: None,
                )

            with zipfile.ZipFile(output) as archive:
                names = archive.namelist()
            self.assertEqual(sum(name.endswith(".osu") for name in names), 6)
            self.assertEqual(sum(name.endswith(".wav") for name in names), 1)
            set_report = json.loads((root / "full-set.full-set.json").read_text(encoding="utf-8"))
            self.assertTrue(set_report["complete"])
            self.assertTrue(set_report["identical_timing_sections"])
            self.assertEqual(set_report["generated_difficulties"], 6)
            self.assertEqual(set_report["mean_star_error"], 0.0)
            self.assertTrue(set_report["all_precision_targets_met"])
            self.assertEqual(set_report["requested_star_precision"], 0.03)
            self.assertEqual(set_report["placement_engine"], "pattern-planner-v1")
            # difficulty_nesting_enforced follows whichever rhythm model is
            # installed, so it is asserted in
            # test_report_describes_the_models_that_actually_produced_the_package
            # where the model is written explicitly.
            for profile in FULL_SET_DIFFICULTIES:
                self.assertTrue((root / f"full-set.{profile.key}.criteria.json").is_file())

    def test_report_describes_the_models_that_actually_produced_the_package(self) -> None:
        def fake_generate(document, _audio, _workspace, _preset, config, _progress):
            profile = next(
                item for item in STANDARD_DIFFICULTIES if item.key == config.difficulty_tier
            )
            return apply_standard_difficulty(document, profile).with_hit_objects(
                [
                    {"x": 128, "y": 192, "time": 1_000, "type": 1},
                    {"x": 384, "y": 192, "time": 32_000, "type": 1},
                ],
                preset="test",
                seed=config.seed,
            )

        audit = {
            "issues": [],
            "summary": {"errors": 0, "warnings": 0, "structural_error_occurrences": 0},
        }
        targets = [profile.default_stars for profile in FULL_SET_DIFFICULTIES]
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            source = root / "standard.osu"
            source.write_bytes((FIXTURES / "standard.osu").read_bytes())
            (root / "audio.wav").write_bytes(b"RIFF-fixture")
            rhythm_model = root / "rhythm-v6"
            rhythm_model.mkdir()
            (rhythm_model / "config.json").write_text(
                json.dumps({"training": {"architecture": "conformer-v6"}}), encoding="utf-8"
            )
            placement_model = root / "placement-v2"
            placement_model.mkdir()
            (placement_model / "config.json").write_text(
                json.dumps({"architecture": "placement-v3"}), encoding="utf-8"
            )
            output = root / "described.osz"

            with (
                patch("osumapper.pipeline.generate_document", side_effect=fake_generate),
                patch(
                    "osumapper.pipeline.resolve_star_calculator",
                    return_value=FakeStarCalculator(targets),
                ),
                patch("osumapper.pipeline.audit_standard_criteria", return_value=audit),
            ):
                generate_package(
                    GenerationRequest(
                        source=source,
                        output=output,
                        mode=GameMode.STANDARD,
                        rhythm_engine="modern",
                        flow_engine="placement",
                        modern_model=rhythm_model,
                        placement_model=placement_model,
                        full_set=True,
                    ),
                    progress=lambda _message: None,
                )

            set_report = json.loads((root / "described.full-set.json").read_text(encoding="utf-8"))
            limitations = " ".join(set_report["limitations"])

            self.assertEqual(set_report["rhythm_architecture"], "conformer-v6")
            self.assertTrue(set_report["difficulty_nesting_enforced"])
            self.assertFalse(set_report["one_pass_full_set_inference"])
            self.assertEqual(set_report["placement_engine"], "placement-v3")
            self.assertIn("monotonically nested", limitations)
            # A newer placement version must describe itself, never fall through
            # to the deterministic PatternPlanner wording.
            self.assertIn("placement-v3", limitations)
            self.assertNotIn("PatternPlanner", limitations)
            self.assertNotIn("requires V5", limitations)

    def test_single_tier_generation_calibrates_before_export(self) -> None:
        def fake_generate(document, _audio, _workspace, _preset, config, _progress):
            profile = standard_difficulty(config.difficulty_tier)
            return apply_standard_difficulty(document, profile).with_hit_objects(
                [
                    {"x": 128, "y": 192, "time": 1_000, "type": 1},
                    {"x": 384, "y": 192, "time": 32_000, "type": 1},
                ],
                preset="test",
                seed=config.seed,
            )

        audit = {
            "issues": [],
            "summary": {
                "errors": 0,
                "warnings": 0,
                "structural_error_occurrences": 0,
            },
        }
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            source = root / "standard.osu"
            source.write_bytes((FIXTURES / "standard.osu").read_bytes())
            (root / "audio.wav").write_bytes(b"RIFF-fixture")
            output = root / "single.osz"

            with (
                patch("osumapper.pipeline.generate_document", side_effect=fake_generate) as run,
                patch(
                    "osumapper.pipeline.resolve_star_calculator",
                    return_value=FakeStarCalculator([4.17, 3.90, 3.50, 3.36]),
                ),
                patch("osumapper.pipeline.audit_standard_criteria", return_value=audit),
            ):
                generate_package(
                    GenerationRequest(
                        source=source,
                        output=output,
                        mode=GameMode.STANDARD,
                        rhythm_engine="modern",
                        difficulty_tier="hard",
                        target_stars=3.35,
                    ),
                    progress=lambda _message: None,
                )

            report = json.loads((root / "single.criteria.json").read_text(encoding="utf-8"))
            self.assertTrue(output.is_file())
            self.assertEqual(run.call_count, 4)
            self.assertEqual(report["star_calibration"]["attempts"], 4)
            self.assertAlmostEqual(report["star_calibration"]["actual_stars"], 3.36)
            self.assertTrue(report["star_calibration"]["precision_target_met"])


if __name__ == "__main__":
    unittest.main()
