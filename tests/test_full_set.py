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
    STANDARD_DIFFICULTIES,
    apply_standard_difficulty,
    standard_difficulty,
)
from osumapper.engine import _deterministic_flow
from osumapper.errors import InputError
from osumapper.pipeline import (
    CalibrationAttempt,
    GenerationRequest,
    _next_density,
    _next_full_set_controls,
    _next_plateau_threshold,
    _next_precision_flow,
    _rhythm_selection_plateaued,
    generate_package,
)

FIXTURES = Path(__file__).parent / "fixtures"


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

    def test_plateau_threshold_adds_candidates_when_stars_are_too_low(self) -> None:
        self.assertLess(_next_plateau_threshold(0.84, 4.65, 3.56), 0.84)

    def test_plateau_threshold_removes_candidates_when_stars_are_too_high(self) -> None:
        self.assertGreater(_next_plateau_threshold(0.84, 4.65, 6.0), 0.84)

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
        targets = [profile.default_stars for profile in STANDARD_DIFFICULTIES]
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            source = root / "standard.osu"
            source.write_bytes((FIXTURES / "standard.osu").read_bytes())
            (root / "audio.wav").write_bytes(b"RIFF-fixture")
            output = root / "full-set.osz"

            with (
                patch("osumapper.pipeline.generate_document", side_effect=fake_generate),
                patch("osumapper.pipeline.calculate_standard_stars", side_effect=targets),
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
            for profile in STANDARD_DIFFICULTIES:
                self.assertTrue((root / f"full-set.{profile.key}.criteria.json").is_file())


if __name__ == "__main__":
    unittest.main()
