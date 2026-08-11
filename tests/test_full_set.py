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
    GenerationRequest,
    _next_density,
    _next_full_set_controls,
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
            for profile in STANDARD_DIFFICULTIES:
                self.assertTrue((root / f"full-set.{profile.key}.criteria.json").is_file())


if __name__ == "__main__":
    unittest.main()
