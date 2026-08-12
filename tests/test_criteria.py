from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from osumapper.beatmap import BeatmapDocument
from osumapper.criteria import audit_standard_criteria

FIXTURES = Path(__file__).parent / "fixtures"


class RankingCriteriaTests(unittest.TestCase):
    def test_generated_draft_reports_policy_and_objective_violations(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            source = root / "generated.osu"
            document = BeatmapDocument.read(FIXTURES / "standard.osu")
            document = document.set_value("Metadata", "Creator", "osumapper")
            document = document.replace_section(
                "HitObjects",
                [
                    "0,0,475,1,0,0:0:0:0:",
                    "0,0,475,1,0,0:0:0:0:",
                ],
            )
            source.write_text(document.text, encoding="utf-8")
            (root / "audio.wav").write_bytes(b"RIFF-fixture")

            report = audit_standard_criteria(source)
            codes = {issue["code"] for issue in report["issues"]}

        self.assertTrue(report["generated_by_osumapper"])
        self.assertEqual(report["rankability"], "not-rankable-generated-draft")
        self.assertIn("generated_content_not_rankable", codes)
        self.assertIn("objects_on_same_tick", codes)
        self.assertIn("objects_partially_offscreen_4_3", codes)
        self.assertFalse(report["automated_structural_pass"])
        self.assertEqual(
            report["pattern_summary"]["object_types"],
            {"circles": 2, "sliders": 0, "spinners": 0},
        )

    def test_structurally_clean_map_still_requires_manual_review(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            source = root / "human.osu"
            report_path = root / "criteria.json"
            document = BeatmapDocument.read(FIXTURES / "standard.osu")
            document = document.set_value("General", "AudioFilename", "audio.mp3")
            document = document.set_value("General", "PreviewTime", "1000")
            document = document.set_value("Metadata", "Creator", "Human Mapper")
            document = document.replace_section("Events", ['0,0,"background.jpg",0,0'])
            document = document.replace_section(
                "Colours",
                ["Combo1 : 255,128,128", "Combo2 : 128,192,255"],
            )
            document = document.replace_section(
                "HitObjects",
                [
                    "128,192,475,1,0,0:0:0:0:",
                    "384,192,30475,1,0,0:0:0:0:",
                ],
            )
            source.write_text(document.text, encoding="utf-8")
            (root / "audio.mp3").write_bytes(b"fixture")
            (root / "background.jpg").write_bytes(b"fixture")

            report = audit_standard_criteria(source, output=report_path)
            stored = json.loads(report_path.read_text(encoding="utf-8"))

        self.assertTrue(report["automated_structural_pass"])
        self.assertFalse(report["generated_by_osumapper"])
        self.assertEqual(report["rankability"], "not-determined-manual-review-required")
        self.assertEqual(report["summary"]["manual_checks"], 8)
        self.assertEqual(report["pattern_summary"]["unique_position_ratio"], 1.0)
        self.assertEqual(stored["audit"], "osu-standard-ranking-criteria-subset")


if __name__ == "__main__":
    unittest.main()
