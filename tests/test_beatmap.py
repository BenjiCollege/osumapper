from __future__ import annotations

import json
import unittest
from pathlib import Path

from osumapper.beatmap import BeatmapDocument, serialize_hit_object
from osumapper.config import GameMode

FIXTURES = Path(__file__).parent / "fixtures"
GOLDEN = Path(__file__).parent / "golden"


class BeatmapTests(unittest.TestCase):
    def test_representative_modes_parse(self) -> None:
        expected = {
            "standard.osu": GameMode.STANDARD,
            "taiko.osu": GameMode.TAIKO,
            "catch.osu": GameMode.CATCH,
            "mania.osu": GameMode.MANIA,
        }
        for filename, mode in expected.items():
            with self.subTest(filename=filename):
                document = BeatmapDocument.read(FIXTURES / filename)
                self.assertEqual(document.mode, mode)
                self.assertEqual(document.audio_filename, "audio.wav")
                self.assertTrue(document.timing_points()["uts"])

    def test_negative_timing_offset_is_preserved(self) -> None:
        document = BeatmapDocument.read(FIXTURES / "standard.osu")
        self.assertEqual(document.timing_points()["uts"][0]["beginTime"], -25)

    def test_golden_hitobject_serialization(self) -> None:
        golden = json.loads((GOLDEN / "hitobjects.json").read_text(encoding="utf-8"))
        objects = {
            "standard": [
                {"x": 128, "y": 192, "time": 500, "type": 1, "hitsounds": 0},
                {
                    "x": 256,
                    "y": 192,
                    "time": 1000,
                    "type": 2,
                    "hitsounds": 0,
                    "sliderGenerator": {"endpoint": [384, 192], "len": 140},
                },
                {"x": 256, "y": 192, "time": 2000, "type": 8, "spinnerEndTime": 3000},
            ],
            "taiko": [
                {"x": 256, "y": 192, "time": 400, "type": 1, "hitsounds": 0},
                {"x": 256, "y": 192, "time": 800, "type": 1, "hitsounds": 8},
            ],
            "catch": [
                {"x": 96, "y": 192, "time": 600, "type": 1},
                {
                    "x": 416,
                    "y": 192,
                    "time": 1200,
                    "type": 2,
                    "sliderGenerator": {"endpoint": [256, 192], "len": 150},
                },
            ],
            "mania": [
                {"x": 64, "y": 192, "time": 500, "type": 1},
                {"x": 192, "y": 192, "time": 1000, "type": 128, "holdEndTime": 1500},
            ],
        }
        for mode, values in objects.items():
            with self.subTest(mode=mode):
                self.assertEqual([serialize_hit_object(value) for value in values], golden[mode])

    def test_unicode_metadata_is_not_removed(self) -> None:
        document = BeatmapDocument.read(FIXTURES / "standard.osu")
        updated = document.set_value("Metadata", "TitleUnicode", "星の歌")
        self.assertEqual(updated.value("Metadata", "TitleUnicode"), "星の歌")

    def test_new_combo_flag_is_preserved_during_serialization(self) -> None:
        circle = serialize_hit_object({"x": 256, "y": 192, "time": 500, "type": 5})
        slider = serialize_hit_object(
            {
                "x": 256,
                "y": 192,
                "time": 1_000,
                "type": 6,
                "sliderGenerator": {"endpoint": [384, 192], "len": 128},
            }
        )

        self.assertEqual(circle.split(",")[3], "5")
        self.assertEqual(slider.split(",")[3], "6")


if __name__ == "__main__":
    unittest.main()
