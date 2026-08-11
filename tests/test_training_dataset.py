from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from osumapper.training.beatmaps import parse_standard_beatmap
from osumapper.training.config import DatasetPaths, QualityConfig
from osumapper.training.dataset import dataset_statistics, rate_map, scan_dataset
from osumapper.training.storage import read_parquet

FIXTURES = Path(__file__).parent / "fixtures"


def _difficulty_text(*, version: str, beatmap_id: int, beatmapset_id: int = 100) -> str:
    text = (FIXTURES / "standard.osu").read_text(encoding="utf-8")
    text = text.replace("Version:Standard", f"Version:{version}")
    text = text.replace(
        "Source:\n", f"BeatmapID:{beatmap_id}\nBeatmapSetID:{beatmapset_id}\nSource:\n"
    )
    return text


class TrainingDatasetTests(unittest.TestCase):
    def test_standard_parser_extracts_slider_timing_and_statistics(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            map_path = root / "map.osu"
            map_path.write_text(_difficulty_text(version="Hard", beatmap_id=1), encoding="utf-8")
            (root / "audio.wav").write_bytes(b"RIFF-test")

            parsed = parse_standard_beatmap(map_path, root)

        self.assertEqual(parsed.metadata["beatmap_id"], 1)
        self.assertEqual(parsed.metadata["beatmapset_id"], 100)
        self.assertEqual(parsed.song_id, "set-100")
        self.assertEqual(parsed.statistics["object_count"], 2)
        self.assertEqual(parsed.statistics["slider_count"], 1)
        self.assertAlmostEqual(parsed.hit_objects[1].slider_duration_ms or 0, 250.0)
        self.assertEqual(parsed.hit_objects[1].slider_curve_type, "L")
        self.assertEqual(parsed.hit_objects[1].slider_path, ((384, 192),))

    def test_scan_groups_difficulties_logs_skips_and_persists_rating(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            songs = root / "Songs"
            mapset = songs / "100 Fixture"
            mapset.mkdir(parents=True)
            (mapset / "audio.wav").write_bytes(b"RIFF-test")
            hard = mapset / "hard.osu"
            insane = mapset / "insane.osu"
            hard.write_text(_difficulty_text(version="Hard", beatmap_id=1), encoding="utf-8")
            insane.write_text(
                _difficulty_text(version="Insane", beatmap_id=2).replace(
                    "128,192,475", "160,192,500"
                ),
                encoding="utf-8",
            )
            (mapset / "taiko.osu").write_bytes((FIXTURES / "taiko.osu").read_bytes())
            (mapset / "malformed.osu").write_text("not an osu file", encoding="utf-8")
            duplicate_dir = songs / "duplicate"
            duplicate_dir.mkdir()
            (duplicate_dir / "hard-copy.osu").write_bytes(hard.read_bytes())
            alternate_dir = songs / "same-id-different-audio"
            alternate_dir.mkdir()
            (alternate_dir / "audio.wav").write_bytes(b"RIFF-other")
            alternate = alternate_dir / "alternate.osu"
            alternate.write_text(
                _difficulty_text(version="Alternate", beatmap_id=3).replace(
                    "128,192,475", "192,192,525"
                ),
                encoding="utf-8",
            )
            data_root = root / "training_data"

            summary = scan_dataset(
                songs,
                dataset_root=data_root,
                quality=QualityConfig(min_objects=1, min_duration_ms=0),
                progress=lambda _message: None,
            )
            rows = read_parquet(DatasetPaths.at(data_root).dataset)

            self.assertEqual(summary.indexed, 3)
            song_ids = {row["song_id"] for row in rows}
            self.assertEqual(len(song_ids), 2)
            self.assertTrue(all(song_id.startswith("set-100-audio-") for song_id in song_ids))
            self.assertEqual({row["quality_status"] for row in rows}, {"candidate_unrated"})
            skips = [
                json.loads(line)
                for line in DatasetPaths.at(data_root)
                .skipped.read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertIn("duplicate_osu", {record["reason"] for record in skips})
            self.assertIn("non_standard_mode", {record["reason"] for record in skips})
            self.assertIn("malformed_map", {record["reason"] for record in skips})

            result = rate_map(hard, "good", dataset_root=data_root)
            self.assertEqual(result["rows_updated"], 1)
            rated_rows = read_parquet(DatasetPaths.at(data_root).dataset)
            rated = next(row for row in rated_rows if row["map_path"] == str(hard.resolve()))
            self.assertEqual(rated["rating"], "good")
            self.assertEqual(rated["quality_status"], "good")

            stats = dataset_statistics(dataset_root=data_root)
            self.assertEqual(stats["songs"], 2)
            self.assertEqual(stats["mapsets"], 1)
            self.assertEqual(stats["difficulties"], 3)
            self.assertEqual(stats["total_hit_objects"], 6)

    def test_scan_rejects_timing_points_that_begin_after_all_objects(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            songs = root / "Songs"
            mapset = songs / "broken-timing"
            mapset.mkdir(parents=True)
            (mapset / "audio.wav").write_bytes(b"RIFF-test")
            map_path = mapset / "late-timing.osu"
            map_path.write_text(
                _difficulty_text(version="Broken", beatmap_id=99, beatmapset_id=999).replace(
                    "-25,500,4,2,1,100,1,0", "100000,500,4,2,1,100,1,0"
                ),
                encoding="utf-8",
            )

            summary = scan_dataset(
                songs,
                dataset_root=root / "training_data",
                quality=QualityConfig(min_objects=1, min_duration_ms=0),
                progress=lambda _message: None,
            )
            row = read_parquet(DatasetPaths.at(root / "training_data").dataset)[0]

        self.assertEqual(summary.rejected, 1)
        self.assertFalse(row["eligible"])
        self.assertEqual(row["quality_status"], "rejected")
        self.assertIn("timing_points_after_objects", row["quality_reasons"])


if __name__ == "__main__":
    unittest.main()
