from __future__ import annotations

import json
import math
import tempfile
import unittest
import wave
from array import array
from pathlib import Path

import numpy as np

from osumapper.training.beatmaps import parse_standard_beatmap
from osumapper.training.config import (
    AudioFeatureConfig,
    DatasetPaths,
    GridConfig,
    QualityConfig,
    prediction_threshold,
)
from osumapper.training.dataset import rate_map, scan_dataset
from osumapper.training.features import (
    extract_audio_features,
    load_features,
)
from osumapper.training.grid import align_features_to_grid, create_timing_grid
from osumapper.training.splits import load_split, split_dataset

FIXTURES = Path(__file__).parent / "fixtures"


def _map_text(*, beatmap_id: int, beatmapset_id: int, version: str) -> str:
    text = (FIXTURES / "standard.osu").read_text(encoding="utf-8")
    text = text.replace("Version:Standard", f"Version:{version}")
    return text.replace(
        "Source:\n",
        f"BeatmapID:{beatmap_id}\nBeatmapSetID:{beatmapset_id}\nSource:\n",
    )


def _write_tone(path: Path, duration_seconds: float = 2.0, sample_rate: int = 22_050) -> None:
    samples = array(
        "h",
        (
            round(10_000 * math.sin(2 * math.pi * 440 * index / sample_rate))
            for index in range(round(sample_rate * duration_seconds))
        ),
    )
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(sample_rate)
        stream.writeframes(samples.tobytes())


class TrainingPreprocessingTests(unittest.TestCase):
    def test_density_aware_threshold_prefers_matching_validation_band(self) -> None:
        config = {
            "training": {"prediction_threshold": 0.5},
            "calibration": {
                "threshold": 0.8,
                "density_bands": [
                    {"minimum_density": 0.0, "maximum_density": 1.5, "threshold": 0.7},
                    {"minimum_density": 1.5, "maximum_density": 3.0, "threshold": 0.9},
                ],
                "difficulty_tiers": [
                    {"name": "hard", "threshold": 0.85},
                ],
            },
        }

        self.assertEqual(prediction_threshold(config, target_density=2.0), 0.9)
        self.assertEqual(prediction_threshold(config, target_density=4.0), 0.8)
        self.assertEqual(prediction_threshold(config, 0.6, target_density=2.0), 0.6)
        self.assertEqual(
            prediction_threshold(config, target_density=2.0, difficulty_tier="hard"),
            0.85,
        )

    def test_song_level_split_is_deterministic_and_has_no_mapset_leakage(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            songs = root / "Songs"
            data_root = root / "training_data"
            all_maps: list[Path] = []
            for mapset_id in range(100, 106):
                folder = songs / str(mapset_id)
                folder.mkdir(parents=True)
                (folder / "audio.wav").write_bytes(b"RIFF-test")
                difficulties = ("Hard", "Insane") if mapset_id == 100 else ("Hard",)
                for difficulty_index, version in enumerate(difficulties):
                    path = folder / f"{version}.osu"
                    path.write_text(
                        _map_text(
                            beatmap_id=mapset_id * 10 + difficulty_index,
                            beatmapset_id=mapset_id,
                            version=version,
                        ),
                        encoding="utf-8",
                    )
                    all_maps.append(path)
                    rate_map(path, "good", dataset_root=data_root)
            scan_dataset(
                songs,
                dataset_root=data_root,
                quality=QualityConfig(min_objects=1, min_duration_ms=0),
                progress=lambda _message: None,
            )

            first = split_dataset(dataset_root=data_root, seed=2026)
            manifest_path = DatasetPaths.at(data_root).splits / "manifest.json"
            first_assignments = json.loads(manifest_path.read_text(encoding="utf-8"))["assignments"]
            second = split_dataset(dataset_root=data_root, seed=2026)
            second_assignments = json.loads(manifest_path.read_text(encoding="utf-8"))[
                "assignments"
            ]

            self.assertEqual(first_assignments, second_assignments)
            self.assertEqual(first.songs, second.songs)
            split_song_ids = {
                split: {row["song_id"] for row in load_split(split, dataset_root=data_root)}
                for split in ("train", "validation", "test")
            }
            self.assertTrue(split_song_ids["train"].isdisjoint(split_song_ids["validation"]))
            self.assertTrue(split_song_ids["train"].isdisjoint(split_song_ids["test"]))
            self.assertTrue(split_song_ids["validation"].isdisjoint(split_song_ids["test"]))
            containing_set_100 = [
                split
                for split, songs_in_split in split_song_ids.items()
                if "set-100" in songs_in_split
            ]
            self.assertEqual(len(containing_set_100), 1)
            rows_for_set_100 = load_split(containing_set_100[0], dataset_root=data_root)
            self.assertEqual(sum(row["song_id"] == "set-100" for row in rows_for_set_100), 2)

    def test_audio_feature_cache_and_grid_alignment_are_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            audio = root / "audio.wav"
            _write_tone(audio)
            map_path = root / "map.osu"
            map_path.write_text(
                _map_text(beatmap_id=1, beatmapset_id=100, version="Hard"),
                encoding="utf-8",
            )
            parsed = parse_standard_beatmap(map_path, root)
            detail = parsed.detail_dict()
            feature_path = root / "features.npz"
            config = AudioFeatureConfig(n_fft=512, hop_length=128, n_mels=16)

            first = extract_audio_features(audio, feature_path, config=config)
            first_values = load_features(feature_path)
            second = extract_audio_features(audio, feature_path, config=config)
            second_values = load_features(feature_path)
            grid = create_timing_grid(detail, config=GridConfig(hit_tolerance_ms=32))
            aligned = align_features_to_grid(feature_path, grid)
            contextual = align_features_to_grid(feature_path, grid, context_radius=2)

            self.assertFalse(first.cached)
            self.assertTrue(second.cached)
            self.assertEqual(first.frames, second.frames)
            np.testing.assert_array_equal(first_values["mel"], second_values["mel"])
            np.testing.assert_array_equal(
                first_values["frame_times_ms"], second_values["frame_times_ms"]
            )
            self.assertTrue(np.all(np.diff(first_values["frame_times_ms"]) > 0))
            self.assertEqual(grid.matched_objects, 2)
            self.assertEqual(sum(candidate.label for candidate in grid.candidates), 2)
            self.assertEqual(aligned.shape, (len(grid.candidates), 20))
            self.assertEqual(contextual.shape, (len(grid.candidates), 100))
            np.testing.assert_array_equal(contextual[:, 40:60], aligned)
            self.assertTrue(np.isfinite(aligned).all())


if __name__ == "__main__":
    unittest.main()
