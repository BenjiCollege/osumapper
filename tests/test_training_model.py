from __future__ import annotations

import math
import tempfile
import unittest
import wave
from array import array
from pathlib import Path

import numpy as np

from osumapper.beatmap import BeatmapDocument
from osumapper.config import GameMode, GenerationConfig
from osumapper.engine import generate_document
from osumapper.errors import InputError
from osumapper.presets import get_preset
from osumapper.training.analysis import analyze_map
from osumapper.training.benchmark import build_dataset_benchmark
from osumapper.training.calibration import calibrate_threshold
from osumapper.training.config import QualityConfig, RhythmTrainingConfig
from osumapper.training.dataset import rate_map, scan_dataset
from osumapper.training.evaluation import evaluate_rhythm
from osumapper.training.features import extract_dataset_features
from osumapper.training.loader import iter_windows, summarize_loader
from osumapper.training.model import build_rhythm_model, load_rhythm_model
from osumapper.training.placement import analyze_placement
from osumapper.training.placement_learning import (
    PLACEMENT_V2_FEATURE_DIMENSION,
    PLACEMENT_V2_TARGET_DIMENSION,
    _reconstruct_v2,
    build_placement_model,
    placement_architecture,
    predict_placement,
)
from osumapper.training.splits import load_split, split_dataset
from osumapper.training.storage import read_json, write_json
from osumapper.training.trainer import historical_empty_epochs, train_rhythm

FIXTURES = Path(__file__).parent / "fixtures"


def _map_text(*, beatmap_id: int, beatmapset_id: int) -> str:
    text = (FIXTURES / "standard.osu").read_text(encoding="utf-8")
    return text.replace(
        "Source:\n",
        f"BeatmapID:{beatmap_id}\nBeatmapSetID:{beatmapset_id}\nSource:\n",
    )


def _write_tone(path: Path, frequency: float) -> None:
    sample_rate = 22_050
    samples = array(
        "h",
        (
            round(8_000 * math.sin(2 * math.pi * frequency * index / sample_rate))
            for index in range(sample_rate * 2)
        ),
    )
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(sample_rate)
        stream.writeframes(samples.tobytes())


class TrainingModelTests(unittest.TestCase):
    def test_historical_empty_epoch_detection(self) -> None:
        self.assertEqual(historical_empty_epochs({"loss": [0.4, 0.0, 0.3, 0.0]}), [2, 4])

    def test_conformer_v2_builds_and_round_trips(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            model = build_rhythm_model(
                audio_dimension=100,
                grid_dimension=6,
                sequence_length=16,
                architecture="conformer-v2",
                model_dimension=48,
                transformer_blocks=1,
                attention_heads=3,
            )
            inputs = {
                "audio_features": np.zeros((1, 16, 100), dtype=np.float32),
                "grid_features": np.zeros((1, 16, 6), dtype=np.float32),
                "difficulty": np.zeros((1, 4), dtype=np.float32),
            }
            prediction = model.predict(inputs, verbose=0)
            destination = Path(name) / "conformer.keras"
            model.save(destination)
            loaded = load_rhythm_model(destination, compile_model=False)
            reloaded = loaded.predict(inputs, verbose=0)

            self.assertEqual(model.name, "osumapper_rhythm_conformer_v2")
            self.assertEqual(prediction.shape, (1, 16, 1))
            np.testing.assert_allclose(prediction, reloaded, rtol=1e-6, atol=1e-6)

    def test_conformer_v3_builds_with_float32_probability_output(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            model = build_rhythm_model(
                audio_dimension=20,
                grid_dimension=6,
                sequence_length=16,
                architecture="conformer-v3",
                model_dimension=40,
                transformer_blocks=1,
                attention_heads=5,
                weight_decay=1e-4,
                jit_compile=False,
            )
            inputs = {
                "audio_features": np.zeros((1, 16, 20), dtype=np.float16),
                "grid_features": np.zeros((1, 16, 6), dtype=np.float16),
                "difficulty": np.zeros((1, 4), dtype=np.float16),
            }
            inputs["grid_features"][0, :4, 0] = 0.4
            prediction = model.predict(inputs, verbose=0)
            destination = Path(name) / "conformer-v3.keras"
            model.save(destination)
            loaded = load_rhythm_model(destination, compile_model=False)
            reloaded = loaded.predict(inputs, verbose=0)

            self.assertEqual(model.name, "osumapper_rhythm_conformer_v3")
            self.assertEqual(prediction.dtype, np.float32)
            self.assertEqual(prediction.shape, (1, 16, 1))
            np.testing.assert_allclose(prediction, reloaded, rtol=1e-6, atol=1e-6)

    def test_conformer_v4_uses_only_star_and_standard_difficulty_inputs(self) -> None:
        model = build_rhythm_model(
            audio_dimension=20,
            grid_dimension=6,
            difficulty_dimension=2,
            sequence_length=16,
            architecture="conformer-v4",
            model_dimension=40,
            transformer_blocks=1,
            attention_heads=5,
            jit_compile=False,
        )
        inputs = {
            "audio_features": np.zeros((1, 16, 20), dtype=np.float32),
            "grid_features": np.zeros((1, 16, 6), dtype=np.float32),
            "difficulty": np.asarray([[0.335, 0.4]], dtype=np.float32),
        }
        inputs["grid_features"][0, :4, 0] = 0.4

        prediction = model.predict(inputs, verbose=0)

        self.assertEqual(model.name, "osumapper_rhythm_conformer_v4")
        self.assertEqual(model.input["difficulty"].shape[-1], 2)
        self.assertEqual(prediction.shape, (1, 16, 1))

    def test_conformer_v5_has_six_round_trip_difficulty_heads(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            model = build_rhythm_model(
                audio_dimension=20,
                grid_dimension=6,
                difficulty_dimension=7,
                sequence_length=16,
                architecture="conformer-v5",
                model_dimension=48,
                transformer_blocks=1,
                attention_heads=6,
                jit_compile=False,
            )
            inputs = {
                "audio_features": np.zeros((1, 16, 20), dtype=np.float32),
                "grid_features": np.zeros((1, 16, 6), dtype=np.float32),
                "difficulty": np.asarray([[0.7, 0, 0, 0, 0, 0, 1]], dtype=np.float32),
            }
            inputs["grid_features"][0, :4, 0] = 0.4
            prediction = model.predict(inputs, verbose=0)
            all_heads = model.get_layer("difficulty_head_probabilities").output
            import keras

            head_model = keras.Model(model.input, all_heads)
            tier_predictions = head_model.predict(inputs, verbose=0)
            destination = Path(name) / "conformer-v5.keras"
            model.save(destination)
            reloaded = load_rhythm_model(destination, compile_model=False).predict(
                inputs, verbose=0
            )

            self.assertEqual(model.name, "osumapper_rhythm_conformer_v5")
            self.assertEqual(prediction.shape, (1, 16, 1))
            self.assertEqual(tier_predictions.shape, (1, 16, 6))
            np.testing.assert_allclose(prediction, tier_predictions[:, :, 5:6], atol=1e-6)
            np.testing.assert_allclose(prediction, reloaded, rtol=1e-6, atol=1e-6)

    def test_conformer_v6_has_nested_round_trip_difficulty_heads(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            model = build_rhythm_model(
                audio_dimension=20,
                grid_dimension=6,
                difficulty_dimension=7,
                sequence_length=16,
                architecture="conformer-v6",
                model_dimension=48,
                transformer_blocks=1,
                attention_heads=6,
                jit_compile=False,
            )
            inputs = {
                "audio_features": np.zeros((6, 16, 20), dtype=np.float32),
                "grid_features": np.zeros((6, 16, 6), dtype=np.float32),
                "difficulty": np.asarray(
                    [
                        [0.150, 1, 0, 0, 0, 0, 0],
                        [0.235, 0, 1, 0, 0, 0, 0],
                        [0.335, 0, 0, 1, 0, 0, 0],
                        [0.465, 0, 0, 0, 1, 0, 0],
                        [0.590, 0, 0, 0, 0, 1, 0],
                        [0.700, 0, 0, 0, 0, 0, 1],
                    ],
                    dtype=np.float32,
                ),
            }
            inputs["grid_features"][:, :4, 0] = 0.4
            prediction = model.predict(inputs, verbose=0)
            import keras

            head_model = keras.Model(
                model.input,
                model.get_layer("difficulty_head_probabilities").output,
            )
            tier_predictions = head_model.predict(inputs, verbose=0)
            destination = Path(name) / "conformer-v6.keras"
            model.save(destination)
            reloaded = load_rhythm_model(destination, compile_model=False).predict(
                inputs, verbose=0
            )

            self.assertEqual(model.name, "osumapper_rhythm_conformer_v6")
            self.assertEqual(prediction.shape, (6, 16, 1))
            self.assertEqual(tier_predictions.shape, (6, 16, 6))
            self.assertTrue(np.all(np.diff(tier_predictions, axis=-1) >= -1e-7))
            self.assertTrue(np.all(np.diff(prediction[:, :, 0], axis=0) >= -1e-7))
            np.testing.assert_allclose(
                prediction[5:6],
                tier_predictions[5:6, :, 5:6],
                rtol=1e-3,
                atol=1e-3,
            )
            np.testing.assert_allclose(prediction, reloaded, rtol=2e-3, atol=1e-3)

    def test_placement_analyzer_reports_safe_fixture(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            _write_tone(root / "audio.wav", 440.0)
            map_path = root / "map.osu"
            map_path.write_text(_map_text(beatmap_id=1, beatmapset_id=2), encoding="utf-8")
            report = analyze_placement(map_path, output=root / "placement.json")

            self.assertEqual(report["metrics"]["offscreen_objects"], 0)
            self.assertEqual(report["quality_score"], 100)
            self.assertTrue(Path(report["output"]).is_file())

    def test_placement_v1_builds_and_round_trips(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            model = build_placement_model(
                sequence_length=16,
                learning_rate=1e-3,
                jit_compile=False,
                architecture="placement-v1",
            )
            inputs = np.zeros((1, 16, 11), dtype=np.float32)
            inputs[0, :4, 6] = 0.2
            targets = np.zeros((1, 16, 8), dtype=np.float32)
            targets[0, :4, 2] = 1.0
            targets[0, :4, 3] = 1.0
            weights = np.zeros((1, 16), dtype=np.float32)
            weights[0, :4] = 1.0
            loss = model.train_on_batch(inputs, targets, sample_weight=weights)
            prediction = model.predict(inputs, verbose=0)
            destination = Path(name) / "placement.keras"
            model.save(destination)
            write_json(
                Path(name) / "config.json",
                {"architecture": "placement-v1", "training": {"sequence_length": 16}},
            )
            loaded = load_rhythm_model(destination, compile_model=False)
            reloaded = loaded.predict(inputs, verbose=0)
            placed = predict_placement(
                BeatmapDocument.read(FIXTURES / "standard.osu"),
                np.asarray([500, 1_000, 1_500, 2_500]),
                model_root=destination,
                target_density=2.0,
                seed=2026,
            )

            self.assertEqual(model.name, "osumapper_placement_v1")
            self.assertEqual(placement_architecture(destination), "placement-v1")
            self.assertTrue(np.isfinite(loss))
            self.assertEqual(prediction.shape, (1, 16, 8))
            self.assertEqual(len(placed), 4)
            self.assertTrue(all(0 <= obj["x"] <= 512 and 0 <= obj["y"] <= 384 for obj in placed))
            np.testing.assert_allclose(prediction, reloaded, rtol=1e-6, atol=1e-6)

    def test_placement_v2_is_the_default_and_reconstructs_legal_objects(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            model = build_placement_model(
                sequence_length=16,
                learning_rate=1e-3,
                jit_compile=False,
                model_dimension=24,
                blocks=1,
                attention_heads=4,
            )
            inputs = np.zeros((1, 16, PLACEMENT_V2_FEATURE_DIMENSION), dtype=np.float32)
            inputs[0, :4, 11] = 1.0
            targets = np.zeros((1, 16, PLACEMENT_V2_TARGET_DIMENSION), dtype=np.float32)
            targets[0, :4, 2] = 1.0
            targets[0, :4, 4] = 1.0
            targets[0, :4, 8] = 0.5
            targets[0, :4, 9] = 0.5
            weights = np.zeros((1, 16), dtype=np.float32)
            weights[0, :4] = 1.0
            loss = model.train_on_batch(inputs, targets, sample_weight=weights)
            prediction = model.predict(inputs, verbose=0)
            destination = Path(name) / "placement.keras"
            model.save(destination)
            write_json(
                Path(name) / "config.json",
                {"architecture": "placement-v2", "training": {"sequence_length": 16}},
            )
            timestamps = np.asarray([500, 1_000, 1_500, 2_500])
            document = BeatmapDocument.read(FIXTURES / "standard.osu")
            placed = predict_placement(
                document,
                timestamps,
                model_root=destination,
                target_density=2.0,
                seed=2026,
                difficulty_tier="expert",
                target_stars=5.9,
            )

            self.assertEqual(model.name, "osumapper_placement_v2")
            self.assertEqual(placement_architecture(destination), "placement-v2")
            self.assertTrue(np.isfinite(loss))
            self.assertEqual(prediction.shape, (1, 16, PLACEMENT_V2_TARGET_DIMENSION))
            self.assertEqual(len(placed), 4)
            self.assertTrue(all(0 <= obj["x"] <= 512 and 0 <= obj["y"] <= 384 for obj in placed))
            self.assertEqual(
                [obj["time"] for obj in placed], [int(value) for value in timestamps]
            )

    def test_placement_v2_spacing_follows_the_calibrated_flow_scale(self) -> None:
        # Full-set star calibration tunes spacing while the rhythm is locked, so a
        # scale change must move the realised distances by the same factor.
        count = 8
        predicted = np.zeros((count, PLACEMENT_V2_TARGET_DIMENSION), dtype=np.float32)
        predicted[:, 0] = 0.15
        predicted[:, 2] = 1.0
        predicted[:, 3] = 1.0
        predicted[:, 8] = 0.5
        predicted[:, 9] = 0.5
        timestamps = [500.0 + index * 300.0 for index in range(count)]

        def spacing(scale: float) -> float:
            objects = _reconstruct_v2(
                timestamps,
                predicted,
                beat_lengths=[600.0] * count,
                slider_lengths=[100.0] * count,
                margin=36.0,
                flow_scale=scale,
                seed=2026,
            )
            return sum(
                math.hypot(
                    float(objects[index]["x"]) - float(objects[index - 1]["x"]),
                    float(objects[index]["y"]) - float(objects[index - 1]["y"]),
                )
                for index in range(1, count)
            )

        single = spacing(1.0)
        doubled = spacing(2.0)

        self.assertAlmostEqual(single, 0.15 * 640.0 * (count - 1), delta=1.0)
        self.assertAlmostEqual(doubled / single, 2.0, delta=0.05)

    def test_loader_training_and_held_out_evaluation_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            songs = root / "Songs"
            data_root = root / "training_data"
            for index, mapset_id in enumerate((201, 202, 203)):
                folder = songs / str(mapset_id)
                folder.mkdir(parents=True)
                _write_tone(folder / "audio.wav", 330.0 + index * 55)
                map_path = folder / "map.osu"
                map_path.write_text(
                    _map_text(beatmap_id=mapset_id * 10, beatmapset_id=mapset_id),
                    encoding="utf-8",
                )
                rate_map(map_path, "good", dataset_root=data_root)
            scan_dataset(
                songs,
                dataset_root=data_root,
                quality=QualityConfig(min_objects=1, min_duration_ms=0),
                progress=lambda _message: None,
            )
            split_dataset(dataset_root=data_root, seed=2026)
            benchmark = build_dataset_benchmark(dataset_root=data_root)
            extract_dataset_features(dataset_root=data_root, progress=lambda _message: None)
            train_rows = load_split("train", dataset_root=data_root)
            examples = list(iter_windows(train_rows, dataset_root=data_root))
            summary = summarize_loader(train_rows, dataset_root=data_root)

            self.assertTrue(examples)
            self.assertGreater(summary.audio_dimension, 100)
            self.assertEqual(summary.grid_dimension, 6)
            self.assertGreater(summary.positives, 0)
            self.assertEqual(benchmark["status"], "curated-prototype")
            self.assertTrue(benchmark["gates"]["no_song_leakage"])

            model_root = root / "modern-model"
            result = train_rhythm(
                dataset_root=data_root,
                output=model_root,
                config=RhythmTrainingConfig(
                    epochs=3,
                    batch_size=1,
                    learning_rate=1e-3,
                    seed=2026,
                    device="cpu",
                    sequence_length=32,
                ),
            )
            resumed = train_rhythm(
                dataset_root=data_root,
                output=model_root,
                config=RhythmTrainingConfig(
                    epochs=4,
                    batch_size=1,
                    learning_rate=1e-3,
                    seed=2026,
                    device="cpu",
                    sequence_length=32,
                ),
                resume=True,
            )
            calibration = calibrate_threshold(
                dataset_root=data_root, model_root=model_root, progress=None
            )
            report = evaluate_rhythm(dataset_root=data_root, model_root=model_root)
            split_manifest_path = data_root / "splits" / "manifest.json"
            split_manifest = read_json(split_manifest_path)
            write_json(split_manifest_path, {**split_manifest, "seed": 7})
            with self.assertRaisesRegex(InputError, "song leakage"):
                evaluate_rhythm(dataset_root=data_root, model_root=model_root)
            write_json(split_manifest_path, split_manifest)
            analysis = analyze_map(
                map_path,
                dataset_root=data_root,
                model_root=model_root,
                output=root / "analysis.json",
                timeline_format="both",
            )
            generation_workspace = root / "generation-workspace"
            generation_workspace.mkdir()
            source_document = BeatmapDocument.read(map_path)
            generated = generate_document(
                source_document,
                map_path.parent / "audio.wav",
                generation_workspace,
                get_preset("default", GameMode.STANDARD),
                GenerationConfig(
                    preset="default",
                    mode=GameMode.STANDARD,
                    seed=2026,
                    flow_engine="deterministic",
                    rhythm_engine="modern",
                    modern_model=model_root,
                    rhythm_threshold=1e-6,
                    target_density=2.0,
                    difficulty_tier="easy",
                    target_stars=1.0,
                ),
                progress=lambda _message: None,
            )

            self.assertTrue(result.model.is_file())
            self.assertTrue(result.best_checkpoint.is_file())
            self.assertEqual(resumed.epochs_completed, 4)
            self.assertTrue((model_root / "last.keras").is_file())
            self.assertTrue((model_root / "config.json").is_file())
            self.assertTrue((model_root / "training_history.json").is_file())
            self.assertTrue((model_root / "training_state.json").is_file())
            self.assertTrue((model_root / "dataset_manifest.json").is_file())
            history = read_json(model_root / "training_history.json")
            self.assertEqual(len(history["loss"]), 4)
            self.assertTrue(all(float(value) > 0.0 for value in history["loss"]))
            self.assertTrue(all(float(value) > 0.0 for value in history["val_loss"]))
            self.assertEqual(report["split"], "test")
            self.assertEqual(calibration["source_split"], "validation")
            self.assertEqual(report["threshold"], calibration["threshold"])
            self.assertEqual(report["test_maps"], 1)
            self.assertGreater(report["candidate_positions"], 0)
            self.assertEqual(analysis["human_objects"], 2)
            self.assertIn("precision", analysis)
            self.assertTrue(Path(analysis["json_output"]).is_file())
            self.assertTrue(Path(analysis["csv_output"]).is_file())
            self.assertTrue(generated.sections()["HitObjects"])
            self.assertIn("modern-rhythm", generated.version_name)
            self.assertEqual(BeatmapDocument.read(map_path).version_name, "Standard")


if __name__ == "__main__":
    unittest.main()
