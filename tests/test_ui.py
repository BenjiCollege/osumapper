from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from osumapper.ui import (
    GenerationOptions,
    build_generate_command,
    default_generation_models,
    default_training_model,
    discover_inputs,
    discover_models,
    model_bundle_paths,
    normalize_input_path,
    summarize_process_error,
    unique_output_path,
)


class UiHelperTests(unittest.TestCase):
    @staticmethod
    def _write_model(root: Path, name: str, architecture: str) -> Path:
        """Create a model folder the way training leaves one behind."""
        folder = root / "models" / "modern" / name
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "model.keras").touch()
        (folder / "config.json").write_text(
            json.dumps({"architecture": architecture}), encoding="utf-8"
        )
        return folder

    def test_untrained_defaults_name_a_model_to_train(self) -> None:
        with tempfile.TemporaryDirectory() as name, patch.dict(os.environ, {}, clear=True):
            rhythm, placement = default_generation_models(Path(name))

        self.assertEqual(rhythm.name, "rhythm-conformer-v5-curated-run1-seed-2026")
        self.assertEqual(placement.name, "placement-v4")

    def test_models_are_discovered_by_recorded_architecture_not_folder_name(self) -> None:
        # Training names folders after the dataset and seed, so matching a fixed
        # folder name silently found nothing and the interface offered a path
        # that did not exist.
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            self._write_model(
                root, "rhythm-conformer-v6-curated-657-songs-seed-2026", "conformer-v6"
            )
            self._write_model(root, "placement-v4-657-seed-2026", "placement-v4")
            with patch.dict(os.environ, {}, clear=True):
                rhythm, placement = default_generation_models(root)

        self.assertEqual(rhythm.name, "rhythm-conformer-v6-curated-657-songs-seed-2026")
        self.assertEqual(placement.name, "placement-v4-657-seed-2026")

    def test_the_measured_best_model_wins_over_the_newest(self) -> None:
        # Conformer-v7 is newer but scored lower on the held-out split, so the
        # default must follow measured quality rather than version order.
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            self._write_model(root, "rhythm-conformer-v7-streams-657-seed-2026", "conformer-v7")
            best = self._write_model(root, "rhythm-conformer-v6-curated-657", "conformer-v6")
            self._write_model(root, "placement-v3-657-seed-2026", "placement-v3")
            top = self._write_model(root, "placement-v4-657-seed-2026", "placement-v4")
            with patch.dict(os.environ, {}, clear=True):
                rhythm, placement = default_generation_models(root)
                ordered = discover_models(root, "rhythm")

        self.assertEqual(rhythm, best)
        self.assertEqual(placement, top)
        self.assertEqual([item.name for item in ordered][0], best.name)

    def test_incomplete_model_folders_are_not_offered(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            partial = root / "models" / "modern" / "rhythm-interrupted"
            partial.mkdir(parents=True)
            (partial / "model.keras").touch()  # no config.json: training died early
            with patch.dict(os.environ, {}, clear=True):
                found = discover_models(root, "rhythm")

        self.assertEqual(found, [])

    def test_model_bundle_accepts_folder_or_keras_file(self) -> None:
        folder_model, folder_config = model_bundle_paths(Path("/models/v5"))
        file_model, file_config = model_bundle_paths(Path("/models/v5/model.keras"))

        self.assertEqual(folder_model, Path("/models/v5/model.keras"))
        self.assertEqual(folder_config, Path("/models/v5/config.json"))
        self.assertEqual(file_model, Path("/models/v5/model.keras"))
        self.assertEqual(file_config, Path("/models/v5/config.json"))

    def test_v6_training_uses_a_new_output_folder(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            output = default_training_model(Path("/home/test/osumapper"))

        self.assertEqual(
            output,
            Path(
                "/home/test/osumapper/models/modern/rhythm-conformer-v6-curated-657-songs-seed-2026"
            ),
        )

    def test_star_calibration_failure_is_summarized_for_queue_row(self) -> None:
        summary = summarize_process_error(
            "error: Star calibration could not reach Expert+ 7.00★ within the requested "
            "±0.030★ after 24 attempts. Best result was 5.73★ at density 12.083 and "
            "spacing 1.000×."
        )

        self.assertEqual(summary, "Expert+: best 5.73★ / target 7.00★")

    @patch("osumapper.ui._running_under_wsl", return_value=True)
    def test_windows_download_path_is_translated_for_wsl(self, _wsl) -> None:
        translated = normalize_input_path(r'"C:\Users\bcten\Downloads\CRAZY_spotdown.org.mp3"')

        self.assertEqual(
            translated,
            Path("/mnt/c/Users/bcten/Downloads/CRAZY_spotdown.org.mp3"),
        )

    def test_folder_discovery_groups_difficulties_and_keeps_audio_only_files(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            mapped = root / "mapped"
            mapped.mkdir()
            (mapped / "a.osu").write_text("map", encoding="utf-8")
            (mapped / "b.osu").write_text("map", encoding="utf-8")
            (mapped / "audio.mp3").write_bytes(b"audio")
            audio_only = root / "audio-only"
            audio_only.mkdir()
            (audio_only / "song.flac").write_bytes(b"audio")

            grouped = discover_inputs([root])
            all_maps = discover_inputs([root], all_difficulties=True)

            self.assertEqual([path.name for path in grouped], ["song.flac", "a.osu"])
            self.assertEqual([path.name for path in all_maps], ["song.flac", "a.osu", "b.osu"])

    def test_command_and_collision_paths_include_user_options(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            source = root / "song.mp3"
            source.write_bytes(b"audio")
            requested = root / "song.osz"
            requested.write_bytes(b"existing")
            output = unique_output_path(requested)
            command = build_generate_command(
                source,
                output,
                GenerationOptions(
                    preset="default",
                    mode="standard",
                    seed=2026,
                    flow_engine="deterministic",
                    rhythm_engine="modern",
                    modern_model=root / "model",
                    placement_model=root / "placement",
                    threshold=0.8,
                    density=3.0,
                    key_count=7,
                    open_in_lazer=True,
                ),
            )

            self.assertEqual(output.name, "song-2.osz")
            self.assertIn("--modern-model", command)
            self.assertIn("--placement-model", command)
            self.assertIn("--rhythm-threshold", command)
            self.assertIn("--target-density", command)
            self.assertIn("--difficulty-tier", command)
            self.assertIn("--target-stars", command)
            self.assertIn("--open", command)
            self.assertEqual(command[command.index("--keys") + 1], "7")

    def test_full_set_command_uses_one_flag_instead_of_individual_target(self) -> None:
        command = build_generate_command(
            Path("song.mp3"),
            Path("song-full-set.osz"),
            GenerationOptions(
                mode="standard",
                rhythm_engine="modern",
                modern_model=Path("model"),
                full_set=True,
            ),
        )

        self.assertIn("--full-set", command)
        self.assertIn("--star-precision", command)
        self.assertEqual(command[command.index("--star-precision") + 1], "0.03")
        self.assertIn("--calibration-attempts", command)
        self.assertEqual(command[command.index("--calibration-attempts") + 1], "24")
        self.assertEqual(command[command.index("--star-calculator") + 1], "auto")
        self.assertNotIn("--difficulty-tier", command)
        self.assertNotIn("--target-stars", command)
        self.assertNotIn("--target-density", command)


if __name__ == "__main__":
    unittest.main()
