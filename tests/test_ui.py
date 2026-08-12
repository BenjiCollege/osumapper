from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from osumapper.ui import (
    GenerationOptions,
    build_generate_command,
    discover_inputs,
    normalize_input_path,
    unique_output_path,
)


class UiHelperTests(unittest.TestCase):
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
        self.assertEqual(command[command.index("--calibration-attempts") + 1], "16")
        self.assertNotIn("--difficulty-tier", command)
        self.assertNotIn("--target-stars", command)
        self.assertNotIn("--target-density", command)


if __name__ == "__main__":
    unittest.main()
