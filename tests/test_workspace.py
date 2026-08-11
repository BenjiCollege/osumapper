from __future__ import annotations

import tempfile
import unittest
import zipfile
from pathlib import Path

from osumapper.errors import InputError, PackageSafetyError
from osumapper.package import validate_osz, write_osz
from osumapper.workspace import prepare_source, safe_extract_osz

FIXTURES = Path(__file__).parent / "fixtures"


class WorkspaceTests(unittest.TestCase):
    def test_rejects_zip_slip(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            archive = root / "bad.osz"
            with zipfile.ZipFile(archive, "w") as package:
                package.writestr("../escape.osu", "bad")
            with self.assertRaises(PackageSafetyError):
                safe_extract_osz(archive, root / "extract")

    def test_explicit_osu_and_audio_are_isolated(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            beatmap = root / "standard.osu"
            beatmap.write_bytes((FIXTURES / "standard.osu").read_bytes())
            (root / "audio.wav").write_bytes(b"RIFF-fixture")
            with prepare_source(beatmap) as workspace:
                self.assertNotEqual(workspace.root, root)
                self.assertTrue(workspace.document.path.is_file())
                self.assertTrue(workspace.audio.is_file())

    def test_osz_output_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            content = root / "content"
            content.mkdir()
            (content / "map.osu").write_bytes((FIXTURES / "standard.osu").read_bytes())
            (content / "audio.wav").write_bytes(b"RIFF-fixture")
            first = root / "first.osz"
            second = root / "second.osz"
            write_osz(content, first)
            write_osz(content, second)
            self.assertEqual(first.read_bytes(), second.read_bytes())
            validate_osz(first, expected_osu_count=1, expected_audio_count=1)
            with self.assertRaises(InputError):
                validate_osz(first, expected_osu_count=6)

            with prepare_source(first) as workspace:
                self.assertEqual(workspace.document.mode, 0)
                self.assertEqual(workspace.audio.name, "audio.wav")
                self.assertTrue(workspace.audio.is_file())


if __name__ == "__main__":
    unittest.main()
