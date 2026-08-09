from __future__ import annotations

import unittest

from osumapper.config import GameMode
from osumapper.models import legacy_model_paths
from osumapper.presets import get_preset, preset_names


class PresetTests(unittest.TestCase):
    def test_all_legacy_models_are_discoverable_for_migration(self) -> None:
        self.assertEqual(len(legacy_model_paths()), 14)

    def test_all_public_presets_resolve(self) -> None:
        for name in preset_names():
            with self.subTest(name=name):
                preset = get_preset(name)
                self.assertTrue(preset.rhythm_model.is_file())

    def test_default_selects_source_ruleset(self) -> None:
        self.assertEqual(get_preset("default", GameMode.TAIKO).name, "taiko")
        self.assertEqual(get_preset("default", GameMode.CATCH).name, "catch")
        self.assertEqual(get_preset("default", GameMode.MANIA).name, "mania-lowkey")


if __name__ == "__main__":
    unittest.main()
