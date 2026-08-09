from __future__ import annotations

import importlib.util
import unittest

from osumapper.models import legacy_model_paths, modern_model_path, verify_all


@unittest.skipUnless(importlib.util.find_spec("tensorflow"), "TensorFlow is not installed")
class ModelPredictionParityTests(unittest.TestCase):
    def test_every_legacy_model_has_a_verified_prediction_parity_manifest(self) -> None:
        sources = legacy_model_paths()
        if not all(modern_model_path(source).is_file() for source in sources):
            self.skipTest("Run `osumapper models migrate` to create parity-gated models")
        results = verify_all()
        self.assertEqual(len(results), len(sources))
        for result in results:
            self.assertLessEqual(result.max_absolute_error, 1e-5)


if __name__ == "__main__":
    unittest.main()
