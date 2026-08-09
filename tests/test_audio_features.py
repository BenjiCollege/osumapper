from __future__ import annotations

import math
import tempfile
import unittest
import wave
from array import array
from pathlib import Path

import numpy as np

from osumapper.engine import _audio_features


class AudioFeatureTests(unittest.TestCase):
    def test_22050_hz_audio_keeps_model_frequency_shape(self) -> None:
        sample_rate = 22_050
        duration_seconds = 4
        samples = array(
            "h",
            (
                round(12_000 * math.sin(2 * math.pi * 440 * index / sample_rate))
                for index in range(sample_rate * duration_seconds)
            ),
        )
        with tempfile.TemporaryDirectory() as name:
            audio = Path(name) / "tone.wav"
            with wave.open(str(audio), "wb") as stream:
                stream.setnchannels(1)
                stream.setsampwidth(2)
                stream.setframerate(sample_rate)
                stream.writeframes(samples.tobytes())

            timestamps = np.arange(500, 3_500, 125)
            features = _audio_features(timestamps, audio)

        self.assertEqual(features.shape, (24, 7, 2, 32))
        self.assertTrue(np.isfinite(features).all())


if __name__ == "__main__":
    unittest.main()
