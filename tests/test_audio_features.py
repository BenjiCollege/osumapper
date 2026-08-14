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


class TimingRefinementTests(unittest.TestCase):
    def test_refined_tempo_lands_on_the_beat_of_a_synthetic_click_track(self) -> None:
        # Beat tracking returns tempo too coarsely for mapping: a 1.4 BPM error is
        # about 3ms per beat, so a map drifts a whole beat off within minutes.
        # A click track at a known tempo must come back at that tempo.
        import math
        import tempfile
        import wave
        from array import array

        from osumapper.timing import estimate_timing

        rate = 22_050
        bpm = 150.0
        beat_samples = int(rate * 60.0 / bpm)
        total = beat_samples * 96
        samples = array("h", bytes(total * 2))
        for beat in range(96):
            start = beat * beat_samples
            for index in range(900):  # short percussive click
                if start + index >= total:
                    break
                decay = math.exp(-index / 120.0)
                samples[start + index] = int(
                    12_000 * decay * math.sin(2 * math.pi * 1_800 * index / rate)
                )
        with tempfile.TemporaryDirectory() as name:
            path = Path(name) / "clicks.wav"
            with wave.open(str(path), "wb") as stream:
                stream.setnchannels(1)
                stream.setsampwidth(2)
                stream.setframerate(rate)
                stream.writeframes(samples.tobytes())

            estimate = estimate_timing(path)

        multiples = (bpm / 2, bpm, bpm * 2)
        self.assertTrue(
            any(abs(estimate.bpm - value) < 1.0 for value in multiples),
            f"expected a multiple of {bpm}, got {estimate.bpm}",
        )
        # Whole or half BPM: real music sits there, and so should the estimate.
        self.assertAlmostEqual(estimate.bpm * 2, round(estimate.bpm * 2), places=3)
