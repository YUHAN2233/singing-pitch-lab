"""Transform, lifecycle and actual UVR ONNX integration tests.

Run from the project root:
    .venv\\Scripts\\python.exe -m unittest tests.test_separation -v
The real-model tests require the explicitly installed project model. Generated
signals are original fixtures; their results verify execution, not music SDR.
"""

import json
from pathlib import Path
import time
import unittest
import uuid

import numpy as np

from backend.separation import (
    COMPENSATION, DIM_F, DIM_T, HOP_LENGTH, MODEL_NAME, MODEL_UVR_HASH,
    N_FFT, SAMPLE_RATE, STFT, CancelledError, _process_chunks, model_info,
    prepare_audio, separate, uvr_model_hash,
)

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models" / MODEL_NAME


def generated_mix(seconds=6.25, sample_rate=SAMPLE_RATE):
    """An original harmonic lead over a stereo bass/chord/percussion signal."""
    time_axis = np.arange(round(seconds * sample_rate), dtype=np.float32) / sample_rate
    frequency = 220 * (1 + 0.005 * np.sin(2 * np.pi * 5 * time_axis))
    phase = 2 * np.pi * np.cumsum(frequency, dtype=np.float64) / sample_rate
    envelope = np.minimum(time_axis * 6, 1) * np.minimum((seconds - time_axis) * 6, 1)
    lead = sum(np.sin(harmonic * phase) / harmonic for harmonic in range(1, 7))
    lead = (0.18 * lead * envelope).astype(np.float32)
    bass = 0.08 * np.sin(2 * np.pi * 82.4069 * time_axis)
    chord_left = 0.07 * np.sin(2 * np.pi * 329.628 * time_axis)
    chord_right = 0.07 * np.sin(2 * np.pi * 391.995 * time_axis)
    rng = np.random.default_rng(415)
    percussion = rng.standard_normal(len(time_axis)).astype(np.float32) * 0.012
    percussion *= np.exp(-18 * (time_axis % 0.5))
    mix = np.column_stack((lead + bass + chord_left + percussion,
                           lead + bass + chord_right + percussion))
    return mix.astype(np.float32), lead


class TransformTests(unittest.TestCase):
    def test_full_band_round_trip_preserves_amplitude_and_stereo(self):
        rng = np.random.default_rng(4)
        audio = rng.normal(0, 0.15, (2, HOP_LENGTH * 255)).astype(np.float32)
        transform = STFT(dim_f=N_FFT // 2 + 1)
        recovered = transform.inverse(transform(audio))
        self.assertEqual(recovered.shape, audio.shape)
        np.testing.assert_allclose(recovered, audio, atol=2e-7, rtol=3e-6)

    def test_frequency_amplitude_and_complex_channel_layout(self):
        samples = np.arange(HOP_LENGTH * 255, dtype=np.float64)
        left = np.cos(2 * np.pi * 60 * samples / N_FFT)
        right = np.sin(2 * np.pi * 90 * samples / N_FFT)
        packed = STFT()(np.array([left, right], dtype=np.float32))
        self.assertEqual(packed.shape, (1, 4, DIM_F, DIM_T))
        self.assertEqual(packed.dtype, np.float32)
        # A unit sinusoid has N/4 raw Hann-STFT amplitude, not 1 or N/2.
        self.assertAlmostEqual(float(packed[0, 0, 60, 15]), N_FFT / 4, places=3)
        self.assertAlmostEqual(float(packed[0, 1, 60, 15]), 0, places=3)
        self.assertAlmostEqual(float(packed[0, 2, 90, 15]), 0, places=3)
        self.assertAlmostEqual(float(packed[0, 3, 90, 15]), -N_FFT / 4, places=3)

    def test_overlap_add_has_no_delay_at_start_end_or_chunk_boundaries(self):
        size = HOP_LENGTH * (DIM_T - 1)
        boundary = size - N_FFT
        n = 2 * boundary + 721
        audio = np.zeros((n, 2), np.float32)
        positions = [0, 1, N_FFT // 2, boundary - 1, boundary,
                     boundary + 1, 2 * boundary, n - 1]
        for i, position in enumerate(positions):
            audio[position] = (0.1 + i * 0.03, -0.2 - i * 0.02)
        transform = STFT(dim_f=N_FFT // 2 + 1)
        restored = _process_chunks(audio, lambda packed: packed,
                                   stft=transform, remove_low_bins=False)
        np.testing.assert_allclose(restored, audio, atol=3e-7)
        np.testing.assert_array_equal(np.flatnonzero(abs(restored[:, 0]) > 1e-5), positions)

    def test_resampling_mono_preserves_pitch_duration_and_onset(self):
        sr = 48000
        t = np.arange(sr, dtype=np.float32) / sr
        input_audio = (0.2 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        stereo = prepare_audio(input_audio, sr)
        self.assertEqual(stereo.shape, (44100, 2))
        np.testing.assert_array_equal(stereo[:, 0], stereo[:, 1])
        peak = np.argmax(np.abs(np.fft.rfft(stereo[:, 0])))
        self.assertEqual(peak, 440)
        impulse = np.zeros((sr, 2), np.float32)
        impulse[24000] = 1
        resampled = prepare_audio(impulse, sr)
        self.assertEqual(int(np.argmax(resampled[:, 0])), 22050)

    def test_invalid_audio_is_rejected(self):
        bad_values = [np.zeros((0, 2), np.float32), np.zeros((5, 3), np.float32),
                      np.zeros((5, 2), np.int16), np.array([np.nan], np.float32),
                      np.array([np.inf], np.float32), np.array([1j], np.complex64)]
        for bad in bad_values:
            with self.subTest(shape=bad.shape, dtype=bad.dtype):
                with self.assertRaises(ValueError):
                    prepare_audio(bad, SAMPLE_RATE)
        for sr in [True, 0, 4000, 200000, 44100.5]:
            with self.subTest(sr=sr), self.assertRaises(ValueError):
                prepare_audio(np.zeros(20, np.float32), sr)

    def test_uvr_hash_uses_last_10000_kib_not_10_mib(self):
        import hashlib
        tail_size = 10000 * 1024
        # Python 3.13's mode-0700 TemporaryDirectory can exclude a restricted
        # Windows token. A uniquely named file inherits project directory ACLs.
        path = ROOT / "evidence" / f"hash-fixture-{uuid.uuid4().hex}.bin"
        try:
            content = b"header" * 50000 + b"T" * tail_size
            path.write_bytes(content)
            self.assertEqual(uvr_model_hash(path), hashlib.md5(b"T" * tail_size).hexdigest())
            path.write_bytes(b"short-model")
            self.assertEqual(uvr_model_hash(path), hashlib.md5(b"short-model").hexdigest())
        finally:
            path.unlink(missing_ok=True)


class LifecycleTests(unittest.TestCase):
    def test_cancel_before_model_access(self):
        with self.assertRaises(CancelledError):
            separate(np.zeros(44100, np.float32), 44100, Path("does-not-exist"),
                     cancel=lambda: True)

    def test_cancel_after_inference_and_then_recover(self):
        mix, _ = generated_mix(seconds=0.25)
        stop = False
        calls = 0

        def prediction(packed):
            nonlocal stop, calls
            calls += 1
            stop = True
            return packed

        with self.assertRaises(CancelledError):
            _process_chunks(mix, prediction, cancel=lambda: stop)
        self.assertEqual(calls, 1)
        progress = []
        recovered = _process_chunks(mix, lambda x: x, progress=progress.append)
        self.assertEqual(recovered.shape, mix.shape)
        self.assertTrue(np.isfinite(recovered).all())
        self.assertEqual(progress[0], 0)
        self.assertEqual(progress[-1], 1)
        self.assertTrue(all(a <= b for a, b in zip(progress, progress[1:])))

    def test_prediction_failure_is_visible_and_next_call_recovers(self):
        mix, _ = generated_mix(seconds=0.1)

        def fail(_packed):
            raise RuntimeError("simulated runtime failure")

        with self.assertRaisesRegex(RuntimeError, "simulated runtime failure"):
            _process_chunks(mix, fail)
        for bad in [lambda x: np.full_like(x, np.nan), lambda x: x[:, :, :-1]]:
            with self.assertRaisesRegex(RuntimeError, "无效数据"):
                _process_chunks(mix, bad)
        self.assertTrue(np.isfinite(_process_chunks(mix, lambda x: x)).all())

    def test_missing_model_is_not_silently_replaced_by_filter(self):
        mix, _ = generated_mix(seconds=0.1)
        with self.assertRaises(FileNotFoundError):
            separate(mix, SAMPLE_RATE, Path("missing-vocal-model.onnx"))


@unittest.skipUnless(MODEL.is_file(), "Install the explicitly documented local ONNX model first")
class ActualModelIntegrationTests(unittest.TestCase):
    def test_actual_onnx_separation_and_reconstruction(self):
        import soundfile as sf

        info = model_info(MODEL)
        self.assertEqual(info["uvr_hash"], MODEL_UVR_HASH)
        self.assertEqual(info["primary_stem"], "Vocals")
        mix, lead = generated_mix()
        original = mix.copy()
        progress = []
        started = time.perf_counter()
        vocals, accompaniment, sr = separate(mix, SAMPLE_RATE, MODEL,
                                             progress=progress.append)
        elapsed = time.perf_counter() - started
        self.assertEqual(sr, SAMPLE_RATE)
        self.assertEqual(vocals.shape, mix.shape)
        self.assertEqual(accompaniment.shape, mix.shape)
        self.assertEqual(vocals.dtype, np.float32)
        self.assertTrue(np.isfinite(vocals).all())
        self.assertTrue(np.isfinite(accompaniment).all())
        np.testing.assert_array_equal(mix, original)
        residual = float(np.max(np.abs(vocals + accompaniment - mix)))
        self.assertLess(residual, 2e-7)
        vocal_rms = float(np.sqrt(np.mean(vocals * vocals)))
        backing_rms = float(np.sqrt(np.mean(accompaniment * accompaniment)))
        self.assertGreater(vocal_rms, 1e-6)
        self.assertGreater(backing_rms, 1e-4)
        self.assertGreater(float(np.max(np.abs(vocals - mix))), 1e-3)
        self.assertEqual(progress[-1], 1)
        self.assertGreaterEqual(len(progress), 3)  # Real inference across chunks.
        artifact_dir = ROOT / "evidence" / "model_integration"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        for name, audio in [("original_generated_mix", mix), ("estimated_vocals", vocals),
                            ("estimated_accompaniment", accompaniment)]:
            sf.write(artifact_dir / f"{name}.wav", audio, sr, subtype="FLOAT")
        report = {
            "fixture": "Original generated harmonic lead plus stereo bass/chord/percussion",
            "quality_scope": "Verifies real model execution and timing plumbing; not a music separation quality benchmark",
            "model": info, "duration_seconds": len(mix) / sr,
            "elapsed_seconds": elapsed, "sample_count": len(mix),
            "output_shape": list(vocals.shape), "reconstruction_max_absolute_error": residual,
            "vocal_rms": vocal_rms, "accompaniment_rms": backing_rms,
            "finite": True, "input_unchanged": True, "progress": progress,
        }
        (artifact_dir / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print("\nMODEL_INTEGRATION " + json.dumps(report, ensure_ascii=False))

    def test_digital_silence_stays_silent(self):
        vocals, accompaniment, sr = separate(np.zeros((3000, 1), np.float32), 48000, MODEL)
        self.assertEqual(vocals.shape, (2757, 2))
        self.assertEqual(sr, SAMPLE_RATE)
        self.assertEqual(float(np.max(np.abs(vocals))), 0)
        self.assertEqual(float(np.max(np.abs(accompaniment))), 0)


if __name__ == "__main__":
    unittest.main()
