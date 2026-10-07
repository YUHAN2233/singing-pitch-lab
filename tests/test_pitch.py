"""Deterministic algorithm tests; run: python -m unittest tests.test_pitch -v."""

import math
from contextlib import contextmanager
from pathlib import Path
import unittest
import uuid
import wave

import numpy as np

from backend.pitch import (
    ANALYSIS_RATE,
    MIN_CONFIDENCE,
    AnalysisCancelled,
    analyze_pitch,
    compare_tracks,
    frequency_details,
)
from scripts.generate_fixtures import generate


def tone(frequency=220.0, duration=0.8, sample_rate=ANALYSIS_RATE, amplitude=0.3):
    time = np.arange(int(duration * sample_rate)) / sample_rate
    return amplitude * np.sin(2 * np.pi * frequency * time)


def voiced(frames):
    return [frame for frame in frames if frame["frequency"] is not None]


def track(frequencies, start=0.064, step=0.02, confidence=0.99):
    frames = []
    for index, frequency in enumerate(frequencies):
        midi, note, cents = frequency_details(frequency) if frequency else (None, None, None)
        frames.append({"time": round(start + index * step, 6), "frequency": frequency,
                       "midi": midi, "note": note, "cents": cents,
                       "confidence": confidence if frequency else 0.0, "rms": 0.2 if frequency else 0.0})
    return frames


@contextmanager
def fixture_directory():
    # Python 3.13 tempfile applies a restrictive Windows ACL (mode 0700), which
    # conflicts with some managed executor identities. Use ordinary inherited
    # workspace ACLs and clean up only this test's known flat directory.
    root = Path(__file__).resolve().parent
    directory = root / f".pitch-fixtures-{uuid.uuid4().hex}"
    directory.mkdir()
    try:
        yield directory
    finally:
        assert directory.resolve().parent == root
        for artifact in directory.iterdir():
            artifact.unlink()
        directory.rmdir()


class PitchTests(unittest.TestCase):
    def assert_pitch(self, data, expected, sample_rate=ANALYSIS_RATE, tolerance=6.0):
        frames = analyze_pitch(data, sample_rate)
        good = voiced(frames)
        self.assertGreater(len(good), len(frames) * 0.95)
        error = 1200 * np.log2(np.array([frame["frequency"] for frame in good]) / expected)
        self.assertLess(float(np.max(np.abs(error))), tolerance)
        return frames

    def test_pure_tones_full_range_and_high_note_quantization(self):
        for frequency in (60.0, 65.406391, 110.0, 220.0, 261.625565, 440.0, 659.255, 880.0, 1000.0, 1046.502, 1100.0):
            with self.subTest(frequency=frequency):
                self.assert_pitch(tone(frequency), frequency)

    def test_second_harmonic_stronger_than_fundamental(self):
        for fundamental in (110.0, 220.0, 440.0):
            with self.subTest(fundamental=fundamental):
                signal = tone(fundamental, amplitude=0.1) + tone(2 * fundamental, amplitude=0.75)
                self.assert_pitch(signal, fundamental)

    def test_missing_fundamental_with_second_and_third_harmonics(self):
        self.assert_pitch(tone(440, amplitude=0.5) + tone(660, amplitude=0.35), 220.0)

    def test_real_octave_jump_is_not_folded(self):
        signal = np.concatenate([tone(220, duration=0.5), tone(440, duration=0.5)])
        frames = analyze_pitch(signal, ANALYSIS_RATE)
        low = [frame["midi"] for frame in frames if 0.12 < frame["time"] < 0.40]
        high = [frame["midi"] for frame in frames if 0.60 < frame["time"] < 0.90]
        self.assertAlmostEqual(float(np.median(high) - np.median(low)), 12.0, delta=0.05)

    def test_silence_and_subthreshold_signal_are_null(self):
        for data in (np.zeros(16000), tone(amplitude=0.0005), np.full(16000, 0.2)):
            frames = analyze_pitch(data, ANALYSIS_RATE)
            self.assertTrue(frames)
            for frame in frames:
                self.assertIsNone(frame["frequency"])
                self.assertIsNone(frame["midi"])
                self.assertIsNone(frame["note"])
                self.assertIsNone(frame["cents"])
                self.assertEqual(frame["confidence"], 0.0)

    def test_seeded_broadband_noise_is_unvoiced(self):
        frames = analyze_pitch(np.random.default_rng(20261005).normal(0, 0.18, 32000), ANALYSIS_RATE)
        self.assertEqual(len(voiced(frames)), 0)
        self.assertTrue(all(frame["confidence"] < MIN_CONFIDENCE for frame in frames))

    def test_short_silent_pauses_are_not_filled_by_neighbouring_notes(self):
        for gap_seconds in (0.08, 0.10):
            for sample_rate in (ANALYSIS_RATE, 44100):
                with self.subTest(gap_seconds=gap_seconds, sample_rate=sample_rate):
                    note = tone(261.625565, duration=0.7, sample_rate=sample_rate)
                    signal = np.concatenate([note, np.zeros(round(gap_seconds * sample_rate)), note])
                    # A DC offset must not turn the centre's silence into energy.
                    frames = analyze_pitch(signal + 0.1, sample_rate)
                    silent_centres = [frame for frame in frames
                                      if 0.7 <= frame["time"] < 0.7 + gap_seconds]
                    self.assertGreaterEqual(len(silent_centres), 3)
                    for frame in silent_centres:
                        self.assertIsNone(frame["frequency"])
                        self.assertIsNone(frame["midi"])
                        self.assertIsNone(frame["note"])
                        self.assertIsNone(frame["cents"])
                        self.assertEqual(frame["confidence"], 0.0)
                        # Full-window energy can be high even though its centre
                        # is silent; retaining it must not override the gate.
                        self.assertGreater(frame["rms"], 0.005)
                    sustained = [frame for frame in frames if 0.2 < frame["time"] < 0.6
                                 or 0.9 < frame["time"] < 1.2]
                    self.assertTrue(all(frame["frequency"] is not None for frame in sustained))

    def test_note_names_and_signed_cents(self):
        midi, note, cents = frequency_details(440 * 2 ** (35 / 1200))
        self.assertAlmostEqual(midi, 69.35)
        self.assertEqual(note, "A4")
        self.assertAlmostEqual(cents, 35.0)
        self.assertAlmostEqual(frequency_details(440 * 2 ** (-35 / 1200))[2], -35.0)
        self.assertEqual(frequency_details(261.625565)[1], "C4")

    def test_resampling_and_samples_by_channels(self):
        for sample_rate in (8000, 22050, 44100, 48000):
            with self.subTest(sample_rate=sample_rate):
                signal = tone(220, sample_rate=sample_rate)
                stereo = np.column_stack([signal, signal * 0.8])
                self.assert_pitch(stereo, 220.0, sample_rate)

    def test_integer_pcm_is_normalized(self):
        signal = (tone(220) * 32767).astype(np.int16)
        self.assert_pitch(signal, 220)
        unsigned = (tone(220) * 128 + 128).astype(np.uint8)
        self.assert_pitch(unsigned, 220)

    def test_times_are_window_centres_at_twenty_ms_hop(self):
        frames = analyze_pitch(tone(duration=1.0), ANALYSIS_RATE)
        self.assertEqual(frames[0]["time"], 0.064)
        np.testing.assert_allclose(np.diff([frame["time"] for frame in frames]), 0.02, atol=1e-9)
        self.assertLessEqual(frames[-1]["time"] + 0.064, 1.0)

    def test_empty_short_and_invalid_input(self):
        self.assertEqual(analyze_pitch(np.array([]), ANALYSIS_RATE), [])
        short = analyze_pitch(tone(duration=0.05), ANALYSIS_RATE)
        self.assertEqual(len(short), 1)
        self.assertEqual(short[0]["time"], 0.025)
        for data in (np.array([np.nan]), np.array([np.inf]), np.zeros((2, 16000)), np.zeros((1, 2, 3))):
            with self.assertRaises(ValueError):
                analyze_pitch(data, ANALYSIS_RATE)
        with self.assertRaises(ValueError):
            analyze_pitch(tone(), 0)

    def test_progress_and_cancellation(self):
        progress = []
        analyze_pitch(tone(duration=1.2), ANALYSIS_RATE, progress=progress.append)
        self.assertEqual(progress[0], 0.0)
        self.assertEqual(progress[-1], 1.0)
        self.assertEqual(progress, sorted(progress))
        counter = {"calls": 0}

        def cancel():
            counter["calls"] += 1
            return counter["calls"] >= 4

        with self.assertRaises(AnalysisCancelled):
            analyze_pitch(tone(), ANALYSIS_RATE, cancel=cancel)
        with self.assertRaises(AnalysisCancelled):
            analyze_pitch(tone(), ANALYSIS_RATE, cancel=lambda: True)


class ComparisonTests(unittest.TestCase):
    def test_signed_deviation_and_accuracy(self):
        reference = track([220.0] * 10)
        recording = track([220 * 2 ** (35 / 1200)] * 8 + [220 * 2 ** (70 / 1200)] * 2)
        result = compare_tracks(reference, recording)
        summary = result["summary"]
        self.assertEqual(summary["comparable_frames"], 10)
        self.assertEqual(summary["voiced_coverage"], 1.0)
        self.assertAlmostEqual(summary["median_abs_cents"], 35)
        self.assertAlmostEqual(summary["within_50_cents_ratio"], 0.8)
        self.assertAlmostEqual(summary["mean_signed_cents"], 42)
        self.assertAlmostEqual(result["points"][0]["cents"], 35)

    def test_up_and_down_transposition(self):
        reference = track([220.0] * 10)
        for transpose in (-5, 3):
            with self.subTest(transpose=transpose):
                recording = track([220 * 2 ** (transpose / 12)] * 10)
                self.assertAlmostEqual(compare_tracks(reference, recording)["summary"]["mean_signed_cents"], transpose * 100)
                self.assertAlmostEqual(compare_tracks(reference, recording, transpose=transpose)["summary"]["median_abs_cents"], 0)

    def test_actual_octave_error_remains_twelve_hundred_cents(self):
        summary = compare_tracks(track([220.0] * 10), track([440.0] * 10))["summary"]
        self.assertEqual(summary["median_abs_cents"], 1200.0)
        self.assertEqual(summary["within_50_cents_ratio"], 0.0)

    def test_positive_offset_recording_time_maps_forward_to_reference(self):
        frequencies = [220 * 2 ** (index / 120) for index in range(25)]
        reference = track(frequencies)
        recording = track(frequencies[5:])
        result = compare_tracks(reference, recording, offset_seconds=0.1)
        self.assertEqual(result["summary"]["comparable_frames"], 20)
        self.assertAlmostEqual(result["summary"]["median_abs_cents"], 0, delta=0.001)
        self.assertEqual(result["points"][0]["reference_time"], 0.164)
        self.assertGreater(compare_tracks(reference, recording)["summary"]["median_abs_cents"], 40)

    def test_negative_offset_compensates_recording_start_delay(self):
        frequencies = [220 * 2 ** (index / 120) for index in range(20)]
        reference = track(frequencies)
        recording = track([None] * 5 + frequencies)
        result = compare_tracks(reference, recording, offset_seconds=-0.1)
        self.assertEqual(result["summary"]["comparable_frames"], 20)
        self.assertEqual(result["summary"]["median_abs_cents"], 0)
        self.assertEqual(result["points"][0]["status"], "out_of_range")

    def test_sub_hop_offset_uses_log_frequency_interpolation(self):
        reference = track([220 * 2 ** (index / 120) for index in range(20)])
        recording = track([220 * 2 ** ((index + 0.5) / 120) for index in range(19)])
        summary = compare_tracks(reference, recording, offset_seconds=0.01)["summary"]
        self.assertEqual(summary["comparable_frames"], 19)
        self.assertAlmostEqual(summary["median_abs_cents"], 0)

    def test_low_confidence_and_silence_do_not_count_as_correct(self):
        reference = track([220.0] * 10)
        recording = track([220.0] * 5 + [None] * 5)
        recording[0]["confidence"] = 0.5
        result = compare_tracks(reference, recording)
        self.assertEqual(result["summary"]["comparable_frames"], 4)
        self.assertEqual(result["summary"]["voiced_coverage"], 0.4)
        self.assertEqual(result["summary"]["comparable_ratio"], 0.4)
        self.assertIsNone(result["points"][0]["cents"])
        self.assertEqual(result["points"][0]["status"], "recording_unvoiced")
        reference[1]["confidence"] = 0.2
        self.assertEqual(compare_tracks(reference, recording)["points"][1]["status"], "reference_unvoiced")

    def test_reference_gaps_are_not_bridged(self):
        reference = track([220.0, 220.0]) + track([220.0, 220.0], start=0.564)
        recording = track([220.0] * 25)
        points = compare_tracks(reference, recording)["points"]
        self.assertEqual(points[10]["status"], "reference_gap")
        self.assertIsNone(points[10]["cents"])

    def test_empty_comparison_has_no_fake_accuracy(self):
        summary = compare_tracks([], [])["summary"]
        self.assertEqual(summary["comparable_frames"], 0)
        self.assertEqual(summary["voiced_coverage"], 0)
        self.assertIsNone(summary["median_abs_cents"])
        self.assertIsNone(summary["within_50_cents_ratio"])

    def test_invalid_alignment_parameters(self):
        with self.assertRaises(ValueError):
            compare_tracks([], [], offset_seconds=math.nan)
        with self.assertRaises(ValueError):
            compare_tracks([], [], transpose=math.inf)


class GeneratedFixtureTests(unittest.TestCase):
    def test_original_fixture_audio_matches_gold_and_sharp_offset(self):
        with fixture_directory() as output:
            gold = generate(output)

            def read_mono(name):
                with wave.open(str(output / name), "rb") as stream:
                    self.assertEqual(stream.getnchannels(), 1)
                    return np.frombuffer(stream.readframes(stream.getnframes()), dtype="<i2"), stream.getframerate()

            dry, sample_rate = read_mono("melody_dry.wav")
            sharp, _ = read_mono("melody_sharp.wav")
            reference = analyze_pitch(dry, sample_rate)
            recording = analyze_pitch(sharp, sample_rate)
            stable_errors = []
            for note in gold["notes"]:
                stable_frames = [frame for frame in reference if note["start"] + 0.1 < frame["time"] < note["end"] - 0.1]
                self.assertTrue(stable_frames)
                self.assertTrue(all(frame["frequency"] is not None for frame in stable_frames))
                stable_errors.extend(1200 * math.log2(frame["frequency"] / note["frequency"]) for frame in stable_frames)
            self.assertLess(max(abs(value) for value in stable_errors), 2.0)
            summary = compare_tracks(reference, recording)["summary"]
            self.assertAlmostEqual(summary["median_signed_cents"], gold["sharp_cents"], delta=1.0)
            self.assertGreater(summary["voiced_coverage"], 0.98)
            self.assertLess(summary["comparable_ratio"], 0.85)  # Silent gaps are not counted as accurate notes.
            with wave.open(str(output / "synthetic_mix.wav"), "rb") as stream:
                self.assertEqual(stream.getnchannels(), 2)
                self.assertEqual(stream.getnframes(), round(gold["duration_seconds"] * sample_rate))
            self.assertEqual(gold["sha256"], generate(output)["sha256"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
