"""Experimental two-F0 analysis for isolated, simultaneous harmonic voices.

This estimates two pitches, not singer identity or separated audio. Spectral
peaks explained by the first harmonic series cannot independently justify a
second voice. Closely spaced, unison, octave-related and noisy voices remain
ambiguous. Inspired by Klapuri's harmonic salience/iterative multi-F0 work:
https://archives.ismir.net/ismir2006/paper/000125.pdf
"""
from __future__ import annotations

import math
from typing import Callable

import numpy as np
from scipy.signal import find_peaks, resample_poly

from .pitch import ANALYSIS_RATE, AnalysisCancelled, MIN_RMS, frequency_details, _mono_float

WINDOW = 4096
FFT = 8192
HOP = 640  # 40 ms
MIN_F0 = 75.0
MAX_F0 = 750.0
MAX_HARMONIC = 8


def _pitch_record(frequency: float | None, confidence: float = 0.0) -> dict:
    midi, note, cents = frequency_details(frequency) if frequency else (None, None, None)
    return {"frequency": round(frequency, 3) if frequency else None,
            "midi": round(midi, 4) if midi is not None else None, "note": note,
            "cents": round(cents, 2) if cents is not None else None,
            "confidence": round(confidence, 3)}


def _frame_candidates(window: np.ndarray) -> tuple[list[dict], float]:
    signal = window - np.mean(window)
    centre = window[len(window)//2-HOP//4:len(window)//2+HOP//4]
    centre = centre - np.mean(centre)
    rms = float(np.sqrt(np.mean(centre * centre)))
    half_rms = [float(np.sqrt(np.mean((part - np.mean(part)) ** 2)))
                for part in np.split(centre, 2)]
    if rms < MIN_RMS or min(half_rms) < MIN_RMS:
        return [], rms
    spectrum = np.abs(np.fft.rfft(signal * np.hanning(len(signal)), FFT))
    strongest = float(np.max(spectrum))
    if strongest < 1e-8:
        return [], rms
    peaks, _ = find_peaks(spectrum, height=strongest * .035,
                          prominence=strongest * .022, distance=3)
    peaks = peaks[(peaks * ANALYSIS_RATE / FFT >= MIN_F0) &
                  (peaks * ANALYSIS_RATE / FFT <= 3600)]
    peaks = sorted(peaks, key=lambda p: spectrum[p], reverse=True)[:36]
    if not peaks:
        return [], rms
    peaks = np.array(peaks, dtype=int)
    left, mid, right = spectrum[peaks - 1], spectrum[peaks], spectrum[peaks + 1]
    curvature = left - 2*mid + right
    shift = np.clip(.5 * (left - right) / np.where(curvature < -1e-12, curvature, -1.), -.5, .5)
    # Only interpolate concave spectral peaks.
    shift = np.where(curvature < 0, shift, 0)
    frequencies = (peaks + shift) * ANALYSIS_RATE / FFT
    amplitudes = spectrum[peaks]
    candidates = []
    for peak_f in frequencies:
        for divisor in range(1, 6):
            f0 = peak_f / divisor
            if MIN_F0 <= f0 <= MAX_F0 and all(abs(1200*math.log2(f0/c)) > 18 for c in candidates):
                candidates.append(float(f0))

    def rank(f0: float, excluded: set[int]) -> dict | None:
        indices = []
        harmonics = []
        for harmonic in range(1, MAX_HARMONIC + 1):
            target = harmonic * f0
            if target > 3600:
                break
            differences = abs(frequencies - target)
            index = int(np.argmin(differences))
            if index in excluded or index in indices or differences[index] > max(4., target * .018):
                continue
            indices.append(index)
            harmonics.append(harmonic)
        if len(indices) < 2 or min(harmonics) > 2:
            return None
        # Two independent partials are the minimum evidence. A single upper
        # harmonic cannot turn one singer into two detected singers.
        strength = sum(math.sqrt(float(amplitudes[i] / strongest)) / math.sqrt(h)
                       for i, h in zip(indices, harmonics))
        strength *= min(1., len(indices) / 3.)
        if 1 not in harmonics:
            strength *= .72
        if strength < .48:
            return None
        # Refine F0 from several matched partials, not one FFT bin.
        refined = float(np.median([frequencies[i] / h for i, h in zip(indices, harmonics)]))
        if not MIN_F0 <= refined <= MAX_F0:
            return None
        return {"frequency": refined, "strength": strength, "indices": set(indices),
                "harmonics": harmonics}

    first = max((item for f in candidates if (item := rank(f, set())) is not None),
                key=lambda item: item["strength"], default=None)
    if first is None:
        return [], rms
    # Exclude every spectral peak near the selected harmonic series, including
    # weak partials not used in its fit. This is intentionally conservative.
    excluded = {i for i, peak in enumerate(frequencies)
                if any(abs(peak - h*first["frequency"]) <= max(5., h*first["frequency"]*.025)
                       for h in range(1, MAX_HARMONIC+1))}
    second = max((item for f in candidates if (item := rank(f, excluded)) is not None
                  and abs(1200*math.log2(item["frequency"]/first["frequency"])) >= 85),
                 key=lambda item: item["strength"], default=None)
    if second and (second["strength"] < .55 or second["strength"] < first["strength"]*.27):
        second = None
    selected = [first] + ([second] if second else [])
    selected.sort(key=lambda item: item["frequency"])
    return [_pitch_record(item["frequency"], min(.99, item["strength"] / 2.5)) for item in selected], rms


def analyze_two_voices(audio: np.ndarray, sample_rate: int,
                       cancel: Callable[[], bool] | None = None,
                       progress: Callable[[float], None] | None = None) -> list[dict]:
    """Return lower/higher simultaneous pitch slots; slots are not identities."""
    if isinstance(sample_rate, bool) or not isinstance(sample_rate, (int, np.integer)) or sample_rate < 4000:
        raise ValueError("采样率必须至少为 4000 Hz")
    signal = _mono_float(audio)
    if cancel and cancel():
        raise AnalysisCancelled("分析已取消")
    if sample_rate != ANALYSIS_RATE:
        gcd = math.gcd(int(sample_rate), ANALYSIS_RATE)
        signal = resample_poly(signal, ANALYSIS_RATE//gcd, int(sample_rate)//gcd)
    if len(signal) < WINDOW:
        signal = np.pad(signal, (0, WINDOW-len(signal)))
    starts = range(0, len(signal)-WINDOW+1, HOP)
    frames = []
    for index, start in enumerate(starts):
        if cancel and cancel():
            raise AnalysisCancelled("分析已取消")
        pitches, rms = _frame_candidates(signal[start:start+WINDOW])
        frames.append({"time": round((start+WINDOW/2)/ANALYSIS_RATE, 4),
                       "lower": pitches[0] if pitches else _pitch_record(None),
                       "higher": pitches[1] if len(pitches) > 1 else _pitch_record(None),
                       "voice_count": len(pitches), "rms": round(rms, 6),
                       "ambiguous": len(pitches) < 2})
        if progress and (index % 10 == 0 or index == len(starts)-1):
            progress((index+1)/len(starts))
    return frames
