"""Local monophonic pitch analysis and time-aligned comparison.

YIN reference: de Cheveigne & Kawahara (2002), JASA 111(4), 1917-1930,
https://pubmed.ncbi.nlm.nih.gov/12002874/ . This is an independent NumPy
implementation with FFT difference calculation and conservative valley selection.

Input must be an isolated vocal / other single pitched source, NOT a musical mix.
Confidence describes periodicity, not the probability of vocal activity or of a
correct singer/note. A separate centre-20-ms energy gate suppresses short pauses;
requiring energy in both halves makes decisions conservative at note boundaries.
Separation bleed, multiple singers, breathiness and rapid note transitions remain
known limitations. We never automatically fold an actual octave difference.
"""

from __future__ import annotations

import math
from bisect import bisect_left
from typing import Callable

import numpy as np

ANALYSIS_RATE = 16000
WINDOW_SAMPLES = 2048  # 128 ms, sufficiently long for several 60 Hz periods.
HOP_SAMPLES = 320  # 20 ms; frame times always identify the window centre.
MIN_FREQUENCY = 60.0
MAX_FREQUENCY = 1100.0
MIN_CONFIDENCE = 0.85
MIN_RMS = 0.005  # About -46 dBFS; quiet input is deliberately left unscored.
NOTE_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")


class AnalysisCancelled(RuntimeError):
    """Raised when the caller requests cancellation between frames."""


def frequency_details(frequency: float) -> tuple[float, str, float]:
    """Return fractional MIDI, nearest equal-tempered note, and signed cents."""
    if not math.isfinite(frequency) or frequency <= 0:
        raise ValueError("频率必须是正数。")
    midi = 69.0 + 12.0 * math.log2(frequency / 440.0)
    nearest = math.floor(midi + 0.5)
    return midi, f"{NOTE_NAMES[nearest % 12]}{nearest // 12 - 1}", 100.0 * (midi - nearest)


def _mono_float(audio: np.ndarray) -> np.ndarray:
    data = np.asarray(audio)
    if data.ndim not in (1, 2):
        raise ValueError("音频必须为单声道数组或 samples-by-channels 数组。")
    if data.ndim == 2 and (data.shape[1] < 1 or data.shape[1] > 32):
        raise ValueError("声道轴应为第二维（samples-by-channels）。")
    if np.issubdtype(data.dtype, np.unsignedinteger):
        info = np.iinfo(data.dtype)
        midpoint = (float(info.max) + 1.0) / 2.0
        data = (data.astype(np.float64) - midpoint) / midpoint
    elif np.issubdtype(data.dtype, np.signedinteger):
        info = np.iinfo(data.dtype)
        data = data.astype(np.float64) / max(abs(int(info.min)), int(info.max))
    else:
        data = data.astype(np.float64)
    if not np.all(np.isfinite(data)):
        raise ValueError("音频包含非有限数值。")
    # Averaging is a standard downmix, not source separation. Opposite-phase
    # channels can cancel; callers should provide the actual separated vocal.
    return np.mean(data, axis=1) if data.ndim == 2 else data


def _estimate_window(window: np.ndarray) -> tuple[float | None, float, float]:
    signal = window - np.mean(window)
    rms = float(np.sqrt(np.mean(signal * signal)))
    centre = len(window) // 2
    centre_signal = window[centre - HOP_SAMPLES // 2 : centre + HOP_SAMPLES // 2]
    # Gate the timestamp's own 20 ms interval independently of the 128 ms
    # periodicity window. Otherwise notes on either side of an 80-100 ms pause
    # can yield a confident frequency even when the frame centre is silent.
    # Remove the centre interval's own DC, not the entire window's DC: the
    # latter can turn an all-zero centre into apparent nonzero energy.
    centre_signal = centre_signal - np.mean(centre_signal)
    centre_rms = float(np.sqrt(np.mean(centre_signal * centre_signal)))
    # Each half must contain energy as well. A centre 4 ms into a pause should
    # not be classified from the note still present in the preceding 6 ms.
    # This deliberately rejects some low-level onset/offset frames, rather
    # than attributing neighbouring periodic sound to a silent timestamp.
    half_rms = [float(np.sqrt(np.mean((part - np.mean(part)) ** 2)))
                for part in np.split(centre_signal, 2)]
    if rms < MIN_RMS or centre_rms < MIN_RMS or min(half_rms) < MIN_RMS:
        return None, 0.0, rms
    n = len(signal)
    min_lag = max(2, int(ANALYSIS_RATE / MAX_FREQUENCY) - 1)
    max_lag = min(n // 2 - 1, math.ceil(ANALYSIS_RATE / MIN_FREQUENCY) + 2)
    fft_size = 1 << (2 * n - 1).bit_length()
    spectrum = np.fft.rfft(signal, fft_size)
    correlation = np.fft.irfft(spectrum * np.conjugate(spectrum), fft_size)[: max_lag + 1]
    energy = np.concatenate(([0.0], np.cumsum(signal * signal)))
    lags = np.arange(max_lag + 1)
    difference = energy[n - lags] + energy[n] - energy[lags] - 2.0 * correlation
    # Correct for decreasing overlap. A plain autocorrelation maximum alone
    # would favour strong harmonics and is not a pitch estimate.
    difference = np.maximum(difference, 0.0) / (n - lags)
    cmnd = np.ones(max_lag + 1, dtype=np.float64)
    denominator = np.cumsum(difference[1:])
    cmnd[1:] = difference[1:] * np.arange(1, max_lag + 1) / np.maximum(denominator, 1e-15)
    minima = [
        lag for lag in range(min_lag, max_lag)
        if cmnd[lag] <= cmnd[lag - 1] and cmnd[lag] < cmnd[lag + 1]
    ]
    if not minima:
        return None, 0.0, rms
    # Compare interpolated valley depths, not integer samples: a 1 kHz period
    # has only ~16 samples and its sampled minimum can otherwise look worse
    # than the next (subharmonic) period solely because of quantization.
    valley_values = {}
    valley_shifts = {}
    for candidate in minima:
        left, centre, right = cmnd[candidate - 1 : candidate + 2]
        curvature = left - 2.0 * centre + right
        shift = float(0.5 * (left - right) / curvature) if curvature > 1e-15 else 0.0
        shift = max(-0.5, min(0.5, shift))
        valley_shifts[candidate] = shift
        valley_values[candidate] = max(0.0, float(centre - 0.5 * curvature * shift * shift))
    best_value = min(valley_values.values())
    confidence = max(0.0, min(1.0, 1.0 - best_value))
    if confidence < MIN_CONFIDENCE:
        return None, confidence, rms
    # Use the earliest valley close to the global periodicity minimum. This
    # rejects an early half-period valley when a strong second harmonic masks
    # a weaker fundamental, while choosing the first genuine period for pure
    # high tones instead of an arbitrary 2x/3x longer subharmonic. Extremely
    # weak/absent odd harmonics can make two octaves physically ambiguous.
    near_best = min(1.0 - MIN_CONFIDENCE, best_value + 0.004 + 0.12 * best_value)
    lag = next(lag for lag in minima if valley_values[lag] <= near_best)
    refined_lag = lag + valley_shifts[lag]
    frequency = ANALYSIS_RATE / refined_lag
    if not MIN_FREQUENCY * 0.99 <= frequency <= MAX_FREQUENCY * 1.01:
        return None, confidence, rms
    # Interpolation errors at the exact search boundaries should not create
    # phantom out-of-range values; this clamp affects only the 1% edge margin.
    frequency = min(MAX_FREQUENCY, max(MIN_FREQUENCY, frequency))
    return frequency, float(max(0.0, min(1.0, 1.0 - valley_values[lag]))), rms


def analyze_pitch(
    audio: np.ndarray,
    sample_rate: int,
    cancel: Callable[[], bool] | None = None,
    progress: Callable[[float], None] | None = None,
) -> list[dict]:
    """Analyze mono or samples-by-channels PCM; emit one frame every 20 ms.

    Integer PCM is normalized; floating point PCM is expected in [-1, 1].
    SciPy's polyphase resampler supplies anti-aliasing when sample rates differ.
    Normal frame times start at 64 ms and end before the trailing half-window.
    Clips shorter than 128 ms receive one zero-padded, duration-centred window.
    A centre-20-ms energy gate suppresses pauses; output rms remains the full
    window RMS for diagnostics, so a null frame may still have nonzero rms.
    `progress` receives fractions in [0, 1]; `cancel` is checked every frame.
    """
    if isinstance(sample_rate, bool) or not isinstance(sample_rate, (int, np.integer)) or sample_rate < 4000:
        raise ValueError("采样率必须为不小于 4000 Hz 的整数。")
    if cancel and cancel():
        raise AnalysisCancelled("音高分析已取消。")
    signal = _mono_float(audio)
    if progress:
        progress(0.0)
    if not len(signal):
        if progress:
            progress(1.0)
        return []
    if sample_rate != ANALYSIS_RATE:
        from scipy.signal import resample_poly

        gcd = math.gcd(int(sample_rate), ANALYSIS_RATE)
        signal = resample_poly(signal, ANALYSIS_RATE // gcd, int(sample_rate) // gcd)
    short_clip = len(signal) < WINDOW_SAMPLES
    if short_clip:
        duration = len(signal) / ANALYSIS_RATE
        pad = WINDOW_SAMPLES - len(signal)
        signal = np.pad(signal, (pad // 2, pad - pad // 2))
        starts = [0]
    else:
        starts = range(0, len(signal) - WINDOW_SAMPLES + 1, HOP_SAMPLES)
    frames = []
    for index, start in enumerate(starts):
        if cancel and cancel():
            raise AnalysisCancelled("音高分析已取消。")
        frequency, confidence, rms = _estimate_window(signal[start : start + WINDOW_SAMPLES])
        time = duration / 2.0 if short_clip else (start + WINDOW_SAMPLES / 2.0) / ANALYSIS_RATE
        midi, note, cents = frequency_details(frequency) if frequency else (None, None, None)
        frames.append({
            "time": round(time, 6),
            "frequency": round(frequency, 4) if frequency is not None else None,
            "midi": round(midi, 6) if midi is not None else None,
            "note": note,
            "cents": round(cents, 3) if cents is not None else None,
            "confidence": round(confidence, 6),
            "rms": round(rms, 7),
        })
        if progress and (index % 25 == 0 or index == len(starts) - 1):
            progress((index + 1) / len(starts))
    return frames


def _valid_frequency(frame: dict) -> float | None:
    frequency = frame.get("frequency")
    confidence = frame.get("confidence", 0.0)
    rms = frame.get("rms", 0.0)
    if any(isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value)
           for value in (frequency, confidence, rms)):
        return None
    if frequency <= 0 or confidence < MIN_CONFIDENCE or rms < MIN_RMS:
        return None
    return float(frequency)


def compare_tracks(
    referenceFrames: list[dict],
    recordingFrames: list[dict],
    offset_seconds: float = 0.0,
    transpose: float = 0.0,
) -> dict:
    """Compare isolated vocal tracks without octave folding or overall scoring.

    recording time t maps to reference time t + offset_seconds. Positive
    transpose raises the expected reference pitch by that many semitones.
    Invalid/low-confidence/silent frames never count as correct notes. Coverage
    is comparable frames / aligned voiced reference frames, separate from pitch
    accuracy. Gaps above 60 ms and boundaries are not interpolated across.
    """
    if not math.isfinite(offset_seconds) or not math.isfinite(transpose):
        raise ValueError("时间偏移和移调必须为有限数值。")
    reference = sorted(referenceFrames, key=lambda frame: frame["time"])
    recordings = sorted(recordingFrames, key=lambda frame: frame["time"])
    times = [float(frame["time"]) for frame in reference]
    if any(not math.isfinite(float(frame["time"])) for frame in reference + recordings):
        raise ValueError("轨迹包含无效时间。")
    positive_steps = np.diff(times)
    positive_steps = positive_steps[positive_steps > 0]
    step = float(np.median(positive_steps)) if len(positive_steps) else HOP_SAMPLES / ANALYSIS_RATE
    tolerance = min(0.03, max(0.005, step * 0.55))
    max_gap = min(0.06, max(0.022, step * 1.6))
    reference_valid = [_valid_frequency(frame) for frame in reference]
    points: list[dict] = []
    deviations = []
    aligned_voiced = 0
    recording_voiced = 0

    for frame in recordings:
        time = float(frame["time"])
        target = time + offset_seconds
        observed = _valid_frequency(frame)
        recording_voiced += int(observed is not None)
        point = {"time": time, "reference_time": round(target, 6), "cents": None,
                 "reference_frequency": None, "recording_frequency": observed, "status": "out_of_range"}
        if not times or target < times[0] - tolerance or target > times[-1] + tolerance:
            points.append(point)
            continue
        index = bisect_left(times, target)
        candidates = [candidate for candidate in (index - 1, index) if 0 <= candidate < len(times)]
        closest = min(candidates, key=lambda candidate: abs(times[candidate] - target))
        expected = reference_valid[closest]
        if abs(times[closest] - target) > tolerance:
            point["status"] = "reference_gap"
            points.append(point)
            continue
        if expected is None:
            point["status"] = "reference_unvoiced"
            points.append(point)
            continue
        # Linear interpolation in log frequency avoids rounding timing offsets
        # to an entire analysis hop; never interpolate a silence or large jump.
        if 0 < index < len(times):
            left, right = reference_valid[index - 1], reference_valid[index]
            gap = times[index] - times[index - 1]
            if left and right and 0 < gap <= max_gap and abs(12 * math.log2(right / left)) < 3.0:
                fraction = (target - times[index - 1]) / gap
                expected = left * (right / left) ** fraction
        expected *= 2.0 ** (transpose / 12.0)
        point["reference_frequency"] = round(expected, 4)
        aligned_voiced += 1
        if observed is None:
            point["status"] = "recording_unvoiced"
            points.append(point)
            continue
        cents = 1200.0 * math.log2(observed / expected)
        deviations.append(cents)
        point.update(cents=round(cents, 3), status="comparable")
        points.append(point)

    count = len(deviations)
    summary = {
        "reference_total_frames": len(reference),
        "reference_voiced_frames": sum(value is not None for value in reference_valid),
        "recording_total_frames": len(recordings),
        "recording_voiced_frames": recording_voiced,
        "aligned_reference_voiced_frames": aligned_voiced,
        "comparable_frames": count,
        "voiced_coverage": count / aligned_voiced if aligned_voiced else 0.0,
        "comparable_ratio": count / len(recordings) if recordings else 0.0,
        "recording_voiced_ratio": recording_voiced / len(recordings) if recordings else 0.0,
        "median_abs_cents": round(float(np.median(np.abs(deviations))), 3) if count else None,
        "within_50_cents_ratio": sum(abs(value) <= 50.0 + 1e-8 for value in deviations) / count if count else None,
        "mean_signed_cents": round(float(np.mean(deviations)), 3) if count else None,
        "median_signed_cents": round(float(np.median(deviations)), 3) if count else None,
        "offset_seconds": offset_seconds,
        "transpose": transpose,
    }
    return {"summary": summary, "points": points}
