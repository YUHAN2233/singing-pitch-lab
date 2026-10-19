"""Quantitative tests with generated two-voice ground truth; no commercial audio."""
import math
import time
import unittest

import numpy as np

from backend.duet import analyze_two_voices

RATE = 16000
DURATION = 1.5
T = np.arange(int(RATE*DURATION)) / RATE


def voice(frequency, level=.2):
    if np.isscalar(frequency):
        phase = 2*np.pi*float(frequency)*T
    else:
        phase = 2*np.pi*np.cumsum(frequency)/RATE
    return sum(level/k*np.sin(k*phase) for k in range(1, 6))


def metrics(frames, expected):
    pairs = [f for f in frames if f['voice_count'] == 2]
    errors = [abs(1200*math.log2(f[key]['frequency']/target(f['time'])))
              for f in pairs for key,target in zip(('lower','higher'),expected)]
    return len(pairs)/len(frames), float(np.median(errors)) if errors else None


class DuetPitchTests(unittest.TestCase):
    def test_silence(self):
        self.assertTrue(all(f['voice_count']==0 for f in analyze_two_voices(np.zeros_like(T),RATE)))

    def test_single_harmonic_voice_is_not_two(self):
        signal = voice(220) + .09*np.sin(4*np.pi*220*T)
        frames = analyze_two_voices(signal,RATE)
        self.assertGreater(sum(f['voice_count']==1 for f in frames)/len(frames),.9)
        self.assertFalse(any(f['voice_count']==2 for f in frames))

    def test_distinct_overlapping_voices(self):
        frames=analyze_two_voices(voice(220)+voice(330,.16),RATE)
        coverage,error=metrics(frames,(lambda t:220,lambda t:330))
        self.assertGreater(coverage,.85)
        self.assertLess(error,20)

    def test_unequal_levels(self):
        frames=analyze_two_voices(voice(220)+voice(330,.06),RATE)
        coverage,error=metrics(frames,(lambda t:220,lambda t:330))
        self.assertGreater(coverage,.7)
        self.assertLess(error,25)

    def test_unison_is_ambiguous(self):
        frames=analyze_two_voices(voice(220)+voice(220,.1),RATE)
        self.assertEqual(sum(f['voice_count']==2 for f in frames),0)

    def test_silent_gap_is_not_filled_from_neighbors(self):
        signal=voice(220)+voice(330,.16)
        signal[(T>=.5)&(T<.85)]=0
        frames=analyze_two_voices(signal,RATE)
        middle=[f for f in frames if .58<=f['time']<=.77]
        self.assertTrue(middle)
        self.assertTrue(all(f['voice_count']==0 for f in middle))

    def test_crossing_reports_pitch_slots_not_identity(self):
        low_to_high=220+140*T/DURATION
        high_to_low=360-140*T/DURATION
        frames=analyze_two_voices(voice(low_to_high)+voice(high_to_low,.17),RATE)
        outside=[f for f in frames if abs(f['time']-DURATION/2)>.25]
        good=[f for f in outside if f['voice_count']==2 and
              abs(1200*math.log2(f['lower']['frequency']/min(220+140*f['time']/DURATION,360-140*f['time']/DURATION)))<50 and
              abs(1200*math.log2(f['higher']['frequency']/max(220+140*f['time']/DURATION,360-140*f['time']/DURATION)))<50]
        self.assertGreater(len(good)/len(outside),.55)


if __name__ == '__main__':
    started=time.perf_counter()
    unittest.main()
