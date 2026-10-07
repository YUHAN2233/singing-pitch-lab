/* Local monophonic pitch analysis. No reference-guided octave correction. */
export const MIN_CONFIDENCE = 0.85;
// Browser microphones with automatic gain disabled can deliver quiet sustained
// singing below -46 dBFS. Periodicity remains the deciding voice gate.
export const MIN_RMS = 0.0015;

export function frequencyToMidi(frequency) {
  return frequency > 0 ? 69 + 12 * Math.log2(frequency / 440) : null;
}

export function midiToFrequency(midi) {
  return 440 * 2 ** ((midi - 69) / 12);
}

export function describePitch(frequency) {
  if (!(frequency > 0)) return { frequency: null, midi: null, note: null, cents: null };
  const midi = frequencyToMidi(frequency);
  const rounded = Math.round(midi);
  const names = ['C', 'C♯', 'D', 'D♯', 'E', 'F', 'F♯', 'G', 'G♯', 'A', 'A♯', 'B'];
  return { frequency, midi, note: names[((rounded % 12) + 12) % 12] + (Math.floor(rounded / 12) - 1), cents: (midi - rounded) * 100 };
}

export function detectPitch(input, inputRate, { minFrequency = 60, maxFrequency = 1100, minRms = MIN_RMS, minConfidence = MIN_CONFIDENCE } = {}) {
  if (!input?.length || !(inputRate > 0)) return { ...describePitch(null), confidence: 0, rms: 0, reason: 'quiet' };
  // Average before integer decimation; keep the expensive YIN difference loop off
  // the audio rendering thread. A 4096-sample input contains low singing notes.
  const stride = Math.max(1, Math.floor(inputRate / 16000));
  const sampleRate = inputRate / stride;
  const n = Math.floor(input.length / stride);
  const samples = new Float64Array(n);
  let mean = 0;
  for (let i = 0; i < n; i++) {
    let sum = 0;
    for (let j = 0; j < stride; j++) sum += input[i * stride + j];
    samples[i] = sum / stride;
    mean += samples[i];
  }
  mean /= n;
  let energy = 0;
  for (let i = 0; i < n; i++) { samples[i] -= mean; energy += samples[i] ** 2; }
  const rms = Math.sqrt(energy / n);
  const empty = (reason, confidence = 0) => ({ ...describePitch(null), confidence, rms, reason });
  if (!Number.isFinite(rms) || rms < minRms) return empty('quiet');
  // A long YIN window may straddle a sung note and a short rest. Gate the
  // timestamp's centre, not only the full window, so adjacent notes cannot
  // manufacture a scored pitch inside the rest. Match the offline analyzer.
  const halfGate = Math.max(1, Math.round(inputRate * .01));
  const gateStart = Math.max(0, Math.floor(input.length / 2) - halfGate);
  for (let half = 0; half < 2; half++) {
    const start = gateStart + half * halfGate, end = Math.min(input.length, start + halfGate);
    let sum = 0, squared = 0;
    for (let i = start; i < end; i++) { sum += input[i]; squared += input[i] ** 2; }
    const length = end - start;
    const gateRms = Math.sqrt(Math.max(0, squared / length - (sum / length) ** 2));
    if (!Number.isFinite(gateRms) || gateRms < minRms) return empty('quiet');
  }
  const minTau = Math.max(2, Math.floor(sampleRate / maxFrequency) - 1);
  const maxTau = Math.min(Math.ceil(sampleRate / minFrequency) + 2, Math.floor(n / 2) - 1);
  if (maxTau <= minTau) return empty('out_of_range');
  const difference = new Float64Array(maxTau + 1);
  const count = n - maxTau;
  for (let tau = 1; tau <= maxTau; tau++) {
    let sum = 0;
    for (let i = 0; i < count; i++) sum += (samples[i] - samples[i + tau]) ** 2;
    difference[tau] = sum;
  }
  let running = 0;
  difference[0] = 1;
  for (let tau = 1; tau <= maxTau; tau++) {
    running += difference[tau];
    difference[tau] = running > 0 ? difference[tau] * tau / running : 1;
  }
  const valleys = [];
  for (let tau = minTau; tau < maxTau; tau++) {
    const left = difference[tau - 1], center = difference[tau], right = difference[tau + 1];
    if (center > left || center >= right) continue;
    const curvature = left - 2 * center + right;
    const shift = curvature > 1e-15 ? Math.max(-.5, Math.min(.5, .5 * (left - right) / curvature)) : 0;
    valleys.push({ tau, shift, value: Math.max(0, center - .5 * curvature * shift * shift) });
  }
  if (!valleys.length) return empty('unstable');
  // Compare interpolated valley depths. Integer-only minima can prefer a
  // subharmonic on high notes; the first threshold crossing can pick a strong
  // second harmonic. Choose the earliest valley near the global minimum.
  const best = Math.min(...valleys.map(valley => valley.value));
  const confidence = Math.max(0, Math.min(1, 1 - best));
  if (confidence < minConfidence) return empty('low_confidence', confidence);
  const tolerance = Math.min(1 - minConfidence, best + .004 + .12 * best);
  const chosen = valleys.find(valley => valley.value <= tolerance);
  const frequency = sampleRate / (chosen.tau + chosen.shift);
  if (frequency < minFrequency * .99 || frequency > maxFrequency * 1.01) return empty('out_of_range', confidence);
  return { ...describePitch(Math.max(minFrequency, Math.min(maxFrequency, frequency))), confidence: Math.max(0, 1 - chosen.value), rms };
}

export function isVoiced(frame) {
  return Boolean(frame && Number.isFinite(frame.midi) && Number.isFinite(frame.frequency) && frame.frequency > 0 && frame.confidence >= MIN_CONFIDENCE && frame.rms >= MIN_RMS);
}

// Presentation only: never pass the returned previous pitch into scoring.
export function displayPitch(frame, lastVoiced, phase = 'active') {
  if (isVoiced(frame)) return { shown: frame, state: 'current' };
  if (frame?.reason === 'quiet') return { shown: isVoiced(lastVoiced) ? lastVoiced : null, state: 'quiet' };
  if (!frame && phase === 'idle') return { shown: isVoiced(lastVoiced) ? lastVoiced : null, state: 'stopped' };
  if (frame) return { shown: null, state: 'unstable' };
  return { shown: null, state: 'waiting' };
}

export function frameAt(frames, time, maxDistance = 0.011) {
  if (!frames?.length || !Number.isFinite(time) || time < 0) return null;
  let lo = 0, hi = frames.length;
  while (lo < hi) { const mid = (lo + hi) >>> 1; if (frames[mid].time < time) lo = mid + 1; else hi = mid; }
  const next = frames[lo], previous = frames[lo - 1];
  const closest = !previous ? next : !next ? previous : Math.abs(previous.time - time) <= Math.abs(next.time - time) ? previous : next;
  if (!closest || Math.abs(closest.time - time) > maxDistance) return null;
  if (previous && next && isVoiced(previous) && isVoiced(next) && next.time - previous.time <= .032 && Math.abs(next.midi - previous.midi) < 3) {
    const fraction = (time - previous.time) / (next.time - previous.time);
    const midi = previous.midi + (next.midi - previous.midi) * fraction;
    return { ...closest, ...describePitch(midiToFrequency(midi)), time };
  }
  return closest;
}

export function referenceTime(playbackTime, windowAgeSeconds, latencyMs, offsetMs) {
  return playbackTime - Math.max(0, windowAgeSeconds) - latencyMs / 1000 + offsetMs / 1000;
}

export function pitchDifference(input, target, transpose = 0) {
  if (!isVoiced(input) || !isVoiced(target)) return null;
  // Keep signed octave errors: one octave sharp is +1200 cents, never zero.
  return 100 * (input.midi - target.midi - transpose);
}

/** Inspect the two curves at one plotted reference-clock position. */
export function timelineReadout(referenceFrames, inputFrames, time, { offsetMs = 0, transpose = 0, usingLive = false } = {}) {
  const target = frameAt(referenceFrames, time);
  const input = frameAt(inputFrames, time - (usingLive ? 0 : offsetMs / 1000));
  return { time, target, input, targetMidi: isVoiced(target) ? target.midi + transpose : null,
    difference: pitchDifference(input, target, transpose) };
}

export function summarizeComparisons(comparisons, eligible = comparisons.length) {
  const values = comparisons.filter(Number.isFinite);
  const absolute = values.map(Math.abs).sort((a, b) => a - b);
  const count = values.length;
  return {
    count, eligible, coverage: eligible ? count / eligible : 0,
    meanAbsoluteCents: count ? absolute.reduce((a, b) => a + b, 0) / count : null,
    medianAbsoluteCents: count ? (absolute[Math.floor((count - 1) / 2)] + absolute[Math.floor(count / 2)]) / 2 : null,
    meanSignedCents: count ? values.reduce((a, b) => a + b, 0) / count : null,
    within50: count ? values.filter(v => Math.abs(v) <= 50).length / count : null,
  };
}

export function compareRecording(reference, recording, { offsetMs = 0, transpose = 0 } = {}) {
  const differences = [];
  let eligible = 0;
  for (const input of recording ?? []) {
    const target = frameAt(reference, input.time + offsetMs / 1000);
    if (!isVoiced(target)) continue;
    eligible++;
    const difference = pitchDifference(input, target, transpose);
    if (difference !== null) differences.push(difference);
  }
  return summarizeComparisons(differences, eligible);
}

// One observation per reference time bucket. Seeking starts a fresh pass, so
// replaying the same passage cannot silently inflate the previous pass's score.
export class LiveScore {
  constructor() { this.reset(); }
  reset() { this.buckets = new Map(); this.lastTime = null; }
  add(time, input, target, transpose = 0, playing = true) {
    if (!playing || !isVoiced(target) || time < 0) return false;
    if (this.lastTime !== null && time < this.lastTime - 0.2) this.reset();
    this.lastTime = time;
    const bucket = Math.floor(time * 20);
    if (this.buckets.has(bucket)) return false;
    this.buckets.set(bucket, pitchDifference(input, target, transpose));
    return true;
  }
  summary() { return summarizeComparisons([...this.buckets.values()], this.buckets.size); }
}
