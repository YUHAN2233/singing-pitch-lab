import test from 'node:test';
import assert from 'node:assert/strict';
import { detectPitch, describePitch, frequencyToMidi, midiToFrequency, isVoiced, displayPitch, frameAt, pitchDifference, timelineReadout, referenceTime, compareRecording, LiveScore } from '../web/pitch.js';
import { JobSlot, MicrophoneController, microphoneError, abortableDelay } from '../web/app.js';
import { PitchHistory } from '../web/history.js';

const tone = (frequency, sampleRate = 48000, amplitude = .3, count = 4096) => Float32Array.from({ length: count }, (_, i) => amplitude * Math.sin(2 * Math.PI * frequency * i / sampleRate));
const frame = (frequency, time = 0, confidence = .99, rms = .2) => ({ ...describePitch(frequency), time, confidence, rms });
const merge = (...signals) => Float32Array.from(signals[0], (_, i) => signals.reduce((sum, signal) => sum + signal[i], 0));
const centsFrom = (frequency, target) => 1200 * Math.log2(frequency / target);
const json = (body, status = 200) => new Response(JSON.stringify(body), { status, headers: { 'content-type': 'application/json' } });
const deferred = () => { let resolve, reject; const promise = new Promise((yes, no) => { resolve = yes; reject = no; }); return { promise, resolve, reject }; };
const tick = () => new Promise(resolve => setImmediate(resolve));

test('pitch history preserves voiced sections, silence breaks, and stop breaks', () => {
  const history = new PitchHistory(100, .1);
  history.add(0, frame(440)); history.add(.05, frame(440)); history.add(.12, frame(440));
  history.add(.15, frame(null)); history.add(.18, frame(440)); history.break(.3);
  assert.deepEqual(history.frames.map(item => item.midi === null), [false, false, true, false, true]);
  assert.equal(history.frames[0].time, 0);
  assert.equal(history.latest, .3);
  history.clear();
  assert.equal(history.frames.length, 0);
  assert.equal(history.latest, 0);
});

test('pitch history keeps a bounded window during long practice and clamps review range', () => {
  const history = new PitchHistory(18000, .1);
  for (let i = 0; i < 25000; i++) history.add(i * .11, frame(440));
  assert.ok(history.frames.length <= 18000 && history.frames.length > 15000);
  assert.ok(history.oldest >= history.latest - 1800 && history.latest > 2700);
  assert.ok(history.visible(history.latest - 20, history.latest).length < 200);
});

test('history cursor snaps only to nearby observations and reports silent gaps', () => {
  const history = new PitchHistory(100, .1);
  history.add(0, frame(440));
  history.add(.2, { ...frame(null), reason: 'quiet' });
  history.add(.4, frame(523.25));
  assert.equal(history.nearest(.04).midi, 69);
  assert.equal(history.nearest(.2).reason, 'quiet');
  assert.equal(history.nearest(.4).midi > 71, true);
  assert.equal(history.nearest(.65), null);
  assert.equal(history.nearest(-1), null);
  history.clear();
  assert.equal(history.nearest(0), null);
});

test('timeline cursor uses plotted time shift and transpose without borrowing distant notes', () => {
  const reference = [frame(440, .1), { ...frame(null, .2), reason: 'quiet' }];
  const recording = [frame(466.164, 0), frame(466.164, .1)];
  const aligned = timelineReadout(reference, recording, .1, { offsetMs: 100 });
  assert.equal(aligned.target.note, 'A4');
  assert.equal(Math.round(aligned.input.midi), 70);
  assert.ok(Math.abs(aligned.difference - 100) < 1);
  assert.ok(Math.abs(timelineReadout(reference, recording, .1, { offsetMs: 100, transpose: 12 }).difference + 1100) < 1);
  assert.equal(timelineReadout(reference, recording, .2).target.frequency, null);
  const gap = timelineReadout(reference, recording, .35, { offsetMs: 100 });
  assert.equal(gap.target, null);
  assert.equal(gap.input, null);
  assert.equal(gap.difference, null);
  assert.equal(timelineReadout(reference, [], .1).input, null);
});

test('silent live display retains only the last valid note without changing score input', () => {
  const previous = frame(440);
  const quiet = { ...frame(null), reason: 'quiet' };
  const unstable = { ...frame(null), reason: 'low_confidence', confidence: .3 };
  assert.deepEqual(displayPitch(quiet, null), { shown: null, state: 'quiet' });
  assert.equal(displayPitch(previous, null).state, 'current');
  assert.equal(displayPitch(quiet, previous).shown.note, 'A4');
  assert.deepEqual(displayPitch(unstable, previous), { shown: null, state: 'unstable' });
  assert.equal(displayPitch(null, previous, 'idle').shown.note, 'A4');
  assert.equal(displayPitch(null, null, 'idle').shown, null);
  assert.equal(pitchDifference(quiet, previous), null);
});

test('note statistics count stable sung segments, silence repeats and genuine note changes', () => {
  const history = new PitchHistory();
  for (const time of [0, .13, .26, .39, .52]) history.add(time, frame(440));
  assert.equal(history.summary().counts.get(69), 1);
  history.add(.65, { ...frame(null), reason: 'quiet' });
  for (const time of [.8, .93, 1.06]) history.add(time, frame(440));
  assert.equal(history.summary().counts.get(69), 2);
  for (const time of [1.2, 1.33, 1.46, 1.59]) history.add(time, frame(587.33));
  const summary = history.summary();
  assert.equal(summary.highest, 74);
  assert.equal(summary.lowest, 69);
  assert.equal(summary.counts.get(74), 1);
  history.break(1.7);
  assert.equal(history.summary().counts.get(74), 1);
  history.clear();
  assert.equal(history.summary().highest, null);
  assert.equal(history.summary().counts.size, 0);
});

test('vibrato at a semitone edge and one-frame peaks do not inflate note counts', () => {
  const history = new PitchHistory();
  for (const [time, midi] of [[0,69],[.13,69],[.26,69],[.39,69.52],[.52,69.48],[.65,69],[.78,81],[.91,69],[1.04,69]]) {
    history.add(time, frame(midiToFrequency(midi)));
  }
  assert.deepEqual([...history.summary().counts], [[69, 1]]);
  const short = new PitchHistory();
  short.add(0, frame(880)); short.add(.13, frame(880));
  assert.equal(short.summary().highest, null);
});

test('YIN detects 60–1100 Hz at common browser sample rates without high-note octave errors', () => {
  for (const rate of [16000, 44100, 48000]) {
    for (const frequency of [60, 65.406, 110, 220, 261.626, 440, 659.255, 880, 1000, 1046.502, 1100]) {
      const result = detectPitch(tone(frequency, rate), rate);
      assert.ok(isVoiced(result), `${frequency} Hz at ${rate} Hz should be voiced`);
      assert.ok(Math.abs(centsFrom(result.frequency, frequency)) < 8, `${frequency} Hz @ ${rate}: got ${result.frequency}`);
    }
  }
});

test('YIN retains a weak fundamental beneath a stronger second harmonic', () => {
  for (const frequency of [110, 220, 440]) {
    const result = detectPitch(merge(tone(frequency, 48000, .1), tone(frequency * 2, 48000, .75)), 48000);
    assert.ok(Math.abs(centsFrom(result.frequency, frequency)) < 8, JSON.stringify(result));
  }
});

test('YIN detects a missing fundamental from second and third harmonics', () => {
  const result = detectPitch(merge(tone(440, 48000, .5), tone(660, 48000, .35)), 48000);
  assert.ok(Math.abs(centsFrom(result.frequency, 220)) < 8);
});

test('silence, DC, very quiet input, and invalid input do not produce notes', () => {
  for (const input of [new Float32Array(4096), new Float32Array(4096).fill(.5), tone(440, 48000, .0001), new Float32Array(4096).fill(NaN), new Float32Array(0)]) {
    const result = detectPitch(input, 48000);
    assert.equal(result.frequency, null); assert.equal(result.note, null); assert.equal(isVoiced(result), false);
  }
});

test('quiet sustained singing remains voiced while weaker input remains quiet', () => {
  for (const frequency of [110, 220, 440]) {
    const result = detectPitch(tone(frequency, 48000, .003), 48000);
    assert.ok(isVoiced(result), JSON.stringify(result));
    assert.ok(Math.abs(centsFrom(result.frequency, frequency)) < 8, JSON.stringify(result));
  }
  assert.equal(detectPitch(tone(440, 48000, .0001), 48000).reason, 'quiet');
});

test('deterministic broadband noise has insufficient periodicity confidence', () => {
  let seed = 20261005;
  const noise = Float32Array.from({ length: 4096 }, () => { seed = (1664525 * seed + 1013904223) >>> 0; return (seed / 2 ** 32 - .5) * .6; });
  const result = detectPitch(noise, 48000);
  assert.equal(result.frequency, null); assert.ok(result.confidence < .85);
});

test('short central rests and one silent centre half cannot borrow pitch from nearby notes', () => {
  for (const restSamples of [960, 3840]) {
    const signal = tone(440);
    const start = Math.floor((signal.length - restSamples) / 2);
    signal.fill(0, start, start + restSamples);
    const result = detectPitch(signal, 48000);
    assert.equal(result.frequency, null); assert.equal(result.confidence, 0);
    assert.ok(result.rms > .005, 'full-window input meter should retain adjacent note energy');
  }
  const boundary = tone(440);
  boundary.fill(0, Math.floor(boundary.length / 2));
  assert.equal(detectPitch(boundary, 48000).frequency, null);
});

test('note names, signed cents and frequency conversions preserve tuning', () => {
  assert.equal(describePitch(440).note, 'A4'); assert.equal(describePitch(261.625565).note, 'C4');
  assert.ok(Math.abs(describePitch(440 * 2 ** (.35 / 12)).cents - 35) < 1e-8);
  assert.ok(Math.abs(describePitch(440 * 2 ** (-.35 / 12)).cents + 35) < 1e-8);
  assert.equal(midiToFrequency(69), 440); assert.equal(frequencyToMidi(880), 81);
});

test('one octave difference remains ±1200 cents rather than being folded', () => {
  assert.equal(pitchDifference(frame(880), frame(440)), 1200);
  assert.equal(pitchDifference(frame(220), frame(440)), -1200);
  assert.equal(pitchDifference(frame(null), frame(440)), null);
  assert.equal(pitchDifference(frame(440, 0, .84), frame(440)), null);
});

test('positive and negative target transposition have explicit semantics', () => {
  assert.ok(Math.abs(pitchDifference(frame(midiToFrequency(71)), frame(440), 2)) < 1e-8);
  assert.ok(Math.abs(pitchDifference(frame(midiToFrequency(67)), frame(440), -2)) < 1e-8);
  assert.equal(pitchDifference(frame(440), frame(440), 12), -1200);
});

test('reference timing subtracts window age and latency, adds positive offset', () => {
  assert.ok(Math.abs(referenceTime(4, .04, 80, 250) - 4.13) < 1e-12);
  assert.ok(Math.abs(referenceTime(4, .04, 80, -250) - 3.63) < 1e-12);
  assert.equal(referenceTime(0, .04, 80, 0), -.12);
});

test('nearest frame uses a tight 20 ms hop tolerance and rejects gaps or distant boundaries', () => {
  const frames = [frame(440, .064), frame(440, .084), frame(440, .104), frame(440, .3)];
  assert.equal(frameAt(frames, -.001), null);
  assert.equal(frameAt(frames, .04), null);
  assert.equal(frameAt(frames, .2), null);
  assert.equal(frameAt(frames, .312), null);
  assert.ok(frameAt(frames, .071));
});

test('reference interpolation never bridges silence or a large note jump', () => {
  const silent = frame(null, .084, 0, 0);
  assert.equal(frameAt([frame(440, .064), silent, frame(440, .104)], .084).frequency, null);
  const glide = frameAt([frame(440, .064), frame(midiToFrequency(70), .084)], .074);
  assert.ok(Math.abs(glide.midi - 69.5) < 1e-8);
  const jump = frameAt([frame(440, .064), frame(880, .084)], .074);
  assert.ok(jump.midi === 69 || jump.midi === 81);
});

test('offline comparison aligns offsets and excludes silence from accuracy but counts coverage', () => {
  const reference = [frame(440, .064), frame(440, .084), frame(440, .104)];
  const recording = [frame(440, .264), frame(null, .284, 0, 0), frame(440 * 2 ** (35 / 1200), .304)];
  const unmatched = compareRecording(reference, recording);
  assert.equal(unmatched.count, 0);
  const aligned = compareRecording(reference, recording, { offsetMs: -200 });
  assert.equal(aligned.count, 2); assert.equal(aligned.eligible, 3);
  assert.equal(aligned.coverage, 2 / 3); assert.ok(Math.abs(aligned.meanAbsoluteCents - 17.5) < 1e-8);
  assert.equal(aligned.within50, 1);
});

test('low-confidence target is never eligible for scoring', () => {
  const score = compareRecording([frame(440, .064, .3)], [frame(440, .064)]);
  assert.equal(score.count, 0); assert.equal(score.eligible, 0); assert.equal(score.meanAbsoluteCents, null);
});

test('live scoring ignores pause, deduplicates time buckets, and resets after backward seek', () => {
  const score = new LiveScore();
  assert.equal(score.add(1, frame(440), frame(440), 0, false), false);
  assert.equal(score.add(1, frame(440), frame(440)), true);
  assert.equal(score.add(1.01, frame(880), frame(440)), false);
  score.add(1.06, frame(null), frame(440));
  assert.equal(score.summary().count, 1); assert.equal(score.summary().coverage, .5);
  score.add(.5, frame(880), frame(440));
  assert.equal(score.summary().count, 1); assert.equal(score.summary().meanAbsoluteCents, 1200);
  score.reset(); assert.equal(score.summary().eligible, 0);
});

test('job upload sends raw file and encoded name then accepts actual frame results', async () => {
  const calls = [];
  const result = { id: 'a', kind: 'reference', name: '中文.wav', frames: [frame(440)], audio: {}, duration: 1, mode: 'dry' };
  const slot = new JobSlot('reference', { fetcher: async (url, options = {}) => {
    calls.push({ url, options });
    return options.method === 'POST' ? json({ id: 'a' }, 202) : json({ id: 'a', status: 'done', result, progress: 1 });
  } });
  const file = new File(['wave'], '中文.wav'); await slot.start(file, 'dry');
  assert.equal(slot.state.status, 'done'); assert.deepEqual(slot.state.result.frames, result.frames);
  assert.equal(calls[0].options.body, file); assert.equal(calls[0].options.headers['X-Filename'], encodeURIComponent('中文.wav'));
  assert.match(calls[0].url, /kind=reference&mode=dry/);
});

test('cancel while POST is pending deletes its eventual server job and never polls it', async () => {
  const post = deferred(), calls = [];
  const slot = new JobSlot('reference', { fetcher: async (url, options = {}) => {
    calls.push({ url, method: options.method });
    if (options.method === 'POST') return post.promise;
    if (options.method === 'DELETE') return json({ status: 'cancelled' });
    throw new Error('Should not poll cancelled job');
  } });
  const running = slot.start(new File(['x'], 'song.wav'), 'mixed');
  await slot.cancel(); post.resolve(json({ id: 'late' }, 202)); await running;
  assert.equal(slot.state.status, 'cancelled');
  assert.deepEqual(calls.map(call => call.method), ['POST', 'DELETE']);
  assert.equal(calls[1].url, '/api/jobs/late');
});

test('switching files rejects stale results and deletes the old job', async () => {
  const firstPoll = deferred(), calls = []; let posts = 0;
  const slot = new JobSlot('recording', { fetcher: async (url, options = {}) => {
    calls.push({ url, method: options.method });
    if (options.method === 'POST') return json({ id: `job${++posts}` }, 202);
    if (options.method === 'DELETE') return json({ status: 'cancelled' });
    if (url.endsWith('job1')) return firstPoll.promise;
    return json({ id: 'job2', status: 'done', result: { frames: [frame(220)], name: 'new.wav' } });
  } });
  const first = slot.start(new File(['1'], 'old.wav'), 'dry'); await tick();
  await slot.start(new File(['2'], 'new.wav'), 'dry');
  firstPoll.resolve(json({ id: 'job1', status: 'done', result: { frames: [frame(880)] } })); await first;
  assert.equal(slot.state.id, 'job2'); assert.equal(slot.state.result.frames[0].frequency, 220);
  assert.ok(calls.some(call => call.url.endsWith('job1') && call.method === 'DELETE'));
});

test('switching completed references preserves the previous library entry', async () => {
  const deleted = [];
  let index = 0;
  const slot = new JobSlot('reference', { fetcher: async (url, options = {}) => {
    if (options.method === 'DELETE') { deleted.push(url); return json({ status: 'cancelled' }); }
    if (options.method === 'POST') return json({ id: `song${++index}`, status: 'queued' }, 202);
    const id = url.split('/').at(-1);
    return json({ id, status: 'done', result: { kind: 'reference', mode: 'dry', name: id, frames: [] } });
  } });
  await slot.start(new File(['a'], 'one.wav'), 'dry');
  await slot.start(new File(['b'], 'two.wav'), 'dry');
  assert.equal(slot.state.id, 'song2');
  assert.deepEqual(deleted, []);
  slot.restore({ id: 'song1', status: 'done', result: { kind: 'reference', mode: 'dry', name: 'one.wav', frames: [] } });
  assert.equal(slot.state.id, 'song1');
  assert.deepEqual(deleted, []);
});

test('stale duplicate response never deletes an existing analyzed song', async () => {
  const pending = deferred(), deleted = [];
  const slot = new JobSlot('reference', { fetcher: async (url, options = {}) => {
    if (options.method === 'DELETE') { deleted.push(url); return json({ status: 'cancelled' }); }
    if (options.method === 'POST') return pending.promise;
    return json({ id: 'old', status: 'done', result: { kind: 'reference', frames: [] } });
  } });
  const upload = slot.start(new File(['a'], 'duplicate.wav'), 'dry');
  await tick(); slot.cancel({ silent: true });
  pending.resolve(json({ id: 'old', status: 'done', reused: true }, 202));
  await upload;
  assert.deepEqual(deleted, []);
});

test('server failures remain explicit and a new upload recovers successfully', async () => {
  let fail = true;
  const slot = new JobSlot('reference', { fetcher: async (_url, options = {}) => {
    if (fail) return json({ detail: '缺少真实分离模型' }, 503);
    if (options.method === 'POST') return json({ id: 'ok' }, 202);
    return json({ id: 'ok', status: 'done', result: { frames: [] } });
  } });
  await slot.start(new File(['x'], 'mix.wav'), 'mixed');
  assert.equal(slot.state.status, 'error'); assert.match(slot.state.message, /真实分离模型/);
  fail = false; await slot.start(new File(['x'], 'dry.wav'), 'dry');
  assert.equal(slot.state.status, 'done');
});

test('done without pitch data is rejected rather than presented as analyzed', async () => {
  const slot = new JobSlot('reference', { fetcher: async (_url, options = {}) => options.method === 'POST' ? json({ id: 'bad' }) : json({ id: 'bad', status: 'done', result: {} }) });
  await slot.start(new File(['x'], 'x.wav'), 'dry');
  assert.equal(slot.state.status, 'error'); assert.match(slot.state.message, /真实音高轨迹/);
});

test('HTTP 200 deletion errors retain a removable job ID and never claim success', async () => {
  const slot = new JobSlot('reference', { fetcher: async () => json({ status: 'error', message: '文件被占用，未清理完成' }) });
  slot.state = { id: 'locked', status: 'done', result: { frames: [] } };
  await slot.cancel();
  assert.equal(slot.state.id, 'locked'); assert.equal(slot.state.status, 'error'); assert.match(slot.state.message, /未清理/);
});

test('old-file deletion failure cannot overwrite a replacement upload result', async () => {
  const deletion = deferred();
  const slot = new JobSlot('reference', { fetcher: async (_url, options = {}) => {
    if (options.method === 'DELETE') return deletion.promise;
    if (options.method === 'POST') return json({ id: 'new' });
    return json({ id: 'new', status: 'done', result: { frames: [] } });
  } });
  slot.state = { id: 'old', status: 'done', result: { frames: [] } };
  await slot.start(new File(['x'], 'new.wav'), 'dry');
  deletion.resolve(json({ status: 'error', message: 'Locked old file' })); await tick();
  assert.equal(slot.state.id, 'new'); assert.equal(slot.state.status, 'done');
});

test('restoring a cache validates kind and preserves a current identical job', () => {
  let deleted = false;
  const slot = new JobSlot('reference', { fetcher: async () => { deleted = true; return json({}); } });
  const job = { id: 'cached', status: 'done', result: { kind: 'reference', frames: [], name: 'cached.wav', mode: 'dry' } };
  slot.restore(job); slot.restore(job); assert.equal(slot.state.id, 'cached'); assert.equal(deleted, false);
  assert.throws(() => slot.restore({ ...job, result: { ...job.result, kind: 'recording' } }));
});

test('poll delay aborts immediately when a job is cancelled', async () => {
  const controller = new AbortController(); const promise = abortableDelay(30000, controller.signal);
  controller.abort(); await assert.rejects(promise, { name: 'AbortError' });
});

function microphoneHarness({ denied = false, pending = null, moduleFailure = false } = {}) {
  const tracks = [], contexts = [], nodes = [], states = [], samples = [];
  const makeStream = () => { const track = { stopped: 0, stop() { this.stopped++; } }; tracks.push(track); return { getTracks: () => [track] }; };
  const mediaDevices = { getUserMedia: async () => {
    if (pending) return pending.promise;
    if (denied) throw Object.assign(new Error('denied'), { name: 'NotAllowedError' });
    return makeStream();
  } };
  const controller = new MicrophoneController({ mediaDevices, createContext: () => {
    const context = { state: 'suspended', currentTime: 1, destination: {}, closed: 0,
      audioWorklet: { addModule: async () => { if (moduleFailure) throw new Error('worklet unavailable'); } },
      resume: async () => { context.state = 'running'; },
      close: async () => { context.closed++; context.state = 'closed'; },
      createMediaStreamSource: () => ({ connect() {}, disconnect() {} }),
    }; contexts.push(context); return context;
  }, createNode: () => { const node = { port: {}, connect() {}, disconnect() {} }; nodes.push(node); return node; },
  onState: (phase, message) => states.push({ phase, message }), onSamples: (...args) => samples.push(args) });
  return { controller, tracks, contexts, nodes, states, samples, mediaDevices, makeStream };
}

test('permission denial returns to idle with actionable Chinese instructions', async () => {
  const harness = microphoneHarness({ denied: true }); await harness.controller.start();
  assert.equal(harness.controller.phase, 'idle'); assert.match(harness.states.at(-1).message, /权限被拒绝/);
  assert.equal(harness.contexts.length, 0);
  harness.mediaDevices.getUserMedia = async () => harness.makeStream();
  await harness.controller.start(); assert.equal(harness.controller.phase, 'active'); await harness.controller.stop();
});

test('repeated starts do not duplicate input; repeated stops release tracks and contexts', async () => {
  const harness = microphoneHarness();
  for (let iteration = 0; iteration < 3; iteration++) {
    await harness.controller.start(); await harness.controller.start();
    assert.equal(harness.tracks.length, iteration + 1);
    await harness.controller.stop(); await harness.controller.stop();
  }
  assert.ok(harness.tracks.every(track => track.stopped === 1));
  assert.ok(harness.contexts.every(context => context.closed === 1));
  assert.equal(harness.controller.phase, 'idle');
});

test('stopping while permission is pending closes a later granted stream', async () => {
  const pending = deferred(), harness = microphoneHarness({ pending });
  const starting = harness.controller.start(); await harness.controller.stop();
  pending.resolve(harness.makeStream()); await starting;
  assert.equal(harness.tracks[0].stopped, 1); assert.equal(harness.contexts.length, 0); assert.equal(harness.controller.phase, 'idle');
});

test('worklet loading failure releases microphone and permits retry', async () => {
  const harness = microphoneHarness({ moduleFailure: true }); await harness.controller.start();
  assert.equal(harness.controller.phase, 'idle'); assert.equal(harness.tracks[0].stopped, 1); assert.equal(harness.contexts[0].closed, 1);
  assert.match(harness.states.at(-1).message, /启动失败/);
});

test('capture timestamps include half-window age and post-stop messages are ignored', async () => {
  const harness = microphoneHarness(); await harness.controller.start();
  const handler = harness.nodes[0].port.onmessage;
  handler({ data: { samples: new Float32Array(4096), sampleRate: 48000, captureTime: .99 } });
  assert.equal(harness.samples.length, 1);
  assert.ok(Math.abs(harness.samples[0][2] - (.01 + 4096 / 96000)) < 1e-12);
  await harness.controller.stop(); handler({ data: { samples: new Float32Array(4096), sampleRate: 48000, captureTime: .99 } });
  assert.equal(harness.samples.length, 1); assert.equal(harness.nodes[0].port.onmessage, null);
});

test('device disconnect closes its AudioContext and reports recovery guidance', async () => {
  const harness = microphoneHarness(); await harness.controller.start(); harness.tracks[0].onended(); await tick();
  assert.equal(harness.controller.phase, 'idle'); assert.equal(harness.contexts[0].closed, 1); assert.match(harness.states.at(-1).message, /断开/);
});

test('unsupported capture and busy devices produce concrete local guidance', async () => {
  let message = '';
  const controller = new MicrophoneController({ mediaDevices: {}, onState: (_phase, text) => { message = text; } }); await controller.start();
  assert.match(message, /localhost/);
  assert.match(microphoneError({ name: 'NotReadableError' }), /占用/);
  assert.match(microphoneError({ name: 'NotFoundError' }), /没有找到/);
});
