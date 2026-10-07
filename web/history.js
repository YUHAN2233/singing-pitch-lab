import { isVoiced } from './pitch.js';

// At most 30 minutes at 10 observations/s. Only pitch metadata is retained.
export class PitchHistory {
  constructor(maxFrames = 18000, interval = .1) {
    this.maxFrames = maxFrames;
    this.interval = interval;
    this.clear();
  }
  clear() { this.frames = []; this.latest = 0; this._summary = null; }
  add(time, frame) {
    if (!Number.isFinite(time) || time < 0) return false;
    const voiced = isVoiced(frame);
    const previous = this.frames.at(-1);
    this.latest = Math.max(this.latest, time);
    if (previous && time - previous.time < this.interval && (previous.midi !== null) === voiced) return false;
    this.frames.push({ time, midi: voiced ? frame.midi : null, reason: voiced ? null : (frame?.reason || 'unvoiced') });
    this._summary = null;
    if (this.frames.length > this.maxFrames) this.frames.splice(0, this.frames.length - this.maxFrames);
    while (this.frames.length && this.frames[0].time < this.latest - 1800) this.frames.shift();
    return true;
  }
  break(time) {
    if (!this.frames.length) return;
    this.latest = Math.max(this.latest, time);
    this.frames.push({ time, midi: null, reason: 'stopped' });
    this._summary = null;
    if (this.frames.length > this.maxFrames) this.frames.splice(0, this.frames.length - this.maxFrames);
    while (this.frames.length && this.frames[0].time < this.latest - 1800) this.frames.shift();
  }
  get oldest() { return this.frames[0]?.time ?? 0; }
  visible(start, end) { return this.frames.filter(frame => frame.time >= start - .3 && frame.time <= end); }
  nearest(time, maxDistance = .13) {
    if (!Number.isFinite(time) || !this.frames.length) return null;
    let lo = 0, hi = this.frames.length;
    while (lo < hi) {
      const middle = (lo + hi) >>> 1;
      if (this.frames[middle].time < time) lo = middle + 1;
      else hi = middle;
    }
    const before = this.frames[lo - 1], after = this.frames[lo];
    const closest = !before ? after : !after ? before : time - before.time <= after.time - time ? before : after;
    return closest && Math.abs(closest.time - time) <= maxDistance ? closest : null;
  }
  summary() {
    if (this._summary) return this._summary;
    const segments = [];
    let active = null, pending = null;
    const minDuration = .24, maxGap = .32, hysteresis = .65;
    const commit = () => {
      if (active && active.last - active.start >= minDuration) segments.push({ note: active.note, start: active.start, end: active.last });
      active = null;
    };
    for (const item of this.frames) {
      if (item.midi === null) {
        pending = null;
        if (item.reason === 'quiet' || item.reason === 'stopped') commit();
        continue;
      }
      if (active && item.time - active.last > maxGap) commit();
      const note = Math.round(item.midi);
      if (!active) {
        if (!pending || pending.note !== note || item.time - pending.last > maxGap) pending = { note, start: item.time, last: item.time };
        else pending.last = item.time;
        if (pending.last - pending.start >= minDuration) { active = pending; pending = null; }
      } else if (Math.abs(item.midi - active.note) <= hysteresis) {
        active.last = item.time; pending = null;
      } else {
        if (!pending || pending.note !== note || item.time - pending.last > maxGap) pending = { note, start: item.time, last: item.time };
        else pending.last = item.time;
        if (pending.last - pending.start >= minDuration) { commit(); active = pending; pending = null; }
      }
    }
    commit();
    const counts = new Map();
    for (const segment of segments) counts.set(segment.note, (counts.get(segment.note) || 0) + 1);
    const notes = [...counts.keys()].sort((a, b) => b - a);
    return this._summary = { highest: notes[0] ?? null, lowest: notes.at(-1) ?? null, counts, segments };
  }
}
