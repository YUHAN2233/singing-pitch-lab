import { detectPitch, describePitch, midiToFrequency, frameAt, isVoiced, displayPitch, pitchDifference, timelineReadout, compareRecording, referenceTime, LiveScore } from './pitch.js';
import { PitchHistory } from './history.js';

const TERMINAL = new Set(['done', 'error', 'cancelled']);

async function responseJSON(response) {
  let body;
  try { body = await response.json(); } catch { throw new Error(`本机服务返回了无效响应（${response.status}）。`); }
  if (!response.ok) throw new Error(body.error || body.detail || body.message || `请求失败（${response.status}）。`);
  return body;
}

export function abortableDelay(milliseconds, signal) {
  return new Promise((resolve, reject) => {
    if (signal?.aborted) { reject(new DOMException('Cancelled', 'AbortError')); return; }
    const timer = setTimeout(() => { signal?.removeEventListener('abort', abort); resolve(); }, milliseconds);
    const abort = () => { clearTimeout(timer); reject(new DOMException('Cancelled', 'AbortError')); };
    signal?.addEventListener('abort', abort, { once: true });
  });
}

/** Race-safe slot; a superseded POST is allowed to return its ID, then deleted.
 * Aborting an upload after the server accepted it would leave an unknown job. */
export class JobSlot {
  constructor(kind, { fetcher = globalThis.fetch.bind(globalThis), delay = abortableDelay, onChange = () => {} } = {}) {
    this.kind = kind; this.fetcher = fetcher; this.delay = delay; this.onChange = onChange;
    this.generation = 0; this.state = { status: 'idle', result: null, id: null };
    this.pollAbort = null;
  }
  emit(update) { this.state = { ...this.state, ...update }; this.onChange(this.state); }
  async deleteId(id) {
    const response = await this.fetcher(`/api/jobs/${encodeURIComponent(id)}`, { method: 'DELETE' });
    if (response.status === 404) return { status: 'cancelled' };
    const job = await responseJSON(response);
    if (job.status === 'error') throw new Error(job.message || '任务已停止显示，但本机文件删除失败；请重试移除。');
    return job;
  }
  cancel({ silent = false, preserveDone = false } = {}) {
    const generation = ++this.generation;
    this.pollAbort?.abort(); this.pollAbort = null;
    const id = this.state.id;
    const keep = preserveDone && this.state.status === 'done';
    this.emit({ status: silent ? 'idle' : 'cancelled', result: null, id: null, name: null, progress: 0, message: silent ? '' : '已请求取消 / 移除；进行中的上传将在返回任务编号后清理。', error: null });
    if (!id || keep) return Promise.resolve();
    return this.deleteId(id).then(job => {
      if (!silent && generation === this.generation) this.emit({ id: job.status === 'cancelled' ? null : id, message: job.status === 'cancelled' ? '已取消并删除本机任务文件。' : '已请求取消，当前计算步骤结束后会删除文件。' });
    }).catch(error => {
      if (!silent && generation === this.generation) this.emit({ status: 'error', id, message: error.message, error: error.message });
    });
  }
  restore(job) {
    if (job.status !== 'done' || job.result?.kind !== this.kind || !Array.isArray(job.result?.frames)) throw new Error('缓存结果不完整，无法恢复。');
    if (job.id !== this.state.id) this.cancel({ silent: true, preserveDone: this.kind === 'reference' });
    this.emit({ ...job, name: job.result.name, mode: job.result.mode, error: null });
  }
  async start(file, mode) {
    this.cancel({ silent: true, preserveDone: this.kind === 'reference' });
    const generation = this.generation;
    if (!file || !['mixed', 'dry'].includes(mode)) throw new Error('请选择有效的音频文件和类型。');
    this.emit({ status: 'uploading', name: file.name, mode, id: null, progress: 0, message: '正在传送到本机处理服务…', error: null });
    try {
      const created = await responseJSON(await this.fetcher(`/api/jobs?kind=${this.kind}&mode=${mode}`, {
        method: 'POST', headers: { 'Content-Type': 'application/octet-stream', 'X-Filename': encodeURIComponent(file.name || 'audio.wav') }, body: file,
      }));
      if (!created.id) throw new Error('本机服务未返回任务编号。');
      if (generation !== this.generation) { if (!created.reused) await this.deleteId(created.id); return; }
      this.emit({ id: created.id, status: created.status || 'queued', message: created.message || '已加入处理队列…', progress: created.progress || 0 });
      const controller = new AbortController(); this.pollAbort = controller;
      while (generation === this.generation) {
        const job = await responseJSON(await this.fetcher(`/api/jobs/${encodeURIComponent(created.id)}`, { signal: controller.signal }));
        if (generation !== this.generation) return;
        if (job.status === 'done' && !Array.isArray(job.result?.frames)) throw new Error('分析结果缺少真实音高轨迹，请重新处理。');
        this.emit({ ...job, progress: Number.isFinite(job.progress) ? job.progress : 0 });
        if (TERMINAL.has(job.status)) break;
        await this.delay(500, controller.signal);
      }
    } catch (error) {
      if (generation !== this.generation || error.name === 'AbortError') return;
      this.emit({ status: 'error', message: error.message || '连接本机服务失败，请检查服务是否启动。', error: error.message });
    }
  }
}

export function microphoneError(error) {
  if (error?.name === 'NotAllowedError' || error?.name === 'PermissionDeniedError') return '麦克风权限被拒绝。请在浏览器地址栏的站点权限中允许麦克风，然后重新开启。';
  if (error?.name === 'NotFoundError' || error?.name === 'DevicesNotFoundError') return '没有找到麦克风。连接输入设备后重新开启。';
  if (error?.name === 'NotReadableError' || error?.name === 'TrackStartError') return '麦克风无法读取，可能被其他程序占用。关闭占用程序后重试。';
  if (error?.name === 'SecurityError') return '浏览器安全设置阻止了麦克风。请使用本机 localhost 地址打开。';
  return `麦克风启动失败：${error?.message || '请检查设备后重试。'}`;
}

export class MicrophoneController {
  constructor({ mediaDevices = globalThis.navigator?.mediaDevices, createContext = () => new (globalThis.AudioContext || globalThis.webkitAudioContext)(), createNode = context => new AudioWorkletNode(context, 'singing-capture'), onState = () => {}, onSamples = () => {} } = {}) {
    this.mediaDevices = mediaDevices; this.createContext = createContext; this.createNode = createNode;
    this.onState = onState; this.onSamples = onSamples;
    this.generation = 0; this.phase = 'idle'; this.stream = null; this.context = null; this.source = null; this.node = null;
  }
  setPhase(phase, message = '') { this.phase = phase; this.onState(phase, message); }
  async start() {
    if (this.phase !== 'idle') return;
    if (!this.mediaDevices?.getUserMedia) { this.setPhase('idle', '浏览器无法使用麦克风。请在支持 AudioWorklet 的浏览器中通过 http://localhost 打开。'); return; }
    const generation = ++this.generation;
    this.setPhase('requesting', '请允许浏览器使用麦克风。');
    try {
      const stream = await this.mediaDevices.getUserMedia({ audio: { channelCount: 1, echoCancellation: false, noiseSuppression: false, autoGainControl: false }, video: false });
      if (generation !== this.generation) { stream.getTracks().forEach(track => track.stop()); return; }
      this.stream = stream;
      const context = this.createContext(); this.context = context;
      if (!context.audioWorklet) throw new Error('此浏览器不支持 AudioWorklet，请使用新版 Edge、Chrome 或 Firefox。');
      await context.audioWorklet.addModule(new URL('./worklet.js', import.meta.url).href);
      if (generation !== this.generation) return;
      await context.resume();
      if (generation !== this.generation) return;
      this.source = context.createMediaStreamSource(stream);
      this.node = this.createNode(context);
      this.node.port.onmessage = event => {
        if (generation !== this.generation || this.phase !== 'active') return;
        const { samples, sampleRate, captureTime } = event.data;
        const windowAge = Math.max(0, context.currentTime - captureTime) + samples.length / (2 * sampleRate);
        this.onSamples(samples, sampleRate, windowAge);
      };
      this.source.connect(this.node); this.node.connect(context.destination);
      stream.getTracks().forEach(track => { track.onended = () => { if (generation === this.generation) this.stop('麦克风已断开，请重新连接后开启。'); }; });
      this.setPhase('active', '麦克风已开启。保持单人干声输入，播放参考音轨即可比对。');
    } catch (error) {
      if (generation !== this.generation) return;
      await this.stop(microphoneError(error));
    }
  }
  async stop(message = '麦克风已停止，设备已释放。') {
    ++this.generation;
    const stream = this.stream, context = this.context, source = this.source, node = this.node;
    this.stream = this.context = this.source = this.node = null;
    if (node) { node.port.onmessage = null; try { node.disconnect(); } catch {} }
    if (source) try { source.disconnect(); } catch {}
    stream?.getTracks().forEach(track => { track.onended = null; track.stop(); });
    this.setPhase('idle', message);
    if (context && context.state !== 'closed') await context.close().catch(() => {});
  }
}

export function initializeApp(document) {
  const $ = id => document.getElementById(id);
  const player = $('audio-player');
  let playbackAudioContext = null, playbackGain = null, desiredPlaybackGain = 1;
  function setPlaybackGain(value) {
    desiredPlaybackGain = Number.isFinite(value) && value > 0 ? value : 1;
    if (playbackGain) playbackGain.gain.value = desiredPlaybackGain;
  }
  function ensurePlaybackGain() {
    if (playbackAudioContext || desiredPlaybackGain === 1) return;
    const Context = globalThis.AudioContext || globalThis.webkitAudioContext;
    if (!Context) return;
    playbackAudioContext = new Context();
    const source = playbackAudioContext.createMediaElementSource(player);
    playbackGain = playbackAudioContext.createGain();
    playbackGain.gain.value = desiredPlaybackGain;
    source.connect(playbackGain).connect(playbackAudioContext.destination);
  }
  const results = { reference: null, recording: null };
  const slots = {}, files = { reference: null, recording: null };
  let health = null, playbackKind = null, playbackKey = '', lastLiveFrame = null, lastVoicedFrame = null, liveFrames = [], samplesSeen = 0;
  let redrawPending = false, playingFrame = null, micErrorShown = false;
  const liveScore = new LiveScore();
  const history = new PitchHistory();
  let historyDuration = 0, historyStartedAt = null, historyFollowing = true, historyStart = 0;
  let historyPointerX = null, historyTouchTime = null;
  let timelinePointerX = null, timelineTouchTime = null;
  let lastStatsRendered = null;
  let libraryRequest = 0;
  const formatSize = bytes => Number.isFinite(bytes) ? `${(bytes / 1024 / 1024).toFixed(1)} MiB` : '大小未知';
  async function refreshLibrary() {
    const request = ++libraryRequest;
    try {
      const jobs = await responseJSON(await fetch('/api/jobs?summary=true'));
      if (request !== libraryRequest) return;
      const references = jobs.filter(job => job.kind === 'reference' && job.status !== 'cancelled').sort((a, b) => b.created - a.created);
      const list = $('reference-library-list');
      list.replaceChildren();
      $('reference-library-status').textContent = `${references.filter(job => job.status === 'done').length} 首已分析`;
      if (!references.length) {
        const empty = document.createElement('p'); empty.textContent = '还没有本机原唱结果。上传歌曲后会显示在这里。'; list.append(empty);
      }
      for (const job of references) {
        const row = document.createElement('div'); row.className = 'reference-library-row';
        const info = document.createElement('div'); info.className = 'reference-library-info';
        const title = document.createElement('strong'); title.textContent = job.name;
        const details = document.createElement('small');
        const status = job.status === 'done' ? '已完成' : job.status === 'error' ? '分析失败' : job.status === 'cancelled' ? '已取消' : '处理中';
        const duration = Number.isFinite(job.duration) ? ` · ${Math.floor(job.duration / 60)}:${String(Math.floor(job.duration % 60)).padStart(2, '0')}` : '';
        const when = Number.isFinite(job.completed_at || job.created) ? ` · ${new Date((job.completed_at || job.created) * 1000).toLocaleString('zh-CN')}` : '';
        details.textContent = `${status}${duration}${when} · ${formatSize(job.disk_bytes)}`;
        info.append(title, details); row.append(info);
        if (job.status === 'done') {
          const use = document.createElement('button'); use.type = 'button'; use.className = 'text-button';
          const selected = slots.reference.state.id === job.id && slots.reference.state.status === 'done';
          use.textContent = selected ? '当前原唱' : '使用这首'; use.disabled = selected;
          use.addEventListener('click', async () => {
            use.disabled = true;
            try {
              const full = await responseJSON(await fetch(`/api/jobs/${encodeURIComponent(job.id)}`));
              stopForChange(); files.reference = null; $('reference-mode').value = full.result.mode;
              slots.reference.restore(full); refreshLibrary();
            } catch (error) { $('reference-library-status').textContent = error.message; use.disabled = false; }
          });
          row.append(use);
        }
        const remove = document.createElement('button'); remove.type = 'button'; remove.className = 'text-button danger'; remove.textContent = '删除';
        remove.setAttribute('aria-label', `删除本机原唱 ${job.name}`);
        remove.addEventListener('click', async () => {
          if (!globalThis.confirm(`要删除「${job.name}」的本机分析和音频吗？此操作无法撤销。`)) return;
          remove.disabled = true;
          try {
            if (slots.reference.state.id === job.id) { stopForChange(); files.reference = null; await slots.reference.cancel(); }
            else await slots.reference.deleteId(job.id);
            await refreshLibrary();
          } catch (error) { $('reference-library-status').textContent = error.message; remove.disabled = false; }
        });
        row.append(remove); list.append(row);
      }
    } catch (error) { if (request === libraryRequest) $('reference-library-status').textContent = `读取曲库失败：${error.message}`; }
  }
  const settings = () => ({ offsetMs: Number($('time-offset').value) || 0, latencyMs: Number($('input-latency').value) || 0, transpose: Number($('transpose').value) || 0 });
  const resetLive = () => { liveScore.reset(); liveFrames = []; lastLiveFrame = null; samplesSeen = 0; renderMetrics(); scheduleDraw(); };
  const signed = n => `${n > 0 ? '+' : ''}${Math.round(n)}`;

  const microphone = new MicrophoneController({
    onState(phase, message) {
      if (phase === 'active' && historyStartedAt === null) {
        historyStartedAt = performance.now();
      } else if (phase === 'idle' && historyStartedAt !== null) {
        historyDuration += (performance.now() - historyStartedAt) / 1000;
        historyStartedAt = null;
        history.break(historyDuration);
      }
      $('mic-badge').textContent = phase === 'active' ? '正在聆听' : phase === 'requesting' ? '等待授权' : '麦克风未开启';
      $('mic-badge').classList.toggle('active', phase === 'active');
      $('mic-toggle').innerHTML = phase === 'idle' ? '<span aria-hidden="true">●</span> 开启麦克风' : '<span aria-hidden="true">■</span> 停止麦克风';
      $('mic-toggle').setAttribute('aria-pressed', phase !== 'idle' ? 'true' : 'false');
      if (message) $('mic-status').textContent = message;
      micErrorShown = phase === 'idle' && /拒绝|失败|无法|没有找到|阻止/.test(message);
      $('mic-status').classList.toggle('error-text', micErrorShown);
      if (phase === 'idle') { $('mic-level').value = 0; $('mic-confidence').textContent = '置信度 —'; renderLive(null); }
      renderMetrics();
    },
    onSamples(samples, sampleRate, windowAge) {
      const before = performance.now();
      const frame = detectPitch(samples, sampleRate);
      if (isVoiced(frame)) lastVoicedFrame = frame;
      history.add(historyDuration + (historyStartedAt === null ? 0 : (performance.now() - historyStartedAt) / 1000), frame);
      const age = windowAge + (performance.now() - before) / 1000;
      lastLiveFrame = frame;
      const inputDb = frame.rms > 0 ? 20 * Math.log10(frame.rms) : -Infinity;
      $('mic-level').value = Math.max(0, Math.min(1, (inputDb + 70) / 55));
      $('mic-confidence').textContent = `输入 ${Number.isFinite(inputDb) ? Math.round(inputDb) : '−∞'} dBFS · 置信度 ${Math.round(frame.confidence * 100)}%`;
      const { offsetMs, latencyMs, transpose } = settings();
      const time = referenceTime(player.currentTime, age, latencyMs, offsetMs);
      const activeComparison = !player.paused && !player.seeking && playbackKind === 'reference' && results.reference;
      const target = activeComparison ? frameAt(results.reference.frames, time) : null;
      renderLive(frame, target);
      if (activeComparison) {
        samplesSeen++;
        liveScore.add(time, frame, target, transpose, true);
        // Plot in the same reference clock as scoring. Breaks remain gaps.
        liveFrames.push({ ...frame, time });
        if (liveFrames.length > 24000) liveFrames.shift();
        if (!isVoiced(target)) $('mic-status').textContent = '当前参考位置无可信人声，暂不计分。';
        else if (!isVoiced(frame)) $('mic-status').textContent = '当前输入无声或置信度偏低，暂不计分。';
        else $('mic-status').textContent = '正在与缓存的原唱人声轨迹比对。';
        renderMetrics();
      } else $('mic-status').textContent = results.reference ? '当前仅测音高；播放原唱参考、人声或伴奏音轨后开始实时比对。' : '当前为独立干声测音；先分析原唱，才能实时比对。';
      if (frame.reason === 'quiet') $('mic-status').textContent = '麦克风仍在采集；输入声音太轻。请靠近麦克风，或检查系统输入音量。';
      else if (!isVoiced(frame)) $('mic-status').textContent = '麦克风仍在采集；检测到声音，但暂时没有稳定的单音音高。';
      scheduleDraw();
    },
  });

  // Changing a reference/recording invalidates alignment, but it must not
  // silently release an already active microphone while the user is singing.
  function stopForChange() { player.pause(); resetLive(); }

  function renderLive(frame, target = null) {
    const { shown, state } = displayPitch(frame, lastVoicedFrame, microphone.phase);
    const voiced = isVoiced(frame);
    $('live-note').firstChild.textContent = shown?.note || '—';
    const previous = shown ? `上次音高 ${shown.note} · ${shown.frequency.toFixed(1)} Hz` : '尚无上次音高';
    $('live-frequency').textContent = state === 'current' ? `${frame.frequency.toFixed(1)} Hz · 最近标准音 ${signed(frame.cents)} 音分`
      : state === 'quiet' ? `无声 · ${previous}`
      : state === 'stopped' ? shown ? `已停止 · ${previous}` : '等待你的声音'
      : state === 'unstable' ? '有声 · 音高不稳定' : '等待你的声音';
    $('live-note').setAttribute('aria-label', `${state === 'quiet' ? '无声' : state === 'stopped' ? '麦克风已停止' : state === 'unstable' ? '音高不稳定' : '当前音高'}：${shown?.note || '无有效音高'}`);
    const difference = pitchDifference(frame, target, settings().transpose);
    $('live-target').textContent = isVoiced(target) ? `目标音：${describePitch(midiToFrequency(target.midi + settings().transpose)).note}` : '目标音：—';
    $('live-delta').textContent = difference !== null ? `${signed(difference)} 音分` : '— 音分';
    const cents = difference ?? (voiced ? frame.cents : null);
    $('tuning-needle').classList.toggle('visible', cents !== null);
    $('tuning-needle').style.left = `${50 + Math.max(-50, Math.min(50, cents ?? 0))}%`;
    document.querySelector('.tuning-scale').setAttribute('aria-label', cents === null ? '当前没有可评分音高' : `${difference === null ? '相对标准音' : '相对原唱目标'}偏差 ${signed(cents)} 音分`);
  }

  function renderMetrics() {
    const live = microphone.phase !== 'idle' || samplesSeen > 0;
    const available = live ? Boolean(results.reference) : Boolean(results.reference && results.recording);
    const score = live ? liveScore.summary() : available ? compareRecording(results.reference.frames, results.recording.frames, settings()) : null;
    $('score-title').textContent = live ? '本轮实时比对' : '录音比对结果';
    $('score-caption').textContent = !available ? live ? '等待参考曲' : '等待两条音高轨迹' : score?.count ? score.coverage < .4 ? '覆盖率偏低，请检查对齐与人声质量' : '只统计有效人声片段' : '还没有可比较的可信人声帧';
    $('metric-cents').innerHTML = `${score?.meanAbsoluteCents !== null && score?.meanAbsoluteCents !== undefined ? Math.round(score.meanAbsoluteCents) : '—'}<small> 音分</small>`;
    $('metric-accuracy').textContent = score?.within50 !== null && score?.within50 !== undefined ? `${Math.round(score.within50 * 100)}%` : '—';
    $('metric-coverage').textContent = score?.eligible ? `${Math.round(score.coverage * 100)}%` : '—';
    $('metric-count').textContent = String(score?.count || 0);
  }

  function refreshSources() {
    const select = $('playback-source');
    const previous = select.value;
    select.replaceChildren();
    for (const kind of ['reference', 'recording']) {
      const result = results[kind];
      if (!result) continue;
      for (const [track, label] of [['original', '原始音频'], ['vocals', '人声'], ['accompaniment', '伴奏']]) {
        if (!result.audio?.[track]) continue;
        const option = document.createElement('option');
        option.value = `${kind}:${track}`; option.textContent = `${kind === 'reference' ? '原唱参考' : '我的录音'} · ${label}`;
        select.appendChild(option);
      }
    }
    select.disabled = select.options.length === 0;
    if (!select.options.length) {
      const option = document.createElement('option'); option.value = ''; option.textContent = '请先分析音频'; select.appendChild(option);
    } else if ([...select.options].some(option => option.value === previous)) select.value = previous;
    else select.value = results.reference?.audio?.accompaniment ? 'reference:accompaniment' : results.reference?.audio?.vocals ? 'reference:vocals' : select.options[0].value;
    updatePlayback(false);
  }

  function updatePlayback(reset = true) {
    const key = $('playback-source').value;
    const [kind, track] = key.split(':');
    const url = results[kind]?.audio?.[track];
    setPlaybackGain(results[kind]?.storage_gains?.[track] || 1);
    const uniqueKey = `${results[kind]?.id || ''}:${key}`;
    if (uniqueKey !== playbackKey) {
      player.pause();
      if (reset) resetLive();
      if (url) player.src = url; else player.removeAttribute('src');
      player.load(); playbackKey = uniqueKey;
    }
    playbackKind = url ? kind : null;
    $('playback-description').textContent = !url ? '分析后可试听原曲、人声与伴奏' : kind === 'reference' ? '播放这条音轨可实时比对 · 请戴耳机' : '演唱录音试听 · 实时比对需切换原唱参考';
    scheduleDraw();
  }

  function updateSlot(kind, state) {
    const oldResult = results[kind];
    results[kind] = state.status === 'done' ? state.result : null;
    $('' + kind + '-name').textContent = state.name || (kind === 'reference' ? '选择原唱音频文件' : '选择你的演唱录音');
    const fallback = { idle: kind === 'reference' ? '还没有参考曲' : '还没有演唱录音', done: '处理完成', error: '处理失败，请选择文件重试。', cancelled: '已取消 / 移除。' };
    let status = state.message || fallback[state.status] || '正在本机分析…';
    if (state.status === 'done') {
      const voiced = state.result.frames.filter(isVoiced).length;
      status = `已分析 ${Number(state.result.duration).toFixed(1)} 秒 · ${voiced} 个可信人声帧 · ${state.result.mode === 'mixed' ? '已完成人声分离' : '干声分析'}`;
    }
    $(kind + '-status').textContent = status;
    $(kind + '-status').classList.toggle('error-text', state.status === 'error');
    const progress = $(kind + '-progress');
    progress.hidden = !['uploading', 'queued', 'running'].includes(state.status);
    if (state.status === 'uploading') progress.removeAttribute('value'); else progress.value = state.progress;
    $(kind + '-cancel').hidden = ['idle', 'cancelled'].includes(state.status) && !state.id;
    const warnings = [...(state.result?.warnings || [])];
    if (state.status === 'done' && !state.result.frames.some(isVoiced)) warnings.push('未检测到可信单声部人声，不能评分。请检查文件、音量或分离结果。');
    $(kind + '-warning').hidden = !warnings.length;
    $(kind + '-warning').textContent = warnings.join(' ');
    if (oldResult !== results[kind]) { timelinePointerX = timelineTouchTime = null; refreshSources(); resetLive(); }
    renderMetrics(); scheduleDraw();
    if (kind === 'reference' && ['queued', 'done', 'error', 'cancelled'].includes(state.status)) refreshLibrary();
  }

  async function submitFile(kind, file, mode = $(kind + '-mode').value) {
    if (!file) return;
    stopForChange();
    files[kind] = file;
    $(kind + '-mode').value = mode;
    await slots[kind].start(file, mode);
  }

  for (const kind of ['reference', 'recording']) {
    slots[kind] = new JobSlot(kind, { onChange: state => updateSlot(kind, state) });
    $(kind + '-file').addEventListener('change', event => { const file = event.target.files[0]; event.target.value = ''; if (file) submitFile(kind, file); });
    $(kind + '-mode').addEventListener('change', () => { if (files[kind]) submitFile(kind, files[kind]); });
    $(kind + '-cancel').addEventListener('click', () => {
      if (kind === 'reference' && slots[kind].state.status === 'done'
          && !globalThis.confirm(`要删除「${slots[kind].state.name}」的本机分析和音频吗？此操作无法撤销。`)) return;
      stopForChange(); files[kind] = null; slots[kind].cancel().then(refreshLibrary);
    });
    $(kind + '-restore').addEventListener('click', async () => {
      const button = $(kind + '-restore'), generation = slots[kind].generation;
      button.disabled = true;
      try {
        const jobs = await responseJSON(await fetch('/api/jobs?summary=true'));
        if (generation !== slots[kind].generation) return;
        const latest = [...jobs].reverse().find(job => job.status === 'done' && job.kind === kind);
        if (!latest) { $(kind + '-status').textContent = '没有可恢复的本机结果。请先分析音频。'; return; }
        const restored = await responseJSON(await fetch(`/api/jobs/${encodeURIComponent(latest.id)}`));
        if (generation !== slots[kind].generation) return;
        stopForChange(); files[kind] = null;
        $(kind + '-mode').value = restored.result.mode;
        slots[kind].restore(restored);
      } catch (error) { $(kind + '-status').textContent = error.message; }
      finally { button.disabled = false; }
    });
    const drop = document.querySelector(`label[for="${kind}-file"]`);
    drop.addEventListener('dragover', event => { event.preventDefault(); drop.classList.add('drag-over'); });
    drop.addEventListener('dragleave', () => drop.classList.remove('drag-over'));
    drop.addEventListener('drop', event => { event.preventDefault(); drop.classList.remove('drag-over'); const file = event.dataTransfer.files[0]; if (file) submitFile(kind, file); });
    $(kind + '-demo').addEventListener('click', async () => {
      const button = $(kind + '-demo'); button.disabled = true;
      try {
        const response = await fetch(`/api/demo?kind=${kind}`);
        if (!response.ok) throw new Error('合成示例读取失败，请检查本机服务。');
        const blob = await response.blob();
        await submitFile(kind, new File([blob], kind === 'reference' ? '合成参考旋律.wav' : '合成偏音录音.wav', { type: 'audio/wav' }), 'dry');
      } catch (error) { $(kind + '-status').textContent = error.message; $(kind + '-status').classList.add('error-text'); }
      finally { button.disabled = false; }
    });
  }

  $('mic-toggle').addEventListener('click', async () => {
    if (microphone.phase === 'idle') { resetLive(); await microphone.start(); }
    else { player.pause(); await microphone.stop('你已手动停止麦克风，设备已释放。'); }
  });
  $('history-follow').addEventListener('click', () => {
    historyFollowing = !historyFollowing;
    historyStart = Math.max(history.oldest, history.latest - 20);
    scheduleDraw();
  });
  $('history-position').addEventListener('input', () => {
    historyFollowing = false;
    historyStart = Number($('history-position').value);
    scheduleDraw();
  });
  $('history-clear').addEventListener('click', () => {
    history.clear(); historyDuration = 0;
    historyStartedAt = microphone.phase === 'active' ? performance.now() : null;
    historyFollowing = true; historyStart = 0; historyPointerX = historyTouchTime = null;
    lastVoicedFrame = null; renderLive(microphone.phase === 'active' ? lastLiveFrame : null);
    scheduleDraw();
  });
  $('clear-cache').addEventListener('click', async () => {
    if (!globalThis.confirm('删除全部本机音频缓存和分析结果？此操作不会删除你原来的音频文件。正在处理的任务也会取消。')) return;
    const button = $('clear-cache'), status = $('cache-status');
    button.disabled = true; stopForChange(); await microphone.stop('清理本机缓存时已停止麦克风，设备已释放。'); files.reference = files.recording = null;
    await Promise.all([slots.reference.cancel({ silent: true }), slots.recording.cancel({ silent: true })]);
    status.textContent = '正在清理本机缓存…';
    try {
      const jobs = await responseJSON(await fetch('/api/jobs?summary=true'));
      let count = 0, pending = 0;
      const failed = [];
      for (const job of jobs) {
        try { const result = await slots.reference.deleteId(job.id); count++; if (result.status !== 'cancelled') pending++; }
        catch (error) { failed.push(error.message); }
      }
      status.textContent = failed.length ? `已提交 ${count} 个清理请求，${failed.length} 个失败：${failed[0]}` : pending ? `已提交 ${count} 个清理请求；${pending} 个正在计算的任务会在当前步骤结束后删除。` : `已删除 ${count} 个本机缓存任务。`;
      status.classList.toggle('error-text', failed.length > 0);
    } catch (error) { status.textContent = `清理失败：${error.message}`; status.classList.add('error-text'); }
    finally { button.disabled = false; }
  });
  $('playback-source').addEventListener('change', () => updatePlayback());
  for (const id of ['time-offset', 'input-latency', 'transpose']) {
    $(id).addEventListener('change', () => {
      const input = $(id); input.value = String(Math.max(Number(input.min), Math.min(Number(input.max), Number(input.value) || 0)));
      resetLive(); renderLive(lastLiveFrame); renderMetrics(); scheduleDraw();
    });
  }
  player.addEventListener('seeking', resetLive);
  player.addEventListener('play', () => {
    ensurePlaybackGain();
    playbackAudioContext?.resume().catch(() => {});
    if (microphone.phase !== 'idle' && playbackKind !== 'reference') $('mic-status').textContent = '这是演唱录音试听；实时比对请选择原唱参考音轨。';
    const tick = () => { scheduleDraw(); if (!player.paused) playingFrame = requestAnimationFrame(tick); };
    cancelAnimationFrame(playingFrame); tick();
  });
  player.addEventListener('pause', () => {
    cancelAnimationFrame(playingFrame); resetLive();
    if (microphone.phase === 'active') { $('mic-status').textContent = '播放已暂停，当前只测音高，不计分。'; renderLive(lastLiveFrame); }
  });
  player.addEventListener('timeupdate', scheduleDraw);
  player.addEventListener('error', () => { if (player.getAttribute('src')) $('playback-description').textContent = '音频播放失败，请重新处理文件。'; });
  document.addEventListener('visibilitychange', () => { if (document.hidden) { player.pause(); microphone.stop('页面进入后台，麦克风已自动停止。'); } });
  globalThis.addEventListener('pagehide', () => { player.pause(); microphone.stop('离开页面时麦克风已停止，设备已释放。'); });

  function scheduleDraw() {
    if (redrawPending) return;
    redrawPending = true;
    requestAnimationFrame(() => { redrawPending = false; drawChart(); drawHistory(); });
  }

  refreshLibrary();

  function drawHistoryCursor(width) {
    const marker = $('history-crosshair'), readout = $('history-cursor');
    const left = 46, right = width - 12;
    const x = historyTouchTime === null ? historyPointerX
      : left + (historyTouchTime - historyStart) / 20 * (right - left);
    if (x === null || x < left || x > right) { marker.hidden = readout.hidden = true; return; }
    const time = historyTouchTime ?? historyStart + (x - left) / (right - left) * 20;
    const found = history.nearest(time);
    const clock = `${Math.floor(time / 60)}:${String(Math.floor(time % 60)).padStart(2, '0')}.${Math.floor((time % 1) * 10)}`;
    let value = '无有效音高';
    if (found?.midi !== null && found) {
      const pitch = describePitch(midiToFrequency(found.midi));
      value = `${pitch.note} · ${pitch.frequency.toFixed(1)} Hz · ${signed(pitch.cents)} 音分`;
    } else if (found?.reason === 'quiet') value = '无声 · 无有效音高';
    else if (found?.reason === 'low_confidence' || found?.reason === 'unstable') value = '有声 · 音高不稳定';
    else if (found?.reason === 'stopped') value = '停止采集 · 无有效音高';
    readout.textContent = `${clock} · ${value}`;
    marker.style.left = `${x}px`;
    readout.style.left = `${Math.max(4, Math.min(width - 250, x + 9))}px`;
    marker.hidden = readout.hidden = false;
  }

  function renderHistoryStats() {
    const summary = history.summary();
    if (summary === lastStatsRendered) return;
    lastStatsRendered = summary;
    const label = midi => midi === null ? '暂无' : `${describePitch(midiToFrequency(midi)).note} · ${Math.round(midiToFrequency(midi))} Hz`;
    $('history-high').textContent = label(summary.highest);
    $('history-low').textContent = label(summary.lowest);
    const holder = $('history-counts');
    holder.replaceChildren();
    if (!summary.counts.size) holder.textContent = '暂无';
    for (const [midi, count] of [...summary.counts].sort((a, b) => b[0] - a[0])) {
      const chip = document.createElement('span');
      chip.className = 'note-count';
      chip.textContent = `${describePitch(midiToFrequency(midi)).note} ${count} 次`;
      holder.appendChild(chip);
    }
  }

  function drawHistory() {
    const canvas = $('history-chart'), rect = canvas.getBoundingClientRect();
    if (!rect.width || !rect.height) return;
    const ratio = globalThis.devicePixelRatio || 1;
    canvas.width = Math.round(rect.width * ratio); canvas.height = Math.round(rect.height * ratio);
    const ctx = canvas.getContext('2d'); ctx.scale(ratio, ratio);
    const width = rect.width, height = rect.height, left = 46, right = width - 12, top = 15, bottom = height - 30;
    const maxStart = Math.max(history.oldest, history.latest - 20);
    if (historyFollowing) historyStart = maxStart;
    historyStart = Math.max(history.oldest, Math.min(maxStart, historyStart));
    const end = historyStart + 20;
    const frames = history.visible(historyStart, end);
    $('history-empty').hidden = history.frames.length > 0;
    $('history-position').disabled = history.latest - history.oldest <= 20;
    $('history-position').min = String(history.oldest);
    $('history-position').max = String(maxStart);
    $('history-position').value = String(historyStart);
    $('history-follow').textContent = historyFollowing ? '暂停回看' : '恢复实时跟随';
    const clock = seconds => `${Math.floor(seconds / 60)}:${String(Math.floor(seconds % 60)).padStart(2, '0')}`;
    $('history-time').textContent = `${clock(historyStart)}–${clock(end)}`;
    $('history-status').textContent = !history.frames.length ? '等待本次演唱；停止麦克风后记录仍保留。'
      : `${historyFollowing ? '实时跟随' : '暂停回看，采集继续'} · ${clock(history.latest)} · ${history.frames.length} 个音高观测点（最近最多 30 分钟）`;
    const notes = frames.filter(frame => frame.midi !== null).map(frame => frame.midi);
    let low = 48, high = 72;
    if (notes.length) {
      low = Math.floor(Math.min(...notes) / 3) * 3 - 3;
      high = Math.ceil(Math.max(...notes) / 3) * 3 + 3;
      if (high - low < 12) { low -= 3; high += 3; }
    }
    const x = time => left + (time - historyStart) / 20 * (right - left);
    const y = midi => bottom - (midi - low) / (high - low) * (bottom - top);
    ctx.font = '10px "Segoe UI", sans-serif'; ctx.textBaseline = 'middle';
    for (let midi = Math.ceil(low / 3) * 3; midi <= high; midi += 3) {
      ctx.strokeStyle = '#edf1e8'; ctx.beginPath(); ctx.moveTo(left, y(midi)); ctx.lineTo(right, y(midi)); ctx.stroke();
      ctx.fillStyle = '#84998c'; ctx.textAlign = 'right'; ctx.fillText(describePitch(midiToFrequency(midi)).note, left - 8, y(midi));
    }
    for (let t = Math.ceil(historyStart / 5) * 5; t <= end; t += 5) {
      ctx.strokeStyle = '#f0f3ec'; ctx.beginPath(); ctx.moveTo(x(t), top); ctx.lineTo(x(t), bottom); ctx.stroke();
      ctx.fillStyle = '#84998c'; ctx.textAlign = 'center'; ctx.fillText(clock(t), x(t), height - 11);
    }
    ctx.strokeStyle = '#d58661'; ctx.lineWidth = 2.5; ctx.lineJoin = 'round'; ctx.lineCap = 'round'; ctx.beginPath();
    let previous = null;
    for (const frame of frames) {
      if (frame.midi === null) { previous = null; continue; }
      const px = x(frame.time), py = y(frame.midi);
      if (previous && frame.time - previous.time <= .24 && Math.abs(frame.midi - previous.midi) < 4) ctx.lineTo(px, py);
      else { ctx.moveTo(px, py); ctx.lineTo(px + .8, py); }
      previous = frame;
    }
    ctx.stroke();
    canvas.setAttribute('aria-label', `本次演唱音高历史，时间 ${clock(historyStart)} 到 ${clock(end)}，纵轴为音名；静音和低置信度处断开。`);
    renderHistoryStats();
    drawHistoryCursor(width);
  }

  function drawTimelineCursor({ width, left, right, duration, reference, input, offsetMs, transpose, usingLive }) {
    const marker = $('timeline-crosshair'), readout = $('timeline-cursor');
    const x = timelineTouchTime === null ? timelinePointerX
      : left + timelineTouchTime / duration * (right - left);
    if (x === null || x < left || x > right) { marker.hidden = readout.hidden = true; return; }
    const time = timelineTouchTime ?? (x - left) / (right - left) * duration;
    const inspected = timelineReadout(reference, input, time, { offsetMs, transpose, usingLive });
    const label = (frame, shift = 0) => {
      if (isVoiced(frame)) {
        const pitch = describePitch(midiToFrequency(frame.midi + shift));
        return `${pitch.note} · ${pitch.frequency.toFixed(1)} Hz`;
      }
      return frame?.reason === 'quiet' ? '无声' : frame?.reason === 'low_confidence' ? '低置信度' : '暂无';
    };
    const clock = `${Math.floor(time / 60)}:${String(Math.floor(time % 60)).padStart(2, '0')}.${Math.floor((time % 1) * 10)}`;
    readout.textContent = `${clock} · 原唱目标 ${label(inspected.target, transpose)} · ${usingLive ? '实时人声' : '我的声音'} ${label(inspected.input)}${inspected.difference === null ? '' : ` · 偏差 ${signed(inspected.difference)} 音分`}`;
    marker.style.left = `${x}px`;
    readout.style.left = `${Math.max(4, Math.min(width - Math.min(260, width - 8), x + 9))}px`;
    marker.hidden = readout.hidden = false;
  }

  function drawChart() {
    const canvas = $('pitch-chart');
    const rect = canvas.getBoundingClientRect();
    if (!rect.width || !rect.height) return;
    const ratio = globalThis.devicePixelRatio || 1;
    canvas.width = Math.round(rect.width * ratio); canvas.height = Math.round(rect.height * ratio);
    const ctx = canvas.getContext('2d'); ctx.scale(ratio, ratio);
    const width = rect.width, height = rect.height, left = 42, top = 13, right = width - 12, bottom = height - 24;
    const { offsetMs, transpose } = settings();
    const reference = results.reference?.frames || [];
    const recorded = results.recording?.frames || [];
    const usingLive = microphone.phase !== 'idle' || samplesSeen > 0;
    const input = usingLive ? liveFrames : recorded;
    const duration = Math.max(results.reference?.duration || 0, (results.recording?.duration || 0) + offsetMs / 1000, 6);
    const hasFrames = reference.length || input.length;
    $('chart-empty').hidden = Boolean(hasFrames);
    const voicedMidi = [];
    for (const frame of reference) if (isVoiced(frame)) voicedMidi.push(frame.midi + transpose);
    for (const frame of input) if (isVoiced(frame)) voicedMidi.push(frame.midi);
    let minMidi = 48, maxMidi = 72;
    if (voicedMidi.length) {
      minMidi = Math.floor(Math.min(...voicedMidi.slice(0, 100000)) / 3) * 3 - 3;
      maxMidi = Math.ceil(Math.max(...voicedMidi.slice(0, 100000)) / 3) * 3 + 3;
      if (maxMidi - minMidi < 12) { minMidi -= 3; maxMidi += 3; }
    }
    const x = time => left + time / duration * (right - left);
    const y = midi => bottom - (midi - minMidi) / (maxMidi - minMidi) * (bottom - top);
    ctx.font = '9px "Segoe UI", sans-serif'; ctx.textBaseline = 'middle';
    const step = maxMidi - minMidi > 36 ? 6 : 3;
    for (let midi = Math.ceil(minMidi / step) * step; midi <= maxMidi; midi += step) {
      ctx.strokeStyle = '#edf1e8'; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(left, y(midi)); ctx.lineTo(right, y(midi)); ctx.stroke();
      ctx.fillStyle = '#a0ad9c'; ctx.textAlign = 'right'; ctx.fillText(describePitch(midiToFrequency(midi)).note, left - 10, y(midi));
    }
    const timeStep = duration <= 12 ? 2 : duration <= 30 ? 5 : duration <= 90 ? 15 : duration <= 240 ? 30 : Math.ceil(duration / 6 / 60) * 60;
    for (let time = 0; time <= duration; time += timeStep) {
      ctx.strokeStyle = '#f2f5ee'; ctx.beginPath(); ctx.moveTo(x(time), top); ctx.lineTo(x(time), bottom); ctx.stroke();
      ctx.fillStyle = '#a0ad9c'; ctx.textAlign = 'center'; ctx.fillText(`${Math.floor(time / 60)}:${String(Math.floor(time % 60)).padStart(2, '0')}`, x(time), height - 9);
    }
    function trace(frames, color, shift, timeShift) {
      ctx.strokeStyle = color; ctx.lineWidth = 1.8; ctx.lineJoin = 'round'; ctx.lineCap = 'round'; ctx.beginPath();
      let previous = null;
      for (const frame of frames) {
        const time = frame.time + timeShift;
        if (!isVoiced(frame) || time < 0 || time > duration) { previous = null; continue; }
        const px = x(time), py = y(frame.midi + shift);
        if (previous && time - previous.time <= .18 && Math.abs(frame.midi - previous.midi) < 4) ctx.lineTo(px, py);
        else { ctx.moveTo(px, py); ctx.lineTo(px + .6, py); }
        previous = { time, midi: frame.midi };
      }
      ctx.stroke();
    }
    trace(reference, '#277b6d', transpose, 0);
    trace(input, '#d58b65', 0, usingLive ? 0 : offsetMs / 1000);
    drawTimelineCursor({ width, left, right, duration, reference, input, offsetMs, transpose, usingLive });
    if (player.getAttribute('src')) {
      const time = player.currentTime + (playbackKind === 'recording' ? offsetMs / 1000 : 0);
      if (time >= 0 && time <= duration) {
        ctx.strokeStyle = '#729482'; ctx.lineWidth = 1; ctx.setLineDash([3, 4]); ctx.beginPath(); ctx.moveTo(x(time), top); ctx.lineTo(x(time), bottom); ctx.stroke(); ctx.setLineDash([]);
      }
    }
    canvas.setAttribute('aria-label', hasFrames ? `音高曲线，时长 ${duration.toFixed(1)} 秒；青绿色为原唱目标，橙色为${usingLive ? '实时干声' : '演唱录音'}。目标转调 ${transpose} 半音，时间偏移 ${offsetMs} 毫秒。` : '音高曲线：等待真实音频分析，横轴为时间，纵轴为音名。');
  }

  async function refreshHealth() {
    try {
      health = await responseJSON(await fetch('/api/health'));
      $('engine-title').textContent = health.model_ready ? '本机分离引擎已就绪' : '干声引擎可用 · 分离模型未就绪';
      $('engine-detail').textContent = health.model_ready ? `${health.model_name || '神经网络'} · 原唱离线分离，麦克风实时检测` : '混音需先准备本机模型；可先用干声或合成示例';
      $('engine-dot').className = `status-dot${health.model_ready ? '' : ' pending'}`;
    } catch {
      $('engine-title').textContent = '无法连接本机处理服务'; $('engine-detail').textContent = '请按说明启动服务；浏览器端麦克风仍可独立测音。'; $('engine-dot').className = 'status-dot error';
    }
  }
  new ResizeObserver(scheduleDraw).observe($('pitch-chart').parentElement);
  new ResizeObserver(scheduleDraw).observe($('history-chart').parentElement);
  const historyCanvas = $('history-chart');
  const timelineCanvas = $('pitch-chart');
  function pointTimeline(event) {
    const rect = timelineCanvas.getBoundingClientRect();
    const x = event.clientX - rect.left;
    if (x < 42 || x > rect.width - 12) return;
    if (event.pointerType === 'touch') {
      const duration = Math.max(results.reference?.duration || 0,
        (results.recording?.duration || 0) + settings().offsetMs / 1000, 6);
      timelineTouchTime = (x - 42) / (rect.width - 54) * duration;
      timelinePointerX = null;
    } else { timelinePointerX = x; timelineTouchTime = null; }
    scheduleDraw();
  }
  timelineCanvas.addEventListener('pointermove', event => { if (event.pointerType !== 'touch') pointTimeline(event); });
  timelineCanvas.addEventListener('pointerdown', pointTimeline);
  timelineCanvas.addEventListener('pointerleave', event => {
    if (event.pointerType !== 'touch') { timelinePointerX = null; scheduleDraw(); }
  });
  function pointHistory(event) {
    const rect = historyCanvas.getBoundingClientRect();
    const x = event.clientX - rect.left;
    if (x < 46 || x > rect.width - 12) return;
    if (event.pointerType === 'touch') {
      historyTouchTime = historyStart + (x - 46) / (rect.width - 58) * 20;
      historyPointerX = null;
    } else {
      historyPointerX = x;
      historyTouchTime = null;
    }
    scheduleDraw();
  }
  historyCanvas.addEventListener('pointermove', event => { if (event.pointerType !== 'touch') pointHistory(event); });
  historyCanvas.addEventListener('pointerdown', pointHistory);
  historyCanvas.addEventListener('pointerleave', event => {
    if (event.pointerType !== 'touch') { historyPointerX = null; scheduleDraw(); }
  });
  renderMetrics(); scheduleDraw(); refreshHealth();
  return { slots, microphone, results, settings, refreshHealth, history };
}

if (typeof document !== 'undefined' && document.getElementById('pitch-chart')) initializeApp(document);
