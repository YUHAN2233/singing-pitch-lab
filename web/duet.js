const $ = id => document.getElementById(id);
const descriptions = {
  independent: '两份彼此独立的单人干声，按文件名标记歌手。B 延迟可手动调整；两份录音须有共同的时间起点。',
  duet_channels: '仅当左声道和右声道确实各录一人时使用。普通立体声歌曲的左右声道不是两位歌手。',
  duet_mix: '一份没有伴奏的双人混录。逐帧寻找最多两个谐波基频；低／高轨不等于固定歌手。',
  duet_song: '先用本机 UVR 模型分离人声与伴奏，再实验性估计两个人声基频。模型未安装时不可用。'
};
let generation = 0, active = new Set(), request = null, current = null, pointer = null, audioUrls = [];
let modelReady = false;
const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));
const finite = n => typeof n === 'number' && Number.isFinite(n);

function modeChanged() {
  const mode = $('mode').value;
  $('mode-help').textContent = descriptions[mode];
  $('file-b-wrap').hidden = $('offset-wrap').hidden = mode !== 'independent';
  $('analyze').disabled = mode === 'duet_song' && !modelReady;
  if (mode === 'duet_song' && !modelReady) $('status').textContent = '伴奏分离模型尚未安装；请按安装说明导入匹配的模型。';
}
function status(message, progress = null) {
  $('status').textContent = message;
  $('progress').hidden = progress === null;
  if (progress !== null) $('progress').value = progress;
}
async function stop() {
  generation++;
  request?.abort(); request = null;
  const ids = [...active]; active.clear();
  await Promise.allSettled(ids.map(id => fetch(`/api/jobs/${id}`, {method:'DELETE'})));
  $('cancel').disabled = true; $('analyze').disabled = $('mode').value === 'duet_song' && !modelReady;
  status('已取消当前任务。');
}
async function upload(file, mode, token) {
  request = new AbortController();
  const response = await fetch(`/api/jobs?kind=recording&mode=${mode}`, {
    method:'POST', headers:{'content-type':'application/octet-stream','x-filename':encodeURIComponent(file.name)},
    body:file, signal:request.signal
  });
  request = null;
  if (!response.ok) throw new Error((await response.json()).detail || `上传失败 ${response.status}`);
  let job = await response.json();
  active.add(job.id);
  while (token === generation && job.status !== 'done') {
    if (job.status === 'error' || job.status === 'cancelled') throw new Error(job.message || '分析未完成');
    status(`${file.name}：${job.message}`, job.progress || 0);
    await sleep(300);
    const poll = await fetch(`/api/jobs/${job.id}`);
    if (!poll.ok) throw new Error('无法读取本机任务状态');
    job = await poll.json();
  }
  if (token !== generation) throw new Error('任务已取消');
  active.delete(job.id);
  return job.result;
}
function releaseAudio() { for (const url of audioUrls) URL.revokeObjectURL(url); audioUrls = []; }
async function loadAudio(element, path) {
  element.hidden = !path;
  element.removeAttribute('src');
  if (!path) return;
  const response = await fetch(path);
  if (!response.ok) return;
  const url = URL.createObjectURL(await response.blob());
  audioUrls.push(url); element.src = url;
}
async function show(result, second = null) {
  releaseAudio();
  const tracks = second ? [
    {label:`A · ${result.name}`,frames:result.frames},
    {label:`B · ${second.name}`,frames:second.frames}
  ] : result.tracks;
  if (!Array.isArray(tracks) || tracks.length !== 2) throw new Error('结果缺少两条音高轨迹');
  current = {tracks, mode:second?'independent':result.mode, result, second,
    duration:Math.max(result.duration || 0, (second?.duration || 0) + (second ? Number($('offset').value) || 0 : 0))};
  $('result').hidden = false;
  $('label-a').textContent = $('stats-a-label').textContent = tracks[0].label;
  $('label-b').textContent = $('stats-b-label').textContent = tracks[1].label;
  $('seek').max = String(Math.max(0, current.duration - 20)); $('seek').value = '0';
  $('warnings').replaceChildren(...(result.warnings || []).map(message => {const li=document.createElement('li');li.textContent=message;return li;}));
  if (second) { const li=document.createElement('li');li.textContent='独立文件由用户标记为 A/B；请用时间偏移对齐它们，不能假设两次录音天然同步。';$('warnings').append(li); }
  const summaries = tracks.map(track => {
    const valid=track.frames.filter(frame=>finite(frame.frequency));
    const sorted=valid.map(f=>f.frequency).sort((a,b)=>a-b);
    return {total:track.frames.length,voiced:valid.length,
      median:sorted.length?sorted[Math.floor(sorted.length/2)]:null,
      low:valid.length?valid.reduce((a,b)=>a.midi<b.midi?a:b).note:'—',
      high:valid.length?valid.reduce((a,b)=>a.midi>b.midi?a:b).note:'—'};
  });
  summaries.forEach((sum,i)=>{$(`stats-${i?'b':'a'}`).textContent=`有效 ${sum.voiced} / ${sum.total} 帧（${Math.round(100*sum.voiced/Math.max(1,sum.total))}%） · 音域 ${sum.low}–${sum.high} · 中位频率 ${sum.median?.toFixed(1) ?? '—'} Hz`;});
  const both = result.paired_frames?.filter(frame=>frame.voice_count===2).length;
  $('coverage').textContent = finite(both) ? `双音同时检出 ${both} / ${result.paired_frames.length} 帧；空白和单音帧不补成第二个人。` : '每条曲线独立统计有效帧；空白处代表无声或低置信度。';
  await Promise.all([loadAudio($('audio-a'), result.audio?.original),loadAudio($('audio-b'),second?.audio?.original)]);
  draw();
}
function nearest(frames, time) {
  let lo=0,hi=frames.length;
  while(lo<hi){const mid=(lo+hi)>>1;if(frames[mid].time<time)lo=mid+1;else hi=mid;}
  const frame=[frames[lo-1],frames[lo]].filter(Boolean).sort((a,b)=>Math.abs(a.time-time)-Math.abs(b.time-time))[0];
  return frame && Math.abs(frame.time-time)<.09 ? frame : null;
}
function format(frame) {return frame && finite(frame.frequency) ? `${frame.note} ${frame.cents>=0?'+':''}${frame.cents?.toFixed(0) ?? '—'} 音分 · ${frame.frequency.toFixed(1)} Hz · ${current.mode==='independent'||current.mode==='duet_channels'?'周期性':'谐波支持度'} ${Math.round((frame.confidence||0)*100)}%`:'无声／不确定';}
function draw() {
  if (!current) return;
  const canvas=$('chart'),rect=canvas.getBoundingClientRect(),dpr=window.devicePixelRatio||1;
  canvas.width=Math.round(rect.width*dpr); canvas.height=Math.round(rect.height*dpr);
  const ctx=canvas.getContext('2d');ctx.scale(dpr,dpr);
  const width=rect.width,height=rect.height,left=47,right=width-17,top=16,bottom=height-28;
  const start=Number($('seek').value),end=start+20,offset=current.mode==='independent'?Number($('offset').value)||0:0;
  const values=current.tracks.flatMap((track,i)=>track.frames.filter(f=>finite(f.midi)&&f.time+(i?offset:0)>=start&&f.time+(i?offset:0)<=end).map(f=>f.midi));
  let min=values.length?Math.floor(Math.min(...values)-2):48,max=values.length?Math.ceil(Math.max(...values)+2):84;
  min=Math.max(12,min);max=Math.min(108,Math.max(min+12,max));
  const x=t=>left+(t-start)/20*(right-left),y=m=>bottom-(m-min)/(max-min)*(bottom-top);
  ctx.fillStyle='#fff';ctx.fillRect(0,0,width,height);ctx.font='12px sans-serif';ctx.lineWidth=1;
  for(let m=Math.ceil(min/12)*12;m<=max;m+=12){const yy=y(m);ctx.strokeStyle='#e7eee8';ctx.beginPath();ctx.moveTo(left,yy);ctx.lineTo(right,yy);ctx.stroke();ctx.fillStyle='#729086';ctx.fillText(`C${m/12-1}`,5,yy+4);}
  for(let t=Math.ceil(start/5)*5;t<=end;t+=5){const xx=x(t);ctx.strokeStyle='#eef1e9';ctx.beginPath();ctx.moveTo(xx,top);ctx.lineTo(xx,bottom);ctx.stroke();ctx.fillStyle='#729086';ctx.fillText(`${t.toFixed(0)}s`,xx-8,height-7);}
  current.tracks.forEach((track,i)=>{
    ctx.strokeStyle=i?'#e28b67':'#167c75';ctx.lineWidth=2.5;ctx.beginPath();let active=false,previous=-Infinity;
    for(const f of track.frames){const t=f.time+(i?offset:0);if(t<start||t>end)continue;
      if(!finite(f.midi)||f.confidence<.15){active=false;continue;}
      if(!active||t-previous>.1)ctx.moveTo(x(t),y(f.midi));else ctx.lineTo(x(t),y(f.midi));
      active=true;previous=t;
    }ctx.stroke();
  });
  if(pointer!==null){ctx.strokeStyle='#54766a';ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(x(pointer),top);ctx.lineTo(x(pointer),bottom);ctx.stroke();
    $('hover').textContent=`${pointer.toFixed(2)} 秒 · ${current.tracks[0].label}: ${format(nearest(current.tracks[0].frames,pointer))} ｜ ${current.tracks[1].label}: ${format(nearest(current.tracks[1].frames,pointer-offset))}`;
  }
  canvas.setAttribute('aria-label',`${current.tracks[0].label} 与 ${current.tracks[1].label}，${start.toFixed(1)} 至 ${end.toFixed(1)} 秒音高曲线`);
}
async function refreshRecent(){try{const response=await fetch('/api/jobs');const jobs=await response.json();const select=$('recent');select.replaceChildren(new Option('选择本机保存的分析',''));for(const job of jobs.filter(j=>j.status==='done'&&j.mode?.startsWith('duet_')).reverse()){select.add(new Option(`${job.name} · ${job.mode}`,job.id));}}catch{}}
$('mode').addEventListener('change',()=>{if(active.size||request)void stop();modeChanged();});
for(const id of ['file-a','file-b']) $(id).addEventListener('change',()=>{if(active.size||request)void stop();});
$('cancel').addEventListener('click',()=>void stop());
$('analyze').addEventListener('click',async()=>{
  const mode=$('mode').value,a=$('file-a').files[0],b=$('file-b').files[0];
  if(!a||(mode==='independent'&&!b)){status('请先选择所需音频文件。');return;}
  const token=++generation;$('analyze').disabled=true;$('cancel').disabled=false;$('result').hidden=true;
  try{const first=await upload(a,mode==='independent'?'dry':mode,token);
    const second=mode==='independent'?await upload(b,'dry',token):null;
    if(token===generation){await show(first,second);status('分析完成。');void refreshRecent();}}
  catch(error){if(token===generation)status(`未完成：${error.message}`);}
  finally{if(token===generation){$('cancel').disabled=true;modeChanged();}}
});
$('restore').addEventListener('click',async()=>{const id=$('recent').value;if(!id)return;try{const r=await fetch(`/api/jobs/${id}`);const j=await r.json();if(j.status!=='done')throw Error('任务未完成');await show(j.result);status('已打开本机历史结果。');}catch(e){status(e.message);}});
$('seek').addEventListener('input',draw);$('offset').addEventListener('input',draw);window.addEventListener('resize',draw);
$('chart').addEventListener('pointermove',event=>{if(!current)return;const rect=$('chart').getBoundingClientRect();pointer=Number($('seek').value)+(event.clientX-rect.left-47)/(rect.width-64)*20;draw();});
$('chart').addEventListener('pointerleave',()=>{pointer=null;$('hover').textContent='将鼠标移到曲线上，查看两条轨迹同一时刻的音名、音分和证据强度。';draw();});
$('download').addEventListener('click',()=>{if(!current)return;const blob=new Blob([JSON.stringify({mode:current.mode,tracks:current.tracks,warnings:current.result.warnings||[]},null,2)],{type:'application/json'});const url=URL.createObjectURL(blob);const a=document.createElement('a');a.href=url;a.download='duet-analysis.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);});
$('delete-result').addEventListener('click',async()=>{
  if(!current||!window.confirm('删除当前分析及本机缓存音频？独立双文件模式会删除这两份分析。'))return;
  const ids=[...new Set([current.result.id,current.second?.id].filter(Boolean))];
  try{for(const id of ids){const response=await fetch(`/api/jobs/${id}`,{method:'DELETE'});if(!response.ok||(await response.json()).status!=='cancelled')throw new Error(`删除 ${id} 失败，请检查音频是否仍在播放`);}
    releaseAudio();current=null;$('result').hidden=true;status('已删除当前分析与本机缓存。');void refreshRecent();
  }catch(error){status(`删除未完成：${error.message}`);}
});
modeChanged();void fetch('/api/health').then(r=>r.json()).then(h=>{modelReady=!!h.model_ready;modeChanged();}).catch(()=>{});void refreshRecent();
