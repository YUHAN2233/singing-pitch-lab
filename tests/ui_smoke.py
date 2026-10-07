"""Run isolated system-Chrome acceptance tests against the local app.

No user's browser profile or real microphone is accessed. Microphone denial is
tested through Chrome permissions; voiced/silent microphone tests inject an
OscillatorNode-backed MediaStream through the actual AudioWorklet/YIN pipeline.

Usage: .venv\\Scripts\\python.exe tests\\ui_smoke.py --base-url http://127.0.0.1:8765
"""
from __future__ import annotations

import argparse
import ctypes
from ctypes import wintypes
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import re
import sys
import time
import traceback

import numpy as np
from playwright.sync_api import sync_playwright, expect
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
EVIDENCE = ROOT / "evidence"
CHROME = r"C:\Program Files\Google\Chrome\Application\chrome.exe"


MIC_INJECTION = r"""
() => {
  const state = window.__qaMic = {mode: 'immediate', frequency: 440, gain: .25,
    calls: 0, contexts: [], streams: [], resolvers: []};
  Object.defineProperty(navigator.mediaDevices, 'getUserMedia', {
    configurable: true,
    value: async () => {
      state.calls++;
      const context = new AudioContext();
      const source = context.createOscillator();
      source.frequency.value = state.frequency;
      const gain = context.createGain(); gain.gain.value = state.gain;
      const destination = context.createMediaStreamDestination();
      source.connect(gain).connect(destination); source.start();
      await context.resume();
      const item = {context, source, gain, stream: destination.stream};
      state.contexts.push(item); state.streams.push(destination.stream);
      if (state.mode === 'pending') return new Promise(resolve => state.resolvers.push(() => resolve(destination.stream)));
      return destination.stream;
    }
  });
}
"""


def tone_bytes(frequency=440, seconds=0.8):
    rate = 44100
    signal = .25 * np.sin(2 * np.pi * frequency * np.arange(int(rate * seconds)) / rate)
    target = io.BytesIO()
    sf.write(target, signal, rate, format="WAV", subtype="PCM_16")
    return target.getvalue()


def wait_js(page, expression, timeout=15000):
    """Poll via CDP evaluation without weakening the application's strict CSP."""
    deadline = time.monotonic() + timeout / 1000
    while time.monotonic() < deadline:
        if page.evaluate(expression):
            return
        page.wait_for_timeout(50)
    raise AssertionError(f"Browser condition timed out: {expression}")


def lock_windows_file(path):
    """Deny deletion of a test-created file until CloseHandle is called."""
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                                  wintypes.LPVOID, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
    kernel.CreateFileW.restype = wintypes.HANDLE
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel.CloseHandle.restype = wintypes.BOOL
    handle = kernel.CreateFileW(str(path), 0x80000000, 1, None, 3, 0x80, None)
    if handle == wintypes.HANDLE(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    return lambda: kernel.CloseHandle(handle)


def run(base_url, chrome):
    EVIDENCE.mkdir(exist_ok=True)
    report = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "base_url": base_url,
        "browser_executable": chrome,
        "real_hardware_microphone_tested": False,
        "microphone_fixture": "WebAudio 440 Hz oscillator MediaStream; actual worklet and detector; no real microphone",
        "tests": [], "console_errors": [], "page_errors": [], "external_requests": [],
    }
    started = time.perf_counter()
    owned_jobs = set()
    current_case = "initialization"
    mixed_report = None

    def record(name, detail=None):
        report["tests"].append({"name": name, "status": "passed", "detail": detail})
        print(f"PASS {name}", flush=True)

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            executable_path=chrome, headless=True,
            args=["--use-fake-device-for-media-stream", "--autoplay-policy=no-user-gesture-required"],
        )
        report["browser_version"] = browser.version
        context = browser.new_context(viewport={"width": 1440, "height": 1120}, locale="zh-CN")
        page = context.new_page()
        page.set_default_timeout(15000)
        page.on("pageerror", lambda error: report["page_errors"].append(str(error)))
        page.on("console", lambda message: report["console_errors"].append({"case": current_case, "text": message.text}) if message.type == "error" else None)

        def observed_response(response):
            if "/api/jobs?" in response.url and response.request.method == "POST" and response.status == 202:
                try:
                    owned_jobs.add(response.json()["id"])
                except Exception:
                    pass

        def observed_request(request):
            if not request.url.startswith((base_url, "data:", "blob:")):
                report["external_requests"].append(request.url)

        page.on("response", observed_response)
        page.on("request", observed_request)

        def analyze_file(kind, data, name="qa-tone.wav", mode="dry"):
            page.locator(f"#{kind}-mode").select_option(mode)
            page.locator(f"#{kind}-file").set_input_files({"name": name, "mimeType": "audio/wav", "buffer": data})
            expect(page.locator(f"#{kind}-status")).to_contain_text("已分析", timeout=180000 if mode == "mixed" else 30000)

        def start_microphone():
            page.locator("#mic-toggle").click()
            expect(page.locator("#mic-toggle")).to_contain_text("停止")
            wait_js(page, "document.querySelector('#mic-badge').classList.contains('active')")

        def stop_microphone():
            page.locator("#mic-toggle").click()
            expect(page.locator("#mic-toggle")).to_contain_text("开启")
            wait_js(page, "window.__qaMic.streams.every(stream => stream.getTracks().every(track => track.readyState === 'ended'))")

        try:
            current_case = "local_page_boot_and_privacy"
            page.goto(base_url, wait_until="networkidle")
            expect(page.locator("#engine-title")).to_contain_text("就绪")
            expect(page.locator(".headphone-tip")).to_contain_text("耳机")
            expect(page.locator(".headphone-tip")).to_contain_text("实时混音分离")
            assert page.locator("#reference-mode").input_value() == "mixed"
            assert page.locator("#recording-mode").input_value() == "mixed"
            record(current_case, "Both uploads default to mixed mode; headphones and offline separation boundary are visible")

            current_case = "permission_denial_and_recovery_ui"
            session = context.new_cdp_session(page)
            session.send("Browser.setPermission", {"permission": {"name": "microphone"}, "setting": "denied", "origin": base_url})
            page.locator("#mic-toggle").click()
            expect(page.locator("#mic-status")).to_contain_text(re.compile("拒绝|权限"))
            expect(page.locator("#mic-toggle")).to_contain_text("开启")
            assert page.evaluate("navigator.permissions.query({name:'microphone'}).then(p=>p.state)") == "denied"
            record(current_case, "Actual Chrome microphone permission denied via CDP; no hardware capture")

            current_case = "real_demo_upload_analysis_and_35_cent_difference"
            page.locator("#reference-demo").click()
            expect(page.locator("#reference-status")).to_contain_text("已分析", timeout=30000)
            current_case = "timeline_single_reference_without_recording"
            single_canvas = page.locator("#pitch-chart")
            single_canvas.scroll_into_view_if_needed()
            single_box = single_canvas.bounding_box()
            page.mouse.move(single_box["x"] + 42 + (single_box["width"] - 54) * .2,
                            single_box["y"] + single_box["height"] / 2)
            expect(page.locator("#timeline-cursor")).to_contain_text("原唱目标")
            expect(page.locator("#timeline-cursor")).to_contain_text("我的声音 暂无")
            assert "偏差" not in page.locator("#timeline-cursor").inner_text()
            record(current_case, "Single reference curve shows no fabricated singer pitch or deviation")
            current_case = "real_demo_upload_analysis_and_35_cent_difference"
            page.locator("#recording-demo").click()
            expect(page.locator("#recording-status")).to_contain_text("已分析", timeout=30000)
            cents = float(re.search(r"[0-9.]+", page.locator("#metric-cents").inner_text()).group())
            assert 33 < cents < 37, cents
            assert int(page.locator("#metric-count").inner_text()) > 230
            expect(page.locator("#chart-empty")).to_be_hidden()
            record(current_case, {"mean_absolute_cents": cents})

            current_case = "timeline_pointer_two_tracks_offset_transpose_and_gaps"
            all_jobs = context.request.get(base_url + "/api/jobs?summary=false").json()
            ref = next(item["result"] for item in all_jobs if item.get("result") and item["result"]["kind"] == "reference")
            rec = next(item["result"] for item in all_jobs if item.get("result") and item["result"]["kind"] == "recording")
            ref_times = {round(item["time"], 2) for item in ref["frames"] if item["frequency"]}
            inspection_time = next(item["time"] for item in rec["frames"]
                                   if item["frequency"] and round(item["time"], 2) in ref_times)
            canvas = page.locator("#pitch-chart")
            canvas.scroll_into_view_if_needed()
            box = canvas.bounding_box()
            def hover_timeline(at):
                nonlocal box
                canvas.scroll_into_view_if_needed()
                box = canvas.bounding_box()
                page.mouse.move(box["x"] + 42 + (box["width"] - 54) * at / max(ref["duration"], rec["duration"], 6),
                                box["y"] + box["height"] / 2)
            hover_timeline(inspection_time)
            expect(page.locator("#timeline-cursor")).to_contain_text("原唱目标")
            expect(page.locator("#timeline-cursor")).to_contain_text("我的声音")
            expect(page.locator("#timeline-cursor")).to_contain_text("Hz")
            expect(page.locator("#timeline-cursor")).to_contain_text("偏差")
            before_transpose = page.locator("#timeline-cursor").inner_text()
            page.locator("#transpose").fill("12")
            page.locator("#transpose").dispatch_event("change")
            box = canvas.bounding_box()
            hover_timeline(inspection_time)
            expect(page.locator("#timeline-cursor")).to_contain_text("偏差")
            expect(page.locator("#timeline-cursor")).not_to_have_text(before_transpose)
            page.locator("#transpose").fill("0")
            page.locator("#transpose").dispatch_event("change")
            box = canvas.bounding_box()
            hover_timeline(max(ref["duration"], rec["duration"], 6) - .02)
            expect(page.locator("#timeline-cursor")).to_contain_text("暂无")
            page.mouse.move(box["x"] + box["width"] + 5, box["y"])
            expect(page.locator("#timeline-cursor")).to_be_hidden()
            record(current_case, {"two_tracks": True, "transpose_changes_readout": True, "trailing_gap": True})

            current_case = "reload_and_restore_cached_local_results"
            page.reload(wait_until="networkidle")
            page.locator("#reference-restore").click()
            expect(page.locator("#reference-status")).to_contain_text("已分析", timeout=20000)
            page.locator("#recording-restore").click()
            expect(page.locator("#recording-status")).to_contain_text("已分析", timeout=20000)
            assert int(page.locator("#metric-count").inner_text()) > 230
            record(current_case, "Both completed tracks restored from local cached results after page reload")

            current_case = "reference_library_switch_reload_and_deduplicate"
            first_name = page.locator("#reference-name").inner_text()
            first_id = next(item["id"] for item in context.request.get(base_url + "/api/jobs").json()
                            if item["kind"] == "reference" and item["status"] == "done" and item["name"] == first_name)
            second_audio = tone_bytes(523.251)
            analyze_file("reference", second_audio, "qa-library-c5.wav")
            second_id = next(item["id"] for item in context.request.get(base_url + "/api/jobs").json()
                             if item["kind"] == "reference" and item["status"] == "done" and item["name"] == "qa-library-c5.wav")
            assert first_id != second_id
            assert context.request.get(base_url + f"/api/jobs/{first_id}").json()["status"] == "done"
            page.locator(".reference-library-row", has_text=first_name).get_by_role("button", name="使用这首").click()
            expect(page.locator("#reference-name")).to_have_text(first_name)
            page.reload(wait_until="networkidle")
            page.locator(".reference-library-row", has_text="qa-library-c5.wav").get_by_role("button", name="使用这首").click()
            expect(page.locator("#reference-name")).to_have_text("qa-library-c5.wav")
            duplicate = context.request.post(base_url + "/api/jobs?kind=reference&mode=dry",
                                             data=second_audio, headers={"X-Filename": "duplicate-c5.wav"}).json()
            assert duplicate["id"] == second_id and duplicate["reused"] is True
            assert context.request.get(base_url + f"/api/jobs/{first_id}").json()["status"] == "done"
            page.locator(".reference-library-row", has_text=first_name).get_by_role("button", name="使用这首").click()
            page.locator("#recording-restore").click()
            expect(page.locator("#recording-status")).to_contain_text("已分析")
            page.once("dialog", lambda dialog: dialog.dismiss())
            page.locator(".reference-library-row", has_text="qa-library-c5.wav").get_by_role("button", name="删除").click()
            assert context.request.get(base_url + f"/api/jobs/{second_id}").json()["status"] == "done"
            page.once("dialog", lambda dialog: dialog.accept())
            page.locator(".reference-library-row", has_text="qa-library-c5.wav").get_by_role("button", name="删除").click()
            wait_js(page, f"!document.body.innerText.includes('qa-library-c5.wav')")
            assert context.request.get(base_url + f"/api/jobs/{second_id}").json()["status"] == "cancelled"
            assert not (Path(context.request.get(base_url + "/api/health").json()["data_directory"]) / second_id).exists()
            assert context.request.get(base_url + f"/api/jobs/{first_id}").json()["status"] == "done"
            record(current_case, {"saved_references_before_explicit_delete": 2, "reused_id": second_id,
                                  "delete_required_confirmation": True})

            current_case = "playback_seek_offset_latency_and_transpose_controls"
            page.locator("#transpose").fill("12")
            page.locator("#transpose").dispatch_event("change")
            octave_error = float(re.search(r"[0-9.]+", page.locator("#metric-cents").inner_text()).group())
            assert 1160 < octave_error < 1170, octave_error
            page.locator("#transpose").fill("0")
            page.locator("#transpose").dispatch_event("change")
            page.locator("#time-offset").fill("300")
            page.locator("#time-offset").dispatch_event("change")
            shifted = float(re.search(r"[0-9.]+", page.locator("#metric-cents").inner_text()).group())
            assert shifted > cents + 10
            page.locator("#time-offset").fill("0")
            page.locator("#time-offset").dispatch_event("change")
            page.locator("#input-latency").fill("160")
            page.locator("#input-latency").dispatch_event("change")
            page.locator("#audio-player").evaluate("audio => {audio.currentTime = 2;}")
            wait_js(page, "document.querySelector('#audio-player').currentTime >= 1.9")
            page.locator("#audio-player").evaluate("audio => audio.play()")
            wait_js(page, "document.querySelector('#audio-player').currentTime > 2.1")
            page.locator("#audio-player").evaluate("audio => audio.pause()")
            record(current_case, {"transpose_12_absolute_cents": octave_error, "offset_300ms_absolute_cents": shifted})
            page.screenshot(path=str(EVIDENCE / "ui-desktop.png"), full_page=True)

            current_case = "invalid_file_failure_and_valid_file_recovery"
            page.locator("#recording-mode").select_option("dry")
            page.locator("#recording-file").set_input_files({"name": "invalid.wav", "mimeType": "audio/wav", "buffer": b"not an audio file"})
            expect(page.locator("#recording-status")).to_contain_text(re.compile("无法解码|失败"), timeout=30000)
            analyze_file("recording", tone_bytes(), "recovered-a4.wav")
            record(current_case, "Invalid file executed real FFmpeg failure; a valid upload then completed")

            current_case = "switch_file_removes_previous_audio_cache"
            previous_ids = {item["id"] for item in context.request.get(base_url + "/api/jobs").json() if item["status"] == "done"}
            previous_source = page.locator("#recording-name").inner_text()
            analyze_file("recording", tone_bytes(523.251), "switched-c5.wav")
            expect(page.locator("#recording-name")).to_contain_text("switched-c5")
            jobs = context.request.get(base_url + "/api/jobs").json()
            after_ids = {item["id"] for item in jobs if item["status"] == "done"}
            assert len(previous_ids - after_ids) >= 1
            record(current_case, {"previous_file": previous_source, "removed_completed_jobs": len(previous_ids - after_ids)})

            current_case = "actual_windows_file_lock_delete_failure_and_retry"
            jobs = context.request.get(base_url + "/api/jobs?summary=false").json()
            locked_job = next(job for job in jobs if job.get("result") and job["result"]["name"] == "switched-c5.wav")
            data_dir = Path(context.request.get(base_url + "/api/health").json()["data_directory"])
            locked_path = data_dir / locked_job["id"] / "original.flac"
            unlock = lock_windows_file(locked_path)
            try:
                page.locator("#recording-cancel").click()
                expect(page.locator("#recording-status")).to_contain_text("未完全清除")
                expect(page.locator("#recording-cancel")).to_be_visible()
                assert locked_path.exists()
            finally:
                unlock()
            page.locator("#recording-cancel").click()
            expect(page.locator("#recording-cancel")).to_be_hidden()
            assert not locked_path.parent.exists()
            analyze_file("recording", tone_bytes(523.251), "after-lock-retry.wav")
            record(current_case, "Actual Windows file handle denied deletion; UI kept retry action, then file deletion succeeded after release")

            current_case = "synthetic_microphone_real_detector_repeated_start_stop"
            page.evaluate(MIC_INJECTION)
            current_case = "sustained_voice_silence_recovery_and_file_change"
            page.evaluate("""() => {
              window.__qaCaptureUpdates = 0;
              new MutationObserver(() => window.__qaCaptureUpdates++)
                .observe(document.querySelector('#live-frequency'), {childList:true, characterData:true, subtree:true});
            }""")
            start_microphone()
            def sample_live(seconds):
                observed = []
                for _ in range(round(seconds * 10)):
                    observed.append(page.evaluate("""() => ({
                      active: document.querySelector('#mic-badge').classList.contains('active'),
                      a4: document.querySelector('#live-note').textContent.includes('A4'),
                      frequency: document.querySelector('#live-frequency').textContent,
                      needle: document.querySelector('#tuning-needle').classList.contains('visible'),
                      updates: window.__qaCaptureUpdates,
                    })"""))
                    page.wait_for_timeout(100)
                return observed
            voice_before = sample_live(2.5)
            assert all(row["active"] for row in voice_before), voice_before
            assert sum(row["a4"] for row in voice_before) >= 20, voice_before
            assert voice_before[-1]["updates"] - voice_before[0]["updates"] >= 20, voice_before
            expect(page.locator("#history-counts")).to_contain_text("A4 1 次")
            expect(page.locator("#history-high")).to_contain_text("A4")
            expect(page.locator("#history-low")).to_contain_text("A4")
            current_case = "pitch_history_cursor_readout_and_silent_last_note"
            history_canvas = page.locator("#history-chart")
            history_canvas.scroll_into_view_if_needed()
            history_box = history_canvas.bounding_box()
            page.mouse.move(history_box["x"] + 46 + (history_box["width"] - 58) * .075, history_box["y"] + history_box["height"] / 2)
            expect(page.locator("#history-cursor")).to_contain_text("A4")
            expect(page.locator("#history-cursor")).to_contain_text("Hz")
            page.mouse.move(history_box["x"] + history_box["width"] + 5, history_box["y"])
            expect(page.locator("#history-cursor")).to_be_hidden()
            current_case = "live_pitch_history_review_retention_clear"
            expect(page.locator("#history-status")).to_contain_text("实时跟随")
            assert int(re.search(r"(\d+) 个音高观测点", page.locator("#history-status").inner_text()).group(1)) > 10
            page.wait_for_timeout(18500)
            assert page.locator("#history-position").is_enabled(), "History must be scrollable after 20 seconds"
            expect(page.locator("#history-counts")).to_contain_text("A4 1 次")
            page.locator("#history-follow").click()
            expect(page.locator("#history-follow")).to_have_text("恢复实时跟随")
            page.locator("#history-position").evaluate("slider => { slider.value = '0'; slider.dispatchEvent(new Event('input', {bubbles:true})); }")
            expect(page.locator("#history-time")).to_contain_text("0:00")
            frozen_time = page.locator("#history-time").inner_text()
            frozen_count = int(re.search(r"(\d+) 个音高观测点", page.locator("#history-status").inner_text()).group(1))
            page.wait_for_timeout(400)
            assert page.locator("#history-time").inner_text() == frozen_time
            assert int(re.search(r"(\d+) 个音高观测点", page.locator("#history-status").inner_text()).group(1)) > frozen_count
            assert page.locator("#mic-badge").get_attribute("class").find("active") >= 0
            page.locator("#history-follow").click()
            expect(page.locator("#history-follow")).to_have_text("暂停回看")
            assert page.locator("#history-time").inner_text() != frozen_time
            page.evaluate("window.__qaMic.contexts.at(-1).gain.gain.value = 0")
            page.wait_for_timeout(250)
            silence = sample_live(1.0)
            assert all(row["active"] for row in silence), silence
            assert sum("无声" in row["frequency"] and row["a4"] for row in silence) >= 8, silence
            assert sum(not row["needle"] for row in silence) >= 8, silence
            expect(page.locator("#history-counts")).to_contain_text("A4 1 次")
            history_box = history_canvas.bounding_box()
            page.mouse.move(history_box["x"] + history_box["width"] - 12 - (history_box["width"] - 58) * .02, history_box["y"] + history_box["height"] / 2)
            expect(page.locator("#history-cursor")).to_contain_text("无声")
            page.mouse.move(history_box["x"] + history_box["width"] + 5, history_box["y"])
            page.evaluate("window.__qaMic.contexts.at(-1).gain.gain.value = .003")
            page.wait_for_timeout(250)
            quiet_voice = sample_live(2.0)
            assert all(row["active"] for row in quiet_voice), quiet_voice
            assert sum(row["a4"] for row in quiet_voice) >= 16, quiet_voice
            page.evaluate("window.__qaMic.contexts.at(-1).gain.gain.value = .25")
            page.wait_for_timeout(250)
            voice_after = sample_live(2.0)
            assert all(row["active"] for row in voice_after), voice_after
            assert sum(row["a4"] for row in voice_after) >= 16, voice_after
            expect(page.locator("#history-counts")).to_contain_text("A4 2 次")
            page.evaluate("window.__qaMic.contexts.at(-1).source.frequency.value = 523.251")
            expect(page.locator("#live-note")).to_contain_text("C5")
            expect(page.locator("#history-counts")).to_contain_text("C5 1 次")
            expect(page.locator("#history-high")).to_contain_text("C5")
            expect(page.locator("#history-low")).to_contain_text("A4")
            page.evaluate("window.__qaMic.contexts.at(-1).source.frequency.value = 440")
            expect(page.locator("#live-note")).to_contain_text("A4")
            page.locator("#history-chart").scroll_into_view_if_needed()
            page.screenshot(path=str(EVIDENCE / "ui-history.png"))
            analyze_file("reference", tone_bytes(seconds=1.5), "reference-while-singing.wav")
            assert page.evaluate("document.querySelector('#mic-badge').classList.contains('active')"), "Uploading a file released the live microphone"
            expect(page.locator("#live-note")).to_contain_text("A4")
            stop_microphone()
            expect(page.locator("#live-frequency")).to_contain_text("已停止 · 上次音高")
            expect(page.locator("#live-note")).to_contain_text("A4")
            retained = int(re.search(r"(\d+) 个音高观测点", page.locator("#history-status").inner_text()).group(1))
            assert retained > 40
            page.locator("#history-clear").click()
            expect(page.locator("#history-empty")).to_be_visible()
            expect(page.locator("#history-status")).to_contain_text("等待本次演唱")
            assert "A4" not in page.locator("#live-note").inner_text()
            expect(page.locator("#history-high")).to_have_text("暂无")
            expect(page.locator("#history-low")).to_have_text("暂无")
            expect(page.locator("#history-counts")).to_have_text("暂无")
            record(current_case, {"paused_view_keeps_capture": True, "retained_after_stop": retained, "clear_resets_view": True})
            record("pitch_history_cursor_readout_and_silent_last_note", "Hover shows note/Hz, silence shows no valid pitch, live note holds A4 without scoring, stop retains, clear resets")
            current_case = "sustained_voice_silence_recovery_and_file_change"
            record(current_case, {
                "continuous_capture_updates": voice_before[-1]["updates"] - voice_before[0]["updates"],
                "voiced_samples_before": sum(row["a4"] for row in voice_before),
                "silent_samples_unvoiced": sum(not row["a4"] for row in silence),
                "quiet_voiced_samples": sum(row["a4"] for row in quiet_voice),
                "voiced_samples_after": sum(row["a4"] for row in voice_after),
                "upload_keeps_microphone_active": True,
            })

            current_case = "synthetic_microphone_real_detector_repeated_start_stop"
            for iteration in range(3):
                start_microphone()
                expect(page.locator("#live-note")).to_contain_text("A4", timeout=20000)
                wait_js(page, "document.querySelector('#mic-level').value > .01")
                stop_microphone()
            record(current_case, {"cycles": 3, "detected_note": "A4", "all_tracks_stopped": True})

            current_case = "synthetic_microphone_silence_is_unscored"
            page.locator("#history-clear").click()
            page.evaluate("window.__qaMic.gain = 0")
            start_microphone()
            expect(page.locator("#live-frequency")).to_contain_text(re.compile("无声|低置信度"), timeout=15000)
            assert "A4" not in page.locator("#live-note").inner_text()
            expect(page.locator("#history-high")).to_have_text("暂无")
            stop_microphone()
            record(current_case, "Silent synthetic MediaStream leaves pitch unvoiced")

            current_case = "cancel_pending_microphone_permission_late_stream_cleanup"
            page.evaluate("window.__qaMic.mode = 'pending'; window.__qaMic.gain = .25")
            page.locator("#mic-toggle").click()
            wait_js(page, "window.__qaMic.resolvers.length === 1")
            expect(page.locator("#mic-toggle")).to_contain_text(re.compile("取消|停止"))
            page.locator("#mic-toggle").click()
            page.evaluate("window.__qaMic.resolvers.shift()()")
            expect(page.locator("#mic-toggle")).to_contain_text("开启")
            wait_js(page, "window.__qaMic.streams.every(stream => stream.getTracks().every(track => track.readyState === 'ended'))")
            page.evaluate("window.__qaMic.mode = 'immediate'")
            start_microphone()
            expect(page.locator("#live-note")).to_contain_text("A4", timeout=20000)
            stop_microphone()
            record(current_case, "Permission Promise resolved after cancellation; late stream tracks stopped and next start works")

            current_case = "live_reference_comparison_and_seek_reset"
            analyze_file("reference", tone_bytes(440, seconds=6.4), "reference-a4.wav")
            page.locator("#playback-source").select_option("reference:vocals")
            page.locator("#time-offset").fill("0")
            page.locator("#time-offset").dispatch_event("change")
            page.locator("#input-latency").fill("0")
            page.locator("#input-latency").dispatch_event("change")
            start_microphone()
            expect(page.locator("#live-note")).to_contain_text("A4", timeout=20000)
            page.locator("#audio-player").evaluate("audio => {audio.currentTime = .5; return audio.play()}")
            wait_js(page, "Number(document.querySelector('#metric-count').textContent) >= 5", timeout=20000)
            live_cents = float(re.search(r"[0-9.]+", page.locator("#metric-cents").inner_text()).group())
            assert live_cents < 5, live_cents
            page.locator("#audio-player").evaluate("audio => audio.pause()")
            expect(page.locator("#metric-count")).to_have_text("0")
            page.locator("#audio-player").evaluate("audio => {audio.currentTime = 0;}")
            expect(page.locator("#metric-count")).to_have_text("0")
            stop_microphone()
            record(current_case, {"A4_reference_A4_live_mean_absolute_cents": live_cents, "pause_resets_count": True, "seek_resets_count": True})

            current_case = "cancel_upload_processing_and_recover"
            page.locator("#reference-mode").select_option("dry")
            page.locator("#reference-file").set_input_files({"name": "cancel-me.wav", "mimeType": "audio/wav", "buffer": tone_bytes(seconds=45)})
            expect(page.locator("#reference-cancel")).to_be_visible()
            page.locator("#reference-cancel").click()
            expect(page.locator("#reference-cancel")).to_be_hidden()
            analyze_file("reference", tone_bytes(), "after-cancel.wav")
            record(current_case, "Cancel during upload/processing then upload new file; deterministic backend cancellation separately tested by pytest")

            current_case = "synthetic_ncm_browser_import_and_pitch"
            from tests.test_ncm import synthetic_ncm
            flac = io.BytesIO()
            signal = .25 * np.sin(2 * np.pi * 440 * np.arange(44100) / 44100)
            sf.write(flac, signal, 44100, format="FLAC")
            assert ".ncm" in page.locator("#reference-file").get_attribute("accept")
            analyze_file("reference", synthetic_ncm(flac.getvalue()), "synthetic-tone.ncm")
            wait_js(page, "document.querySelector('#pitch-chart').getAttribute('aria-label').includes('原唱目标')")
            record(current_case, "Local NCM container extraction, FLAC decoding and visible pitch trace; generated tone, not the supplied song")

            current_case = "actual_model_mixed_upload_to_stems_and_pitch"
            # Small original test fixture: actual ONNX model, never a simulated separator.
            sys.path.insert(0, str(ROOT / "tests"))
            from test_separation import generated_mix
            mixed, _ = generated_mix(seconds=6)
            buffer = io.BytesIO()
            sf.write(buffer, mixed, 44100, format="WAV", subtype="PCM_16")
            model_started = time.perf_counter()
            analyze_file("reference", buffer.getvalue(), "original-generated-mix.wav", mode="mixed")
            options = page.locator("#playback-source option").all_text_contents()
            assert any("伴奏" in option for option in options)
            jobs = context.request.get(base_url + "/api/jobs?summary=false").json()
            mixed_job = next(job for job in jobs if job.get("result") and job["result"]["name"] == "original-generated-mix.wav")
            assert mixed_job["result"]["separation"] == "UVR_MDXNET_Main"
            assert set(mixed_job["result"]["audio"]) == {"original", "vocals", "accompaniment"}
            assert mixed_job["result"]["frames"]
            downloaded = {}
            stems = {}
            for stem, url in mixed_job["result"]["audio"].items():
                response = context.request.get(base_url + url)
                assert response.status == 200
                downloaded[stem] = len(response.body())
                stems[stem], _ = sf.read(io.BytesIO(response.body()), dtype="float32", always_2d=True)
                stems[stem] *= mixed_job["result"]["storage_gains"][stem]
            stem_rms = {name: float(np.sqrt(np.mean(data.astype(np.float64) ** 2))) for name, data in stems.items()}
            assert stem_rms["vocals"] > .01 and stem_rms["accompaniment"] > .01, stem_rms
            from backend.pitch import analyze_pitch
            vocal_frames = analyze_pitch(stems["vocals"], 44100)
            saved_frames = mixed_job["result"]["frames"]
            assert len(vocal_frames) == len(saved_frames)
            assert [frame["frequency"] is None for frame in vocal_frames] == [frame["frequency"] is None for frame in saved_frames]
            deviations = [abs(a["frequency"] - b["frequency"]) for a, b in zip(vocal_frames, saved_frames)
                          if a["frequency"] is not None and b["frequency"] is not None]
            assert max(deviations, default=0) < .02, "Pitch track differs from high-precision compact vocal stem"
            for stem in ("original", "vocals", "accompaniment"):
                page.locator("#playback-source").select_option(f"reference:{stem}")
                wait_js(page, "document.querySelector('#audio-player').readyState >= 2 && document.querySelector('#audio-player').duration > 5.9")
            residual = float(np.max(np.abs(stems["vocals"] + stems["accompaniment"] - stems["original"])))
            assert residual < 5e-7, residual
            mixed_report = {
                "status": "passed", "job_id": mixed_job["id"],
                "fixture": "Original 6 second generated harmonic lead plus stereo bass/chord/percussion",
                "separation": mixed_job["result"]["separation"], "frame_count": len(mixed_job["result"]["frames"]),
                "downloaded_stem_bytes": downloaded, "reconstruction_max_absolute_error": residual,
                "stem_rms": stem_rms, "pitch_equals_exported_vocals": True,
                "all_three_tracks_browser_playable": True,
                "elapsed_seconds": round(time.perf_counter() - model_started, 2),
                "quality_scope": "Actual model/pipeline execution on generated audio; not a singing separation benchmark",
            }
            record(current_case, {"elapsed_seconds": round(time.perf_counter() - model_started, 2), "duration_seconds": 6, "quality_scope": "Actual model/pipeline execution on generated audio; not a singing separation benchmark"})

            current_case = "compact_flac_peak_gain_playback"
            over_peak = np.stack([1.2 * np.sin(2 * np.pi * 440 * np.arange(44100) / 44100)] * 2, axis=1)
            over_peak_bytes = io.BytesIO()
            sf.write(over_peak_bytes, over_peak, 44100, format="WAV", subtype="FLOAT")
            analyze_file("recording", over_peak_bytes.getvalue(), "qa-over-peak.wav", mode="dry")
            gain_job = next(job for job in context.request.get(base_url + "/api/jobs?summary=false").json()
                            if job.get("result") and job["result"]["name"] == "qa-over-peak.wav")
            assert gain_job["result"]["storage_gains"]["original"] > 1.2
            page.locator("#playback-source").select_option("recording:original")
            wait_js(page, "document.querySelector('#audio-player').readyState >= 2")
            page.locator("#audio-player").evaluate("audio => audio.play()")
            wait_js(page, "document.querySelector('#audio-player').currentTime > 0.1")
            page.locator("#audio-player").evaluate("audio => audio.pause()")
            record(current_case, {"peak_gain": gain_job["result"]["storage_gains"]["original"], "played_in_chrome": True})

            current_case = "responsive_mobile_layout"
            page.set_viewport_size({"width": 390, "height": 844})
            page.wait_for_timeout(200)
            page.locator("#history-chart").evaluate("canvas => { const box = canvas.getBoundingClientRect(); canvas.dispatchEvent(new PointerEvent('pointerdown', {pointerType:'touch', clientX:box.left+55, clientY:box.top+box.height/2, bubbles:true})); }")
            expect(page.locator("#history-cursor")).to_be_visible()
            page.locator("#pitch-chart").evaluate("canvas => { const box = canvas.getBoundingClientRect(); canvas.dispatchEvent(new PointerEvent('pointerdown', {pointerType:'touch', clientX:box.left+55, clientY:box.top+box.height/2, bubbles:true})); }")
            expect(page.locator("#timeline-cursor")).to_be_visible()
            assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth + 1"), "Horizontal page overflow"
            page.screenshot(path=str(EVIDENCE / "ui-mobile.png"), full_page=True)
            record(current_case, {"viewport": "390x844", "horizontal_overflow": False, "touch_pitch_readout": True})

            current_case = "console_and_network_privacy"
            assert not report["page_errors"], report["page_errors"]
            unexpected = [item for item in report["console_errors"] if item["case"] not in {"invalid_file_failure_and_valid_file_recovery", "cancel_upload_processing_and_recover"}]
            assert not unexpected, unexpected
            assert not report["external_requests"], report["external_requests"]
            record(current_case, {"javascript_errors": 0, "external_requests": 0, "expected_fault_console_entries": len(report["console_errors"])})
            report["status"] = "passed"
        except Exception as error:
            report["status"] = "failed"
            report["failed_case"] = current_case
            report["error"] = str(error)
            report["traceback"] = traceback.format_exc()
            page.screenshot(path=str(EVIDENCE / "ui-failure.png"), full_page=True)
            raise
        finally:
            for identifier in owned_jobs:
                context.request.delete(base_url + "/api/jobs/" + identifier)
            if mixed_report:
                data_dir = Path(context.request.get(base_url + "/api/health").json()["data_directory"])
                mixed_report["cache_deleted"] = not (data_dir / mixed_report["job_id"]).exists()
                (EVIDENCE / "mixed-api.json").write_text(json.dumps(mixed_report, ensure_ascii=False, indent=2), encoding="utf-8")
                assert mixed_report["cache_deleted"]
            try:
                page.evaluate("Promise.all((window.__qaMic?.contexts || []).map(item => item.context.close()))")
            except Exception:
                pass
            report["elapsed_seconds"] = round(time.perf_counter() - started, 3)
            report["finished_utc"] = datetime.now(timezone.utc).isoformat()
            (EVIDENCE / "ui-tests.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            context.close()
            browser.close()
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8765")
    parser.add_argument("--chrome", default=CHROME)
    arguments = parser.parse_args()
    result = run(arguments.base_url.rstrip("/"), arguments.chrome)
    print(json.dumps({"status": result["status"], "passed": len(result["tests"]), "elapsed_seconds": result["elapsed_seconds"]}))
